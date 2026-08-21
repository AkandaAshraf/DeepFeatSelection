"""Falsification gate for the proposed source-detection statistic.

Derivation and predictions: paper/source_detection_note.md, written before
this ran. The three checks are the ones that would kill the statistic:

  S1 separation  designed sources score high outflow / low excess; sinks the
                 reverse. If they do not separate, the statistic fails.
  S2 ghost       a circularly shifted copy must score outflow ~ 0. If not,
                 the statistic reads typicality rather than influence
                 (companion paper, Mechanism 2) and is dead.
  S3 maturity    outflow must not decay with training epochs. If it does, it
                 is a difference-based importance score in disguise
                 (Mechanism 1) and must be discarded.

    python scripts/source_outflow_gate.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from wormwideweb_gate import MaskedAE  # architecture reused verbatim

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
E, TAU, ALPHA = 3, 1, 1.0
BOTTLENECK, MASK, BATCH, MODELS = 16, 0.25, 64, 2


# ----------------------------------------------------------------- system

def coupled_system(n=4000, n_src=3, n_sink=6, n_iso=6, seed=0):
    """Sources drive sinks; isolated variables are coupled to nothing.

    Ground truth by construction: sources receive nothing, sinks receive
    from sources, isolated variables receive nothing and send nothing.
    Sources and isolated variables are indistinguishable to MACE (both have
    zero inflow) - that is the blindness this statistic must resolve.
    """
    rng = np.random.default_rng(seed)
    V = n_src + n_sink + n_iso
    x = np.zeros((n, V))
    x[0] = rng.uniform(0.2, 0.8, V)
    r_src = rng.uniform(3.7, 3.9, n_src)
    r_iso = rng.uniform(3.7, 3.9, n_iso)
    r_snk = rng.uniform(3.5, 3.7, n_sink)
    # each sink is driven by one source
    parent = rng.integers(0, n_src, n_sink)
    for t in range(n - 1):
        s = x[t, :n_src]
        x[t + 1, :n_src] = np.clip(r_src * s * (1 - s), 0, 1)
        k = x[t, n_src:n_src + n_sink]
        drive = 0.35 * x[t, parent]
        x[t + 1, n_src:n_src + n_sink] = np.clip(
            r_snk * k * (1 - k) + drive * (1 - k), 0, 1)
        i = x[t, n_src + n_sink:]
        x[t + 1, n_src + n_sink:] = np.clip(r_iso * i * (1 - i), 0, 1)
    x += 0.01 * rng.standard_normal((n, V))
    role = np.array(["source"] * n_src + ["sink"] * n_sink
                    + ["isolated"] * n_iso)
    return x, role


# ------------------------------------------------------------- machinery

def embed(x):
    span = (E - 1) * TAU
    n = x.shape[0] - span
    return np.concatenate(
        [np.stack([x[span - k * TAU: span - k * TAU + n, j]
                   for k in range(E)], axis=1) for j in range(x.shape[1])],
        axis=1)


def poly2(a):
    cols = [a]
    e = a.shape[1]
    for i in range(e):
        for j in range(i, e):
            cols.append((a[:, i] * a[:, j])[:, None])
    return np.hstack(cols)


def ridge_r2(Xtr, ytr, Xte, yte) -> float:
    """R^2 of a ridge fit; y may be multivariate (mean over columns)."""
    Xt = torch.as_tensor(Xtr, dtype=torch.float64, device=DEV)
    yt = torch.as_tensor(ytr, dtype=torch.float64, device=DEV)
    Xe = torch.as_tensor(Xte, dtype=torch.float64, device=DEV)
    ye = torch.as_tensor(yte, dtype=torch.float64, device=DEV)
    if yt.ndim == 1:
        yt, ye = yt[:, None], ye[:, None]
    A = Xt.T @ Xt + ALPHA * torch.eye(Xt.shape[1], device=DEV,
                                      dtype=torch.float64)
    w = torch.linalg.solve(A, Xt.T @ yt)
    err = ((Xe @ w - ye) ** 2).mean(0)
    var = ye.var(0) + 1e-12
    return float((1 - err / var).clamp(min=0).mean())


def train_ae(zs, V, tr, epochs, seed):
    torch.manual_seed(seed)
    net = MaskedAE(zs.shape[1], BOTTLENECK).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    ztr = torch.as_tensor(zs[tr], device=DEV)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(ztr.shape[0], generator=g)
        for i in range(0, len(perm), BATCH):
            b = ztr[perm[i:i + BATCH]]
            msk = torch.rand(b.shape[0], V, device=DEV) < MASK
            mc = msk.repeat_interleave(E, dim=1)
            out = net(b.masked_fill(mc, 0.0))
            loss = ((out - b)[mc] ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


def code_history(zq):
    """Delay embedding of the code: [z(t), z(t-1), ..., z(t-E+1)]."""
    span = E - 1
    cols = [np.roll(zq, k, axis=0) for k in range(E)]
    h = np.concatenate(cols, axis=1)
    h[:span] = 0.0
    return h


def codes_with_mask(net, zs, V, q=None):
    """Code from the full system, or with channel q masked at the input."""
    zt = torch.as_tensor(zs, device=DEV).clone()
    if q is not None:
        zt[:, q * E:(q + 1) * E] = 0.0
    with torch.no_grad():
        return net.enc(zt).cpu().numpy()


def analyse(x, epochs=25, ghost_of=0):
    """Return excess and outflow per channel, plus the ghost's scores."""
    xg = np.concatenate(
        [x, np.roll(x[:, [ghost_of]], x.shape[0] // 3, axis=0)], axis=1)
    V = xg.shape[1]
    emb = embed(xg)
    n = emb.shape[0]
    a, b = int(0.6 * n), int(0.8 * n)
    tr = slice(0, a)
    tr_i = np.arange(0, a - 1)
    te_i = np.arange(b, n - 1)
    mu, sd = emb[tr].mean(0), emb[tr].std(0) + 1e-12
    zs = np.clip((emb - mu) / sd, -20, 20).astype(np.float32)
    lead = zs[:, [j * E for j in range(V)]]
    feats = [poly2(zs[:, q * E:(q + 1) * E]) for q in range(V)]

    exc = np.zeros(V)
    out = np.zeros(V)
    for m in range(MODELS):
        net = train_ae(zs, V, tr, epochs, SEED + m)
        z_full = codes_with_mask(net, zs, V)
        for q in range(V):
            # inflow: what the code adds to predicting q
            base = ridge_r2(feats[q][tr_i], lead[tr_i + 1, q],
                            feats[q][te_i], lead[te_i + 1, q])
            with_code = ridge_r2(
                np.hstack([feats[q][tr_i], z_full[tr_i]]), lead[tr_i + 1, q],
                np.hstack([feats[q][te_i], z_full[te_i]]), lead[te_i + 1, q])
            exc[q] += (with_code - base) / MODELS

            # outflow: what q adds to predicting the rest-of-system code.
            # The baseline is a DELAY EMBEDDING of the code, not a single
            # time point: otherwise any channel improves the fit merely by
            # supplying temporal depth the baseline lacks, and source,
            # isolated and ghost all score alike.
            zq = codes_with_mask(net, zs, V, q=q)
            zh = code_history(zq)
            span = E - 1
            ti = tr_i[tr_i >= span]
            si = te_i[te_i >= span]
            self_base = ridge_r2(zh[ti], zq[ti + 1], zh[si], zq[si + 1])
            with_q = ridge_r2(
                np.hstack([zh[ti], feats[q][ti]]), zq[ti + 1],
                np.hstack([zh[si], feats[q][si]]), zq[si + 1])
            out[q] += (with_q - self_base) / MODELS
    return exc, out


def main() -> int:
    print(f"device: {DEV}")
    x, role = coupled_system()
    print(f"system: {x.shape[1]} variables "
          f"({(role=='source').sum()} source, {(role=='sink').sum()} sink, "
          f"{(role=='isolated').sum()} isolated), n={x.shape[0]}")

    t0 = time.time()
    exc, out = analyse(x, epochs=25)
    print(f"analysed in {time.time()-t0:.0f}s\n")

    g_exc, g_out = exc[-1], out[-1]
    exc, out = exc[:-1], out[:-1]

    print("S1 SEPARATION")
    print(f"  {'role':10s} {'excess (inflow)':>18s} {'outflow':>12s}")
    for r in ("source", "sink", "isolated"):
        m = role == r
        print(f"  {r:10s} {np.median(exc[m]):+18.4f} "
              f"{np.median(out[m]):+12.4f}")
    src, iso = role == "source", role == "isolated"
    sep = np.median(out[src]) - np.median(out[iso])
    print(f"  source vs isolated, outflow gap: {sep:+.4f}")
    print(f"  -> {'SEPARATES' if sep > 0.01 else 'FAILS TO SEPARATE'}"
          "  (this is the blindness MACE cannot resolve)\n")

    print("S2 GHOST")
    print(f"  ghost excess {g_exc:+.4f}   ghost outflow {g_out:+.4f}")
    thr = max(0.02, 0.25 * float(np.median(out[src])))
    print(f"  -> {'PASS' if abs(g_out) < thr else 'FAIL'} "
          f"(|outflow| < {thr:.3f})\n")

    print("S3 MATURITY (outflow must not decay with training)")
    for ep in (5, 15, 40):
        e2, o2 = analyse(x, epochs=ep)
        o2 = o2[:-1]
        print(f"  epochs {ep:3d}: source outflow median "
              f"{np.median(o2[src]):+.4f}   isolated "
              f"{np.median(o2[iso]):+.4f}")
    print("  -> decay toward zero with epochs would mean this is a "
          "difference-based\n     importance score after all, and it dies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
