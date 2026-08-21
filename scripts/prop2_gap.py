"""Decompose Proposition 2's lower-bound gap into readout and compression.

Pre-registration: paper/prop2_gap_protocol.md, committed before this was
written. Four estimators, defined there:

    BASE      own history only
    AFFINE    own history + code, linear in the code  (as implemented)
    INTERACT  own history + code + (own linear lags) x code
    ORACLE    own history + the full uncompressed state of every other channel

ORACLE - AFFINE is the total gap. INTERACT - AFFINE is what a richer readout
recovers; ORACLE - INTERACT is what the bottleneck loses and no readout can
recover. The ghost must stay near zero for every estimator, which is the
check that a richer readout is not simply fitting noise.

    python scripts/prop2_gap.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from wormwideweb_gate import MaskedAE  # architecture reused verbatim

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
E, TAU, ALPHA = 3, 1, 1.0
BOTTLENECK, MASK, EPOCHS, BATCH, MODELS = 32, 0.25, 25, 64, 4
OUT = Path("ExpOutput/prop2_gap")


def driven_system(n=6000, n_drivers=4, n_driven=10, coupling=0.30, seed=0):
    """Drivers are autonomous; driven channels receive from one driver each."""
    rng = np.random.default_rng(seed)
    V = n_drivers + n_driven
    x = np.zeros((n, V))
    x[0] = rng.uniform(0.2, 0.8, V)
    r = rng.uniform(3.6, 3.9, V)
    parent = rng.integers(0, n_drivers, n_driven)
    for t in range(n - 1):
        d = x[t, :n_drivers]
        x[t + 1, :n_drivers] = np.clip(r[:n_drivers] * d * (1 - d), 0, 1)
        k = x[t, n_drivers:]
        x[t + 1, n_drivers:] = np.clip(
            r[n_drivers:] * k * (1 - k)
            + coupling * x[t, parent] * (1 - k), 0, 1)
    x += 0.005 * rng.standard_normal((n, V))
    is_driven = np.zeros(V, bool)
    is_driven[n_drivers:] = True
    return x, is_driven


def embed(x):
    span = (E - 1) * TAU
    n = x.shape[0] - span
    return np.concatenate(
        [np.stack([x[span - k * TAU: span - k * TAU + n, j]
                   for k in range(E)], axis=1) for j in range(x.shape[1])],
        axis=1)


def poly3(a):
    cols = [a]
    e = a.shape[1]
    for i in range(e):
        for j in range(i, e):
            cols.append((a[:, i] * a[:, j])[:, None])
    for i in range(e):
        for j in range(i, e):
            for k in range(j, e):
                cols.append((a[:, i] * a[:, j] * a[:, k])[:, None])
    return np.hstack(cols)


def ridge_r2(Xtr, ytr, Xte, yte) -> float:
    Xt = torch.as_tensor(Xtr, dtype=torch.float64, device=DEV)
    yt = torch.as_tensor(ytr, dtype=torch.float64, device=DEV)
    Xe = torch.as_tensor(Xte, dtype=torch.float64, device=DEV)
    ye = torch.as_tensor(yte, dtype=torch.float64, device=DEV)
    A = Xt.T @ Xt + ALPHA * torch.eye(Xt.shape[1], device=DEV,
                                      dtype=torch.float64)
    w = torch.linalg.solve(A, Xt.T @ yt)
    err = float(((Xe @ w - ye) ** 2).mean())
    return max(0.0, 1.0 - err / (float(ye.var()) + 1e-12))


def interact(own_lin, code):
    """Bilinear features: each own-lag times each code dimension."""
    return (own_lin[:, :, None] * code[:, None, :]).reshape(len(own_lin), -1)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEV}")
    x, is_driven = driven_system()
    # ghost: circularly shifted copy of a driven channel, coupled to nothing
    ghost_src = int(np.where(is_driven)[0][0])
    xg = np.concatenate(
        [x, np.roll(x[:, [ghost_src]], len(x) // 3, axis=0)], axis=1)
    V = xg.shape[1]
    driven = np.append(is_driven, False)
    print(f"system: {V-1} channels ({is_driven.sum()} driven) + 1 ghost, "
          f"n={len(x)}")

    emb = embed(xg)
    n = emb.shape[0]
    a, b = int(0.6 * n), int(0.8 * n)
    tr = slice(0, a)
    tr_i = np.arange(0, a - 1)
    te_i = np.arange(b, n - 1)
    mu, sd = emb[tr].mean(0), emb[tr].std(0) + 1e-12
    zs = np.clip((emb - mu) / sd, -20, 20).astype(np.float32)
    lead = zs[:, [j * E for j in range(V)]]
    own_lin = [zs[:, q * E:(q + 1) * E] for q in range(V)]
    feats = [poly3(o) for o in own_lin]

    # train the code ensemble
    ztr = torch.as_tensor(zs[tr], device=DEV)
    zfull = torch.as_tensor(zs, device=DEV)
    codes = []
    for m in range(MODELS):
        torch.manual_seed(SEED + m)
        net = MaskedAE(zs.shape[1], BOTTLENECK).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        g = torch.Generator().manual_seed(SEED + m)
        for _ in range(EPOCHS):
            perm = torch.randperm(ztr.shape[0], generator=g)
            for i in range(0, len(perm), BATCH):
                bt = ztr[perm[i:i + BATCH]]
                msk = torch.rand(bt.shape[0], V, device=DEV) < MASK
                mc = msk.repeat_interleave(E, dim=1)
                loss = ((net(bt.masked_fill(mc, 0.0)) - bt)[mc] ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            codes.append(net.enc(zfull).cpu().numpy())

    t0 = time.time()
    rows = []
    for q in range(V):
        y_tr, y_te = lead[tr_i + 1, q], lead[te_i + 1, q]
        base = ridge_r2(feats[q][tr_i], y_tr, feats[q][te_i], y_te)

        # ORACLE: the full uncompressed state of every OTHER channel
        others = np.hstack([feats[p] for p in range(V) if p != q])
        oracle = ridge_r2(np.hstack([feats[q][tr_i], others[tr_i]]), y_tr,
                          np.hstack([feats[q][te_i], others[te_i]]), y_te)

        aff, itr = [], []
        for c in codes:
            aff.append(ridge_r2(
                np.hstack([feats[q][tr_i], c[tr_i]]), y_tr,
                np.hstack([feats[q][te_i], c[te_i]]), y_te))
            ix = interact(own_lin[q], c)
            itr.append(ridge_r2(
                np.hstack([feats[q][tr_i], c[tr_i], ix[tr_i]]), y_tr,
                np.hstack([feats[q][te_i], c[te_i], ix[te_i]]), y_te))
        rows.append({"q": q, "driven": bool(driven[q]),
                     "ghost": q == V - 1,
                     "base": base,
                     "ex_affine": float(np.mean(aff)) - base,
                     "ex_interact": float(np.mean(itr)) - base,
                     "ex_oracle": oracle - base})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "gap.csv", index=False)
    print(f"computed in {time.time()-t0:.0f}s\n")

    dr = d[d.driven & ~d.ghost]
    gh = d[d.ghost].iloc[0]
    aff, itc, orc = (dr.ex_affine.mean(), dr.ex_interact.mean(),
                     dr.ex_oracle.mean())
    total = orc - aff

    print("P-P1  GAP EXISTS?")
    print(f"  affine   {aff:+.4f}    interact {itc:+.4f}    "
          f"oracle {orc:+.4f}")
    print(f"  total gap (oracle - affine) = {total:+.4f}  "
          f"({100*total/orc:.1f}% of oracle)")
    print(f"  -> {'YES' if total > 0.002 else 'NO GAP TO DECOMPOSE'}\n")

    print("P-P2  DECOMPOSITION")
    if total > 1e-9:
        readout = (itc - aff) / total
        compress = (orc - itc) / total
        print(f"  readout share      {100*readout:5.1f}%   "
              f"(recoverable by a richer readout)")
        print(f"  compression share  {100*compress:5.1f}%   "
              f"(bottleneck; no readout recovers this)")
    print()

    print("P-P3  GHOST (must stay near zero under the richer readout)")
    print(f"  affine {gh.ex_affine:+.4f}   interact {gh.ex_interact:+.4f}   "
          f"oracle {gh.ex_oracle:+.4f}")
    ok = abs(gh.ex_interact) < 0.02
    print(f"  -> {'PASS' if ok else 'FAIL - richer readout fits noise'}\n")

    print("P-P4  COST (readout feature count per channel)")
    nb, nc = feats[0].shape[1], BOTTLENECK
    print(f"  affine   {nb + nc:5d}")
    print(f"  interact {nb + nc + E * nc:5d}   "
          f"({(nb + nc + E * nc) / (nb + nc):.1f}x affine)")
    print(f"  oracle   {nb + (V - 1) * nb:5d}   "
          f"(grows with V - this is what MACE avoids)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
