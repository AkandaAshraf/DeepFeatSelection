"""Boundary map for MACE against known ground truth.

Pre-registration: paper/boundary_map_protocol.md, committed before this was
written. Four axes varied one at a time from a fixed centre; precision,
recall, ghost, saturation and source false-positive rate measured per cell.

Replaces a proposed survey of real datasets for premise satisfaction, which
an adversarial pass killed: the iEEG cohort satisfied every premise and then
failed replication, so premise satisfaction does not predict validity.

    python scripts/boundary_map.py            # full grid
    python scripts/boundary_map.py --quick    # centre cell only, for checks
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from wormwideweb_gate import MaskedAE  # architecture reused verbatim

DEV = "cuda" if torch.cuda.is_available() else "cpu"
E, TAU, ALPHA = 3, 1, 1.0
BOTTLENECK, MASK, EPOCHS, BATCH, MODELS = 32, 0.25, 20, 64, 2
N_GHOSTS, DONOR_R2, MIN_DONORS = 30, 0.9, 8
OUT = Path("ExpOutput/boundary_map")

CENTRE = dict(n=4000, V=60, coupling=0.20, redundancy=0)
AXES = {
    "n":          [1000, 2000, 4000, 8000, 16000],
    "V":          [15, 30, 60, 120, 240],
    "coupling":   [0.05, 0.10, 0.20, 0.35, 0.50],
    "redundancy": [0, 1, 2, 4, 8],
}
SEEDS = (0, 1, 2)


def make_system(n, V, coupling, redundancy, seed):
    """Drivers are autonomous; driven channels receive from one driver.

    `redundancy` duplicates of each driver's signal (with noise) are added as
    extra DRIVEN channels, so the code can reach a target through a duplicate
    - the Takens redundancy the method is expected to be limited by.
    """
    rng = np.random.default_rng(seed)
    n_src = max(3, V // 6)
    n_drv = V - n_src
    x = np.zeros((n, V))
    x[0] = rng.uniform(0.2, 0.8, V)
    r = rng.uniform(3.6, 3.9, V)
    parent = rng.integers(0, n_src, n_drv)
    for t in range(n - 1):
        s = x[t, :n_src]
        x[t + 1, :n_src] = np.clip(r[:n_src] * s * (1 - s), 0, 1)
        k = x[t, n_src:]
        x[t + 1, n_src:] = np.clip(
            r[n_src:] * k * (1 - k) + coupling * x[t, parent] * (1 - k), 0, 1)
    is_driven = np.zeros(V, bool)
    is_driven[n_src:] = True
    is_source = ~is_driven

    if redundancy:
        dup = []
        for d in range(min(redundancy, n_src)):
            for _ in range(1):
                dup.append(x[:, d] + 0.02 * rng.standard_normal(n))
        if dup:
            x = np.concatenate([x, np.stack(dup, axis=1)], axis=1)
            # duplicates carry a driver's signal; they are neither driven nor
            # sources, and are excluded from both truth sets
            is_driven = np.append(is_driven, np.zeros(len(dup), bool))
            is_source = np.append(is_source, np.zeros(len(dup), bool))
    x = x + 0.005 * rng.standard_normal(x.shape)
    return x, is_driven, is_source


def embed(x):
    span = (E - 1) * TAU
    m = x.shape[0] - span
    return np.concatenate(
        [np.stack([x[span - k * TAU: span - k * TAU + m, j]
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


def run_cell(n, V, coupling, redundancy, seed):
    x, is_driven, is_source = make_system(n, V, coupling, redundancy, seed)
    Vt = x.shape[1]
    emb = embed(x)
    m = emb.shape[0]
    a, b = int(0.6 * m), int(0.8 * m)
    tr = slice(0, a)
    tr_i = np.arange(0, a - 1)
    te_i = np.arange(b, m - 1)
    mu, sd = emb[tr].mean(0), emb[tr].std(0) + 1e-12
    zs = np.clip(np.nan_to_num((emb - mu) / sd), -20, 20).astype(np.float32)
    lead = zs[:, [j * E for j in range(Vt)]]
    feats = [poly3(zs[:, q * E:(q + 1) * E]) for q in range(Vt)]

    self_r2 = np.array([ridge_r2(f[tr_i], lead[tr_i + 1, q],
                                 f[te_i], lead[te_i + 1, q])
                        for q, f in enumerate(feats)])

    ztr = torch.as_tensor(zs[tr], device=DEV)
    zfull = torch.as_tensor(zs, device=DEV)
    codes = []
    for mm in range(MODELS):
        torch.manual_seed(seed * 100 + mm)
        net = MaskedAE(zs.shape[1], BOTTLENECK).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        g = torch.Generator().manual_seed(seed * 100 + mm)
        for _ in range(EPOCHS):
            perm = torch.randperm(ztr.shape[0], generator=g)
            for i in range(0, len(perm), BATCH):
                bt = ztr[perm[i:i + BATCH]]
                msk = torch.rand(bt.shape[0], Vt, device=DEV) < MASK
                mc = msk.repeat_interleave(E, dim=1)
                loss = ((net(bt.masked_fill(mc, 0.0)) - bt)[mc] ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            codes.append(net.enc(zfull).cpu().numpy())

    def excess_of(f, target):
        base = ridge_r2(f[tr_i], target[tr_i + 1], f[te_i], target[te_i + 1])
        return float(np.mean([
            ridge_r2(np.hstack([f[tr_i], c[tr_i]]), target[tr_i + 1],
                     np.hstack([f[te_i], c[te_i]]), target[te_i + 1]) - base
            for c in codes]))

    excess = np.array([excess_of(feats[q], lead[:, q]) for q in range(Vt)])

    rng = np.random.default_rng(seed + 4242)
    qual = np.where(self_r2 > DONOR_R2)[0]
    fallback = len(qual) < MIN_DONORS
    pool = np.arange(Vt) if fallback else qual
    donors = rng.choice(pool, size=min(N_GHOSTS, len(pool)), replace=False)
    ghosts = []
    for d in donors:
        s = int(rng.integers(m // 4, 3 * m // 4))
        gz = np.roll(zs[:, d * E:(d + 1) * E], s, axis=0)
        ghosts.append(excess_of(poly3(gz), np.roll(lead[:, d], s)))
    ghosts = np.array(ghosts)
    thr = max(0.0, float(ghosts.max()))

    flagged = excess > thr
    tp = int((flagged & is_driven).sum())
    return {
        "n": n, "V": V, "coupling": coupling, "redundancy": redundancy,
        "seed": seed, "channels": Vt,
        "precision": tp / max(int(flagged.sum()), 1),
        "recall": tp / max(int(is_driven.sum()), 1),
        "n_flagged": int(flagged.sum()),
        "ghost_med": float(np.median(ghosts)),
        "ghost_max": float(ghosts.max()),
        "ghost_void": bool(np.median(ghosts) > 0.005),
        "saturation": float((self_r2 > 0.9).mean()),
        "donor_fallback": bool(fallback),
        "source_fp": float((flagged & is_source).sum()
                           / max(int(is_source.sum()), 1)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEV}   centre: {CENTRE}")

    cells = []
    if args.quick:
        cells = [dict(CENTRE, seed=0)]
    else:
        seen = set()
        for axis, values in AXES.items():
            for v in values:
                cfg = dict(CENTRE)
                cfg[axis] = v
                for s in SEEDS:
                    key = (cfg["n"], cfg["V"], cfg["coupling"],
                           cfg["redundancy"], s)
                    if key in seen:
                        continue
                    seen.add(key)
                    cells.append(dict(cfg, seed=s))
    print(f"cells to run: {len(cells)}\n")

    rows, t0 = [], time.time()
    dest = OUT / "boundary_map.csv"
    if dest.exists():
        rows = pd.read_csv(dest).to_dict("records")
        done = {(r["n"], r["V"], r["coupling"], r["redundancy"], r["seed"])
                for r in rows}
        cells = [c for c in cells
                 if (c["n"], c["V"], c["coupling"], c["redundancy"],
                     c["seed"]) not in done]
        print(f"resuming: {len(rows)} cells done, {len(cells)} to go\n")

    for i, c in enumerate(cells, 1):
        try:
            r = run_cell(**c)
        except Exception as exc:
            print(f"  [{i}/{len(cells)}] {c} FAILED: {str(exc)[:70]}",
                  flush=True)
            continue
        rows.append(r)
        pd.DataFrame(rows).to_csv(dest, index=False)
        print(f"  [{i}/{len(cells)}] n={r['n']:<6} V={r['V']:<4} "
              f"c={r['coupling']:<5} red={r['redundancy']}  "
              f"prec {r['precision']:.2f} rec {r['recall']:.2f}  "
              f"sat {r['saturation']:.2f}  srcFP {r['source_fp']:.2f}"
              f"{'  GHOST-VOID' if r['ghost_void'] else ''}", flush=True)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
