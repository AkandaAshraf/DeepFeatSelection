"""Separate width from capacity in the saturation premise.

Pre-registration: paper/crossed_saturation_protocol.md, committed before this
was written or run.

The earlier saturation-gate run found source false positives ordered by
width at matched self-R2 (V=15: 0.00, V=30: 0.20-0.40) but held b at 32 for
both, so b/V was 2.1 against 1.07 and width was confounded with capacity.
Its result section said the confound could not be resolved by any further
reading of those cells, and specified this design.

Here b = 2V in EVERY cell, so the capacity ratio is constant at 2.0 and any
surviving width effect is not a capacity effect. Crossed with the noise
ladder and with k in {0, 2}, because the original finding came only from
k = 2 cells (Rule 86).

    python scripts/crossed_saturation.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from boundary_map import (BATCH, DEV, DONOR_R2, E, EPOCHS, MASK,  # noqa: E402
                          MIN_DONORS, MODELS, N_GHOSTS, embed, make_system,
                          poly3, ridge_r2)
from wormwideweb_gate import MaskedAE  # noqa: E402

OUT = Path("ExpOutput/crossed_saturation")
N = 4000
COUPLING = 0.20
WIDTHS = (15, 30, 60)          # b = 2V in each: 30, 60, 120
NOISE = (0.0, 0.02, 0.05, 0.10, 0.30)
KS = (0, 2)
SEEDS = (0, 1, 2)


def run_cell(V, b, k, seed, obs_noise):
    """boundary_map's scan with b and redundancy parameterised."""
    x, is_driven, is_source = make_system(N, V, COUPLING, k, seed)
    if obs_noise:
        rng = np.random.default_rng(seed + 777)
        x = x + obs_noise * rng.standard_normal(x.shape)
    Vt = x.shape[1]
    emb = embed(x)
    m = emb.shape[0]
    a, bnd = int(0.6 * m), int(0.8 * m)
    tr, tr_i, te_i = slice(0, a), np.arange(0, a - 1), np.arange(bnd, m - 1)
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
        net = MaskedAE(zs.shape[1], b).to(DEV)          # b, not BOTTLENECK
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
    pool = np.arange(Vt) if len(qual) < MIN_DONORS else qual
    donors = rng.choice(pool, size=min(N_GHOSTS, len(pool)), replace=False)
    ghosts = np.array([
        excess_of(poly3(np.roll(zs[:, d * E:(d + 1) * E], s_, axis=0)),
                  np.roll(lead[:, d], s_))
        for d, s_ in zip(donors, rng.integers(m // 4, 3 * m // 4, len(donors)))])
    thr = max(0.0, float(ghosts.max()))
    flagged = excess > thr

    return {
        "V": V, "b": b, "b_over_V": b / V, "k": k, "seed": seed,
        "noise": obs_noise, "channels": Vt,
        "self_r2_med": float(np.median(self_r2)),
        "source_fp": float((flagged & is_source).sum()
                           / max(int(is_source.sum()), 1)),
        "recall": float((flagged & is_driven).sum()
                        / max(int(is_driven.sum()), 1)),
        "precision": float((flagged & is_driven).sum()
                           / max(int(flagged.sum()), 1)),
        "ghost_med": float(np.median(ghosts)),
        "ghost_max": float(ghosts.max()),
        "ghost_ok": bool(np.median(ghosts) <= 0.005),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEV}   b = 2V in every cell   coupling={COUPLING}")
    print(f"{len(WIDTHS)*len(NOISE)*len(KS)*len(SEEDS)} cells\n")
    rows, t0 = [], time.time()
    for k in KS:
        for V in WIDTHS:
            b = 2 * V
            for nz in NOISE:
                for s in SEEDS:
                    r = run_cell(V, b, k, s, nz)
                    rows.append(r)
                print(f"  k={k} V={V:<3} b={b:<3} nz={nz:<5} "
                      f"selfR2 {np.median([x['self_r2_med'] for x in rows[-3:]]):.3f}  "
                      f"srcFP {np.median([x['source_fp'] for x in rows[-3:]]):.2f}  "
                      f"recall {np.median([x['recall'] for x in rows[-3:]]):.2f}  "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "cells.csv", index=False)
    print(f"\n({(time.time()-t0)/60:.1f} min)\n")

    clean = d[d.ghost_ok]
    print(f"X5 ghost: {len(d)-len(clean)} of {len(d)} cells dirty "
          f"(excluded from X2)\n")

    print("X1 REPRODUCTION: does source FP rise as self-R2 falls, at k=2?")
    for V in WIDTHS:
        g = d[(d.k == 2) & (d.V == V)].groupby("noise").agg(
            sr=("self_r2_med", "median"), fp=("source_fp", "median"))
        print(f"   V={V:<3} " + "  ".join(
            f"{nz}:{g.loc[nz,'sr']:.2f}/{g.loc[nz,'fp']:.2f}"
            for nz in NOISE if nz in g.index))
    x1 = d[d.k == 2].source_fp.max() > 0.05
    print(f"   -> {'REPRODUCES' if x1 else 'DOES NOT REPRODUCE - VOID'}")
    if not x1:
        return 0

    print("\nX3: is the failure weaker at k=0 than k=2?")
    for k in KS:
        print(f"   k={k}: max source FP {d[d.k==k].source_fp.max():.2f}, "
              f"median {d[d.k==k].source_fp.median():.2f}")

    print("\nX2 DECISIVE: at b/V=2 throughout, is source FP ordered by width")
    print("   at matched self-R2? (k=2, ghost-clean cells)")
    sub = clean[clean.k == 2]
    for nz in NOISE:
        g = sub[sub.noise == nz].groupby("V").agg(
            sr=("self_r2_med", "median"), fp=("source_fp", "median"))
        if len(g) < 2:
            continue
        line = "  ".join(f"V={V}: selfR2 {g.loc[V,'sr']:.2f} fp {g.loc[V,'fp']:.2f}"
                         for V in g.index)
        print(f"   nz={nz:<5} {line}")
    piv = sub.groupby("V").source_fp.median()
    spread = float(piv.max() - piv.min())
    print(f"\n   median source FP by width: "
          + "  ".join(f"V={V}: {piv[V]:.3f}" for V in piv.index))
    print(f"   spread across widths: {spread:.3f}")

    print("\nVERDICT (rule fixed before running)")
    if spread <= 0.05:
        print("   -> CAPACITY. At constant b/V the width ordering "
              "disappears; the earlier\n      width effect was "
              "under-capacity, and the saturation premise is one-\n"
              "      dimensional after all.")
    else:
        print("   -> TWO DIMENSIONS. The width ordering survives at constant "
              "b/V, so width is\n      a real second dimension and the "
              "licensing premise needs two numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
