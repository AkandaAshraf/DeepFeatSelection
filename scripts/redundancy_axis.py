"""Redundancy as a real axis: five levels, not two.

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

OUT = Path("ExpOutput/redundancy_axis")
N = 4000
COUPLING = 0.20
WIDTHS = (15, 30, 60)
NOISE = (0.0, 0.05, 0.10)
KS = (0, 1, 2, 4, 8)
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
                    rows.append(run_cell(V, b, k, s, nz))
                last = rows[-3:]
                print(f"  k={k} V={V:<3} b={b:<3} nz={nz:<5} "
                      f"selfR2 {np.median([x['self_r2_med'] for x in last]):.3f}  "
                      f"srcFP {np.median([x['source_fp'] for x in last]):.2f}  "
                      f"recall {np.median([x['recall'] for x in last]):.2f}  "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "cells.csv", index=False)
    print(f"\n({(time.time()-t0)/60:.1f} min)\n")

    dirty = int((~d.ghost_ok).sum())
    print(f"R5 ghost: {dirty} of {len(d)} cells dirty (excluded from R2/R3)")
    clean = d[d.ghost_ok]
    noisy = clean[clean.noise > 0]

    print("\nR1 REPRODUCTION: width ordering at k=0?")
    z = noisy[noisy.k == 0]
    rho0 = z[["V", "source_fp"]].corr(method="spearman").iloc[0, 1]
    print(f"   Spearman(source_fp, V) at k=0: {rho0:+.3f}  (crossed run: +0.672)")
    r1 = rho0 > 0.4
    print(f"   -> {'REPRODUCES' if r1 else 'DOES NOT - k=0 ordering was noise; DISSOLVED'}")
    if not r1:
        return 0

    print("\nR2 DECISIVE: does source FP rise monotonically with k?")
    g = noisy.groupby("k").source_fp.mean()
    print("   mean source_fp by k: " + "  ".join(f"k={k}: {g[k]:.3f}" for k in g.index))
    r2 = all(g[KS[i]] <= g[KS[i + 1]] + 0.02 for i in range(len(KS) - 1))
    print(f"   monotone (2pp tolerance)? {r2}")

    print("\nR3: does the width ordering weaken as k rises?")
    rhos = {}
    for k in KS:
        s = noisy[noisy.k == k]
        rhos[k] = s[["V", "source_fp"]].corr(method="spearman").iloc[0, 1]
        print(f"   k={k}: Spearman(source_fp, V) = {rhos[k]:+.3f}   mean fp {s.source_fp.mean():.3f}")
    r3 = rhos[KS[-1]] < rhos[KS[0]]
    print(f"   -> ordering {'weakens' if r3 else 'does NOT weaken'} ({rhos[KS[0]]:+.3f} -> {rhos[KS[-1]]:+.3f})")

    pd.DataFrame([{"k": k, "spearman_V": rhos[k],
                   "mean_fp": noisy[noisy.k == k].source_fp.mean()}
                  for k in KS]).to_csv(OUT / "by_k.csv", index=False)

    print("\nVERDICT (rule fixed before running)")
    if r2 and r3:
        print("   -> INTERACTION CHARACTERISED. Redundancy saturates the "
              "failure; width matters\n      only at low redundancy. The "
              "crossed run's disagreement is explained.")
    else:
        print("   -> NOT AS THEORISED. "
              + ("R2 fails: source FP is not monotone in k. " if not r2 else "")
              + ("R3 fails: the width ordering does not weaken. " if not r3 else "")
              + "\n      The crossed disagreement stands unexplained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
