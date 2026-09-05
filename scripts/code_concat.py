"""Is the ensemble leaving readout capacity unused?

Pre-registration: paper/code_concat_protocol.md, committed before this was
written or run.

The deployed estimator trains two encoders and AVERAGES their two gains, so
each readout sees only one width-32 code. Concatenating the same two codes
gives the readout 64 columns for identical encoder cost. Capacity is the
known binding constraint on recall (0.18 -> 0.78 when b goes 32 -> 128), so
this may be free recall -- or estimation variance may eat it, since the
paper's own argument says the raw alternative fails as VE/n grows.

Arms A and B share the SAME two trained encoders and differ only in how the
readout consumes them, so their comparison is exactly paired. Arm C trains
one encoder at double width to separate "readout width" from "ensemble
diversity".

    python scripts/code_concat.py
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
                          MIN_DONORS, N_GHOSTS, embed, make_system, poly3,
                          ridge_r2)
from wormwideweb_gate import MaskedAE  # noqa: E402

OUT = Path("ExpOutput/code_concat")
N, COUPLING, REDUNDANCY = 4000, 0.20, 0
WIDTHS = (30, 60, 120)
SEEDS = (0, 1, 2)
B = 32                 # deployed per-model width; arm C uses 2*B


def train(zs, Vt, tr, b, seed, tag):
    torch.manual_seed(seed * 100 + tag)
    net = MaskedAE(zs.shape[1], b).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(seed * 100 + tag)
    ztr = torch.as_tensor(zs[tr], device=DEV)
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
        return net.enc(torch.as_tensor(zs, device=DEV)).cpu().numpy()


def cell(V, seed):
    """Return one row per arm; A and B share the identical two encoders."""
    x, is_driven, is_source = make_system(N, V, COUPLING, REDUNDANCY, seed)
    Vt = x.shape[1]
    emb = embed(x)
    m = emb.shape[0]
    a, bnd = int(0.6 * m), int(0.8 * m)
    tr, tr_i, te_i = slice(0, a), np.arange(0, a - 1), np.arange(bnd, m - 1)
    mu, sd = emb[tr].mean(0), emb[tr].std(0) + 1e-12
    zs = np.clip(np.nan_to_num((emb - mu) / sd), -20, 20).astype(np.float32)
    lead = zs[:, [j * E for j in range(Vt)]]
    feats = [poly3(zs[:, q * E:(q + 1) * E]) for q in range(Vt)]

    t0 = time.time()
    pair = [train(zs, Vt, tr, B, seed, k) for k in range(2)]   # A and B share
    t_pair = time.time() - t0
    t0 = time.time()
    wide = [train(zs, Vt, tr, 2 * B, seed, 7)]                 # arm C
    t_wide = time.time() - t0

    def make_scorer(codes, mode):
        def excess_of(f, target):
            base = ridge_r2(f[tr_i], target[tr_i + 1],
                            f[te_i], target[te_i + 1])
            if mode == "average":
                return float(np.mean([
                    ridge_r2(np.hstack([f[tr_i], c[tr_i]]), target[tr_i + 1],
                             np.hstack([f[te_i], c[te_i]]), target[te_i + 1])
                    - base for c in codes]))
            C = np.hstack(codes)          # concat / single wide code
            return float(ridge_r2(np.hstack([f[tr_i], C[tr_i]]),
                                  target[tr_i + 1],
                                  np.hstack([f[te_i], C[te_i]]),
                                  target[te_i + 1]) - base)
        return excess_of

    self_r2 = np.array([ridge_r2(f[tr_i], lead[tr_i + 1, q],
                                 f[te_i], lead[te_i + 1, q])
                        for q, f in enumerate(feats)])

    rows = []
    for arm, codes, mode, secs in (("AVERAGE", pair, "average", t_pair),
                                   ("CONCAT", pair, "concat", t_pair),
                                   ("WIDE", wide, "concat", t_wide)):
        t1 = time.time()
        score = make_scorer(codes, mode)
        excess = np.array([score(feats[q], lead[:, q]) for q in range(Vt)])
        # each arm gets its own ghost panel under its own aggregation
        rng = np.random.default_rng(seed + 4242)
        qual = np.where(self_r2 > DONOR_R2)[0]
        pool = np.arange(Vt) if len(qual) < MIN_DONORS else qual
        donors = rng.choice(pool, size=min(N_GHOSTS, len(pool)),
                            replace=False)
        ghosts = np.array([
            score(poly3(np.roll(zs[:, d * E:(d + 1) * E], s_, axis=0)),
                  np.roll(lead[:, d], s_))
            for d, s_ in zip(donors,
                             rng.integers(m // 4, 3 * m // 4, len(donors)))])
        thr = max(0.0, float(ghosts.max()))
        flagged = excess > thr
        rows.append({
            "V": V, "seed": seed, "arm": arm,
            "readout_width": B if arm == "AVERAGE" else 2 * B,
            "recall": float((flagged & is_driven).sum()
                            / max(int(is_driven.sum()), 1)),
            "precision": float((flagged & is_driven).sum()
                               / max(int(flagged.sum()), 1)),
            "source_fp": float((flagged & is_source).sum()
                               / max(int(is_source.sum()), 1)),
            "n_flagged": int(flagged.sum()),
            "ghost_med": float(np.median(ghosts)),
            "ghost_ok": bool(np.median(ghosts) <= 0.005),
            "encoder_secs": secs, "readout_secs": time.time() - t1,
        })
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEV}   b={B} per model   arms: AVERAGE / CONCAT / WIDE")
    print(f"{len(WIDTHS)*len(SEEDS)} cells x 3 arms\n")
    rows, t0 = [], time.time()
    for V in WIDTHS:
        for s in SEEDS:
            rows += cell(V, s)
            r = rows[-3:]
            print(f"  V={V:<4} seed={s}  " + "  ".join(
                f"{x['arm']} rec {x['recall']:.2f}" for x in r)
                + f"   ({(time.time()-t0)/60:.1f} min)", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "cells.csv", index=False)
    print(f"\n({(time.time()-t0)/60:.1f} min)\n")

    piv = d.pivot_table(index="V", columns="arm", values="recall",
                        aggfunc="median")[["AVERAGE", "CONCAT", "WIDE"]]
    print("RECALL (median over 3 seeds)")
    print("   " + piv.round(3).to_string().replace("\n", "\n   "))

    print("\nC3 DISQUALIFYING: precision and source false positives")
    for arm, g in d.groupby("arm"):
        print(f"   {arm:8s} precision min {g.precision.min():.3f}   "
              f"source_fp max {g.source_fp.max():.3f}")
    c3 = bool((d.precision.min() >= 0.999) and (d.source_fp.max() <= 1e-9))
    print(f"   -> {'HOLDS' if c3 else 'FAILS - concatenation is REJECTED'}")

    print(f"\nC4 ghost clean in every cell: "
          f"{'YES' if bool(d.ghost_ok.all()) else 'NO'}")

    print("\nC6 cost")
    for arm, g in d.groupby("arm"):
        print(f"   {arm:8s} encoder {g.encoder_secs.median():6.1f}s   "
              f"readout {g.readout_secs.median():5.1f}s")

    v60 = d[d.V == 60].pivot_table(index="seed", columns="arm",
                                   values="recall")
    avg, con = v60["AVERAGE"].median(), v60["CONCAT"].median()
    wide = v60["WIDE"].median()
    print(f"\nC1 DECISIVE at V=60: AVERAGE {avg:.3f}  CONCAT {con:.3f}"
          f"  (WIDE {wide:.3f})")
    print("   per seed: " + "  ".join(
        f"s{i}: {v60.loc[i,'AVERAGE']:.2f}->{v60.loc[i,'CONCAT']:.2f}"
        for i in v60.index))

    print("\nVERDICT (rule fixed before running)")
    if not c3:
        print("   -> REJECT. Concatenation broke precision or source "
              "blindness; averaging stands.")
    elif con > avg:
        print(f"   -> ADOPT. CONCAT {con:.3f} > AVERAGE {avg:.3f} at V=60 "
              "with precision and source\n      blindness intact, at no "
              "extra encoder cost. Part of the published recall\n"
              "      shortfall was an aggregation choice.")
        print(f"   C2 (no prediction): CONCAT {con:.3f} vs WIDE {wide:.3f} -> "
              + ("diversity adds beyond width" if con > wide
                 else "width explains it; one wide code suffices"))
    else:
        print(f"   -> NULL. CONCAT {con:.3f} does not beat AVERAGE {avg:.3f}. "
              "Averaging is not\n      leaving capacity on the table and the "
              "recall limit is not an aggregation\n      artefact. Incumbent "
              "stands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
