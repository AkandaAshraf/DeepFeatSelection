"""Re-derive the replacement numbers the second review supplies, before they
are written into the paper. Nothing here is taken on trust."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from network_scale import random_dag, simulate  # noqa: E402

BAR = "=" * 76
V, MEMBERS, N, COUPLING = 1000, 100, 2000, 0.3

print(BAR)
print("1. EEG GHOST SIGNS")
print(BAR)
w = pd.read_csv("ExpOutput/eeg_excess/windows.csv")
pos = w[w.ghost > 0]
print(w[["record", "kind", "ghost"]].to_string(index=False))
print(f"\npositive ghosts: {len(pos)}/{len(w)}   max {w.ghost.max():+.6f}   "
      f"min {w.ghost.min():+.6f}   max|.| {w.ghost.abs().max():.6f}")
clean = w.groupby("record").ghost.apply(lambda g: (g <= 0).all())
print(f"records with BOTH windows ghost-clean: "
      f"{sorted(clean[clean].index.tolist())}")

print("\n" + BAR)
print("2. EEG TOTAL DRIVE: does it fall?")
print(BAR)
c = pd.read_csv("ExpOutput/eeg_excess/concentration_null.csv")
piv = c.pivot_table(index="record", columns="kind",
                    values=["total_positive", "n_positive"]).dropna()
tp = piv["total_positive"]
print(tp.to_string())
print(f"\nmean   ictal {tp['ictal'].mean():.3f}  interictal {tp['interictal'].mean():.3f}")
print(f"median ictal {tp['ictal'].median():.3f}  interictal {tp['interictal'].median():.3f}")
print(f"falls in {(tp['ictal'] < tp['interictal']).sum()}/{len(tp)} records")
print(f"paired Wilcoxon two-sided p = "
      f"{wilcoxon(tp['ictal'], tp['interictal']).pvalue:.3f}")
npv = piv["n_positive"]
print(f"\nn_positive: ictal {npv['ictal'].mean():.1f} vs interictal "
      f"{npv['interictal'].mean():.1f}, falls in "
      f"{(npv['ictal'] < npv['interictal']).sum()}/{len(npv)}")
print(f"  one-sided Wilcoxon (ictal < interictal) p = "
      f"{wilcoxon(npv['ictal'], npv['interictal'], alternative='less').pvalue:.3f}")

print("\n" + BAR)
print("3. STRICT GHOST RULE: concentration on ghost-clean records only")
print(BAR)
keep = sorted(clean[clean].index.tolist())
sub = c[c.record.isin(keep)].pivot_table(
    index="record", columns="kind", values="top4_share_clamped").dropna()
print(sub.to_string())
if len(sub) >= 3:
    r = wilcoxon(sub["ictal"], sub["interictal"], alternative="greater")
    print(f"paired one-sided Wilcoxon on {len(sub)} clean records: p = {r.pvalue:.3f}")

print("\n" + BAR)
print("4. IN-DEGREE NULL OVER THREE DISTINCT GRAPHS ONLY")
print(BAR)


def degrees(seed):
    ne = max(MEMBERS, int(1.2 * MEMBERS))
    for a in range(20):
        rng = np.random.default_rng(seed + 100 * a)
        cand = random_dag(MEMBERS, ne, rng)
        try:
            simulate(N, cand, MEMBERS, COUPLING, seed)
            ind = np.zeros(MEMBERS, int)
            for _, j in cand:
                ind[j] += 1
            return ind
        except ValueError:
            continue


tail = pd.read_csv("ExpOutput/recall/ghost_tail.csv")
flag_i, miss_i, zero_flagged, zero_total = [], [], 0, 0
for label, path, seed in [("seed0", "ExpOutput/excess_poly/excess_consensus.npy", 0),
                          ("seed1", "ExpOutput/excess_s1/excess_consensus.npy", 1),
                          ("seed2", "ExpOutput/excess_s2/excess_consensus.npy", 2)]:
    ex = np.load(path)
    ok = tail[(tail.system == label) & (tail.self_predictability > 0.99)]
    thr = max(0.0, float(ok.excess.max()))
    ind = degrees(seed)
    flagged = ex[:MEMBERS] > thr
    driven = ind > 0
    flag_i += list(ind[driven & flagged])
    miss_i += list(ind[driven & ~flagged])
    zero_total += int((~driven).sum())
    zero_flagged += int((~driven & flagged).sum())
f, m = np.array(flag_i), np.array(miss_i)
print(f"flagged driven    n={len(f):3d}  mean in-degree {f.mean():.3f}")
print(f"undetected driven n={len(m):3d}  mean in-degree {m.mean():.3f}")
u1 = mannwhitneyu(f, m, alternative="greater")
u2 = mannwhitneyu(f, m, alternative="two-sided")
print(f"Mann-Whitney one-sided p = {u1.pvalue:.3f}, two-sided p = {u2.pvalue:.3f}")
print(f"in-degree-zero channels: {zero_flagged}/{zero_total} flagged")

print("\n" + BAR)
print("5. ENCODER RECIPE PER DEPLOYMENT (from archived weights)")
print(BAR)
try:
    import h5py
    for tag, wf in [("synthetic", "ExpOutput/ensemble/models/m0.weights.h5"),
                    ("worm", "ExpOutput/celegans_excess/models/m0.weights.h5"),
                    ("climate", "ExpOutput/climate_excess/models/m0.weights.h5")]:
        p = Path(wf)
        if not p.exists():
            print(f"  {tag}: {wf} absent")
            continue
        with h5py.File(p, "r") as h:
            shapes = []

            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
                    shapes.append((name, obj.shape))
            h.visititems(visit)
            enc = [sh for nm, sh in shapes if "encoder" in nm]
            print(f"  {tag}: encoder 2-D shapes {enc[:3]}")
except ImportError:
    print("  h5py unavailable")
