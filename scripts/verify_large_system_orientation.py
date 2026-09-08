"""Recompute every ap_source value in ExpOutput/large_system/cells.csv,
arm by arm, width by width, from the archived raw_*.npz score arrays. Confirms
the AP orientation convention (-score for a SOURCE-positive AP) is applied
identically by every arm and matches what is on disk exactly.

Written for the 2026-09-08 scale-diagnosis; read-only, no dataset, checkpoint
or result file touched.

    python scripts/verify_large_system_orientation.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

d = pd.read_csv("ExpOutput/large_system/cells.csv")
mismatches = []
checked = 0

for V in [120, 240, 500, 1000]:
    for seed in [100, 101, 102]:
        z = np.load(f"ExpOutput/large_system/raw_V{V}_s{seed}.npz")
        is_source = z["is_source"]
        for arm in ["FLAT", "SELFR2", "LAGCORR", "HIER-CLUST-TRAIN",
                   "HIER-RAND-SIZED", "HIER-TRUE"]:
            recorded = d[(d.V == V) & (d.seed == seed) & (d.arm == arm)].ap_source
            if len(recorded) == 0:
                mismatches.append((V, seed, arm, "MISSING FROM cells.csv"))
                continue
            recorded = float(recorded.values[0])
            if arm in ("HIER-CLUST-TRAIN", "HIER-RAND-SIZED", "HIER-TRUE"):
                score = z[f"{arm}_e2"] + z[f"{arm}_e3"]
            else:
                score = z[arm]
            recomputed = average_precision_score(is_source, -score)
            checked += 1
            if abs(recomputed - recorded) > 1e-9:
                mismatches.append((V, seed, arm, f"recorded={recorded:.6f} "
                                  f"recomputed={recomputed:.6f}"))

print(f"checked {checked} of 72 (V x seed x arm) rows")
if mismatches:
    print(f"MISMATCHES: {len(mismatches)}")
    for m in mismatches:
        print(" ", m)
else:
    print("ALL MATCH EXACTLY. No orientation or transcription defect in "
          "cells.csv's ap_source column, any arm, any width, any seed.")
