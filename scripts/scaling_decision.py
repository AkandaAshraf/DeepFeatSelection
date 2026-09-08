"""Does any DEPLOYABLE learned arm beat the two cheap baselines at V=500/1000?

Closing analysis for the large-system scaling investigation (large_system,
the 2026-09-08 diagnosis, ridge_alpha_scaling). Read-only against archived
files; no retraining, no dataset touched, no raw arrays copied out.

    python scripts/scaling_decision.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

cells = pd.read_csv("ExpOutput/large_system/cells.csv")
rows = []
for V in [120, 240, 500, 1000]:
    for seed in [100, 101, 102]:
        z = np.load(f"ExpOutput/large_system/raw_V{V}_s{seed}.npz")
        is_source = z["is_source"]
        prevalence = float(is_source.mean())

        for arm in ["FLAT", "SELFR2", "LAGCORR"]:
            recorded = float(cells[(cells.V == V) & (cells.seed == seed)
                                   & (cells.arm == arm)].ap_source.iloc[0])
            recomputed = average_precision_score(is_source, -z[arm])
            assert abs(recorded - recomputed) < 1e-9, (V, seed, arm,
                                                        recorded, recomputed)
            rows.append(dict(V=V, seed=seed, arm=arm, ap=recomputed,
                             prevalence=prevalence, deployable=True))

        e2 = z["HIER-CLUST-TRAIN_e2"]
        e3 = z["HIER-CLUST-TRAIN_e3"]
        recorded = float(cells[(cells.V == V) & (cells.seed == seed) &
                               (cells.arm == "HIER-CLUST-TRAIN")]
                        .ap_source.iloc[0])
        deployed_ap = average_precision_score(is_source, -(e2 + e3))
        assert abs(recorded - deployed_ap) < 1e-9, (V, seed, "deployed",
                                                     recorded, deployed_ap)
        e2_only_ap = average_precision_score(is_source, -e2)
        rows.append(dict(V=V, seed=seed, arm="HIER-CLUST-TRAIN (deployed)",
                         ap=deployed_ap, prevalence=prevalence,
                         deployable=True))
        rows.append(dict(V=V, seed=seed,
                         arm="HIER-CLUST-TRAIN (e2-only, POST HOC)",
                         ap=e2_only_ap, prevalence=prevalence,
                         deployable=True))

t = pd.DataFrame(rows)   # in-memory only, nothing written to the repo

print("ALL RECORDED VALUES MATCHED RECOMPUTED VALUES EXACTLY (asserts passed)\n")

ARMS = ["FLAT", "HIER-CLUST-TRAIN (deployed)",
       "HIER-CLUST-TRAIN (e2-only, POST HOC)", "SELFR2", "LAGCORR"]

print("=== PER-SEED, V=500 and V=1000 (the decisive widths) ===")
for V in [500, 1000]:
    print(f"\nV={V}  prevalence={t[t.V==V].prevalence.iloc[0]:.3f}")
    piv = t[t.V == V].pivot_table(index="seed", columns="arm", values="ap")
    print(piv[ARMS].round(4).to_string())

print("\n=== MEAN OVER 3 SEEDS, ALL FOUR WIDTHS ===")
piv_all = t.pivot_table(index="V", columns="arm", values="ap", aggfunc="mean")
print(piv_all[ARMS].round(4).to_string())

print("\n=== CROSSOVER CHECK, ALL FOUR WIDTHS, PER-SEED COUNTS ===")
for V in [120, 240, 500, 1000]:
    sub = t[t.V == V].pivot_table(index="seed", columns="arm", values="ap")
    for arm in ["FLAT", "HIER-CLUST-TRAIN (deployed)"]:
        beats_both = ((sub[arm] > sub["SELFR2"]) &
                     (sub[arm] > sub["LAGCORR"])).sum()
        print(f"  V={V:<5} {arm:30s} beats BOTH in {beats_both}/3 seeds")

print("\n=== DOES ANY DEPLOYABLE ARM BEAT BOTH BASELINES, PER SEED? ===")
for V in [500, 1000]:
    sub = t[t.V == V].pivot_table(index="seed", columns="arm", values="ap")
    for arm in ["FLAT", "HIER-CLUST-TRAIN (deployed)"]:
        beats_both = ((sub[arm] > sub["SELFR2"]) &
                     (sub[arm] > sub["LAGCORR"])).sum()
        print(f"  V={V}  {arm:38s} beats BOTH baselines in {beats_both}/3 seeds")
    # e2-only labelled separately: post hoc, not a deployed rule
    arm = "HIER-CLUST-TRAIN (e2-only, POST HOC)"
    beats_both = ((sub[arm] > sub["SELFR2"]) &
                 (sub[arm] > sub["LAGCORR"])).sum()
    print(f"  V={V}  {arm:38s} beats BOTH baselines in {beats_both}/3 seeds"
          f"  [POST HOC, no deployed rule uses this score]")
