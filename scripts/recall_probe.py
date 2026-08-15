"""What kind of member does MACE actually find, and where does the ranking
stop being informative?

Two facts from the operating curve need explaining before either can be
written up.

1.  Nearly every non-member scores above the ghost, so the ghost cannot be
    a detection threshold; it is a validity check only. Quantify where the
    ghost actually sits in the score distribution.

2.  Precision is perfect at k=10 and collapses below chance further down
    (AUROC 0.28). So the bulk of members rank BELOW the bulk of
    non-members while a handful rank at the very top. Characterise the two
    populations: what distinguishes a member MACE finds from one it buries?

    python scripts/recall_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from bottleneck_membership import build_system  # noqa: E402
from network_scale import random_dag, simulate  # noqa: E402

SYSTEMS = [("seed 0", "ExpOutput/excess_poly/excess_consensus.npy", 0),
           ("seed 1", "ExpOutput/excess_s1/excess_consensus.npy", 1),
           ("seed 2", "ExpOutput/excess_s2/excess_consensus.npy", 2)]
V, MEMBERS, N, COUPLING = 1000, 100, 2000, 0.3


def edges_for(seed):
    n_edges = max(MEMBERS, int(1.2 * MEMBERS))
    for attempt in range(20):
        rng = np.random.default_rng(seed + 100 * attempt)
        cand = random_dag(MEMBERS, n_edges, rng)
        try:
            simulate(N, cand, MEMBERS, COUPLING, seed)
            return cand
        except ValueError:
            continue
    raise RuntimeError(seed)


rows, comp = [], []
for label, path, seed in SYSTEMS:
    ex = np.load(path)
    _, truth = build_system(V, MEMBERS, N, COUPLING, seed)
    truth_all = np.append(truth, False)
    ghost = float(ex[-1])
    indeg = np.zeros(MEMBERS, dtype=int)
    outdeg = np.zeros(MEMBERS, dtype=int)
    for i, j in edges_for(seed):
        indeg[j] += 1
        outdeg[i] += 1

    pct = float((ex < ghost).mean() * 100)
    rows.append({"system": label, "ghost": ghost,
                 "ghost_percentile": pct,
                 "member_median": float(np.median(ex[:MEMBERS])),
                 "nonmember_median": float(np.median(ex[MEMBERS:V])),
                 "member_max": float(ex[:MEMBERS].max()),
                 "nonmember_max": float(ex[MEMBERS:V].max())})

    order = np.argsort(-ex)
    for k in (10, 30):
        top = order[:k]
        mem = top[top < MEMBERS]
        comp.append({"system": label, "k": k, "members_in_top": len(mem),
                     "mean_indeg": float(indeg[mem].mean()) if len(mem) else np.nan,
                     "mean_outdeg": float(outdeg[mem].mean()) if len(mem) else np.nan,
                     "all_member_indeg": float(indeg.mean()),
                     "all_member_outdeg": float(outdeg.mean())})

    # Rank of every member, split by whether it has parents.
    rank = np.empty(len(ex), dtype=int)
    rank[order] = np.arange(len(ex))
    print(f"\n--- {label} --- ghost at percentile {pct:.1f} of all scores")
    for d in range(indeg.max() + 1):
        sel = indeg == d
        print(f"  in-degree {d}: n={sel.sum():3d}  "
              f"median rank {np.median(rank[:MEMBERS][sel]):6.0f}  "
              f"best rank {rank[:MEMBERS][sel].min():4d}  "
              f"median excess {np.median(ex[:MEMBERS][sel]):+.5f}")
    nm_rank = rank[MEMBERS:V]
    print(f"  non-members: n={len(nm_rank)}  "
          f"median rank {np.median(nm_rank):6.0f}  "
          f"median excess {np.median(ex[MEMBERS:V]):+.5f}")
    # Out-degree: are the found members the hubs?
    hi = outdeg >= 3
    if hi.any():
        print(f"  members with out-degree>=3: n={hi.sum()}, "
              f"median rank {np.median(rank[:MEMBERS][hi]):.0f}")

pd.set_option("display.width", 160)
pd.set_option("display.float_format", "{:.5f}".format)
print("\n" + "=" * 76)
print("WHERE THE GHOST SITS")
print("=" * 76)
print(pd.DataFrame(rows).to_string(index=False))
print("\n" + "=" * 76)
print("WHAT IS IN THE TOP-k")
print("=" * 76)
print(pd.DataFrame(comp).to_string(index=False))
