"""How many driven variables does MACE miss, and which ones?

The paper reports top-k precision but never characterises recall, which is
the question a user faces when acting on a null: if a variable does not
appear in the driven core, does that mean it is autonomous, or that the
statistic could not see it?

Two things are measured here on the synthetic systems, where membership and
drive strength are both known by construction.

1.  The operating curve. Precision and recall at every k, so a user can see
    the price of going deeper into the ranking rather than only the top-10
    figure the paper quotes.

2.  Recall stratified by in-degree, the number of parents a member has in
    the generating graph. In-degree is the available proxy for how much of
    a member's next state is externally determined, and members with
    in-degree zero are pure sources within the web: invisible to a
    drivenness statistic by design rather than by failure. Separating those
    out says how much of the miss rate is designed blindness.

The threshold a user actually has, absent ground truth, is the ghost
channel's score, so that is reported alongside.

    python scripts/recall_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from bottleneck_membership import build_system  # noqa: E402
from network_scale import random_dag, simulate  # noqa: E402

SYSTEMS = [
    ("seed 0", "ExpOutput/excess_poly/excess_consensus.npy", 0),
    ("seed 1", "ExpOutput/excess_s1/excess_consensus.npy", 1),
    ("seed 2", "ExpOutput/excess_s2/excess_consensus.npy", 2),
]
V, MEMBERS, N, COUPLING = 1000, 100, 2000, 0.3
KS = [10, 20, 30, 50, 75, 100, 150, 200]


def edges_for(seed: int) -> list[tuple[int, int]]:
    """Reproduce the edge list build_system accepted for this seed.

    build_system retries with rng seeded ``seed + 100 * attempt`` and stops
    at the first non-divergent draw; the same loop is replayed here so the
    in-degrees match the graph that actually generated the data.
    """
    n_edges = max(MEMBERS, int(1.2 * MEMBERS))
    for attempt in range(20):
        rng = np.random.default_rng(seed + 100 * attempt)
        cand = random_dag(MEMBERS, n_edges, rng)
        try:
            simulate(N, cand, MEMBERS, COUPLING, seed)
            return cand
        except ValueError:
            continue
    raise RuntimeError(f"no non-divergent graph found for seed {seed}")


def main() -> int:
    curves, strat, summary = [], [], []

    for label, path, seed in SYSTEMS:
        ex = np.load(path)
        _, truth = build_system(V, MEMBERS, N, COUPLING, seed)
        truth_all = np.append(truth, False)     # ghost appended, never a member
        ghost = float(ex[-1])
        assert ex.shape == truth_all.shape, (ex.shape, truth_all.shape)

        indeg = np.zeros(MEMBERS, dtype=int)
        for _, j in edges_for(seed):
            indeg[j] += 1

        order = np.argsort(-ex)
        n_true = int(truth_all.sum())
        for k in KS:
            hit = int(truth_all[order[:k]].sum())
            curves.append({"system": label, "k": k, "hits": hit,
                           "precision": hit / k, "recall": hit / n_true})

        # Threshold a user can actually apply: score above the ghost.
        above = ex > ghost
        summary.append({
            "system": label, "ghost": ghost,
            "members_above_ghost": int((above & truth_all).sum()),
            "recall_at_ghost": float((above & truth_all).sum() / n_true),
            "nonmembers_above_ghost": int((above & ~truth_all).sum()),
            "nonmembers": int((~truth_all).sum()),
        })

        detected = ex[:MEMBERS] > ghost
        is_mem = truth[:MEMBERS]
        for d in range(indeg.max() + 1):
            sel = is_mem & (indeg == d)
            if not sel.any():
                continue
            strat.append({"system": label, "in_degree": d, "n": int(sel.sum()),
                          "detected": int((detected & sel).sum()),
                          "recall": float(detected[sel].mean()),
                          "median_excess": float(np.median(ex[:MEMBERS][sel]))})

    cur = pd.DataFrame(curves)
    st = pd.DataFrame(strat)
    sm = pd.DataFrame(summary)
    out = Path("ExpOutput/recall")
    out.mkdir(parents=True, exist_ok=True)
    cur.to_csv(out / "operating_curve.csv", index=False)
    st.to_csv(out / "recall_by_indegree.csv", index=False)
    sm.to_csv(out / "recall_at_ghost.csv", index=False)

    pd.set_option("display.width", 150)
    pd.set_option("display.float_format", "{:.3f}".format)

    print("=" * 76)
    print("OPERATING CURVE  (100 members among 1000 variables + 1 ghost)")
    print("=" * 76)
    piv = cur.pivot_table(index="k", columns="system",
                          values=["precision", "recall"])
    print(piv.to_string())
    print("\nmean over the three systems:")
    print(cur.groupby("k")[["precision", "recall"]].mean().to_string())

    print("\n" + "=" * 76)
    print("AT THE GHOST THRESHOLD  (the rule a user without truth can apply)")
    print("=" * 76)
    print(sm.to_string(index=False))

    print("\n" + "=" * 76)
    print("RECALL BY IN-DEGREE  (parents in the generating graph)")
    print("=" * 76)
    print(st.pivot_table(index="in_degree", columns="system",
                         values="recall").to_string())
    print("\npooled across systems:")
    pool = st.groupby("in_degree").agg(
        members=("n", "sum"), detected=("detected", "sum"))
    pool["recall"] = pool.detected / pool.members
    print(pool.to_string())

    src = int(pool.loc[0, "members"]) if 0 in pool.index else 0
    tot = int(pool.members.sum())
    print(f"\n{src}/{tot} members ({src/tot:.0%}) have in-degree 0: pure "
          "sources within")
    print("the web, invisible to a drivenness statistic by construction.")
    dn = pool[pool.index > 0]
    print(f"recall among genuinely driven members (in-degree > 0): "
          f"{dn.detected.sum()/dn.members.sum():.0%}")
    print(f"\nwrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
