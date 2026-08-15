"""The corrected ghost panel: donors must be self-predictable.

The null a ghost is supposed to certify follows from saturation of the
self-baseline: if a channel's own history already determines its next value,
no code can add anything and its excess is zero for any code whatsoever. A
circularly shifted copy of such a channel inherits that property, so its
excess is zero by the same argument.

That argument needs the donor to be autonomous. A shifted copy of a DRIVEN
channel inherits a self-baseline that does not saturate, leaving room for a
code to score against it, and the null is no longer guaranteed. The archived
runs drew their single ghost from the coupled web, which is exactly the
population where the guarantee fails, and the measured tail shows it: every
ghost above +0.001 came from a driven donor.

Donor autonomy is not observable, but the property the argument actually
needs -- a saturating self-baseline -- is directly measurable without any
ground truth. This script re-derives the detection threshold from a panel
restricted to self-predictable donors and reports what it buys.

    python scripts/ghost_corrected.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from bottleneck_membership import build_system, build_system_hetero  # noqa: E402
from network_scale import random_dag, simulate  # noqa: E402

V, MEMBERS, N, COUPLING = 1000, 100, 2000, 0.3
CUT = 0.99          # donor self-predictability required to qualify as a ghost
CASES = [("seed0", 0, False, "ExpOutput/excess_poly/excess_consensus.npy"),
         ("seed1", 1, False, "ExpOutput/excess_s1/excess_consensus.npy"),
         ("seed2", 2, False, "ExpOutput/excess_s2/excess_consensus.npy"),
         ("hetero", 0, True, "ExpOutput/excess_het/excess_consensus.npy")]


def degrees(seed):
    n_edges = max(MEMBERS, int(1.2 * MEMBERS))
    for attempt in range(20):
        rng = np.random.default_rng(seed + 100 * attempt)
        cand = random_dag(MEMBERS, n_edges, rng)
        try:
            simulate(N, cand, MEMBERS, COUPLING, seed)
            ind = np.zeros(MEMBERS, int)
            outd = np.zeros(MEMBERS, int)
            for i, j in cand:
                outd[i] += 1
                ind[j] += 1
            return ind, outd
        except ValueError:
            continue
    raise RuntimeError(seed)


tail = pd.read_csv("ExpOutput/recall/ghost_tail.csv")
rows, roles = [], []

for label, seed, hetero, path in CASES:
    ex = np.load(path)
    builder = build_system_hetero if hetero else build_system
    _, truth = builder(V, MEMBERS, N, COUPLING, seed)
    truth_all = np.append(truth, False)
    sub = tail[tail.system == label]
    ok = sub[sub.self_predictability > CUT]
    bad = sub[sub.self_predictability <= CUT]

    # Theory bounds an autonomous channel's excess by zero, so a panel
    # maximum that lands below zero reflects sampling, not a looser null;
    # the threshold is floored there rather than trusted downward.
    thr = max(0.0, float(ok.excess.max()))
    sel = ex > thr
    tp = int((sel & truth_all).sum())
    fp = int((sel & ~truth_all).sum())
    rows.append({"system": label, "qualifying_donors": len(ok),
                 "rejected_donors": len(bad), "threshold": thr,
                 "true_pos": tp, "false_pos": fp,
                 "precision": tp / max(tp + fp, 1)})

    ind, outd = degrees(seed)
    role = np.where(ind > 0, "driven", np.where(outd > 0, "source", "isolated"))
    flag = ex[:MEMBERS] > thr
    for nm in ("driven", "source", "isolated"):
        s = role == nm
        if s.any():
            roles.append({"system": label, "role": nm, "n": int(s.sum()),
                          "flagged": int(flag[s].sum())})

frame = pd.DataFrame(rows)
rf = pd.DataFrame(roles)
frame.to_csv("ExpOutput/recall/corrected_threshold.csv", index=False)
rf.to_csv("ExpOutput/recall/corrected_roles.csv", index=False)

pd.set_option("display.width", 170)
pd.set_option("display.float_format", "{:.6f}".format)
print("=" * 88)
print(f"THRESHOLD FROM A PANEL OF SELF-PREDICTABLE DONORS (self-pred > {CUT})")
print("=" * 88)
print(frame.to_string(index=False))
pool = frame[["true_pos", "false_pos"]].sum()
print(f"\npooled: {int(pool.true_pos)} true positives, "
      f"{int(pool.false_pos)} false positives, precision "
      f"{pool.true_pos/(pool.true_pos+pool.false_pos):.3f}")

print("\n" + "=" * 88)
print("RECALL BY ROLE IN THE GENERATING GRAPH")
print("=" * 88)
p = rf.groupby("role").agg(n=("n", "sum"), flagged=("flagged", "sum"))
p["recall"] = p.flagged / p.n
print(p.to_string())
d = p.loc["driven"]
print(f"\nrecall among genuinely driven channels: {int(d.flagged)}/{int(d.n)} "
      f"= {d.flagged/d.n:.0%}")
print("sources and isolated channels are silent by design, and are: "
      f"{int(p.loc['source','flagged'])} and "
      f"{int(p.loc['isolated','flagged'])} flagged respectively.")
