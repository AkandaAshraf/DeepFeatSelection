"""Is there a detection threshold a user without ground truth can apply?

The ghost's own score is not one: it sits at the 4th-22nd percentile of all
scores, so "above the ghost" admits nearly every channel. But the ghost's
score is a draw from the null, so its MAGNITUDE estimates the noise scale of
the statistic on this dataset. This script tests whether a multiple of that
magnitude separates driven members from autonomous channels, and reports
what precision and recall such a rule buys.

Absolute thresholds are included for reference; they are not available to a
user, since the scale of excess depends on the system.

    python scripts/recall_threshold.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from bottleneck_membership import build_system  # noqa: E402

SYSTEMS = [("seed 0", "ExpOutput/excess_poly/excess_consensus.npy", 0),
           ("seed 1", "ExpOutput/excess_s1/excess_consensus.npy", 1),
           ("seed 2", "ExpOutput/excess_s2/excess_consensus.npy", 2)]
V, MEMBERS, N, COUPLING = 1000, 100, 2000, 0.3
MULTIPLES = [1, 2, 5, 10, 20]

rows = []
for label, path, seed in SYSTEMS:
    ex = np.load(path)
    _, truth = build_system(V, MEMBERS, N, COUPLING, seed)
    truth_all = np.append(truth, False)
    g = abs(float(ex[-1]))
    n_true = int(truth_all.sum())

    for m in MULTIPLES:
        thr = m * g
        sel = ex > thr
        tp = int((sel & truth_all).sum())
        fp = int((sel & ~truth_all).sum())
        rows.append({"system": label, "rule": f"{m}x|ghost|", "threshold": thr,
                     "flagged": int(sel.sum()), "true_pos": tp,
                     "false_pos": fp,
                     "precision": tp / max(tp + fp, 1),
                     "recall": tp / n_true})

frame = pd.DataFrame(rows)
out = Path("ExpOutput/recall")
out.mkdir(parents=True, exist_ok=True)
frame.to_csv(out / "ghost_scaled_threshold.csv", index=False)

pd.set_option("display.width", 160)
pd.set_option("display.float_format", "{:.4f}".format)
print("=" * 78)
print("GHOST-SCALED THRESHOLD:  flag variable q if excess(q) > m x |excess(ghost)|")
print("=" * 78)
print(frame.to_string(index=False))

print("\n" + "=" * 78)
print("POOLED OVER THE THREE SYSTEMS")
print("=" * 78)
pool = frame.groupby("rule", sort=False).agg(
    flagged=("flagged", "sum"), true_pos=("true_pos", "sum"),
    false_pos=("false_pos", "sum"))
pool["precision"] = pool.true_pos / (pool.true_pos + pool.false_pos)
pool["recall"] = pool.true_pos / (3 * MEMBERS)
print(pool.to_string())
print(f"\nwrote {out}/ghost_scaled_threshold.csv")
