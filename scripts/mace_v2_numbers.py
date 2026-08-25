"""Audit gate for paper/mace_v2.tex: re-derive every quoted number.

The companion DepMap paper has scripts/depmap_paper_numbers.py as its audit
gate; this is the equivalent for the v2 manuscript's new sections. Each check
re-derives a number from its primary output file and compares it to the value
written in the manuscript. Any MISMATCH is a defect in the paper, not here.

    python scripts/mace_v2_numbers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OK, BAD, SKIP = [], [], []


def chk(label, claimed, actual, tol=5e-4):
    if actual is None:
        SKIP.append(f"{label}: source file missing")
        return
    good = abs(float(claimed) - float(actual)) <= tol
    (OK if good else BAD).append(
        f"{label}: paper {claimed}  file {actual:.4f}"
        + ("" if good else "   <-- MISMATCH"))


def read(p):
    p = Path(p)
    return pd.read_csv(p) if p.exists() else None


# ---------------------------------------------------------------- boundary
d = read("ExpOutput/boundary_map/boundary_map.csv")
if d is None:
    SKIP.append("boundary map: ExpOutput/boundary_map/boundary_map.csv missing")
else:
    chk("boundary: cells = 51", 51, len(d), tol=0)
    chk("boundary: source FP 0.000 in every cell", 0.0, d.source_fp.max())
    # the paper's Table 4 is the V SWEEP: other axes held at their centre
    # (n=4000, coupling=0.2, redundancy=0). Pooling all cells at a given V
    # would mix in the coupling and redundancy sweeps.
    sl = d[(d.n == 4000) & (d.coupling == 0.2) & (d.redundancy == 0)]
    g = sl.groupby("V").agg(flag=("n_flagged", "median"),
                            rec=("recall", "median"))
    for V, claim in [(15, 1.00), (30, 0.88), (60, 0.18), (120, 0.23),
                     (240, 0.14)]:
        if V in g.index:
            chk(f"boundary: recall at V={V}", claim, g.loc[V, "rec"], tol=0.02)
    if "coupling" in d:
        c = d.groupby("coupling").recall.median()
        if 0.50 in c.index:
            chk("boundary: recall collapses to 0.06 at coupling 0.50",
                0.06, c.loc[0.50], tol=0.02)

# -------------------------------------------------------------- bottleneck
d = read("ExpOutput/bottleneck/bottleneck.csv")
if d is None:
    d = read("ExpOutput/bottleneck_scaling/bottleneck_scaling.csv")
if d is None:
    SKIP.append("bottleneck: output csv missing")
else:
    chk("bottleneck: cells = 90", 90, len(d), tol=0)
    f = d[d.code == "float"] if "code" in d else d
    g = f.groupby(["V", "b"]).recall.median()
    for (V, b), claim in [((60, 32), 0.18), ((60, 128), 0.78),
                          ((30, 32), 0.88), ((120, 128), 0.41)]:
        if (V, b) in g.index:
            chk(f"bottleneck: recall V={V} b={b}", claim, g.loc[(V, b)],
                tol=0.02)
    if "precision" in d:
        chk("bottleneck: precision 1.00 in all cells", 1.0, d.precision.min())
    if "source_fp" in d:
        chk("bottleneck: source FP 0.000 in all cells", 0.0, d.source_fp.max())
    if "ghost_max" in d:
        # the paper's claim is that the MEDIAN of ghost_max is flat across
        # widths; the per-cell maximum is reported separately beside it
        med = d.groupby("b").ghost_max.median()
        chk("bottleneck: median ghost_max flat, low end", 0.0012, med.min(),
            tol=2e-4)
        chk("bottleneck: median ghost_max flat, high end", 0.0015, med.max(),
            tol=2e-4)
        chk("bottleneck: worst-cell ghost_max", 0.0030, d.ghost_max.max(),
            tol=2e-4)

# ------------------------------------------------------------------ outflow
d = read("ExpOutput/source_outflow/coupling_sweep.csv")
if d is None:
    SKIP.append("outflow coupling sweep missing")
else:
    g = d.groupby("coupling").agg(m=("margin", "median"),
                                  sink=("out_sink", "median"))
    chk("outflow: margin at coupling 0.50", 0.0107, g.loc[0.50, "m"], tol=1e-3)
    chk("outflow: margin at coupling 0.70", 0.0103, g.loc[0.70, "m"], tol=1e-3)
    chk("outflow: sink proxy at 0.50", 0.0027, g.loc[0.50, "sink"], tol=1e-3)
    chk("outflow: sink proxy at 0.70", 0.0062, g.loc[0.70, "sink"], tol=1e-3)

# ------------------------------------------------------ conditional outflow
d = read("ExpOutput/conditional_outflow/channels.csv")
if d is None:
    SKIP.append("conditional outflow channels.csv missing")
else:
    sys.path.insert(0, "scripts")
    from error_metrics import auc  # noqa: E402
    d0 = d[d.copies == 0]
    for c, a1, c1 in [(0.30, 0.768, 0.856), (0.50, 0.655, 0.877),
                      (0.70, 0.517, 0.872)]:
        g = d0[d0.coupling == c]
        s, k = g[g.role == "source"], g[g.role == "sink"]
        chk(f"conditional: A1 AUC at coupling {c}", a1,
            auc(s.A1.values, k.A1.values), tol=0.01)
        chk(f"conditional: C1 AUC at coupling {c}", c1,
            auc(s.C1.values, k.C1.values), tol=0.01)

# ------------------------------------------------------------ chamber, real
d = read("ExpOutput/real_conditional/summary.csv")
if d is None:
    SKIP.append("real_conditional summary.csv missing")
else:
    p = d[d.dataset == "wt_intake_impulse_v1"]
    if len(p):
        r = p.iloc[0]
        chk("chamber: A1 marginal AUC (primary)", 0.916, r.A1_auc, tol=0.005)
        chk("chamber: C1 conditional AUC (primary)", 0.751, r.C1_auc,
            tol=0.005)
    s = d[d.dataset == "wt_walks_v1"]
    if len(s):
        r = s.iloc[0]
        chk("chamber: A1 AUC (secondary)", 0.605, r.A1_auc, tol=0.005)
        chk("chamber: C1 AUC (secondary)", 0.632, r.C1_auc, tol=0.005)

# ------------------------------------------------------------ fitness gate
d = read("ExpOutput/dataset_fitness/reference.csv")
if d is None:
    SKIP.append("dataset_fitness reference.csv missing")
else:
    r = d.set_index("coupling").lag_info
    chk("fitness: L50 pass mark", 0.0136, r.loc[0.50], tol=2e-4)
    chk("fitness: L30 floor", 0.0114, r.loc[0.30], tol=2e-4)

d = read("ExpOutput/chamber_screen/screen.csv")
if d is None:
    SKIP.append("chamber_screen/screen.csv missing")
else:
    q = d[(d.dataset == "wt_intake_impulse_v1") & (d.m == 10)]
    if len(q):
        chk("fitness: impulse dataset at m=10", 0.0224,
            q.iloc[0].lag_info, tol=5e-4)

# ---------------------------------------------------------------- report
print("MACE v2 audit gate\n")
for line in OK:
    print("  PASS  " + line)
for line in BAD:
    print("  FAIL  " + line)
for line in SKIP:
    print("  SKIP  " + line)
print(f"\n{len(OK)} passed, {len(BAD)} failed, {len(SKIP)} skipped")
if BAD:
    print("\nA FAIL means the manuscript disagrees with its own output file. "
          "Fix the paper.")
raise SystemExit(1 if BAD else 0)
