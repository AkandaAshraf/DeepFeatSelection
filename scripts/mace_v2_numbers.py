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

d = read("ExpOutput/chamber_cluster_check/summary.csv")
if d is None:
    SKIP.append("chamber_cluster_check summary.csv missing")
else:
    r = d.iloc[0]
    chk("chamber: run-cluster CI lower bound", 0.862, r.ci_lo, tol=0.01)
    chk("chamber: run-cluster CI upper bound", 0.962, r.ci_hi, tol=0.01)
    chk("chamber: run-clustering variance ratio", 3.45, r.clustering_ratio,
        tol=0.1)

d = read("ExpOutput/ccm_pcmci_baseline/results_corrected.csv")
if d is None:
    SKIP.append("ccm_pcmci results_corrected.csv missing")
else:
    r = d.set_index("method").membership_auroc
    chk("baseline: CCM membership AUROC", 1.000, r.loc["CCM"], tol=0.005)
    chk("baseline: PCMCI membership AUROC", 0.728, r.loc["PCMCI"], tol=0.005)
    chk("baseline: MACE membership AUROC (same cell)", 1.000, r.loc["MACE"],
        tol=0.005)
    e = d.set_index("method").true_edge_auroc
    chk("baseline: CCM true-edge AUROC", 1.000, e.loc["CCM"], tol=0.005)
    chk("baseline: PCMCI true-edge AUROC", 0.984, e.loc["PCMCI"], tol=0.005)

d = read("ExpOutput/ccm_pcmci_v60/results.csv")
if d is None:
    SKIP.append("ccm_pcmci_v60 results.csv missing")
else:
    g = d.groupby("method").membership_auroc.median()
    chk("V60: MACE membership AUROC (median)", 0.976, g.loc["MACE"], tol=0.005)
    chk("V60: CCM membership AUROC (median)", 0.674, g.loc["CCM"], tol=0.005)
    chk("V60: PCMCI membership AUROC (median)", 0.590, g.loc["PCMCI"],
        tol=0.005)
    e = d.groupby("method").true_edge_auroc.median()
    chk("V60: CCM true-edge AUROC (median)", 0.984, e.loc["CCM"], tol=0.005)
    chk("V60: PCMCI true-edge AUROC (median)", 0.950, e.loc["PCMCI"],
        tol=0.005)
    w = d[d.method == "CCM"].membership_auroc
    chk("V60: CCM worst seed at chance", 0.490, w.min(), tol=0.005)

d = read("ExpOutput/sink_bar/summary.csv")
if d is None:
    SKIP.append("sink_bar summary.csv missing")
else:
    r = d.iloc[0]
    chk("sink bar: q95(sink) calibrated bar", 0.01051, r.bar, tol=5e-5)
    chk("sink bar: sensitivity at coupling 0.50", 0.53, r.sensitivity_c050,
        tol=0.02)
    chk("sink bar: source median at 0.50", 0.01064, r.source_median_c050,
        tol=5e-5)

d = read("ExpOutput/crossed_saturation/cells.csv")
if d is None:
    SKIP.append("crossed_saturation cells.csv missing")
else:
    chk("crossed sat: cells", 90, len(d), tol=0)
    chk("crossed sat: ghost-dirty cells", 0, int((~d.ghost_ok).sum()), tol=0)
    n = d[(d.k == 2) & (d.noise > 0)]
    chk("crossed sat: Spearman(rate,V) at k=2", 0.002,
        n[["V", "source_fp"]].corr(method="spearman").iloc[0, 1], tol=0.02)
    z = d[(d.k == 0) & (d.noise > 0)]
    chk("crossed sat: Spearman(rate,V) at k=0", 0.672,
        z[["V", "source_fp"]].corr(method="spearman").iloc[0, 1], tol=0.02)
    chk("crossed sat: recall at V=60, b=2V, no noise", 0.61,
        d[(d.V == 60) & (d.noise == 0)].recall.median(), tol=0.02)

d = read("ExpOutput/aggregation_check/results.csv")
if d is None:
    SKIP.append("aggregation_check results.csv missing")
else:
    c = d[(d.V == 60) & (d.method == "CCM")].groupby("aggregation").auroc.median()
    chk("aggregation: CCM MAX at V=60", 0.674, c.loc["MAX"], tol=0.005)
    chk("aggregation: CCM MEAN at V=60", 0.680, c.loc["MEAN"], tol=0.005)
    chk("aggregation: best alternative still below MACE", True,
        float(c.drop("MAX").max()) < 0.976, tol=0)

d = read("ExpOutput/chamber_shape/results.csv")
if d is None:
    SKIP.append("chamber_shape results.csv missing")
else:
    r = d.set_index("shape")
    syn = [i for i in r.index if i.startswith("synthetic")][0]
    cham = [i for i in r.index if i.startswith("chamber (")][0]
    ch15 = [i for i in r.index if "V=15" in i][0]
    chk("shape: synthetic sensitivity", 0.667, r.loc[syn, "sensitivity"],
        tol=0.01)
    chk("shape: chamber sensitivity", 0.867, r.loc[cham, "sensitivity"],
        tol=0.01)
    chk("shape: chamber at V=15 sensitivity", 0.967,
        r.loc[ch15, "sensitivity"], tol=0.01)
    chk("shape: chamber AUC", 0.864, r.loc[cham, "auc_median"], tol=0.01)
    chk("shape: AUC never exceeds sensitivity (C3 failed)", True,
        bool((r.auc_median <= r.sensitivity + 1e-9).all()), tol=0)

d = read("ExpOutput/redundancy_axis/by_k.csv")
if d is None:
    SKIP.append("redundancy_axis by_k.csv missing")
else:
    r = d.set_index("k")
    chk("redundancy: Spearman(fp,V) at k=0", 0.681, r.loc[0, "spearman_V"],
        tol=0.01)
    chk("redundancy: Spearman(fp,V) at k=8", -0.159, r.loc[8, "spearman_V"],
        tol=0.01)
    chk("redundancy: mean fp at k=0", 0.122, r.loc[0, "mean_fp"], tol=0.005)
    chk("redundancy: mean fp at k=8", 0.309, r.loc[8, "mean_fp"], tol=0.005)
    chk("redundancy: fp monotone in k", True,
        bool(all(r.mean_fp.iloc[i] <= r.mean_fp.iloc[i+1] + 0.02
                 for i in range(len(r)-1))), tol=0)

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

# ------------------------------------------------- reopen at the chamber shape
d = read("ExpOutput/chamber_shape_reopen/summary.csv")
if d is None:
    SKIP.append("chamber_shape_reopen summary.csv missing")
else:
    r = d.iloc[0]
    chk("reopen: calibrated bar", 0.01498, r.bar, tol=5e-5)
    chk("reopen: sensitivity at 2/11/2", 0.833, r.sensitivity, tol=0.01)
    chk("reopen: AUC", 0.864, r.auc_median, tol=0.01)
    chk("reopen: source median", 0.02312, r.source_med, tol=5e-5)
    chk("reopen: clears the declared 0.80 bar", True,
        bool(r.sensitivity >= 0.80), tol=0)

d = read("ExpOutput/chamber_shape_reopen/runs.csv")
if d is None:
    SKIP.append("chamber_shape_reopen runs.csv missing")
else:
    te = d[d.set == "test"]
    bar = float(np.quantile(d[d.set == "cal"].sink, 0.95))
    chk("reopen: test runs clearing the bar", 25, int((te.source > bar).sum()),
        tol=0)
    chk("reopen: dead test runs", 2, int((te.source < 0.002).sum()), tol=0)

# ------------------------------------------------------------- ratio sweep
d = read("ExpOutput/ratio_sweep/results.csv")
if d is None:
    SKIP.append("ratio_sweep results.csv missing")
else:
    r = d.sort_values("ratio").reset_index(drop=True)
    chk("ratio sweep: T2 Spearman(C1 AUC, ratio)", 0.941,
        r[["ratio", "C1_auc"]].corr(method="spearman").iloc[0, 1], tol=0.01)
    chk("ratio sweep: T2 FAILS - conditional AUC does not fall", True,
        bool(r[["ratio", "C1_auc"]].corr(method="spearman").iloc[0, 1] >= 0),
        tol=0)
    chk("ratio sweep: conditional never behind the marginal variant", True,
        bool((r.C1_auc >= r.A1_auc - 1e-9).all()), tol=0)
    chk("ratio sweep: no strict AUC sign change (no crossover)", 0,
        int((np.sign((r.A1_auc - r.C1_auc).values)[:-1]
             * np.sign((r.A1_auc - r.C1_auc).values)[1:] < 0).sum()), tol=0)
    chk("ratio sweep: A1 sensitivity at ratio 1", 0.45, r.A1_sens.iloc[0],
        tol=0.01)
    chk("ratio sweep: C1 sensitivity at ratio 1", 0.90, r.C1_sens.iloc[0],
        tol=0.01)

# --------------------------------------------------------- generator audit
d = read("ExpOutput/generator_audit/r_scan.csv")
if d is None:
    SKIP.append("generator_audit r_scan.csv missing")
else:
    chk("audit: locked fraction of U(3.7, 3.9)", 0.192, d.locked.mean(),
        tol=0.002)
    chk("audit: P(locked source) at n_src=3", 0.47,
        1 - (1 - d.locked.mean()) ** 3, tol=0.01)
    chk("audit: P(locked source) at n_src=2", 0.35,
        1 - (1 - d.locked.mean()) ** 2, tol=0.01)

d = read("ExpOutput/generator_audit/locked_runs.csv")
if d is None:
    SKIP.append("generator_audit locked_runs.csv missing")
else:
    for name, n_locked, n_dead in [("sink_bar 3/6/6", 26, 8),
                                   ("reopen 2/11/2", 24, 4)]:
        e = d[d.experiment == name]
        chk(f"audit: {name} runs with a locked source", n_locked,
            int((e.n_locked > 0).sum()), tol=0)
        chk(f"audit: {name} dead runs", n_dead,
            int((e.source < 0.002).sum()), tol=0)
        chk(f"audit: {name} every dead run has a locked source", True,
            bool(((e.source < 0.002) <= (e.n_locked > 0)).all()), tol=0)

d = read("ExpOutput/generator_audit/gate_by_parent.csv")
if d is None:
    SKIP.append("generator_audit gate_by_parent.csv missing")
else:
    g = d.groupby("parent_locked").dR2.median()
    chk("audit: gate dR2, chaotic parent", 0.0211, g.loc[False], tol=5e-4)
    chk("audit: gate dR2, locked parent", 0.0005, g.loc[True], tol=5e-4)
    chk("audit: locked parents fail the L50 gate (+0.0136)", True,
        bool(g.loc[True] < 0.0136 < g.loc[False]), tol=0)

# ------------------------------------------------------- clean generator
d = read("ExpOutput/clean_generator/summary.csv")
if d is None:
    SKIP.append("clean_generator summary.csv missing")
else:
    r = d.set_index(d.columns[0])
    chk("clean generator: ghost clean at both shapes", True,
        bool((r.ghost_clear <= 0.05).all()), tol=0)
    chk("clean generator: no locked source survives rejection", True,
        bool((r.src_max_ac < 0.90).all()), tol=0)
    chk("clean generator: 3/6/6 bar", 0.00876, r.loc["3/6/6", "bar"], tol=5e-5)
    chk("clean generator: 3/6/6 sensitivity", 0.867, r.loc["3/6/6", "sens"],
        tol=0.01)
    chk("clean generator: 3/6/6 AUC", 0.889, r.loc["3/6/6", "auc"], tol=0.01)
    chk("clean generator: 2/11/2 bar", 0.01360, r.loc["2/11/2", "bar"],
        tol=5e-5)
    chk("clean generator: 2/11/2 sensitivity", 1.000, r.loc["2/11/2", "sens"],
        tol=0.01)
    chk("clean generator: 2/11/2 AUC", 0.977, r.loc["2/11/2", "auc"], tol=0.01)
    chk("clean generator: Q1 clears the 0.80 bar the dirty run failed", True,
        bool(r.loc["3/6/6", "sens"] >= 0.80), tol=0)
    chk("clean generator: dead runs within the declared limit of 3", True,
        bool((r.dead <= 3).all()), tol=0)

d = read("ExpOutput/clean_generator/runs.csv")
c = read("ExpOutput/clean_generator/channels.csv")
if d is None or c is None:
    SKIP.append("clean_generator runs.csv / channels.csv missing")
else:
    dead = d[d.source < 0.002]
    chk("clean generator: dead runs total", 2, len(dead), tol=0)
    chk("clean generator: every dead run has an orphan source", True,
        bool((dead.n_orphan > 0).all()), tol=0)
    chk("clean generator: no dead run has a locked source", True,
        bool((dead.src_max_ac < 0.90).all()), tol=0)
    chk("clean generator: orphan runs at 3/6/6", 22,
        int((d[d["shape"] == "3/6/6"].n_orphan > 0).sum()), tol=0)
    chk("clean generator: orphan runs at 2/11/2", 1,
        int((d[d["shape"] == "2/11/2"].n_orphan > 0).sum()), tol=0)
    # post-hoc, orphan-free: both shapes detect in every test run
    for shape, k in [("3/6/6", 20), ("2/11/2", 29)]:
        a = d[(d["shape"] == shape) & (d.n_orphan == 0)]
        bar = float(np.quantile(a[a.set == "cal"].sink, 0.95))
        te = a[a.set == "test"]
        chk(f"clean generator: orphan-free sensitivity at {shape}", k,
            int((te.source > bar).sum()), tol=0)
        chk(f"clean generator: orphan-free test runs at {shape}", k, len(te),
            tol=0)
    src = c[(c.role == "source") & (c.n_sinks > 0)]
    g = src.groupby("n_sinks").outflow.median()
    chk("clean generator: outflow rises with sinks driven, 1 to 8", True,
        bool(all(g.loc[i] <= g.loc[i + 1] + 1e-4 for i in range(1, 8))), tol=0)
    chk("clean generator: outflow median at 1 sink", 0.00813, g.loc[1],
        tol=5e-5)
    chk("clean generator: outflow median at 8 sinks", 0.03182, g.loc[8],
        tol=5e-5)
    orph = c[(c.role == "source") & (c.n_sinks == 0)]
    chk("clean generator: orphan sources score ~zero", 0.0,
        float(orph.outflow.median()), tol=0.002)

# ------------------------------------- premises measured, not assumed
# Section sec:practice:premises and the ghost-calibration table in
# sec:theory:finite. Added 2026-09-06 with the audit repairs.
p = Path("ExpOutput/per_channel_null/cells.csv")
if p.exists():
    d = pd.read_csv(p)
    inc = d[d.arm == "GLOBAL-MAX"]
    chk("ghost calibration: nominal level at K=30", 0.032, 1 / 31, tol=5e-4)
    for nz, want in [(0.05, 0.128), (0.30, 0.244)]:
        chk(f"ghost calibration: measured source FP at noise {nz}", want,
            float(inc[inc.noise == nz].source_fp.mean()), tol=5e-4)
    chk("ghost calibration: ghost diagnostic clean in those cells", True,
        bool((inc[inc.noise > 0].ghost_med <= 0.005).all()), tol=0)
else:
    SKIP.append("per-channel null cells.csv absent")

p = Path("ExpOutput/resnet_shortcut/cells.csv")
if p.exists():
    d = pd.read_csv(p)
    # V=30 throughout that run; 6 paired cells = 2 noise levels x 3 seeds
    piv = d.pivot_table(index=["noise", "seed"], columns="arm", values="auroc")
    for label, a, b, want in [
            ("learned self-baseline for the fixed cubic",
             "RES-SC-INCL", "RIDGE-INCL", 0.003),
            ("target withheld from the code",
             "RES-SC-EXCL", "RES-SC-INCL", 0.008)]:
        chk(f"premise repair: {label}, change in ranking accuracy", want,
            float((piv[a] - piv[b]).mean()), tol=5e-4)
        chk(f"premise repair: {label}, cells won", 2,
            int((piv[a] > piv[b]).sum()), tol=0)
        chk(f"premise repair: {label}, cells total", 6, len(piv), tol=0)
    ch = pd.read_csv("ExpOutput/resnet_shortcut/channels.csv")
    z0 = ch[ch.noise == 0.0]
    chk("premise repair: learned trunk beats the cubic on 99% of channels",
        0.99, float((z0.r2_trunk_own > z0.r2_ridge_own).mean()), tol=0.005)
else:
    SKIP.append("resnet shortcut cells.csv absent")

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
