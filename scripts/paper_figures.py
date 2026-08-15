"""Generate all figures for the excess-over-self paper from primary data.

Every panel is drawn from files in ExpOutput/ (or deterministic
reconstructions of the generating systems), never from numbers typed in by
hand, so the figures cannot drift from the evidence. Output: paper/figs/*.pdf.

Design rules applied throughout (colour-blind-safe by construction):
Okabe-Ito categorical palette in fixed order; sequential magnitude uses a
single-hue ramp; one axis per panel; thin marks; direct labels over legends
where few series; recessive grids.

    python scripts/paper_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

# Okabe-Ito, fixed assignment across the whole paper.
BLUE = "#0072B2"      # members / primary series
ORANGE = "#E69F00"    # non-members / secondary series
GREEN = "#009E73"     # tertiary
VERM = "#D55E00"      # ghost / warnings
GRAY = "#7F7F7F"      # context

FIGS = Path("paper/figs")
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def fig_maturity():
    d = pd.read_csv("ExpOutput/maturity_synth/synthetic_curve.csv")
    m = d.groupby("epochs").mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(m.index, m.hub_gain, "-o", color=BLUE, lw=2, ms=4)
    ax.plot(m.index, m.few_gain, "-s", color=ORANGE, lw=2, ms=4)
    ax.annotate("hub parent (6 children)\nredundant imprint",
                (m.index[-1], m.hub_gain.iloc[-1]), xytext=(-5, 12),
                textcoords="offset points", ha="right", color=BLUE)
    ax.annotate("single-child parents\nunique imprint",
                (m.index[-1], m.few_gain.iloc[-1]), xytext=(-5, -22),
                textcoords="offset points", ha="right", color=ORANGE)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("training epochs")
    ax.set_ylabel("leave-one-out gain")
    ax.set_xscale("log")
    ax.set_xticks(m.index)
    ax.set_xticklabels([str(int(e)) for e in m.index])
    fig.savefig(FIGS / "fig_maturity.pdf")
    plt.close(fig)


def _v1000_classes():
    """Reconstruct membership truth and loner periodicity for system seed 0."""
    from bottleneck_membership import build_system
    x, truth = build_system(1000, 100, 2000, 0.3, 0)
    lone = x[:, 100:]
    uniq = np.array([len(np.unique(np.round(lone[-500:, j], 6)))
                     for j in range(900)])
    return truth, uniq < 50


def fig_typicality(truth, periodic):
    scores = np.load("ExpOutput/ensemble/all_scores.npy").mean(axis=0)
    groups = [
        ("periodic\nautonomous", scores[100:1000][periodic], GRAY),
        ("ghost\n(shifted copy)", np.array([scores[-1]]), VERM),
        ("chaotic\nautonomous", scores[100:1000][~periodic], GRAY),
        ("coupled\nmembers", scores[:100][truth[:100]], BLUE),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for i, (label, vals, c) in enumerate(groups):
        jitter = (np.random.default_rng(0).uniform(-0.16, 0.16, len(vals))
                  if len(vals) > 1 else np.zeros(1))
        ax.plot(i + jitter, vals, "o", ms=2.5, alpha=0.35, color=c,
                markeredgewidth=0)
        ax.plot([i - 0.25, i + 0.25], [vals.mean()] * 2, "-", color=c, lw=2.5)
        ax.annotate(f"{vals.mean():+.2f}", (i, vals.mean()),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", color=c, fontsize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_ylabel("masked-recreation consensus $R^2$")
    fig.savefig(FIGS / "fig_typicality.pdf")
    plt.close(fig)


def fig_lln(truth):
    from sklearn.metrics import average_precision_score
    scores = np.load("ExpOutput/ensemble/all_scores.npy")
    truth_all = np.append(truth, False)
    Ms = range(1, scores.shape[0] + 1)
    ap = [average_precision_score(truth_all, scores[:m].mean(axis=0))
          for m in Ms]
    ghost = [scores[:m].mean(axis=0)[-1] for m in Ms]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.3))
    axes[0].plot(list(Ms), ap, "-o", color=BLUE, lw=2, ms=4)
    axes[0].axhline(truth_all.mean(), color=GRAY, lw=0.8, ls="--")
    axes[0].annotate("baseline", (1, truth_all.mean()), xytext=(0, 4),
                     textcoords="offset points", color=GRAY, fontsize=8)
    axes[0].set_xlabel("ensemble size $M$")
    axes[0].set_ylabel("consensus AP")
    axes[1].plot(list(Ms), ghost, "-o", color=VERM, lw=2, ms=4)
    axes[1].axhline(0, color=GRAY, lw=0.8)
    axes[1].set_xlabel("ensemble size $M$")
    axes[1].set_ylabel("ghost recreation $R^2$")
    axes[1].set_ylim(-0.05, 0.85)
    fig.savefig(FIGS / "fig_lln.pdf")
    plt.close(fig)


def fig_synthetic(truth):
    ex = np.load("ExpOutput/excess_poly/excess_consensus.npy")
    truth_all = np.append(truth, False)
    order = np.argsort(-ex)
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    xs = np.arange(len(ex))
    colors = np.where(truth_all[order], BLUE, ORANGE)
    ghost_rank = int(np.where(order == len(ex) - 1)[0][0])
    ax.vlines(xs[:60], 0, ex[order][:60], colors=colors[:60], lw=1.4)
    ax.plot(xs[60:], ex[order][60:], ".", ms=1.2,
            color=ORANGE, alpha=0.4, markeredgewidth=0)
    for i in range(60):
        if truth_all[order[i]]:
            continue
    ax.plot(ghost_rank, ex[len(ex) - 1], "*", ms=9, color=VERM)
    ax.annotate("ghost", (ghost_rank, ex[len(ex) - 1]), xytext=(6, 8),
                textcoords="offset points", color=VERM)
    ax.annotate("members", (8, ex[order][8]), xytext=(12, 6),
                textcoords="offset points", color=BLUE)
    ax.annotate("autonomous channels pinned at $\\leq 0$",
                (400, 0), xytext=(0, -16), textcoords="offset points",
                color=ORANGE)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("rank by consensus excess (V = 1001)")
    ax.set_ylabel("excess")
    ax.set_xlim(-8, 1010)
    fig.savefig(FIGS / "fig_synthetic.pdf")
    plt.close(fig)


def _worm_profiles():
    def load(paths):
        fs = []
        for w, path in paths:
            d = pd.read_csv(path)
            d["neuron"] = d.neuron.astype(str).str.replace("\x00", "", regex=False)
            d = d[d.neuron != "GHOST"].copy()
            d["identified"] = ~d.neuron.str.isdigit()
            d["pct"] = d.excess.rank(pct=True)
            d["cell"] = d.neuron.str.replace(r"[LR]$", "", regex=True)
            fs.append(d[d.identified][["cell", "pct"]].assign(worm=w))
        return pd.concat(fs)
    wt = load([(w, f"ExpOutput/celegans_excess/worm{w}_excess.csv") for w in (0, 1, 2)]
              + [(w, f"ExpOutput/celegans_excess_heldout/worm{w}_excess.csv") for w in (3, 4)])
    ko = load([(w, f"ExpOutput/celegans_excess_avahiscl/worm{w}_excess.csv") for w in range(5)])
    prof = lambda df: df.groupby("cell").agg(n=("worm", "nunique"), pct=("pct", "mean"))
    w, k = prof(wt), prof(ko)
    j = w.join(k, lsuffix="_wt", rsuffix="_ko", how="inner")
    return j[(j.n_wt >= 3) & (j.n_ko >= 3)]


def fig_worm(j):
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=0.8)
    ax.plot(j.pct_wt, j.pct_ko, "o", ms=5, color=GRAY, alpha=0.55,
            markeredgewidth=0)
    highlight = {"AVA": VERM, "AVE": BLUE, "RIM": GREEN, "AIB": GREEN,
                 "AVB": ORANGE}
    offsets = {"AVA": (6, -10), "AVE": (6, 2), "RIM": (-10, 6),
               "AIB": (6, 4), "AVB": (6, -2)}
    for cell, c in highlight.items():
        if cell in j.index:
            r = j.loc[cell]
            ax.plot(r.pct_wt, r.pct_ko, "o", ms=7, color=c)
            ax.annotate(cell, (r.pct_wt, r.pct_ko),
                        xytext=offsets.get(cell, (6, 2)),
                        textcoords="offset points", color=c, fontsize=9)
    ax.set_xlabel("drivenness percentile, wild type (5 worms)")
    ax.set_ylabel("drivenness percentile, AVA silenced (5 worms)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    fig.savefig(FIGS / "fig_worm.pdf")
    plt.close(fig)


def fig_fish():
    data = np.load("ExpOutput/zapbench_full/excess_sample.npy")
    idx = data[:, 0].astype(int)
    ex = data[:, 1]
    d = json.loads(Path("Data/zapbench/segmentation_dataframe.json").read_text())
    cx = np.array([d["centroid_x"][str(i)] for i in range(71721)])
    cy = np.array([d["centroid_y"][str(i)] for i in range(71721)])
    order = np.argsort(-ex)
    top = idx[order[:1000]]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.plot(cx, cy, ".", ms=0.4, color=GRAY, alpha=0.12, markeredgewidth=0)
    ax.plot(cx[top], cy[top], ".", ms=1.8, color=BLUE, alpha=0.6,
            markeredgewidth=0)
    ax.annotate("top-1000 driven neurons", (cx[top].mean(), cy[top].mean()),
                xytext=(30, 46), textcoords="offset points", color=BLUE)
    ax.annotate("anterior", (cx.min() + 30, cy.min() + 8), color=GRAY,
                fontsize=8)
    ax.annotate("posterior", (cx.max() - 190, cy.min() + 8), color=GRAY,
                fontsize=8)
    ax.set_xlabel("x (rostro-caudal, px)")
    ax.set_ylabel("y (left-right, px)")
    ax.set_aspect("equal")
    ax.grid(False)
    fig.savefig(FIGS / "fig_fish.pdf")
    plt.close(fig)


def fig_eeg():
    w = pd.read_csv("ExpOutput/eeg_excess/windows.csv")
    recs = sorted(set(w[w.kind == "ictal"].record))
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    for rec in recs:
        ict = w[(w.record == rec) & (w.kind == "ictal")].top4_share.mean()
        inter = w[(w.record == rec) & (w.kind == "interictal")].top4_share.mean()
        exception = ict < inter
        c = VERM if exception else BLUE
        ax.plot([0, 1], [inter, ict], "-o", color=c, lw=1.6, ms=4,
                alpha=0.85)
        if exception:
            ax.annotate(rec.replace(".edf", ""), (1, ict), xytext=(6, 0),
                        textcoords="offset points", color=VERM, fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["interictal", "ictal"])
    ax.set_ylabel("top-4 concentration of excess")
    ax.set_xlim(-0.25, 1.45)
    fig.savefig(FIGS / "fig_eeg.pdf")
    plt.close(fig)


def fig_climate():
    ex = np.load("ExpOutput/climate_excess/excess.npy")[:10512]
    latlon = np.load("ExpOutput/climate_excess/latlon.npy")
    la = latlon[0].reshape(73, 144)
    lo = latlon[1].reshape(73, 144)
    grid = ex.reshape(73, 144)
    lo_c = np.where(lo > 180, lo - 360, lo)
    order = np.argsort(lo_c[0])
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4),
                             gridspec_kw={"width_ratios": [3, 1]})
    pm = axes[0].pcolormesh(lo_c[:, order], la[:, order], grid[:, order],
                            cmap="Blues", shading="auto")
    fig.colorbar(pm, ax=axes[0], label="excess", shrink=0.85)
    axes[0].set_xlabel("longitude")
    axes[0].set_ylabel("latitude")
    axes[0].grid(False)
    zonal = grid.mean(axis=1)
    axes[1].plot(zonal, la[:, 0], "-", color=BLUE, lw=2)
    axes[1].axhspan(-30, 30, color=BLUE, alpha=0.08)
    axes[1].set_xlabel("zonal mean excess")
    axes[1].set_ylabel("latitude")
    axes[1].set_ylim(-90, 90)
    fig.savefig(FIGS / "fig_climate.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_maturity(); print("maturity")
    truth, periodic = _v1000_classes()
    fig_typicality(truth, periodic); print("typicality")
    fig_lln(truth); print("lln")
    fig_synthetic(truth); print("synthetic")
    j = _worm_profiles()
    fig_worm(j); print("worm")
    fig_fish(); print("fish")
    fig_eeg(); print("eeg")
    fig_climate(); print("climate")
    print("all figures written to", FIGS)
