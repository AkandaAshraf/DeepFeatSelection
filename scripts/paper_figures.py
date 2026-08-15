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
    xs = range(len(m.index))
    fig, ax = plt.subplots(figsize=(3.8, 2.7))
    ax.plot(xs, m.hub_gain, "-o", color=BLUE, lw=2, ms=4)
    ax.plot(xs, m.few_gain, "-s", color=ORANGE, lw=2, ms=4)
    ax.annotate("hub parent (6 children),\nredundant imprint",
                (2, m.hub_gain.iloc[2]), xytext=(0, -30),
                textcoords="offset points", ha="center", color=BLUE,
                fontsize=8)
    ax.annotate("single-child parents,\nunique imprint",
                (2.6, 0.12), ha="center", color=ORANGE, fontsize=8)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("training epochs")
    ax.set_ylabel("leave-one-out gain")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(int(e)) for e in m.index])
    ax.set_ylim(-0.05, 0.32)
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
    """Horizontal group summary: median, IQR box, and full point spread.

    The earlier vertical strip plot collided its x-tick labels and hid the
    bimodality of each group behind a single mean bar; horizontal layout
    gives the labels room and lets the spread be read directly.
    """
    scores = np.load("ExpOutput/ensemble/all_scores.npy").mean(axis=0)
    groups = [
        ("coupled members", scores[:100][truth[:100]], BLUE),
        ("chaotic autonomous", scores[100:1000][~periodic], GRAY),
        ("ghost (shifted copy)", np.array([scores[-1]]), VERM),
        ("periodic autonomous", scores[100:1000][periodic], GRAY),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    rng = np.random.default_rng(0)
    for i, (label, vals, c) in enumerate(groups):
        if len(vals) > 1:
            jit = rng.uniform(-0.17, 0.17, len(vals))
            ax.plot(vals, i + jit, "o", ms=2.2, alpha=0.28, color=c,
                    markeredgewidth=0, zorder=1)
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            ax.plot([q1, q3], [i, i], "-", color=c, lw=5, alpha=0.55,
                    solid_capstyle="butt", zorder=2)
            ax.plot(med, i, "|", color=c, ms=16, mew=2.2, zorder=3)
            txt = f"median {med:.2f}"
        else:
            ax.plot(vals[0], i, "D", ms=7, color=c, zorder=3)
            txt = f"{vals[0]:.2f}"
        ax.annotate(txt, (1.03, i), xycoords=("axes fraction", "data"),
                    va="center", ha="left", color=c, fontsize=8)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g[0] for g in groups], fontsize=9)
    ax.set_xlabel("masked-reconstruction consensus $R^2$")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.6, len(groups) - 0.4)
    ax.grid(axis="y", alpha=0)
    fig.subplots_adjust(right=0.78)
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
    axes[1].set_ylabel("ghost reconstruction $R^2$")
    axes[1].set_ylim(-0.05, 0.85)
    fig.savefig(FIGS / "fig_lln.pdf")
    plt.close(fig)


def fig_synthetic(truth):
    """Top-25 ranks as stems with markers, so zero-valued entries stay visible.

    A plain bar chart drew the non-members at zero height, making them
    invisible and leaving a legend entry with no corresponding mark. The
    marker at each stem tip fixes that, and it is the substantive point:
    the few non-members that rank inside the top 20 sit at exactly zero,
    so they are ranked high only because everything below them is zero too.
    """
    ex = np.load("ExpOutput/excess_poly/excess_consensus.npy")
    truth_all = np.append(truth, False)
    order = np.argsort(-ex)
    vals, is_mem = ex[order], truth_all[order]
    k = 25
    xs = np.arange(1, k + 1)

    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    ax.axhline(0, color=GRAY, lw=0.9, zorder=1)
    for i in range(k):
        c = BLUE if is_mem[i] else ORANGE
        ax.plot([xs[i], xs[i]], [0, vals[i]], "-", color=c, lw=2.6,
                solid_capstyle="round", zorder=2)
        ax.plot(xs[i], vals[i], "o", color=c, ms=5,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)

    n_mem_top = int(is_mem[:20].sum())
    ax.annotate(f"top 20: {n_mem_top}/20 are true members",
                (0.03, 0.90), xycoords="axes fraction", fontsize=8,
                color=BLUE)
    note = ("non-members here sit at exactly 0;\n"
            "they rank high only because the\n"
            "961 channels below are also 0")
    ax.annotate(note, (0.44, 0.42), xycoords="axes fraction", fontsize=8,
                color=ORANGE)
    handles = [plt.Line2D([], [], color=BLUE, marker="o", lw=2.6, ms=5),
               plt.Line2D([], [], color=ORANGE, marker="o", lw=2.6, ms=5)]
    ax.legend(handles, ["coupled member", "autonomous channel"],
              loc="upper right", frameon=False, fontsize=8)
    ax.set_xlabel("rank by consensus excess ($V = 1001$)")
    ax.set_ylabel("excess")
    ax.set_xlim(0.3, k + 0.7)
    ax.set_xticks([1, 5, 10, 15, 20, 25])
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
