"""Figure for the recall section: the operating curve and the ghost null.

Panel (a): precision and recall against k, so the price of going deeper into
the ranking is visible rather than implied by a single top-10 number. Both
series are fractions on the same 0-1 axis, so one axis serves both.

Panel (b): the ghost null distribution against the flagged members, on a
symmetric log scale because the null spans four decades below zero and the
detections sit two decades above it. This is the panel that shows why the
statistic is a top-k detector: the separation at the top is enormous and the
bulk of members is buried in the null.

    python scripts/fig_recall.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from bottleneck_membership import build_system  # noqa: E402

BLUE, ORANGE, VERM, GRAY = "#0072B2", "#E69F00", "#D55E00", "#7F7F7F"
FIGS = Path("paper/figs")
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 2.9))

# ---- (a) operating curve -------------------------------------------------
cur = pd.read_csv("ExpOutput/recall/operating_curve.csv")
m = cur.groupby("k")[["precision", "recall"]].mean()
for sysname, sub in cur.groupby("system"):
    s = sub.set_index("k")
    ax1.plot(s.index, s.precision, "-", color=BLUE, lw=0.7, alpha=0.3)
    ax1.plot(s.index, s.recall, "-", color=ORANGE, lw=0.7, alpha=0.3)
ax1.plot(m.index, m.precision, "-o", color=BLUE, lw=2, ms=4, zorder=3)
ax1.plot(m.index, m.recall, "-s", color=ORANGE, lw=2, ms=4, zorder=3)
ax1.annotate("precision", (m.index[1], m.precision.iloc[1]),
             xytext=(6, 8), textcoords="offset points", color=BLUE, fontsize=8.5)
ax1.annotate("recall", (m.index[-2], m.recall.iloc[-2]),
             xytext=(-4, 9), textcoords="offset points", color=ORANGE,
             fontsize=8.5)
ax1.set_xscale("log")
ax1.set_xticks(list(m.index))
ax1.set_xticklabels([str(k) for k in m.index])
ax1.set_xlabel("$k$ (variables reported)")
ax1.set_ylabel("fraction")
ax1.set_ylim(-0.03, 1.05)
ax1.set_title("(a) no depth gives a complete inventory", fontsize=9, loc="left")

# ---- (b) null versus detections ------------------------------------------
ex = np.load("ExpOutput/excess_poly/excess_consensus.npy")
_, truth = build_system(1000, 100, 2000, 0.3, 0)
ghosts = np.load("ExpOutput/recall/ghosts_seed0.npy")
tail = pd.read_csv("ExpOutput/recall/ghost_tail.csv")
ok = tail[(tail.system == "seed0") & (tail.self_predictability > 0.99)]
thr = max(0.0, float(ok.excess.max()))

mem, non = ex[:100][truth[:100]], ex[100:1000]
groups = [("ghost panel\n(qualifying donors)", ok.excess.values, VERM),
          ("autonomous\nchannels", non, GRAY),
          ("coupled\nmembers", mem, BLUE)]
for i, (lab, vals, col) in enumerate(groups):
    jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.34
    ax2.scatter(vals, np.full(len(vals), i) + jitter, s=7, color=col,
                alpha=0.55, linewidths=0)
ax2.axvline(thr, color="black", lw=1.1, ls="--", zorder=4)
ax2.annotate(f"panel threshold\n{thr:.1e}", (thr, 2.42),
             xytext=(9, 0), textcoords="offset points", fontsize=7.6,
             va="center")
ax2.set_xscale("symlog", linthresh=1e-6)
# Explicit decade ticks: the automatic symlog locator crowds the linear
# region and renders the labels unreadable.
ax2.set_xticks([-1e-1, -1e-3, -1e-5, 0, 1e-5, 1e-3])
ax2.set_xticklabels(["$-10^{-1}$", "$-10^{-3}$", "$-10^{-5}$", "0",
                     "$10^{-5}$", "$10^{-3}$"], fontsize=8)
ax2.set_yticks(range(len(groups)))
ax2.set_yticklabels([g[0] for g in groups], fontsize=8)
ax2.set_xlabel("consensus excess (symmetric log)")
ax2.set_ylim(-0.6, len(groups) - 0.25)
ax2.set_title("(b) detections clear the ghost panel", fontsize=9, loc="left")
ax2.grid(axis="y", visible=False)

n_over = int((mem > thr).sum())
print(f"panel threshold {thr:.3e}; members above {n_over}/{len(mem)}; "
      f"autonomous above {int((non > thr).sum())}/{len(non)}; "
      f"qualifying ghosts {len(ok)}")

fig.tight_layout()
out = FIGS / "fig_recall.pdf"
fig.savefig(out)
print(f"wrote {out}")
