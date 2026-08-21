"""Cover figure: the redundancy-vs-equivalence calibration curve with CIs.

Single-series line with a 95% gene-bootstrap band, base-rate reference, and
the sparse top bins pooled into one honest point: 0 of 13 pairs, drawn with a
rule-of-three upper whisker rather than the bootstrap's overconfident [0,0].

    python scripts/fig_depmap_calibration.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("ExpOutput/depmap_calibration")

# palette: reference instance, light mode
BLUE = "#2a78d6"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
SURF = "#fcfcfb"

curve = pd.read_csv(OUT / "curve_prox_all.csv")
ci = pd.read_csv(OUT / "bootstrap_ci.csv")
m = curve.merge(ci, on="r_lo")

# main span: bins with real mass, below the pooled tail
main = m[(m.r_lo < 0.80) & (m.pairs > 0)].copy()
x = (main.r_lo + 0.025).to_numpy()
y = main.p_equiv.to_numpy() * 100
lo = main["p_lo2.5"].to_numpy() * 100
hi = main["p_hi97.5"].to_numpy() * 100

# pooled tail: r2 >= 0.80
tail = m[m.r_lo >= 0.80]
n_tail = int(tail.pairs.sum())
tail_x, tail_y = 0.90, 0.0
tail_upper = 3.0 / n_tail * 100          # rule of three

base_rate = float((curve.pairs * curve.p_equiv).sum() / curve.pairs.sum()) * 100

# Pooled ceiling CI over the r2 0.60-0.70 bins, computed here from the
# bootstrap replicates rather than hardcoded: the previous label was a
# literal string and survived a correction to the underlying number.
# Float-safe bin mask - the 0.60 edge is stored as 0.6000000000000001.
_bp = pd.read_csv(OUT / "bootstrap_prox.csv")
_sel = _bp[(_bp.r_lo >= 0.60 - 1e-9) & (_bp.r_lo < 0.70 - 1e-9)]
_per = _sel.groupby("boot").apply(
    lambda g: (g.pairs * g.p_equiv).sum() / g.pairs.sum(), include_groups=False)
CEIL, CLO, CHI = (float(_per.median()), float(_per.quantile(0.025)),
                  float(_per.quantile(0.975)))

fig, ax = plt.subplots(figsize=(12.0, 6.44), dpi=200)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)

ax.fill_between(x, lo, hi, color=BLUE, alpha=0.14, linewidth=0)
ax.plot(x, y, color=BLUE, linewidth=2.2, zorder=3)
ax.plot(x, y, "o", color=BLUE, markersize=6, markeredgecolor=SURF,
        markeredgewidth=1.4, zorder=4)

# pooled sparse tail
ax.errorbar([tail_x], [tail_y], yerr=[[0], [tail_upper]], fmt="o",
            color=BLUE, markersize=7, markerfacecolor=SURF,
            markeredgewidth=2.0, elinewidth=1.6, capsize=5, zorder=4)
ax.annotate(f"0 of {n_tail} pairs at the most\nextreme redundancy\n"
            f"(rule-of-three bound ≤ {tail_upper:.0f}%)",
            xy=(tail_x, tail_upper), xytext=(0.66, 26.5),
            fontsize=10.5, color=INK2, ha="left",
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8,
                            shrinkA=2, shrinkB=3))

# base rate reference
ax.axhline(base_rate, color=INK2, linewidth=1.2, linestyle=(0, (5, 4)),
           alpha=0.75)
ax.annotate(f"base rate {base_rate:.2f}%", xy=(0.015, base_rate),
            xytext=(0.015, 2.1), fontsize=10.5, color=INK2)

# ceiling annotation
pk = np.argmax(y)
ax.annotate(f"ceiling ≈ {CEIL*100:.0f}%\n[{CLO*100:.0f}, {CHI*100:.0f}] 95% CI",
            xy=(x[pk], y[pk]), xytext=(0.36, 20.5), fontsize=11, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8,
                            shrinkA=2, shrinkB=4))

# small-n counts beside the sparse-bin points themselves
for rl, n, dy in [(0.70, 79, 1.6), (0.75, 20, 1.6)]:
    yy = float(main.loc[np.isclose(main.r_lo, rl), "p_equiv"].iloc[0]) * 100
    ax.annotate(f"n={n}", xy=(rl + 0.025, yy + dy), fontsize=9, color=INK2,
                ha="center")

ax.set_xlim(0, 1.0)
ax.set_ylim(0, 33)
ax.set_xlabel("expression redundancy between the two genes  (r², binned)",
              fontsize=12, color=INK)
ax.set_ylabel("% of pairs with the same knockout effect", fontsize=12,
              color=INK)
ax.set_title("Genes that look interchangeable usually aren’t",
             fontsize=17, color=INK, loc="left", y=1.10, fontweight="bold")
ax.text(0, 1.045, "P(same knockout phenotype | expression redundancy) — "
        "43.9M gene pairs, 1,103 cancer cell lines (DepMap 24Q4), "
        "95% gene-bootstrap band",
        transform=ax.transAxes, fontsize=10.5, color=INK2)

ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(INK2)
    ax.spines[s].set_linewidth(0.8)
ax.tick_params(colors=INK2, labelsize=10)

fig.tight_layout(pad=1.6)
out = OUT / "fig_calibration_cover.png"
fig.savefig(out, facecolor=SURF, bbox_inches="tight")
print(f"wrote {out}")
