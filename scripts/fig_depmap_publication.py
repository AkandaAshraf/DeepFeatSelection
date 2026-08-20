"""Publication-format version of the calibration figure for PLOS.

PLOS requires TIFF (LZW) or EPS, 300-600 dpi, width 789-2250 px and height
at most 2625 px. The screen version is 12 in wide at 200 dpi, which is 3600
px at 300 dpi and too wide, so this rebuilds at 7.5 in with type sized for
print rather than rescaling the raster.

    python scripts/fig_depmap_publication.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("ExpOutput/depmap_calibration")
BLUE, INK, INK2, GRID, SURF = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1", "#ffffff"
DPI = 300

curve = pd.read_csv(OUT / "curve_prox_all.csv")
ci = pd.read_csv(OUT / "bootstrap_ci.csv")
m = curve.merge(ci, on="r_lo")

main = m[(m.r_lo < 0.80) & (m.pairs > 0)].copy()
x = (main.r_lo + 0.025).to_numpy()
y = main.p_equiv.to_numpy() * 100
lo = main["p_lo2.5"].to_numpy() * 100
hi = main["p_hi97.5"].to_numpy() * 100

tail = m[m.r_lo >= 0.80]
n_tail = int(tail.pairs.sum())
tail_upper = 3.0 / n_tail * 100
base_rate = float((curve.pairs * curve.p_equiv).sum() / curve.pairs.sum()) * 100

_bp = pd.read_csv(OUT / "bootstrap_prox.csv")
_sel = _bp[(_bp.r_lo >= 0.60 - 1e-9) & (_bp.r_lo < 0.70 - 1e-9)]
_per = _sel.groupby("boot").apply(
    lambda g: (g.pairs * g.p_equiv).sum() / g.pairs.sum(), include_groups=False)
CEIL, CLO, CHI = (float(_per.median()), float(_per.quantile(0.025)),
                  float(_per.quantile(0.975)))

fig, ax = plt.subplots(figsize=(7.5, 4.3), dpi=DPI)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)

ax.fill_between(x, lo, hi, color=BLUE, alpha=0.15, linewidth=0)
ax.plot(x, y, color=BLUE, linewidth=1.6, zorder=3)
ax.plot(x, y, "o", color=BLUE, markersize=4.0, markeredgecolor=SURF,
        markeredgewidth=0.9, zorder=4)

ax.errorbar([0.90], [0.0], yerr=[[0], [tail_upper]], fmt="o", color=BLUE,
            markersize=4.5, markerfacecolor=SURF, markeredgewidth=1.3,
            elinewidth=1.2, capsize=3.5, zorder=4)
ax.annotate(f"0 of {n_tail} pairs at\nextreme redundancy\n"
            f"(rule of three, ≤{tail_upper:.0f}%)",
            xy=(0.90, tail_upper), xytext=(0.635, 25.0),
            fontsize=7.5, color=INK2, ha="left",
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6,
                            shrinkA=2, shrinkB=3))

ax.axhline(base_rate, color=INK2, linewidth=0.9, linestyle=(0, (5, 4)),
           alpha=0.8)
ax.annotate(f"base rate {base_rate:.2f}%", xy=(0.012, base_rate),
            xytext=(0.012, 1.9), fontsize=7.5, color=INK2)

pk = int(np.argmax(y))
ax.annotate(f"ceiling {CEIL*100:.0f}%\n[{CLO*100:.0f}, {CHI*100:.0f}] 95% CI",
            xy=(x[pk], y[pk]), xytext=(0.335, 21.0), fontsize=8, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6,
                            shrinkA=2, shrinkB=4))

for rl in (0.70, 0.75):
    row = main.loc[np.isclose(main.r_lo, rl)]
    yy = float(row.p_equiv.iloc[0]) * 100
    ax.annotate(f"n={int(row.pairs.iloc[0])}", xy=(rl + 0.025, yy + 1.4),
                fontsize=6.5, color=INK2, ha="center")

ax.set_xlim(0, 1.0)
ax.set_ylim(0, 32)
ax.set_xlabel("expression redundancy between the two genes  ($r^2$, binned)",
              fontsize=9, color=INK)
ax.set_ylabel("% of pairs interventionally equivalent", fontsize=9, color=INK)
ax.grid(axis="y", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(INK2)
    ax.spines[s].set_linewidth(0.6)
ax.tick_params(colors=INK2, labelsize=8)

fig.tight_layout(pad=0.6)
for ext, kw in (("tiff", dict(pil_kwargs={"compression": "tiff_lzw"})),
                ("eps", {}), ("png", {})):
    p = OUT / f"Fig1.{ext}"
    fig.savefig(p, facecolor=SURF, dpi=DPI, **kw)
    px = fig.get_size_inches() * DPI
    print(f"wrote {p}  {int(px[0])}x{int(px[1])} px @ {DPI} dpi  "
          f"{p.stat().st_size/1e6:.2f} MB")

# PLOS gate: width 789-2250 px, height <= 2625 px, size <= 10 MB
w, h = (fig.get_size_inches() * DPI).astype(int)
ok = 789 <= w <= 2250 and h <= 2625
print(f"PLOS dimension check: width {w} (789-2250), height {h} (<=2625) -> "
      f"{'PASS' if ok else 'FAIL'}")
