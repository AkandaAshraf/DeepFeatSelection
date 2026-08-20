"""Re-derive every number quoted in the DepMap short paper from its outputs.

The companion paper carries an audit gate for exactly this reason: a number
in prose must be reproducible from a file, not from memory. Run before any
edit to the manuscript is considered final.

    python scripts/depmap_paper_numbers.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("ExpOutput/depmap_calibration")


def f(x, n=4):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


def weighted(curve: pd.DataFrame, lo: float = None, hi: float = None) -> float:
    d = curve
    if lo is not None:
        d = d[d.r_lo >= lo]
    if hi is not None:
        d = d[d.r_lo < hi]
    tot = d.pairs.sum()
    return float((d.pairs * d.p_equiv).sum() / tot) if tot else float("nan")


print("=" * 72)
print("HEADLINE (prox axis, lineage-corrected)")
curve = pd.read_csv(OUT / "curve_prox_all.csv")
ci = pd.read_csv(OUT / "bootstrap_ci.csv")
base = weighted(curve)
ceil_ = weighted(curve, 0.60, 0.70)
top = curve[curve.r_lo >= 0.80]
print(f"  total pairs scored      {curve.pairs.sum():,.0f}")
print(f"  base rate               {base*100:.2f}%")
print(f"  ceiling (r2 0.60-0.70)  {ceil_*100:.1f}%")
print(f"  lift                    {ceil_/base:.1f}x")
print(f"  pairs in top bin r2>0.8 {int(top.pairs.sum())}")
print(f"  equivalent in top bin   {int((top.pairs*top.p_equiv).sum())}")
print(f"  rule-of-three bound     {300/max(top.pairs.sum(),1):.0f}%")

print("\nBOOTSTRAP CIs (B=100, gene-level)")
for name, path in (("prox", "bootstrap_prox.csv"),):
    b = pd.read_csv(OUT / path)
    bb = b.groupby("boot").apply(
        lambda g: pd.Series({
            "base": (g.pairs * g.p_equiv).sum() / g.pairs.sum(),
            "ceil": weighted(g, 0.60, 0.70)}), include_groups=False)
    bb["lift"] = bb.ceil / bb.base
    for k in ("base", "ceil", "lift"):
        q = bb[k].quantile([0.025, 0.5, 0.975])
        scale = 100 if k != "lift" else 1
        unit = "%" if k != "lift" else "x"
        print(f"  {k:5s} {q[0.5]*scale:.2f}{unit} "
              f"[{q[0.025]*scale:.2f}, {q[0.975]*scale:.2f}]")

print("\nLINEAGE INFLATION")
cc = pd.read_csv(OUT / "curve_corrected.csv")
cu = pd.read_csv(OUT / "curve_uncorrected.csv")
hc = cc[cc.r_lo >= 0.60].pairs.sum()
hu = cu[cu.r_lo >= 0.60].pairs.sum()
print(f"  high-redundancy pairs uncorrected {hu:,.0f} -> corrected {hc:,.0f}")
print(f"  removed by lineage correction     {(1-hc/hu)*100:.0f}%")

print("\nARM B  many-to-one")
g = pd.read_csv(OUT / "many_to_one_genes.csv")


def auc(score, label):
    order = np.argsort(score)
    r = np.empty(len(score), float)
    r[order] = np.arange(1, len(score) + 1)
    pos, neg = label.sum(), (~label).sum()
    return float((r[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


lab = g.has_equiv_partner.to_numpy().astype(bool)
print(f"  genes                       {len(g):,}")
print(f"  base P(has equiv partner)   {lab.mean():.3f}")
for ax in ("best_pair_r2", "panel_r2"):
    hi = g[g[ax] >= 0.6]
    print(f"  {ax:13s} AUC {auc(g[ax].to_numpy(), lab):.3f}   "
          f"P(equiv|>=0.6) {hi.has_equiv_partner.mean():.3f} (n={len(hi)})")

print("\nARM C  real gene families vs symbol-root proxy")
for name in ("real_nofamily", "real_familyonly", "proxy_nofamily"):
    p = OUT / f"curve_{name}.csv"
    if p.exists():
        c = pd.read_csv(p)
        print(f"  {name:16s} base {weighted(c)*100:.3f}%  "
              f"ceiling(r2>=0.6) {weighted(c, 0.60)*100:.1f}%  "
              f"pairs>=0.6 {int(c[c.r_lo>=0.60].pairs.sum())}")

print("\nARM A  TNBC")
for name in ("tnbc", "breast_all"):
    p = OUT / f"curve_{name}.csv"
    if p.exists():
        c = pd.read_csv(p)
        hi = c[c.r_lo >= 0.60].pairs.sum()
        print(f"  {name:11s} lines {int(c.n_lines.iloc[0]):3d}  "
              f"base {weighted(c)*100:.2f}%  ceiling {weighted(c,0.60)*100:.1f}%  "
              f"pairs>=0.6 {int(hi):,} ({hi/c.pairs.sum()*100:.1f}% of all)")
print("=" * 72)
