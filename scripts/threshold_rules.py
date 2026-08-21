"""Compare ghost-panel decision rules on identical scans.

Pre-registration: paper/threshold_rule_protocol.md, committed before this was
written. Five rules are evaluated against the SAME per-channel excess values
and the SAME ghost panels saved by boundary_map.py, so no rule benefits from
a different scan or a different seed.

The choice is a multiplicity decision: the panel maximum is a max-statistic
family-wise control, a quantile is a per-channel control. Neither is
arbitrary; the question is whether the conservatism of the maximum buys
anything given that measured precision is already at ceiling.

    python scripts/threshold_rules.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("ExpOutput/boundary_map")
CENTRE = dict(n=4000, V=60, coupling=0.20, redundancy=0)

RULES = {
    "MAX": lambda g: g.max(),
    "Q99": lambda g: np.percentile(g, 99),
    "Q95": lambda g: np.percentile(g, 95),
    "Q90": lambda g: np.percentile(g, 90),
    "MEAN3SD": lambda g: g.mean() + 3 * g.std(),
}
PAT = re.compile(r"raw_n(\d+)_V(\d+)_c([\d.]+)_r(\d+)_s(\d+)\.npz")


def main() -> int:
    files = sorted(OUT.glob("raw_*.npz"))
    if not files:
        print("no raw cells found; run boundary_map.py first")
        return 1
    rows = []
    for f in files:
        m = PAT.match(f.name)
        if not m:
            continue
        n, V, c, r, s = (int(m[1]), int(m[2]), float(m[3]), int(m[4]),
                         int(m[5]))
        z = np.load(f)
        ex, gh = z["excess"], z["ghosts"]
        dr, sr = z["is_driven"], z["is_source"]
        for name, fn in RULES.items():
            thr = max(0.0, float(fn(gh)))
            fl = ex > thr
            tp = int((fl & dr).sum())
            rows.append({
                "rule": name, "n": n, "V": V, "coupling": c,
                "redundancy": r, "seed": s,
                "precision": tp / max(int(fl.sum()), 1),
                "recall": tp / max(int(dr.sum()), 1),
                "source_fp": float((fl & sr).sum() / max(int(sr.sum()), 1)),
                "threshold": thr,
            })
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "threshold_rules.csv", index=False)
    print(f"cells: {d.groupby('rule').size().iloc[0]}   rules: {len(RULES)}\n")

    print("T1/T2  ACROSS THE WHOLE GRID (median)")
    print(f"  {'rule':10s}{'precision':>11}{'recall':>9}{'source FP':>11}")
    for name in RULES:
        g = d[d.rule == name]
        print(f"  {name:10s}{g.precision.median():>11.3f}"
              f"{g.recall.median():>9.3f}{g.source_fp.max():>11.3f}")

    print("\nT3  DEFECT CLAIM: recall variability across n and seed")
    print("     (cells sharing V, coupling, redundancy; sd of recall)")
    print(f"  {'rule':10s}{'sd(recall)':>12}{'min':>8}{'max':>8}")
    for name in RULES:
        g = d[(d.rule == name) & (d.V == CENTRE["V"])
              & (d.coupling == CENTRE["coupling"])
              & (d.redundancy == CENTRE["redundancy"])]
        print(f"  {name:10s}{g.recall.std():>12.3f}"
              f"{g.recall.min():>8.2f}{g.recall.max():>8.2f}")

    print("\nT4  SOURCE FALSE POSITIVES (max over all cells)")
    for name in RULES:
        mx = d[d.rule == name].source_fp.max()
        print(f"  {name:10s}{mx:>8.3f}   "
              f"{'PASS' if mx == 0 else 'DISQUALIFIED'}")

    print("\nDECISION (declared: precision >= 0.95 AND zero source FP AND\n"
          "          lower recall sd than MAX; most conservative qualifier)")
    base = d[(d.rule == "MAX") & (d.V == CENTRE["V"])
             & (d.coupling == CENTRE["coupling"])
             & (d.redundancy == CENTRE["redundancy"])].recall.std()
    qualifying = []
    for name in RULES:
        if name == "MAX":
            continue
        g = d[d.rule == name]
        gc = d[(d.rule == name) & (d.V == CENTRE["V"])
               & (d.coupling == CENTRE["coupling"])
               & (d.redundancy == CENTRE["redundancy"])]
        ok = (g.precision.median() >= 0.95 and g.source_fp.max() == 0
              and gc.recall.std() < base)
        print(f"  {name:10s} precision {g.precision.median():.3f}  "
              f"srcFP {g.source_fp.max():.3f}  sd {gc.recall.std():.3f} "
              f"vs MAX {base:.3f}   {'QUALIFIES' if ok else 'no'}")
        if ok:
            qualifying.append(name)
    order = ["Q99", "MEAN3SD", "Q95", "Q90"]
    winner = next((r for r in order if r in qualifying), None)
    print(f"\n  -> {'ADOPT ' + winner if winner else 'NO RULE QUALIFIES; MAX stands'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
