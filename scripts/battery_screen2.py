"""Split-sample battery re-screen: grid from physics, verdict on held-out cells.

Pre-registration: paper/dataset_fitness_protocol.md, split-sample addendum,
committed before this ran. The first battery screen was DISQUALIFIED at its
declared 60 s grid, with lag_info ascending toward the truncation edge; any
regrid is therefore post-hoc motivated, and this design contains that: the
new base grid is fixed from thermal physics (not from the observed numbers),
four named DISCOVERY cells may be examined freely, and the verdict is the
median over the 24 HELD-OUT cells alone. This is the final screen of this
dataset; no third grid.

    python scripts/battery_screen2.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import battery_screen as B  # noqa: E402
from dataset_fitness import lag_info  # noqa: E402
from chamber_screen import reference  # noqa: E402

OUT = Path("ExpOutput/battery_screen2")
DT = 300.0            # 5 min: declared from the 18650 thermal time constant
MIN_N = 2000          # Rule 98: segments ~2,400 points at this grid -> m=1 only
DISCOVERY = {"RW25", "RW3", "RW13", "RW9"}   # one per stratum, named in advance


def cell_stat(f):
    B.DT = DT                          # base grid for segments()
    T, X = B.load_cell(f)
    segs = B.segments(T, X)
    cv, cg = [], []
    for s in segs:
        if len(s) < MIN_N or s[:, 0].std() < 1e-9:
            continue
        cv.append(lag_info(s, 1))
        cg.append(lag_info(s, 1, ghost=True))
    if not cv:
        return None
    return {"cell": Path(f).stem,
            "family": Path(f).parts[-4].replace("_DataSet_2Post", ""),
            "n_segs": len(cv), "lag_info": float(np.median(cv)),
            "ghost": float(np.median(cg))}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    L30, L50 = reference()
    print(f"REFERENCE recomputed: L30 = {L30:+.4f}   L50 = {L50:+.4f}")
    print(f"base grid DT = {DT:.0f}s (declared from physics)   m = 1 only "
          f"(Rule 98 arithmetic)\n")

    rows = []
    for f in sorted(glob.glob(B.DATA)):
        r = cell_stat(f)
        if r is None:
            print(f"  {Path(f).stem}: no valid segment")
            continue
        r["set"] = "discovery" if r["cell"] in DISCOVERY else "heldout"
        rows.append(r)
        print(f"  {r['set']:9s} {r['family'][:40]:42s}{r['cell']:6s} "
              f"segs {r['n_segs']:>2}  lag_info {r['lag_info']:+.4f}  "
              f"ghost {r['ghost']:+.4f}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "cells.csv", index=False)

    disc = d[d.set == "discovery"]
    held = d[d.set == "heldout"]
    print(f"\nDISCOVERY ({len(disc)} cells, informational only): "
          f"median {disc.lag_info.median():+.4f}")
    print(f"HELD-OUT  ({len(held)} cells, the verdict): "
          f"median {held.lag_info.median():+.4f}  "
          f"[{held.lag_info.min():+.4f}, {held.lag_info.max():+.4f}]  "
          f"ghost {held.ghost.median():+.4f}")
    print("  per family (held-out):")
    for fam, g in held.groupby("family"):
        print(f"    {fam[:42]:44s} {g.lag_info.median():+.4f}  "
              f"clear L50: {(g.lag_info >= L50).sum()}/{len(g)}")

    v = float(held.lag_info.median())
    print("\nVERDICT (held-out median against unchanged L50; rule fixed "
          "before running)")
    if v >= L50:
        print(f"   -> QUALIFIES: {v:+.4f} >= {L50:+.4f}. Next: the "
              "pre-registered replication test,\n      marginal statistic "
              "primary, on the held-out cells.")
    else:
        print(f"   -> DISQUALIFIED: {v:+.4f} < {L50:+.4f}. Final for this "
              "dataset; no third grid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
