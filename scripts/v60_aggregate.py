"""Aggregate the three per-seed V=60 runs and apply the declared rule.

Pre-registration: paper/ccm_pcmci_v60_protocol.md. Each seed ran in its own
process for memory isolation, writing results_s{seed}.csv; this collects
them, adds MACE's AUROC re-derived from the published boundary-map cells,
and applies the verdict rule fixed before any score was seen.

    python scripts/v60_aggregate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

OUT = Path("ExpOutput/ccm_pcmci_v60")
N, V, COUPLING, REDUNDANCY = 4000, 60, 0.20, 0
SEEDS = (0, 1, 2)


def auc(pos, neg):
    """Identical to error_metrics.auc; see scripts/test_auc_identical.py."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def main() -> int:
    parts = []
    for s in SEEDS:
        f = OUT / f"results_s{s}.csv"
        if not f.exists():
            print(f"missing {f}")
            return 1
        parts.append(pd.read_csv(f))

        m = np.load(f"ExpOutput/boundary_map/raw_n{N}_V{V}_c{COUPLING}"
                    f"_r{REDUNDANCY}_s{s}.npz")
        parts.append(pd.DataFrame([{
            "seed": s, "method": "MACE",
            "membership_auroc": auc(m["excess"][m["is_driven"]],
                                    m["excess"][m["is_source"]]),
            "true_edge_auroc": np.nan, "minutes": 0.0}]))

    d = pd.concat(parts, ignore_index=True)
    d.to_csv(OUT / "results.csv", index=False)

    piv = d.pivot(index="seed", columns="method", values="membership_auroc")
    print("MEMBERSHIP AUROC per seed")
    print("   " + piv.round(3).to_string().replace("\n", "\n   "))

    g = d.groupby("method").membership_auroc.agg(["median", "min", "max"])
    print("\nover three seeds")
    print("   " + g.round(3).to_string().replace("\n", "\n   "))

    e = d.groupby("method").true_edge_auroc.median()
    print("\nTRUE-EDGE AUROC (median)")
    for k in ("CCM", "PCMCI"):
        print(f"   {k:6s} {e.loc[k]:.3f}  "
              f"{'functions' if e.loc[k] >= 0.7 else 'FAILS to function'}")

    t = d.groupby("method").minutes.sum()
    print(f"\ncompute: CCM {t.loc['CCM']:.0f} min, "
          f"PCMCI {t.loc['PCMCI']:.0f} min, MACE 0 (reused published cells)")

    mace, ccm_m = g.loc["MACE", "median"], g.loc["CCM", "median"]
    both_ceiling = mace > 0.99 and ccm_m > 0.99
    h3_ok = all(e.loc[k] >= 0.7 for k in ("CCM", "PCMCI"))

    print("\nVERDICT (rule fixed before running)")
    if both_ceiling or not h3_ok:
        print("   -> NOT INFORMATIVE: "
              + ("both at ceiling" if both_ceiling else "a baseline failed H3"))
    elif mace > ccm_m:
        print(f"   -> H1 HOLDS. MACE {mace:.3f} > CCM {ccm_m:.3f} at the "
              f"width where MACE's own\n      recall has collapsed to "
              f"0.14-0.18. The V=30 tie was a ceiling effect.")
    else:
        print(f"   -> BASELINE AHEAD. CCM {ccm_m:.3f} >= MACE {mace:.3f}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
