"""Which ghost statistic should the gate watch?

Pre-registration: paper/gate_statistic_protocol.md, committed before this was
written.

G3 currently tests the ghost panel's MEDIAN. The duplicate-channel experiment
found nine scans where up to half the genuine sources were flagged and every
one passed G3, while the panel's maximum rose 30-80 fold. This compares five
candidate statistics on FRESH seeds at MATCHED specificity, so that a
statistic cannot win by rejecting more.

    python scripts/gate_statistic.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from duplicate_channel import make_system, scan  # noqa: E402

OUT = Path("ExpOutput/gate_statistic")
OBS_NOISE = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50)
SEEDS = (10, 11, 12, 13, 14)          # fresh: not used in the motivating run
BAD_THRESHOLD = 0.05                  # source FP above this = BAD scan
TARGET_SPECIFICITY = 0.90

STATS = {
    "MEDIAN": lambda g: float(np.median(g)),
    "MAX": lambda g: float(g.max()),
    "P95": lambda g: float(np.percentile(g, 95)),
    "IQR": lambda g: float(np.subtract(*np.percentile(g, [75, 25]))),
    "STD": lambda g: float(g.std()),
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    dest = OUT / "gate_statistic.csv"
    if dest.exists():
        rows = pd.read_csv(dest).to_dict("records")
        done = {(r["obs_noise"], r["seed"]) for r in rows}
    else:
        done = set()

    for ob in OBS_NOISE:
        for seed in SEEDS:
            if (ob, seed) in done:
                continue
            x, is_src, is_drv, _, _, _ = make_system(2, 0.05, seed, ob)
            ex, gh = scan(x, seed)
            thr = max(0.0, float(gh.max()))
            fl = ex > thr
            src_fp = float((fl & is_src).sum() / max(int(is_src.sum()), 1))
            r = {"obs_noise": ob, "seed": seed, "source_fp": src_fp,
                 "bad": src_fp > BAD_THRESHOLD,
                 "recall": float((fl & is_drv).sum()
                                 / max(int(is_drv.sum()), 1))}
            for name, fn in STATS.items():
                r[name] = fn(gh)
            rows.append(r)
            pd.DataFrame(rows).to_csv(dest, index=False)
            print(f"  obs={ob:<5} seed={seed}  srcFP {src_fp:.2f}  "
                  f"{'BAD ' if r['bad'] else 'good'}  "
                  f"med {r['MEDIAN']:+.4f}  max {r['MAX']:+.4f}  "
                  f"iqr {r['IQR']:.4f}", flush=True)

    d = pd.DataFrame(rows)
    good, bad = d[~d.bad], d[d.bad]
    print(f"\n({(time.time()-t0)/60:.1f} min)   "
          f"scans: {len(d)}   good: {len(good)}   BAD: {len(bad)}\n")
    if len(bad) == 0 or len(good) == 0:
        print("need both good and bad scans; widen the range")
        return 1

    print(f"MATCHED SPECIFICITY: threshold set to reject {100-100*TARGET_SPECIFICITY:.0f}% "
          f"of good scans")
    print(f"  {'stat':<9}{'threshold':>11}{'specificity':>13}{'SENSITIVITY':>13}")
    res = []
    for name in STATS:
        # reject when the statistic EXCEEDS the threshold
        thr = float(np.quantile(good[name], TARGET_SPECIFICITY))
        spec = float((good[name] <= thr).mean())
        sens = float((bad[name] > thr).mean())
        res.append((name, thr, spec, sens))
        print(f"  {name:<9}{thr:>11.4f}{spec:>13.2f}{sens:>13.2f}")

    res.sort(key=lambda t: -t[3])
    best, med = res[0], next(r for r in res if r[0] == "MEDIAN")
    print(f"\nG1  is MEDIAN the worst?  MEDIAN sens {med[3]:.2f}, "
          f"lowest is {min(r[3] for r in res):.2f} "
          f"({'YES' if med[3] <= min(r[3] for r in res) + 1e-9 else 'NO - motivating observation may have been coincidence'})")
    spread = max(r[3] for r in res if r[0] in ("IQR", "STD"))
    mx = next(r[3] for r in res if r[0] == "MAX")
    print(f"G2  spread beats MAX?  best spread {spread:.2f} vs MAX {mx:.2f}  "
          f"({'YES' if spread > mx else 'NO'})")
    print(f"\nDECISION (declared: replace MEDIAN only if sensitivity exceeds "
          f"it by >= 0.20)")
    print(f"  best candidate {best[0]} sens {best[3]:.2f} vs MEDIAN "
          f"{med[3]:.2f}  delta {best[3]-med[3]:+.2f}")
    if best[3] - med[3] >= 0.20 and best[0] != "MEDIAN":
        print(f"  -> {best[0]} QUALIFIES, pending confirmation on a held-out "
              f"generating process")
    else:
        print("  -> NO STATISTIC QUALIFIES. The ghost panel does not protect "
              "against\n     this failure; saturation must be reported "
              "alongside every scan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
