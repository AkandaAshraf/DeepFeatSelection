"""Calibrate the outflow bar against the SINK distribution, not the ghost.

Pre-registration: paper/sink_bar_protocol.md, committed before any bar was
computed.

The ghost-calibrated attempt failed: the bar came out at -0.00005 and sinks
cleared it 28/30, because the ghost is a null for "influences nothing" while
the thing that actually confuses outflow is "carries a proxy of an
influencer". Sinks are that confuser, so sinks are the null.

    bar = q95 of the sink outflow distribution   (5% sink false-alarm rate)

Calibration seeds 300-329 define the bar and supply nothing else; disjoint
test seeds 400-429 measure sensitivity at coupling 0.50, with coupling 0.30
reported for description only.

    python scripts/sink_bar.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import source_outflow_gate as G  # noqa: E402
from source_outflow_coupling import coupled  # noqa: E402

OUT = Path("ExpOutput/sink_bar")
B = 64                            # 4V, the outflow capacity requirement
ALPHA = 0.05
CAL_SEEDS = range(300, 330)
TEST_SEEDS = range(400, 430)
SENS_BAR = 0.80
GHOST_BAR = -0.00005              # what the previous, wrong calibration gave
OLD_BAR = 0.01


def one(coupling, seed):
    G.BOTTLENECK, G.SEED = B, seed
    x, role = coupled(coupling=coupling, seed=seed)
    exc, out = G.analyse(x, epochs=25)
    ghost = float(out[-1])
    out = out[:-1]
    return {"coupling": coupling, "seed": seed, "ghost": ghost,
            "source": float(np.median(out[role == "source"])),
            "sink": float(np.median(out[role == "sink"])),
            "iso": float(np.median(out[role == "isolated"]))}


def sweep(coupling, seeds, tag):
    rows = []
    for s in seeds:
        t = time.time()
        r = one(coupling, s)
        r["set"] = tag
        rows.append(r)
        print(f"  {tag:11s} c={coupling} s={s}  source {r['source']:+.4f}  "
              f"sink {r['sink']:+.4f}  iso {r['iso']:+.4f}  "
              f"({time.time()-t:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device: {G.DEV}   b={B} (=4V)   alpha={ALPHA}\n")
    t0 = time.time()

    print("CALIBRATION (sink distribution only; nothing else taken from it)")
    cal = sweep(0.50, CAL_SEEDS, "calibration")
    print("\nTEST  coupling 0.50")
    test = sweep(0.50, TEST_SEEDS, "test")
    print("\nDESCRIPTIVE  coupling 0.30 (weak coupling, no pass/fail)")
    weak = sweep(0.30, TEST_SEEDS, "weak")
    d = pd.concat([cal, test, weak], ignore_index=True)
    d.to_csv(OUT / "runs.csv", index=False)
    print(f"\n({(time.time()-t0)/60:.1f} min)\n")

    bar = float(np.quantile(cal.sink, 1 - ALPHA))
    print(f"S1  sink-calibrated bar q{int((1-ALPHA)*100)}(sink) = {bar:+.5f}")
    print(f"    sink distribution: median {cal.sink.median():+.5f}  "
          f"[{cal.sink.min():+.5f}, {cal.sink.max():+.5f}]")
    print(f"    ghost-calibrated bar was {GHOST_BAR:+.5f}; declared 0.01 bar "
          f"was {OLD_BAR}")
    print(f"    higher than the ghost bar? "
          f"{'YES, as declared' if bar > GHOST_BAR else 'NO'}")

    sens = float((test.source > bar).mean())
    print(f"\nS2  DECISIVE: sensitivity at a {int(ALPHA*100)}% SINK "
          f"false-alarm rate, coupling 0.50")
    print(f"    {int((test.source > bar).sum())}/{len(test)} runs clear the "
          f"bar  ->  {sens:.2f}   {'PASS' if sens >= SENS_BAR else 'FAIL'} "
          f"(bar {SENS_BAR})")
    print(f"    source median {test.source.median():+.5f}   "
          f"sink median {test.sink.median():+.5f}   "
          f"margin {test.source.median()-bar:+.5f}")

    overlap = float((test.source <= bar).mean())
    print(f"\nS3  do sources and sinks overlap at this false-alarm rate? "
          f"{overlap:.2f} of runs fail to clear")

    ws = float((weak.source > bar).mean())
    print(f"\nDESCRIPTIVE  coupling 0.30: {ws:.2f} of runs clear the bar "
          "(no pass/fail attached)")

    pd.DataFrame([{"bar": bar, "alpha": ALPHA, "sensitivity_c050": sens,
                   "overlap": overlap, "clear_rate_c030": ws,
                   "source_median_c050": float(test.source.median()),
                   "sink_median_c050": float(test.sink.median())}]).to_csv(
        OUT / "summary.csv", index=False)

    print("\nVERDICT (rule fixed before running)")
    if sens >= SENS_BAR:
        print(f"    -> ADOPTED. Outflow reports sensitivity {sens:.2f} at a "
              f"{int(ALPHA*100)}% sink\n       false-alarm rate, bar "
              f"{bar:+.5f}. The declared 0.01 is retired from this line.")
    else:
        print(f"    -> CLOSED. At a {int(ALPHA*100)}% sink false-alarm rate "
              f"sensitivity is {sens:.2f},\n       below the declared "
              f"{SENS_BAR}. The margin the line was reopened on does not\n"
              "       survive calibration against the right null.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
