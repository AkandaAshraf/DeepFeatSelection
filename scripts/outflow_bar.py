"""Replace the declared 0.01 outflow bar with a calibrated one.

Pre-registration: paper/outflow_bar_protocol.md, committed before this was
written or run.

The ghost - a circularly shifted copy of a real channel - is a draw from
exactly the null the bar must exclude. The bar becomes q95 of that null, a
5% false-alarm rate, fitted on calibration seeds and applied to disjoint test
and control seeds.

    python scripts/outflow_bar.py
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

OUT = Path("ExpOutput/outflow_bar")
B = 64
ALPHA = 0.05                      # stated operating point: 5% false alarms
CAL_SEEDS = range(100, 130)
TEST_SEEDS = range(200, 230)
SENS_BAR, CTRL_BAR = 0.80, 0.20
OLD_BAR = 0.01


def one(coupling, seed):
    G.BOTTLENECK, G.SEED = B, seed
    x, role = coupled(coupling=coupling, seed=seed)
    exc, out = G.analyse(x, epochs=25)
    ghost = float(out[-1])
    out = out[:-1]
    return {"coupling": coupling, "seed": seed, "ghost": ghost,
            "source": float(np.median(out[role == "source"])),
            "sink": float(np.median(out[role == "sink"]))}


def sweep(coupling, seeds, tag):
    rows = []
    for s in seeds:
        t = time.time()
        r = one(coupling, s)
        r["set"] = tag
        rows.append(r)
        print(f"  {tag:11s} c={coupling} s={s}  source {r['source']:+.4f}  "
              f"sink {r['sink']:+.4f}  ghost {r['ghost']:+.4f}  "
              f"({time.time()-t:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device: {G.DEV}   b={B}   alpha={ALPHA}\n")
    t0 = time.time()

    print("CALIBRATION (ghost null only; nothing else is taken from these)")
    cal = sweep(0.50, CAL_SEEDS, "calibration")
    print("\nTEST  coupling 0.50, the regime where outflow is claimed to work")
    test = sweep(0.50, TEST_SEEDS, "test")
    print("\nCONTROL  coupling 0.30, known not to work")
    ctrl = sweep(0.30, TEST_SEEDS, "control")
    d = pd.concat([cal, test, ctrl], ignore_index=True)
    d.to_csv(OUT / "runs.csv", index=False)
    print(f"\n({(time.time()-t0)/60:.1f} min)\n")

    # ---- O1 -------------------------------------------------------------
    gm = float(cal.ghost.median())
    print(f"O1  calibration ghost null: median {gm:+.4f}  "
          f"[{cal.ghost.min():+.4f}, {cal.ghost.max():+.4f}]")
    if abs(gm) > 0.002:
        print("   -> NOT a null. Experiment VOID.")
        return 0
    print("   -> centred at zero, usable as a null")

    bar = float(np.quantile(cal.ghost, 1 - ALPHA))
    print(f"\nO2  calibrated bar q{int((1-ALPHA)*100)}(ghost) = {bar:+.4f}   "
          f"(declared constant was {OLD_BAR})")
    print("   quantiles of the null: " + "  ".join(
        f"q{int(q*100)} {np.quantile(cal.ghost, q):+.4f}"
        for q in (0.50, 0.75, 0.90, 0.95, 0.99)))

    # ---- O5 checked before anything is claimed --------------------------
    if bar >= OLD_BAR:
        print(f"\nO5  the calibrated bar {bar:+.4f} is at or ABOVE the "
              f"declared {OLD_BAR}.")
        print("   The reopening margin of 0.0107 was inside the null.")
        print("   -> THE LINE CLOSES, as declared before running.")
        return 0
    print(f"   -> below {OLD_BAR}: the declared constant was conservative")

    # ---- O3, O4 ---------------------------------------------------------
    sens = float((test.source > bar).mean())
    spec_fail = float((ctrl.source > bar).mean())
    print(f"\nO3  DECISIVE: sensitivity at a {int(ALPHA*100)}% false-alarm "
          f"rate, coupling 0.50")
    print(f"   {int((test.source > bar).sum())}/{len(test)} runs clear the "
          f"bar  ->  sensitivity {sens:.2f}   "
          f"{'PASS' if sens >= SENS_BAR else 'FAIL'} (bar {SENS_BAR})")
    print(f"   source outflow median {test.source.median():+.4f}   "
          f"sink {test.sink.median():+.4f}")

    print(f"\nO4  CONTROL at coupling 0.30")
    print(f"   {int((ctrl.source > bar).sum())}/{len(ctrl)} clear the bar  ->  "
          f"{spec_fail:.2f}   "
          f"{'PASS' if spec_fail <= CTRL_BAR else 'FAIL'} (bar {CTRL_BAR})")

    print(f"\nO6  margin against the calibrated bar (no prediction was made): "
          f"{test.source.median() - bar:+.4f}")

    print("\nVERDICT (rule fixed before running)")
    if sens >= SENS_BAR and spec_fail <= CTRL_BAR:
        print(f"   -> ADOPTED. The reported quantity for outflow becomes "
              f"sensitivity at a\n      {int(ALPHA*100)}% false-alarm rate "
              f"against the ghost null: {sens:.2f} at coupling 0.50,\n"
              f"      {spec_fail:.2f} at 0.30. The constant {OLD_BAR} is "
              "retired from this line.")
    elif sens < SENS_BAR:
        print("   -> NOT USABLE. At a stated false-alarm rate the statistic "
              "does not reach\n      the declared sensitivity. The margin of "
              "0.0107 overstated what it can do.")
    else:
        print("   -> DOES NOT SEPARATE. The control clears the bar at nearly "
              "the test rate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
