"""Does the chamber's SHAPE explain the synthetic/real tension?

Pre-registration: paper/chamber_shape_protocol.md, committed before any
chamber-shaped system was run.

The sink-calibrated bar closed the synthetic line at sensitivity 0.53 while
the chamber sits at AUC 0.916, and the paper admits three candidate
explanations without choosing. The first is testable: the synthetic family
is 3 sources against 6 sinks; the chamber is 2 actuators against 11 sensors.
Same bar, same alpha, same seeds discipline, same b=4V, same coupling --
only the shape changes.

    python scripts/chamber_shape.py
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

OUT = Path("ExpOutput/chamber_shape")
ALPHA, COUPLING = 0.05, 0.50
CAL_SEEDS, TEST_SEEDS = range(500, 530), range(600, 630)
INCUMBENT_SENS = 0.53

SHAPES = {
    "synthetic (3 src / 6 sink / 6 iso)":  dict(n_src=3, n_sink=6,  n_iso=6),
    "chamber (2 src / 11 sink / 0 iso)":   dict(n_src=2, n_sink=11, n_iso=0),
    "chamber at V=15 (2 / 11 / 2)":        dict(n_src=2, n_sink=11, n_iso=2),
}


def auc(pos, neg):
    """Identical to error_metrics.auc; see scripts/test_auc_identical.py."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def one(shape, seed):
    V = shape["n_src"] + shape["n_sink"] + shape["n_iso"] + 1   # +1 ghost
    G.BOTTLENECK, G.SEED = 4 * V, seed
    x, role = coupled(coupling=COUPLING, seed=seed, **shape)
    exc, out = G.analyse(x, epochs=25)
    out = out[:-1]                       # drop ghost
    src = out[role == "source"]
    snk = out[role == "sink"]
    return {"seed": seed, "b": 4 * V,
            "source": float(np.median(src)), "sink": float(np.median(snk)),
            "auc": auc(src, snk)}        # per-run AUC: source vs sink channels


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"alpha={ALPHA}  coupling={COUPLING}  b=4V  "
          f"incumbent sensitivity {INCUMBENT_SENS}\n")
    rows, t0 = [], time.time()

    for name, shape in SHAPES.items():
        print(f"=== {name} ===", flush=True)
        cal = pd.DataFrame([one(shape, s) for s in CAL_SEEDS])
        test = pd.DataFrame([one(shape, s) for s in TEST_SEEDS])
        bar = float(np.quantile(cal.sink, 1 - ALPHA))
        sens = float((test.source > bar).mean())
        a = float(test.auc.median())
        rows.append({"shape": name, "bar": bar, "sensitivity": sens,
                     "auc_median": a,
                     "source_med": float(test.source.median()),
                     "sink_med": float(test.sink.median())})
        print(f"  bar q95(sink) {bar:+.5f}   source med "
              f"{test.source.median():+.5f}   sink med "
              f"{test.sink.median():+.5f}")
        print(f"  sensitivity {sens:.2f}   AUC {a:.3f}   "
              f"({(time.time()-t0)/60:.1f} min)\n", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "results.csv", index=False)
    print(d.round(4).to_string(index=False))

    inc = d.iloc[0].sensitivity
    cham = d.iloc[1].sensitivity
    cham15 = d.iloc[2].sensitivity

    print("\nC4  shape vs width entanglement check: chamber shapes at V=13 "
          f"and V=15 differ by {abs(cham-cham15):.2f}")
    c4 = abs(cham - cham15) <= 0.15
    print(f"    -> {'consistent, arms interpretable' if c4 else 'ENTANGLED - no arm interpretable'}")

    print(f"\nC3  AUC exceeds thresholded sensitivity in every arm? "
          f"{bool((d.auc_median > d.sensitivity).all())} (recorded, expected)")

    print(f"\nC1  DECISIVE: chamber-shape sensitivity {cham:.2f} vs "
          f"incumbent {inc:.2f}")
    print("\nVERDICT (rule fixed before running)")
    if not c4:
        print("   -> VOID on C4: shape and width are entangled.")
    elif cham > inc:
        print(f"   -> SHAPE EXPLAINS. {cham:.2f} > {inc:.2f}. The chamber's "
              "2-against-11 structure is\n      more favourable than the "
              "3-against-6 family; the closure is a statement\n      about "
              "that family, not about the statistic everywhere.")
        if cham >= 0.80:
            print("      (It also clears 0.80 -- NOT predicted in advance "
                  "(C2), and a separate,\n       higher bar that would "
                  "reopen the line. Reported, not claimed.)")
    else:
        print(f"   -> SHAPE DOES NOT EXPLAIN. {cham:.2f} <= {inc:.2f}. Shape "
              "is eliminated; two\n      candidates remain (AUC forgiveness, "
              "or fortune) and the chamber result\n      becomes materially "
              "more suspect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
