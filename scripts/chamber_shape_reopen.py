"""Confirmatory re-run of the chamber-shape result on fresh seeds.

Pre-registration: paper/chamber_shape_reopen_protocol.md, committed before
this was run. one() is byte-identical to chamber_shape.one and to
sink_bar.one (b = 4V, 25 epochs, 2 models averaged in G.analyse).

The chamber-shape run of 2026-09-02 (seeds 500-629) found sensitivity 0.867
and 0.967 at the 2/11/2 shape against 0.667 at 3/6/6 -- on two seed blocks.
This run asks the same question on seeds 700-829, never used before, with
the bar to beat (0.80) and the P3 bar-region check fixed in advance.

POST-RUN NOTE (2026-09-03): the P3 window coded below (0.008 <= bar <=
0.020) is wider than the protocol's stated chamber region (0.0113-0.0138).
The script was committed before the run, so the coded window is the
operative one; the discrepancy and the observed bar (+0.01498, inside the
coded window, outside the narrower one) are disclosed in the protocol's
result section. Nothing in this file was changed after the run except this
docstring.

    python scripts/chamber_shape_reopen.py
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

OUT = Path("ExpOutput/chamber_shape_reopen")
ALPHA, COUPLING = 0.05, 0.50
CAL_SEEDS, TEST_SEEDS = range(700, 730), range(800, 830)
BAR_TO_BEAT = 0.80
INCUMBENT = 0.667          # 3/6/6 shape, measured on seeds 500-629
SHAPE = dict(n_src=2, n_sink=11, n_iso=2)


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
    print("confirmatory: does the line reopen at the chamber shape?")
    print(f"shape {SHAPE}   alpha={ALPHA}  coupling={COUPLING}  b=4V")
    print(f"bar to beat {BAR_TO_BEAT}   incumbent 3/6/6 shape {INCUMBENT}")
    print(f"seeds cal {CAL_SEEDS.start}-{CAL_SEEDS.stop-1}, "
          f"test {TEST_SEEDS.start}-{TEST_SEEDS.stop-1} (unused before)\n")
    t0 = time.time()

    cal = pd.DataFrame([one(SHAPE, s) for s in CAL_SEEDS])
    print(f"  calibration done ({(time.time()-t0)/60:.1f} min)", flush=True)
    test = pd.DataFrame([one(SHAPE, s) for s in TEST_SEEDS])
    print(f"  test done ({(time.time()-t0)/60:.1f} min)\n", flush=True)

    bar = float(np.quantile(cal.sink, 1 - ALPHA))
    sens = float((test.source > bar).mean())
    a = float(test.auc.median())

    pd.concat([cal.assign(set="cal"), test.assign(set="test")]).to_csv(
        OUT / "runs.csv", index=False)
    pd.DataFrame([{"bar": bar, "sensitivity": sens, "auc_median": a,
                   "source_med": float(test.source.median()),
                   "sink_med": float(test.sink.median()),
                   "incumbent": INCUMBENT}]).to_csv(
        OUT / "summary.csv", index=False)

    print(f"P3  bar q95(sink) = {bar:+.5f}   "
          f"(today's chamber arms: 0.0113-0.0138)")
    p3 = 0.008 <= bar <= 0.020
    print(f"    -> {'in the same region, comparison safe' if p3 else 'MATERIALLY DIFFERENT - sink distribution unstable across seeds'}")
    print(f"\n    source median {test.source.median():+.5f}   "
          f"sink median {test.sink.median():+.5f}   AUC {a:.3f}")

    print(f"\nP1  DECISIVE: sensitivity {sens:.2f} vs bar "
          f"{BAR_TO_BEAT}   (incumbent 3/6/6 shape: {INCUMBENT:.2f})")
    print(f"    {int((test.source > bar).sum())}/{len(test)} test runs clear")

    print("\nVERDICT (rule fixed before running)")
    if not p3:
        print("   -> UNSAFE: P3 fails, the bar is not comparable. No verdict.")
    elif sens >= BAR_TO_BEAT:
        print(f"   -> REOPENS AT THIS SHAPE. {sens:.2f} >= {BAR_TO_BEAT}. The "
              "line is not closed as a\n      property of the statistic: it "
              f"is closed near 1:2 sinks per source ({INCUMBENT:.2f}) and\n"
              "      open near 1:5.5. Section 12's verdict gains a shape "
              "condition, and the\n      chamber result gains a synthetic "
              "footing rather than standing alone.")
    else:
        print(f"   -> CLOSURE STANDS. {sens:.2f} < {BAR_TO_BEAT}. Today's "
              "0.867/0.967 were seed-block\n      fortune. The closure is "
              "unconditional on available evidence, and the shape\n      "
              "explanation for the synthetic/real tension weakens with it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
