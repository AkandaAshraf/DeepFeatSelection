"""Does the rejected source-detection statistic work at sufficient capacity?

Declared in paper/source_detection_note.md before running.

The rejection diagnosed a lossy 16-dimensional code: outflow conditions on
the code, and an insufficient code lets driver information leak into a
sink's increment, so sinks score like sources. The width experiment finds
MACE needs b of order 2V regardless, so the code is close to sufficient at
the widths it actually requires. This sweeps b and asks whether the
source-sink gap opens.

R2 is decisive: sources must separate from SINKS, not merely from isolated
channels.

    python scripts/source_outflow_capacity.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import source_outflow_gate as G  # noqa: E402

OUT = Path("ExpOutput/source_outflow")
B_VALUES = (16, 32, 64, 128)
SEEDS = (0, 1, 2)


def run(b, seed):
    G.BOTTLENECK = b                      # the swept parameter
    G.SEED = seed
    x, role = G.coupled_system(seed=seed)
    exc, out = G.analyse(x, epochs=25)
    g_exc, g_out = float(exc[-1]), float(out[-1])
    exc, out = exc[:-1], out[:-1]
    src, snk, iso = role == "source", role == "sink", role == "isolated"
    return {
        "b": b, "seed": seed,
        "out_source": float(np.median(out[src])),
        "out_sink": float(np.median(out[snk])),
        "out_iso": float(np.median(out[iso])),
        "exc_source": float(np.median(exc[src])),
        "exc_sink": float(np.median(exc[snk])),
        "gap_src_sink": float(np.median(out[src]) - np.median(out[snk])),
        "gap_src_iso": float(np.median(out[src]) - np.median(out[iso])),
        "ghost_out": g_out, "ghost_exc": g_exc,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device: {G.DEV}   sweeping b over {B_VALUES}")
    print("R2 is decisive: sources must separate from SINKS\n")
    rows, t0 = [], time.time()
    for b in B_VALUES:
        for s in SEEDS:
            r = run(b, s)
            rows.append(r)
            print(f"  b={b:<4} s={s}  source {r['out_source']:+.4f}  "
                  f"sink {r['out_sink']:+.4f}  iso {r['out_iso']:+.4f}  "
                  f"| src-sink {r['gap_src_sink']:+.4f}  "
                  f"ghost {r['ghost_out']:+.4f}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "capacity_sweep.csv", index=False)
    print(f"\n({(time.time()-t0)/60:.1f} min)\n")

    g = d.groupby("b").agg(
        src=("out_source", "median"), sink=("out_sink", "median"),
        iso=("out_iso", "median"), gap_ss=("gap_src_sink", "median"),
        gap_si=("gap_src_iso", "median"), ghost=("ghost_out", "median"))
    print("MEDIAN OVER SEEDS")
    print("   " + g.round(4).to_string().replace("\n", "\n   "))

    print("\nR2  DECISIVE: does the source-sink gap open with capacity?")
    for b in B_VALUES:
        v = d[d.b == b].gap_src_sink
        print(f"  b={b:<4} gap {v.median():+.4f}  "
              f"[{v.min():+.4f}, {v.max():+.4f}]  "
              f"positive in {int((v > 0).sum())}/{len(v)} seeds")
    best = g.gap_ss.max()
    bb = int(g.gap_ss.idxmax())
    print(f"  -> best gap {best:+.4f} at b={bb}")

    print("\nR3  GHOST at each width")
    for b in B_VALUES:
        v = d[d.b == b].ghost_out
        print(f"  b={b:<4} {v.median():+.4f}  "
              f"{'clean' if abs(v.median()) < 0.02 else 'ELEVATED'}")

    ghost_ok = abs(d[d.b == bb].ghost_out.median()) < 0.02
    lifted = best > 0.01 and ghost_ok
    print(f"\nVERDICT (declared: lift only if R2 AND R3 hold)")
    print(f"  -> {'REJECTION LIFTED at b=' + str(bb) if lifted else 'CONFIRMED REJECTED: the barrier is the confound, not the compression'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
