"""Does outflow survive training maturity where it actually works?

Declared in paper/source_detection_note.md before running. Third and final
test of this statistic.

Mechanism 1 of the companion paper: difference-based importance scores rise,
peak and collapse as a model matures and learns alternative routes. Outflow
did not decay at coupling 0.35 - but it does not WORK at 0.35. At the
couplings where it works, the sink carries a strong proxy of its driver, so
a maturing encoder has more opportunity to route around the source through
that proxy. That is exactly the condition under which Mechanism 1 bites, and
it has never been tested.

    python scripts/source_outflow_maturity.py
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

OUT = Path("ExpOutput/source_outflow")
EPOCHS = (5, 15, 25, 40, 80)
COUPLINGS = (0.5, 0.7)
SEEDS = (0, 1, 2)
B = 64


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    G.BOTTLENECK = B
    print(f"device: {G.DEV}   b={B}   couplings {COUPLINGS}   "
          f"epochs {EPOCHS}")
    print("decay toward zero fails; flat or rising passes\n")
    rows, t0 = [], time.time()
    for c in COUPLINGS:
        for ep in EPOCHS:
            for s in SEEDS:
                G.SEED = s
                x, role = coupled(coupling=c, seed=s)
                exc, out = G.analyse(x, epochs=ep)
                g_out = float(out[-1])
                exc, out = exc[:-1], out[:-1]
                src, snk = role == "source", role == "sink"
                r = {"coupling": c, "epochs": ep, "seed": s,
                     "out_source": float(np.median(out[src])),
                     "out_sink": float(np.median(out[snk])),
                     "ghost": g_out}
                r["margin"] = r["out_source"] - r["ghost"]
                r["gap"] = r["out_source"] - r["out_sink"]
                rows.append(r)
                print(f"  c={c} ep={ep:<3} s={s}  src {r['out_source']:+.4f}  "
                      f"sink {r['out_sink']:+.4f}  gap {r['gap']:+.4f}  "
                      f"margin {r['margin']:+.4f}  ghost {r['ghost']:+.4f}",
                      flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "maturity_sweep.csv", index=False)
    print(f"\n({(time.time()-t0)/60:.1f} min)\n")

    verdicts = []
    for c in COUPLINGS:
        sub = d[d.coupling == c]
        g = sub.groupby("epochs").agg(
            src=("out_source", "median"), sink=("out_sink", "median"),
            gap=("gap", "median"), margin=("margin", "median"),
            ghost=("ghost", "median")).round(4)
        print(f"COUPLING {c}")
        print("   " + g.to_string().replace("\n", "\n   "))
        first, last = g.src.iloc[0], g.src.iloc[-1]
        gfirst, glast = g.gap.iloc[0], g.gap.iloc[-1]
        m1 = last >= 0.6 * first          # no collapse toward zero
        m2 = glast > 0 and glast >= 0.6 * gfirst
        m3 = sub.ghost.abs().max() < 0.02
        print(f"  M1 source {first:+.4f} -> {last:+.4f}   "
              f"{'PASS' if m1 else 'FAIL - decays with training'}")
        print(f"  M2 gap    {gfirst:+.4f} -> {glast:+.4f}   "
              f"{'PASS' if m2 else 'FAIL - closes with training'}")
        print(f"  M3 ghost  max |{sub.ghost.abs().max():.4f}|   "
              f"{'PASS' if m3 else 'FAIL'}\n")
        verdicts.append(m1 and m2 and m3)

    print("VERDICT (declared: any M1 or M2 failure closes the line "
          "permanently)")
    if all(verdicts):
        print("  -> SURVIVES. Outflow is not a difference-based importance")
        print("     score in the regime where it works. Next step is real")
        print("     data, not another synthetic variant.")
    else:
        print("  -> CLOSED PERMANENTLY. Outflow decays or its gap closes with")
        print("     training, so it is Mechanism 1 after all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
