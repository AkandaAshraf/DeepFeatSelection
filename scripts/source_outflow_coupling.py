"""Does source outflow strengthen with coupling? This decides the line.

Declared in paper/source_detection_note.md before running.

The capacity retry removed the source-sink confound but left sources only
0.002 above the ghost. Either that margin is intrinsic - and the statistic
is dead - or it scales with how hard sources drive.

C3 names the risk: stronger driving raises the source signal AND strengthens
the sink's proxy of its driver, so the confound could return at strong
coupling. Which grows faster is what this measures.

    python scripts/source_outflow_coupling.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import source_outflow_gate as G  # noqa: E402

OUT = Path("ExpOutput/source_outflow")
COUPLINGS = (0.05, 0.15, 0.30, 0.50, 0.70)
SEEDS = (0, 1, 2)
B = 64            # the width that gave the cleanest separation in the retry
MARGIN_BAR = 0.01  # the bar declared for the retry


def coupled(n=4000, n_src=3, n_sink=6, n_iso=6, coupling=0.35, seed=0):
    """As source_outflow_gate.coupled_system but with coupling exposed."""
    rng = np.random.default_rng(seed)
    V = n_src + n_sink + n_iso
    x = np.zeros((n, V))
    x[0] = rng.uniform(0.2, 0.8, V)
    r_src = rng.uniform(3.7, 3.9, n_src)
    r_iso = rng.uniform(3.7, 3.9, n_iso)
    r_snk = rng.uniform(3.5, 3.7, n_sink)
    parent = rng.integers(0, n_src, n_sink)
    for t in range(n - 1):
        s = x[t, :n_src]
        x[t + 1, :n_src] = np.clip(r_src * s * (1 - s), 0, 1)
        k = x[t, n_src:n_src + n_sink]
        x[t + 1, n_src:n_src + n_sink] = np.clip(
            r_snk * k * (1 - k) + coupling * x[t, parent] * (1 - k), 0, 1)
        i = x[t, n_src + n_sink:]
        x[t + 1, n_src + n_sink:] = np.clip(r_iso * i * (1 - i), 0, 1)
    x += 0.01 * rng.standard_normal((n, V))
    role = np.array(["source"] * n_src + ["sink"] * n_sink
                    + ["isolated"] * n_iso)
    return x, role


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    G.BOTTLENECK = B
    print(f"device: {G.DEV}   b={B}   couplings {COUPLINGS}")
    print("C2 decisive: source must clear the ghost by "
          f"{MARGIN_BAR}\n")
    rows, t0 = [], time.time()
    for c in COUPLINGS:
        for s in SEEDS:
            G.SEED = s
            x, role = coupled(coupling=c, seed=s)
            exc, out = G.analyse(x, epochs=25)
            g_out = float(out[-1])
            exc, out = exc[:-1], out[:-1]
            src, snk, iso = (role == "source", role == "sink",
                             role == "isolated")
            r = {"coupling": c, "seed": s,
                 "out_source": float(np.median(out[src])),
                 "out_sink": float(np.median(out[snk])),
                 "out_iso": float(np.median(out[iso])),
                 "exc_sink": float(np.median(exc[snk])),
                 "ghost": g_out}
            r["margin"] = r["out_source"] - r["ghost"]
            r["gap_src_sink"] = r["out_source"] - r["out_sink"]
            rows.append(r)
            print(f"  c={c:<5} s={s}  src {r['out_source']:+.4f}  "
                  f"sink {r['out_sink']:+.4f}  ghost {r['ghost']:+.4f}  "
                  f"| margin {r['margin']:+.4f}  "
                  f"gap {r['gap_src_sink']:+.4f}  "
                  f"(sink inflow {r['exc_sink']:+.4f})", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "coupling_sweep.csv", index=False)
    print(f"\n({(time.time()-t0)/60:.1f} min)\n")

    g = d.groupby("coupling").agg(
        src=("out_source", "median"), sink=("out_sink", "median"),
        ghost=("ghost", "median"), margin=("margin", "median"),
        gap=("gap_src_sink", "median"),
        sink_inflow=("exc_sink", "median")).round(4)
    print("MEDIAN OVER SEEDS")
    print("   " + g.to_string().replace("\n", "\n   "))

    print(f"\nC1  does source outflow grow with coupling?")
    lo, hi = g.src.iloc[0], g.src.iloc[-1]
    print(f"  {g.index[0]}: {lo:+.4f}  ->  {g.index[-1]}: {hi:+.4f}   "
          f"{'GROWS' if hi > lo * 1.5 else 'FLAT - the weak signal is intrinsic'}")

    print(f"\nC2  DECISIVE: margin over ghost >= {MARGIN_BAR}?")
    for c in COUPLINGS:
        m = d[d.coupling == c].margin
        print(f"  c={c:<5} margin {m.median():+.4f}  "
              f"[{m.min():+.4f}, {m.max():+.4f}]  "
              f"{'CLEARS' if m.median() >= MARGIN_BAR else 'below bar'}")

    print("\nC3  do sinks stay separated as coupling rises?")
    for c in COUPLINGS:
        gp = d[d.coupling == c].gap_src_sink
        print(f"  c={c:<5} src-sink gap {gp.median():+.4f}  "
              f"positive in {int((gp > 0).sum())}/{len(gp)}")

    print("\nC4  ghost")
    print(f"  max |ghost| over all cells: {d.ghost.abs().max():.4f}")

    best_c = g.margin.idxmax()
    ok = (g.margin.max() >= MARGIN_BAR
          and g.loc[best_c, "gap"] > 0
          and d.ghost.abs().max() < 0.02)
    print("\nVERDICT (declared: reopen only if C2 and C3 and C4 all hold)")
    print(f"  best margin {g.margin.max():+.4f} at coupling {best_c}, "
          f"gap there {g.loc[best_c,'gap']:+.4f}")
    print(f"  -> {'LINE REOPENED' if ok else 'LINE CLOSED: unconfounded, correctly signed, too weak to use'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
