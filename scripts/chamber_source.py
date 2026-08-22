"""Source detection on a real physical system with known ground truth.

Pre-registration: paper/chamber_source_protocol.md, committed before any
statistic was computed on this data.

Causal chamber wind tunnel (Gamella et al. 2025), 28 random-walk runs. The
experimenter SETS hatch, pot_1 and pot_2; everything else is a physical
consequence. Ground truth is structural, not annotated.

    python scripts/chamber_source.py
"""

from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import source_outflow_gate as G  # noqa: E402

DATA = "Data/causalchamber/wt_walks_v1/*.csv"
OUT = Path("ExpOutput/chamber_source")
SOURCES = ["hatch", "pot_1", "pot_2"]
SENSORS = ["load_in", "load_out", "current_in", "current_out",
           "rpm_in", "rpm_out", "pressure_upwind", "pressure_downwind",
           "pressure_ambient", "pressure_intake", "mic",
           "signal_1", "signal_2"]
COLS = SOURCES + SENSORS
B = 32                      # b ~ 2V with V = 16
BAR = 0.01                  # the margin bar used throughout this line


def load_runs():
    runs = []
    for f in sorted(glob.glob(DATA)):
        d = pd.read_csv(f)
        if not set(COLS).issubset(d.columns):
            continue
        runs.append(d[COLS].to_numpy(np.float64))
    return runs


def analyse(x, seed, epochs=25):
    """Inflow (excess) and outflow per channel, plus the ghost."""
    G.BOTTLENECK = B
    G.SEED = seed
    # standardise: chamber units differ by orders of magnitude
    x = (x - x.mean(0)) / (x.std(0) + 1e-12)
    exc, out = G.analyse(x, epochs=epochs)
    return exc, out


def summarise(exc, out, tag):
    n = len(SOURCES)
    g_exc, g_out = float(exc[-1]), float(out[-1])
    exc, out = exc[:-1], out[:-1]
    src_out, sen_out = np.median(out[:n]), np.median(out[n:])
    return {
        "arm": tag,
        "src_inflow": float(np.median(exc[:n])),
        "sen_inflow": float(np.median(exc[n:])),
        "src_outflow": float(src_out),
        "sen_outflow": float(sen_out),
        "margin": float(src_out - g_out),
        "gap": float(src_out - sen_out),
        "ghost_out": g_out, "ghost_exc": g_exc,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    print(f"runs: {len(runs)}   n per run: {runs[0].shape[0]}   "
          f"V = {runs[0].shape[1]}  (b={B})")
    print(f"sources: {SOURCES}\n")
    t0 = time.time()

    # ---- arm 1: per run, n below the validated floor, declared ----------
    rows = []
    for i, x in enumerate(runs):
        try:
            exc, out = analyse(x, seed=i)
            r = summarise(exc, out, "per_run")
            r["run"] = i
            rows.append(r)
        except Exception as exc_:
            print(f"  run {i} failed: {str(exc_)[:60]}")
    d1 = pd.DataFrame(rows)
    print(f"PER-RUN arm: {len(d1)} runs at n={runs[0].shape[0]} "
          f"(below the ~2000 floor, declared)")
    print(f"  source outflow  {d1.src_outflow.median():+.4f}   "
          f"sensor outflow {d1.sen_outflow.median():+.4f}")
    print(f"  source inflow   {d1.src_inflow.median():+.4f}   "
          f"sensor inflow  {d1.sen_inflow.median():+.4f}")
    print(f"  margin over ghost {d1.margin.median():+.4f}   "
          f"gap {d1.gap.median():+.4f}   "
          f"positive in {int((d1.gap > 0).sum())}/{len(d1)} runs")
    print(f"  ghost {d1.ghost_out.median():+.4f}")

    # ---- arm 2: concatenated ------------------------------------------
    xc = np.concatenate(runs, axis=0)
    exc, out = analyse(xc, seed=0)
    d2 = summarise(exc, out, "concat")
    print(f"\nCONCATENATED arm: n={xc.shape[0]} "
          f"({len(runs)-1} declared discontinuities)")
    print(f"  source outflow  {d2['src_outflow']:+.4f}   "
          f"sensor outflow {d2['sen_outflow']:+.4f}")
    print(f"  source inflow   {d2['src_inflow']:+.4f}   "
          f"sensor inflow  {d2['sen_inflow']:+.4f}")
    print(f"  margin over ghost {d2['margin']:+.4f}   gap {d2['gap']:+.4f}")
    print(f"  ghost {d2['ghost_out']:+.4f}")

    pd.concat([d1, pd.DataFrame([d2])]).to_csv(
        OUT / "chamber_source.csv", index=False)

    # ---- the declared synchrony diagnostic -----------------------------
    from source_outflow_gate import E, embed, poly3, ridge_r2
    z = (xc - xc.mean(0)) / (xc.std(0) + 1e-12)
    emb = embed(z)
    m = emb.shape[0]
    a, b = int(0.6 * m), int(0.8 * m)
    tr_i, te_i = np.arange(0, a - 1), np.arange(b, m - 1)
    mu, sd = emb[:a].mean(0), emb[:a].std(0) + 1e-12
    zs = np.clip((emb - mu) / sd, -20, 20).astype(np.float32)
    lead = zs[:, [j * E for j in range(len(COLS))]]
    self_r2 = np.array([
        ridge_r2(poly3(zs[:, q * E:(q + 1) * E])[tr_i], lead[tr_i + 1, q],
                 poly3(zs[:, q * E:(q + 1) * E])[te_i], lead[te_i + 1, q])
        for q in range(len(COLS))])
    sen_self = float(np.median(self_r2[len(SOURCES):]))
    print(f"\nSYNCHRONY DIAGNOSTIC (declared): sensor self-R2 median "
          f"{sen_self:.3f}")
    synchronous = sen_self > 0.95
    print(f"  {'NEAR-SYNCHRONOUS at this sampling rate' if synchronous else 'not synchronous'}")

    print("\nVERDICT (declared before running)")
    ch2 = (d1.margin.median() >= BAR and d1.gap.median() > 0
           and d2["margin"] >= BAR and d2["gap"] > 0)
    ch3 = abs(d1.ghost_out.median()) <= 0.02 and abs(d2["ghost_out"]) <= 0.02
    ch4 = (d1.gap.median() > 0) == (d2["gap"] > 0)
    print(f"  CH2 margin >= {BAR} and actuators above sensors, BOTH arms: "
          f"{'PASS' if ch2 else 'FAIL'}")
    print(f"  CH3 ghost clean: {'PASS' if ch3 else 'FAIL'}")
    print(f"  CH4 arms agree on ordering: {'PASS' if ch4 else 'FAIL'}")
    if ch2 and ch3 and ch4:
        print("  -> PASSES. Known sources in a real physical system are "
              "detected.")
    elif synchronous:
        print("  -> UNINFORMATIVE. The system is near-synchronous at this "
              "sampling rate;\n     no claim either way.")
    else:
        print("  -> FAILS. Works on synthetic data, not on a real system "
              "whose sources\n     are known with certainty. The line closes.")
    print(f"\n({(time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
