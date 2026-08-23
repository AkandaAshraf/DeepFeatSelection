"""Does a dataset carry lagged driver-to-target information at its own rate?

Pre-registration: paper/dataset_fitness_protocol.md, committed before this
quantity was computed on any data.

The pass mark is CALIBRATED, not declared: it is the value measured on the
weakest coupled-logistic system at which outflow is known to work (coupling
0.50), with coupling 0.30 - the strongest system where it is known to fail -
as the floor. Linear only; no autoencoder is trained and no outflow value is
produced.

    python scripts/dataset_fitness.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from source_outflow_gate import E, TAU, embed, poly2, ridge_r2  # noqa: E402
from source_outflow_coupling import coupled  # noqa: E402

DATA = "Data/causalchamber/wt_walks_v1/actuators_random_walk_*.csv"
OUT = Path("ExpOutput/dataset_fitness")
SRC = ["hatch", "pot_1", "pot_2"]
SEN = ["load_in", "load_out", "current_in", "current_out", "rpm_in",
       "rpm_out", "pressure_upwind", "pressure_downwind", "pressure_ambient",
       "pressure_intake", "mic", "signal_1", "signal_2"]
COUPLINGS = (0.05, 0.15, 0.30, 0.50, 0.70)
SEEDS = (0, 1, 2)
GHOST_BAR = 0.002


def lag_info(x, n_drivers, ghost=False):
    """Median over targets of R2 gain from the drivers' delay embedding."""
    z = (x - x.mean(0)) / (x.std(0) + 1e-12)
    emb = embed(z)
    m, V = emb.shape[0], x.shape[1]
    a, b = int(0.6 * m), int(0.8 * m)
    tr, te = np.arange(0, a - 1), np.arange(b, m - 1)
    mu, sd = emb[:a].mean(0), emb[:a].std(0) + 1e-12
    zs = np.clip((emb - mu) / sd, -20, 20).astype(np.float32)
    lead = zs[:, [j * E for j in range(V)]]

    drv = zs[:, :n_drivers * E]
    if ghost:                       # declared control: shift the driver block
        drv = np.roll(drv, m // 3, axis=0)

    gains = []
    for q in range(n_drivers, V):
        own = poly2(zs[:, q * E:(q + 1) * E])
        base = ridge_r2(own[tr], lead[tr + 1, q], own[te], lead[te + 1, q])
        both = np.hstack([own, drv])
        with_d = ridge_r2(both[tr], lead[tr + 1, q],
                          both[te], lead[te + 1, q])
        gains.append(with_d - base)
    return float(np.median(gains))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"E={E} TAU={TAU}  ghost bar {GHOST_BAR}\n")

    # ---- reference: where the statistic is known to work and to fail -----
    print("REFERENCE  coupled logistic, 3 sources -> 6 sinks (+6 isolated)")
    ref = []
    for c in COUPLINGS:
        v = [lag_info(coupled(coupling=c, seed=s)[0][:, :9], 3)
             for s in SEEDS]
        g = [lag_info(coupled(coupling=c, seed=s)[0][:, :9], 3, ghost=True)
             for s in SEEDS]
        ref.append({"coupling": c, "lag_info": float(np.median(v)),
                    "ghost": float(np.median(g))})
        works = "outflow WORKS" if c >= 0.50 else "outflow fails"
        print(f"   coupling {c:<5} lag_info {ref[-1]['lag_info']:+.4f}   "
              f"ghost {ref[-1]['ghost']:+.4f}   ({works})")
    R = pd.DataFrame(ref)
    R.to_csv(OUT / "reference.csv", index=False)
    L30 = float(R[R.coupling == 0.30].lag_info.iloc[0])
    L50 = float(R[R.coupling == 0.50].lag_info.iloc[0])
    print(f"\n   L30 = {L30:+.4f} (floor)   L50 = {L50:+.4f} (pass mark)")

    # ---- candidate: the 16 clean chamber runs ----------------------------
    files = sorted(glob.glob(DATA))
    rows = []
    for f in files:
        d = pd.read_csv(f)
        x = d[SRC + SEN].to_numpy(float)
        rows.append({"file": Path(f).name, "n": len(x),
                     "lag_info": lag_info(x, len(SRC)),
                     "ghost": lag_info(x, len(SRC), ghost=True)})
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "chamber.csv", index=False)
    chamber = float(C.lag_info.median())
    ghost = float(C.ghost.median())
    print(f"\nCANDIDATE  {len(C)} actuators_random_walk runs "
          f"(one family, all sources vary)")
    print(f"   lag_info median {chamber:+.4f}   "
          f"[{C.lag_info.min():+.4f}, {C.lag_info.max():+.4f}]")
    print(f"   ghost    median {ghost:+.4f}   "
          f"[{C.ghost.min():+.4f}, {C.ghost.max():+.4f}]")

    print("\nVERDICT (rule fixed before running)")
    if max(abs(ghost), abs(R.ghost.abs().max())) > GHOST_BAR:
        print(f"   CHECK VOID: ghost exceeds {GHOST_BAR}")
        return 0
    print(f"   ghost clean on both systems (<= {GHOST_BAR})")
    if chamber >= L50:
        print(f"   {chamber:+.4f} >= L50 {L50:+.4f}  -> QUALIFIES. The data "
              "carries at least as much\n      lagged driver information as "
              "the weakest system where outflow works.\n      A clean re-run "
              "at b = 4V is worth its cost.")
    elif chamber <= L30:
        print(f"   {chamber:+.4f} <= L30 {L30:+.4f}  -> DISQUALIFIED. No more "
              "lagged information than\n      systems where the statistic is "
              "known to fail. No re-run; excluded on\n      evidence.")
    else:
        print(f"   L30 {L30:+.4f} < {chamber:+.4f} < L50 {L50:+.4f}  -> "
              "AMBIGUOUS, treated as\n      DISQUALIFIED for the purpose of "
              "spending compute, as declared.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
