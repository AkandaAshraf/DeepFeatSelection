"""Screen chamber datasets against the fitness gate, across a decimation grid.

Pre-registration: paper/dataset_fitness_protocol.md and its 2026-08-23
addendum, committed before any lag_info was computed on the new data.

The pass mark is unchanged and calibrated: L50 = +0.0136, the value measured
on the weakest coupled-logistic system where outflow is known to work. A
dataset qualifies at the SMALLEST decimation clearing it. Every candidate at
every decimation is reported, whatever the numbers say.

Linear only. No autoencoder is trained and no outflow value is produced.

    python scripts/chamber_screen.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from dataset_fitness import lag_info  # noqa: E402
from source_outflow_coupling import coupled  # noqa: E402
from dataset_fitness import lag_info as _li  # noqa: E402,F401

OUT = Path("ExpOutput/chamber_screen")
SETTABLE = ["hatch", "pot_1", "pot_2", "load_in", "load_out"]
MEASURED = ["current_in", "current_out", "rpm_in", "rpm_out",
            "pressure_upwind", "pressure_downwind", "pressure_ambient",
            "pressure_intake", "mic", "signal_1", "signal_2"]
GRID = (1, 2, 5, 10, 20, 50)
MIN_N = 2000
CANDIDATES = {
    "wt_walks_v1": "Data/causalchamber/wt_walks_v1/*.csv",
    "wt_changepoints_v1": "Data/causalchamber/wt_changepoints_v1/**/*.csv",
    "wt_bernoulli_v1": "Data/causalchamber/wt_bernoulli_v1/**/*.csv",
    "wt_intake_impulse_v1":
        "Data/causalchamber/wt_intake_impulse_v1/**/*.csv",
}


def reference():
    """L30 and L50, recomputed with the identical code path."""
    out = {}
    for c in (0.30, 0.50):
        v = [lag_info(coupled(coupling=c, seed=s)[0][:, :9], 3)
             for s in (0, 1, 2)]
        out[c] = float(np.median(v))
    return out[0.30], out[0.50]


def screen(pattern, m):
    """Median lag_info and ghost over the runs of one dataset at decimation m."""
    vals, ghosts, srcs, dropped, used = [], [], [], 0, 0
    for f in sorted(glob.glob(pattern, recursive=True)):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if not set(SETTABLE + MEASURED).issubset(d.columns):
            continue
        x = d[SETTABLE + MEASURED].to_numpy(float)[::m]
        if len(x) < MIN_N:
            dropped += 1
            continue
        # Rule 80: a settable variable is a source only if it varies here
        sd = x[:, :len(SETTABLE)].std(0)
        live = np.where(sd > 1e-9)[0]
        if len(live) == 0:
            dropped += 1
            continue
        # a measured column constant in this run carries no target variance
        # and would dilute the median with structural zeros; excluded, as
        # constant settables are excluded from the source set (Rule 80)
        meas = np.arange(len(SETTABLE), x.shape[1])
        meas = meas[x[:, meas].std(0) > 1e-9]
        cols = np.concatenate([live, meas])
        xi = x[:, cols]
        srcs.append(len(live))
        vals.append(lag_info(xi, len(live)))
        ghosts.append(lag_info(xi, len(live), ghost=True))
        used += 1
    if not vals:
        return None
    return {"m": m, "runs": used, "dropped": dropped,
            "n_sources": f"{min(srcs)}-{max(srcs)}",
            "lag_info": float(np.median(vals)),
            "lag_max": float(np.max(vals)),
            "ghost": float(np.median(ghosts))}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    L30, L50 = reference()
    print(f"REFERENCE recomputed: L30 = {L30:+.4f}   L50 = {L50:+.4f} "
          f"(pass mark, unchanged)\n")

    rows = []
    for name, pat in CANDIDATES.items():
        if not glob.glob(pat, recursive=True):
            print(f"{name}: NOT PRESENT, skipped\n")
            continue
        print(f"{name}")
        for m in GRID:
            r = screen(pat, m)
            if r is None:
                print(f"   m={m:<3} no run retains {MIN_N} samples")
                continue
            r["dataset"] = name
            rows.append(r)
            flag = ("QUALIFIES" if r["lag_info"] >= L50
                    else "below L50" if r["lag_info"] > L30 else "below floor")
            print(f"   m={m:<3} runs {r['runs']:<3} src {r['n_sources']:<5}"
                  f"lag_info {r['lag_info']:+.4f}  (max {r['lag_max']:+.4f})  "
                  f"ghost {r['ghost']:+.4f}   {flag}")
        print()
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "screen.csv", index=False)

    print("VERDICT (rule fixed before running)")
    ok = d[d.lag_info >= L50]
    if ok.empty:
        print(f"   -> NO candidate reaches L50 = {L50:+.4f} at any decimation.")
        print("      Best observed: "
              f"{d.lag_info.max():+.4f} "
              f"({d.loc[d.lag_info.idxmax(), 'dataset']} at "
              f"m={int(d.loc[d.lag_info.idxmax(), 'm'])}).")
        print("      All four chamber datasets are DISQUALIFIED. The apparatus")
        print("      does not carry lagged actuator influence at any sampling")
        print("      rate reachable by decimating its logs.")
    else:
        best = ok.sort_values(["dataset", "m"]).groupby("dataset").first()
        print("   -> QUALIFIES, at the smallest clearing decimation:")
        for name, r in best.iterrows():
            print(f"      {name}  m={int(r.m)}  lag_info {r.lag_info:+.4f}  "
                  f"ghost {r.ghost:+.4f}  ({int(r.runs)} runs)")
        pick = best.lag_info.idxmax()
        print(f"      selected for the conditional-outflow test: {pick} "
              f"at m={int(best.loc[pick,'m'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
