"""Screen the PRONTO multiphase flow facility against the fitness gate.

Pre-registration: paper/dataset_fitness_protocol.md, PRONTO addendum,
committed before any lag_info was computed on this data.

Cranfield 2" multiphase flow rig (Stief et al., J. Process Control 79, 2019;
Zenodo 1341583, CC-BY). DeltaV logs every process variable at 1 Hz
continuously for a whole test day, so setpoint changes and their transients
are both in the record.

  SOURCES   the four flow-controller SET POINTS (FIC302/301/102/101 SP.CV):
            operator-commanded, exogenous to the rig's physics
  DRIVEN    the measured process variables - flow transmitters, pressures,
            temperatures, density, separator levels and the controllers'
            PROCESS VALUES (PV.CV)
  EXCLUDED  the controllers' OUTPUT values (FIC*/PID1/OUT.CV, PIC501, LVC502
            valve openings). These are feedback-computed from SP and PV: not
            exogenous, not pure physical consequences - the same class
            excluded as the UR5's target velocities. Note that "/OUT.CV" on
            a bare transmitter tag (FT305/OUT.CV) is DeltaV's name for the
            sensor reading and IS a measurement; only "/PID1/OUT.CV" is a
            controller output.

    python scripts/pronto_screen.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from dataset_fitness import lag_info  # noqa: E402
from chamber_screen import reference  # noqa: E402

OUT = Path("ExpOutput/pronto_screen")
DATA = "Data/pronto/*Testday*.csv"
GRID = (1, 2, 5, 10)          # Rule 98/99 arithmetic, computed in advance
MIN_N = 2000


def load_day(path):
    d = pd.read_csv(path, header=1, skiprows=[2], low_memory=False)
    d.columns = [str(c).strip() for c in d.columns]
    num = d.drop(columns=d.columns[0]).apply(pd.to_numeric, errors="coerce")
    num = num.ffill().bfill()
    src = [c for c in num.columns if c.endswith("SP.CV")]
    drv = [c for c in num.columns
           if not c.endswith("SP.CV") and "/PID1/OUT.CV" not in c]
    return num[src].to_numpy(float), num[drv].to_numpy(float), src, drv


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    L30, L50 = reference()
    print(f"REFERENCE recomputed: L30 = {L30:+.4f}   L50 = {L50:+.4f}\n")

    days = []
    for f in sorted(glob.glob(DATA)):
        S, D, sn, dn = load_day(f)
        days.append((Path(f).name, S, D, sn, dn))
        live = (S.std(0) > 1e-9).sum()
        print(f"  {Path(f).name:22s} rows {len(S):>6,}  sources varying "
              f"{live}/{len(sn)}  driven {len(dn)}")

    print("\npronto (sources = flow setpoints; driven = measured process "
          "variables)")
    rows = []
    for m in GRID:
        vals, ghosts, used, dropped = [], [], 0, 0
        for name, S, D, sn, dn in days:
            s, d_ = S[::m], D[::m]
            if len(s) < MIN_N:
                dropped += 1
                continue
            live = np.where(s.std(0) > 1e-9)[0]
            dlive = np.where(d_.std(0) > 1e-9)[0]
            if len(live) == 0 or len(dlive) == 0:
                dropped += 1
                continue
            x = np.concatenate([s[:, live], d_[:, dlive]], axis=1)
            vals.append(lag_info(x, len(live)))
            ghosts.append(lag_info(x, len(live), ghost=True))
            used += 1
        if not vals:
            print(f"   m={m:<3} no test-day retains {MIN_N} samples")
            continue
        r = {"m": m, "days": used, "dropped": dropped,
             "lag_info": float(np.median(vals)),
             "lag_max": float(np.max(vals)),
             "ghost": float(np.median(ghosts))}
        rows.append(r)
        flag = ("QUALIFIES" if r["lag_info"] >= L50
                else "below L50" if r["lag_info"] > L30 else "below floor")
        print(f"   m={m:<3} days {r['days']}  lag_info {r['lag_info']:+.4f}  "
              f"(max {r['lag_max']:+.4f})  ghost {r['ghost']:+.4f}   {flag}",
              flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "screen.csv", index=False)

    print("\nVERDICT (rule fixed before running)")
    ok = d[d.lag_info >= L50]
    if ok.empty:
        print(f"   -> DISQUALIFIED: no reachable decimation reaches "
              f"L50 = {L50:+.4f}. Best {d.lag_info.max():+.4f}.")
    else:
        r = ok.sort_values("m").iloc[0]
        print(f"   -> QUALIFIES at m={int(r.m)}: lag_info {r.lag_info:+.4f}, "
              f"ghost {r.ghost:+.4f}, {int(r.days)} test-days.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
