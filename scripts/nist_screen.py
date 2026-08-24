"""Screen the NIST UR5 degradation dataset against the fitness gate.

Pre-registration: paper/dataset_fitness_protocol.md, NIST UR5 addendum,
committed before any lag_info was computed on this data.

The apparatus: a UR5 arm repeatedly runs a preprogrammed trajectory
(randomly-selected stop points) under varied speed, payload and temperature;
the controller logs TARGET (commanded) and ACTUAL signals at 125 Hz.

  SOURCES   the six TARGET_JOINT_POSITIONS - the preprogrammed trajectory,
            exogenous to the physical plant by construction
  DRIVEN    ACTUAL joint positions, velocities and currents, the control
            currents, the TCP pose and force, and the joint temperatures
  EXCLUDED  TARGET velocities / accelerations / currents / torques: these
            are deterministic functions of the same preprogrammed
            trajectory (interpolator derivatives and the controller's
            feedforward model), so they are neither independent sources to
            test nor physical consequences

Same decimation grid, same 2,000-sample floor, same calibrated pass mark
L50, ghost beside every cell. Linear only; no autoencoder is trained.

    python scripts/nist_screen.py
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from dataset_fitness import lag_info  # noqa: E402
from chamber_screen import reference  # noqa: E402

OUT = Path("ExpOutput/nist_screen")
DATA = "Data/nist_ur5/*.csv"
GRID = (1, 2, 5, 10, 20, 50)
MIN_N = 2000

# bracket-group layout, from UR5TestResult_header.xlsx
GROUPS = ["time", "target_pos", "actual_pos", "target_vel", "actual_vel",
          "target_cur", "actual_cur", "target_acc", "target_torq",
          "control_cur", "tcp_pose", "tcp_force", "joint_temp"]
SRC_GROUPS = ["target_pos"]
DRV_GROUPS = ["actual_pos", "actual_vel", "actual_cur", "control_cur",
              "tcp_pose", "tcp_force", "joint_temp"]


def load_run(path):
    """Parse the bracketed-tuple CSV into (n, 6*len(groups)) blocks."""
    rows = []
    with open(path) as fh:
        for line in fh:
            gs = re.findall(r"\[([^\]]*)\]", line)
            if len(gs) != len(GROUPS):
                continue
            rows.append([float(v) for g in gs[1:] for v in g.split(",")])
    x = np.asarray(rows, dtype=np.float64)
    names = [g for g in GROUPS[1:] for _ in range(6)]
    return x, np.asarray(names)


def screen(m):
    vals, ghosts, srcs, dropped, used = [], [], [], 0, 0
    for f in sorted(glob.glob(DATA)):
        x, names = load_run(f)
        x = x[::m]
        if len(x) < MIN_N:
            dropped += 1
            continue
        src_cols = np.where(np.isin(names, SRC_GROUPS))[0]
        drv_cols = np.where(np.isin(names, DRV_GROUPS))[0]
        # Rule 80 and its symmetric counterpart: only varying columns count
        src_cols = src_cols[x[:, src_cols].std(0) > 1e-9]
        drv_cols = drv_cols[x[:, drv_cols].std(0) > 1e-9]
        if len(src_cols) == 0 or len(drv_cols) == 0:
            dropped += 1
            continue
        xi = np.concatenate([x[:, src_cols], x[:, drv_cols]], axis=1)
        srcs.append(len(src_cols))
        vals.append(lag_info(xi, len(src_cols)))
        ghosts.append(lag_info(xi, len(src_cols), ghost=True))
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
    print(f"REFERENCE recomputed: L30 = {L30:+.4f}   L50 = {L50:+.4f}\n")
    print("nist_ur5 (18 runs, 125 Hz, sources = 6 target joint positions)")
    rows = []
    for m in GRID:
        r = screen(m)
        if r is None:
            print(f"   m={m:<3} no run retains {MIN_N} samples")
            continue
        rows.append(r)
        flag = ("QUALIFIES" if r["lag_info"] >= L50
                else "below L50" if r["lag_info"] > L30 else "below floor")
        print(f"   m={m:<3} runs {r['runs']:<3} src {r['n_sources']:<5}"
              f"lag_info {r['lag_info']:+.4f}  (max {r['lag_max']:+.4f})  "
              f"ghost {r['ghost']:+.4f}   {flag}")
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "screen.csv", index=False)

    print("\nVERDICT (rule fixed before running)")
    ok = d[d.lag_info >= L50]
    if ok.empty:
        print(f"   -> DISQUALIFIED: no decimation reaches L50 = {L50:+.4f}. "
              f"Best {d.lag_info.max():+.4f}.")
    else:
        r = ok.sort_values("m").iloc[0]
        print(f"   -> QUALIFIES at m={int(r.m)}: lag_info {r.lag_info:+.4f}, "
              f"ghost {r.ghost:+.4f}, {int(r.runs)} runs.")
        print("      Next step: the pre-registered replication test, marginal "
              "statistic primary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
