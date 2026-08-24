"""Screen the NASA PCoE randomized battery usage data against the fitness gate.

Pre-registration: paper/dataset_fitness_protocol.md, battery addendum,
committed before any lag_info was computed on this data.

The driver is the applied current: a randomized load profile, redrawn every
60 s from a declared distribution - exogenous by construction. The driven
variables are terminal voltage (RC polarisation at seconds, state-of-charge
integration over the whole history) and cell temperature (thermal mass at
minutes). Base grid 60 s - the excitation's own timescale; segments split at
recording voids > 300 s; the decimation grid then explores 1-50 minutes.

Linear only; no autoencoder is trained.

    python scripts/battery_screen.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).parent))
from dataset_fitness import lag_info  # noqa: E402
from chamber_screen import reference  # noqa: E402

OUT = Path("ExpOutput/battery_screen")
DATA = "Data/battery_rw/*/data/Matlab/*.mat"
DT = 60.0                  # base grid, seconds
VOID = 300.0               # split segments at recording voids longer than this
GRID = (1, 2, 5, 10, 20, 50)
MIN_N = 2000


def load_cell(path):
    """Concatenated (time, current, voltage, temperature) for one cell."""
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    steps = np.atleast_1d(m["data"].step)
    T = np.concatenate([np.atleast_1d(s.time) for s in steps])
    I = np.concatenate([np.atleast_1d(s.current) for s in steps])
    V = np.concatenate([np.atleast_1d(s.voltage) for s in steps])
    C = np.concatenate([np.atleast_1d(s.temperature) for s in steps])
    o = np.argsort(T, kind="stable")
    return T[o], np.column_stack([I[o], V[o], C[o]])


def segments(T, X):
    """Uniform-DT segments, split at voids > VOID, interpolated within."""
    brk = np.where(np.diff(T) > VOID)[0]
    lo = np.concatenate([[0], brk + 1])
    hi = np.concatenate([brk, [len(T) - 1]])
    out = []
    for a, b in zip(lo, hi):
        if T[b] - T[a] < MIN_N * DT / 4:      # skip clearly hopeless spans
            continue
        g = np.arange(T[a], T[b], DT)
        out.append(np.column_stack(
            [np.interp(g, T[a:b + 1], X[a:b + 1, j]) for j in range(3)]))
    return out


def screen_cells(cells, m):
    vals, ghosts, used, dropped = [], [], 0, 0
    for _, segs in cells:
        cv, cg = [], []
        for x in segs:
            xi = x[::m]
            if len(xi) < MIN_N or xi[:, 0].std() < 1e-9:
                continue
            cv.append(lag_info(xi, 1))
            cg.append(lag_info(xi, 1, ghost=True))
        if cv:
            vals.append(float(np.median(cv)))
            ghosts.append(float(np.median(cg)))
            used += 1
        else:
            dropped += 1
    if not vals:
        return None
    return {"m": m, "cells": used, "dropped": dropped,
            "lag_info": float(np.median(vals)),
            "lag_min": float(np.min(vals)), "lag_max": float(np.max(vals)),
            "ghost": float(np.median(ghosts))}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    L30, L50 = reference()
    print(f"REFERENCE recomputed: L30 = {L30:+.4f}   L50 = {L50:+.4f}\n")

    files = sorted(glob.glob(DATA))
    print(f"cells found: {len(files)}")
    cells = []
    for f in files:
        T, X = load_cell(f)
        segs = segments(T, X)
        ns = [len(s) for s in segs]
        cells.append((f, segs))
        print(f"  {Path(f).parts[-4][:44]:46s}{Path(f).name:10s} "
              f"segs {len(segs):>3}  pts {sum(ns):>7,}", flush=True)

    print("\nbattery_rw (source = applied current; driven = voltage, "
          "temperature; base grid 60 s)")
    rows = []
    for m in GRID:
        r = screen_cells(cells, m)
        if r is None:
            print(f"   m={m:<3} no cell retains {MIN_N} samples")
            continue
        rows.append(r)
        flag = ("QUALIFIES" if r["lag_info"] >= L50
                else "below L50" if r["lag_info"] > L30 else "below floor")
        print(f"   m={m:<3} cells {r['cells']:<3} lag_info {r['lag_info']:+.4f}"
              f"  [{r['lag_min']:+.4f}, {r['lag_max']:+.4f}]  "
              f"ghost {r['ghost']:+.4f}   {flag}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "screen.csv", index=False)

    print("\nVERDICT (rule fixed before running)")
    ok = d[d.lag_info >= L50]
    if ok.empty:
        print(f"   -> DISQUALIFIED: no decimation reaches L50 = {L50:+.4f}. "
              f"Best {d.lag_info.max():+.4f}.")
    else:
        r = ok.sort_values("m").iloc[0]
        print(f"   -> QUALIFIES at m={int(r.m)} (grid {DT*int(r.m):.0f}s): "
              f"lag_info {r.lag_info:+.4f}, ghost {r.ghost:+.4f}, "
              f"{int(r.cells)} cells.")
        print("      Next: the pre-registered replication test, marginal "
              "statistic primary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
