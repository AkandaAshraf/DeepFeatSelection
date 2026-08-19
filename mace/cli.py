"""Command line: mace-scan data.csv --out drivenness.csv

The CSV is (timepoints x channels) with an optional header row of channel
names. The gate report prints to stdout; scores go to --out.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .core import MaceConfig, scan


def main() -> int:
    ap = argparse.ArgumentParser(prog="mace-scan", description=__doc__)
    ap.add_argument("data", help="CSV, timepoints x channels")
    ap.add_argument("--out", default="drivenness.csv")
    ap.add_argument("--no-difference", action="store_true",
                    help="skip first-differencing (already-stationary data)")
    ap.add_argument("--tau", type=int, default=1)
    ap.add_argument("--degree", type=int, default=3, choices=(2, 3))
    args = ap.parse_args()

    frame = pd.read_csv(args.data)
    names = list(frame.columns)
    cfg = MaceConfig(tau=args.tau, degree=args.degree,
                     difference=not args.no_difference)
    result = scan(frame.to_numpy(np.float64), cfg, channel_names=names)
    print(result.summary())
    result.to_frame().to_csv(args.out, index=False)
    print(f"scores written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
