"""Sweep the gate L1 strength and print the resulting sparsity path.

Evidence that the penalty does what the README claims: as ``l1_gate`` rises the
gates should collapse onto a shrinking subset of features, and held-out AUC
should hold up until the penalty starts removing features that matter.

    python scripts/l1_sweep.py --n-models 5
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from deepfeatselect.experiment import run_experiment, summarise
from deepfeatselect.train import TrainConfig, configure_devices


def effective_features(shares: np.ndarray, threshold: float = 0.01) -> int:
    """Count features holding at least ``threshold`` of the total gate mass."""
    return int((shares >= threshold).sum())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="Data/processed.cleveland.data")
    p.add_argument("--n-models", type=int, default=5)
    p.add_argument("--workers", type=int, default=1, help="models to train concurrently")
    p.add_argument("--task", default="binary")
    # With proximal soft-thresholding a gate travels roughly `lr * l1` per step,
    # so on a dataset this small (a few hundred steps) the interesting range is
    # far higher than the values that would matter for a loss-based L1.
    p.add_argument(
        "--l1-values",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.3, 1.0, 3.0, 10.0],
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=3e-3)
    # A fixed epoch budget, not early stopping. Under early stopping each l1
    # trains for a different number of steps, and since a gate shrinks per step
    # the resulting "path" tracks when training happened to stop rather than the
    # penalty strength -- which shows up as a non-monotone path.
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=10**6)
    args = p.parse_args()

    print(f"compute devices: {', '.join(configure_devices())}")

    rows = []
    for l1 in args.l1_values:
        print(f"\n=== l1_gate={l1} ===")
        runs = run_experiment(
            csv_path=args.data,
            config=TrainConfig(
                task=args.task,
                l1_gate=l1,
                batch_size=args.batch_size,
                patience=args.patience,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
            ),
            n_models=args.n_models,
            workers=args.workers,
        )
        gate_cols = [
            c for c in runs.columns
            if c not in ("seed", "epochs_run") and not c.startswith("test_")
        ]
        try:
            summary = summarise(runs, gate_cols)
        except ValueError as exc:
            print(f"  {exc}")
            rows.append({"l1_gate": l1, "n_effective": 0, "test_auc": np.nan, "top3": "collapsed"})
            continue

        shares = summary["importance"].to_numpy()
        rows.append(
            {
                "l1_gate": l1,
                "n_effective": effective_features(shares),
                "test_auc": runs["test_auc"].mean() if "test_auc" in runs else np.nan,
                "test_f1": runs["test_f1"].mean(),
                "top3": ", ".join(summary["feature"].head(3)),
            }
        )

    print("\n\nsparsity path")
    print("=" * 78)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
