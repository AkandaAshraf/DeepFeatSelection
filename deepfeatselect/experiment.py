"""Repeated training runs, aggregated into a feature ranking with uncertainty.

A single run's gates are noisy, so the original trained many models and averaged
them.  Two changes here:

* every run draws its own stratified split as well as its own weight
  initialisation, so the spread across runs reflects sampling variability rather
  than initialisation alone -- much closer to a stability-selection estimate;
* the ranking is reported with bootstrap confidence intervals and rank stability,
  so "feature A beats feature B" can be distinguished from noise.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .data import prepare
from .train import TrainConfig, configure_devices, train_one


def _run_one(
    csv_path: str,
    feature_names: list[str] | None,
    config: TrainConfig,
    seed: int,
    val_size: float,
    test_size: float,
    verbose: int,
) -> dict[str, float]:
    """Prepare data, train one model, and return a flat row of results.

    Kept at module level and taking only picklable arguments so it can be
    dispatched to a :class:`ProcessPoolExecutor` worker.
    """
    configure_devices()
    data = prepare(
        csv_path,
        feature_names=feature_names,
        task=config.task,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )
    result = train_one(data, config, seed=seed, verbose=verbose)
    return result.as_row(data.feature_names)


def run_experiment(
    csv_path: str | Path,
    feature_names: list[str] | None = None,
    config: TrainConfig | None = None,
    n_models: int = 20,
    val_size: float = 0.2,
    test_size: float = 0.2,
    workers: int = 1,
    seed0: int = 0,
    verbose: int = 0,
) -> pd.DataFrame:
    """Train ``n_models`` models and return one row of results per model."""
    config = config or TrainConfig()
    csv_path = str(csv_path)
    seeds = [seed0 + i for i in range(n_models)]
    args = [(csv_path, feature_names, config, s, val_size, test_size, verbose) for s in seeds]

    if workers <= 1:
        rows = []
        for i, a in enumerate(args, 1):
            rows.append(_run_one(*a))
            print(f"  model {i}/{n_models} done (test_f1={rows[-1]['test_f1']:.3f})")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_run_one, *zip(*args)))

    return pd.DataFrame(rows)


def _bootstrap_ci(
    values: np.ndarray, n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile bootstrap interval for the mean of each column of ``values``."""
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = values[idx].mean(axis=1)
    lo = np.percentile(means, 100 * alpha / 2, axis=0)
    hi = np.percentile(means, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


def summarise(runs: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Aggregate per-run gates into a ranked importance table.

    Each run's gates are normalised to sum to one before averaging, so a run that
    happened to settle on uniformly larger gates does not dominate.  Runs whose
    gates all collapsed to zero (an over-strong L1) are dropped rather than
    producing a division by zero.
    """
    gates = runs[feature_names].to_numpy(dtype=float)

    totals = gates.sum(axis=1, keepdims=True)
    usable = totals.reshape(-1) > 1e-12
    if not usable.any():
        raise ValueError(
            "every run collapsed to all-zero gates -- l1_gate is too strong for this data"
        )
    if not usable.all():
        print(f"warning: dropped {(~usable).sum()} run(s) whose gates collapsed to zero")

    shares = gates[usable] / totals[usable]
    lo, hi = _bootstrap_ci(shares)

    # Rank within each run, so stability is measured on the ordering itself
    # rather than on the magnitudes, which vary in scale between runs.
    ranks = (-shares).argsort(axis=1).argsort(axis=1) + 1

    summary = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": shares.mean(axis=0),
            "ci_low": lo,
            "ci_high": hi,
            "std": shares.std(axis=0, ddof=1) if len(shares) > 1 else 0.0,
            "mean_rank": ranks.mean(axis=0),
            "rank_std": ranks.std(axis=0, ddof=1) if len(shares) > 1 else 0.0,
            "selected_frac": (shares > 1e-4).mean(axis=0),
        }
    )
    return summary.sort_values("importance", ascending=False).reset_index(drop=True)


def report(runs: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Print held-out performance first, then the ranking."""
    metric_cols = [c for c in runs.columns if c.startswith("test_")]

    print("\nheld-out performance across runs")
    print("-" * 64)
    for col in metric_cols:
        v = runs[col]
        print(f"  {col:<20} mean={v.mean():.3f}  std={v.std(ddof=1) if len(v) > 1 else 0:.3f}"
              f"  min={v.min():.3f}  max={v.max():.3f}")
    print(f"  {'epochs_run':<20} mean={runs['epochs_run'].mean():.0f}")
    print(
        "\n  Importances are only as trustworthy as these scores. A run that did"
        "\n  not learn still produces a full set of gates."
    )

    print("\nfeature importance (higher is better, 95% bootstrap CI over runs)")
    print("-" * 64)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary.to_string(index=False))
