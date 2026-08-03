"""Compare the gated network's ranking with the classical baselines on Cleveland.

The gates are the point of this package, so the honest question is whether they
say anything a mutual-information filter or a random forest does not say for a
thousandth of the compute.  This script answers it three ways: the ranked
per-feature table under every method, the pairwise Spearman agreement between the
orderings, and where the top-5 sets overlap.

Two things it is careful about.

* Every baseline scores *columns* and Cleveland's four nominal attributes are
  one-hot encoded, so the raw vectors are length 22 while the gates are length
  13.  Everything is pushed through
  :func:`~deepfeatselect.baselines.aggregate_to_features` before it is compared.
* The baselines see the same rows the network's fitting procedure saw -- the
  training split plus the validation split it early-stopped on -- drawn from the
  same seeds, so a difference in the ranking is a difference in method rather
  than in data.  The test split is not shown to anything.

    python scripts/compare_baselines.py --n-models 20
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# TensorFlow reads both of these at import time and the package imports keras
# eagerly, so they have to be set before deepfeatselect comes in. CPU on purpose:
# the models are a few thousand parameters and GPU launch overhead dominates.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from deepfeatselect.baselines import (  # noqa: E402
    aggregate_to_features,
    all_baselines,
    rank_agreement,
)
from deepfeatselect.data import load_feature_names, prepare  # noqa: E402
from deepfeatselect.experiment import report, run_experiment, summarise  # noqa: E402
from deepfeatselect.train import config_from_namespace, configure_devices  # noqa: E402

GATE_METHOD = "gated_network"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rank Cleveland features with the gated network and with four "
        "classical baselines, then measure how much the rankings agree.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", default="Data/processed.cleveland.data")
    p.add_argument("--attributes", default=None, help="single-line CSV of feature names")
    p.add_argument("--outdir", default="ExpOutput")
    p.add_argument("--task", choices=("binary", "multiclass"), default="binary")
    p.add_argument("-n", "--n-models", type=int, default=20, help="gated models to train")
    p.add_argument("--workers", type=int, default=1, help="models to train concurrently")
    p.add_argument("--seed", type=int, default=0, help="seed of the first run")
    p.add_argument(
        "--baseline-seeds",
        type=int,
        default=5,
        help="splits to average each baseline over. Every method gets the same "
        "seed inside one call, so a single call is reproducible but not an "
        "independent estimate -- averaging over splits is what gives a spread",
    )
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--top-k", type=int, default=5, help="size of the top-k overlap sets")

    mdl = p.add_argument_group("model")
    mdl.add_argument("--l1-gate", type=float, default=1.0)
    mdl.add_argument("--learning-rate", type=float, default=3e-3)
    mdl.add_argument("--batch-size", type=int, default=32)
    mdl.add_argument("--epochs", type=int, default=2000)
    mdl.add_argument("--patience", type=int, default=40)
    return p


def gate_importance(
    args, raw_names: list[str] | None, feature_names: list[str]
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Train the gated models and return per-feature shares in ``feature_names`` order.

    ``raw_names`` is the CSV column order that :func:`~deepfeatselect.data.prepare`
    expects; ``feature_names`` is the order it emits, numeric block first.  The
    two are different lists whenever any attribute is nominal, and feeding the
    second one back in would silently relabel every column.

    :func:`~deepfeatselect.experiment.summarise` sorts by importance, so the
    vector has to be re-indexed before it can sit next to a baseline column.
    """
    config = config_from_namespace(args)
    runs = run_experiment(
        csv_path=args.data,
        feature_names=raw_names,
        config=config,
        n_models=args.n_models,
        val_size=args.val_size,
        test_size=args.test_size,
        workers=args.workers,
        seed0=args.seed,
    )
    summary = summarise(runs, feature_names)
    shares = summary.set_index("feature").loc[feature_names, "importance"].to_numpy()
    return shares, runs, summary


def baseline_importance(args, raw_names: list[str] | None) -> dict[str, np.ndarray]:
    """Run every baseline on each split and average the per-feature shares.

    The network's importance is an average over ``n_models`` splits, so a
    baseline read off a single split would be compared against a quantity with
    far less variance in it.  Each seed here reproduces one of the splits
    ``run_experiment`` drew, and the same rows are used: train plus validation.
    """
    totals: dict[str, np.ndarray] = {}
    for offset in range(args.baseline_seeds):
        seed = args.seed + offset
        data = prepare(
            args.data,
            feature_names=raw_names,
            task=args.task,
            val_size=args.val_size,
            test_size=args.test_size,
            seed=seed,
        )
        x = np.vstack([data.x_train, data.x_val])
        y = np.concatenate([data.y_train, data.y_val])

        for name, column_scores in all_baselines(x, y, seed=seed).items():
            per_feature = aggregate_to_features(column_scores, data.groups, data.n_features)
            totals[name] = totals.get(name, 0.0) + per_feature
        print(f"  baselines {offset + 1}/{args.baseline_seeds} done (seed {seed})")

    return {name: total / args.baseline_seeds for name, total in totals.items()}


def rank_frame(scores: dict[str, np.ndarray], feature_names: list[str]) -> pd.DataFrame:
    """Per-feature scores with a rank column per method, ranked by mean rank.

    ``method="min"`` so tied scores share a rank rather than being separated by
    the order the features happen to sit in.  Cleveland gates do hit exact zero,
    so the ties are real.
    """
    table = pd.DataFrame(scores, index=pd.Index(feature_names, name="feature"))
    ranks = table.rank(ascending=False, method="min").astype(int)
    ranks.columns = [f"{c}_rank" for c in ranks.columns]

    out = table.join(ranks)
    out.insert(0, "mean_rank", ranks.mean(axis=1))
    return out.sort_values("mean_rank")


def top_k_sets(scores: dict[str, np.ndarray], feature_names: list[str], k: int) -> dict[str, list[str]]:
    """The ``k`` highest-scoring features per method, best first."""
    names = np.asarray(feature_names)
    return {
        method: list(names[np.argsort(-np.asarray(vector), kind="stable")[:k]])
        for method, vector in scores.items()
    }


def overlap_matrix(top: dict[str, list[str]]) -> pd.DataFrame:
    """How many features each pair of methods shares in its top-k."""
    methods = list(top)
    counts = [[len(set(top[a]) & set(top[b])) for b in methods] for a in methods]
    return pd.DataFrame(counts, index=methods, columns=methods)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"compute devices: {', '.join(configure_devices())}")

    raw_names = load_feature_names(args.attributes) if args.attributes else None
    reference = prepare(
        args.data,
        feature_names=raw_names,
        task=args.task,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    feature_names = reference.feature_names
    print(
        f"{reference.n_features} features spanning {reference.n_columns} columns; "
        f"{len(reference.y_train)} train / {len(reference.y_val)} val / "
        f"{len(reference.y_test)} test rows"
    )

    print(f"\ntraining {args.n_models} gated model(s)")
    shares, runs, summary = gate_importance(args, raw_names, feature_names)
    report(runs, summary)

    print(f"\nbaselines over {args.baseline_seeds} split(s)")
    scores = {GATE_METHOD: shares}
    scores.update(baseline_importance(args, raw_names))

    table = rank_frame(scores, feature_names)
    print("\nper-feature importance, all methods (shares of each method's own total)")
    print("=" * 78)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(table.to_string())

    print("\npairwise Spearman rank agreement")
    print("=" * 78)
    with pd.option_context("display.float_format", "{:+.3f}".format, "display.width", 200):
        print(rank_agreement(scores).to_string())

    top = top_k_sets(scores, feature_names, args.top_k)
    print(f"\ntop-{args.top_k} by method (best first)")
    print("=" * 78)
    for method, members in top.items():
        print(f"  {method:<20} {', '.join(members)}")

    sets = [set(v) for v in top.values()]
    consensus = sorted(set.intersection(*sets))
    union = sorted(set.union(*sets))
    counts = pd.Series(
        {name: sum(name in s for s in sets) for name in union}, name="n_methods"
    ).sort_values(ascending=False)

    print(f"\n  in every method's top-{args.top_k}: {', '.join(consensus) or '(none)'}")
    print(f"  in at least one:               {', '.join(union)}")
    print(f"\n  how many of the {len(sets)} methods put each feature in its top-{args.top_k}")
    print(counts.to_string())

    print(f"\npairwise top-{args.top_k} overlap (out of {args.top_k})")
    print(overlap_matrix(top).to_string())

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(outdir / "baseline_comparison.csv")
    rank_agreement(scores).to_csv(outdir / "baseline_rank_agreement.csv")
    print(
        f"\nwrote {outdir / 'baseline_comparison.csv'} and "
        f"{outdir / 'baseline_rank_agreement.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
