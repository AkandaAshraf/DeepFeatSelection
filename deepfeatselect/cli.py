"""Command line entry point.

Replaces the original's blocking ``input()`` prompt, which made the experiment
impossible to script, schedule or sweep.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_feature_names
from .experiment import report, run_experiment, summarise
from .train import config_from_namespace, configure_devices


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deepfeatselect",
        description="Rank features by training gated neural networks and reading the gates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io = p.add_argument_group("data")
    io.add_argument("--data", default="Data/processed.cleveland.data", help="input CSV")
    io.add_argument("--attributes", default=None, help="single-line CSV of feature names")
    io.add_argument("--outdir", default="ExpOutput", help="directory for result CSVs")
    io.add_argument(
        "--task",
        choices=("binary", "multiclass"),
        default="binary",
        help="binary collapses the 0-4 severity target to absent/present",
    )
    io.add_argument("--val-size", type=float, default=0.2)
    io.add_argument("--test-size", type=float, default=0.2)

    exp = p.add_argument_group("experiment")
    exp.add_argument("-n", "--n-models", type=int, default=20, help="models to train")
    exp.add_argument("--workers", type=int, default=1, help="models to train concurrently")
    exp.add_argument("--seed", type=int, default=0, help="seed of the first run")

    mdl = p.add_argument_group("model")
    mdl.add_argument(
        "--l1-gate",
        type=float,
        default=1.0,
        help="per-step gate shrinkage (proximal L1); 0 reproduces the original "
        "unregularised model. Larger than usual L1 values by design -- see README",
    )
    mdl.add_argument("--l2-dense", type=float, default=1e-3, help="weight decay on dense layers")
    mdl.add_argument("--hidden-units", type=int, default=128)
    mdl.add_argument("--n-hidden-layers", type=int, default=3)
    mdl.add_argument("--dropout", type=float, default=0.5)
    mdl.add_argument("--noise", type=float, default=0.005)
    mdl.add_argument("--learning-rate", type=float, default=1e-3)
    mdl.add_argument("--loss", choices=("ce", "soft_f1"), default="ce")
    mdl.add_argument("--epochs", type=int, default=2000)
    mdl.add_argument("--batch-size", type=int, default=64)
    mdl.add_argument("--patience", type=int, default=25)
    mdl.add_argument("--no-class-weight", dest="class_weight", action="store_false")
    mdl.add_argument(
        "--no-proximal",
        dest="proximal",
        action="store_false",
        help="enforce L1 through the loss instead of soft-thresholding; shrinks "
        "gates but does not drive them to exactly zero",
    )
    mdl.add_argument(
        "--no-hierarchy",
        dest="hierarchy",
        action="store_false",
        help="drop the constraint tying first-layer weights to their gate; without "
        "it the network compensates for a shrinking gate and nothing is selected",
    )
    mdl.add_argument(
        "--hierarchy-m",
        type=float,
        default=1.0,
        help="how much first-layer weight one unit of gate buys; only binds when "
        "comparable to the weight scale (Glorot init is around 0.2 here)",
    )

    p.add_argument("-v", "--verbose", action="count", default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    devices = configure_devices()
    print(f"compute devices: {', '.join(devices)}")

    feature_names = load_feature_names(args.attributes) if args.attributes else None
    config = config_from_namespace(args)

    print(f"training {args.n_models} model(s), task={config.task}, l1_gate={config.l1_gate}")
    runs = run_experiment(
        csv_path=args.data,
        feature_names=feature_names,
        config=config,
        n_models=args.n_models,
        val_size=args.val_size,
        test_size=args.test_size,
        workers=args.workers,
        seed0=args.seed,
        verbose=args.verbose,
    )

    gate_cols = [c for c in runs.columns if c not in ("seed", "epochs_run")]
    gate_cols = [c for c in gate_cols if not c.startswith("test_")]
    summary = summarise(runs, gate_cols)
    report(runs, summary)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(outdir / "runs.csv", index=False)
    summary.to_csv(outdir / "importance.csv", index=False)
    print(f"\nwrote {outdir / 'runs.csv'} and {outdir / 'importance.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
