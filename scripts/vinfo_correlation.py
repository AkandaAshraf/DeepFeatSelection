"""What quantity do the internal signatures actually track?

The paper claims they track conditional V-information.  That claim deserves a
direct test rather than an argument, and there is a specific reason to doubt the
*conditional* form: under redundancy the conditional quantity
``R_F(N \\ {j}) - R_F(N)`` is close to zero for every informative feature, which
cannot order anything.  The ordering the probe produces has to come from
somewhere else.

Three candidate quantities are estimated here, all with the *same* network class
the probe uses, since V-information is defined relative to a class:

* ``marginal``    -- how well the class predicts the target from feature j alone.
* ``conditional`` -- the full-model loss minus the leave-j-out loss.
* ``shapley``     -- the average marginal contribution of j over coalitions,
  which interpolates between the two.

Each is rank-correlated against the probe's internal signature across several
systems.  Whichever correlates is the quantity the probe measures.

    python scripts/vinfo_correlation.py
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_internals import build_dataset  # noqa: E402

from deepfeatselect.data import Dataset  # noqa: E402
from deepfeatselect.netstats import train_and_measure  # noqa: E402
from deepfeatselect.probe import ablate_feature  # noqa: E402
from deepfeatselect.synthetic import manifold_redundancy, parity_redundancy  # noqa: E402
from deepfeatselect.train import TrainConfig  # noqa: E402


def _config(args, width=None) -> TrainConfig:
    return TrainConfig(
        task="binary", l1_gate=0.0, dropout=0.0, noise=0.0,
        hidden_units=width or args.hidden_units, n_hidden_layers=2,
        learning_rate=3e-3, epochs=args.epochs, batch_size=128,
        hierarchy=False, class_weight=False,
    )


def _keep_only(data: Dataset, keep: set[int]) -> Dataset:
    """Zero every column outside ``keep``, holding the architecture fixed."""
    reduced = data
    for j in range(data.n_features):
        if j not in keep:
            reduced = ablate_feature(reduced, j)
    return reduced


def _loss(data: Dataset, config: TrainConfig, seed: int) -> float:
    return train_and_measure(data, config, seed=seed).metrics["val_loss_final"]


def value_function(data, config, seeds):
    """``v(S) = -loss`` with only ``S`` visible, averaged over seeds and cached."""
    cache: dict[frozenset[int], float] = {}

    def v(subset: frozenset[int]) -> float:
        if subset not in cache:
            reduced = _keep_only(data, set(subset))
            cache[subset] = -np.mean([_loss(reduced, config, s) for s in range(seeds)])
        return cache[subset]

    return v


def vinfo_variants(data, names, config, seeds, shapley_max_features=5):
    """Marginal, conditional and (when affordable) Shapley V-information."""
    d = len(names)
    v = value_function(data, config, seeds)
    everything = frozenset(range(d))
    empty = frozenset()

    rows = []
    for j in range(d):
        rows.append({
            "feature": names[j],
            "marginal": v(frozenset({j})) - v(empty),
            "conditional": v(everything) - v(everything - {j}),
        })
    table = pd.DataFrame(rows)

    if d <= shapley_max_features:
        phi = np.zeros(d)
        for j in range(d):
            others = [i for i in range(d) if i != j]
            for size in range(len(others) + 1):
                weight = factorial(size) * factorial(d - size - 1) / factorial(d)
                for subset in combinations(others, size):
                    s = frozenset(subset)
                    phi[j] += weight * (v(s | {j}) - v(s))
        table["shapley"] = phi
    return table


def internal_signature(data, names, config, seeds):
    """Sum of |z| over all functionals, the probe's aggregate detection strength."""
    rows = []
    for seed in range(seeds):
        for arm in ["full"] + names:
            arm_data = data if arm == "full" else ablate_feature(data, names.index(arm))
            rows.append({"arm": arm, "seed": seed,
                         **train_and_measure(arm_data, config, seed=seed).metrics})
    df = pd.DataFrame(rows)
    metrics = [c for c in df.columns if c not in ("arm", "seed")]
    full = df[df.arm == "full"].set_index("seed")

    deltas = []
    for arm in names:
        sub = df[df.arm == arm].set_index("seed")
        for seed in sub.index:
            deltas.append({"arm": arm,
                           **{m: sub.loc[seed, m] - full.loc[seed, m] for m in metrics}})
    dd = pd.DataFrame(deltas)

    nulls = [n for n in names if n.startswith("null_") or n in ("unrelated",)]
    null_d = dd[dd.arm.isin(nulls)] if nulls else dd

    totals = {arm: 0.0 for arm in names}
    for m in metrics:
        sd = null_d[m].std(ddof=1)
        if sd == 0 or not np.isfinite(sd):
            continue
        mu = null_d[m].mean()
        for arm in names:
            totals[arm] += abs((dd[dd.arm == arm][m].mean() - mu) / sd)
    return pd.Series(totals, name="internal_sum_abs_z")


def analyse(data, names, title, args):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    config = _config(args)

    table = vinfo_variants(data, names, config, args.seeds)
    table = table.set_index("feature").join(internal_signature(data, names, config, args.seeds))

    with pd.option_context("display.float_format", "{:+.4f}".format):
        print(table.to_string())

    print("\n  Spearman rank correlation with the internal signature:")
    results = {}
    for column in ("marginal", "conditional", "shapley"):
        if column not in table:
            continue
        rho, p = spearmanr(table[column], table["internal_sum_abs_z"])
        results[column] = rho
        print(f"    {column:<12} rho = {rho:+.3f}   p = {p:.3f}")
    return table, results


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--hidden-units", type=int, default=16)
    p.add_argument("--outdir", default="ExpOutput/vinfo")
    args = p.parse_args()

    all_results = {}

    for label in ("successor_median", "driver_threshold"):
        data, names = build_dataset(args.n, 1, seed=0, label=label)
        _, res = analyse(data, names, f"redundancy_demo [{label}]", args)
        all_results[f"redundancy_demo:{label}"] = res

    # The other two families, wrapped into the same Dataset shape.
    for builder, title in [(parity_redundancy, "parity_redundancy"),
                           (manifold_redundancy, "manifold_redundancy")]:
        system = builder(n=args.n, seed=0)
        x = np.asarray(system["x"], dtype=np.float64)
        y = np.asarray(system["y"]).astype(np.int64)
        a, b = int(0.6 * len(x)), int(0.8 * len(x))
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(x[:a])
        data = Dataset(
            x_train=scaler.transform(x[:a]), y_train=y[:a],
            x_val=scaler.transform(x[a:b]), y_val=y[a:b],
            x_test=scaler.transform(x[b:]), y_test=y[b:],
            feature_names=list(system["feature_names"]),
            groups=np.arange(len(system["feature_names"]), dtype=np.int32),
            n_classes=2,
        )
        _, res = analyse(data, data.feature_names, title, args)
        all_results[title] = res

    print("\n" + "=" * 78)
    print("WHICH QUANTITY DOES THE PROBE TRACK?")
    print("=" * 78)
    summary = pd.DataFrame(all_results).T
    with pd.option_context("display.float_format", "{:+.3f}".format):
        print(summary.to_string())
    print("\n  mean rank correlation across systems:")
    for column in summary.columns:
        print(f"    {column:<12} {summary[column].mean():+.3f}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / "vinfo_correlation.csv")
    print(f"\nwrote {outdir}/vinfo_correlation.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
