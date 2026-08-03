"""Score importance and causality methods against known ground truth.

Every ranking on the Cleveland data is unfalsifiable -- nobody knows the true
parents of ``num`` -- so the only way to find out whether a method answers the
causal question is to run it on a system whose structure was written down first.
:mod:`deepfeatselect.synthetic` provides those systems; this script scores
against them.

Three sections, aimed at three separate claims.

(a) Cross-sectional, on :func:`~deepfeatselect.synthetic.nonlinear_scm`.  Six
    importance methods are ranked twice: once against the true direct causes and
    once against the true Markov blanket.  The blanket contains ``x_effect``, a
    noisy readout of the target with zero causal effect, and the prediction is
    that every method scores well against the blanket and badly against the
    causes.  A method that maximises predictive accuracy is *supposed* to rank
    ``x_effect`` first; the gap between the two columns is the point.

(b) Dynamical, on the coupled and uncoupled chaotic systems.  CCM has to recover
    the coupling direction, with a convergence statistic and a surrogate p-value,
    and has to stay silent on the negative controls.

(c) :func:`~deepfeatselect.synthetic.redundancy_demo`.  A genuine, unique driver
    with a perfect substitute in the table.  Removal-based importance -- refit
    LOCO, whether the model is a forest or the gated network -- has to report
    zero for it, because refitting recovers the signal from the substitute.  CCM
    is asked the same question and separates the driver from the distractor.

Runtimes are dominated by the networks, not the causality work: a full CCM sweep
plus a hundred surrogates on 3000 points takes under two seconds, while each
trained model costs ten to twenty.  ``--scm-models`` and ``--loco`` are the knobs
that matter.

    python scripts/benchmark_causal.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# TensorFlow reads both at import time and the package imports keras eagerly, so
# they have to be set before deepfeatselect comes in. CPU on purpose: these
# networks are a few thousand parameters and GPU launch overhead dominates.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from deepfeatselect import synthetic as syn  # noqa: E402
from deepfeatselect.baselines import aggregate_to_features, all_baselines  # noqa: E402
from deepfeatselect.ccm import ccm, optimal_embedding_dimension, surrogate_test  # noqa: E402
from deepfeatselect.data import Dataset  # noqa: E402
from deepfeatselect.probe import loco_importance  # noqa: E402
from deepfeatselect.train import TrainConfig, configure_devices, train_one  # noqa: E402

CROSS_SECTIONAL = "cross_sectional"
DYNAMICAL = "dynamical"
REDUNDANCY = "redundancy"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score feature-importance and causality methods against the "
        "known structure of the synthetic systems.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--outdir", default="ExpOutput")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--sections",
        nargs="+",
        default=[CROSS_SECTIONAL, DYNAMICAL, REDUNDANCY],
        choices=[CROSS_SECTIONAL, DYNAMICAL, REDUNDANCY],
    )

    a = p.add_argument_group("cross-sectional (a)")
    a.add_argument("--scm-n", type=int, default=1500, help="samples drawn from nonlinear_scm")
    a.add_argument("--scm-models", type=int, default=5, help="gated models to average")
    a.add_argument(
        "--no-loco",
        dest="loco",
        action="store_false",
        help="skip the retraining probes, which cost 1 + n_features models each",
    )

    b = p.add_argument_group("dynamical (b)")
    b.add_argument("--series-n", type=int, default=3000, help="length of each map series")
    b.add_argument(
        "--embedding",
        type=int,
        default=3,
        help="E for the logistic maps. The simplex-optimal E is reported alongside "
        "and is 1 for a one-dimensional map, which is degenerate for a cross map",
    )
    b.add_argument(
        "--rl-embedding",
        type=int,
        default=7,
        help="E for Rossler-Lorenz. Three is not enough to unfold a 3-D attractor "
        "plus its forcing and CCM reports the arrow backwards there; the same "
        "series is also run at --embedding so the sensitivity is visible",
    )
    b.add_argument("--n-bootstrap", type=int, default=50, help="library subsets per size")
    b.add_argument("--n-surrogates", type=int, default=100)
    b.add_argument(
        "--rl-points", type=int, default=1500, help="samples kept from the Rossler-Lorenz run"
    )
    b.add_argument(
        "--rl-stride",
        type=int,
        default=10,
        help="subsampling stride for Rossler-Lorenz. dt=0.01 oversamples both "
        "attractors badly, and on consecutive samples cross mapping scores high "
        "in both directions on autocorrelation alone",
    )
    b.add_argument(
        "--rl-exclusion",
        type=int,
        default=5,
        help="Theiler window for Rossler-Lorenz, in subsampled steps; belt and "
        "braces alongside the stride",
    )

    m = p.add_argument_group("network")
    m.add_argument("--hidden-units", type=int, default=64)
    m.add_argument("--n-hidden-layers", type=int, default=2)
    m.add_argument("--epochs", type=int, default=200)
    m.add_argument("--patience", type=int, default=20)
    m.add_argument("--batch-size", type=int, default=64)
    m.add_argument("--l1-gate", type=float, default=1.0)
    m.add_argument("--learning-rate", type=float, default=3e-3)
    return p


def _row(
    section: str,
    system: str,
    method: str,
    subject: str,
    metric: str,
    value: float = float("nan"),
    text: str = "",
) -> dict[str, object]:
    """One tidy record.  Every number this script produces goes through here."""
    return {
        "section": section,
        "system": system,
        "method": method,
        "subject": subject,
        "metric": metric,
        "value": float(value),
        "text": text,
    }


def _config(args) -> TrainConfig:
    """A deliberately small network.

    The shipped defaults (128 units, 3 layers, up to 2000 epochs) are tuned for
    297 Cleveland rows.  Here the probes train 1 + n_features models per system,
    so the architecture is cut down; the ranking is what is being measured, not
    the last point of AUC.
    """
    return TrainConfig(
        task="binary",
        l1_gate=args.l1_gate,
        hidden_units=args.hidden_units,
        n_hidden_layers=args.n_hidden_layers,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


def as_dataset(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    seed: int = 0,
    val_size: float = 0.2,
    test_size: float = 0.2,
) -> Dataset:
    """Split and standardise a synthetic design matrix the way ``prepare`` does.

    Scaling statistics come from the training split alone, and every synthetic
    feature owns exactly one column, so ``groups`` is the identity map.  A
    constant column would divide by zero; its scale is left at one, which keeps
    it constant rather than turning it into NaN and poisoning every layer.
    """
    x_fit, x_test, y_fit, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=seed
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_fit, y_fit, test_size=val_size / (1.0 - test_size), stratify=y_fit, random_state=seed
    )

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std > 0.0, std, 1.0)

    def scale(a: np.ndarray) -> np.ndarray:
        return (a - mean) / std

    return Dataset(
        x_train=scale(x_train),
        y_train=y_train,
        x_val=scale(x_val),
        y_val=y_val,
        x_test=scale(x_test),
        y_test=y_test,
        feature_names=list(feature_names),
        groups=np.arange(x.shape[1], dtype=np.int32),
        n_classes=int(y.max()) + 1,
    )


def gate_scores(x, y, feature_names, config, n_models: int, seed0: int) -> np.ndarray:
    """Mean per-run normalised gates, over models that each drew their own split.

    Redrawing the split per seed matches
    :func:`~deepfeatselect.experiment.run_experiment`: the spread across runs then
    reflects sampling variability as well as initialisation.  Runs whose gates
    collapsed to zero are dropped rather than divided by zero.
    """
    shares = []
    for i in range(n_models):
        data = as_dataset(x, y, feature_names, seed=seed0 + i)
        run = train_one(data, config, seed=seed0 + i)
        total = float(run.gates.sum())
        if total <= 1e-12:
            print(f"  gate model {i + 1}/{n_models}: gates collapsed to zero, dropped")
            continue
        shares.append(run.gates / total)
        print(
            f"  gate model {i + 1}/{n_models} done "
            f"(test_auc={run.metrics.get('test_auc', float('nan')):.3f}, "
            f"{int((run.gates == 0).sum())} gates exactly zero)"
        )
    if not shares:
        raise ValueError("every gated run collapsed to zero gates; lower --l1-gate")
    return np.mean(shares, axis=0)


def rank_quality(scores: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """AUROC and average precision at ranking the ``mask`` features above the rest.

    Both are computed over ``n_features`` points -- nine here -- so they move in
    coarse steps and a single swapped pair is visible.  That is a property of the
    question, not a defect: there are only nine features to order.
    """
    s = np.asarray(scores, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    return (
        float(roc_auc_score(m, s)),
        float(average_precision_score(m, s)),
    )


def cross_sectional(args) -> list[dict[str, object]]:
    """Section (a): rank the nonlinear SCM's features under every method."""
    scm = syn.nonlinear_scm(n=args.scm_n, seed=args.seed)
    names = scm.feature_names
    print(f"\nnonlinear_scm: n={scm.n_samples}, {scm.n_features} features, "
          f"positive rate {scm.y.mean():.3f}")
    print(f"  direct causes:  {', '.join(scm.names(scm.direct_causes))}")
    print(f"  markov blanket: {', '.join(scm.names(scm.markov_blanket))}")
    print(f"  confounded:     {', '.join(scm.names(scm.confounded))}")
    print(f"  irrelevant:     {', '.join(scm.names(scm.irrelevant))}")

    data = as_dataset(scm.x, scm.y, names, seed=args.seed)
    config = _config(args)

    scores: dict[str, np.ndarray] = {}

    print(f"\n  training {args.scm_models} gated model(s)")
    scores["gate"] = gate_scores(scm.x, scm.y, names, config, args.scm_models, args.seed)

    # The classical baselines see the training split only: the same rows the
    # network fitted on, and nothing from the test split.  groups is the identity
    # here, but the aggregation step is kept so the call site is the same one the
    # Cleveland comparison uses.
    for method, column_scores in all_baselines(data.x_train, data.y_train, seed=args.seed).items():
        scores[method] = aggregate_to_features(column_scores, data.groups, data.n_features)
    print("  baselines done")

    if args.loco:
        print(f"\n  LOCO probe: {1 + scm.n_features} models")
        table = loco_importance(data, config, seed=args.seed).set_index("feature")
        scores["loco_delta_auc"] = table.loc[names, "delta_auc"].to_numpy()
        # The other half of what the probe measures: how far the penultimate
        # representation moved. A feature with a perfect substitute costs no
        # accuracy and still forces the network to rebuild, which delta_auc
        # cannot see and CKA can.
        scores["loco_1_minus_cka"] = 1.0 - table.loc[names, "representation_cka"].to_numpy()

    rows: list[dict[str, object]] = []
    for method, vector in scores.items():
        for name, value in zip(names, vector):
            rows.append(_row(CROSS_SECTIONAL, "nonlinear_scm", method, name, "score", value))
        for target, mask in (
            ("direct_causes", scm.direct_causes),
            ("markov_blanket", scm.markov_blanket),
        ):
            auroc, ap = rank_quality(vector, mask)
            rows.append(_row(CROSS_SECTIONAL, "nonlinear_scm", method, target, "auroc", auroc))
            rows.append(
                _row(CROSS_SECTIONAL, "nonlinear_scm", method, target, "avg_precision", ap)
            )

    _report_cross_sectional(scores, scm)
    return rows


def _report_cross_sectional(scores: dict[str, np.ndarray], scm: syn.SyntheticDataset) -> None:
    names = scm.feature_names
    table = pd.DataFrame(scores, index=pd.Index(names, name="feature"))
    role = pd.Series("-", index=table.index, name="role")
    role[scm.confounded] = "confounded"
    role[scm.irrelevant] = "irrelevant"
    role[scm.direct_causes] = "cause"
    role[scm.effects] = "effect"

    print("\nper-feature scores (each method in its own units)")
    print("=" * 78)
    with pd.option_context("display.float_format", "{:+.4f}".format, "display.width", 200):
        print(pd.concat([role, table], axis=1).to_string())

    ranks = table.rank(ascending=False, method="min").astype(int)
    print("\nrank under each method (1 = most important)")
    with pd.option_context("display.width", 200):
        print(pd.concat([role, ranks], axis=1).to_string())

    quality = pd.DataFrame(
        {
            "auroc_direct_causes": {m: rank_quality(v, scm.direct_causes)[0] for m, v in scores.items()},
            "ap_direct_causes": {m: rank_quality(v, scm.direct_causes)[1] for m, v in scores.items()},
            "auroc_markov_blanket": {m: rank_quality(v, scm.markov_blanket)[0] for m, v in scores.items()},
            "ap_markov_blanket": {m: rank_quality(v, scm.markov_blanket)[1] for m, v in scores.items()},
        }
    )
    quality["blanket_minus_causes_auroc"] = (
        quality["auroc_markov_blanket"] - quality["auroc_direct_causes"]
    )

    base_causes = scm.direct_causes.mean()
    base_blanket = scm.markov_blanket.mean()
    print(
        f"\nranking quality (chance AUROC 0.5; chance AP {base_causes:.3f} for causes, "
        f"{base_blanket:.3f} for blanket)"
    )
    print("=" * 78)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(quality.to_string())

    effect = scm.names(scm.effects)[0]
    print(f"\nrank of {effect} (a child of the target, zero causal effect):")
    for method in table.columns:
        print(f"  {method:<20} rank {ranks.loc[effect, method]} of {len(names)}")


def _subsampled(data: syn.SystemData, stride: int) -> syn.SystemData:
    """Keep every ``stride``-th sample of both observed series."""
    out = dict(data)
    out["x"] = np.ascontiguousarray(data["x"][::stride])
    out["y"] = np.ascontiguousarray(data["y"][::stride])
    return out


def _dynamical_systems(args) -> list[tuple[str, syn.SystemData, str, int, int, str]]:
    """``(label, data, ground truth, E, Theiler window, note)`` for each system."""
    n = args.series_n
    cases: list[tuple[str, syn.SystemData, str, int, int, str]] = []

    driven = syn.coupled_logistic(n=n, seed=args.seed)
    cases.append(("coupled_logistic", driven, driven["true_direction"], args.embedding, 0, ""))

    # The documented recipe for the reversed case: swapping the growth rates
    # rather than only the couplings. Leaving r_y=3.5 in the driver's seat puts a
    # period-4 cycle there, which entrains x onto a cycle too and leaves both
    # series with four distinct values and no information for any method to find.
    reversed_case = syn.coupled_logistic(
        n=n, r_x=3.5, r_y=3.8, coupling_x_to_y=0.0, coupling_y_to_x=0.32, seed=args.seed
    )
    cases.append(
        (
            "coupled_logistic_reversed",
            reversed_case,
            reversed_case["true_direction"],
            args.embedding,
            0,
            "",
        )
    )

    rl = _subsampled(
        syn.rossler_lorenz(n=args.rl_points * args.rl_stride, coupling=2.0, seed=args.seed),
        args.rl_stride,
    )
    cases.append(
        ("rossler_lorenz", rl, rl["true_direction"], args.rl_embedding, args.rl_exclusion, "")
    )
    # The same series at the embedding the maps use. It is here because CCM gets
    # this system backwards at E=3 and the sensitivity belongs in the output
    # rather than in a tuned default: three delay coordinates cannot unfold a
    # 3-D response driven by a 3-D forcing, and at coupling=2.0 the forcing is
    # strong enough that the under-unfolded manifolds cross-map about equally
    # well in both directions.
    cases.append(
        (
            f"rossler_lorenz_E{args.embedding}",
            rl,
            rl["true_direction"],
            args.embedding,
            args.rl_exclusion,
            "same series as rossler_lorenz, run at the logistic-map embedding",
        )
    )

    rl_null = _subsampled(
        syn.rossler_lorenz(n=args.rl_points * args.rl_stride, coupling=0.0, seed=args.seed),
        args.rl_stride,
    )
    # rossler_lorenz hardcodes true_direction="x->y" whatever the coupling, but at
    # coupling=0 there is no arrow at all and its own docstring calls this the
    # continuous-time negative control. The ground truth used here is the one the
    # generating equations support, not the field.
    cases.append(
        (
            "rossler_lorenz_uncoupled",
            rl_null,
            syn.DIRECTION_NONE,
            args.rl_embedding,
            args.rl_exclusion,
            f"generator reports true_direction={rl_null['true_direction']!r}; "
            f"scored against 'none'",
        )
    )

    control = syn.independent_logistic(n=n, seed=args.seed)
    cases.append(
        ("independent_logistic", control, control["true_direction"], args.embedding, 0, "")
    )
    return cases


def dynamical(args) -> list[dict[str, object]]:
    """Section (b): CCM direction detection against the ground-truth coupling."""
    rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []

    for label, data, truth, embedding, exclusion, note in _dynamical_systems(args):
        x, y = np.asarray(data["x"]), np.asarray(data["y"])
        # Reported rather than used: the logistic map is a one-dimensional
        # deterministic system, so simplex projection picks E=1, which is a
        # degenerate embedding for a cross map. The sweep uses the fixed E.
        auto_e = optimal_embedding_dimension(x, max_E=8, exclusion_radius=exclusion)

        result = ccm(
            x,
            y,
            E=embedding,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            exclusion_radius=exclusion,
        )
        test = surrogate_test(
            x,
            y,
            E=embedding,
            n_surrogates=args.n_surrogates,
            seed=args.seed,
            exclusion_radius=exclusion,
        )
        dominant = result.dominant_direction()
        detected = syn.DIRECTION_NONE if dominant is None else f"{dominant.cause}->{dominant.effect}"

        print(
            f"\n{label}  (n={len(x)}, truth={truth}, E={embedding}, "
            f"Theiler={exclusion}, simplex-optimal E={auto_e})"
        )
        print("-" * 78)
        print(result.describe())
        print(test.describe())
        print(f"  detected {detected}  truth {truth}  {'MATCH' if detected == truth else 'MISS'}")
        if note:
            print(f"  note: {note}")

        entry: dict[str, object] = {
            "system": label,
            "n": len(x),
            "E": embedding,
            "truth": truth,
            "detected": detected,
            "match": detected == truth,
        }
        for direction, tested in (
            (result.x_causes_y, test.x_causes_y),
            (result.y_causes_x, test.y_causes_x),
        ):
            subject = f"{direction.cause}->{direction.effect}"
            lo, hi = direction.delta_rho_ci()
            values = {
                "rho_min_lib": direction.rho_at_min_lib,
                "rho_max_lib": direction.rho_at_max_lib,
                "delta_rho": direction.delta_rho,
                "delta_rho_ci_low": lo,
                "delta_rho_ci_high": hi,
                "is_convergent": float(direction.is_convergent()),
                "surrogate_rho": tested.rho,
                "surrogate_p": tested.p_value,
                "surrogate_null_mean": float(np.nanmean(tested.null_rho)),
            }
            for metric, value in values.items():
                rows.append(_row(DYNAMICAL, label, "ccm", subject, metric, value))
            entry[f"rho_{subject}"] = direction.rho_at_max_lib
            entry[f"delta_{subject}"] = direction.delta_rho
            entry[f"conv_{subject}"] = bool(direction.is_convergent())
            entry[f"p_{subject}"] = tested.p_value

        rows.append(
            _row(
                DYNAMICAL,
                label,
                "ccm",
                "verdict",
                "direction_match",
                float(detected == truth),
                f"detected={detected} truth={truth}" + (f"; {note}" if note else ""),
            )
        )
        rows.append(_row(DYNAMICAL, label, "ccm", "both", "embedding_dimension", embedding))
        rows.append(_row(DYNAMICAL, label, "simplex", "x", "optimal_embedding", auto_e))
        summary.append(entry)

    print("\ndirection detection versus ground truth")
    print("=" * 78)
    with pd.option_context("display.float_format", "{:+.3f}".format, "display.width", 250):
        print(pd.DataFrame(summary).to_string(index=False))
    return rows


def _rf_r2(x: np.ndarray, y: np.ndarray, seed: int, n_estimators: int = 200) -> float:
    """Held-out R^2 of a forest refitted on whichever columns it is given.

    Refit, not reuse: the question a removal-based importance asks is how much of
    the signal the *remaining* columns can recover, which needs a new fit.  A
    forest rather than the network because the point of this section is that the
    failure is a property of the joint distribution and not of one model class.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)
    forest = RandomForestRegressor(n_estimators=n_estimators, random_state=seed)
    forest.fit(x_train, y_train)
    return float(forest.score(x_test, y_test))


def redundancy(args) -> list[dict[str, object]]:
    """Section (c): removal importance against CCM on a deterministic system."""
    demo = syn.redundancy_demo(n=args.series_n, seed=args.seed)
    x = np.asarray(demo["x"])
    y = np.asarray(demo["y"])
    names: list[str] = list(demo["feature_names"])
    driver = str(demo["driver"])

    print(f"\nredundancy_demo: n={len(y)}, features {', '.join(names)}")
    print(f"  ground-truth driver: {driver}")
    print(f"  minimal sufficient sets: {demo['minimal_sufficient_sets']}")
    print(f"  irrelevant: {demo['irrelevant']}")

    rows: list[dict[str, object]] = []
    table = pd.DataFrame(index=pd.Index(names, name="feature"))

    full_r2 = _rf_r2(x, y, args.seed)
    print(f"\n  forest R^2 with every column: {full_r2:.5f}")
    drops = []
    for j, name in enumerate(names):
        kept = np.delete(x, j, axis=1)
        drops.append(full_r2 - _rf_r2(kept, y, args.seed))
    table["rf_loco_r2_drop"] = drops

    # The corrected claim from the module docstring: no single deletion costs
    # anything, and the smallest one that does is this pair, because a set
    # determines y exactly when it contains driver or proxy_cos.
    pair = ("driver", "proxy_cos")
    pair_idx = [names.index(n) for n in pair]
    pair_drop = full_r2 - _rf_r2(np.delete(x, pair_idx, axis=1), y, args.seed)
    print(f"  forest R^2 drop from deleting {pair}: {pair_drop:+.5f}")

    if args.loco:
        # The probes are classifiers, so the continuous target is split at its
        # median. The redundancy survives the split untouched: a set that
        # determines y determines any function of y, so driver and proxy_cos are
        # each still individually sufficient.
        y_binary = (y > np.median(y)).astype(np.int64)
        data = as_dataset(x, y_binary, names, seed=args.seed)
        print(f"\n  LOCO probe on the median-split target: {1 + len(names)} models")
        probe = loco_importance(data, _config(args), seed=args.seed).set_index("feature")
        table["net_loco_delta_auc"] = probe.loc[names, "delta_auc"]
        table["net_loco_cka"] = probe.loc[names, "representation_cka"]

    print("\n  CCM, each feature against the target")
    # rho at the smallest library is reported next to rho at the largest because
    # this system is deterministic and noiseless: a real driver is already
    # reconstructed almost perfectly from ten library points, which leaves
    # is_convergent's "skill must rise by min_delta" leg nothing to measure.
    # Convergence is a criterion for noisy systems; here the separation between a
    # driver and a distractor lives in rho itself.
    ccm_rho, ccm_rho_min, ccm_conv, ccm_p, ccm_reverse = [], [], [], [], []
    for j, name in enumerate(names):
        result = ccm(
            x[:, j], y, E=args.embedding, n_bootstrap=args.n_bootstrap, seed=args.seed
        )
        test = surrogate_test(
            x[:, j], y, E=args.embedding, n_surrogates=args.n_surrogates, seed=args.seed
        )
        forward = result.x_causes_y
        ccm_rho.append(forward.rho_at_max_lib)
        ccm_rho_min.append(forward.rho_at_min_lib)
        ccm_conv.append(forward.is_convergent())
        ccm_p.append(test.x_causes_y.p_value)
        ccm_reverse.append(result.y_causes_x.rho_at_max_lib)
        print(
            f"    {name:<10} rho(feature->target) {forward.rho_at_min_lib:+.4f} "
            f"(L={int(forward.lib_sizes[0])}) -> {forward.rho_at_max_lib:+.4f} "
            f"(L={int(forward.lib_sizes[-1])})  delta={forward.delta_rho:+.4f} "
            f"p={test.x_causes_y.p_value:.4f} convergent={forward.is_convergent()}"
        )
        for metric, value in (
            ("ccm_rho_feature_causes_target", forward.rho_at_max_lib),
            ("ccm_rho_at_min_lib", forward.rho_at_min_lib),
            ("ccm_delta_rho", forward.delta_rho),
            ("ccm_is_convergent", float(forward.is_convergent())),
            ("ccm_surrogate_p", test.x_causes_y.p_value),
            ("ccm_rho_target_causes_feature", result.y_causes_x.rho_at_max_lib),
        ):
            rows.append(_row(REDUNDANCY, "redundancy_demo", "ccm", name, metric, value))

    table["ccm_rho_min_lib"] = ccm_rho_min
    table["ccm_rho"] = ccm_rho
    table["ccm_convergent"] = ccm_conv
    table["ccm_p"] = ccm_p
    table["ccm_rho_reverse"] = ccm_reverse
    table["is_true_driver"] = [name == driver for name in names]

    # Booleans go out as 0/1 rather than being skipped, so the CSV carries the
    # whole printed table and not just its float columns.
    for column in table.columns:
        for name, value in table[column].items():
            rows.append(_row(REDUNDANCY, "redundancy_demo", "table", str(name), column, float(value)))
    rows.append(
        _row(REDUNDANCY, "redundancy_demo", "rf_refit", "all_columns", "r2", full_r2)
    )
    rows.append(
        _row(
            REDUNDANCY,
            "redundancy_demo",
            "rf_refit",
            "+".join(pair),
            "r2_drop_pair_deletion",
            pair_drop,
            "smallest deletion that costs anything",
        )
    )

    print("\nremoval-based importance versus CCM")
    print("=" * 78)
    with pd.option_context("display.float_format", "{:+.5f}".format, "display.width", 200):
        print(table.to_string())
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"compute devices: {', '.join(configure_devices())}")

    rows: list[dict[str, object]] = []
    if CROSS_SECTIONAL in args.sections:
        rows += cross_sectional(args)
    if DYNAMICAL in args.sections:
        rows += dynamical(args)
    if REDUNDANCY in args.sections:
        rows += redundancy(args)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "causal_benchmark.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nwrote {path} ({len(rows)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
