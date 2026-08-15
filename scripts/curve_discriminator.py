"""Which feature of a training curve, if any, recovers causal role?

Borrowed from simplex projection, where an embedding is judged by whether
prediction skill *decays smoothly* with horizon rather than by its peak value:
the shape of the curve diagnoses the setup, and the endpoint alone does not.

Applied here, the question is whether some property of the loss trajectory --
its slope, its curvature, when the descent happens, how smooth it is -- carries
information that the final loss does not, and specifically whether any of them
separates a cause from an effect.

That comparison is the hard one and the reason for this experiment. On
``nonlinear_scm`` every importance measure tried so far ranks ``x_effect``
first. It is a *child* of the target, so it is the most predictive column in
the table and the least useful one to intervene on. If a curve statistic ranks
the true causes above it, that is a signal no endpoint measure has produced.

Nothing is assumed about which statistic wins. Fourteen candidates are computed
per run and each is scored against ground truth, including the final loss as the
incumbent baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from deepfeatselect.data import Dataset
from deepfeatselect.model import build_model
from deepfeatselect.synthetic import nonlinear_scm, redundancy_demo


# --------------------------------------------------------------------------
# curve statistics

def curve_features(curve: np.ndarray) -> dict[str, float]:
    """Fourteen summaries of one validation-loss trajectory.

    Grouped by what they could plausibly capture: where the curve ends, how
    fast it got there, when the learning happened, and how cleanly it descended.
    Several are near-duplicates by construction; that is intentional, since the
    point is to find which formulation discriminates rather than to pick one in
    advance.
    """
    n = len(curve)
    total_drop = curve[0] - curve[-1]
    diffs = np.diff(curve)
    third = max(1, n // 3)
    fifth = max(1, n // 5)

    # Where the descent has got to, as a fraction of its eventual total.
    progress = (curve[0] - curve) / (total_drop + 1e-12)

    def first_epoch_reaching(fraction: float) -> float:
        hit = np.flatnonzero(progress >= fraction)
        return float(hit[0]) / n if hit.size else 1.0

    return {
        # endpoint
        "final_loss": float(curve[-5:].mean()),
        "total_drop": float(total_drop),
        "area": float(curve.mean()),
        # rates
        "slope_early": float((curve[third] - curve[0]) / third),
        "slope_mid": float((curve[2 * third] - curve[third]) / third),
        "slope_late": float((curve[-1] - curve[-fifth]) / fifth),
        "slope_max": float(diffs.min()),
        # timing: when the learning actually happened
        "epoch_of_max_slope": float(np.argmin(diffs)) / n,
        "t_half": first_epoch_reaching(0.5),
        "t_ninety": first_epoch_reaching(0.9),
        # shape and quality
        "monotone_frac": float((diffs < 0).mean()),
        "smoothness": float(np.abs(diffs).sum() / (total_drop + 1e-12)),
        "curvature": float(np.abs(np.diff(diffs)).mean()),
        "plateau_frac": float((np.abs(curve - curve[-1]) < 0.02 * total_drop).mean()),
    }


# --------------------------------------------------------------------------
# training that returns the whole trajectory

def train_curve(data: Dataset, seed: int, epochs: int, width: int) -> np.ndarray:
    """Train once at a fixed budget and return the validation-loss curve.

    Fixed budget rather than early stopping, so every arm is measured over the
    same window and the curve shapes stay comparable. Deliberately small and
    noise-free: dropout would smear the trajectory with its own randomness,
    which is the thing being measured.
    """
    keras.utils.set_random_seed(seed)
    model = build_model(
        n_columns=data.n_columns, groups=data.groups, n_classes=2, task="binary",
        l1_gate=0.0, dropout=0.0, noise=0.0, hidden_units=width,
        n_hidden_layers=2, learning_rate=3e-3,
    )
    history = model.fit(
        data.x_train, data.y_train.astype("float32").reshape(-1, 1),
        validation_data=(data.x_val, data.y_val.astype("float32").reshape(-1, 1)),
        epochs=epochs, batch_size=128, shuffle=True, verbose=0,
    )
    return np.asarray(history.history["val_loss"], dtype=np.float64)


def ablate(data: Dataset, index: int) -> Dataset:
    """Zero one feature's column, holding the architecture fixed."""
    from dataclasses import replace
    def blank(x):
        out = x.copy()
        out[:, index] = 0.0
        return out
    return replace(data, x_train=blank(data.x_train), x_val=blank(data.x_val),
                   x_test=blank(data.x_test))


def as_dataset(x: np.ndarray, y: np.ndarray, names: list[str]) -> Dataset:
    a, b = int(0.6 * len(x)), int(0.8 * len(x))
    scaler = StandardScaler().fit(x[:a])
    return Dataset(
        x_train=scaler.transform(x[:a]), y_train=y[:a],
        x_val=scaler.transform(x[a:b]), y_val=y[a:b],
        x_test=scaler.transform(x[b:]), y_test=y[b:],
        feature_names=names, groups=np.arange(len(names), dtype=np.int32), n_classes=2,
    )


# --------------------------------------------------------------------------

def run_system(name: str, builder, args) -> pd.DataFrame:
    """Ablation sweep over S independent *draws of the system*, not just seeds.

    Resampling the initialisation alone would leave the role counts fixed -- three
    causes against one effect on nonlinear_scm -- and an AUROC over 3 versus 1
    takes four possible values, which is not a measurement. Redrawing the system
    gives 3S against S and makes the comparison meaningful.
    """
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    rows = []
    for draw in range(args.draws):
        data, roles = builder(draw)
        base = curve_features(train_curve(data, draw, args.epochs, args.width))
        for j, feature in enumerate(data.feature_names):
            stats = curve_features(
                train_curve(ablate(data, j), draw, args.epochs, args.width))
            row = {"feature": feature, "role": roles.get(feature, "?"), "draw": draw}
            # Paired against the full model from the same draw and seed.
            row.update({k: stats[k] - base[k] for k in stats})
            rows.append(row)
        print(f"  draw {draw + 1}/{args.draws} done")
    return pd.DataFrame(rows)


def score(frame: pd.DataFrame, stats: list[str], positive: set[str],
          negative: set[str], label: str) -> pd.DataFrame:
    """AUROC and PR-AUC per statistic, pooled over draws.

    AUROC leads because the three comparisons have different class balances and
    PR-AUC's baseline is the positive rate, so its values are not comparable
    across them. PR-AUC is carried for completeness.
    """
    sub = frame[frame.role.isin(positive | negative)]
    # One observation per (draw, feature): pooling over draws is what supplies
    # the sample size.
    obs = sub.groupby(["draw", "feature", "role"])[stats].mean().reset_index()
    y = obs.role.isin(positive).astype(int)
    if y.nunique() < 2:
        return pd.DataFrame()

    out = []
    for s in stats:
        v = obs[s].to_numpy()
        if np.allclose(v, v[0]):
            continue
        auc = roc_auc_score(y, v)
        # Direction is not known in advance, so distance from chance is the
        # informative quantity; orient PR-AUC to match whichever way it fell.
        oriented = v if auc >= 0.5 else -v
        out.append({
            "statistic": s,
            label: max(auc, 1 - auc),
            f"{label}_raw": auc,
            f"{label}_prauc": average_precision_score(y, oriented),
        })
    return pd.DataFrame(out).set_index("statistic")


def orientations(frame: pd.DataFrame, stats: list[str]) -> dict[str, int]:
    """Fix each statistic's sign on a contrast independent of cause-versus-effect.

    A statistic has no a-priori direction: for some, ablating an important
    feature raises the value, for others it lowers it. That sign must be settled
    *before* the metric of interest is computed, and it must not be settled by
    the metric of interest -- choosing the orientation that minimises
    P(effect first) would be fitting a free parameter to the reported number,
    and on a handful of draws it reaches zero by chance alone.

    The sign is therefore taken from the cause-versus-irrelevant contrast, which
    only assumes that ablating a cause disturbs training more than ablating
    noise, and says nothing about how causes compare with effects.
    """
    sub = frame[frame.role.isin({"cause", "irrelevant"})]
    means = sub.groupby("role")[stats].mean()
    signs = {}
    for s in stats:
        gap = means.loc["cause", s] - means.loc["irrelevant", s]
        signs[s] = 1 if gap >= 0 else -1
    return signs


def effect_first_rate(frame: pd.DataFrame, stats: list[str],
                      signs: dict[str, int]) -> pd.Series:
    """How often each statistic puts the effect at rank 1, at fixed orientation.

    The metric that matches the failure. Seven existing methods score 1.0 here;
    the null is the share of informative features that are effects, not 0.5,
    because the role counts are unequal.
    """
    sub = frame[frame.role.isin({"cause", "effect"})]
    draws = sorted(sub.draw.unique())
    out = {}
    for s in stats:
        hits = 0
        for d in draws:
            block = sub[sub.draw == d]
            top = block.loc[(signs[s] * block[s]).idxmax()]
            hits += int(top.role == "effect")
        out[s] = hits / len(draws)
    return pd.Series(out, name="effect_first")


def build_scm(args):
    def builder(draw: int):
        scm = nonlinear_scm(n=args.n, seed=draw)
        roles = {}
        for i, nm in enumerate(scm.feature_names):
            roles[nm] = ("cause" if scm.direct_causes[i] else
                         "effect" if scm.effects[i] else
                         "irrelevant" if scm.irrelevant[i] else "confounded")
        return as_dataset(scm.x, scm.y.astype(np.int64), scm.feature_names), roles
    return builder


def build_demo(args):
    def builder(draw: int):
        demo = redundancy_demo(n=args.n, seed=draw)
        y = (np.asarray(demo["y"]) > np.median(demo["y"])).astype(np.int64)
        roles = {"driver": "cause", "proxy_cos": "effect",
                 "proxy_sin": "effect", "unrelated": "irrelevant"}
        return as_dataset(np.asarray(demo["x"]), y, list(demo["feature_names"])), roles
    return builder


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--draws", type=int, default=8,
                   help="independent draws of each system; role counts are fixed "
                        "within a draw, so this is what supplies the sample size")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--outdir", default="ExpOutput/curves")
    args = p.parse_args()

    scm_frame = run_system("nonlinear_scm", build_scm(args), args)
    demo_frame = run_system("redundancy_demo", build_demo(args), args)

    stats = [c for c in scm_frame.columns if c not in ("feature", "role", "draw")]

    print(f"\n{'=' * 78}\nRANKING QUALITY (AUROC, chance = 0.5)\n{'=' * 78}")
    table = pd.concat([
        score(scm_frame, stats, {"cause"}, {"effect"}, "scm_cause_vs_effect"),
        score(scm_frame, stats, {"cause"}, {"irrelevant"}, "scm_cause_vs_noise"),
        score(demo_frame, stats, {"cause"}, {"effect"}, "demo_cause_vs_effect"),
    ], axis=1)
    auroc_cols = [c for c in table.columns
                  if not c.endswith("_raw") and not c.endswith("_prauc")]
    table["mean_auroc"] = table[auroc_cols].mean(axis=1)
    table = table.sort_values("mean_auroc", ascending=False)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 220):
        print(table[auroc_cols + ["mean_auroc"]].to_string())

    print(f"\n{'=' * 78}\nP(EFFECT RANKS FIRST) -- the metric that matches the failure")
    print("=" * 78)
    # Orientation is fixed on cause-vs-irrelevant, which is independent of the
    # cause-vs-effect question being scored.
    signs = orientations(scm_frame, stats)
    first = pd.DataFrame({
        "scm_effect_first": effect_first_rate(scm_frame, stats, signs),
        "demo_effect_first": effect_first_rate(demo_frame, stats, signs),
    })
    first["mean"] = first.mean(axis=1)
    first = first.sort_values("mean")
    print(f"  nulls: nonlinear_scm 1/4 = 0.250 (1 effect of 4 causal roles)")
    print(f"         redundancy_demo 2/3 = 0.667 (2 effects of 3 informative)")
    print("  every existing method scores 1.000 on both\n")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(first.to_string())

    best_auroc = table.index[0]
    best_first = first.index[0]
    print(f"\n  best AUROC:          {best_auroc} ({table.loc[best_auroc, 'mean_auroc']:.3f}) "
          f"vs final_loss ({table.loc['final_loss', 'mean_auroc']:.3f})")
    print(f"  lowest effect-first: {best_first} "
          f"(scm {first.loc[best_first, 'scm_effect_first']:.3f}, "
          f"demo {first.loc[best_first, 'demo_effect_first']:.3f})")
    print(f"                       final_loss "
          f"(scm {first.loc['final_loss', 'scm_effect_first']:.3f}, "
          f"demo {first.loc['final_loss', 'demo_effect_first']:.3f})")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    scm_frame.to_csv(outdir / "curves_scm.csv", index=False)
    demo_frame.to_csv(outdir / "curves_demo.csv", index=False)
    table.to_csv(outdir / "curve_auroc.csv")
    first.to_csv(outdir / "curve_effect_first.csv")
    print(f"\nwrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
