"""Can a *combination* of curve statistics recover causal role?

Individually, none of the fourteen curve statistics separated cause from effect
on both systems. That does not settle the question: a signal can be distributed
across several weak features and invisible in any one of them. This fits a model
on all fourteen and asks whether the combination does what the parts could not.

The design point that decides whether the answer means anything is what gets
held out.

Within-system cross-validation is nearly worthless here. Both systems are small
and fixed in structure, so a model can learn which *column position* tends to be
the effect and score well without having learned anything about causality. The
honest test is **cross-system**: fit on one generating process and predict on a
different one. A combination that recovers causal role should transfer; one that
has memorised a system's idiosyncrasies will not.

A label-permutation null is run alongside, because fourteen features against a
few dozen observations will produce a flattering number by chance and there is
no way to judge the real one without knowing what chance looks like.

    python scripts/curve_regressor.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Chance rate at which the effect takes the top rank, per system: the share of
# informative features that are effects. Not 0.5, and not equal across systems.
NULLS = {"scm": 1 / 4, "demo": 2 / 3}


def load(indir: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for key, name in (("scm", "curves_scm.csv"), ("demo", "curves_demo.csv")):
        frame = pd.read_csv(indir / name)
        # Only the two roles the question is about.
        frames[key] = frame[frame.role.isin({"cause", "effect"})].copy()
    return frames


def normalise_within_draw(frame: pd.DataFrame, stats: list[str]) -> pd.DataFrame:
    """Z-score each statistic within its own draw.

    Absolute scales differ between systems -- different feature counts, different
    loss levels -- so a model fitted on raw values would transfer nothing but
    those scales. Normalising within draw makes the input a *relative* profile
    across the features of one system, which is the only thing that could
    plausibly generalise. Uses no labels, so it leaks nothing.
    """
    out = frame.copy()
    for stat in stats:
        grouped = out.groupby("draw")[stat]
        spread = grouped.transform("std").replace(0.0, np.nan)
        out[stat] = ((out[stat] - grouped.transform("mean")) / spread).fillna(0.0)
    return out


def effect_first(frame: pd.DataFrame, scores: np.ndarray) -> float:
    """Fraction of draws whose highest-scoring feature is an effect.

    Scores are P(cause), so a working model puts a cause on top and this goes to
    zero. Every method tried so far sits at 1.0.
    """
    work = frame.assign(score=scores)
    hits = [int(block.loc[block.score.idxmax(), "role"] == "effect")
            for _, block in work.groupby("draw")]
    return float(np.mean(hits))


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, stats: list[str],
                model) -> tuple[float, float]:
    y_train = (train.role == "cause").astype(int)
    y_test = (test.role == "cause").astype(int)
    model.fit(train[stats], y_train)
    scores = model.predict_proba(test[stats])[:, 1]
    auc = roc_auc_score(y_test, scores) if y_test.nunique() > 1 else np.nan
    return auc, effect_first(test, scores)


def leave_one_draw_out(frame: pd.DataFrame, stats: list[str], model_fn):
    aucs, firsts = [], []
    for draw in sorted(frame.draw.unique()):
        train, test = frame[frame.draw != draw], frame[frame.draw == draw]
        if (train.role == "cause").nunique() < 2:
            continue
        model = model_fn()
        y = (train.role == "cause").astype(int)
        model.fit(train[stats], y)
        scores = model.predict_proba(test[stats])[:, 1]
        firsts.append(effect_first(test, scores))
        if (test.role == "cause").nunique() > 1:
            aucs.append(roc_auc_score((test.role == "cause").astype(int), scores))
    return (float(np.mean(aucs)) if aucs else np.nan), float(np.mean(firsts))


def permutation_null(frame: pd.DataFrame, stats: list[str], model_fn,
                     n: int = 200, seed: int = 0) -> tuple[float, float]:
    """Same protocol with labels shuffled *within* each draw.

    Shuffling within a draw preserves how many causes and effects each draw
    contains, so the null keeps the class balance and destroys only the
    association being tested.
    """
    rng = np.random.default_rng(seed)
    aucs, firsts = [], []
    for _ in range(n):
        shuffled = frame.copy()
        shuffled["role"] = (shuffled.groupby("draw")["role"]
                            .transform(lambda s: rng.permutation(s.values)))
        auc, first = leave_one_draw_out(shuffled, stats, model_fn)
        aucs.append(auc)
        firsts.append(first)
    return float(np.nanmean(aucs)), float(np.nanmean(firsts))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--indir", default="ExpOutput/curves")
    p.add_argument("--null-reps", type=int, default=100)
    p.add_argument("--outdir", default="ExpOutput/curves")
    args = p.parse_args()

    frames = load(Path(args.indir))
    stats = [c for c in frames["scm"].columns if c not in ("feature", "role", "draw")]
    normed = {k: normalise_within_draw(v, stats) for k, v in frames.items()}

    for key, frame in frames.items():
        counts = frame.role.value_counts().to_dict()
        print(f"  {key}: {len(frame)} observations {counts}, "
              f"{frame.draw.nunique()} draws, {len(stats)} statistics")

    models = {
        "logistic": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
        "forest": lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, random_state=0),
    }

    rows = []
    for model_name, model_fn in models.items():
        # Within-system: generous, and reported mainly so the gap to the
        # cross-system numbers is visible.
        for key in ("scm", "demo"):
            auc, first = leave_one_draw_out(normed[key], stats, model_fn)
            null_auc, null_first = permutation_null(
                normed[key], stats, model_fn, n=args.null_reps)
            rows.append({"model": model_name, "eval": f"within_{key}",
                         "auroc": auc, "effect_first": first,
                         "null_auroc": null_auc, "null_effect_first": null_first,
                         "chance_effect_first": NULLS[key]})

        # Cross-system: the test that matters.
        for train_key, test_key in (("scm", "demo"), ("demo", "scm")):
            auc, first = fit_predict(normed[train_key], normed[test_key],
                                     stats, model_fn())
            rows.append({"model": model_name, "eval": f"{train_key}->{test_key}",
                         "auroc": auc, "effect_first": first,
                         "null_auroc": np.nan, "null_effect_first": np.nan,
                         "chance_effect_first": NULLS[test_key]})

    table = pd.DataFrame(rows)
    print("\n" + "=" * 96)
    print("COMBINED CURVE STATISTICS AS A CAUSAL-ROLE CLASSIFIER")
    print("=" * 96)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 220):
        print(table.to_string(index=False))

    print("\nreading")
    print("  auroc            : ranking causes above effects, 0.5 is chance")
    print("  effect_first     : how often the top-ranked feature is an EFFECT;")
    print("                     compare against chance_effect_first, not 0.5")
    print("  null_*           : same protocol with labels shuffled within draw")

    cross = table[table["eval"].str.contains("->")]
    beats = cross[cross.effect_first < cross.chance_effect_first]
    print("\n" + "=" * 96)
    if len(beats):
        print("Cross-system transfer beat chance in:")
        for _, r in beats.iterrows():
            print(f"  {r.model} {r['eval']}: effect_first {r.effect_first:.3f} "
                  f"vs chance {r.chance_effect_first:.3f}")
        print("\n  Check against the usable-information reading before believing it:")
        print("  the effect is also the cheapest feature to read on both systems.")
    else:
        print("No cross-system transfer beats chance.")
        print("A combination of the fourteen statistics does not recover causal role")
        print("any better than the individual statistics did.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(outdir / "curve_regressor.csv", index=False)
    print(f"\nwrote {outdir}/curve_regressor.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
