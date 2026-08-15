"""A map of where the cheap importance methods actually break.

Before anyone proposes a remedy, the problem has to be located.  So far this
project has exactly one documented failure of a cheap method: on
``redundancy_demo`` leave-one-out and permutation importance both return
exactly 0.0000 for a deterministically redundant true cause, while impurity,
mutual information and Shapley recover it (``scripts/blindness_scope.py``).
Everywhere else random forest impurity and permutation have won or tied.  A
single anchor case is not a failure map, and "it broke on the one system we
built to break it" is not a result anyone should act on.

This sweeps the two axes that ought to hurt a tree, and reports whether they do.

* **Interaction order k.**  A tree can only express a k-way coupling by
  assembling all k members on one root-to-leaf path.  The number of paths that
  do so falls off sharply with k, so impurity credit for an interaction member
  should thin out as k grows even though the feature is fully causal.
* **Dimension d.**  ``max_features`` sampling means each split considers a
  shrinking fraction of the columns as d grows, so the chance of ever getting
  all k members onto one path falls again, for a different reason.

Both are run on ``parity_interaction`` (axis-aligned, native to a tree) and
``oblique_interaction`` (smooth and tilted to every axis, so the tree must
build a staircase).  ``redundancy_demo`` is included as the anchor.

Two rules this project paid for, and how they are honoured here
---------------------------------------------------------------
*Never score a method against a target the fitted model cannot learn.*  A
method cannot be blamed for missing structure that the model class never
represented, and the failure is silent -- it returns a perfectly well-formed
AUROC near chance.  Per that rule the scored contrast is
``causal = interaction OR marginal`` against irrelevant; ``auc_interaction`` is a
diagnostic only.  The guard has three parts, because an earlier version of this
script had only the first and it passed cleanly on cells that were in fact
uninterpretable.

1. *Can the model class represent it?*  Held-out AUC of a plain forest on all
   columns, on the interaction columns alone, and on the marginal columns alone.
2. *Is the guard in the same metric the method is?*  ``permutation`` and ``loco``
   difference a **score**, so the number that governs them is the fitted model's
   held-out value of *that score*, not of AUC.  These parity targets are the
   trap: with two marginal causes the marginals-only Bayes *classifier* is the
   constant classifier, so at large d the forest sits at or below the
   majority-class rate while its AUC still reads a healthy 0.68.  An
   accuracy-differencing method then has nothing to resolve and returns noise,
   and an AUC-only guard calls the cell fine.  Every cell therefore reports
   held-out accuracy beside the majority-class rate, and every method is run
   under both scores (``_acc`` and ``_auc``) off the same refits, so the choice
   of score is visible rather than assumed.
3. *Does the fitted model use these columns at all?*  ``permutation`` and
   ``rf_impurity`` explain one fitted model; they cannot find a column that model
   ignores, and reporting that as method blindness is the same error one level
   down.  So each cell also joint-permutes the interaction group, the marginal
   group and the irrelevant group on held-out data and records the AUC each
   costs the fitted model.  When the interaction group costs no more than the
   irrelevant group, the model demonstrably ignores it and no model-explaining
   method's interaction score in that cell is evidence about the method.

*Never orient a statistic to maximise the reported metric.*  Nothing here has a
free sign.  All five methods are defined so that larger means more important,
fixed by their definitions before any data was seen, and that orientation is
used unchanged.

Shapley is priced, not assumed away.  Its value function is a cross-validated
forest, so one evaluation costs about 0.7 s at d=20 and 0.9 s at d=80 on this
machine; the sampled estimator needs ``n_permutations * d`` of them, which is
roughly 9 minutes per seed at d=20 and 46 minutes per seed at d=80.  It is
therefore run only where the feature count allows (``--shapley-max-d``, default
4, which covers ``redundancy_demo`` exactly), and every cell where it is skipped
says so in the table rather than quietly showing four methods instead of five.

    python scripts/failure_atlas.py
    python scripts/failure_atlas.py --n 3000 --seeds 2
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from deepfeatselect.scaling import ScalingSystem, oblique_interaction, parity_interaction
from deepfeatselect.shapley import shapley_importance
from deepfeatselect.synthetic import redundancy_demo

sys.path.insert(0, str(Path(__file__).resolve().parent))
# score_permutation is deliberately NOT imported: the local one below is the
# same estimator with a second scorer attached, and importing both would leave
# two functions of that name in the module.
from scaling_benchmark import (  # noqa: E402
    detection_auc, score_forest, score_mutual_info,
)

# Below this the detection is not usefully better than guessing; used only to
# name failures in the summary, never to alter a score.
FAILURE_THRESHOLD = 0.60

# A guard model this close to 0.5 has learned nothing, so every detection score
# in that cell is measuring the target's learnability rather than the method.
CHANCE_CEILING = 0.55

# Held-out AUC a group of columns must cost the fitted model, over what the
# irrelevant columns cost it, before that model can be said to use the group.
# Below this the model ignores them and cannot be interrogated about them.  The
# exact value is not load-bearing: on this sweep the cells separate into ones
# where the excess is around 0.10 to 0.23 and ones where it is negative, two
# orders of magnitude apart, so any threshold in between sorts them identically.
GROUP_USE_MARGIN = 0.01

# Methods that explain one fitted model rather than the data.  They are bounded
# by what that model uses, so a low score on columns the model ignores is the
# method working, not failing.  LOCO refits per ablation and is not in this set.
# The group-usage test is run on the forest ``score_permutation`` fits, which is
# that method's model exactly; ``rf_impurity``'s forest is a sibling of it --
# same class, same data, 400 trees instead of 300 and no held-out split -- so
# for impurity the disqualification is an inference about an equivalent model
# rather than a measurement of its own.
MODEL_EXPLAINING = frozenset({"permutation_acc", "permutation_auc", "rf_impurity"})

METHOD_ORDER = ["loco_acc", "loco_auc", "permutation_acc", "permutation_auc",
                "rf_impurity", "mutual_info", "mutual_info_typed", "shapley"]

# Scores this close together, relative to the largest score in the cell, are one
# value.  Not cosmetic: the anchor result is that permutation gives the true
# cause *exactly* zero, and permutation under an AUC scorer returns 1.1e-17
# rather than 0.0 for it.  Untreated, that residue outranks a genuine 0.0 noise
# column, and the headline blindness silently becomes "finds it".
TIE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Case:
    """One (system, condition) pair: the columns to score and how to score them."""
    system: str
    condition: str
    x: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    causal: np.ndarray
    irrelevant: np.ndarray
    # Empty for redundancy_demo, which has no interaction/marginal split.
    interaction: np.ndarray | None
    marginal: np.ndarray | None
    # What ``causal`` actually means here. On redundancy_demo the positive group
    # is "informative", not "causal": proxy_cos and proxy_sin are deterministic
    # functions of the driver, so they are siblings of the cause rather than
    # causes, and calling the contrast causal there would overstate it.
    contrast: str = "causal (interaction OR marginal) vs irrelevant"


def score_loco(x: np.ndarray, y: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Retrained leave-one-out, read off the same refits under two scores.

    The estimator is the one in ``blindness_scope.py`` -- retrained rather than
    marginalised -- so the anchor cell reproduces the documented exact zero
    rather than a near miss that would have to be argued about.

    It reports accuracy *and* AUC because on these systems accuracy is not
    always a live metric.  A LOCO score is a difference of two values of the
    chosen score, so when the fitted model sits at the majority-class rate that
    difference is bounded by nothing but sampling noise, and the resulting
    ranking is noise wearing a well-formed number.  Both come out of one set of
    ``d + 1`` refits, so carrying both costs nothing and removes the need to
    pick one -- picking one after seeing the answer is the second rule.
    """
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.3, random_state=seed, stratify=y)

    def fit(cols: list[int]) -> tuple[float, float]:
        model = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        model.fit(x_tr[:, cols], y_tr)
        return (float(model.score(x_te[:, cols], y_te)),
                float(roc_auc_score(y_te, model.predict_proba(x_te[:, cols])[:, 1])))

    full_acc, full_auc = fit(list(range(x.shape[1])))
    ablated = [fit([c for c in range(x.shape[1]) if c != j])
               for j in range(x.shape[1])]
    return {
        "loco_acc": np.array([full_acc - a for a, _ in ablated]),
        "loco_auc": np.array([full_auc - u for _, u in ablated]),
    }


def score_permutation(x: np.ndarray, y: np.ndarray,
                      seed: int) -> dict[str, np.ndarray]:
    """Permutation importance on held-out data, under both scores at once.

    Identical in construction to ``scaling_benchmark.score_permutation`` -- same
    split, same forest, same ten repeats -- so ``permutation_acc`` reproduces
    that function exactly; sklearn draws the shuffling indices before scoring,
    so adding a second scorer does not perturb the first.  The AUC arm exists
    for the reason given in ``score_loco``: the default scorer is accuracy, and
    accuracy is degenerate on a target whose Bayes classifier is constant.
    """
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.3, random_state=seed, stratify=y)
    model = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    model.fit(x_tr, y_tr)
    result = permutation_importance(model, x_te, y_te, n_repeats=10,
                                    random_state=seed, n_jobs=-1,
                                    scoring=["accuracy", "roc_auc"])
    return {"permutation_acc": result["accuracy"].importances_mean,
            "permutation_auc": result["roc_auc"].importances_mean}


def score_mutual_info_typed(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Mutual information with the discrete/continuous flag set per column.

    ``scaling_benchmark.score_mutual_info`` takes sklearn's default,
    ``discrete_features='auto'``, which resolves to *continuous* for any dense
    array and estimates the information with a k-nearest-neighbour rule.  The
    parity systems are +-1 bits: every distance is 0 or a constant, sklearn has
    to jitter the columns to break the ties, and the estimate that comes back is
    mostly jitter.  That is an estimator misconfiguration rather than a property
    of mutual information, and it shows up as the marginal control -- an
    individually informative binary cause that must score near 1.0 -- coming
    back at 0.56.  Flagging binary columns as discrete uses the exact plug-in
    estimator instead.  Both variants are reported, so the artefact is visible.
    """
    discrete = np.array([len(np.unique(x[:, j])) <= 2 for j in range(x.shape[1])])
    if discrete.any():
        return mutual_info_classif(x, y, discrete_features=discrete,
                                   random_state=seed)
    return mutual_info_classif(x, y, random_state=seed)


def held_out_auc(x: np.ndarray, y: np.ndarray, cols: np.ndarray | None,
                 seed: int) -> float:
    """AUC of a plain forest restricted to ``cols``, on a held-out split.

    ``cols`` of ``None`` means every column.  Returns NaN for an empty column
    set, which is the honest answer when a system has no such group at all.
    """
    subset = x if cols is None else x[:, cols]
    if subset.shape[1] == 0:
        return float("nan")
    x_tr, x_te, y_tr, y_te = train_test_split(
        subset, y, test_size=0.3, random_state=seed, stratify=y)
    model = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    model.fit(x_tr, y_tr)
    return float(roc_auc_score(y_te, model.predict_proba(x_te)[:, 1]))


def snap_ties(scores: np.ndarray) -> np.ndarray:
    """Round away differences too small to be a ranking.

    Applied to every method in every cell, before any statistic is taken, so it
    is a property of the harness rather than a choice made per result.  An
    importance estimated by refitting forests does not resolve 1e-17; treating
    such a gap as an ordering is how a measure that returns exactly zero for the
    true cause gets scored as though it had ranked it first.
    """
    scale = max(float(np.max(np.abs(scores))), np.finfo(float).tiny)
    return np.round(scores / (TIE_TOLERANCE * scale)) * (TIE_TOLERANCE * scale)


def univariate_strength(x: np.ndarray, y: np.ndarray,
                        groups: dict[str, np.ndarray]) -> dict[str, float]:
    """Mean ``|AUC - 0.5|`` of each column on its own, per group.  No model.

    This is the question of whether a group is a *joint*-structure test at all.
    A parity member is exactly uninformative alone, so any method that ranks it
    above noise has had to represent the coupling.  If instead the members are
    individually informative, a method can rank them first with no
    representation of the coupling whatever, and a perfect score on that group
    is not evidence that the coupling was found.  Reported per cell because the
    two generators here differ on exactly this point and the difference decides
    how their results may be read.
    """
    out = {}
    for name, mask in groups.items():
        if mask is None or not mask.any():
            out[f"univariate_{name}"] = float("nan")
            continue
        out[f"univariate_{name}"] = float(np.mean(
            [abs(roc_auc_score(y, x[:, j]) - 0.5) for j in np.where(mask)[0]]))
    return out


def fitted_model_guard(x: np.ndarray, y: np.ndarray, groups: dict[str, np.ndarray],
                       seed: int) -> dict[str, float]:
    """What the model that permutation explains can and cannot do.

    Fits the *same* forest on the *same* split that ``score_permutation`` uses,
    then reports three things about it: its held-out AUC, its held-out accuracy
    beside the majority-class rate (accuracy below that rate means an
    accuracy-differencing method is resolving nothing), and the AUC it loses
    when each named group of columns is permuted *jointly* on the held-out set.

    The joint permutation is the part that matters.  A parity is invisible one
    column at a time by construction, so permuting members singly cannot
    distinguish "the model encodes the parity" from "the model ignores these
    columns".  Breaking the whole group at once can: if the model has the
    parity, destroying it costs AUC.  The irrelevant group is permuted the same
    way as the null reference, since disturbing any columns at all costs a
    little through the trees' spurious splits.
    """
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.3, random_state=seed, stratify=y)
    model = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    model.fit(x_tr, y_tr)
    base = float(roc_auc_score(y_te, model.predict_proba(x_te)[:, 1]))
    majority = float(max(y_te.mean(), 1.0 - y_te.mean()))
    out = {"rf_auc_all": base,
           "rf_acc_all": float(model.score(x_te, y_te)),
           "majority_rate": majority}

    rng = np.random.default_rng(seed)
    for name, mask in groups.items():
        if mask is None or not mask.any():
            out[f"group_cost_{name}"] = float("nan")
            continue
        columns = np.where(mask)[0]
        losses = []
        for _ in range(10):
            shuffled = x_te.copy()
            shuffled[:, columns] = x_te[rng.permutation(len(x_te))][:, columns]
            losses.append(base - roc_auc_score(
                y_te, model.predict_proba(shuffled)[:, 1]))
        out[f"group_cost_{name}"] = float(np.mean(losses))
    return out


def scaling_case(generator: str, d: int, k: int, n: int, seed: int) -> Case:
    system: ScalingSystem = (parity_interaction if generator == "parity"
                             else oblique_interaction)(
        n=n, n_features=d, k=k, n_marginal=2, seed=seed)
    return Case(
        system=generator,
        condition=f"d={d},k={k}",
        x=system.x,
        y=system.y,
        feature_names=system.feature_names,
        causal=system.causal,
        irrelevant=system.irrelevant,
        interaction=system.interaction,
        marginal=system.marginal,
    )


def redundancy_case(n: int, seed: int) -> Case:
    """The anchor.  Binarised at the median exactly as ``blindness_scope`` does.

    The target is a continuous logistic-map successor; the classifiers and the
    detection AUROC both need a binary label, and the median split is the
    choice already made in the documented run, so the anchor stays comparable.
    """
    demo = redundancy_demo(n=n, seed=seed)
    x = np.asarray(demo["x"], dtype=np.float64)
    y = (np.asarray(demo["y"]) > np.median(demo["y"])).astype(np.int64)
    names = list(demo["feature_names"])
    causal = np.array([name != "unrelated" for name in names])
    return Case(
        system="redundancy_demo",
        condition="d=4,deterministic",
        x=x, y=y, feature_names=names,
        causal=causal, irrelevant=~causal,
        interaction=None, marginal=None,
        contrast="informative (driver, both proxies) vs irrelevant",
    )


def run_case(
    case: Case, seed: int, shapley_max_d: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Score every method on one case, with the guard attached to every row.

    Returns the per-method summary rows and the per-feature raw scores, the
    latter so a diagnostic thought of later costs a groupby rather than a refit.
    """
    x, y = case.x, case.y
    d = x.shape[1]

    # The guard, computed first: if the forest is at chance here -- in the metric
    # the method is scored in -- everything below it is uninterpretable, and the
    # flag has to travel with the numbers rather than sit in a separate table.
    groups = {"int": case.interaction, "marg": case.marginal,
              "irr": case.irrelevant}
    fitted = fitted_model_guard(x, y, groups, seed)
    fitted.update(univariate_strength(x, y, groups))
    guard_all = fitted["rf_auc_all"]
    guard_int = (held_out_auc(x, y, case.interaction, seed)
                 if case.interaction is not None else float("nan"))
    guard_marg = (held_out_auc(x, y, case.marginal, seed)
                  if case.marginal is not None else float("nan"))
    # Does the model that permutation and impurity explain use the interaction
    # columns at all?  Anything it ignores it cannot be interrogated about.  A
    # system with no interaction group cannot be disqualified on this ground, so
    # the missing test passes rather than failing.
    uses_interaction = bool(
        np.isnan(fitted["group_cost_int"])
        or fitted["group_cost_int"] - fitted["group_cost_irr"] > GROUP_USE_MARGIN)
    # Likewise in the metric: an accuracy difference on a model pinned at the
    # majority-class rate is noise however well formed the AUROC it produces.
    accuracy_live = fitted["rf_acc_all"] > fitted["majority_rate"]

    def shapley_scores(a: np.ndarray, b: np.ndarray, s: int) -> np.ndarray:
        # shapley_importance returns a frame sorted by value, so the scores have
        # to be put back into column order by name before they can be masked.
        lookup = shapley_importance(
            a, b.astype(np.float64), case.feature_names, seed=s
        ).set_index("feature")["sage"]
        return np.array([lookup[name] for name in case.feature_names])

    methods = {
        # Two of these return a dict: one set of refits, scored two ways.
        "loco": score_loco,
        "permutation": score_permutation,
        "rf_impurity": score_forest,
        "mutual_info": score_mutual_info,
        "mutual_info_typed": score_mutual_info_typed,
    }
    if d <= shapley_max_d:
        methods["shapley"] = shapley_scores

    rows: list[dict[str, object]] = []
    per_feature: list[dict[str, object]] = []
    computed: dict[str, tuple[np.ndarray, float]] = {}
    for name, function in methods.items():
        start = time.time()
        produced = function(x, y, seed)
        elapsed = time.time() - start
        if isinstance(produced, dict):
            # The two scorings share the refits, so they share the cost too.
            computed.update({k: (v, elapsed / len(produced))
                             for k, v in produced.items()})
        else:
            computed[name] = (produced, elapsed)

    for name, (raw_scores, elapsed) in computed.items():
        scores = snap_ties(raw_scores)
        # Kept so any later diagnostic can be recomputed without refitting.
        role = np.where(case.causal, "causal", "irrelevant")
        per_feature.extend(
            {"system": case.system, "condition": case.condition, "seed": seed,
             "method": name, "feature": feature, "role": str(role[i]),
             "score": float(scores[i])}
            for i, feature in enumerate(case.feature_names))
        # AUROC credits a tie as half a point, which on a system with a single
        # irrelevant column is far too forgiving: a method that scores two of
        # three true causes at exactly zero still reads 0.667. That is precisely
        # the documented redundancy failure, so it needs a statistic that ties
        # cannot launder. This one counts causal features the method ranks no
        # higher than the best irrelevant feature.
        buried = int(np.sum(scores[case.causal] <= scores[case.irrelevant].max()))
        rows.append({
            "system": case.system,
            "condition": case.condition,
            "d": d,
            "seed": seed,
            "method": name,
            "auc_causal": detection_auc(scores, case.causal, case.irrelevant),
            "n_causal": int(case.causal.sum()),
            "causal_buried": buried,
            "auc_interaction": (detection_auc(scores, case.interaction, case.irrelevant)
                                if case.interaction is not None else float("nan")),
            "auc_marginal": (detection_auc(scores, case.marginal, case.irrelevant)
                             if case.marginal is not None else float("nan")),
            "contrast": case.contrast,
            "rf_auc_all": guard_all,
            "rf_auc_int_only": guard_int,
            "rf_auc_marg_only": guard_marg,
            "rf_acc_all": fitted["rf_acc_all"],
            "majority_rate": fitted["majority_rate"],
            "group_cost_int": fitted["group_cost_int"],
            "group_cost_marg": fitted["group_cost_marg"],
            "group_cost_irr": fitted["group_cost_irr"],
            "univariate_int": fitted["univariate_int"],
            "univariate_marg": fitted["univariate_marg"],
            "univariate_irr": fitted["univariate_irr"],
            # Why a low auc_interaction in this cell may not be about the method.
            "model_uses_interaction": uses_interaction,
            "metric_live": accuracy_live if name.endswith("_acc") else True,
            "model_dependent": name in MODEL_EXPLAINING,
            "seconds": elapsed,
        })
    if d > shapley_max_d:
        # Recorded as a row so the omission is visible in the table itself.
        rows.append({
            "system": case.system, "condition": case.condition, "d": d,
            "seed": seed, "method": "shapley",
            "auc_causal": float("nan"), "n_causal": int(case.causal.sum()),
            "causal_buried": float("nan"), "auc_interaction": float("nan"),
            "auc_marginal": float("nan"), "contrast": case.contrast,
            "rf_auc_all": guard_all,
            "rf_auc_int_only": guard_int, "rf_auc_marg_only": guard_marg,
            "rf_acc_all": fitted["rf_acc_all"],
            "majority_rate": fitted["majority_rate"],
            "group_cost_int": fitted["group_cost_int"],
            "group_cost_marg": fitted["group_cost_marg"],
            "group_cost_irr": fitted["group_cost_irr"],
            "univariate_int": fitted["univariate_int"],
            "univariate_marg": fitted["univariate_marg"],
            "univariate_irr": fitted["univariate_irr"],
            "model_uses_interaction": uses_interaction,
            "metric_live": True, "model_dependent": False,
            "seconds": float("nan"), "skipped": "infeasible at this d",
        })
    return rows, per_feature


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 4, 6],
                        help="interaction orders, swept at --order-dim")
    parser.add_argument("--order-dim", type=int, default=20)
    parser.add_argument("--dims", type=int, nargs="+", default=[20, 40, 80],
                        help="feature counts, swept at --dim-k")
    parser.add_argument("--dim-k", type=int, default=4)
    parser.add_argument("--shapley-max-d", type=int, default=4,
                        help="largest d at which Shapley is affordable; above "
                             "it the run is skipped and marked, not dropped")
    parser.add_argument("--outdir", default="ExpOutput/failure_atlas")
    args = parser.parse_args()

    # Union of the two sweeps, so the shared (order-dim, dim-k) corner is run once.
    conditions = sorted({(args.order_dim, k) for k in args.ks}
                        | {(d, args.dim_k) for d in args.dims})

    rows: list[dict[str, object]] = []
    scores: list[dict[str, object]] = []
    for seed in range(args.seeds):
        for generator in ("parity", "oblique"):
            for d, k in conditions:
                case = scaling_case(generator, d, k, args.n, seed)
                new_rows, new_scores = run_case(case, seed, args.shapley_max_d)
                rows.extend(new_rows)
                scores.extend(new_scores)
                print(f"  {generator:<8} d={d:<3} k={k}  seed={seed} done", flush=True)
        case = redundancy_case(args.n, seed)
        new_rows, new_scores = run_case(case, seed, args.shapley_max_d)
        rows.extend(new_rows)
        scores.extend(new_scores)
        print(f"  redundancy_demo         seed={seed} done", flush=True)

    frame = pd.DataFrame(rows)
    score_frame = pd.DataFrame(scores)
    if "skipped" not in frame.columns:
        frame["skipped"] = np.nan

    summary = (frame.groupby(["system", "condition", "method"], sort=False)
               .agg(auc_causal=("auc_causal", "mean"),
                    causal_buried=("causal_buried", "mean"),
                    n_causal=("n_causal", "max"),
                    auc_interaction=("auc_interaction", "mean"),
                    auc_marginal=("auc_marginal", "mean"),
                    rf_auc_all=("rf_auc_all", "mean"),
                    rf_auc_int_only=("rf_auc_int_only", "mean"),
                    rf_auc_marg_only=("rf_auc_marg_only", "mean"),
                    rf_acc_all=("rf_acc_all", "mean"),
                    majority_rate=("majority_rate", "mean"),
                    group_cost_int=("group_cost_int", "mean"),
                    group_cost_marg=("group_cost_marg", "mean"),
                    group_cost_irr=("group_cost_irr", "mean"),
                    univariate_int=("univariate_int", "mean"),
                    univariate_marg=("univariate_marg", "mean"),
                    univariate_irr=("univariate_irr", "mean"),
                    # "all" so a cell counts as guarded only if every seed agrees.
                    model_uses_interaction=("model_uses_interaction", "all"),
                    metric_live=("metric_live", "all"),
                    model_dependent=("model_dependent", "first"),
                    seconds=("seconds", "mean"),
                    # count ignores NaN, so this counts only the marked rows.
                    n_skipped=("skipped", "count"),
                    n_seeds=("auc_causal", "size"))
               .reset_index())
    # A cell that was never run must not be able to look like a cell that ran
    # and scored badly, so the two are separated before anything is judged.
    summary["ran"] = summary.n_skipped == 0
    summary["method"] = pd.Categorical(summary.method, METHOD_ORDER, ordered=True)
    summary = summary.sort_values(["system", "condition", "method"], kind="stable")

    print("\n" + "=" * 118)
    print("FAILURE ATLAS -- DETECTION AUROC OF CAUSAL (interaction OR marginal) "
          "VERSUS IRRELEVANT")
    print("=" * 118)
    print(f"  n={args.n}, {args.seeds} seeds per cell, 0.5 = chance")
    print("  _acc / _auc are the same refits differenced under accuracy and under AUC.")
    print("  mutual_info is the house default (kNN); mutual_info_typed flags binary")
    print("  columns as discrete, which is what the parity systems actually are.\n")
    display = summary.copy()
    display["note"] = np.where(~display.ran, "SKIPPED (infeasible)",
                               np.where(display.auc_causal.isna(),
                                        "DEGENERATE (no ranking)", ""))
    with pd.option_context("display.float_format", "{:.3f}".format,
                           "display.width", 240, "display.max_rows", 400):
        print(display[["system", "condition", "method", "auc_causal",
                       "auc_interaction", "auc_marginal", "rf_auc_all",
                       "seconds", "note"]].to_string(index=False))

    # ------------------------------------------------------------------ guard
    print("\n" + "=" * 118)
    print("GUARD -- WHAT THE FITTED MODEL IN EACH CELL CAN AND CANNOT DO")
    print("=" * 118)
    print("  rf_auc_all      : all columns. At chance, every score above is meaningless.")
    print("  rf_auc_int_only : interaction columns only -- is that structure")
    print("                    learnable by this model class at all?")
    print("  rf_auc_marg_only: marginal columns only -- the control.")
    print("  rf_acc_all/major: held-out ACCURACY against the majority-class rate. An")
    print("                    accuracy-differencing method (_acc) on a model at or")
    print("                    below the majority rate is resolving nothing.")
    print("  cost_int/marg/irr: AUC the fitted model loses when that group of columns")
    print("                    is permuted JOINTLY. cost_irr is the null reference;")
    print("                    cost_int no bigger than it means the model ignores the")
    print("                    interaction columns, so nothing that explains that")
    print("                    model can be blamed for missing them.")
    print("  uni_int/marg/irr: mean |AUC-0.5| of each column ALONE, no model. Where")
    print("                    uni_int sits at uni_irr the interaction is a genuine")
    print("                    joint-structure test; where it sits at uni_marg the")
    print("                    members are individually detectable and a perfect")
    print("                    score on them is not evidence a coupling was found.\n")
    guard = (summary.groupby(["system", "condition"], sort=False)
             .agg(rf_auc_all=("rf_auc_all", "first"),
                  rf_auc_int_only=("rf_auc_int_only", "first"),
                  rf_auc_marg_only=("rf_auc_marg_only", "first"),
                  rf_acc_all=("rf_acc_all", "first"),
                  majority_rate=("majority_rate", "first"),
                  cost_int=("group_cost_int", "first"),
                  cost_marg=("group_cost_marg", "first"),
                  cost_irr=("group_cost_irr", "first"),
                  uni_int=("univariate_int", "first"),
                  uni_marg=("univariate_marg", "first"),
                  uni_irr=("univariate_irr", "first"),
                  uses_int=("model_uses_interaction", "first"))
             .reset_index())
    guard["verdict"] = np.where(
        guard.rf_auc_all < CHANCE_CEILING, "AT CHANCE -- SCORES MEANINGLESS",
        np.where(guard.rf_acc_all <= guard.majority_rate,
                 "ACCURACY DEAD -- _acc methods uninterpretable",
                 np.where(~guard.uses_int.fillna(True),
                          "MODEL IGNORES INTERACTION COLUMNS", "ok")))
    with pd.option_context("display.float_format", "{:.3f}".format,
                           "display.width", 240):
        print(guard.to_string(index=False))

    dead = guard[guard.rf_auc_all < CHANCE_CEILING]
    if len(dead):
        print("\n  WARNING: the cells above marked AT CHANCE have no learnable signal;")
        print("  their detection AUROCs measure the target, not the method.")
    else:
        print("\n  No cell is at chance on AUC.")

    flat = guard[guard.rf_acc_all <= guard.majority_rate]
    if len(flat):
        print("\n  ACCURACY IS DEAD in these cells -- the fitted forest does not beat")
        print("  the constant classifier, so loco_acc and permutation_acc are")
        print("  differencing a quantity with no signal in it. Read the _auc arms:")
        for _, row in flat.iterrows():
            print(f"    {row.system:<16} {row.condition:<18} "
                  f"rf_acc_all={row.rf_acc_all:.4f} vs majority={row.majority_rate:.4f}")

    ignored = guard[~guard.uses_int.fillna(True)]
    if len(ignored):
        print("\n  THE FITTED MODEL IGNORES THE INTERACTION COLUMNS in these cells --")
        print("  permuting all of them jointly costs it no more AUC than permuting the")
        print("  noise columns. permutation_* and rf_impurity explain that model, so")
        print("  their auc_interaction here is not evidence about the method:")
        for _, row in ignored.iterrows():
            print(f"    {row.system:<16} {row.condition:<18} "
                  f"cost_int={row.cost_int:+.4f} vs cost_irr={row.cost_irr:+.4f} "
                  f"(cost_marg={row.cost_marg:+.4f}, so the model does use those)")

    # Halfway between the noise floor and the marginal causes is enough: past
    # that point a member is detectable on its own and the group stops being a
    # test of whether a method can represent a coupling.
    marginally_visible = guard[
        guard.uni_int > 0.5 * (guard.uni_irr + guard.uni_marg)]
    if len(marginally_visible):
        print("\n  NOT A JOINT-STRUCTURE TEST in these cells -- each interaction")
        print("  member is individually about as detectable as a marginal cause, so")
        print("  ranking them first needs no representation of the coupling and a")
        print("  score of 1.000 on them proves nothing about interaction detection:")
        for _, row in marginally_visible.iterrows():
            print(f"    {row.system:<16} {row.condition:<18} "
                  f"uni_int={row.uni_int:.4f} vs uni_marg={row.uni_marg:.4f}, "
                  f"noise floor uni_irr={row.uni_irr:.4f}")

    unlearnable = guard[(guard.rf_auc_int_only < CHANCE_CEILING)
                        & guard.rf_auc_int_only.notna()]
    if len(unlearnable):
        print("\n  NOTE: in these cells a forest given ONLY the interaction columns is")
        print("  itself at chance, so an interaction member is invisible to the model")
        print("  class rather than to the importance method:")
        for _, row in unlearnable.iterrows():
            print(f"    {row.system:<16} {row.condition:<18} "
                  f"rf_auc_int_only={row.rf_auc_int_only:.3f}")

    # ----------------------------------------------------------- anchor check
    print("\n" + "=" * 118)
    print("ANCHOR CHECK -- DOES THE DOCUMENTED REDUNDANCY FAILURE REPRODUCE?")
    print("=" * 118)
    print("  blindness_scope reports LOCO and permutation returning exactly 0.0000")
    print("  for 'driver', the true cause. AUROC hides this: with one irrelevant")
    print("  column, ties are worth half a point and a method can bury two of three")
    print("  true causes and still read 0.667. The raw scores are the honest view.\n")
    anchor = score_frame[score_frame.system == "redundancy_demo"]
    if len(anchor):
        pivot = (anchor.groupby(["method", "feature"], sort=False).score.mean()
                 .unstack("feature"))
        order = [c for c in ["driver", "proxy_cos", "proxy_sin", "unrelated"]
                 if c in pivot.columns]
        with pd.option_context("display.float_format", "{:+.4f}".format,
                               "display.width", 200):
            print(pivot[order].to_string())
        print("\n  true cause 'driver' versus the irrelevant 'unrelated':")
        for method in pivot.index:
            drives, null = pivot.loc[method, "driver"], pivot.loc[method, "unrelated"]
            # Tie counts as blind: the claim being checked is that the measure
            # gives the true cause nothing, not that it ranks it last.
            blind = drives <= null + TIE_TOLERANCE * max(abs(drives), abs(null), 1e-12)
            print(f"    {method:<18} driver {drives:+.4f}  unrelated {null:+.4f}"
                  f"   -> {'BLIND TO THE TRUE CAUSE' if blind else 'finds it'}")

    # --------------------------------------------------------------- failures
    print("\n" + "=" * 118)
    print(f"FAILURE CELLS -- every (system, condition, method) below "
          f"AUROC {FAILURE_THRESHOLD:.2f}, adjudicated against the guard")
    print("=" * 118)
    print("  A low AUROC is only evidence about the METHOD when the guard says the")
    print("  method had something to find. Two disqualifications apply here:")
    print("    CONFOUNDED (metric) -- an _acc method in a cell where the fitted")
    print("      forest does not beat the majority-class rate.")
    print("    CONFOUNDED (model)  -- a model-explaining method in a cell where the")
    print("      fitted forest provably ignores the interaction columns.")
    print("  Confounded cells are listed, never silently dropped, and never counted.\n")
    scored = summary[summary.ran].copy()

    def verdict(row: pd.Series) -> str:
        if row.model_dependent and not row.model_uses_interaction:
            return "CONFOUNDED (model)"
        if not row.metric_live:
            return "CONFOUNDED (metric)"
        return "METHOD FAILURE"

    scored["failure_verdict"] = scored.apply(verdict, axis=1)
    failures = scored[scored.auc_causal < FAILURE_THRESHOLD].sort_values("auc_causal")
    degenerate = scored[scored.auc_causal.isna()]
    if len(failures):
        for _, row in failures.iterrows():
            print(f"  {row.system:<16} {row.condition:<18} {str(row.method):<18} "
                  f"auc_causal {row.auc_causal:.3f}  "
                  f"(int {row.auc_interaction:.3f}, marg {row.auc_marginal:.3f})  "
                  f"-> {row.failure_verdict}")
    else:
        print("  none")

    real = failures[failures.failure_verdict == "METHOD FAILURE"]
    print(f"\n  {len(real)} of {len(failures)} sub-{FAILURE_THRESHOLD:.2f} cells "
          f"survive the guard as genuine method failures.")
    if len(degenerate):
        print("\n  DEGENERATE -- the method returned no ranking at all (every score")
        print("  identical, so AUROC is undefined). This is a failure, not a gap:")
        for _, row in degenerate.iterrows():
            print(f"  {row.system:<16} {row.condition:<18} {str(row.method):<18} "
                  f"auc_causal undefined   (guard rf_auc_all {row.rf_auc_all:.3f})")

    print("\n  BURIED CAUSES -- causal features scored no higher than the best")
    print("  irrelevant feature. AUROC can stay respectable while this is nonzero,")
    print("  so it is reported separately rather than folded into the headline.")
    print("  The bar is the MAXIMUM over irrelevant columns, an order statistic that")
    print("  rises with d, so counts are comparable across methods within a cell but")
    print("  not across cells of different width:")
    buried = scored[scored.causal_buried > 0].sort_values(
        "causal_buried", ascending=False)
    if len(buried):
        for _, row in buried.iterrows():
            auc = ("undefined" if pd.isna(row.auc_causal)
                   else f"{row.auc_causal:.3f}")
            print(f"    {row.system:<16} {row.condition:<18} {str(row.method):<18} "
                  f"{row.causal_buried:.1f} of {int(row.n_causal)} buried "
                  f"(auc_causal {auc})")
    else:
        print("    none")

    # ------------------------------------------------------------ closing call
    print("\n" + "=" * 118)
    print("METHODS WITH NO GUARD-SURVIVING FAILURE ACROSS THIS SWEEP")
    print("=" * 118)
    for method in METHOD_ORDER:
        cells = scored[scored.method == method]
        n_skipped_cells = int((summary.method.eq(method) & ~summary.ran).sum())
        if not len(cells):
            print(f"  {method:<18} NOT RUN anywhere ({n_skipped_cells} cells skipped "
                  f"as infeasible) -- no claim either way")
            continue
        confounded = cells[cells.failure_verdict != "METHOD FAILURE"]
        judged = cells[cells.failure_verdict == "METHOD FAILURE"]
        coverage = (f" over {len(judged)} of {len(cells)} cells the guard admits"
                    + (f" ({n_skipped_cells} skipped as infeasible)"
                       if n_skipped_cells else "")
                    + (f" ({len(confounded)} disqualified by the guard)"
                       if len(confounded) else ""))
        low = judged[judged.auc_causal < FAILURE_THRESHOLD]
        dud = judged[judged.auc_causal.isna()]
        if len(low) or len(dud):
            parts = []
            if len(low):
                worst = low.loc[low.auc_causal.idxmin()]
                parts.append(f"worst {worst.auc_causal:.3f} at "
                             f"{worst.system} {worst.condition}")
            if len(dud):
                parts.append(f"{len(dud)} degenerate ("
                             + ", ".join(f"{r.system} {r.condition}"
                                         for _, r in dud.iterrows()) + ")")
            print(f"  {method:<18} FAILS in {len(low) + len(dud)}{coverage} -- "
                  + "; ".join(parts))
        else:
            floor = ("n/a" if judged.auc_causal.isna().all()
                     else f"{judged.auc_causal.min():.3f}")
            note = ""
            if (cells.causal_buried > 0).any():
                worst_bury = cells.loc[cells.causal_buried.idxmax()]
                note = (f"; but buries up to {worst_bury.causal_buried:.1f} of "
                        f"{int(worst_bury.n_causal)} causal features at "
                        f"{worst_bury.system} {worst_bury.condition}")
            print(f"  {method:<18} no guard-surviving failure{coverage} "
                  f"(min {floor}){note}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outdir / "atlas_raw.csv", index=False)
    summary.to_csv(outdir / "atlas_summary.csv", index=False)
    guard.to_csv(outdir / "atlas_guard.csv", index=False)
    score_frame.to_csv(outdir / "atlas_scores.csv", index=False)
    print(f"\nwrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
