"""Classical feature-importance baselines to compare the gated network against.

A ranking produced by twenty trained networks is only interesting relative to
what a filter or a forest produces in a second, so this module provides four
cheap references: mutual information, L1-penalised logistic regression,
random-forest impurity importance and random-forest permutation importance.
They disagree with each other often enough that :func:`rank_agreement` is worth
looking at before treating any single ordering as the answer.

Two traps this module exists to avoid:

* every method here scores *columns*, but the project reports *features*, and a
  one-hot feature owns several columns.  :func:`aggregate_to_features` is the
  supported way back to per-feature numbers; comparing a raw column vector
  against :class:`~deepfeatselect.model.FeatureGate` gates would compare
  vectors of different lengths, or worse, silently line up if the dataset
  happens to have no nominal attributes.
* permutation importance measured on the rows the forest was fitted to is a
  well-known way to make high-cardinality noise look informative.
  :func:`permutation_rf` always measures on held-out rows.

Inputs are assumed to be scaled the way :func:`deepfeatselect.data.prepare`
leaves them.  Two of the four baselines (mutual information, impurity) do not
care, but coefficient magnitude from :func:`l1_logistic` is only comparable
across columns when the columns share a scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Forest size used inside permutation_rf.  Smaller than the impurity default
# because the cost there is n_columns * (n_repeats + 1) full-forest predictions
# rather than one fit, and the permutation ranking has stopped moving by 200
# trees on data of this size.
_PERM_N_ESTIMATORS = 200

# Fraction of (x, y) held out when permutation_rf is not given its own
# evaluation set.
_PERM_HOLDOUT = 0.25


def _l1_kwargs() -> dict[str, object]:
    """How to ask :class:`LogisticRegression` for an L1 penalty on this install.

    scikit-learn 1.8 deprecated ``penalty`` in favour of ``l1_ratio`` and 1.10
    removes it, but on 1.5-1.7 ``l1_ratio`` is *ignored* unless
    ``penalty="elasticnet"`` -- so hardcoding either spelling means one of a
    FutureWarning now or a silently L2-penalised fit on the older versions
    ``pyproject.toml`` still allows.  Version strings only move forward, so an
    unparseable one is assumed to be new.
    """
    try:
        major, minor = (int(p) for p in sklearn.__version__.split(".")[:2])
    except ValueError:
        return {"l1_ratio": 1.0}
    return {"l1_ratio": 1.0} if (major, minor) >= (1, 8) else {"penalty": "l1"}


def _as_xy(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to arrays and check the shapes line up."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y).reshape(-1)
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D (n_samples, n_columns), got shape {x.shape}")
    if len(x) != len(y):
        raise ValueError(f"x has {len(x)} rows but y has {len(y)}")
    return x, y


def _stratify(y: np.ndarray) -> np.ndarray | None:
    """Stratify only when every class can land on both sides of a split.

    The multiclass Cleveland target has a level with 13 examples, and a
    caller slicing it further can leave a class with one member, which
    ``train_test_split`` refuses to stratify rather than warn about.
    """
    _, counts = np.unique(y, return_counts=True)
    return y if counts.min() >= 2 else None


def _forest(n_estimators: int, seed: int) -> RandomForestClassifier:
    """A forest configured to match how the network is trained.

    ``class_weight="balanced"`` mirrors ``TrainConfig.class_weight``: without it
    the baseline is allowed to ignore the minority class that the network is
    explicitly pushed to fit, and the comparison stops being like for like.

    Left single-threaded on purpose.  On a few hundred rows ``n_jobs=-1`` loses
    to the default at fitting and loses badly inside
    :func:`permutation_importance`, where the thread pool is set up and torn
    down again for every one of the hundreds of small predictions.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        class_weight="balanced",
    )


def _normalise(scores: np.ndarray) -> np.ndarray:
    """Clip to non-negative and scale to sum to one.

    A score vector can legitimately come out all zero -- ``C=0.01`` on this data
    drops every L1 coefficient, permutation importance can find nothing -- and
    dividing by that total would put NaN into every downstream comparison.  A
    uniform vector instead says "this method separated nothing", which is what
    happened, and preserves the sum-to-one invariant callers rely on.

    Zero columns is the one input with no uniform fallback: ``1 / size`` is the
    second division by zero here and would raise before the empty result could
    be returned, so it is short-circuited rather than guarded after the fact.
    """
    scores = np.clip(np.asarray(scores, dtype=np.float64), 0.0, None)
    if scores.size == 0:
        return scores
    total = scores.sum()
    if total <= 0.0:
        return np.full(scores.shape, 1.0 / scores.size, dtype=np.float64)
    return scores / total


def mutual_information(x, y, seed: int = 0) -> np.ndarray:
    """Per-column mutual information with the target, in nats.

    Non-parametric and so the only baseline here that sees a non-monotone
    relationship, but it is univariate: it scores each column in isolation and
    therefore gives two redundant copies of the same signal the same high score,
    where the L1 and forest baselines split the credit between them.

    The estimator is k-nearest-neighbour based and adds tie-breaking noise, so
    the result moves with the seed; it is threaded through rather than left to
    the global RNG.  Values are already clipped at zero by scikit-learn.
    """
    x, y = _as_xy(x, y)
    return np.asarray(mutual_info_classif(x, y, random_state=seed), dtype=np.float64)


def l1_logistic(x, y, C: float = 0.1, seed: int = 0) -> np.ndarray:
    """Magnitude of the coefficients of an L1-penalised logistic regression.

    ``saga`` rather than ``liblinear``, which is the usual choice for L1 on
    small data but refuses three or more classes outright from scikit-learn 1.8
    -- ``saga`` is the solver that covers both the binary and the multiclass
    task without a one-vs-rest wrapper.  With more than two classes ``coef_``
    gains a class axis and the magnitudes are averaged over it, so a column
    that separates a single class from the rest still scores, in proportion to
    how many classes it helps.

    ``C`` is inverse regularisation strength: smaller means sparser.  The
    default of 0.1 is deliberately on the sparse side -- an unpenalised fit
    ranks nothing, it just assigns every column some coefficient.  Being a
    linear model this sees only linear effects, which is the point of having it
    next to the forest baselines rather than instead of them.
    """
    x, y = _as_xy(x, y)
    model = LogisticRegression(
        solver="saga",
        C=C,
        random_state=seed,
        max_iter=5000,
        **_l1_kwargs(),
    )
    model.fit(x, y)
    return np.abs(model.coef_).mean(axis=0)


def random_forest(x, y, n_estimators: int = 500, seed: int = 0) -> np.ndarray:
    """Mean impurity decrease over a random forest.

    Free once the forest is fitted, and biased in a specific direction: a
    column offering many distinct split points collects impurity decrease that a
    binary indicator cannot, so continuous columns outrank one-hot columns even
    when they carry the same information.  That bias is the reason
    :func:`permutation_rf` is also here rather than this being the only
    tree-based baseline.
    """
    x, y = _as_xy(x, y)
    forest = _forest(n_estimators, seed).fit(x, y)
    return np.asarray(forest.feature_importances_, dtype=np.float64)


def permutation_rf(
    x,
    y,
    x_eval=None,
    y_eval=None,
    n_repeats: int = 20,
    seed: int = 0,
) -> np.ndarray:
    """Drop in held-out balanced accuracy when a column is shuffled.

    Held-out is the whole point.  A fully grown forest memorises its training
    rows, so shuffling a column it memorised destroys that memorisation and
    scores highly -- which is why permutation importance computed on training
    data reliably promotes high-cardinality noise columns.  When no evaluation
    set is given this carves a stratified quarter out of ``(x, y)`` and fits on
    the rest rather than quietly reusing the training rows.

    Scoring is balanced accuracy rather than plain accuracy so that a column
    only the minority class depends on still registers.

    Negative importances -- a column whose shuffling happened to help -- are
    clipped to zero.  They are sampling noise around zero, and a negative
    "importance" has no meaning to aggregate or normalise.
    """
    x, y = _as_xy(x, y)
    if (x_eval is None) != (y_eval is None):
        raise ValueError("x_eval and y_eval must be given together, or neither")

    if x_eval is None:
        x_fit, x_eval, y_fit, y_eval = train_test_split(
            x, y, test_size=_PERM_HOLDOUT, stratify=_stratify(y), random_state=seed
        )
    else:
        x_fit, y_fit = x, y
        x_eval, y_eval = _as_xy(x_eval, y_eval)
        if x_eval.shape[1] != x.shape[1]:
            raise ValueError(
                f"x_eval has {x_eval.shape[1]} columns but x has {x.shape[1]}"
            )

    forest = _forest(_PERM_N_ESTIMATORS, seed).fit(x_fit, y_fit)
    result = permutation_importance(
        forest,
        x_eval,
        y_eval,
        scoring="balanced_accuracy",
        n_repeats=n_repeats,
        random_state=seed,
    )
    return np.clip(np.asarray(result.importances_mean, dtype=np.float64), 0.0, None)


def all_baselines(x, y, seed: int = 0) -> dict[str, np.ndarray]:
    """Run every baseline and return per-column scores normalised to sum to one.

    The four methods report in incomparable units -- nats, coefficient
    magnitudes, impurity decrease, accuracy points -- so each vector is
    converted into a share of its own total.  Shares also survive
    :func:`aggregate_to_features` unchanged: summing a partition of a vector
    that sums to one still sums to one, so the per-feature tables are directly
    comparable to the normalised gate shares from
    :func:`deepfeatselect.experiment.summarise`.

    Every method gets the same ``seed``, which makes a single call reproducible
    but does *not* make the scores independent estimates -- for a spread, call
    this with several seeds.
    """
    return {
        "mutual_information": _normalise(mutual_information(x, y, seed=seed)),
        "l1_logistic": _normalise(l1_logistic(x, y, seed=seed)),
        "random_forest": _normalise(random_forest(x, y, seed=seed)),
        "permutation_rf": _normalise(permutation_rf(x, y, seed=seed)),
    }


def aggregate_to_features(column_scores, groups, n_features: int) -> np.ndarray:
    """Sum column scores within each one-hot group, giving one score per feature.

    Sum, not mean and not max.  Every measure in this module is additive over
    columns -- impurity decrease accumulates over splits, a permutation drop is
    a loss of accuracy, an L1 coefficient contributes additively to the logit --
    so a categorical feature's contribution is *split* across its indicator
    columns rather than repeated in each of them.  Summing puts it back
    together.  A mean would divide that contribution by the number of levels and
    systematically under-rank wide features; a max would keep one level and
    throw the rest away.

    The flip side, and the reason additivity has to be checked rather than
    assumed: summing is only safe because these scores decompose.  For a
    statistic computed independently per column -- each indicator's own mutual
    information with the target, say -- the k terms are k separate
    positively-biased estimates rather than a decomposition, and summing them
    lets a wide one-hot feature look artificially important purely because it is
    wide.  That is the same inflation
    :func:`deepfeatselect.data._group_scale` corrects for on the input side, and
    it is worth remembering that :func:`mutual_information` is the one baseline
    here whose per-column values are estimates of that kind.

    ``groups`` is the length-``n_columns`` vector from
    :class:`~deepfeatselect.data.Dataset`.  Features owning no columns come back
    as zero rather than being dropped, so the result always lines up with
    ``feature_names``.
    """
    scores = np.asarray(column_scores, dtype=np.float64).reshape(-1)
    groups = np.asarray(groups, dtype=np.int64).reshape(-1)

    if scores.shape != groups.shape:
        raise ValueError(
            f"column_scores has {scores.size} entries but groups has {groups.size}; "
            "both must be one per column"
        )
    if n_features < 0:
        raise ValueError(f"n_features must be non-negative, got {n_features}")
    if groups.size and (groups.min() < 0 or groups.max() >= n_features):
        raise ValueError(
            f"groups values must be in [0, {n_features}), got "
            f"[{groups.min()}, {groups.max()}]"
        )

    return np.bincount(groups, weights=scores, minlength=n_features).astype(np.float64)


def rank_agreement(scores: dict[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise Spearman correlation between the orderings the methods produce.

    Spearman rather than overlap-at-k because it uses the whole ordering and
    needs no arbitrary cut-off, and rank-based rather than Pearson because the
    scores are on different scales even after normalisation.

    A low off-diagonal entry is the informative case: it says the methods
    disagree about the ordering, so no single ranking -- the network's included
    -- should be read as the answer on its own.

    The diagonal is set to one explicitly rather than computed.  A method that
    gave every column the same score has no ranking to correlate, so its
    correlation is undefined even against itself -- but a NaN on the diagonal
    would read as a defect in the table rather than a fact about that method.
    Its off-diagonal entries are left NaN, which is the honest answer, and are
    filled in without calling ``spearmanr`` only to keep it from warning about
    an input the caller cannot do anything about.
    """
    names = list(scores)
    vectors = [np.asarray(scores[name], dtype=np.float64).reshape(-1) for name in names]

    lengths = {v.size for v in vectors}
    if len(lengths) > 1:
        raise ValueError(f"all score vectors must have the same length, got {sorted(lengths)}")

    flat = [bool(v.size == 0 or np.ptp(v) == 0.0) for v in vectors]

    matrix = np.eye(len(names), dtype=np.float64)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if flat[i] or flat[j]:
                rho = np.nan
            else:
                rho = float(spearmanr(vectors[i], vectors[j]).statistic)
            matrix[i, j] = matrix[j, i] = rho

    return pd.DataFrame(matrix, index=names, columns=names)
