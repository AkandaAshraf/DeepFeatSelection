"""Detecting when importance rankings are not identifiable.

Proposition 1 says that if a feature is a deterministic function of the others,
every risk-difference importance measure scores it exactly zero.  That result is
only actionable if the condition can be checked on data, which is what this
module does: regress each feature on the rest and see how much is left.

The point is diagnostic rather than decorative.  A high leave-one-out
predictability means any single-feature importance for that column is
uninterpretable -- not small, not noisy, but *undefined up to the choice of
representative*.  Reporting a ranking over such a set without saying so is how a
biomarker that cannot replicate gets published.

All scores are held out.  In-sample R-squared from a flexible learner is close
to one for everything and would report the whole table as redundant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

# A feature this predictable from the others contributes nothing a predictor
# could not reconstruct, so its individual importance is not identifiable.
REDUNDANT_R2 = 0.95


def _cv_r2(inputs: np.ndarray, target: np.ndarray, seed: int = 0, n_splits: int = 3) -> float:
    """Cross-validated R-squared of predicting ``target`` from ``inputs``."""
    if inputs.shape[1] == 0:
        return 0.0
    predictions = np.empty_like(target, dtype=np.float64)
    for train_idx, test_idx in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(inputs):
        model = RandomForestRegressor(
            n_estimators=120, min_samples_leaf=5, random_state=seed, n_jobs=-1
        )
        model.fit(inputs[train_idx], target[train_idx])
        predictions[test_idx] = model.predict(inputs[test_idx])
    variance = np.var(target)
    if variance <= 0.0:
        return 0.0
    return float(1.0 - np.mean((target - predictions) ** 2) / variance)


def redundancy_scores(
    x: np.ndarray,
    feature_names: list[str],
    seed: int = 0,
    threshold: float = REDUNDANT_R2,
) -> pd.DataFrame:
    """How well each feature is reconstructed from all the others.

    This is the Proposition 1 condition, estimated.  ``redundant`` marks the
    columns whose individual risk-difference importance should not be read as a
    measure of anything.
    """
    rows = []
    for j, name in enumerate(feature_names):
        others = np.delete(x, j, axis=1)
        r2 = _cv_r2(others, x[:, j], seed=seed)
        rows.append({"feature": name, "r2_from_others": r2, "redundant": r2 >= threshold})
    return pd.DataFrame(rows).sort_values("r2_from_others", ascending=False).reset_index(drop=True)


def pairwise_predictability(
    x: np.ndarray, feature_names: list[str], seed: int = 0
) -> pd.DataFrame:
    """Directional matrix: entry [i, j] is the R-squared of predicting j from i.

    Kept directional because the asymmetry is informative.  A squared readout is
    predictable from the pathway it measures while the pathway is not
    recoverable from it, and that gap is exactly the non-invertibility that
    residual-asymmetry methods can later exploit to orient the pair.
    """
    n = len(feature_names)
    matrix = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = _cv_r2(x[:, [i]], x[:, j], seed=seed)
    return pd.DataFrame(matrix, index=feature_names, columns=feature_names)


def minimal_removal_set(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    min_drop: float = 0.10,
    max_size: int = 4,
    seed: int = 0,
    max_evals: int = 300,
    beam_width: int = 8,
) -> tuple[list[str], float]:
    """Smallest set of features whose joint removal actually destroys the signal.

    Proposition 1 says single-feature ablation is uninformative under redundancy.
    The constructive complement is to keep removing until the information is
    genuinely gone: the resulting set is the coarsest grouping at which ablation
    importance becomes meaningful again, and its size is the number of distinct
    ways the target is reachable.

    Searched by subset size rather than greedily.  Greedy fails here for exactly
    the reason the function is needed: under redundancy every *single* removal
    scores about zero, so the first greedy step has no signal and picks on noise,
    after which it is committed to the wrong branch.  The search therefore sweeps
    all subsets of size 1, then 2, and so on, returning the first size at which
    some subset destroys the signal -- which is minimal by construction.

    Exhaustive while the number of subsets stays under ``max_evals``, then a beam
    over the best partial subsets, since the exact search is exponential.

    Returns the set and the predictive drop achieved.  An empty set means no
    removal of up to ``max_size`` features reached ``min_drop``.
    """
    from itertools import combinations

    target = y.astype(np.float64)
    all_cols = list(range(x.shape[1]))
    base = _cv_r2(x, target, seed=seed)

    def drop_for(subset: tuple[int, ...]) -> float:
        keep = [c for c in all_cols if c not in subset]
        if not keep:
            return base
        return base - _cv_r2(x[:, keep], target, seed=seed)

    beam: list[tuple[int, ...]] = [()]
    best_overall: tuple[float, tuple[int, ...]] = (0.0, ())

    for size in range(1, max_size + 1):
        n_exhaustive = len(list(combinations(all_cols, size))) if size <= 3 else max_evals + 1
        if n_exhaustive <= max_evals:
            candidates = list(combinations(all_cols, size))
        else:
            # Extend the surviving partial subsets rather than starting over.
            candidates = sorted({
                tuple(sorted(prefix + (j,)))
                for prefix in beam
                for j in all_cols
                if j not in prefix
            })

        scored = sorted(((drop_for(c), c) for c in candidates), reverse=True)
        if not scored:
            break
        if scored[0][0] > best_overall[0]:
            best_overall = scored[0]
        if scored[0][0] >= min_drop:
            return [feature_names[j] for j in scored[0][1]], float(scored[0][0])
        beam = [c for _, c in scored[:beam_width]]

    return [feature_names[j] for j in best_overall[1]], float(best_overall[0])


def group_loco(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    groups: list[list[str]],
    seed: int = 0,
) -> pd.DataFrame:
    """Ablation importance for whole redundancy-closed groups rather than columns.

    The repair for Proposition 1: a group that contains every route to a piece
    of information does have a well-defined importance, even though each of its
    members individually has none.
    """
    index = {name: i for i, name in enumerate(feature_names)}
    base = _cv_r2(x, y.astype(np.float64), seed=seed)

    rows = []
    for group in groups:
        drop_cols = {index[name] for name in group if name in index}
        keep = [c for c in range(x.shape[1]) if c not in drop_cols]
        rows.append({
            "group": " + ".join(sorted(group)),
            "size": len(group),
            "r2_drop": base - _cv_r2(x[:, keep], y.astype(np.float64), seed=seed),
        })
    return pd.DataFrame(rows).sort_values("r2_drop", ascending=False).reset_index(drop=True)


def equivalence_classes(
    x: np.ndarray,
    feature_names: list[str],
    seed: int = 0,
    threshold: float = REDUNDANT_R2,
) -> list[list[str]]:
    """Group features that determine each other in both directions.

    Members of a class are interchangeable as predictors, so a ranking within a
    class carries no information about which one the biology runs through.
    Only mutual predictability counts: a one-way link means one column is a
    coarsening of the other, which is a different relationship and must not be
    collapsed into the same class.
    """
    matrix = pairwise_predictability(x, feature_names, seed=seed).to_numpy()
    n = len(feature_names)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i, j] >= threshold and matrix[j, i] >= threshold:
                parent[find(i)] = find(j)

    groups: dict[int, list[str]] = {}
    for i, name in enumerate(feature_names):
        groups.setdefault(find(i), []).append(name)
    return [g for g in groups.values() if len(g) > 1]
