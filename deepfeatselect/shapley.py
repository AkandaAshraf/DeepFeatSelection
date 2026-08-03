"""Shapley-style global importance, the method built for redundant features.

Proposition 1 kills every importance measure defined as a single risk
difference: if ``X_j`` is a deterministic function of the rest then
``R*(N \\ {j}) = R*(N)`` and leave-one-out importance is exactly zero.  But that
difference is only *one* of the coalitions a Shapley value averages over --
specifically the ``S = N \\ {j}`` term -- and the others need not vanish.  For a
feature that alone determines the target, the ``S = empty`` term is as large as
the signal itself.

So Shapley importance is expected to survive where leave-one-out does not, and
this module measures whether it does.  That matters for honesty about scope: the
blindness result is about risk-difference importance, not about every importance
measure, and SAGE (Covert et al., 2020) is the natural counterexample to check
rather than to assume away.

The value function here is the *retrained* one, ``v(S) = R^2`` of a model fitted
on ``X_S`` alone.  SAGE as published marginalises absent features out of a
single fixed model, which measures how much that model relies on a feature; the
retrained variant measures how much the data supports it.  The second is the one
comparable to leave-one-out refitting, and the one Proposition 1 speaks about.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd

from .redundancy import _cv_r2

# Exact enumeration is 2^d value evaluations; beyond this, sample permutations.
EXACT_MAX_SUBSETS = 1024


def _value(x: np.ndarray, y: np.ndarray, cols: tuple[int, ...],
           cache: dict[frozenset[int], float], seed: int) -> float:
    """Cached ``v(S)``, with ``v(empty) = 0`` as the no-information baseline."""
    key = frozenset(cols)
    if key not in cache:
        cache[key] = 0.0 if not cols else _cv_r2(x[:, sorted(cols)], y, seed=seed)
    return cache[key]


def shapley_importance(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    seed: int = 0,
    n_permutations: int = 40,
) -> pd.DataFrame:
    """Global Shapley importance over a retrained value function.

    Exact when the feature count allows enumerating every coalition, otherwise
    estimated by sampling permutations, which is the standard unbiased estimator
    and reuses the cache across draws.
    """
    d = len(feature_names)
    target = y.astype(np.float64)
    cache: dict[frozenset[int], float] = {}
    phi = np.zeros(d)

    if 2**d <= EXACT_MAX_SUBSETS:
        for j in range(d):
            others = [i for i in range(d) if i != j]
            for size in range(len(others) + 1):
                # Shapley weight: the probability that a uniformly random
                # permutation places exactly this coalition before j.
                weight = (math.factorial(size) * math.factorial(d - size - 1)
                          / math.factorial(d))
                for subset in combinations(others, size):
                    with_j = tuple(sorted(subset + (j,)))
                    phi[j] += weight * (
                        _value(x, target, with_j, cache, seed)
                        - _value(x, target, subset, cache, seed)
                    )
        method = "exact"
    else:
        rng = np.random.default_rng(seed)
        for _ in range(n_permutations):
            order = rng.permutation(d)
            running: list[int] = []
            previous = 0.0
            for j in order:
                running.append(int(j))
                current = _value(x, target, tuple(sorted(running)), cache, seed)
                phi[j] += current - previous
                previous = current
        phi /= n_permutations
        method = f"sampled({n_permutations})"

    return (
        pd.DataFrame({"feature": feature_names, "sage": phi, "method": method})
        .sort_values("sage", ascending=False)
        .reset_index(drop=True)
    )


def shapley_noise_floor(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    seed: int = 0,
    n_nulls: int = 2,
) -> float:
    """Resolution limit of the estimator, from injected known-null columns.

    The value function is a cross-validated forest, so ``v(S)`` is estimated
    rather than exact: adding even a constant column perturbs the internal
    feature sampling and moves the score slightly.  Those perturbations
    propagate into every coalition and give each feature an irreducible
    uncertainty, so a Shapley value smaller than this floor carries no
    information regardless of the axioms it satisfies.

    Estimated as the largest absolute value assigned to columns of independent
    noise, which are null players by construction.
    """
    rng = np.random.default_rng(seed + 977)
    augmented = np.hstack([x, rng.standard_normal((len(x), n_nulls))])
    names = list(feature_names) + [f"__null_{i}" for i in range(n_nulls)]
    result = shapley_importance(augmented, y, names, seed=seed).set_index("feature")
    return float(result.loc[[n for n in names if n.startswith("__null_")], "sage"].abs().max())


def loco_for_comparison(
    x: np.ndarray, y: np.ndarray, feature_names: list[str], seed: int = 0
) -> pd.DataFrame:
    """Single-coalition importance, i.e. the ``S = N \\ {j}`` term on its own."""
    target = y.astype(np.float64)
    base = _cv_r2(x, target, seed=seed)
    rows = []
    for j, name in enumerate(feature_names):
        keep = [c for c in range(x.shape[1]) if c != j]
        rows.append({"feature": name, "loco": base - _cv_r2(x[:, keep], target, seed=seed)})
    return pd.DataFrame(rows)
