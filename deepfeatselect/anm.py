"""Pairwise causal orientation by residual-independence asymmetry (RESIT).

The one tie-breaker observational data still offers after everything else in
this project came up symmetric.  For an additive noise model ``Y = f(X) + N``
with ``N`` independent of ``X`` and ``f`` nonlinear, regressing the effect on
the cause leaves residuals independent of the input, while the reverse
regression generally does not (Hoyer et al. 2009; Peters et al. 2014).  Fit
both directions, test both residuals, and read the direction off the asymmetry.

Two regimes beyond the textbook one are handled explicitly, because the
redundancy benchmark lives in them:

* Deterministic non-invertible pairs (``proxy = cos(2*pi*u)`` with no noise)
  are orientable without any noise argument: the effect is a function of the
  cause but not conversely, so one regression is exact and the other has an
  irreducible, input-dependent residual.  The verdict says so.
* Deterministic bijective pairs are genuinely unidentifiable this way -- both
  regressions are exact and there is no residual left to interrogate.  The
  honest output is a refusal, not a guess.

Known limits, stated rather than hidden: the linear-Gaussian case is
unidentifiable (both directions admit an ANM -> "undecided"), a common cause
typically breaks independence both ways ("no ANM either direction" -- which is
the correct answer for a confounded pair, not a failure), and the permutation
test assumes exchangeable rows, so autocorrelated series should be thinned
before testing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from .netstats import residual_hsic

# Residual variance below this fraction of the target's variance counts as an
# exact functional relationship.  Set well below what a strong-but-noisy fit
# reaches (sigma = 0.05 on a unit-scale signal leaves a ratio near 3e-3).
DETERMINISTIC_RATIO = 1e-3


@dataclass(frozen=True)
class ANMResult:
    """Both directions of one pair, plus the categorical verdict.

    ``verdict`` is one of: ``independent``, ``x->y``, ``y->x``,
    ``x->y (deterministic)``, ``y->x (deterministic)``,
    ``deterministic bijection: unidentifiable``, ``undecided: both admissible``,
    ``no ANM in either direction``.
    """

    verdict: str
    p_marginal: float
    p_forward: float  # HSIC p of residual(y|x) against x; high = independent
    p_backward: float  # HSIC p of residual(x|y) against y
    residual_ratio_forward: float
    residual_ratio_backward: float
    r2_forward: float
    r2_backward: float


def _fit_residuals(inputs: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """One-dimensional regression with the estimator chosen by cross-validation.

    Candidates: kNN (flexible, handles any shape including the branch-averaged
    reverse of a non-invertible function) and low-degree polynomials.  The
    polynomials matter because kNN carries a density-dependent smoothing bias
    -- its window widens in the tails of the input distribution, so where the
    true function has curvature the bias varies *with the input*, and the test
    downstream reads that as residual dependence.  On ``x = z^2 + noise`` this
    artefact alone pushed the true causal direction below alpha.

    Selection is by CV mean-squared error, never by the independence test's
    outcome: choosing the regressor that maximises the p-value would bias the
    test toward whatever answer was wanted.
    """
    x = inputs.reshape(-1, 1)

    def make_candidates(n: int) -> dict[str, object]:
        k = int(np.clip(np.sqrt(n), 5, 50))
        candidates: dict[str, object] = {
            "knn": KNeighborsRegressor(n_neighbors=k),
            "knn_half": KNeighborsRegressor(n_neighbors=max(3, k // 2)),
            # Splines are the workhorse: piecewise-cubic fits smooth
            # non-polynomial shapes (tanh tails, one full cosine period) that a
            # global polynomial cannot, and at large n the permutation test has
            # enough power to read any leftover approximation bias as residual
            # dependence -- which is how an impoverished family turns the true
            # causal direction into a false rejection.
            "spline": make_pipeline(
                SplineTransformer(n_knots=max(8, min(25, n // 40)), degree=3),
                Ridge(alpha=1e-3),
            ),
        }
        for degree in (3, 5, 7):
            candidates[f"poly{degree}"] = make_pipeline(
                SplineTransformer(n_knots=2, degree=degree), Ridge(alpha=1e-6)
            )
        return candidates

    # Held-out MSE via a deterministic alternating split; the winner is refit
    # on everything.
    even = np.arange(len(inputs)) % 2 == 0
    odd = ~even
    scores: list[tuple[float, str]] = []
    for name in make_candidates(len(inputs)):
        mse = 0.0
        for fit_idx, val_idx in ((even, odd), (odd, even)):
            model = make_candidates(int(fit_idx.sum()))[name]
            model.fit(x[fit_idx], target[fit_idx])
            pred = model.predict(x[val_idx])
            mse += float(((target[val_idx] - pred) ** 2).mean())
        scores.append((mse, name))

    _, best_name = min(scores)
    best = make_candidates(len(inputs))[best_name]
    best.fit(x, target)
    predicted = best.predict(x)

    residuals = target - predicted
    total = np.var(target)
    r2 = 1.0 - np.var(residuals) / total if total > 0 else 0.0
    return residuals, float(r2)


def anm_orient(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
    n_permutations: int = 200,
    seed: int = 0,
) -> ANMResult:
    """Orient a single pair, or refuse with the reason encoded in the verdict."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)

    # Marginal dependence first: an independent pair fits a constant both ways
    # and would otherwise fall through to "undecided", which overstates what
    # was found.
    _, p_marginal = residual_hsic(y, x.reshape(-1, 1), n_permutations=n_permutations, seed=seed)

    r_fwd, r2_fwd = _fit_residuals(x, y)
    r_bwd, r2_bwd = _fit_residuals(y, x)
    ratio_fwd = float(np.var(r_fwd))  # inputs are standardised, so var(target)=1
    ratio_bwd = float(np.var(r_bwd))

    _, p_fwd = residual_hsic(r_fwd, x.reshape(-1, 1), n_permutations=n_permutations, seed=seed)
    _, p_bwd = residual_hsic(r_bwd, y.reshape(-1, 1), n_permutations=n_permutations, seed=seed)

    det_fwd = ratio_fwd < DETERMINISTIC_RATIO
    det_bwd = ratio_bwd < DETERMINISTIC_RATIO

    if p_marginal > alpha:
        verdict = "independent"
    elif det_fwd and det_bwd:
        verdict = "deterministic bijection: unidentifiable"
    elif det_fwd:
        verdict = "x->y (deterministic)"
    elif det_bwd:
        verdict = "y->x (deterministic)"
    elif p_fwd > alpha and p_bwd <= alpha:
        verdict = "x->y"
    elif p_bwd > alpha and p_fwd <= alpha:
        verdict = "y->x"
    elif p_fwd > alpha and p_bwd > alpha:
        verdict = "undecided: both admissible"
    else:
        verdict = "no ANM in either direction"

    return ANMResult(
        verdict=verdict,
        p_marginal=p_marginal,
        p_forward=p_fwd,
        p_backward=p_bwd,
        residual_ratio_forward=ratio_fwd,
        residual_ratio_backward=ratio_bwd,
        r2_forward=r2_fwd,
        r2_backward=r2_bwd,
    )
