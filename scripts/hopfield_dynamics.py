"""Modern Hopfield retrieval on trajectories, where its energy has a referent.

An earlier experiment in this project asked Hopfield to regress column j of an
i.i.d. tabular design from the remaining columns.  On exchangeable rows that is
Nadaraya-Watson kernel regression with a softmax kernel and nothing more: there
is no attractor, so "stored pattern" means "another random draw" and the energy
of a query is a statement about sampling density in feature space, not about
retrieval.  Everything distinctive about the operator was discarded before the
first number was computed.

On a trajectory the objects mean what the theory says they mean.  Stored
patterns are states on an attractor, retrieval is the reconstruction of a
neighbouring state, and energy measures whether a query lies on the sampled part
of that attractor.  Two experiments follow from that.

EXPERIMENT 1 -- SWAP THE OPERATOR INSIDE CCM.
CCM's inner loop is simplex projection: take the E+1 nearest neighbours of a
query on the source manifold and return an exponentially-weighted average of the
contemporaneous target values.  Modern Hopfield retrieval has the same shape with
a softmax over ALL stored patterns in place of the hard k-NN truncation.  So this
holds the embedding, the library draws, the Theiler window, the prediction set
and the skill statistic fixed, and changes only the weighting operator.  Both
methods see bit-identical library subsets at every size, so the difference
reported is the operator's and nothing else.

Two similarity kernels are run, and the distinction is load-bearing:

* ``dot``  -- ``s(q, m) = q . m``, literally Ramsauer et al. (2020) and literally
  what :func:`deepfeatselect.probe.hopfield_energy` scores.
* ``dist`` -- ``s(q, m) = -||q - m||^2 / 2``, a Gaussian kernel on the manifold.

They are NOT the same operator.  Since ``||q - m||^2 = ||q||^2 - 2 q.m + ||m||^2``
and the ``||q||^2`` term cancels in a softmax, the distance kernel equals the dot
kernel with a per-memory bias of ``-||m||^2 / 2``.  The two coincide only when
every stored pattern has the same norm, which no delay embedding of a real
trajectory satisfies.  Only ``dist`` orders memories the way a k-NN search does,
so only ``dist`` can converge to a neighbour method as beta grows; ``dot``
converges to maximum-inner-product retrieval, which prefers large-norm states
regardless of proximity.  Both are reported.

WHAT THE OPERATOR SWAP ACTUALLY CHANGES -- TWO CONTROLS THAT DECIDE IT.
It is not enough to show that Hopfield behaves differently from simplex; the
question is whether the difference is the *softmax over the whole bank* or
merely *averaging more than one neighbour*.  Two controls separate those, and
without them the mechanism cannot be read off the result:

* ``knn8`` -- an unweighted mean of the eight nearest admissible neighbours.
  No softmax, no kernel, no energy.  If this reproduces whatever Hopfield does,
  the effect is not the operator.
* ``hopfield-dist-top32`` -- the identical Gaussian softmax, truncated to the
  32 nearest admissible memories.  If this matches untruncated Hopfield, then
  "over the WHOLE memory bank" is doing no work and the distinction the module
  is built on is cosmetic.

The second control also explains why ``simplex`` is the odd one out rather than
Hopfield being special: ``ccm._simplex_predict`` weights by ``exp(-d_i/d_1)``
floored at ``_MIN_WEIGHT = 1e-6``, so a neighbour past about ``14 * d_1``
contributes nothing at all.  Raising ``n_neighbours`` in that routine therefore
does NOT give a wider average -- k=8 and k=128 are the same estimator -- and
simplex sits much closer to 1-NN than its nominal k=E+1 suggests.

THE HIGH-BETA LIMIT, STATED CORRECTLY.  The premise this experiment was
commissioned under -- "high beta approaches hard nearest neighbour, so Hopfield
converges to simplex projection" -- is right in its first clause and off by one
step in its second.  Simplex projection is not 1-NN.  It averages E+1 neighbours
with weights ``exp(-d_i / d_1)`` normalised to sum to one, and the nearest
neighbour never takes more than a bounded share of that mass.  A softmax with
beta -> infinity puts ALL mass on one memory.  So the true limit of ``dist``
Hopfield is 1-NN cross mapping, not simplex, and this script verifies convergence
against 1-NN -- computed here with the same code path as simplex, at k=1 -- and
reports the residual 1-NN-versus-simplex gap separately.  Checking against
simplex directly would have declared a correct implementation broken.

EXPERIMENT 2 -- ENERGY AS A PER-PREDICTION CONFIDENCE SIGNAL.
Simplex has no analogue of the energy, and this is the only part of the exercise
where Hopfield can offer something CCM structurally cannot.  Energy is low when a
query sits near stored patterns and high when it is extrapolating off the sampled
attractor, so it should predict cross-map error per query with no ground truth.
CCM's known failure mode is a confident skill reported from a library that does
not cover the attractor, and a signal that flags exactly those queries would
address it.

The honest version of that claim needs a control, so one is included: the
distance from the query to its nearest admissible neighbour.  Simplex already
computes that quantity on its way to a prediction -- it is the ``d_1`` in the
weights.  If ``d_1`` ranks queries by error as well as the energy does, then the
energy is not offering anything the incumbent lacks, and the correct conclusion
is that CCM was never missing a confidence signal, only failing to report one.

TEMPORAL EXCLUSION -- MATCHED TO ccm.py, NOT CHOSEN HERE.
Consecutive points on a trajectory are autocorrelated, so a query that retrieves
its own temporal neighbours is being scored on interpolation rather than on
cross mapping.  ``ccm.py`` implements a Theiler window as ``exclusion_radius``
and defaults it to 0, where a radius of 0 still excludes the query itself.
``scripts/benchmark_causal.py`` is this project's settled configuration and uses
radius 0 with E=3 for the logistic maps and radius 5 with E=7 for Rossler-Lorenz
at stride 10.  Those are the values used here, applied identically to both
operators, and the Hopfield memory bank has the query's whole window removed
rather than merely down-weighted.  Nothing about the exclusion is tuned in
Hopfield's favour; a second pass at a wider radius is reported as a robustness
check.

BETA SELECTION -- FIXED A PRIORI, NEVER ON THE CROSS MAP.
Retrieval sharpness depends entirely on beta and one arbitrary value would mean
nothing, so beta is swept over four orders of magnitude.  For the headline
comparison a single beta must be named, and picking the one that maximises the
cross-map skill would be choosing a statistic to maximise the reported metric.
So the headline beta is the one that maximises the source manifold's own
one-step SELF-forecast skill -- a quantity computed without any reference to the
target series or to the ground-truth arrow, and the same quantity that satisfies
this project's rule about never reporting a detection score without the fitted
model's held-out performance beside it.  The cross-map-maximising beta is also
reported, labelled as the oracle it is.

Series are standardised before embedding, so a one-step self-forecast MSE near
1.0 means the manifold predicts nothing and every cross-map number on that row is
noise.  Such rows are flagged in the output rather than quietly averaged in.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Private imports, deliberately.  _simplex_predict is the only way to get
# (actual, predicted) pairs out of the incumbent -- simplex_cross_map returns a
# correlation and throws the predictions away, and this script needs the
# residuals for the MSE guard and the per-query error of Experiment 2.
# _default_lib_sizes and _pearson are imported so that the library grid and the
# skill statistic are bit-identical to CCM's rather than a reimplementation that
# happens to agree.
from deepfeatselect.ccm import (
    DirectionResult,
    _default_lib_sizes,
    _pearson,
    _simplex_predict,
    simplex_cross_map,
    time_delay_embed,
)
from deepfeatselect.probe import hopfield_energy
from deepfeatselect.synthetic import (
    coupled_logistic,
    independent_logistic,
    rossler_lorenz,
)

# Kernels swept over four orders of magnitude.  The bottom of the range averages
# the whole memory bank into the global mean and the top is single-memory
# retrieval; the interesting behaviour is somewhere between and is not knowable
# in advance.
BETAS: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)

# Ladder for the beta -> infinity check, continued well past the sweep so the
# approach is visible as a trend rather than asserted from one point. It has to
# reach this far: at beta=1e6 nineteen queries out of a thousand still disagree
# with 1-NN, and every one of them is a genuine near-tie -- two admissible
# memories 1.6e-7 apart in squared distance, which needs beta ~ 1e8 before the
# softmax commits to one of them. Agreement is exact by 1e10, and reading the
# residual at 1e6 as a bug would have been wrong.
BETA_LADDER: tuple[float, ...] = (1e3, 1e4, 1e5, 1e6, 1e8, 1e10)

KERNELS: tuple[str, ...] = ("dist", "dot")

# Controls for the mechanism claim, not results in their own right.  KNN_MEAN_K
# is an unweighted average over that many nearest admissible neighbours;
# TOP_K_CONTROL truncates the Gaussian softmax to that many memories.  Both are
# below the smallest library size used (10) only where noted, and any k at or
# above it is skipped rather than silently degraded to "average everything".
KNN_MEAN_K: tuple[int, ...] = (4, 8)
TOP_K_CONTROL: int = 32


def standardise(series: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance.

    Not cosmetic.  It fixes the scale that beta is measured against, so a beta
    sweep means the same thing across systems whose raw amplitudes differ by two
    orders of magnitude, and it makes a one-step forecast MSE of 1.0 the exact
    score of predicting the mean -- the reference this project's first standing
    rule is stated against.
    """
    series = np.asarray(series, dtype=np.float64).reshape(-1)
    return (series - series.mean()) / (series.std() + 1e-12)


@dataclass(frozen=True)
class Manifold:
    """A delay embedding plus every pairwise quantity the retrieval needs.

    Both similarity matrices are formed once per manifold and then sliced.  The
    sweep evaluates tens of thousands of (library, beta) combinations and
    recomputing an n-by-n Gram matrix inside that loop would dominate the
    runtime; beta only rescales an existing matrix and a library subset is only a
    column selection.
    """

    points: np.ndarray  # (n, E) delay coordinates
    times: np.ndarray  # (n,) original time index of each row
    dot: np.ndarray  # (n, n) inner products
    dist2: np.ndarray  # (n, n) squared euclidean distances
    admissible: np.ndarray  # (n, n) bool, True where a memory may serve a query
    exclusion_radius: int

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    @property
    def embedding_dim(self) -> int:
        return self.points.shape[1]

    def similarity(self, kernel: str) -> np.ndarray:
        if kernel == "dot":
            return self.dot
        if kernel == "dist":
            # exp(beta * -d^2/2) is a Gaussian kernel of bandwidth 1/sqrt(beta).
            # The halving is what makes this the dot kernel offset by exactly
            # -||m||^2/2, which is the statement the module docstring makes.
            return -0.5 * self.dist2
        raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")


def build_manifold(series: np.ndarray, E: int, tau: int, exclusion_radius: int) -> Manifold:
    """Embed a series and precompute its retrieval geometry."""
    points, times = time_delay_embed(series, E, tau)
    dot = points @ points.T
    sq = np.einsum("ij,ij->i", points, points)
    # Rounding can make a self-distance a small negative number, which becomes a
    # nan under any later sqrt and a spurious largest similarity under none.
    dist2 = np.maximum(sq[:, None] - 2.0 * dot + sq[None, :], 0.0)
    admissible = np.abs(times[:, None] - times[None, :]) > exclusion_radius
    return Manifold(
        points=points,
        times=times,
        dot=dot,
        dist2=dist2,
        admissible=admissible,
        exclusion_radius=exclusion_radius,
    )


def library_rows(n_points: int, lib_size: int | None, seed: int) -> np.ndarray:
    """Library draw reproducing ``simplex_cross_map``'s own sampling exactly.

    Same generator, same call, same argument order, so passing ``seed`` to
    ``simplex_cross_map`` and this function to the Hopfield path puts both
    operators on the identical subset of stored states.  Any difference in skill
    is then the operator's and not the draw's.
    """
    if lib_size is None or lib_size >= n_points:
        return np.arange(n_points)
    return np.random.default_rng(seed).choice(n_points, size=lib_size, replace=False)


def hopfield_predict(
    manifold: Manifold,
    kernel: str,
    values: np.ndarray,
    lib_rows: np.ndarray,
    pred_rows: np.ndarray,
    beta: float,
    top_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Softmax retrieval over the whole memory bank.

    Signature mirrors :func:`deepfeatselect.ccm._simplex_predict` so the two are
    drop-in substitutes: ``values[i]`` is whatever row ``i`` of the manifold is
    meant to predict, which makes the same function serve the cross map (target
    series, contemporaneous) and the self-forecast (own series, tp ahead).

    ``top_k`` restricts each query's softmax to its ``top_k`` most similar
    admissible memories.  ``None`` is the operator as Ramsauer et al. define it,
    over everything stored.  A finite value exists only as a control: if
    truncating changes nothing, then "over the whole bank" is not the property
    doing the work, and no conclusion may be attributed to it.

    Returns ``(actual, predicted)`` over the prediction rows that retained at
    least one admissible memory, dropping the rest exactly as the simplex path
    drops rows with too few admissible neighbours.
    """
    full = manifold.n_points
    whole = len(pred_rows) == full and len(lib_rows) == full
    if whole:
        # np.ix_ on the full index set copies two n-by-n matrices for nothing.
        sim = manifold.similarity(kernel)
        mask = manifold.admissible
    else:
        sim = manifold.similarity(kernel)[np.ix_(pred_rows, lib_rows)]
        mask = manifold.admissible[np.ix_(pred_rows, lib_rows)]

    if top_k is not None and top_k < sim.shape[1]:
        # Rank on the masked similarity so an excluded memory can never occupy a
        # slot; -inf sorts last. A row with fewer than top_k admissible memories
        # keeps only its admissible ones, which the mask below re-imposes.
        ranked = np.where(mask, sim, -np.inf)
        keep_idx = np.argpartition(-ranked, top_k - 1, axis=1)[:, :top_k]
        trunc = np.zeros_like(mask)
        np.put_along_axis(trunc, keep_idx, True, axis=1)
        mask = mask & trunc

    # Row-max subtraction before the exponential, with the excluded entries sent
    # to -inf first so they cannot supply the max. A row whose memories are all
    # excluded has max -inf; it is carried through with a finite placeholder and
    # dropped afterwards, because -inf minus -inf is nan and would silently
    # poison the weights instead of failing.
    z = np.where(mask, beta * sim, -np.inf)
    row_max = z.max(axis=1, keepdims=True)
    usable = np.isfinite(row_max).reshape(-1)
    weights = np.exp(z - np.where(np.isfinite(row_max), row_max, 0.0))
    weights[~mask] = 0.0

    total = weights.sum(axis=1)
    # Underflow: at very large beta every weight in a row can round to zero
    # except the maximiser, which is exp(0) = 1, so a zero total means the row is
    # degenerate rather than merely sharp.
    usable &= total > 0.0
    if not usable.any():
        return np.empty(0), np.empty(0)

    predicted = (weights[usable] @ values[lib_rows]) / total[usable]
    return values[pred_rows][usable], predicted


def hopfield_cross_map(
    manifold: Manifold,
    target: np.ndarray,
    kernel: str,
    beta: float,
    lib_rows: np.ndarray,
    top_k: int | None = None,
) -> float:
    """Cross-map skill of a Hopfield retrieval, scored exactly as CCM scores simplex.

    "SOURCE xmap TARGET": memories are states on the source manifold, the values
    retrieved are contemporaneous target values, and convergence of this skill is
    evidence that TARGET causes SOURCE.  The direction convention is ccm.py's and
    is not restated here; see that module.
    """
    actual, predicted = hopfield_predict(
        manifold=manifold,
        kernel=kernel,
        values=target[manifold.times],
        lib_rows=lib_rows,
        pred_rows=np.arange(manifold.n_points),
        beta=beta,
        top_k=top_k,
    )
    if len(actual) < 3:
        return float("nan")
    return _pearson(actual, predicted)


def knn_cross_map(manifold: Manifold, target: np.ndarray, k: int, lib_rows: np.ndarray) -> float:
    """Cross-map skill with a hard k-neighbour truncation.

    At ``k = E + 1`` this is simplex projection and agrees with
    ``simplex_cross_map`` to the last bit -- it is the same routine underneath.
    At ``k = 1`` it is the operator a softmax converges to as beta grows, which
    is the reference the high-beta check is measured against.

    Note what this is NOT: because ``_simplex_predict`` floors its
    ``exp(-d_i/d_1)`` weights at ``1e-6``, raising ``k`` here does not widen the
    average.  Use :func:`knn_mean_cross_map` for that.
    """
    actual, predicted = _simplex_predict(
        manifold=manifold.points,
        times=manifold.times,
        values=target[manifold.times],
        lib_rows=lib_rows,
        pred_rows=np.arange(manifold.n_points),
        n_neighbours=k,
        exclusion_radius=manifold.exclusion_radius,
    )
    if len(actual) < 3:
        return float("nan")
    return _pearson(actual, predicted)


def knn_mean_cross_map(
    manifold: Manifold, target: np.ndarray, k: int, lib_rows: np.ndarray
) -> float:
    """Cross-map skill from an UNWEIGHTED mean of the k nearest admissible memories.

    The control that decides whether anything reported about Hopfield is about
    Hopfield.  It has no kernel, no temperature and no energy; the only thing it
    shares with the softmax operator is that it averages more than one or two
    neighbours.  Rows with fewer than k admissible library points are dropped,
    matching how the simplex path drops under-supplied rows rather than filling
    them in.
    """
    values = target[manifold.times]
    d2 = manifold.dist2[:, lib_rows].copy()
    d2[~manifold.admissible[:, lib_rows]] = np.inf
    if k > d2.shape[1]:
        return float("nan")
    idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
    usable = np.isfinite(np.take_along_axis(d2, idx, axis=1)).all(axis=1)
    if usable.sum() < 3:
        return float("nan")
    predicted = values[lib_rows][idx[usable]].mean(axis=1)
    return _pearson(values[usable], predicted)


def self_forecast(
    manifold: Manifold,
    series: np.ndarray,
    predictor: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    tp: int = 1,
) -> tuple[float, float]:
    """One-step-ahead forecast of the manifold's OWN series, leave-one-out.

    This is the held-out performance that has to sit beside every detection
    number in this project.  A cross-map skill computed from a manifold that
    cannot forecast its own next value is a correlation between two failures.
    The series is standardised, so an MSE at or above 1.0 is the score of
    predicting the mean and marks exactly that case.

    Returns ``(rho, mse)``.
    """
    keep = manifold.times + tp < len(series)
    rows = np.flatnonzero(keep)
    if len(rows) < 3:
        return float("nan"), float("nan")

    values = np.full(manifold.n_points, np.nan)
    values[rows] = series[manifold.times[rows] + tp]
    actual, predicted = predictor(rows, values)
    if len(actual) < 3:
        return float("nan"), float("nan")
    return _pearson(actual, predicted), float(np.mean((actual - predicted) ** 2))


def simplex_self_forecast(
    manifold: Manifold, series: np.ndarray, n_neighbours: int | None = None
) -> tuple[float, float]:
    """Self-forecast of the k-neighbour path, at k=E+1 by default.

    ``n_neighbours`` is explicit because the protocol requires each method's own
    held-out score beside its own detection score.  Reporting the simplex
    self-forecast on a 1-NN row would be another model's number.
    """
    k = manifold.embedding_dim + 1 if n_neighbours is None else n_neighbours

    def predict(rows: np.ndarray, values: np.ndarray):
        return _simplex_predict(
            manifold=manifold.points,
            times=manifold.times,
            values=values,
            lib_rows=rows,
            pred_rows=rows,
            n_neighbours=k,
            exclusion_radius=manifold.exclusion_radius,
        )

    return self_forecast(manifold, series, predict)


def knn_mean_self_forecast(
    manifold: Manifold, series: np.ndarray, k: int
) -> tuple[float, float]:
    """Self-forecast of the unweighted k-NN mean control, on its own terms."""

    def predict(rows: np.ndarray, values: np.ndarray):
        d2 = manifold.dist2[np.ix_(rows, rows)].copy()
        d2[~manifold.admissible[np.ix_(rows, rows)]] = np.inf
        kk = min(k, d2.shape[1])
        idx = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
        usable = np.isfinite(np.take_along_axis(d2, idx, axis=1)).all(axis=1)
        if usable.sum() < 3:
            return np.empty(0), np.empty(0)
        return values[rows][usable], values[rows][idx[usable]].mean(axis=1)

    return self_forecast(manifold, series, predict)


def hopfield_self_forecast(
    manifold: Manifold, series: np.ndarray, kernel: str, beta: float, top_k: int | None = None
) -> tuple[float, float]:
    def predict(rows: np.ndarray, values: np.ndarray):
        return hopfield_predict(
            manifold=manifold,
            kernel=kernel,
            values=values,
            lib_rows=rows,
            pred_rows=rows,
            beta=beta,
            top_k=top_k,
        )

    return self_forecast(manifold, series, predict)


def direction_result(
    cause: str, effect: str, lib_sizes: np.ndarray, samples: np.ndarray
) -> DirectionResult:
    """Wrap a skill sweep in CCM's own result type.

    Reusing :class:`DirectionResult` rather than reimplementing a convergence
    rule means Hopfield is judged convergent by the identical three-part test
    CCM applies to itself -- high rho at the largest library, a delta above the
    floor, and a bootstrap interval on that delta excluding zero.  A rule written
    afresh here could not be trusted to be the same rule.
    """
    return DirectionResult(
        cause=cause,
        effect=effect,
        xmap=f"{effect} xmap {cause}",
        lib_sizes=lib_sizes,
        rho=np.nanmean(samples, axis=1),
        rho_ci_low=np.nanpercentile(samples, 2.5, axis=1),
        rho_ci_high=np.nanpercentile(samples, 97.5, axis=1),
        rho_samples=samples,
    )


def sweep_library(
    manifold: Manifold,
    target: np.ndarray,
    lib_sizes: np.ndarray,
    n_bootstrap: int,
    seed: int,
    scorer: Callable[[np.ndarray], float],
) -> np.ndarray:
    """Skill at each library size, ``n_bootstrap`` draws each.

    The draw seeds are a deterministic function of (size, replicate) alone, so
    every operator scored through this function sees the same libraries in the
    same order regardless of how many operators ran before it.  At the full
    library every replicate would be the identical draw, so it is evaluated once
    and broadcast -- the same shortcut ``_cross_map_sweep`` takes, and it keeps
    the zero spread at that end honest rather than merely small.
    """
    samples = np.empty((len(lib_sizes), n_bootstrap), dtype=np.float64)
    for i, lib_size in enumerate(lib_sizes):
        if lib_size >= manifold.n_points:
            samples[i, :] = scorer(np.arange(manifold.n_points))
            continue
        for b in range(n_bootstrap):
            rows = library_rows(manifold.n_points, int(lib_size), library_seed(seed, lib_size, b))
            samples[i, b] = scorer(rows)
    return samples


def library_seed(seed: int, lib_size: int, b: int) -> int:
    """Draw seed determined by (run seed, size, replicate) and nothing else.

    Deliberately not a running generator: a shared stream would make a library
    depend on how many operators had drawn before it, and the whole comparison
    rests on both operators seeing the same subsets.
    """
    return int(np.random.SeedSequence([seed, int(lib_size), b]).generate_state(1)[0])


# --------------------------------------------------------------------------
# Experiment 2: energy against error
# --------------------------------------------------------------------------


def softmin_energy(dist2_row: np.ndarray, beta: float, n_memories: int) -> float:
    """Distance-kernel analogue of the Hopfield energy.

    ``hopfield_energy`` implements Ramsauer's equation (2), which is built on the
    inner-product similarity; there is no published counterpart for the Gaussian
    kernel, so this is the matching construction rather than a citation:
    ``-lse(beta, -d^2/2)`` plus the same ``log(n)/beta`` term, which tends to
    ``min_i d_i^2 / 2`` as beta grows.  Low where the query sits inside the
    sampled attractor, high where it does not, which is the only property
    Experiment 2 asks of it.

    READ THIS BEFORE COMPARING IT TO ``nn_dist``.  That limit is exactly
    ``0.5 * nn_dist^2``, a strictly increasing function of the control it is
    being scored against, so at large beta the two have identical ranks and any
    Spearman comparison between them is an identity rather than a measurement.
    Measured on coupled_logistic, ``spearman(softmin_energy, nn_dist)`` runs
    +0.06 at beta=1, +0.84 at beta=1e3, +1.000 at beta=1e6.  A "tie" between the
    energy and the baseline at high beta is therefore arithmetic; only the
    low-beta rows, where the two statistics still differ, carry information.
    """
    z = -0.5 * beta * dist2_row
    top = z.max()
    lse = (top + np.log(np.exp(z - top).sum())) / beta
    return float(-lse + np.log(n_memories) / beta)


def energy_versus_error(
    manifold: Manifold,
    target: np.ndarray,
    kernel: str,
    beta: float,
    max_queries: int,
    seed: int,
) -> dict[str, float]:
    """Per-query energy against per-query cross-map error, at the full library.

    Each query is scored against a memory bank with its own Theiler window
    removed -- the same bank that produced its prediction.  Leaving the query in
    would make every energy trivially low and the correlation meaningless, and
    leaving its temporal neighbours in would make the energy a measure of local
    sampling rate rather than of attractor coverage.

    ORIENTATION IS FIXED BEFORE THE RUN.  Energy is high where the query is
    extrapolating, so the prediction is a POSITIVE correlation with absolute
    error.  A negative correlation is a failed prediction and is reported as a
    negative number, not folded into a magnitude.

    The two distance columns are controls, not results.  ``nn_dist`` is the
    distance to the nearest admissible neighbour -- the ``d_1`` simplex already
    computes on its way to a prediction -- and ``knn_dist`` is the mean over the
    E+1 neighbours it actually uses.  If either ranks queries as well as the
    energy does, the energy is not information CCM lacks.
    """
    values = target[manifold.times]
    all_rows = np.arange(manifold.n_points)

    rng = np.random.default_rng(seed)
    if manifold.n_points > max_queries:
        query_rows = np.sort(rng.choice(manifold.n_points, size=max_queries, replace=False))
    else:
        query_rows = all_rows

    sim_full = manifold.similarity(kernel)
    errors = np.full(len(query_rows), np.nan)
    energy = np.full(len(query_rows), np.nan)
    nn_dist = np.full(len(query_rows), np.nan)
    knn_dist = np.full(len(query_rows), np.nan)
    k = manifold.embedding_dim + 1

    for i, row in enumerate(query_rows):
        bank = np.flatnonzero(manifold.admissible[row])
        if len(bank) < k:
            continue

        # Retrieval, using the same bank the energy is computed against.
        z = beta * sim_full[row, bank]
        z = z - z.max()
        w = np.exp(z)
        total = w.sum()
        if not np.isfinite(total) or total <= 0.0:
            continue
        errors[i] = abs(values[row] - float(w @ values[bank]) / total)

        if kernel == "dot":
            # The project's own function, unmodified, on the query's own bank.
            energy[i] = float(hopfield_energy(manifold.points[row], manifold.points[bank], beta)[0])
        else:
            energy[i] = softmin_energy(manifold.dist2[row, bank], beta, len(bank))

        d = np.sort(manifold.dist2[row, bank])
        nn_dist[i] = np.sqrt(d[0])
        knn_dist[i] = float(np.sqrt(d[:k]).mean())

    ok = np.isfinite(errors) & np.isfinite(energy)
    out: dict[str, float] = {"n_queries": int(ok.sum())}
    if ok.sum() < 10:
        for name in ("energy", "nn_dist", "knn_dist"):
            out[f"spearman_{name}"] = float("nan")
            out[f"pearson_{name}"] = float("nan")
        out["mean_abs_error"] = float("nan")
        return out

    for name, signal in (("energy", energy), ("nn_dist", nn_dist), ("knn_dist", knn_dist)):
        valid = ok & np.isfinite(signal)
        if valid.sum() < 10 or np.ptp(signal[valid]) == 0.0:
            out[f"spearman_{name}"] = float("nan")
            out[f"pearson_{name}"] = float("nan")
            continue
        out[f"spearman_{name}"] = float(spearmanr(signal[valid], errors[valid]).statistic)
        out[f"pearson_{name}"] = float(_pearson(signal[valid], errors[valid]))

    # Degeneracy diagnostic. The energy and its "free baseline" are not
    # independent statistics: the distance-kernel energy tends to
    # 0.5 * nn_dist^2, so once this reaches 1.0 a tie between the two columns
    # above is arithmetic and says nothing about confidence estimation.
    both = ok & np.isfinite(nn_dist)
    if both.sum() >= 10 and np.ptp(energy[both]) > 0 and np.ptp(nn_dist[both]) > 0:
        out["spearman_energy_vs_nn_dist"] = float(
            spearmanr(energy[both], nn_dist[both]).statistic
        )
    else:
        out["spearman_energy_vs_nn_dist"] = float("nan")

    out["mean_abs_error"] = float(np.mean(errors[ok]))
    return out


# --------------------------------------------------------------------------
# Systems
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One dataset, with the embedding and Theiler window ccm.py's callers use."""

    label: str
    family: str
    seed: int
    x: np.ndarray
    y: np.ndarray
    E: int
    exclusion_radius: int
    truth_x_to_y: bool
    truth_y_to_x: bool


def build_cases(n: int, seeds: int, rl_stride: int, wide_theiler: bool) -> list[Case]:
    """The four systems, at this project's settled embedding and exclusion values.

    E=3 and radius 0 for the logistic maps, E=7 and radius 5 for Rossler-Lorenz
    at stride 10: these are ``scripts/benchmark_causal.py``'s defaults, taken
    rather than chosen.  ``wide_theiler`` widens every window as a robustness
    pass; it is applied to both operators at once, so it cannot favour either.
    """
    cases: list[Case] = []
    for seed in range(seeds):
        cl = coupled_logistic(n=n, seed=seed)
        cases.append(
            Case(
                label=f"coupled_logistic[{seed}]",
                family="coupled_logistic",
                seed=seed,
                x=standardise(cl["x"]),
                y=standardise(cl["y"]),
                E=3,
                exclusion_radius=10 if wide_theiler else 0,
                truth_x_to_y=True,
                truth_y_to_x=False,
            )
        )

        il = independent_logistic(n=n, seed=seed)
        cases.append(
            Case(
                label=f"independent_logistic[{seed}]",
                family="independent_logistic",
                seed=seed,
                x=standardise(il["x"]),
                y=standardise(il["y"]),
                E=3,
                exclusion_radius=10 if wide_theiler else 0,
                truth_x_to_y=False,
                truth_y_to_x=False,
            )
        )

        rl = rossler_lorenz(n=n * rl_stride, seed=seed)
        cases.append(
            Case(
                label=f"rossler_lorenz[{seed}]",
                family="rossler_lorenz",
                seed=seed,
                x=standardise(np.asarray(rl["x"])[::rl_stride]),
                y=standardise(np.asarray(rl["y"])[::rl_stride]),
                E=7,
                exclusion_radius=15 if wide_theiler else 5,
                truth_x_to_y=True,
                truth_y_to_x=False,
            )
        )

        # White noise: the floor. No attractor, no manifold, nothing for either
        # operator to retrieve. Any skill here is the measurement apparatus
        # talking to itself and invalidates every other row.
        rng = np.random.default_rng(1000 + seed)
        cases.append(
            Case(
                label=f"white_noise[{seed}]",
                family="white_noise",
                seed=seed,
                x=standardise(rng.standard_normal(n)),
                y=standardise(rng.standard_normal(n)),
                E=3,
                exclusion_radius=10 if wide_theiler else 0,
                truth_x_to_y=False,
                truth_y_to_x=False,
            )
        )
    return cases


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def run_case(
    case: Case, n_bootstrap: int, betas: tuple[float, ...], max_queries: int
) -> dict[str, list[dict[str, Any]]]:
    """Every measurement for one dataset: both directions, both operators, all betas."""
    sweep_rows: list[dict[str, Any]] = []
    limit_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []

    manifolds = {
        "x": build_manifold(case.x, case.E, 1, case.exclusion_radius),
        "y": build_manifold(case.y, case.E, 1, case.exclusion_radius),
    }
    series = {"x": case.x, "y": case.y}

    # (hypothesis cause, hypothesis effect). Evidence for "cause -> effect" is
    # the skill of (effect xmap cause): the EFFECT supplies the manifold and the
    # CAUSE is the series predicted. ccm.py's convention, unchanged.
    for cause, effect in (("x", "y"), ("y", "x")):
        source = manifolds[effect]
        target = series[cause]
        lib_sizes = _default_lib_sizes(source.n_points, case.E)
        truth = case.truth_x_to_y if cause == "x" else case.truth_y_to_x

        base = {
            "system": case.label,
            "family": case.family,
            "seed": case.seed,
            "E": case.E,
            "exclusion_radius": case.exclusion_radius,
            "n_points": source.n_points,
            "hypothesis": f"{cause}->{effect}",
            "xmap": f"{effect} xmap {cause}",
            "truth": truth,
        }

        simplex_rho, simplex_mse = simplex_self_forecast(source, series[effect])

        # --- incumbent -------------------------------------------------------
        # knn_cross_map at k=E+1 IS simplex_cross_map -- same _simplex_predict,
        # same weights, same window -- but it accepts an explicit row set, which
        # is what lets both operators be handed identical libraries. The
        # equivalence is asserted below at the full library rather than assumed.
        simplex_samples = sweep_library(
            source,
            target,
            lib_sizes,
            n_bootstrap,
            case.seed,
            lambda rows: knn_cross_map(source, target, case.E + 1, rows),
        )
        public = simplex_cross_map(
            source.points,
            source.times,
            target,
            lib_size=None,
            exclusion_radius=source.exclusion_radius,
        )
        assert np.allclose(simplex_samples[-1, 0], public, equal_nan=True), (
            f"simplex reimplementation diverged from the public API on {case.label}: "
            f"{simplex_samples[-1, 0]} vs {public}"
        )
        simplex_dir = direction_result(cause, effect, lib_sizes, simplex_samples)
        sweep_rows.append(
            {
                **base,
                "method": "simplex",
                "kernel": "",
                "beta": np.nan,
                "rho_min_lib": simplex_dir.rho_at_min_lib,
                "rho_max_lib": simplex_dir.rho_at_max_lib,
                "delta_rho": simplex_dir.delta_rho,
                "delta_ci_low": simplex_dir.delta_rho_ci()[0],
                "delta_ci_high": simplex_dir.delta_rho_ci()[1],
                "convergent": simplex_dir.is_convergent(),
                "self_rho": simplex_rho,
                "self_mse": simplex_mse,
            }
        )

        # 1-NN: the operator a softmax actually tends to, and therefore the
        # reference for the high-beta check.
        nn_samples = sweep_library(
            source,
            target,
            lib_sizes,
            n_bootstrap,
            case.seed,
            lambda rows: knn_cross_map(source, target, 1, rows),
        )
        # The protocol requires each row's OWN held-out score, so 1-NN gets its
        # own self-forecast rather than a copy of simplex's.
        nn_self_rho, nn_self_mse = simplex_self_forecast(source, series[effect], n_neighbours=1)
        nn_dir = direction_result(cause, effect, lib_sizes, nn_samples)
        sweep_rows.append(
            {
                **base,
                "method": "knn1",
                "kernel": "",
                "beta": np.nan,
                "rho_min_lib": nn_dir.rho_at_min_lib,
                "rho_max_lib": nn_dir.rho_at_max_lib,
                "delta_rho": nn_dir.delta_rho,
                "delta_ci_low": nn_dir.delta_rho_ci()[0],
                "delta_ci_high": nn_dir.delta_rho_ci()[1],
                "convergent": nn_dir.is_convergent(),
                "self_rho": nn_self_rho,
                "self_mse": nn_self_mse,
            }
        )

        # --- CONTROL: unweighted k-NN mean -----------------------------------
        # No kernel, no temperature, no energy. Whatever this reproduces is not
        # a property of the Hopfield operator, and any mechanism claim that
        # survives has to survive this row first.
        for k_mean in KNN_MEAN_K:
            if k_mean >= int(lib_sizes[0]):
                continue  # would swallow the whole smallest library
            km_rho, km_mse = knn_mean_self_forecast(source, series[effect], k_mean)
            km_samples = sweep_library(
                source,
                target,
                lib_sizes,
                n_bootstrap,
                case.seed,
                lambda rows, k=k_mean: knn_mean_cross_map(source, target, k, rows),
            )
            km_dir = direction_result(cause, effect, lib_sizes, km_samples)
            sweep_rows.append(
                {
                    **base,
                    "method": f"knn{k_mean}mean",
                    "kernel": "",
                    "beta": np.nan,
                    "rho_min_lib": km_dir.rho_at_min_lib,
                    "rho_max_lib": km_dir.rho_at_max_lib,
                    "delta_rho": km_dir.delta_rho,
                    "delta_ci_low": km_dir.delta_rho_ci()[0],
                    "delta_ci_high": km_dir.delta_rho_ci()[1],
                    "convergent": km_dir.is_convergent(),
                    "self_rho": km_rho,
                    "self_mse": km_mse,
                }
            )

        # --- Hopfield --------------------------------------------------------
        # variants: (method name, kernel, top_k). top_k=None is the operator as
        # published; the truncated variant is the control for the claim that
        # retrieving over the WHOLE bank is what matters.
        variants = [(f"hopfield", k, None) for k in KERNELS]
        variants += [("hopfield-top", "dist", TOP_K_CONTROL)]
        for method_name, kernel, top_k in variants:
            for beta in betas:
                h_rho, h_mse = hopfield_self_forecast(
                    source, series[effect], kernel, beta, top_k=top_k
                )
                samples = sweep_library(
                    source,
                    target,
                    lib_sizes,
                    n_bootstrap,
                    case.seed,
                    lambda rows, k=kernel, b=beta, t=top_k: hopfield_cross_map(
                        source, target, k, b, rows, top_k=t
                    ),
                )
                d = direction_result(cause, effect, lib_sizes, samples)
                sweep_rows.append(
                    {
                        **base,
                        "method": method_name,
                        "kernel": kernel,
                        "beta": beta,
                        "top_k": top_k if top_k is not None else -1,
                        "rho_min_lib": d.rho_at_min_lib,
                        "rho_max_lib": d.rho_at_max_lib,
                        "delta_rho": d.delta_rho,
                        "delta_ci_low": d.delta_rho_ci()[0],
                        "delta_ci_high": d.delta_rho_ci()[1],
                        "convergent": d.is_convergent(),
                        "self_rho": h_rho,
                        "self_mse": h_mse,
                    }
                )

        # --- high-beta limit -------------------------------------------------
        # Compared on predictions, not on rho. Two operators can land on the same
        # correlation by different routes; agreement of the per-query estimates
        # is the check that one has actually become the other.
        full = np.arange(source.n_points)
        values = target[source.times]
        _, simplex_pred = _simplex_predict(
            manifold=source.points,
            times=source.times,
            values=values,
            lib_rows=full,
            pred_rows=full,
            n_neighbours=case.E + 1,
            exclusion_radius=source.exclusion_radius,
        )
        _, nn_pred = _simplex_predict(
            manifold=source.points,
            times=source.times,
            values=values,
            lib_rows=full,
            pred_rows=full,
            n_neighbours=1,
            exclusion_radius=source.exclusion_radius,
        )
        for kernel in KERNELS:
            for beta in (*betas, *BETA_LADDER):
                _, h_pred = hopfield_predict(source, kernel, values, full, full, beta)
                if len(h_pred) != len(nn_pred):
                    continue
                diff_nn = np.abs(h_pred - nn_pred)
                limit_rows.append(
                    {
                        **base,
                        "kernel": kernel,
                        "beta": beta,
                        "rho_hopfield": _pearson(values, h_pred),
                        "rho_simplex": _pearson(values, simplex_pred),
                        "rho_knn1": _pearson(values, nn_pred),
                        # Max and count alongside the RMSE: an RMSE can look
                        # converged while a handful of queries still retrieve an
                        # entirely wrong memory, which is the failure this check
                        # exists to catch.
                        "n_differ_vs_knn1": int((diff_nn > 1e-9).sum()),
                        "n_pred": len(h_pred),
                        "max_abs_diff_vs_knn1": float(diff_nn.max()),
                        "max_abs_diff_vs_simplex": float(np.abs(h_pred - simplex_pred).max()),
                        "rmse_vs_knn1": float(np.sqrt(np.mean(diff_nn**2))),
                        "rmse_vs_simplex": float(np.sqrt(np.mean((h_pred - simplex_pred) ** 2))),
                    }
                )

        # --- Experiment 2 ----------------------------------------------------
        for kernel in KERNELS:
            for beta in betas:
                stats = energy_versus_error(source, target, kernel, beta, max_queries, case.seed)
                energy_rows.append({**base, "kernel": kernel, "beta": beta, **stats})

    return {"sweep": sweep_rows, "limit": limit_rows, "energy": energy_rows}


def pick_headline(sweep: pd.DataFrame) -> pd.DataFrame:
    """Hopfield at the beta chosen by SELF-forecast skill, never by cross-map skill.

    Selection uses only the source manifold's own one-step forecast, so it is
    blind to the target series and to the ground-truth arrow.  The
    cross-map-maximising beta is carried alongside as an explicitly labelled
    oracle so the size of the gap between an honest choice and the best possible
    one is visible rather than assumed.
    """
    keys = ["system", "family", "seed", "hypothesis", "truth", "method", "kernel"]
    hop = sweep[sweep.method.isin(["hopfield", "hopfield-top"])].copy()

    picked = []
    for _, group in hop.groupby(keys, dropna=False):
        valid = group[np.isfinite(group.self_rho)]
        if valid.empty:
            continue
        honest = valid.loc[valid.self_rho.idxmax()].copy()
        oracle = group.loc[group.rho_max_lib.idxmax()]
        honest["beta_rule"] = "self_forecast"
        honest["oracle_beta"] = oracle.beta
        honest["oracle_rho_max_lib"] = oracle.rho_max_lib
        honest["oracle_convergent"] = oracle.convergent
        picked.append(honest)

    return pd.DataFrame(picked) if picked else pd.DataFrame()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n", type=int, default=2000, help="samples per series after subsampling")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--bootstrap", type=int, default=10, help="library draws per size")
    p.add_argument("--rl-stride", type=int, default=10)
    p.add_argument("--max-queries", type=int, default=800, help="queries scored in Experiment 2")
    p.add_argument(
        "--wide-theiler",
        action="store_true",
        help="robustness pass at a wider exclusion window, applied to both operators",
    )
    p.add_argument("--smoke", action="store_true", help="tiny run to check the wiring")
    p.add_argument("--outdir", default="ExpOutput/hopfield_dynamics")
    args = p.parse_args()

    betas = BETAS
    if args.smoke:
        args.n, args.seeds, args.bootstrap, args.max_queries = 400, 1, 3, 150
        betas = (0.3, 3.0, 100.0)

    cases = build_cases(args.n, args.seeds, args.rl_stride, args.wide_theiler)
    sweep_rows: list[dict[str, Any]] = []
    limit_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []

    for case in cases:
        t0 = time.time()
        out = run_case(case, args.bootstrap, betas, args.max_queries)
        sweep_rows += out["sweep"]
        limit_rows += out["limit"]
        energy_rows += out["energy"]
        print(f"  {case.label}: E={case.E} theiler={case.exclusion_radius} "
              f"({time.time() - t0:.0f}s)")

    sweep = pd.DataFrame(sweep_rows)
    limit = pd.DataFrame(limit_rows)
    energy = pd.DataFrame(energy_rows)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(outdir / "crossmap_sweep.csv", index=False)
    limit.to_csv(outdir / "highbeta_limit.csv", index=False)
    energy.to_csv(outdir / "energy_vs_error.csv", index=False)

    fmt = {"display.float_format": "{:.3f}".format, "display.width": 240,
           "display.max_columns": 50}

    # ---- protocol: held-out performance first, always -----------------------
    print("\n" + "=" * 100)
    print("MODEL SANITY: ONE-STEP SELF-FORECAST OF THE SOURCE MANIFOLD (standardised, MSE 1.0 = mean)")
    print("=" * 100)
    sanity = (
        sweep[sweep.method == "simplex"]
        .groupby(["family", "hypothesis"])
        .agg(self_rho=("self_rho", "mean"), self_mse=("self_mse", "mean"))
        .reset_index()
    )
    sanity["FAILED"] = sanity.self_mse >= 0.9
    with pd.option_context(*[i for kv in fmt.items() for i in kv]):
        print(sanity.to_string(index=False))
    if sanity.FAILED.any():
        print("\n  FLAGGED: manifolds above forecast nothing. Cross-map numbers on")
        print("  those rows are noise and are reported only to show they are noise.")

    # ---- Experiment 1 -------------------------------------------------------
    headline = pick_headline(sweep)
    print("\n" + "=" * 100)
    print("EXPERIMENT 1: CROSS-MAP SKILL AND CONVERGENCE, IDENTICAL DATA AND LIBRARIES")
    print("=" * 100)
    is_incumbent = sweep.method.isin(["simplex", "knn1"]) | sweep.method.str.endswith("mean")
    incumbent = sweep[is_incumbent].copy()
    incumbent["label"] = incumbent.method
    if not headline.empty:
        headline = headline.copy()
        headline["label"] = headline.method + "-" + headline.kernel
    combined = pd.concat([incumbent, headline], ignore_index=True)
    table = (
        combined.groupby(["family", "hypothesis", "truth", "label"])
        .agg(
            rho_min=("rho_min_lib", "mean"),
            rho_max=("rho_max_lib", "mean"),
            delta=("delta_rho", "mean"),
            conv=("convergent", "mean"),
            self_mse=("self_mse", "mean"),
        )
        .reset_index()
    )
    with pd.option_context(*[i for kv in fmt.items() for i in kv]):
        print(table.to_string(index=False))
    print("\n  'conv' is the fraction of seeds passing DirectionResult.is_convergent(),")
    print("  CCM's own three-part test, applied unchanged to every operator.")
    print("  Hopfield beta chosen by self-forecast skill; see oracle columns in the CSV.")
    print("\n  READ THE CONTROLS BEFORE THE RESULT:")
    print("    knn4mean / knn8mean  -- unweighted mean of the k nearest admissible")
    print("       neighbours. No softmax, no energy. Anything these reproduce is not")
    print("       attributable to the Hopfield operator.")
    print(f"    hopfield-top-dist    -- the same Gaussian softmax truncated to the")
    print(f"       {TOP_K_CONTROL} nearest memories. If it matches hopfield-dist, then")
    print("       'retrieval over the whole memory bank' is doing no work.")

    # ---- beta sweep ---------------------------------------------------------
    print("\n" + "=" * 100)
    print("BETA SWEEP: rho AT LARGEST LIBRARY (true-direction rows only)")
    print("=" * 100)
    true_rows = sweep[sweep.truth]
    if not true_rows.empty:
        pivot = (
            true_rows[true_rows.method.isin(["hopfield", "hopfield-top"])]
            .pivot_table(index=["family", "method", "kernel"], columns="beta",
                         values="rho_max_lib")
        )
        ref = (
            true_rows[~true_rows.method.isin(["hopfield", "hopfield-top"])]
            .pivot_table(index="family", columns="method", values="rho_max_lib")
        )
        with pd.option_context(*[i for kv in fmt.items() for i in kv]):
            print(pivot.to_string())
            print("\n  reference:")
            print(ref.to_string())

    # ---- high-beta limit ----------------------------------------------------
    print("\n" + "=" * 100)
    print("HIGH-BETA LIMIT: DOES SOFTMAX RETRIEVAL BECOME A NEIGHBOUR METHOD?")
    print("=" * 100)
    lim = (
        limit.groupby(["kernel", "beta"])
        .agg(
            n_differ_knn1=("n_differ_vs_knn1", "mean"),
            max_diff_knn1=("max_abs_diff_vs_knn1", "max"),
            rmse_knn1=("rmse_vs_knn1", "mean"),
            rmse_simplex=("rmse_vs_simplex", "mean"),
        )
        .reset_index()
    )
    with pd.option_context(*[i for kv in fmt.items() for i in kv]):
        print(lim.to_string(index=False))
    gap = limit.groupby("family").agg(
        rho_simplex=("rho_simplex", "mean"), rho_knn1=("rho_knn1", "mean")
    )
    print("\n  residual gap between the true limit (1-NN) and simplex, which no beta closes:")
    with pd.option_context(*[i for kv in fmt.items() for i in kv]):
        print(gap.to_string())

    # ---- Experiment 2 -------------------------------------------------------
    print("\n" + "=" * 100)
    print("EXPERIMENT 2: ENERGY vs |CROSS-MAP ERROR|  (positive predicted a priori)")
    print("=" * 100)
    e = (
        energy.groupby(["family", "kernel", "beta"])
        .agg(
            energy=("spearman_energy", "mean"),
            nn_dist=("spearman_nn_dist", "mean"),
            knn_dist=("spearman_knn_dist", "mean"),
            degeneracy=("spearman_energy_vs_nn_dist", "mean"),
        )
        .reset_index()
    )
    with pd.option_context(*[i for kv in fmt.items() for i in kv]):
        print(e.to_string(index=False))
    print("\n  nn_dist and knn_dist are what simplex already computes for free.")
    print("  Energy is only news where it beats them.")
    print("\n  'degeneracy' is spearman(energy, nn_dist) itself. The distance-kernel")
    print("  energy tends to 0.5*nn_dist^2 as beta grows, so where this column is")
    print("  near 1.0 the two are the same statistic and a tie between them is")
    print("  arithmetic, not evidence. Only rows well below 1.0 are a real contest.")

    print(f"\nwrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
