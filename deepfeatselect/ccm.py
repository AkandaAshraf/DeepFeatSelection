"""Convergent cross mapping: detecting directional causality between two series.

Sugihara et al., "Detecting Causality in Complex Ecosystems", Science 336 (2012)
496-500.

DIRECTION CONVENTION -- READ THIS BEFORE CHANGING ANYTHING ELSE
---------------------------------------------------------------
``"Y xmap X"`` means: build the shadow manifold of ``Y``, and use it to predict
``X``.  The causal arrow runs *against* the direction of prediction::

    convergence of (Y xmap X)  is evidence that  X CAUSES Y

This is not a convention we are free to pick.  It follows from Takens: if ``X``
drives ``Y`` then ``Y``'s trajectory is shaped by ``X``, so ``Y``'s history
carries an imprint of ``X`` and ``X`` is the variable recoverable *from* ``Y``.
The reverse does not hold -- a variable that drives nothing leaves no trace in
the other series, so its manifold cannot reconstruct that series.

Reading the skill of (Y xmap X) as evidence for "Y causes X" is the one bug this
module exists to prevent, and it is completely silent: every number stays in
range, the curves still converge, and every conclusion is inverted.  So the
direction is carried in the data rather than left to the caller's memory.
:class:`DirectionResult` records ``cause``, ``effect`` and the ``xmap`` label it
was computed from, and :class:`CCMResult` exposes the two directions as
``x_causes_y`` and ``y_causes_x`` instead of as a bare pair of skills.

DETECTION REQUIRES CONVERGENCE, NOT SKILL
-----------------------------------------
A high cross-map skill on its own means very little: two series sharing a
seasonal cycle, or driven by a common third variable, predict each other
perfectly well.  What separates causation is that skill *rises with library
size* and then saturates -- more of the attractor filled in means closer
neighbours, hence a better reconstruction.  Correlation-like measures have no
such signature.  :meth:`DirectionResult.is_convergent` therefore demands both a
high rho at the largest library and a bootstrap-significant increase from the
smallest; neither alone is evidence.

Two further traps this module handles explicitly:

* **Temporal autocorrelation.**  In a smooth or slowly varying series the
  nearest neighbours on the manifold are simply the temporally adjacent points,
  and "predicting" a value from its immediate neighbours in time is trivial.
  ``exclusion_radius`` (a Theiler window) discards library points within that
  many time steps of the prediction point.  Leave it at 0 only for series with
  no appreciable autocorrelation.
* **Library resampling.**  "Bootstrap" here means repeated random library
  subsets drawn *without* replacement, which is the resampling scheme CCM
  actually uses.  A with-replacement bootstrap would place duplicate library
  points at zero distance from each other and corrupt the neighbour weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "CCMResult",
    "DirectionResult",
    "DirectionTest",
    "SurrogateTestResult",
    "ccm",
    "circular_shift_surrogate",
    "ebisuzaki_surrogate",
    "optimal_embedding_dimension",
    "simplex_cross_map",
    "surrogate_test",
    "time_delay_embed",
]

# Floor on the simplex weights.  Without it a neighbour far from the rest
# underflows to exactly zero and effectively reduces E+1 to fewer neighbours;
# the same guard appears in rEDM.
_MIN_WEIGHT = 1e-6


def time_delay_embed(x, E: int, tau: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Takens time-delay embedding of a scalar series.

    Row ``i`` of the returned manifold is ``[x(t), x(t - tau), ...,
    x(t - (E-1)*tau)]`` for ``t = times[i]``.  Present value first, so column 0
    is the series itself -- that ordering is what makes a cross map of ``x`` from
    its own manifold trivially near-perfect, and it keeps the "state at time t"
    reading of a row obvious.

    The first ``(E-1)*tau`` samples have no full history and are dropped, so
    ``times`` is returned rather than assumed: every downstream step needs the
    original time index to line a manifold row up with the target series and to
    apply the Theiler window.

    Returns:
        ``(manifold, times)`` with shapes ``(n - (E-1)*tau, E)`` and
        ``(n - (E-1)*tau,)``.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if E < 1:
        raise ValueError(f"E must be at least 1, got {E}")
    if tau < 1:
        raise ValueError(f"tau must be at least 1, got {tau}")

    span = (E - 1) * tau
    if len(x) <= span:
        raise ValueError(
            f"series of length {len(x)} is too short for E={E}, tau={tau}; "
            f"needs more than {span} points"
        )

    times = np.arange(span, len(x))
    offsets = np.arange(E) * tau
    return x[times[:, None] - offsets[None, :]], times


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, returning nan for a constant input rather than raising."""
    a = a - a.mean()
    b = b - b.mean()
    na = float(np.sqrt(a @ a))
    nb = float(np.sqrt(b @ b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float((a @ b) / (na * nb))


def _simplex_predict(
    manifold: np.ndarray,
    times: np.ndarray,
    values: np.ndarray,
    lib_rows: np.ndarray,
    pred_rows: np.ndarray,
    n_neighbours: int,
    exclusion_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simplex projection: predict ``values`` at ``pred_rows`` from ``lib_rows``.

    ``values[i]`` is whatever quantity row ``i`` of the manifold is supposed to
    predict -- the contemporaneous value of another series for a cross map, or
    the same series ``tp`` steps ahead for a self-prediction.  Keeping that
    general is what lets :func:`simplex_cross_map` and
    :func:`optimal_embedding_dimension` share one implementation.

    Returns ``(actual, predicted)`` over the prediction rows that had enough
    admissible neighbours; rows that did not are dropped rather than filled in.
    """
    lib_points = manifold[lib_rows]
    lib_times = times[lib_rows]
    lib_values = values[lib_rows]
    pred_times = times[pred_rows]

    # Query extra neighbours so the Theiler window can be applied by discarding
    # instead of re-querying: at most 2r+1 library points can fall inside the
    # window of any one prediction point, and a self-match is one of them.
    k = min(n_neighbours + 2 * exclusion_radius + 1, len(lib_rows))
    dists, idx = cKDTree(lib_points).query(manifold[pred_rows], k=k)
    if k == 1:
        dists = dists[:, None]
        idx = idx[:, None]

    # Push excluded neighbours to infinity, then re-sort. The query already
    # returned ascending distances, so this is a stable partition that keeps the
    # admissible neighbours in distance order.
    excluded = np.abs(lib_times[idx] - pred_times[:, None]) <= exclusion_radius
    dists = np.where(excluded, np.inf, dists)
    order = np.argsort(dists, axis=1, kind="stable")[:, :n_neighbours]
    d = np.take_along_axis(dists, order, axis=1)
    j = np.take_along_axis(idx, order, axis=1)

    usable = np.isfinite(d).all(axis=1)
    if not usable.any():
        return np.empty(0), np.empty(0)
    d, j = d[usable], j[usable]

    # Sugihara's exp(-d_i/d_1) is undefined when the nearest neighbour sits
    # exactly on the prediction point, which happens whenever a state repeats
    # (short, quantised or periodic series). There the limit puts all the mass
    # on the zero-distance neighbours, so do that directly.
    d1 = d[:, :1]
    degenerate = (d1 <= 0.0).reshape(-1)
    w = np.maximum(np.exp(-d / np.where(d1 > 0.0, d1, 1.0)), _MIN_WEIGHT)
    if degenerate.any():
        w[degenerate] = np.where(d[degenerate] <= 0.0, 1.0, _MIN_WEIGHT)
    w /= w.sum(axis=1, keepdims=True)

    predicted = (w * lib_values[j]).sum(axis=1)
    return values[pred_rows][usable], predicted


def simplex_cross_map(
    source_manifold: np.ndarray,
    source_times: np.ndarray,
    target,
    lib_size: int | None = None,
    seed: int | np.random.Generator = 0,
    exclusion_radius: int = 0,
) -> float:
    """Cross-map skill: how well ``source_manifold`` reconstructs ``target``.

    This computes "SOURCE xmap TARGET" -- neighbours come from the *source*
    manifold and the *target* series is what gets predicted.  Convergence here
    is evidence that TARGET causes SOURCE, not the other way round; see the
    module docstring.

    Each prediction point takes its ``E+1`` nearest neighbours in the source
    manifold (``E`` being the embedding dimension, one simplex vertex more than
    the dimension), weighted by ``exp(-d_i / d_1)`` normalised to sum to one.

    Args:
        source_times: Original time index of each manifold row, as returned by
            :func:`time_delay_embed`.  Used both to index ``target`` and to
            apply the Theiler window.
        target: Full-length target series, indexed by the *original* time index.
        lib_size: Number of library points to draw, without replacement, from
            which neighbours may be taken.  ``None`` uses every point.
            Prediction always runs over every point, so rho at different library
            sizes is measured on the same prediction set.
        exclusion_radius: Theiler window; library points within this many time
            steps of a prediction point are not eligible as its neighbours.
            A radius of 0 still excludes the point itself.

    Returns:
        Pearson correlation between the actual and predicted target values, or
        nan if either is constant.
    """
    source_manifold = np.asarray(source_manifold, dtype=np.float64)
    source_times = np.asarray(source_times, dtype=np.int64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)

    if source_manifold.ndim != 2:
        raise ValueError(f"source_manifold must be 2-D, got shape {source_manifold.shape}")
    n_points, embedding_dim = source_manifold.shape
    if len(source_times) != n_points:
        raise ValueError(
            f"source_times has {len(source_times)} entries but the manifold has {n_points} rows"
        )
    if n_points and int(source_times.max()) >= len(target):
        raise ValueError(
            f"source_times reaches index {int(source_times.max())} but target has "
            f"only {len(target)} points"
        )
    if exclusion_radius < 0:
        raise ValueError(f"exclusion_radius must be non-negative, got {exclusion_radius}")

    n_neighbours = embedding_dim + 1
    # One spare point beyond the simplex, because the prediction point itself is
    # excluded whenever it happens to be in the library.
    min_lib = n_neighbours + 1
    if lib_size is None:
        lib_rows = np.arange(n_points)
    else:
        lib_size = int(lib_size)
        if lib_size > n_points:
            raise ValueError(f"lib_size {lib_size} exceeds the {n_points} available points")
        if lib_size < min_lib:
            raise ValueError(
                f"lib_size {lib_size} is below the {min_lib} points needed for "
                f"{n_neighbours} neighbours at E={embedding_dim}"
            )
        lib_rows = np.random.default_rng(seed).choice(n_points, size=lib_size, replace=False)

    if n_points < min_lib:
        raise ValueError(
            f"{n_points} embedded points is below the {min_lib} needed at E={embedding_dim}"
        )

    actual, predicted = _simplex_predict(
        manifold=source_manifold,
        times=source_times,
        values=target[source_times],
        lib_rows=lib_rows,
        pred_rows=np.arange(n_points),
        n_neighbours=n_neighbours,
        exclusion_radius=exclusion_radius,
    )
    if len(actual) < 3:
        return float("nan")
    return _pearson(actual, predicted)


@dataclass(frozen=True)
class DirectionResult:
    """Cross-map skill for ONE causal direction, at increasing library sizes.

    ``cause`` and ``effect`` name the hypothesis this object is evidence for,
    and ``xmap`` records the cross map the numbers actually came from.  For a
    valid result the two always disagree in the way the module docstring
    describes: ``xmap`` is ``"<effect> xmap <cause>"``.
    """

    cause: str
    effect: str
    xmap: str
    lib_sizes: np.ndarray
    rho: np.ndarray
    rho_ci_low: np.ndarray
    rho_ci_high: np.ndarray
    rho_samples: np.ndarray

    @property
    def rho_at_min_lib(self) -> float:
        return float(self.rho[0])

    @property
    def rho_at_max_lib(self) -> float:
        return float(self.rho[-1])

    @property
    def delta_rho(self) -> float:
        """Increase in skill from the smallest to the largest library."""
        return float(self.rho[-1] - self.rho[0])

    def delta_rho_ci(self, alpha: float = 0.05) -> tuple[float, float]:
        """Percentile interval for :attr:`delta_rho`.

        The replicates at the two library sizes are independent draws, so
        pairing them by index is arbitrary -- but differencing independent
        samples still gives a correct Monte Carlo estimate of the distribution
        of the difference, which is all the interval needs.

        This is deliberately the spread of *individual* replicate deltas and not
        a confidence interval on the mean delta.  An interval on the mean would
        shrink like ``1/sqrt(n_bootstrap)``, so raising a purely computational
        knob would eventually declare any arbitrarily small rise significant.
        The question that matters is instead whether a randomly drawn small
        library reliably does worse than the full one, and that spread does not
        shrink with more replicates.

        At the largest library size every replicate uses the whole manifold and
        the spread is genuinely zero, so the interval reflects only the
        variability of the small-library end.  That is the intended reading.
        """
        deltas = self.rho_samples[-1] - self.rho_samples[0]
        lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return float(lo), float(hi)

    def is_convergent(
        self, min_rho: float = 0.3, min_delta: float = 0.05, alpha: float = 0.05
    ) -> bool:
        """Whether this direction shows convergent cross mapping.

        All three conditions must hold.  Skill alone is met by any two series
        sharing a cycle or a common driver; an increase alone is met by a pair
        that never gets anywhere near predictive.  Requiring the increase to be
        bootstrap-significant as well keeps a couple of hundredths of drift from
        being reported as causation.

        The defaults are a permissive floor, not a recommendation.  Strong
        unidirectional forcing leaves a partial imprint in the reverse direction
        too -- the driven series' history constrains the driver's -- so on a
        genuinely one-way system the wrong direction can still reach a moderate
        rho and, on some trajectories, a weakly significant rise.  Compare the
        two directions against each other (:meth:`CCMResult.dominant_direction`)
        rather than reading either verdict in isolation.
        """
        if not np.isfinite(self.rho_at_max_lib):
            return False
        return (
            self.rho_at_max_lib >= min_rho
            and self.delta_rho >= min_delta
            and self.delta_rho_ci(alpha)[0] > 0.0
        )

    def describe(self, alpha: float = 0.05) -> str:
        lo, hi = self.delta_rho_ci(alpha)
        verdict = "detected" if self.is_convergent(alpha=alpha) else "not detected"
        return (
            f"{self.cause} -> {self.effect}  [{self.xmap}]  {verdict}\n"
            f"  rho {self.rho_at_min_lib:+.3f} (L={int(self.lib_sizes[0])}) -> "
            f"{self.rho_at_max_lib:+.3f} (L={int(self.lib_sizes[-1])})\n"
            f"  delta_rho {self.delta_rho:+.3f}  "
            f"{100 * (1 - alpha):.0f}% CI [{lo:+.3f}, {hi:+.3f}]"
        )


@dataclass(frozen=True)
class CCMResult:
    """Both causal directions between ``x`` and ``y``.

    Each field is named for the hypothesis it supports, never for the cross map
    it was computed from -- see ``DirectionResult.xmap`` for that.
    """

    E: int
    tau: int
    lib_sizes: np.ndarray
    x_causes_y: DirectionResult
    y_causes_x: DirectionResult

    def dominant_direction(self, **kwargs) -> DirectionResult | None:
        """The better-supported causal direction, or ``None`` if neither converges.

        Worth preferring over two independent verdicts.  Under strong one-way
        forcing the reverse direction picks up real but much weaker convergence,
        so both can pass :meth:`DirectionResult.is_convergent` while the skills
        are nowhere near comparable; taking the larger max-library rho recovers
        the driver.  Genuinely bidirectional coupling looks the same from here,
        which is why this returns the stronger direction and not "the" answer:
        inspect both curves before concluding the weaker one is absent.

        ``kwargs`` are forwarded to :meth:`DirectionResult.is_convergent`.
        """
        converging = [d for d in (self.x_causes_y, self.y_causes_x) if d.is_convergent(**kwargs)]
        if not converging:
            return None
        return max(converging, key=lambda d: d.rho_at_max_lib)

    def describe(self, alpha: float = 0.05) -> str:
        return (
            f"CCM  E={self.E}  tau={self.tau}  "
            f"libraries {int(self.lib_sizes[0])}..{int(self.lib_sizes[-1])}\n"
            f"{self.x_causes_y.describe(alpha)}\n"
            f"{self.y_causes_x.describe(alpha)}"
        )

    def report(self, alpha: float = 0.05) -> None:
        print(self.describe(alpha))


def _default_lib_sizes(n_points: int, embedding_dim: int, n_sizes: int = 8) -> np.ndarray:
    """Geometrically spaced library sizes.

    Geometric rather than linear because the skill curve saturates: nearly all
    of the rise happens over the first fraction of the range, and linear spacing
    spends most of its points on the flat part where there is nothing to see.
    """
    smallest = max(embedding_dim + 2, 10)
    if smallest > n_points:
        raise ValueError(
            f"{n_points} embedded points is too few for E={embedding_dim}; "
            f"need at least {smallest}"
        )
    return np.unique(np.geomspace(smallest, n_points, num=n_sizes).astype(np.int64))


def _cross_map_sweep(
    manifold: np.ndarray,
    times: np.ndarray,
    target: np.ndarray,
    lib_sizes: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    exclusion_radius: int,
) -> np.ndarray:
    """Cross-map skill at each library size, ``n_bootstrap`` replicates each."""
    n_points = manifold.shape[0]
    samples = np.empty((len(lib_sizes), n_bootstrap), dtype=np.float64)

    for i, lib_size in enumerate(lib_sizes):
        if lib_size >= n_points:
            # Every replicate would draw the identical full library, so the
            # spread is zero by construction; computing it n_bootstrap times
            # buys nothing.
            samples[i, :] = simplex_cross_map(
                manifold, times, target, lib_size=None, exclusion_radius=exclusion_radius
            )
            continue
        for b in range(n_bootstrap):
            samples[i, b] = simplex_cross_map(
                manifold,
                times,
                target,
                lib_size=int(lib_size),
                seed=rng,
                exclusion_radius=exclusion_radius,
            )
    return samples


def ccm(
    x,
    y,
    E: int = 3,
    tau: int = 1,
    lib_sizes=None,
    n_bootstrap: int = 50,
    seed: int = 0,
    exclusion_radius: int = 0,
    alpha: float = 0.05,
) -> CCMResult:
    """Convergent cross mapping in both directions between two series.

    Args:
        x, y: Equal-length scalar series, sampled on the same time base.
        E: Embedding dimension; :func:`optimal_embedding_dimension` picks it.
        lib_sizes: Library sizes to sweep.  Defaults to eight geometrically
            spaced sizes up to the full manifold.
        n_bootstrap: Random library subsets per size.
        exclusion_radius: Theiler window; see the module docstring.

    Returns:
        A :class:`CCMResult` whose ``x_causes_y`` field holds the skill of
        (Y xmap X) and whose ``y_causes_x`` field holds the skill of (X xmap Y).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) != len(y):
        raise ValueError(f"x and y must be the same length, got {len(x)} and {len(y)}")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be at least 1, got {n_bootstrap}")

    manifold_x, times_x = time_delay_embed(x, E, tau)
    manifold_y, times_y = time_delay_embed(y, E, tau)
    n_points = manifold_x.shape[0]

    if lib_sizes is None:
        lib_sizes = _default_lib_sizes(n_points, E)
    else:
        lib_sizes = np.unique(np.asarray(lib_sizes, dtype=np.int64).reshape(-1))
        if lib_sizes.min() < E + 2:
            raise ValueError(f"library sizes must be at least E+2={E + 2}, got {lib_sizes.min()}")
        if lib_sizes.max() > n_points:
            raise ValueError(
                f"library size {lib_sizes.max()} exceeds the {n_points} embedded points"
            )
    if len(lib_sizes) < 2:
        raise ValueError("at least two library sizes are needed to judge convergence")

    # An independent stream per direction, rather than one shared generator, so
    # a direction's library draws do not depend on how many draws the other
    # direction happened to consume first.
    stream_a, stream_b = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))

    # THE CONVENTION LIVES HERE. Read these two blocks together.
    #
    # (Y xmap X): neighbours come from Y's manifold, X is the series predicted.
    # X being recoverable from Y's trajectory means X left an imprint on Y, so
    # this is the evidence for X CAUSES Y. It is *not* evidence for Y -> X.
    y_xmap_x = _cross_map_sweep(
        manifold_y, times_y, x, lib_sizes, n_bootstrap, stream_a, exclusion_radius
    )
    # (X xmap Y): the mirror image, and therefore the evidence for Y CAUSES X.
    x_xmap_y = _cross_map_sweep(
        manifold_x, times_x, y, lib_sizes, n_bootstrap, stream_b, exclusion_radius
    )

    def _direction(cause: str, effect: str, samples: np.ndarray) -> DirectionResult:
        lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=1)
        return DirectionResult(
            cause=cause,
            effect=effect,
            xmap=f"{effect} xmap {cause}",
            lib_sizes=lib_sizes,
            rho=samples.mean(axis=1),
            rho_ci_low=lo,
            rho_ci_high=hi,
            rho_samples=samples,
        )

    return CCMResult(
        E=E,
        tau=tau,
        lib_sizes=lib_sizes,
        # Evidence for "x causes y" is the skill of (Y xmap X): y_xmap_x.
        x_causes_y=_direction("x", "y", y_xmap_x),
        # Evidence for "y causes x" is the skill of (X xmap Y): x_xmap_y.
        y_causes_x=_direction("y", "x", x_xmap_y),
    )


def optimal_embedding_dimension(
    x,
    max_E: int = 10,
    tau: int = 1,
    tp: int = 1,
    exclusion_radius: int = 0,
) -> int:
    """Embedding dimension with the best simplex-projection self-prediction skill.

    Standard univariate simplex projection (Sugihara and May, 1990): embed ``x``
    at each candidate ``E``, predict ``x`` ``tp`` steps ahead by leave-one-out
    simplex, and keep the ``E`` with the highest correlation.  Too small an ``E``
    leaves the attractor self-intersecting, so unrelated states get averaged
    together; too large an ``E`` spreads the same number of points through more
    volume and the neighbours stop being near.

    Every candidate is scored on the *same* prediction points -- those valid at
    ``max_E`` -- because a larger ``E`` otherwise discards more of the start of
    the series and would be compared on an easier or harder subset.

    Ties keep the smaller ``E``: the skill curve usually plateaus, and the
    smallest adequate embedding is the one with the most neighbours per unit
    volume.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if max_E < 1:
        raise ValueError(f"max_E must be at least 1, got {max_E}")
    if tp < 0:
        raise ValueError(f"tp must be non-negative, got {tp}")

    span = (max_E - 1) * tau
    common_times = np.arange(span, len(x) - tp)
    if len(common_times) < max_E + 3:
        raise ValueError(
            f"series of length {len(x)} is too short to compare embeddings up to "
            f"E={max_E} at tau={tau}, tp={tp}"
        )

    best_E, best_rho = 1, -np.inf
    for E in range(1, max_E + 1):
        manifold, times = time_delay_embed(x, E, tau)
        keep = np.isin(times, common_times)
        rows = np.arange(int(keep.sum()))
        actual, predicted = _simplex_predict(
            manifold=manifold[keep],
            times=times[keep],
            values=x[times[keep] + tp],
            lib_rows=rows,
            pred_rows=rows,
            n_neighbours=E + 1,
            exclusion_radius=exclusion_radius,
        )
        rho = _pearson(actual, predicted) if len(actual) >= 3 else float("nan")
        if np.isfinite(rho) and rho > best_rho:
            best_E, best_rho = E, rho

    return best_E


def ebisuzaki_surrogate(x, seed: int | np.random.Generator = 0) -> np.ndarray:
    """Random-phase surrogate with the same power spectrum (Ebisuzaki, 1997).

    Randomises the Fourier phases while keeping every amplitude, so the
    surrogate has the same power spectrum -- hence the same autocorrelation and
    the same mean -- as the original, but none of its deterministic structure.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)

    n = len(x)
    spectrum = np.fft.rfft(x)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=spectrum.shape)
    # The DC term must stay real to preserve the mean, and for even n so must
    # the Nyquist term -- it has no conjugate partner, so only its sign is free.
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = rng.choice([0.0, np.pi])
    return np.fft.irfft(np.abs(spectrum) * np.exp(1j * phases), n=n)


def circular_shift_surrogate(x, seed: int | np.random.Generator = 0) -> np.ndarray:
    """Surrogate formed by rotating the series by a random offset.

    Keeps the trajectory itself -- every value, and all the nonlinear structure
    -- and destroys only the alignment with the other series.  Stricter than a
    phase surrogate, but it draws from a pool of just ``n-1`` distinct series
    and the wrap-around introduces one discontinuity.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) < 2:
        return x.copy()
    shift = int(np.random.default_rng(seed).integers(1, len(x)))
    return np.roll(x, shift)


_SURROGATE_METHODS = {
    "ebisuzaki": ebisuzaki_surrogate,
    "circular": circular_shift_surrogate,
}


@dataclass(frozen=True)
class DirectionTest:
    """Surrogate-test outcome for ONE causal direction.

    As in :class:`DirectionResult`, ``cause``/``effect`` name the hypothesis and
    ``xmap`` records the cross map the skill came from.
    """

    cause: str
    effect: str
    xmap: str
    rho: float
    p_value: float
    null_rho: np.ndarray

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value <= alpha

    def describe(self, alpha: float = 0.05) -> str:
        verdict = "significant" if self.is_significant(alpha) else "not significant"
        return (
            f"{self.cause} -> {self.effect}  [{self.xmap}]  {verdict}\n"
            f"  rho {self.rho:+.3f}  p={self.p_value:.4f}  "
            f"null mean {np.nanmean(self.null_rho):+.3f} "
            f"max {np.nanmax(self.null_rho):+.3f}"
        )


@dataclass(frozen=True)
class SurrogateTestResult:
    """Surrogate p-values for both directions."""

    method: str
    n_surrogates: int
    E: int
    tau: int
    x_causes_y: DirectionTest
    y_causes_x: DirectionTest

    def describe(self, alpha: float = 0.05) -> str:
        return (
            f"surrogate test  method={self.method}  n={self.n_surrogates}  "
            f"E={self.E}  tau={self.tau}\n"
            f"{self.x_causes_y.describe(alpha)}\n"
            f"{self.y_causes_x.describe(alpha)}"
        )

    def report(self, alpha: float = 0.05) -> None:
        print(self.describe(alpha))


def surrogate_test(
    x,
    y,
    E: int = 3,
    tau: int = 1,
    n_surrogates: int = 100,
    lib_size: int | None = None,
    method: str = "ebisuzaki",
    seed: int = 0,
    exclusion_radius: int = 0,
) -> SurrogateTestResult:
    """Significance of the cross-map skill in each direction against a null.

    WHICH SURROGATE, AND WHY.  The default is Ebisuzaki (1997) random-phase
    surrogates.  The failure mode a CCM p-value most needs to rule out is two
    series that merely share strong autocorrelation, a seasonal cycle, or a
    common driver -- all of which produce a respectable cross-map skill with no
    causal link whatsoever.  A random-phase surrogate reproduces exactly that
    much of the series (the power spectrum is the autocorrelation, by
    Wiener-Khinchin) while destroying the deterministic trajectory, so the null
    distribution is "what skill is achievable from this spectrum alone".  It
    also supplies an unlimited pool of independent surrogates.

    ``method="circular"`` rotates the series instead.  That preserves the whole
    nonlinear trajectory and breaks only the alignment between the two series,
    which is the stricter null when the driver is strongly non-Gaussian and the
    spectral null looks too permissive -- but only ``n-1`` distinct surrogates
    exist, which floors the achievable p-value, and the wrap seam injects one
    artificial discontinuity.

    WHICH SERIES GETS RANDOMISED.  The putative *cause* -- the series being
    predicted.  Testing "x causes y" means testing the skill of (Y xmap X), so
    ``x`` is replaced by surrogates while ``y``'s manifold is left exactly as it
    is.  Any skill that survives came from ``x``'s spectrum rather than from a
    causal imprint on ``y``.

    The p-value is ``(1 + #{rho_surrogate >= rho_observed}) / (n_surrogates + 1)``:
    the observed value counts as one of its own null draws, so a p-value of zero
    is not reportable no matter how many surrogates are run.  A cross map that
    could not be evaluated at all (nan, e.g. an exclusion radius that leaves no
    admissible neighbour) reports ``p = 1``: undefined skill is not significant
    skill.

    This tests skill at a single library size and is a complement to, not a
    replacement for, the convergence check in :func:`ccm`.
    """
    if method not in _SURROGATE_METHODS:
        raise ValueError(f"method must be one of {sorted(_SURROGATE_METHODS)}, got {method!r}")
    if n_surrogates < 1:
        raise ValueError(f"n_surrogates must be at least 1, got {n_surrogates}")

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) != len(y):
        raise ValueError(f"x and y must be the same length, got {len(x)} and {len(y)}")

    make_surrogate = _SURROGATE_METHODS[method]
    manifold_x, times_x = time_delay_embed(x, E, tau)
    manifold_y, times_y = time_delay_embed(y, E, tau)
    rng = np.random.default_rng(seed)

    def _test(cause: str, effect: str, driver, manifold, times) -> DirectionTest:
        # The library subset is fixed across the observed run and every
        # surrogate, so the comparison isolates the driver and not the draw.
        lib_seed = int(rng.integers(0, 2**31 - 1))
        observed = simplex_cross_map(
            manifold, times, driver, lib_size=lib_size, seed=lib_seed,
            exclusion_radius=exclusion_radius,
        )
        null = np.array(
            [
                simplex_cross_map(
                    manifold, times, make_surrogate(driver, rng), lib_size=lib_size,
                    seed=lib_seed, exclusion_radius=exclusion_radius,
                )
                for _ in range(n_surrogates)
            ]
        )
        # nan compares False against everything, so a bare `null >= observed`
        # scores an unevaluable cross map as "no surrogate beat it" and returns
        # the smallest p-value the test can produce -- the exact inversion of the
        # truth. An undefined observed skill cannot reject anything, and a
        # surrogate that failed to evaluate is not evidence for rejection either,
        # so it counts towards the null instead of against it.
        if not np.isfinite(observed):
            p = 1.0
        else:
            beat_it = ~np.isfinite(null) | (null >= observed)
            p = (1.0 + float(np.count_nonzero(beat_it))) / (n_surrogates + 1.0)
        return DirectionTest(
            cause=cause,
            effect=effect,
            xmap=f"{effect} xmap {cause}",
            rho=observed,
            p_value=p,
            null_rho=null,
        )

    return SurrogateTestResult(
        method=method,
        n_surrogates=n_surrogates,
        E=E,
        tau=tau,
        # "x causes y" is tested with (Y xmap X): y's manifold, x as the target.
        x_causes_y=_test("x", "y", x, manifold_y, times_y),
        # "y causes x" is tested with (X xmap Y): x's manifold, y as the target.
        y_causes_x=_test("y", "x", y, manifold_x, times_x),
    )
