"""Tests for convergent cross mapping.

The coupled-logistic tests below are the reason this module exists.  If one of
them fails claiming the wrong causal direction, the convention in ``ccm.py`` is
backwards -- fix the code, never the assertion.
"""

import numpy as np
import pytest

from deepfeatselect.ccm import (
    CCMResult,
    ccm,
    circular_shift_surrogate,
    ebisuzaki_surrogate,
    optimal_embedding_dimension,
    simplex_cross_map,
    surrogate_test,
    time_delay_embed,
)


def coupled_logistic(
    n: int,
    b_xy: float,
    b_yx: float,
    x0: float = 0.4,
    y0: float = 0.2,
    burn: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """The Sugihara (2012) coupled logistic map.

        X(t+1) = X(t) * (3.8 - 3.8*X(t) - b_xy*Y(t))
        Y(t+1) = Y(t) * (3.5 - 3.5*Y(t) - b_yx*X(t))

    ``b_xy`` is the effect of Y on X and ``b_yx`` the effect of X on Y, i.e. the
    subscripts read "effect on the first index from the second".  The burn-in is
    discarded so both series sit on the attractor rather than on the transient.
    """
    total = n + burn
    x = np.empty(total)
    y = np.empty(total)
    x[0], y[0] = x0, y0
    for t in range(total - 1):
        x[t + 1] = x[t] * (3.8 - 3.8 * x[t] - b_xy * y[t])
        y[t + 1] = y[t] * (3.5 - 3.5 * y[t] - b_yx * x[t])
    return x[burn:], y[burn:]


def logistic(n: int, r: float, x0: float, burn: int = 500) -> np.ndarray:
    """A standalone logistic map, for building independent series."""
    total = n + burn
    v = np.empty(total)
    v[0] = x0
    for t in range(total - 1):
        v[t + 1] = r * v[t] * (1 - v[t])
    return v[burn:]


def henon(n: int, burn: int = 500, a: float = 1.4, b: float = 0.3) -> np.ndarray:
    """The x component of the Henon map, a two-dimensional attractor."""
    total = n + burn
    u = np.empty(total)
    v = np.empty(total)
    u[0], v[0] = 0.1, 0.1
    for t in range(total - 1):
        u[t + 1] = 1 - a * u[t] ** 2 + v[t]
        v[t + 1] = b * u[t]
    return u[burn:]


# --------------------------------------------------------------------------
# time_delay_embed
# --------------------------------------------------------------------------


def test_time_delay_embed_matches_hand_built_example():
    """Row t must read [x(t), x(t-tau), x(t-2*tau)], present value first."""
    x = np.arange(8.0)
    manifold, times = time_delay_embed(x, E=3, tau=2)

    # The first four samples have no full history at E=3, tau=2, so t starts at 4.
    assert times.tolist() == [4, 5, 6, 7]
    assert manifold.tolist() == [
        [4.0, 2.0, 0.0],
        [5.0, 3.0, 1.0],
        [6.0, 4.0, 2.0],
        [7.0, 5.0, 3.0],
    ]


def test_time_delay_embed_tau_one_hand_built():
    x = np.array([10.0, 20.0, 30.0, 40.0])
    manifold, times = time_delay_embed(x, E=2, tau=1)
    assert times.tolist() == [1, 2, 3]
    assert manifold.tolist() == [[20.0, 10.0], [30.0, 20.0], [40.0, 30.0]]


def test_time_delay_embed_E1_is_the_series_itself():
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    manifold, times = time_delay_embed(x, E=1)
    assert manifold.shape == (5, 1)
    assert times.tolist() == [0, 1, 2, 3, 4]
    assert manifold.ravel().tolist() == x.tolist()


def test_time_delay_embed_row_count():
    for E, tau in [(2, 1), (3, 1), (4, 3), (1, 5)]:
        manifold, times = time_delay_embed(np.arange(50.0), E, tau)
        assert manifold.shape == (50 - (E - 1) * tau, E)
        assert len(times) == manifold.shape[0]


def test_time_delay_embed_rejects_bad_parameters():
    with pytest.raises(ValueError, match="E must be at least 1"):
        time_delay_embed(np.arange(10.0), E=0)
    with pytest.raises(ValueError, match="tau must be at least 1"):
        time_delay_embed(np.arange(10.0), E=2, tau=0)
    with pytest.raises(ValueError, match="too short"):
        time_delay_embed(np.arange(4.0), E=3, tau=2)


# --------------------------------------------------------------------------
# simplex_cross_map
# --------------------------------------------------------------------------


def test_cross_map_of_a_series_from_its_own_manifold_is_near_perfect():
    """Column 0 of the manifold *is* the target, so this is an upper bound."""
    x, _ = coupled_logistic(500, b_xy=0.0, b_yx=0.32)
    manifold, times = time_delay_embed(x, E=2)
    assert simplex_cross_map(manifold, times, x) > 0.99


def test_cross_map_of_independent_noise_is_near_zero():
    x, _ = coupled_logistic(500, b_xy=0.0, b_yx=0.32)
    noise = np.random.default_rng(7).normal(size=len(x))
    manifold, times = time_delay_embed(x, E=2)
    assert abs(simplex_cross_map(manifold, times, noise)) < 0.15


def test_exclusion_radius_suppresses_autocorrelation_driven_skill():
    """Two independent random walks cross-map well purely through smoothness.

    Without a Theiler window the nearest manifold neighbours are the temporally
    adjacent points, so the "prediction" is little more than interpolation. The
    window is what stops that being reported as a relationship.
    """
    rng = np.random.default_rng(3)
    walk = np.cumsum(rng.normal(size=600))
    unrelated = np.cumsum(rng.normal(size=600))
    manifold, times = time_delay_embed(walk, E=3)

    naive = simplex_cross_map(manifold, times, unrelated, exclusion_radius=0)
    windowed = simplex_cross_map(manifold, times, unrelated, exclusion_radius=50)
    assert naive > 0.7
    assert windowed < naive - 0.2

    # Genuine self-prediction survives the window; only the spurious part goes.
    assert simplex_cross_map(manifold, times, walk, exclusion_radius=50) > 0.95


def test_cross_map_handles_zero_distance_neighbours():
    """A periodic series repeats states exactly, so d_1 == 0 and exp(-d/d_1) blows up."""
    periodic = np.tile([0.1, 0.5, 0.9, 0.3], 100)
    manifold, times = time_delay_embed(periodic, E=2)
    rho = simplex_cross_map(manifold, times, periodic)
    assert np.isfinite(rho)
    assert rho > 0.99


def test_cross_map_returns_nan_rather_than_a_number_when_it_cannot_predict():
    """Degenerate cases must be visible, not quietly filled in.

    A Theiler window wider than the series leaves no admissible neighbour at
    all, and a constant target has no variance to correlate against. Both are
    caller errors of a kind that a plausible-looking rho would hide.
    """
    x = np.sin(np.arange(200) * 0.3)
    manifold, times = time_delay_embed(x, E=3)
    assert np.isnan(simplex_cross_map(manifold, times, x, exclusion_radius=500))
    assert np.isnan(simplex_cross_map(manifold, times, np.ones(200)))


def test_cross_map_works_at_E_one():
    """E=1 means two neighbours and a manifold that is just the series."""
    x = np.sin(np.arange(200) * 0.3)
    manifold, times = time_delay_embed(x, E=1)
    assert manifold.shape == (200, 1)
    assert simplex_cross_map(manifold, times, x) > 0.99


def test_cross_map_is_reproducible_and_seed_dependent():
    x, y = coupled_logistic(400, b_xy=0.0, b_yx=0.32)
    manifold, times = time_delay_embed(y, E=2)
    a = simplex_cross_map(manifold, times, x, lib_size=50, seed=1)
    b = simplex_cross_map(manifold, times, x, lib_size=50, seed=1)
    c = simplex_cross_map(manifold, times, x, lib_size=50, seed=2)
    assert a == b
    assert a != c


def test_cross_map_rejects_inconsistent_inputs():
    x = np.arange(30.0)
    manifold, times = time_delay_embed(x, E=2)
    with pytest.raises(ValueError, match="source_times has"):
        simplex_cross_map(manifold, times[:-1], x)
    with pytest.raises(ValueError, match="exceeds the"):
        simplex_cross_map(manifold, times, x, lib_size=1000)
    with pytest.raises(ValueError, match="below the"):
        simplex_cross_map(manifold, times, x, lib_size=3)
    with pytest.raises(ValueError, match="target has"):
        simplex_cross_map(manifold, times, x[:10])
    with pytest.raises(ValueError, match="exclusion_radius"):
        simplex_cross_map(manifold, times, x, exclusion_radius=-1)


# --------------------------------------------------------------------------
# The canonical validation
# --------------------------------------------------------------------------


def test_ccm_recovers_the_direction_of_a_unidirectionally_coupled_system():
    """THE test. b_xy=0 and b_yx=0.32, so X drives Y and Y does not drive X.

    A failure here saying "y causes x" means the direction convention in
    ``ccm.py`` is inverted. Fix the module, not this assertion.

    E=2 matches Sugihara (2012) for this system, and it is the right embedding:
    the driver is a one-dimensional map, and over-embedding inflates the
    spurious reverse skill (see the note in the module docstring).
    """
    x, y = coupled_logistic(1000, b_xy=0.0, b_yx=0.32)
    result = ccm(x, y, E=2, tau=1, n_bootstrap=50, seed=0)

    forward = result.x_causes_y
    reverse = result.y_causes_x

    # The field names must line up with the cross maps they were computed from:
    # evidence for "x causes y" is the skill of (Y xmap X).
    assert forward.xmap == "y xmap x"
    assert reverse.xmap == "x xmap y"

    assert forward.is_convergent(), forward.describe()
    assert not reverse.is_convergent(), reverse.describe()

    # Convergence, not just skill: the forward curve climbs a long way.
    assert forward.rho_at_max_lib > 0.9
    assert forward.delta_rho > 0.4
    assert forward.delta_rho_ci()[0] > 0.0

    # The reverse direction keeps some skill -- a driven series constrains its
    # driver's history -- but it is far weaker and does not converge.
    assert reverse.rho_at_max_lib < 0.7
    assert forward.rho_at_max_lib > reverse.rho_at_max_lib + 0.4
    assert forward.delta_rho > 3 * reverse.delta_rho

    assert result.dominant_direction() is forward


def test_ccm_direction_is_not_an_artefact_of_argument_order():
    """Swapping the arguments must swap the reported direction, not the verdict.

    The driver is now the *second* argument, so the surviving direction has to
    be ``y_causes_x``. Only the full-library skill is compared numerically: the
    smaller libraries are random draws, and which independent stream feeds which
    direction follows argument order.
    """
    x, y = coupled_logistic(1000, b_xy=0.0, b_yx=0.32)
    swapped = ccm(y, x, E=2, tau=1, n_bootstrap=50, seed=0)

    assert swapped.y_causes_x.is_convergent(), swapped.y_causes_x.describe()
    assert not swapped.x_causes_y.is_convergent(), swapped.x_causes_y.describe()
    assert swapped.y_causes_x.xmap == "x xmap y"

    direct = ccm(x, y, E=2, tau=1, n_bootstrap=50, seed=0)
    assert swapped.y_causes_x.rho_at_max_lib == pytest.approx(direct.x_causes_y.rho_at_max_lib)
    assert swapped.x_causes_y.rho_at_max_lib == pytest.approx(direct.y_causes_x.rho_at_max_lib)


def test_ccm_cannot_demonstrate_convergence_on_a_periodic_system():
    """A limitation worth pinning down: near-perfect skill without a rise.

    Setting ``b_yx=0`` leaves Y as a bare r=3.5 logistic map, which sits on a
    period-4 orbit rather than a chaotic attractor, and it drags the X it drives
    onto a period-4 orbit too. Four states is an attractor a library of ten
    already covers completely, so both directions cross-map almost perfectly at
    every library size and neither converges. Reporting causation from the skill
    alone would claim a bidirectional link that is not there.
    """
    x, y = coupled_logistic(1000, b_xy=0.32, b_yx=0.0)
    assert len(np.unique(np.round(y, 6))) == 4

    result = ccm(x, y, E=2, tau=1, n_bootstrap=50, seed=0)
    for direction in (result.x_causes_y, result.y_causes_x):
        assert direction.rho_at_max_lib > 0.99
        assert direction.delta_rho < 0.01
        assert not direction.is_convergent(), direction.describe()
    assert result.dominant_direction() is None


def test_ccm_finds_nothing_between_independent_logistic_maps():
    """Two uncoupled chaotic maps: no detection in either direction."""
    for r_x, r_y, x0, y0 in [(3.8, 3.7, 0.4, 0.2), (3.7, 3.9, 0.11, 0.83)]:
        x = logistic(1000, r_x, x0)
        y = logistic(1000, r_y, y0)
        result = ccm(x, y, E=2, tau=1, n_bootstrap=50, seed=0)

        assert not result.x_causes_y.is_convergent(), result.x_causes_y.describe()
        assert not result.y_causes_x.is_convergent(), result.y_causes_x.describe()
        assert result.dominant_direction() is None
        assert abs(result.x_causes_y.rho_at_max_lib) < 0.3
        assert abs(result.y_causes_x.rho_at_max_lib) < 0.3


def test_reverse_direction_can_converge_weakly_and_comparison_resolves_it():
    """A limitation of any per-direction threshold, pinned down so it stays visible.

    The system is still strictly one-way, but on this trajectory the reverse
    direction clears :meth:`DirectionResult.is_convergent` at the permissive
    defaults, reproducibly and for every seed tried -- the driven series
    constrains its driver's history enough for a small but real rise. It is not
    a knife-edge artefact of one library draw, and no single-direction threshold
    excludes it reliably: across initial conditions the reverse rise ranges from
    roughly -0.01 to +0.22, overlapping anything one might set.

    So the comparison between directions, not either verdict alone, is what
    identifies the driver.
    """
    x, y = coupled_logistic(600, b_xy=0.0, b_yx=0.32, x0=0.7, y0=0.3)

    for seed in range(4):
        result = ccm(x, y, E=2, tau=1, n_bootstrap=50, seed=seed)
        assert result.y_causes_x.is_convergent(), result.y_causes_x.describe()
        # Yet it is nowhere near the real direction, and the comparison says so.
        assert result.y_causes_x.rho_at_max_lib < 0.5
        assert result.x_causes_y.rho_at_max_lib > 0.9
        assert result.dominant_direction() is result.x_causes_y


def test_canonical_verdict_is_stable_across_bootstrap_seeds():
    """The headline result must not depend on which library subsets were drawn."""
    x, y = coupled_logistic(1000, b_xy=0.0, b_yx=0.32)
    for seed in range(4):
        result = ccm(x, y, E=2, tau=1, n_bootstrap=50, seed=seed)
        assert result.x_causes_y.is_convergent(), result.x_causes_y.describe()
        assert not result.y_causes_x.is_convergent(), result.y_causes_x.describe()


def test_ccm_finds_nothing_when_both_couplings_are_zero():
    """The canonical system with the coupling switched off in both directions."""
    x, y = coupled_logistic(1000, b_xy=0.0, b_yx=0.0)
    result = ccm(x, y, E=2, tau=1, n_bootstrap=50, seed=0)
    assert not result.x_causes_y.is_convergent(), result.x_causes_y.describe()
    assert not result.y_causes_x.is_convergent(), result.y_causes_x.describe()


# --------------------------------------------------------------------------
# ccm bookkeeping
# --------------------------------------------------------------------------


def test_ccm_exposes_rho_at_every_library_size_with_intervals():
    x, y = coupled_logistic(400, b_xy=0.0, b_yx=0.32)
    result = ccm(x, y, E=2, tau=1, lib_sizes=[20, 60, 180, 399], n_bootstrap=20, seed=0)

    assert isinstance(result, CCMResult)
    assert result.lib_sizes.tolist() == [20, 60, 180, 399]
    for direction in (result.x_causes_y, result.y_causes_x):
        assert direction.rho.shape == (4,)
        assert direction.rho_samples.shape == (4, 20)
        assert np.all(direction.rho_ci_low <= direction.rho + 1e-12)
        assert np.all(direction.rho_ci_high >= direction.rho - 1e-12)
        assert direction.delta_rho == pytest.approx(direction.rho[-1] - direction.rho[0])


def test_ccm_skill_rises_monotonically_enough_for_the_driven_direction():
    """Convergence means the curve climbs, not that it is exactly monotone."""
    x, y = coupled_logistic(1000, b_xy=0.0, b_yx=0.32)
    rho = ccm(x, y, E=2, tau=1, n_bootstrap=50, seed=0).x_causes_y.rho
    assert np.all(np.diff(rho) > -0.02)
    assert rho[-1] > rho[0]


def test_ccm_full_library_replicates_are_identical():
    """At the full library every draw is the same, so the interval collapses."""
    x, y = coupled_logistic(300, b_xy=0.0, b_yx=0.32)
    result = ccm(x, y, E=2, tau=1, lib_sizes=[20, 299], n_bootstrap=10, seed=0)
    samples = result.x_causes_y.rho_samples[-1]
    assert np.allclose(samples, samples[0])
    assert result.x_causes_y.rho_ci_low[-1] == pytest.approx(result.x_causes_y.rho_ci_high[-1])


def test_ccm_is_reproducible():
    x, y = coupled_logistic(300, b_xy=0.0, b_yx=0.32)
    a = ccm(x, y, E=2, n_bootstrap=10, seed=5)
    b = ccm(x, y, E=2, n_bootstrap=10, seed=5)
    assert np.allclose(a.x_causes_y.rho_samples, b.x_causes_y.rho_samples)
    assert np.allclose(a.y_causes_x.rho_samples, b.y_causes_x.rho_samples)


def test_ccm_default_library_sizes_span_the_manifold():
    x, y = coupled_logistic(500, b_xy=0.0, b_yx=0.32)
    result = ccm(x, y, E=3, tau=1, n_bootstrap=5, seed=0)
    n_points = 500 - (3 - 1)
    assert result.lib_sizes[0] >= 3 + 2
    assert result.lib_sizes[-1] == n_points
    assert np.all(np.diff(result.lib_sizes) > 0)


def test_ccm_rejects_bad_arguments():
    x, y = coupled_logistic(200, b_xy=0.0, b_yx=0.32)
    with pytest.raises(ValueError, match="same length"):
        ccm(x, y[:-1])
    with pytest.raises(ValueError, match="n_bootstrap"):
        ccm(x, y, n_bootstrap=0)
    with pytest.raises(ValueError, match="at least E\\+2"):
        ccm(x, y, E=3, lib_sizes=[4, 100])
    with pytest.raises(ValueError, match="exceeds"):
        ccm(x, y, E=3, lib_sizes=[10, 10_000])
    with pytest.raises(ValueError, match="at least two library sizes"):
        ccm(x, y, E=3, lib_sizes=[50])


def test_describe_names_both_the_hypothesis_and_the_cross_map():
    x, y = coupled_logistic(300, b_xy=0.0, b_yx=0.32)
    text = ccm(x, y, E=2, n_bootstrap=10, seed=0).describe()
    assert "x -> y  [y xmap x]" in text
    assert "y -> x  [x xmap y]" in text


# --------------------------------------------------------------------------
# optimal_embedding_dimension
# --------------------------------------------------------------------------


def test_optimal_embedding_dimension_of_a_one_dimensional_map():
    assert optimal_embedding_dimension(logistic(1000, 3.8, 0.4), max_E=8) <= 2


def test_optimal_embedding_dimension_of_the_henon_map():
    """Henon is two-dimensional, and one lag is provably not enough to unfold it."""
    assert optimal_embedding_dimension(henon(1000), max_E=8) == 2


def test_optimal_embedding_dimension_stays_in_range():
    x = logistic(600, 3.9, 0.31)
    for max_E in (1, 3, 6):
        E = optimal_embedding_dimension(x, max_E=max_E)
        assert 1 <= E <= max_E


def test_optimal_embedding_dimension_rejects_a_too_short_series():
    with pytest.raises(ValueError, match="too short"):
        optimal_embedding_dimension(np.arange(6.0), max_E=5)
    with pytest.raises(ValueError, match="max_E"):
        optimal_embedding_dimension(np.arange(60.0), max_E=0)


# --------------------------------------------------------------------------
# surrogates
# --------------------------------------------------------------------------


def test_ebisuzaki_surrogate_preserves_the_power_spectrum():
    x, _ = coupled_logistic(512, b_xy=0.0, b_yx=0.32)
    s = ebisuzaki_surrogate(x, seed=0)
    assert np.allclose(np.abs(np.fft.rfft(x)), np.abs(np.fft.rfft(s)))
    # Same spectrum implies the same mean (DC term) and variance (Parseval).
    assert s.mean() == pytest.approx(x.mean())
    assert s.var() == pytest.approx(x.var())
    assert not np.allclose(x, s)
    assert np.isrealobj(s)


def test_ebisuzaki_surrogate_handles_odd_lengths():
    x, _ = coupled_logistic(511, b_xy=0.0, b_yx=0.32)
    s = ebisuzaki_surrogate(x, seed=1)
    assert len(s) == 511
    assert np.allclose(np.abs(np.fft.rfft(x)), np.abs(np.fft.rfft(s)))


def test_ebisuzaki_surrogates_differ_between_draws():
    x, _ = coupled_logistic(256, b_xy=0.0, b_yx=0.32)
    assert not np.allclose(ebisuzaki_surrogate(x, seed=0), ebisuzaki_surrogate(x, seed=1))


def test_circular_shift_surrogate_keeps_every_value():
    x, _ = coupled_logistic(200, b_xy=0.0, b_yx=0.32)
    s = circular_shift_surrogate(x, seed=0)
    assert np.allclose(np.sort(s), np.sort(x))
    assert not np.allclose(s, x)


def test_surrogate_test_rejects_the_null_for_the_driven_direction():
    x, y = coupled_logistic(600, b_xy=0.0, b_yx=0.32)
    result = surrogate_test(x, y, E=2, tau=1, n_surrogates=100, seed=0)

    assert result.x_causes_y.xmap == "y xmap x"
    assert result.x_causes_y.is_significant()
    assert result.x_causes_y.rho > 0.9
    # The null is what a series with the same spectrum but no coupling achieves.
    assert np.nanmax(result.x_causes_y.null_rho) < result.x_causes_y.rho


def test_surrogate_significance_alone_does_not_resolve_direction():
    """Documented limitation: this is why :meth:`is_convergent` exists.

    Under strong one-way forcing the reverse cross map still beats a
    spectrum-matched null comfortably, so the surrogate test flags both
    directions. Only the convergence criterion separates them.
    """
    x, y = coupled_logistic(1000, b_xy=0.0, b_yx=0.32)
    test = surrogate_test(x, y, E=2, tau=1, n_surrogates=100, seed=0)
    assert test.x_causes_y.is_significant()
    assert test.y_causes_x.is_significant()

    result = ccm(x, y, E=2, tau=1, n_bootstrap=50, seed=0)
    assert result.x_causes_y.is_convergent()
    assert not result.y_causes_x.is_convergent()


def test_surrogate_test_accepts_the_null_for_independent_series():
    x = logistic(600, 3.8, 0.4)
    y = logistic(600, 3.7, 0.83)
    for method in ("ebisuzaki", "circular"):
        result = surrogate_test(x, y, E=2, tau=1, n_surrogates=100, method=method, seed=0)
        assert not result.x_causes_y.is_significant(), result.x_causes_y.describe()
        assert not result.y_causes_x.is_significant(), result.y_causes_x.describe()


def test_surrogate_p_value_can_never_be_zero():
    """The observed value counts as one of its own draws, so p >= 1/(n+1)."""
    x, y = coupled_logistic(400, b_xy=0.0, b_yx=0.32)
    result = surrogate_test(x, y, E=2, n_surrogates=50, seed=0)
    for direction in (result.x_causes_y, result.y_causes_x):
        assert direction.p_value >= 1.0 / 51.0
        assert len(direction.null_rho) == 50


def test_surrogate_test_cannot_be_significant_when_the_skill_is_undefined():
    """An unevaluable cross map must fail to reject, not maximally reject.

    ``nan`` compares False against everything, so counting exceedances with a
    bare ``null >= observed`` scores a nan rho as "no surrogate beat it" and
    hands back p = 1/(n+1). Both the exclusion radius here and the constant
    series below make every cross map undefined; a p-value near zero would be
    reported as the strongest possible evidence for a link that could not even
    be measured.
    """
    x, y = coupled_logistic(300, b_xy=0.0, b_yx=0.32)

    blind = surrogate_test(x, y, E=2, n_surrogates=50, seed=0, exclusion_radius=10_000)
    for direction in (blind.x_causes_y, blind.y_causes_x):
        assert np.isnan(direction.rho)
        assert direction.p_value == 1.0
        assert not direction.is_significant()

    flat = surrogate_test(np.ones(300), y, E=2, n_surrogates=50, seed=0)
    assert np.isnan(flat.x_causes_y.rho)
    assert not flat.x_causes_y.is_significant()


def test_surrogate_test_counts_unevaluable_nulls_against_rejection(monkeypatch):
    """A surrogate whose own skill is undefined must not help the observed win.

    Same trap from the other side: a nan null draw is not "a surrogate that
    scored below the observed value". Counting it that way would let a null
    distribution that silently failed to evaluate manufacture significance.
    """
    from deepfeatselect import ccm as ccm_module

    monkeypatch.setitem(ccm_module._SURROGATE_METHODS, "broken", lambda v, rng: np.ones(len(v)))

    x, y = coupled_logistic(300, b_xy=0.0, b_yx=0.32)
    result = surrogate_test(x, y, E=2, n_surrogates=20, method="broken", seed=0)

    for direction in (result.x_causes_y, result.y_causes_x):
        assert np.isfinite(direction.rho)
        assert np.all(np.isnan(direction.null_rho))
        assert direction.p_value == 1.0
        assert not direction.is_significant()


def test_surrogate_test_rejects_bad_arguments():
    x, y = coupled_logistic(200, b_xy=0.0, b_yx=0.32)
    with pytest.raises(ValueError, match="method must be one of"):
        surrogate_test(x, y, method="bogus")
    with pytest.raises(ValueError, match="n_surrogates"):
        surrogate_test(x, y, n_surrogates=0)
    with pytest.raises(ValueError, match="same length"):
        surrogate_test(x, y[:-1])
