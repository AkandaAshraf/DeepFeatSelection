"""Tests for the synthetic generators.

The generators exist so other methods can be scored against them, which only
works if the ground truth they advertise is actually the ground truth of the data
they emit.  These tests check that correspondence, not just that arrays come out
the right shape: that the declared direct causes really do carry more information
about the target than the declared irrelevant columns, that the declared coupling
direction really is the only direction the dynamics run in, and that the
redundancy demonstration really does destroy leave-one-out importance.

Statistics here are deliberately plain and deterministic -- a fixed-bin plug-in
mutual information rather than a nearest-neighbour estimator -- so a failure
means the data changed, not that an estimator drew different random neighbours.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from deepfeatselect.synthetic import (
    SyntheticDataset,
    coupled_logistic,
    independent_logistic,
    nonlinear_scm,
    redundancy_demo,
    rossler_lorenz,
)

BINS = 16


def _codes(v: np.ndarray, bins: int) -> np.ndarray:
    """Equal-count bin indices; inputs that are already discrete pass through.

    Equal-count rather than equal-width because the logistic map's invariant
    density piles up at the edges of the unit interval, and equal-width bins
    there leave most of the range nearly empty.
    """
    v = np.asarray(v, dtype=np.float64)
    unique = np.unique(v)
    if unique.size <= bins:
        return np.searchsorted(unique, v)
    edges = np.quantile(v, np.linspace(0.0, 1.0, bins + 1))
    return np.clip(np.digitize(v, edges[1:-1]), 0, bins - 1)


def mutual_info(a: np.ndarray, b: np.ndarray, bins: int = BINS) -> float:
    """Plug-in mutual information in nats, from a binned joint histogram."""
    ia, ib = _codes(a, bins), _codes(b, bins)
    joint = np.zeros((int(ia.max()) + 1, int(ib.max()) + 1))
    np.add.at(joint, (ia, ib), 1.0)
    p = joint / joint.sum()
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())


def shuffle_floor(a: np.ndarray, b: np.ndarray, seed: int = 0, repeats: int = 5) -> float:
    """The plug-in estimate's positive bias, measured on shuffled surrogates.

    A binned MI is biased upwards by roughly ``(bins-1)**2 / (2n)`` even for
    independent inputs, so the tests below compare against a measured baseline
    rather than a hard-coded number.  Valid only for independent draws; for a
    time series use :func:`time_shift_floor`.
    """
    rng = np.random.default_rng(seed)
    return float(np.mean([mutual_info(a, rng.permutation(b)) for _ in range(repeats)]))


def time_shift_floor(a: np.ndarray, b: np.ndarray, repeats: int = 9) -> float:
    """Bias floor for two serially correlated series, from cyclic time shifts.

    Shuffling is the wrong null here: it destroys each series' autocorrelation as
    well as the pairing, so the effective sample size of the surrogate is much
    larger than that of the real data and the floor comes out too low.  A cyclic
    shift keeps both trajectories and both autocorrelation structures exactly
    intact and only breaks the alignment between them, which is the standard
    surrogate for this question and the one CCM significance tests use.
    """
    n = len(b)
    offsets = np.linspace(n // 10, n - n // 10, repeats).astype(int)
    return float(np.mean([mutual_info(a, np.roll(b, int(k))) for k in offsets]))


def feature_mi(data: SyntheticDataset) -> np.ndarray:
    return np.array([mutual_info(col, data.y) for col in data.x.T])


# --------------------------------------------------------------------------
# nonlinear_scm
# --------------------------------------------------------------------------


def test_scm_shapes():
    data = nonlinear_scm(n=500, seed=0)
    assert data.x.shape == (500, 9)
    assert data.y.shape == (500,)
    assert data.n_samples == 500
    assert data.n_features == 9
    assert set(np.unique(data.y)) <= {0, 1}
    for mask in (data.direct_causes, data.markov_blanket, data.irrelevant):
        assert mask.dtype == bool
        assert mask.shape == (9,)


def test_scm_groups_is_the_identity_map():
    """Every synthetic feature is continuous, so column j belongs to feature j."""
    data = nonlinear_scm(n=100, seed=0)
    assert np.array_equal(data.groups, np.arange(data.x.shape[1]))


def test_scm_is_reproducible_under_seed():
    a = nonlinear_scm(n=400, seed=7)
    b = nonlinear_scm(n=400, seed=7)
    assert np.array_equal(a.x, b.x)
    assert np.array_equal(a.y, b.y)

    c = nonlinear_scm(n=400, seed=8)
    assert not np.array_equal(a.x, c.x)


def test_scm_masks_are_consistent():
    data = nonlinear_scm(n=200, seed=0)
    assert np.all(data.markov_blanket[data.direct_causes])
    assert not np.any(data.irrelevant & data.direct_causes)
    assert not np.any(data.irrelevant & data.markov_blanket)
    # The child of the target is the reason the two sets differ at all.
    assert data.markov_blanket.sum() > data.direct_causes.sum()
    assert np.all(data.markov_blanket[data.effects])
    assert not np.any(data.effects & data.direct_causes)


def test_scm_masks_name_the_expected_columns():
    data = nonlinear_scm(n=200, seed=0)
    assert data.names(data.direct_causes) == ["z", "x_cause1", "x_cause2"]
    assert data.names(data.markov_blanket) == ["z", "x_cause1", "x_cause2", "x_effect"]
    assert data.names(data.irrelevant) == ["x_noise1", "x_noise2", "x_noise3"]
    assert data.names(data.effects) == ["x_effect"]


def test_scm_confounder_proxies_are_in_no_mask():
    """They are predictive but neither causal nor blanket members; that is the trap."""
    data = nonlinear_scm(n=200, seed=0)
    proxies = data.confounded
    assert data.names(proxies) == ["x_conf1", "x_conf2"]
    assert not np.any(proxies & data.direct_causes)
    assert not np.any(proxies & data.markov_blanket)
    assert not np.any(proxies & data.irrelevant)


def test_scm_classes_are_roughly_balanced():
    for seed in range(4):
        rate = nonlinear_scm(n=2000, seed=seed).y.mean()
        assert 0.4 < rate < 0.6, f"seed {seed} gave a {rate:.2f} positive rate"


def test_scm_effect_is_the_most_informative_column():
    """The headline property: a non-cause outranks every cause on prediction.

    ``x_effect`` is a child of the target, so its causal effect is zero, yet it
    is a low-noise readout of the realised label and beats every true cause by a
    wide margin.  If this ever stops holding the module has lost its point.
    """
    data = nonlinear_scm(n=4000, seed=0)
    mi = feature_mi(data)
    best = int(np.argmax(mi))
    assert data.feature_names[best] == "x_effect"
    assert mi[data.effects][0] > 4.0 * mi[~data.effects].max()
    assert roc_auc_score(data.y, data.x[:, best]) > 0.95


def test_scm_direct_causes_beat_every_irrelevant_column():
    data = nonlinear_scm(n=4000, seed=0)
    mi = feature_mi(data)
    floor = mi[data.irrelevant].max()
    assert mi[data.direct_causes].min() > 5.0 * floor


def test_scm_quadratic_cause_is_nearly_invisible_to_monotone_association():
    """``x_cause2`` enters ``g`` through its square, so rank statistics miss it.

    A method that scores features by correlation or AUC ranks it barely above the
    noise columns while its mutual information with the target is as large as any
    other cause's.  This is the second way the table punishes a shortcut.
    """
    data = nonlinear_scm(n=4000, seed=0)
    mi = feature_mi(data)
    i1 = data.feature_names.index("x_cause1")
    i2 = data.feature_names.index("x_cause2")

    def auc_gap(i: int) -> float:
        return abs(roc_auc_score(data.y, data.x[:, i]) - 0.5)

    assert auc_gap(i1) > 5.0 * auc_gap(i2)
    assert mi[i2] > 0.5 * mi[i1]


def test_scm_confounder_proxies_are_marginally_predictive():
    """Not causes, but not ignorable either: both carry real marginal signal."""
    data = nonlinear_scm(n=4000, seed=0)
    mi = feature_mi(data)
    assert mi[data.confounded].min() > 3.0 * mi[data.irrelevant].max()


def test_scm_confounder_proxies_are_associated_with_each_other():
    """Driven by a common z with no edge between them: a spurious pairwise link."""
    data = nonlinear_scm(n=4000, seed=0)
    conf = data.x[:, data.confounded]
    noise = data.x[:, data.irrelevant]
    linked = mutual_info(conf[:, 0], conf[:, 1])
    assert linked > 3.0 * shuffle_floor(conf[:, 0], conf[:, 1])
    assert linked > 4.0 * mutual_info(noise[:, 0], noise[:, 1])


def test_scm_zero_noise_makes_the_structural_equations_exact():
    """A direct check that each column is wired to the parents it claims."""
    data = nonlinear_scm(n=500, noise=0.0, seed=3)
    col = {name: data.x[:, i] for i, name in enumerate(data.feature_names)}
    assert np.allclose(col["x_conf1"], np.tanh(2.0 * col["z"]))
    assert np.allclose(col["x_conf2"], col["z"] ** 2 - 1.0)
    assert np.allclose(
        col["x_effect"], 1.5 * data.y + 0.5 * np.sin(3.0 * col["x_cause1"])
    )


# --------------------------------------------------------------------------
# coupled_logistic and independent_logistic
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_coupled_logistic_stays_in_the_unit_interval(seed):
    """The map is only meaningful while it is invariant on [0, 1]."""
    data = coupled_logistic(n=3000, seed=seed)
    for name in ("x", "y"):
        series = data[name]
        assert np.isfinite(series).all()
        assert series.min() >= 0.0
        assert series.max() <= 1.0


def test_coupled_logistic_returns_n_points_after_the_burn_in():
    data = coupled_logistic(n=1234, burn_in=321, seed=0)
    assert data["x"].shape == (1234,)
    assert data["y"].shape == (1234,)
    assert data["burn_in"] == 321


def test_coupled_logistic_is_chaotic_not_a_fixed_point():
    """A collapsed trajectory would pass every bound check and carry no signal."""
    data = coupled_logistic(n=3000, seed=0)
    for name in ("x", "y"):
        assert data[name].std() > 0.1
        assert np.unique(np.round(data[name], 6)).size > 2000


def test_coupled_logistic_is_reproducible_under_seed():
    a = coupled_logistic(n=500, seed=3)
    b = coupled_logistic(n=500, seed=3)
    c = coupled_logistic(n=500, seed=4)
    assert np.array_equal(a["x"], b["x"]) and np.array_equal(a["y"], b["y"])
    assert not np.array_equal(a["x"], c["x"])


@pytest.mark.parametrize(
    ("x_to_y", "y_to_x", "expected"),
    [
        (0.32, 0.0, "x->y"),
        (0.0, 0.32, "y->x"),
        (0.2, 0.2, "x<->y"),
        (0.0, 0.0, "none"),
    ],
)
def test_coupled_logistic_reports_the_true_direction(x_to_y, y_to_x, expected):
    data = coupled_logistic(n=500, coupling_x_to_y=x_to_y, coupling_y_to_x=y_to_x, seed=0)
    assert data["true_direction"] == expected
    assert data["coupling_x_to_y"] == x_to_y
    assert data["coupling_y_to_x"] == y_to_x


def test_coupled_logistic_default_coupling_runs_only_one_way():
    """The declared direction has to be the only direction in the dynamics.

    With ``coupling_y_to_x=0`` the driver is autonomous, so changing how hard it
    pushes the response must leave the driver's own trajectory bit-identical.
    """
    base = coupled_logistic(n=500, seed=1)
    stronger = coupled_logistic(n=500, coupling_x_to_y=0.1, seed=1)
    reversed_ = coupled_logistic(n=500, coupling_x_to_y=0.0, coupling_y_to_x=0.2, seed=1)

    assert np.array_equal(base["x"], stronger["x"])
    assert not np.array_equal(base["y"], stronger["y"])
    # And the reverse coupling does reach x, so the test above is not vacuous.
    assert not np.array_equal(base["x"], reversed_["x"])


def test_coupled_logistic_naive_reversal_collapses_onto_a_cycle():
    """Reversing the coupling without reversing the growth rates is a trap.

    ``r_y=3.5`` is in the period-4 window, so promoting ``y`` to driver makes the
    driver a four-point cycle, and it entrains ``x`` onto a cycle as well.  Both
    returned series then take four distinct values.  The ``y->x`` label stays
    true while the data stops being able to support the inference, which is the
    combination that produces a benchmark every method passes.  Pinned here so
    nobody builds a reversed-direction case on it by accident.
    """
    data = coupled_logistic(n=3000, coupling_x_to_y=0.0, coupling_y_to_x=0.32, seed=0)
    assert data["true_direction"] == "y->x"
    assert np.unique(np.round(data["x"], 8)).size == 4
    assert np.unique(np.round(data["y"], 8)).size == 4


@pytest.mark.parametrize("seed", range(4))
def test_coupled_logistic_reversed_case_needs_swapped_growth_rates(seed):
    """The documented recipe for a usable ``y->x`` case, checked end to end."""
    data = coupled_logistic(
        n=3000, r_x=3.5, r_y=3.8, coupling_x_to_y=0.0, coupling_y_to_x=0.32, seed=seed
    )
    assert data["true_direction"] == "y->x"
    for name in ("x", "y"):
        assert 0.0 <= data[name].min() and data[name].max() <= 1.0
        assert np.unique(np.round(data[name], 6)).size > 2500
        assert data[name].std() > 0.1

    # The imprint runs the way the label says: the response carries the driver's
    # signature one step later, mirroring the default x->y case.
    floor = time_shift_floor(data["x"], data["y"])
    assert mutual_info(data["y"][:-1], data["x"][1:]) > 4.0 * floor


def test_coupled_logistic_rejects_parameters_that_diverge():
    with pytest.raises(ValueError, match="left \\[0, 1\\]"):
        coupled_logistic(n=100, coupling_x_to_y=5.0, seed=0)


def test_independent_logistic_labels_no_coupling():
    data = independent_logistic(n=500, seed=0)
    assert data["system"] == "independent_logistic"
    assert data["true_direction"] == "none"
    assert data["coupling_x_to_y"] == 0.0
    assert data["coupling_y_to_x"] == 0.0


@pytest.mark.parametrize("seed", range(4))
def test_independent_logistic_is_genuinely_chaotic(seed):
    """A control has to be hard, and this one was silently trivial once.

    ``coupled_logistic``'s ``r_y=3.5`` is in the period-4 window, so an uncoupled
    response takes four distinct values forever.  Every independence test passes
    on a four-point cycle, which proves nothing about the test.
    """
    data = independent_logistic(n=3000, seed=seed)
    for name in ("x", "y"):
        assert np.unique(np.round(data[name], 6)).size > 2000
        assert data[name].std() > 0.15


@pytest.mark.parametrize("seed", range(4))
def test_independent_logistic_has_no_mutual_information(seed):
    """The negative control has to be genuinely uninformative, not merely weak.

    Compared against the estimator's own bias on time-shifted surrogates of the
    same two series, so the threshold does not have to be guessed.
    """
    data = independent_logistic(n=3000, seed=seed)
    floor = time_shift_floor(data["x"], data["y"])

    observed = mutual_info(data["x"], data["y"])
    assert observed < 2.0 * floor, f"{observed:.4f} nats against a {floor:.4f} floor"

    # The same at a one-step lag, which is where a driver's imprint would sit.
    lagged = mutual_info(data["x"][:-1], data["y"][1:])
    assert lagged < 2.0 * floor


def test_coupled_logistic_carries_far_more_information_than_the_control():
    """Confirms the control is calibrating a threshold, not just a quiet dataset."""
    coupled = coupled_logistic(n=3000, seed=0)
    control = independent_logistic(n=3000, seed=0)
    linked = mutual_info(coupled["x"][:-1], coupled["y"][1:])
    unlinked = mutual_info(control["x"][:-1], control["y"][1:])
    assert linked > 4.0 * time_shift_floor(coupled["x"], coupled["y"])
    assert linked > 5.0 * unlinked


# --------------------------------------------------------------------------
# rossler_lorenz
# --------------------------------------------------------------------------


def test_rossler_lorenz_shapes_and_observables():
    data = rossler_lorenz(n=800, seed=0)
    assert data["rossler"].shape == (800, 3)
    assert data["lorenz"].shape == (800, 3)
    # The scalars are the components the coupling actually runs through.
    assert np.array_equal(data["x"], data["rossler"][:, 1])
    assert np.array_equal(data["y"], data["lorenz"][:, 1])
    assert data["true_direction"] == "x->y"
    assert data["coupling_y_to_x"] == 0.0


@pytest.mark.parametrize("coupling", [0.0, 1.0, 2.0, 3.0])
def test_rossler_lorenz_does_not_blow_up(coupling):
    """RK4 on a forced Lorenz is the part most likely to go unstable."""
    data = rossler_lorenz(n=800, coupling=coupling, seed=0)
    assert np.isfinite(data["rossler"]).all()
    assert np.isfinite(data["lorenz"]).all()
    # Both attractors are known to live well inside these bounds; a diverging
    # integration leaves them long before it reaches inf.
    assert np.abs(data["rossler"]).max() < 50.0
    assert np.abs(data["lorenz"]).max() < 150.0
    # And has not collapsed onto a fixed point.
    assert data["rossler"].std(axis=0).min() > 1.0
    assert data["lorenz"].std(axis=0).min() > 1.0


def test_rossler_lorenz_is_reproducible_under_seed():
    a = rossler_lorenz(n=300, seed=2)
    b = rossler_lorenz(n=300, seed=2)
    c = rossler_lorenz(n=300, seed=5)
    assert np.array_equal(a["lorenz"], b["lorenz"])
    assert not np.array_equal(a["lorenz"], c["lorenz"])


def test_rossler_lorenz_coupling_is_strictly_one_way():
    """The Rossler is autonomous, so its trajectory cannot depend on ``coupling``."""
    driven = rossler_lorenz(n=800, coupling=2.0, seed=0)
    free = rossler_lorenz(n=800, coupling=0.0, seed=0)
    assert np.array_equal(driven["rossler"], free["rossler"])
    assert not np.array_equal(driven["lorenz"], free["lorenz"])


def test_rossler_lorenz_rejects_a_step_too_large_to_integrate():
    with pytest.raises(ValueError, match="diverged"):
        rossler_lorenz(n=100, dt=1.0, seed=0)


# --------------------------------------------------------------------------
# redundancy_demo
# --------------------------------------------------------------------------


def _knn_r2(x: np.ndarray, y: np.ndarray, columns: list[int]) -> float:
    """Held-out R^2 of a nearest-neighbour fit on a subset of columns.

    A stand-in for the refit step of LOCO. Nearest neighbours rather than a
    network because the point is a property of the data, and any consistent
    learner shows it -- with far less machinery and no training time.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x[:, columns], y, test_size=0.3, random_state=0
    )
    model = KNeighborsRegressor(n_neighbors=3).fit(x_train, y_train)
    return float(r2_score(y_test, model.predict(x_test)))


def test_redundancy_demo_metadata():
    data = redundancy_demo(n=500, seed=0)
    assert data["feature_names"] == ["driver", "proxy_cos", "proxy_sin", "unrelated"]
    assert data["driver"] == "driver"
    assert data["feature_names"][data["driver_index"]] == data["driver"]
    assert data["x"].shape == (500, 4)
    assert data["y"].shape == (500,)


def test_redundancy_demo_is_reproducible_under_seed():
    a = redundancy_demo(n=400, seed=1)
    b = redundancy_demo(n=400, seed=1)
    c = redundancy_demo(n=400, seed=2)
    assert np.array_equal(a["x"], b["x"]) and np.array_equal(a["y"], b["y"])
    assert not np.array_equal(a["x"], c["x"])


def test_redundancy_driver_is_recoverable_from_the_proxies_exactly():
    """The claim the whole demonstration rests on, checked to machine precision."""
    data = redundancy_demo(n=3000, seed=0)
    x = data["x"]
    recovered = np.mod(np.arctan2(x[:, 2], x[:, 1]) / (2.0 * np.pi), 1.0)
    assert np.abs(recovered - x[:, 0]).max() < 1e-12
    # Neither proxy alone would do it: each is two-to-one on the unit interval.
    assert 0.0 < x[:, 0].min() and x[:, 0].max() < 1.0


def test_redundancy_target_is_a_deterministic_function_of_the_driver():
    data = redundancy_demo(n=1000, seed=0)
    driver = data["x"][:, 0]
    assert np.allclose(data["y"], 3.9 * driver * (1.0 - driver))


def test_redundancy_removing_the_true_driver_costs_nothing():
    """LOCO importance of the system's only cause is zero, and stays zero.

    This is the failure the module exists to make concrete: the refit model
    reconstructs the driver from its two proxies and loses no accuracy at all, so
    a removal-based score attributes nothing to a variable that causes
    everything.
    """
    data = redundancy_demo(n=3000, seed=0)
    x, y = data["x"], data["y"]
    names = data["feature_names"]

    full = _knn_r2(x, y, list(range(len(names))))
    without_driver = _knn_r2(x, y, [i for i, n in enumerate(names) if n != "driver"])

    assert full > 0.99
    assert full - without_driver < 0.01

    # Every single-column deletion is equally free, for the same reason.
    for dropped in range(len(names)):
        kept = [i for i in range(len(names)) if i != dropped]
        assert full - _knn_r2(x, y, kept) < 0.01, f"dropping {names[dropped]} was not free"


def test_redundancy_proxy_cos_alone_determines_the_target():
    """One proxy carries the whole target, even though it cannot carry the driver.

    ``cos(2*pi*u)`` cannot tell ``u`` from ``1-u``, and ``y = 3.9*u*(1-u)`` is
    invariant under exactly that swap, so the cosine's two-to-one ambiguity is
    the one ambiguity the target is blind to.  Either branch of the arccosine
    reproduces ``y``, which is why ``minimal_sufficient_sets`` lists the cosine
    on its own rather than the pair.
    """
    data = redundancy_demo(n=3000, seed=0)
    cos_col = data["x"][:, data["feature_names"].index("proxy_cos")]
    branch = np.arccos(np.clip(cos_col, -1.0, 1.0)) / (2.0 * np.pi)
    assert np.abs(data["y"] - 3.9 * branch * (1.0 - branch)).max() < 1e-12


def test_redundancy_proxy_sin_alone_does_not_determine_the_target():
    """The sine gets no such reprieve: its ambiguity pairs ``u`` with ``0.5-u``.

    The map separates those two, so the sine is genuinely insufficient.  The
    asymmetry between the proxies is a property of the system, not an accident
    of the estimator, and the two must not be assumed interchangeable.
    """
    data = redundancy_demo(n=3000, seed=0)
    names = data["feature_names"]
    assert _knn_r2(data["x"], data["y"], [names.index("proxy_sin")]) < 0.6


def test_redundancy_declared_sets_are_exactly_the_inclusion_minimal_ones():
    """Sufficiency alone is too weak a check, so every subset is enumerated.

    ``(proxy_cos, proxy_sin)`` would pass a sufficiency test and still be the
    wrong ground truth: its proper subset ``(proxy_cos,)`` already determines the
    target, so declaring the pair would mark a correct, smaller answer as
    incomplete.  This test fails on any declared set that has a sufficient proper
    subset, and on any minimal set left out.
    """
    data = redundancy_demo(n=3000, seed=0)
    x, y, names = data["x"], data["y"], data["feature_names"]
    declared = {frozenset(subset) for subset in data["minimal_sufficient_sets"]}

    # The gap between sufficient and insufficient subsets is enormous here --
    # roughly 1.0 against below 0.5 -- so the cut point needs no tuning.
    sufficient = {
        frozenset(names[i] for i in combo)
        for size in range(1, len(names) + 1)
        for combo in combinations(range(len(names)), size)
        if _knn_r2(x, y, list(combo)) > 0.9
    }
    minimal = {s for s in sufficient if not any(other < s for other in sufficient)}
    assert minimal == declared, f"minimal sets are {sorted(map(sorted, minimal))}"


def test_redundancy_the_smallest_costly_deletion_is_a_pair():
    """Single deletions are free; dropping the driver and its substitute is not.

    A set determines the target exactly when it contains ``driver`` or
    ``proxy_cos``, so those two have to go together before any loss moves.
    ``proxy_sin`` and ``unrelated`` survive and are not enough.
    """
    data = redundancy_demo(n=3000, seed=0)
    x, y, names = data["x"], data["y"], data["feature_names"]
    full = _knn_r2(x, y, list(range(len(names))))
    survivors = [names.index("proxy_sin"), names.index("unrelated")]
    assert full - _knn_r2(x, y, survivors) > 0.5


def test_redundancy_removing_the_whole_redundant_group_does_cost():
    """Only a joint deletion registers, which is why the sets are reported."""
    data = redundancy_demo(n=3000, seed=0)
    x, y = data["x"], data["y"]
    names = data["feature_names"]

    unrelated_only = [names.index("unrelated")]
    assert _knn_r2(x, y, unrelated_only) < 0.1

    for subset in data["minimal_sufficient_sets"]:
        columns = [names.index(name) for name in subset]
        assert _knn_r2(x, y, columns) > 0.99, f"{subset} should determine the target"
