import numpy as np
import pytest

from deepfeatselect.redundancy import _cv_r2
from deepfeatselect.shapley import loco_for_comparison, shapley_importance
from deepfeatselect.synthetic import redundancy_demo


@pytest.fixture(scope="module")
def demo():
    system = redundancy_demo(n=600, seed=0)
    x = np.asarray(system["x"], dtype=np.float64)
    y = (np.asarray(system["y"]) > np.median(system["y"])).astype(np.float64)
    return x, y, list(system["feature_names"])


def test_efficiency_axiom(demo):
    """Shapley values must sum to v(N) - v(empty); the tightest exactness check."""
    x, y, names = demo
    total = shapley_importance(x, y, names, seed=0).sage.sum()
    assert total == pytest.approx(_cv_r2(x, y, seed=0), abs=1e-9)


def test_null_player_is_within_the_estimator_noise_floor(demo):
    """A constant column is a null player, but only up to estimation noise.

    v(S) is a cross-validated forest rather than an exact quantity, and adding a
    column perturbs its internal feature sampling, so the null-player axiom holds
    numerically rather than identically. Pinned here so the limit is documented
    instead of hidden in a loose tolerance elsewhere.
    """
    x, y, names = demo
    x_aug = np.hstack([x, np.zeros((len(x), 1))])
    result = shapley_importance(x_aug, y, names + ["constant"], seed=0).set_index("feature")
    assert abs(result.loc["constant", "sage"]) < 0.01


def test_noise_floor_is_small_relative_to_real_signal(demo):
    """The resolution limit must sit well below the values being interpreted."""
    from deepfeatselect.shapley import shapley_noise_floor

    x, y, names = demo
    floor = shapley_noise_floor(x, y, names, seed=0)
    signal = shapley_importance(x, y, names, seed=0).set_index("feature")
    assert floor < 0.10
    assert signal.loc["driver", "sage"] > 3 * floor


def test_symmetric_features_get_equal_credit():
    """Duplicate columns are interchangeable in every coalition, so must tie."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=400)
    y = np.tanh(2.0 * a) + 0.1 * rng.normal(size=400)
    x = np.column_stack([a, a.copy(), rng.normal(size=400)])
    result = shapley_importance(x, y, ["a", "a_copy", "noise"], seed=0).set_index("feature")
    assert result.loc["a", "sage"] == pytest.approx(result.loc["a_copy", "sage"], abs=1e-9)


def test_shapley_survives_where_loco_is_blind(demo):
    """The point of the comparison: Prop 1 kills one coalition, not the average."""
    x, y, names = demo
    sage = shapley_importance(x, y, names, seed=0).set_index("feature")
    loco = loco_for_comparison(x, y, names, seed=0).set_index("feature")

    informative = ["driver", "proxy_cos", "proxy_sin"]
    assert loco.loc[informative, "loco"].abs().max() < 0.02
    assert sage.loc[informative, "sage"].min() > 0.05
    # The comparison that matters is against the irrelevant column, not against
    # zero: the estimator has a noise floor and the claim is separation from it.
    assert sage.loc[informative, "sage"].min() > 2 * abs(sage.loc["unrelated", "sage"])


def test_sampled_estimator_approximates_the_exact_one(demo):
    """Permutation sampling is unbiased, so it must land near the exact answer."""
    x, y, names = demo
    exact = shapley_importance(x, y, names, seed=0).set_index("feature").sage

    import deepfeatselect.shapley as mod
    original = mod.EXACT_MAX_SUBSETS
    try:
        mod.EXACT_MAX_SUBSETS = 1  # force the sampling branch
        sampled = shapley_importance(x, y, names, seed=0, n_permutations=60)
    finally:
        mod.EXACT_MAX_SUBSETS = original

    sampled = sampled.set_index("feature").sage
    # Tolerance covers both sampling error and the value function's own noise;
    # what must hold is that the ordering and scale survive.
    for name in names:
        assert sampled[name] == pytest.approx(exact[name], abs=0.10), name
    assert sampled.idxmax() in {"driver", "proxy_cos"}
