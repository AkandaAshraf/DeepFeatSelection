import warnings

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from deepfeatselect.baselines import (
    _normalise,
    aggregate_to_features,
    all_baselines,
    l1_logistic,
    mutual_information,
    permutation_rf,
    random_forest,
    rank_agreement,
)

METHODS = (mutual_information, l1_logistic, random_forest, permutation_rf)


def _signal_data(n=300, n_noise=6, seed=0):
    """Only columns 0 and 1 carry signal; the rest are independent noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2 + n_noise))
    logit = 3.0 * (x[:, 0] + x[:, 1])
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(np.int64)
    return x, y


@pytest.mark.parametrize("method", METHODS, ids=lambda f: f.__name__)
def test_every_baseline_finds_the_signal_columns(method):
    x, y = _signal_data()
    scores = method(x, y, seed=0)
    assert scores.shape == (x.shape[1],)
    assert set(np.argsort(-scores)[:2].tolist()) == {0, 1}


def test_all_baselines_finds_the_signal_columns():
    x, y = _signal_data()
    for name, scores in all_baselines(x, y, seed=0).items():
        assert set(np.argsort(-scores)[:2].tolist()) == {0, 1}, name


def test_all_baselines_returns_non_negative_shares():
    x, y = _signal_data()
    out = all_baselines(x, y, seed=0)
    assert set(out) == {"mutual_information", "l1_logistic", "random_forest", "permutation_rf"}
    for name, scores in out.items():
        assert (scores >= 0).all(), name
        assert scores.sum() == pytest.approx(1.0), name


def test_baselines_handle_multiclass():
    """l1_logistic in particular has to average over a class axis that only exists here."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 5))
    y = np.digitize(x[:, 0] + x[:, 1], [-1.0, 0.0, 1.0]).astype(np.int64)
    for method in METHODS:
        scores = method(x, y, seed=0)
        assert scores.shape == (5,)
        assert set(np.argsort(-scores)[:2].tolist()) == {0, 1}, method.__name__


def test_l1_logistic_produces_exact_zeros():
    """The one property that separates an L1 fit from an L2 one.

    Every other test in this file passes just as well against an L2-penalised
    fit -- it ranks columns 0 and 1 first too -- so without this the version
    gate in ``_l1_kwargs`` could silently pick the wrong spelling and nothing
    would notice.
    """
    x, y = _signal_data()
    scores = l1_logistic(x, y, C=0.1, seed=0)
    assert (scores[[0, 1]] > 0).all()
    assert (scores[2:] == 0.0).all()


def test_l1_logistic_multiclass_reduction_stays_sparse():
    """Averaging |coef| over the class axis keeps a zero only if every class
    dropped the column, which is the reduction we want."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 8))
    y = np.digitize(x[:, 0] + x[:, 1], [-1.0, 0.0, 1.0]).astype(np.int64)
    scores = l1_logistic(x, y, C=0.1, seed=0)
    assert (scores == 0.0).any()
    assert set(np.argsort(-scores)[:2].tolist()) == {0, 1}


def test_l1_logistic_sparsity_tracks_C():
    """C is inverse regularisation strength, so smaller must mean sparser."""
    x, y = _signal_data()
    counts = [int((l1_logistic(x, y, C=c, seed=0) == 0.0).sum()) for c in (0.05, 0.5, 10.0)]
    assert counts[0] > counts[1] > counts[2]


def test_l1_logistic_does_not_trip_a_deprecated_parameter():
    """``_l1_kwargs`` exists to pick the spelling this scikit-learn wants; a
    FutureWarning means it guessed wrong for the installed version."""
    x, y = _signal_data(n=120)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        l1_logistic(x, y, seed=0)


def test_normalise_survives_an_all_zero_and_an_empty_vector():
    """Both denominators -- the total and the length -- can be zero."""
    assert _normalise(np.zeros(4)) == pytest.approx([0.25] * 4)
    assert _normalise(np.array([-1.0, -2.0])) == pytest.approx([0.5, 0.5])
    assert _normalise(np.array([])).shape == (0,)


def test_aggregate_sums_within_groups():
    # Feature 0 owns column 0, feature 1 is a three-level one-hot spanning
    # columns 1-3, feature 2 owns column 4.
    groups = np.array([0, 1, 1, 1, 2])
    column_scores = np.array([0.5, 0.1, 0.2, 0.3, 0.4])
    out = aggregate_to_features(column_scores, groups, n_features=3)
    assert out == pytest.approx([0.5, 0.6, 0.4])


def test_aggregate_preserves_the_total():
    """Shares stay shares, which is what makes the per-feature tables comparable."""
    x, y = _signal_data()
    groups = np.array([0, 0, 1, 1, 1, 2, 3, 3])
    for name, scores in all_baselines(x, y, seed=0).items():
        assert aggregate_to_features(scores, groups, 4).sum() == pytest.approx(1.0), name


def test_aggregate_handles_unsorted_and_non_contiguous_groups():
    """Nothing guarantees a feature's columns are adjacent or that the group
    vector is sorted -- ``_encode`` only happens to emit it that way today."""
    groups = np.array([3, 0, 3, 1, 0, 3, 4])
    column_scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    out = aggregate_to_features(column_scores, groups, n_features=5)
    # feature 3 owns columns 0, 2 and 5; feature 2 owns none.
    assert out == pytest.approx([7.0, 4.0, 0.0, 10.0, 7.0])
    assert out.sum() == pytest.approx(column_scores.sum())
    # A permutation of the columns must give exactly the same per-feature totals.
    order = np.array([6, 0, 4, 2, 1, 5, 3])
    assert aggregate_to_features(column_scores[order], groups[order], 5) == pytest.approx(out)


def test_aggregate_keeps_features_owning_no_columns():
    out = aggregate_to_features([1.0, 2.0], np.array([0, 0]), n_features=3)
    assert out == pytest.approx([3.0, 0.0, 0.0])


def test_aggregate_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="one per column"):
        aggregate_to_features([1.0, 2.0, 3.0], np.array([0, 1]), n_features=2)


def test_aggregate_rejects_out_of_range_groups():
    with pytest.raises(ValueError, match=r"must be in \[0, 2\)"):
        aggregate_to_features([1.0, 2.0], np.array([0, 2]), n_features=2)


def test_rank_agreement_is_symmetric_with_unit_diagonal():
    scores = {
        "a": np.array([0.4, 0.3, 0.2, 0.1]),
        "b": np.array([0.9, 0.05, 0.03, 0.02]),
        "c": np.array([0.1, 0.4, 0.2, 0.3]),
    }
    out = rank_agreement(scores)
    assert list(out.index) == list(out.columns) == ["a", "b", "c"]
    values = out.to_numpy()
    assert np.diag(values) == pytest.approx(1.0)
    assert values == pytest.approx(values.T)
    assert (np.abs(values) <= 1.0 + 1e-12).all()


def test_rank_agreement_reports_perfect_and_reversed_orderings():
    """Guards against a symmetric matrix that is symmetric for the wrong reason."""
    base = np.array([0.4, 0.3, 0.2, 0.1])
    out = rank_agreement({"a": base, "same_order": base * 10, "reversed": base[::-1]})
    assert out.loc["a", "same_order"] == pytest.approx(1.0)
    assert out.loc["a", "reversed"] == pytest.approx(-1.0)


def test_rank_agreement_diagonal_survives_a_constant_vector():
    """A method that ranked nothing has no correlation, but still matches itself."""
    out = rank_agreement({"flat": np.ones(5), "real": np.arange(5.0)})
    assert out.loc["flat", "flat"] == pytest.approx(1.0)
    assert np.isnan(out.loc["flat", "real"])


def test_rank_agreement_rejects_ragged_input():
    with pytest.raises(ValueError, match="same length"):
        rank_agreement({"a": np.arange(3.0), "b": np.arange(4.0)})


def _memorisable_data(n=240, seed=1):
    """Column 0 drives y; column 1 is pure noise with a distinct value per row.

    The label noise is essential, not incidental: with a deterministic
    ``y = signal > 0`` the forest separates the classes on column 0 alone, never
    splits on column 1, and there is nothing memorised for a training-data
    permutation to destroy. Residual the forest cannot explain is what pushes it
    into fitting the noise column.
    """
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-1.5 * signal))).astype(np.int64)
    return np.column_stack([signal, noise]), y


def test_permutation_importance_is_measured_on_heldout_data():
    """The trap: on training rows, shuffling a memorised noise column scores highly."""
    x, y = _memorisable_data()
    held_out = permutation_rf(x, y, n_repeats=10, seed=0)

    forest = RandomForestClassifier(
        n_estimators=200, random_state=0, class_weight="balanced"
    ).fit(x, y)
    on_train = permutation_importance(
        forest, x, y, scoring="balanced_accuracy", n_repeats=10, random_state=0
    ).importances_mean

    assert on_train[1] > 0.1, "the noise column was not memorised; the test is vacuous"
    assert held_out[1] < 0.25 * on_train[1]
    assert held_out[0] > held_out[1]


def test_permutation_rf_accepts_an_explicit_eval_set():
    x, y = _signal_data(n=400)
    scores = permutation_rf(x[:300], y[:300], x[300:], y[300:], n_repeats=10, seed=0)
    assert scores.shape == (x.shape[1],)
    assert (scores >= 0).all()
    assert set(np.argsort(-scores)[:2].tolist()) == {0, 1}


def test_permutation_rf_rejects_a_half_given_eval_set():
    x, y = _signal_data(n=120)
    with pytest.raises(ValueError, match="must be given together"):
        permutation_rf(x, y, x_eval=x, n_repeats=2)


def test_permutation_rf_rejects_mismatched_eval_width():
    x, y = _signal_data(n=120)
    with pytest.raises(ValueError, match="columns"):
        permutation_rf(x, y, x[:, :3], y, n_repeats=2)


def test_baselines_are_reproducible():
    x, y = _signal_data()
    for method in METHODS:
        assert method(x, y, seed=0) == pytest.approx(method(x, y, seed=0)), method.__name__


def test_baselines_reject_ragged_input():
    x, y = _signal_data(n=50)
    for method in METHODS:
        with pytest.raises(ValueError, match="rows"):
            method(x, y[:10], seed=0)
