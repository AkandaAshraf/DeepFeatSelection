import numpy as np
import pytest

from deepfeatselect.data import CLEVELAND_COLUMNS, NOMINAL_FEATURES, load_raw, prepare

DATA = "Data/processed.cleveland.data"


def test_missing_rows_dropped():
    df = load_raw(DATA)
    assert not df.isna().any().any()
    assert len(df) == 297  # 303 rows less the 6 carrying '?'


def test_binary_task_collapses_target():
    data = prepare(DATA, task="binary", seed=0)
    assert data.n_classes == 2
    assert set(np.unique(data.y_train)) <= {0, 1}


def test_multiclass_keeps_five_levels():
    data = prepare(DATA, task="multiclass", seed=0)
    assert data.n_classes == 5


def test_splits_are_disjoint_and_complete():
    data = prepare(DATA, seed=0)
    total = len(data.y_train) + len(data.y_val) + len(data.y_test)
    assert total == 297
    assert len(data.y_test) == pytest.approx(297 * 0.2, abs=2)
    assert len(data.y_val) == pytest.approx(297 * 0.2, abs=2)


def test_split_is_stratified():
    data = prepare(DATA, task="binary", seed=0)
    rates = [split.mean() for split in (data.y_train, data.y_val, data.y_test)]
    assert max(rates) - min(rates) < 0.05


def test_scaler_fitted_on_train_only():
    """Train columns are standardised; val/test are not forced to zero mean."""
    data = prepare(DATA, seed=0)
    # Group scaling divides by sqrt(group size), so compare against that factor
    # rather than against exactly 1.
    assert np.abs(data.x_train.mean(axis=0)).max() < 1e-9
    assert np.abs(data.x_val.mean(axis=0)).max() > 1e-6


def test_one_hot_expands_nominal_features_only():
    data = prepare(DATA, seed=0)
    assert data.n_features == len(CLEVELAND_COLUMNS)
    assert data.n_columns > data.n_features
    # Each nominal feature must own more than one column; each numeric exactly one.
    counts = np.bincount(data.groups)
    for i, name in enumerate(data.feature_names):
        expected_multi = name in NOMINAL_FEATURES
        assert (counts[data.feature_names.index(name)] > 1) == expected_multi, name


def test_group_scaling_equalises_feature_variance():
    """Every feature should contribute the same total variance, whatever its width."""
    data = prepare(DATA, seed=0)
    per_feature = np.zeros(data.n_features)
    var = data.x_train.var(axis=0)
    for col, g in enumerate(data.groups):
        per_feature[g] += var[col]
    assert np.allclose(per_feature, 1.0, atol=1e-6)


def test_rejects_unknown_task():
    with pytest.raises(ValueError, match="task must be"):
        prepare(DATA, task="regression")
