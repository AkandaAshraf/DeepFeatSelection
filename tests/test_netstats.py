import keras
import numpy as np
import pytest

from deepfeatselect.model import build_model
from deepfeatselect.netstats import (
    activation_stats,
    participation_ratio,
    residual_autocorr,
    residual_hsic,
    weight_stats,
)


def _small_model(n_columns=4, hidden=8, layers=2, seed=0):
    keras.utils.set_random_seed(seed)
    return build_model(
        n_columns=n_columns, groups=np.arange(n_columns), n_classes=2,
        task="binary", l1_gate=0.0, dropout=0.0, noise=0.0,
        hidden_units=hidden, n_hidden_layers=layers,
    )


def _permute_hidden(model, layer_name, next_name, perm):
    """Relabel the units of one hidden layer without changing the function."""
    layer, nxt = model.get_layer(layer_name), model.get_layer(next_name)
    k = keras.ops.convert_to_numpy(layer.kernel)[:, perm]
    b = keras.ops.convert_to_numpy(layer.bias)[perm]
    k2 = keras.ops.convert_to_numpy(nxt.kernel)[perm, :]
    layer.kernel.assign(k)
    layer.bias.assign(b)
    nxt.kernel.assign(k2)


def test_participation_ratio_bounds():
    rng = np.random.default_rng(0)
    rank1 = np.outer(rng.normal(size=500), rng.normal(size=8))
    assert participation_ratio(rank1) == pytest.approx(1.0, abs=1e-6)
    isotropic = rng.normal(size=(5000, 8))
    assert participation_ratio(isotropic) == pytest.approx(8.0, rel=0.05)
    assert participation_ratio(np.zeros((10, 4))) == 0.0


def test_activation_stats_invariant_to_unit_permutation():
    model = _small_model()
    rng = np.random.default_rng(1)
    probe = rng.normal(size=(64, 4)).astype("float32")
    before = activation_stats(model, probe, prefix="p")
    _permute_hidden(model, "fc_1", "fc_2", np.random.default_rng(2).permutation(8))
    after = activation_stats(model, probe, prefix="p")
    for key in before:
        assert after[key] == pytest.approx(before[key], abs=1e-6), key


def test_weight_stats_invariant_to_unit_permutation():
    model = _small_model()
    before = weight_stats(model)
    _permute_hidden(model, "fc_1", "fc_2", np.random.default_rng(3).permutation(8))
    after = weight_stats(model)
    for key in before:
        assert after[key] == pytest.approx(before[key], abs=1e-6), key


def test_weight_entropy_edge_cases():
    model = _small_model()
    stats = weight_stats(model)
    # Glorot-initialised weights are continuous draws: entropy well above zero,
    # bounded by the 8-bit quantisation ceiling.
    assert 2.0 < stats["w_entropy_bits"] <= 8.0
    assert stats["w_gzip_bits"] > 0.0


def test_residual_hsic_detects_dependence():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(300, 2))
    dependent = np.sin(3.0 * x[:, 0]) + 0.05 * rng.normal(size=300)
    stat_dep, p_dep = residual_hsic(dependent, x, seed=0)
    independent = rng.normal(size=300)
    stat_ind, p_ind = residual_hsic(independent, x, seed=0)
    assert p_dep < 0.01
    assert p_ind > 0.05
    assert stat_dep > stat_ind


def test_residual_autocorr():
    rng = np.random.default_rng(5)
    assert residual_autocorr(rng.normal(size=2000)) < 0.05
    t = np.arange(2000)
    assert residual_autocorr(np.sin(0.3 * t)) > 0.5
    assert residual_autocorr(np.zeros(100)) == 0.0
