import keras
import numpy as np
import pytest

from deepfeatselect.attribution import (
    _gini,
    _mean_pairwise_cosine,
    input_relevance,
    relevance_structure,
    unit_relevance,
)
from deepfeatselect.model import build_model


def _model(n_columns=4, hidden=8, seed=0):
    keras.utils.set_random_seed(seed)
    return build_model(
        n_columns=n_columns, groups=np.arange(n_columns), n_classes=2, task="binary",
        l1_gate=0.0, dropout=0.0, noise=0.0, hidden_units=hidden, n_hidden_layers=2,
    )


def test_gini_bounds():
    assert _gini(np.ones(10)) == pytest.approx(0.0, abs=1e-9)
    spike = np.zeros(100)
    spike[0] = 1.0
    assert _gini(spike) > 0.95
    assert _gini(np.zeros(5)) == 0.0


def test_mean_pairwise_cosine():
    identical = np.tile(np.array([1.0, 2.0, 3.0]), (5, 1))
    assert _mean_pairwise_cosine(identical) == pytest.approx(1.0, abs=1e-6)
    orthogonal = np.eye(4)
    assert _mean_pairwise_cosine(orthogonal) == pytest.approx(0.0, abs=1e-6)


def test_unit_relevance_is_nonnegative_and_shaped():
    model = _model()
    x = np.random.default_rng(0).normal(size=(32, 4)).astype("float32")
    relevance = unit_relevance(model, x)
    assert relevance.shape == (32, 8)
    assert (relevance >= 0).all()


def test_unit_relevance_matches_grad_cam_definition():
    """ReLU(w_k * h_k) with w the output kernel, checked against a manual pass."""
    model = _model()
    x = np.random.default_rng(1).normal(size=(16, 4)).astype("float32")

    acts = keras.Model(model.inputs, model.get_layer("fc_2").output).predict(x, verbose=0)
    w = keras.ops.convert_to_numpy(model.get_layer("output").kernel)[:, 0]
    expected = np.maximum(np.asarray(acts) * w[None, :], 0.0)

    assert np.allclose(unit_relevance(model, x), expected, atol=1e-5)


def test_relevance_structure_is_permutation_invariant():
    """Relabelling hidden units must leave every exported scalar unchanged."""
    rng = np.random.default_rng(2)
    relevance = np.abs(rng.normal(size=(60, 8)))
    y = rng.integers(0, 2, size=60)

    before = relevance_structure(relevance, y)
    perm = rng.permutation(8)
    after = relevance_structure(relevance[:, perm], y)
    for key in before:
        assert after[key] == pytest.approx(before[key], abs=1e-9), key


def test_class_separation_detects_discriminative_maps():
    """A map that differs by class must score above one that does not."""
    rng = np.random.default_rng(3)
    y = np.repeat([0, 1], 40)
    shared = np.abs(rng.normal(size=(80, 6))) + 1.0

    discriminative = shared.copy()
    discriminative[y == 1] += np.array([5.0, 0, 0, 0, 0, 0])
    discriminative[y == 0] += np.array([0, 0, 0, 0, 0, 5.0])

    sep_disc = relevance_structure(discriminative, y)["class_separation"]
    sep_shared = relevance_structure(shared, y)["class_separation"]
    assert sep_disc > sep_shared
    assert sep_shared < 0.05


def test_input_relevance_zero_for_a_disconnected_feature():
    """A feature the network cannot see must receive no attribution."""
    model = _model(n_columns=3)
    gate = model.get_layer("feature_gate")
    gate.gate.assign(np.array([1.0, 1.0, 0.0], dtype="float32"))

    x = np.random.default_rng(4).normal(size=(24, 3)).astype("float32")
    relevance = input_relevance(model, x)
    assert relevance.shape == (24, 3)
    assert np.abs(relevance[:, 2]).max() < 1e-6
    assert np.abs(relevance[:, :2]).max() > 1e-6
