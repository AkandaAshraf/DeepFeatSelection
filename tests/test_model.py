import keras
import numpy as np
import pytest

from deepfeatselect.model import (
    FeatureGate,
    NonNegSoftThreshold,
    build_model,
    soft_f1_loss,
)


def test_soft_threshold_shrinks_and_clips_at_zero():
    out = keras.ops.convert_to_numpy(
        NonNegSoftThreshold(0.25)(np.array([1.0, 0.3, 0.25, 0.1, -5.0], dtype="float32"))
    )
    # Shrink by the threshold, and anything at or below it lands on exactly zero.
    assert np.allclose(out, [0.75, 0.05, 0.0, 0.0, 0.0])


def test_soft_threshold_is_serialisable():
    restored = NonNegSoftThreshold.from_config(NonNegSoftThreshold(0.1).get_config())
    assert restored.threshold == 0.1


def test_proximal_gate_reaches_exact_zero():
    """The point of the proximal operator: real zeros, not just small numbers."""
    groups = np.arange(4)
    model = build_model(
        n_columns=4, groups=groups, n_classes=2, task="binary",
        l1_gate=5.0, learning_rate=1e-2, proximal=True,
    )
    x = np.random.default_rng(0).normal(size=(128, 4)).astype("float32")
    y = np.random.default_rng(1).integers(0, 2, size=(128, 1)).astype("float32")
    model.fit(x, y, epochs=10, batch_size=32, verbose=0)
    # Pure noise: no feature earns its keep, so every gate should be zeroed.
    assert (model.get_layer("feature_gate").gate_values() == 0).all()


def test_gate_shares_one_weight_per_feature_group():
    groups = np.array([0, 1, 1, 1, 2])
    gate = FeatureGate(groups=groups)
    gate.build((None, 5))
    assert gate.gate.shape == (3,)


def test_gate_rejects_mismatched_input_width():
    gate = FeatureGate(groups=np.array([0, 1, 1]))
    with pytest.raises(ValueError, match="expected 3 input columns"):
        gate.build((None, 7))


def test_gate_starts_as_identity():
    """Initialised at one, so the untrained network passes features through."""
    groups = np.array([0, 1, 1])
    gate = FeatureGate(groups=groups)
    x = np.array([[1.0, 2.0, 3.0]], dtype="float32")
    assert np.allclose(keras.ops.convert_to_numpy(gate(x)), x)


def test_gate_broadcasts_within_a_group():
    groups = np.array([0, 1, 1])
    gate = FeatureGate(groups=groups)
    gate.build((None, 3))
    gate.gate.assign(np.array([2.0, 3.0], dtype="float32"))
    x = np.array([[1.0, 1.0, 1.0]], dtype="float32")
    out = keras.ops.convert_to_numpy(gate(x))
    assert np.allclose(out, [[2.0, 3.0, 3.0]])


def test_gate_is_serialisable():
    groups = np.array([0, 1, 1])
    restored = FeatureGate.from_config(FeatureGate(groups=groups, l1=0.01).get_config())
    assert restored.l1 == 0.01
    assert list(restored.groups) == [0, 1, 1]


def test_gate_stays_non_negative_under_training():
    """The NonNeg constraint must hold even when L1 pushes gates down hard."""
    groups = np.arange(4)
    model = build_model(n_columns=4, groups=groups, n_classes=2, task="binary", l1_gate=1.0)
    x = np.random.default_rng(0).normal(size=(64, 4)).astype("float32")
    y = np.random.default_rng(1).integers(0, 2, size=(64, 1)).astype("float32")
    model.fit(x, y, epochs=5, batch_size=16, verbose=0)
    assert (model.get_layer("feature_gate").gate_values() >= 0).all()


def test_hierarchy_projection_zeroes_weights_under_a_closed_gate():
    """A gate of zero must make the feature unreachable, not merely quiet."""
    from deepfeatselect.model import HierarchyProjection

    groups = np.arange(3)
    model = build_model(n_columns=3, groups=groups, n_classes=2, task="binary")
    gate_layer = model.get_layer("feature_gate")
    gate_layer.gate.assign(np.array([0.0, 0.0, 1.0], dtype="float32"))

    projection = HierarchyProjection(gate_layer, m=1.0)
    projection.set_model(model)
    projection.on_train_batch_end(0)

    kernel = keras.ops.convert_to_numpy(model.get_layer("fc_1").kernel)
    assert np.abs(kernel[0]).max() == 0.0
    assert np.abs(kernel[1]).max() == 0.0
    assert np.abs(kernel[2]).max() > 0.0


def test_hierarchy_projection_bounds_weights_by_gate():
    from deepfeatselect.model import HierarchyProjection

    groups = np.arange(2)
    model = build_model(n_columns=2, groups=groups, n_classes=2, task="binary")
    gate_layer = model.get_layer("feature_gate")
    gate_layer.gate.assign(np.array([0.01, 1.0], dtype="float32"))

    projection = HierarchyProjection(gate_layer, m=2.0)
    projection.set_model(model)
    projection.on_train_batch_end(0)

    kernel = keras.ops.convert_to_numpy(model.get_layer("fc_1").kernel)
    assert np.abs(kernel[0]).max() <= 2.0 * 0.01 + 1e-6


def test_binary_and_multiclass_output_shapes():
    groups = np.arange(4)
    binary = build_model(n_columns=4, groups=groups, n_classes=2, task="binary")
    multi = build_model(n_columns=4, groups=groups, n_classes=5, task="multiclass")
    assert binary.output_shape == (None, 1)
    assert multi.output_shape == (None, 5)


def test_multiclass_head_is_a_simplex():
    """Softmax, not the original's independent sigmoids: rows must sum to one."""
    groups = np.arange(4)
    model = build_model(n_columns=4, groups=groups, n_classes=5, task="multiclass")
    probs = model.predict(np.zeros((8, 4), dtype="float32"), verbose=0)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_soft_f1_loss_rewards_perfect_predictions():
    y = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    perfect = float(keras.ops.convert_to_numpy(soft_f1_loss(y, y)))
    inverted = float(keras.ops.convert_to_numpy(soft_f1_loss(y, 1 - y)))
    assert perfect < 1e-6
    assert inverted > perfect
