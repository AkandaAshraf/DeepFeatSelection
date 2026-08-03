"""The gated feature-selection network.

The idea carried over from the original implementation: put a single learnable
non-negative scalar in front of every input feature, train the network, and read
those scalars back as importances.

The original built this as one ``Dense(1, use_bias=False)`` layer per feature and
concatenated the results.  :class:`FeatureGate` below is the same computation as
one vectorised layer, which makes two things possible that were awkward before:

* an L1 penalty on the gate vector, which is what actually makes the importances
  identifiable (see the note in :func:`build_model`);
* one gate shared across all one-hot columns of a categorical feature, so a
  feature's importance does not depend on how many levels it has.

It also drops the ``relu`` the original applied to each gate.  With the old
all-positive L2-normalised inputs that activation was a no-op; with standardised
inputs it would clip every negative value and silently discard half the signal.
"""

from __future__ import annotations

import keras
import numpy as np
from keras import ops


@keras.saving.register_keras_serializable(package="deepfeatselect")
class NonNegSoftThreshold(keras.constraints.Constraint):
    """Proximal operator for an L1 penalty combined with a non-negativity constraint.

    Adding an L1 term to the loss only shrinks weights asymptotically -- with an
    adaptive optimiser and a few hundred update steps it never reaches zero, so
    the "selection" half of the penalty never happens.  Applying the proximal
    operator instead, ``max(w - t, 0)`` after every update, is the standard way
    to get L1 to produce genuine zeros, and Keras constraints run at exactly the
    right point for it.

    ``threshold`` is the per-step shrinkage, i.e. learning rate times penalty
    strength.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, w):
        return ops.maximum(w - self.threshold, 0.0)

    def get_config(self):
        return {"threshold": self.threshold}


@keras.saving.register_keras_serializable(package="deepfeatselect")
class FeatureGate(keras.layers.Layer):
    """Scales every input column by a non-negative, per-feature learnable weight.

    Args:
        groups: Length-``n_columns`` vector where ``groups[j]`` is the index of
            the feature that produced column ``j``.  Columns sharing an index
            share a gate.
        l1: Strength of the L1 penalty applied to the gate vector.  Zero
            reproduces the original unregularised behaviour.
        prox_threshold: Per-step soft-threshold.  When set, L1 is enforced
            proximally (producing exact zeros) rather than through the loss.
    """

    def __init__(self, groups, l1: float = 0.0, prox_threshold: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.groups = np.asarray(groups, dtype="int32")
        self.l1 = l1
        self.prox_threshold = prox_threshold
        self.n_features = int(self.groups.max()) + 1

    def build(self, input_shape):
        if input_shape[-1] != len(self.groups):
            raise ValueError(
                f"expected {len(self.groups)} input columns to match the group "
                f"vector, got {input_shape[-1]}"
            )
        # Initialised at one so the network starts by passing every feature
        # through untouched, and the L1 term has to earn each reduction.
        constraint = (
            NonNegSoftThreshold(self.prox_threshold)
            if self.prox_threshold
            else keras.constraints.NonNeg()
        )
        self.gate = self.add_weight(
            name="gate",
            shape=(self.n_features,),
            initializer="ones",
            constraint=constraint,
            regularizer=keras.regularizers.L1(self.l1) if self.l1 else None,
            trainable=True,
        )
        self._group_index = ops.convert_to_tensor(self.groups)
        super().build(input_shape)

    def call(self, inputs):
        return inputs * ops.take(self.gate, self._group_index, axis=0)

    def gate_values(self) -> np.ndarray:
        """The current per-feature gate weights, one entry per feature."""
        return np.asarray(keras.ops.convert_to_numpy(self.gate))

    def get_config(self):
        return {
            **super().get_config(),
            "groups": self.groups.tolist(),
            "l1": self.l1,
            "prox_threshold": self.prox_threshold,
        }


class HierarchyProjection(keras.callbacks.Callback):
    """Ties first-layer weights to their gate, so a closed gate really closes.

    Without this the L1 penalty is toothless in a way that is easy to miss: as a
    gate shrinks, the matching column of the first dense layer simply grows to
    compensate, the network's output is unchanged, and every gate feels the same
    push-back.  The observed symptom is gates drifting down *uniformly* and
    stalling, which reorders nothing and selects nothing.

    After each batch this projects the first layer so that column ``j`` satisfies
    ``max|W[j, :]| <= M * gate[feature(j)]``.  A gate of exactly zero therefore
    forces that feature's weights to zero, making the feature genuinely
    unreachable rather than merely attenuated.  This is the hierarchy constraint
    from LassoNet (Lemhadri et al., 2021), applied as an alternating projection.
    """

    def __init__(self, gate_layer: "FeatureGate", dense_layer_name: str = "fc_1", m: float = 1.0):
        super().__init__()
        self.gate_layer = gate_layer
        self.dense_layer_name = dense_layer_name
        self.m = m

    def on_train_batch_end(self, batch, logs=None):
        kernel = self.model.get_layer(self.dense_layer_name).kernel
        bound = self.m * ops.take(self.gate_layer.gate, self.gate_layer._group_index, axis=0)
        norms = ops.max(ops.absolute(kernel), axis=1)
        scale = ops.minimum(1.0, bound / (norms + 1e-12))
        kernel.assign(kernel * ops.expand_dims(scale, axis=-1))


def soft_f1_loss(y_true, y_pred):
    """Macro soft-F1 loss, kept from the original implementation.

    Optimising a differentiable relaxation of F1 directly is a reasonable answer
    to the class imbalance here.  Reduction is over the batch axis, so with very
    rare classes small batches give noisy gradients -- prefer ``"ce"`` with class
    weights when a class has only a handful of examples.
    """
    tp = ops.sum(y_pred * y_true, axis=0)
    fp = ops.sum(y_pred * (1 - y_true), axis=0)
    fn = ops.sum((1 - y_pred) * y_true, axis=0)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    return ops.mean(1 - soft_f1)


def build_model(
    n_columns: int,
    groups,
    n_classes: int,
    task: str = "binary",
    l1_gate: float = 1e-3,
    l2_dense: float = 1e-3,
    hidden_units: int = 128,
    n_hidden_layers: int = 3,
    dropout: float = 0.5,
    noise: float = 0.005,
    learning_rate: float = 1e-3,
    loss: str = "ce",
    proximal: bool = True,
) -> keras.Model:
    """Build and compile the gated network.

    On why both penalties matter: an L1 term on the gate alone does not make the
    importances well defined, because the network can shrink a gate and grow the
    matching column of the first dense layer to compensate at no cost -- gate
    magnitude and downstream weights are freely interchangeable.  Weight decay on
    the dense layers is what removes that escape route and gives the L1 term real
    influence over the ranking.  Setting ``l2_dense=0`` with ``l1_gate>0`` is
    therefore not a meaningful configuration.

    ``proximal`` selects *how* the L1 is enforced.  Through the loss it only
    shrinks gates asymptotically, which on a dataset this small means a few
    hundred update steps that move a gate by a fraction of its initial value and
    never select anything.  Proximally it soft-thresholds after every step and
    produces exact zeros, which is what makes the ranking a selection rather than
    a reordering.
    """
    reg = keras.regularizers.L2(l2_dense) if l2_dense else None

    inputs = keras.layers.Input(shape=(n_columns,), name="features")
    gate = FeatureGate(
        groups=groups,
        l1=0.0 if proximal else l1_gate,
        prox_threshold=learning_rate * l1_gate if proximal else 0.0,
        name="feature_gate",
    )
    h = gate(inputs)

    for i in range(n_hidden_layers):
        h = keras.layers.Dense(
            hidden_units,
            activation="relu",
            kernel_regularizer=reg,
            bias_regularizer=reg,
            name=f"fc_{i + 1}",
        )(h)
        if noise:
            h = keras.layers.GaussianNoise(noise)(h)
        if dropout:
            h = keras.layers.Dropout(dropout)(h)

    if task == "binary":
        outputs = keras.layers.Dense(1, activation="sigmoid", name="output")(h)
        default_loss = keras.losses.BinaryCrossentropy()
        metrics = [keras.metrics.AUC(name="auc"), keras.metrics.BinaryAccuracy(name="acc")]
    else:
        # A softmax over mutually exclusive severity levels, rather than the
        # original's independent sigmoid per class, which let the model assign
        # high probability to several levels at once.
        outputs = keras.layers.Dense(n_classes, activation="softmax", name="output")(h)
        default_loss = keras.losses.CategoricalCrossentropy()
        metrics = [keras.metrics.CategoricalAccuracy(name="acc")]

    model = keras.Model(inputs=inputs, outputs=outputs, name="deep_feat_selection")
    model.compile(
        loss=soft_f1_loss if loss == "soft_f1" else default_loss,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=metrics,
    )
    return model
