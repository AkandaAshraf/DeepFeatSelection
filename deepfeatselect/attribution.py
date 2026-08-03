"""Grad-CAM-style attribution for tabular networks, and its structure statistics.

Grad-CAM (Selvaraju et al., 2017) weights a convolutional feature map by the
pooled gradient of a class score, ``alpha_k = GAP(d y^c / d A^k)``, and reports
``ReLU(sum_k alpha_k A^k)``.  Two ingredients: gradient-times-activation
weighting, and pooling over spatial positions.  Only the first transfers to a
tabular network -- there are no spatial axes to pool -- so the unit-level
analogue is

    relevance_k(x) = ReLU( (d s / d h_k) * h_k )

with ``s`` the pre-sigmoid score.  Evaluated at the penultimate layer this needs
no autodiff at all: the head is affine, so ``d s / d h_k`` is exactly the output
kernel and the relevance is ``ReLU(w_k h_k)``.  That is the same location
Grad-CAM uses in a CNN, the last layer before the classifier head.

This channel answers a question the activation statistics in ``netstats`` cannot.
Those count how many units fire; they are output-agnostic.  Relevance asks which
units the *decision* rests on, which is the quantity that carries structure when
a model has learned something and does not when it has not.

Permutation invariance is handled the same way as elsewhere: a per-unit
relevance vector is tied to unit identity and is not comparable across two
trainings, but every statistic exported here is a scalar computed *within* one
model -- concentration of the relevance vector, cosine distance between the
class-conditional relevance maps, agreement of maps across samples of one class.
Relabelling units permutes the vectors and leaves all three unchanged.

Input-level relevance has no such problem at all: input features keep their
identity across models, so ``d s / d x_i * x_i`` can be compared directly
between a full model and an ablated one.  That comparison is the literal
measurement of a network re-deriving a removed feature's contribution from the
remaining inputs.
"""

from __future__ import annotations

import keras
import numpy as np


def _penultimate(model: keras.Model) -> keras.layers.Layer:
    """The last hidden dense layer, i.e. the Grad-CAM attachment point."""
    dense = [l for l in model.layers if l.name.startswith("fc_")]
    if not dense:
        raise ValueError("no fc_* hidden layer found to attach attribution to")
    return dense[-1]


def unit_relevance(model: keras.Model, x: np.ndarray) -> np.ndarray:
    """Per-unit Grad-CAM relevance at the penultimate layer, shape (n, units).

    The head is affine in the penultimate activations, so the Grad-CAM gradient
    weight is the output kernel itself and the whole map is available in closed
    form.  The ReLU is Grad-CAM's, and it is load-bearing: it keeps only units
    pushing the score toward the positive class, which is what makes the map
    class-discriminative rather than a general sensitivity measure.
    """
    layer = _penultimate(model)
    sub = keras.Model(model.inputs, layer.output)
    acts = np.asarray(sub.predict(x, verbose=0))

    kernel = np.asarray(keras.ops.convert_to_numpy(model.get_layer("output").kernel))
    if kernel.shape[1] != 1:
        raise ValueError("unit_relevance currently assumes a single-logit head")
    return np.maximum(acts * kernel[:, 0][None, :], 0.0)


def input_relevance(model: keras.Model, x: np.ndarray) -> np.ndarray:
    """Gradient-times-input attribution on the pre-sigmoid score, shape (n, features).

    Differentiating the score rather than the probability keeps the measure from
    being squashed to zero wherever the model is confident, which is exactly
    where a well-fit network spends most of its mass.
    """
    import tensorflow as tf

    layer = _penultimate(model)
    features = keras.Model(model.inputs, layer.output)
    out = model.get_layer("output")
    kernel = tf.convert_to_tensor(keras.ops.convert_to_numpy(out.kernel), dtype=tf.float32)
    bias = tf.convert_to_tensor(keras.ops.convert_to_numpy(out.bias), dtype=tf.float32)

    x_t = tf.convert_to_tensor(np.asarray(x, dtype="float32"))
    with tf.GradientTape() as tape:
        tape.watch(x_t)
        score = tf.matmul(features(x_t, training=False), kernel) + bias
    grads = tape.gradient(score, x_t)
    return np.asarray(grads) * np.asarray(x)


def _gini(values: np.ndarray) -> float:
    """Concentration of a non-negative vector: 0 uniform, approaching 1 if one unit carries it."""
    v = np.sort(np.abs(values))
    n = len(v)
    total = v.sum()
    if n == 0 or total <= 0.0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * (index * v).sum()) / (n * total) - (n + 1.0) / n)


def _mean_pairwise_cosine(maps: np.ndarray) -> float:
    """Average cosine similarity between all pairs of rows.

    Computed from the norm of the summed unit vectors rather than the full
    pairwise matrix, which is the same quantity in O(n d) instead of O(n^2 d).
    """
    norms = np.linalg.norm(maps, axis=1, keepdims=True)
    keep = norms.reshape(-1) > 0
    if keep.sum() < 2:
        return 0.0
    unit = maps[keep] / norms[keep]
    n = len(unit)
    total = float((unit.sum(axis=0) ** 2).sum())
    return float((total - n) / (n * (n - 1)))


def relevance_structure(relevance: np.ndarray, y: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Scalar summaries of a relevance map: does the attribution have structure?

    * ``concentration`` -- Gini of the mean relevance over units.  A decision
      carried by a few units scores high; one smeared across the layer scores low.
    * ``class_separation`` -- cosine distance between the mean relevance maps of
      the two classes.  This is Grad-CAM's actual claim, class-discriminative
      attribution: a map that looks identical whatever the answer explains nothing.
    * ``consistency`` -- mean pairwise cosine similarity of maps within a class,
      minus the same quantity computed on label-shuffled groups.  Subtracting the
      shuffled baseline matters because any two non-negative vectors already have
      positive cosine, so the raw statistic is large even for unstructured maps.
    """
    mean_map = relevance.mean(axis=0)
    stats = {f"{prefix}concentration": _gini(mean_map)}

    classes = np.unique(y)
    if len(classes) == 2:
        a = relevance[y == classes[0]].mean(axis=0)
        b = relevance[y == classes[1]].mean(axis=0)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cos = float(a @ b / denom) if denom > 0 else 1.0
        stats[f"{prefix}class_separation"] = 1.0 - cos

        within = np.mean([_mean_pairwise_cosine(relevance[y == c]) for c in classes])
        rng = np.random.default_rng(0)
        shuffled = rng.permutation(y)
        baseline = np.mean([_mean_pairwise_cosine(relevance[shuffled == c]) for c in classes])
        stats[f"{prefix}consistency"] = float(within - baseline)

    return stats


def attribution_shift(
    full_relevance: np.ndarray,
    ablated_relevance: np.ndarray,
    feature_names: list[str],
    ablated_index: int,
) -> dict[str, float]:
    """How input attribution redistributes when a feature is removed.

    ``redistribution`` is the total absolute attribution the remaining features
    gain.  If the removed feature's contribution really is recoverable from its
    imprints, the network must find it somewhere, and this is where it shows up.
    """
    full = np.abs(full_relevance).mean(axis=0)
    ablated = np.abs(ablated_relevance).mean(axis=0)

    others = [i for i in range(len(feature_names)) if i != ablated_index]
    gained = ablated[others] - full[others]

    return {
        "attr_lost_on_ablated": float(full[ablated_index]),
        "attr_redistribution": float(np.abs(gained).sum()),
        "attr_max_gainer": feature_names[others[int(np.argmax(gained))]] if others else "",
        "attr_max_gain": float(gained.max()) if others else 0.0,
    }
