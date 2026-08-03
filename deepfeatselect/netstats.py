"""Permutation-invariant functionals of a single trained network's internals.

Why this module exists.  Accuracy-based ablation is provably blind to a
deterministically redundant feature: if ``X_j = g(X_-j)`` almost surely, the
sigma-algebras of the full and reduced inputs coincide, so the achievable risk
is identical and leave-one-out importance is exactly zero however strongly
``X_j`` drives the target.  The redundancy_demo benchmark measured precisely
that zero.  Comparing the raw weights of a with/without-feature pair does not
work either, because the loss is invariant under hidden-unit permutation and
training is stochastic besides -- two equally good networks can have arbitrarily
different weight tensors.

The way out is to compare functionals that are invariant to hidden-unit
relabelling.  Everything here is: activation statistics are unit-order-free
aggregates, spectral quantities are invariant to permutation (and orthogonal
rotation), and the compression statistics are computed on a canonicalised
weight ordering.  Two networks trained with and without a feature can then be
compared functional by functional, with significance calibrated against the
same contrast run on known-irrelevant features.

The capacity caveat matters: with a wide enough network the reduced model
synthesises the missing feature at no representational cost and every contrast
here converges to zero along with the accuracy contrast.  These probes are
informative in the *constrained* regime, where re-deriving ``g`` costs
capacity.  Use a deliberately small network.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import keras
import numpy as np

from .attribution import relevance_structure, unit_relevance
from .data import Dataset
from .model import build_model
from .train import TrainConfig

# ---------------------------------------------------------------------------
# activation-side functionals


def participation_ratio(acts: np.ndarray) -> float:
    """Effective dimensionality of a set of activation vectors.

    PR = (sum lambda)^2 / sum lambda^2 over covariance eigenvalues: 1 when a
    single direction carries everything, the layer width when the units are
    isotropic.  Invariant to unit permutation and to any orthogonal rotation of
    the representation, which is what makes it comparable across retrainings.
    """
    centred = acts - acts.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centred, compute_uv=False)
    lam = s**2
    total = lam.sum()
    if total <= 0.0:
        return 0.0
    return float(total**2 / (lam**2).sum())


def activation_stats(model: keras.Model, x_probe: np.ndarray, prefix: str) -> dict[str, float]:
    """Distributional statistics of the hidden ReLU activations on a probe set.

    All statistics are aggregates over units (means, counts, entropies of the
    sorted firing profile), never tied to a unit's index, so they survive
    permutation.  ``distinct_frac`` is the Hanin-Rolnick style count of distinct
    binary activation patterns, as a fraction of the probe count.
    """
    hidden = [l for l in model.layers if l.name.startswith("fc_")]
    sub = keras.Model(model.inputs, [l.output for l in hidden])
    outs = sub.predict(x_probe, verbose=0)
    if not isinstance(outs, list):
        outs = [outs]

    active, dead, pr_norm, fire_h, distinct = [], [], [], [], []
    for acts in outs:
        on = acts > 0.0
        rates = on.mean(axis=0)
        active.append(on.mean())
        dead.append(float((~on.any(axis=0)).mean()))
        pr_norm.append(participation_ratio(acts) / acts.shape[1])
        # Mean per-unit Bernoulli entropy of the firing rate, in bits. A layer
        # of always-on or always-off units scores 0 regardless of which units
        # they are.
        p = np.clip(rates, 1e-12, 1.0 - 1e-12)
        fire_h.append(float(np.mean(-p * np.log2(p) - (1 - p) * np.log2(1 - p))))
        distinct.append(len(np.unique(on, axis=0)) / on.shape[0])

    return {
        f"act_active_frac_{prefix}": float(np.mean(active)),
        f"act_dead_frac_{prefix}": float(np.mean(dead)),
        f"act_pr_norm_{prefix}": float(np.mean(pr_norm)),
        f"act_firing_entropy_{prefix}": float(np.mean(fire_h)),
        f"act_distinct_frac_{prefix}": float(np.mean(distinct)),
    }


# ---------------------------------------------------------------------------
# weight-side functionals


def _dense_chain(model: keras.Model) -> list[keras.layers.Dense]:
    chain = [l for l in model.layers if l.name.startswith("fc_")]
    chain.append(model.get_layer("output"))
    return chain


def _quantise(values: np.ndarray) -> np.ndarray:
    """Symmetric 8-bit quantisation, the common ruler for both entropy estimates."""
    scale = np.abs(values).max()
    if scale == 0.0:
        return np.zeros(values.size, dtype=np.uint8)
    return np.round((values / scale) * 127.0 + 128.0).astype(np.uint8)


def weight_stats(model: keras.Model) -> dict[str, float]:
    """Spectral and compression statistics of the dense-layer kernels.

    Two entropy estimates on the same 8-bit quantisation:

    * ``w_entropy_bits`` -- Shannon entropy of the pooled value histogram.
      Order-free by construction; measures the weight *distribution* only.
    * ``w_gzip_bits`` -- zlib bits per weight on a canonicalised ordering
      (hidden units sorted by incoming-kernel norm, the permutation propagated
      to the next layer's rows).  Canonicalisation makes the byte stream
      permutation-invariant while preserving within-unit structure, so this
      also sees correlations between weights, which a histogram cannot.

    If the two disagree strongly, structure -- not the value distribution -- is
    what changed.
    """
    chain = _dense_chain(model)
    kernels = [np.asarray(keras.ops.convert_to_numpy(l.kernel)) for l in chain]

    spectral, stable = [], []
    for w in kernels:
        s = np.linalg.svd(w, compute_uv=False)
        energy = s**2
        total = energy.sum()
        if total <= 0.0:
            spectral.append(0.0)
            stable.append(0.0)
            continue
        if len(s) < 2:
            # A rank-one shape (the single-logit output head) has no spectral
            # spread to measure; entropy is zero by definition, not NaN.
            spectral.append(0.0)
            stable.append(float(total / energy.max()))
            continue
        p = energy / total
        p = p[p > 0]
        spectral.append(float(-(p * np.log(p)).sum() / np.log(len(s))))
        stable.append(float(total / energy.max()))

    # Canonical order: sort each hidden layer's units by incoming norm and carry
    # the permutation into the next kernel's rows, so any relabelling of hidden
    # units maps to the same byte stream (up to norm ties, which are measure
    # zero after training).
    canon = [k.copy() for k in kernels]
    for i in range(len(canon) - 1):
        order = np.argsort(-np.linalg.norm(canon[i], axis=0), kind="stable")
        canon[i] = canon[i][:, order]
        canon[i + 1] = canon[i + 1][order, :]

    pooled = np.concatenate([k.ravel() for k in canon])
    q = _quantise(pooled)
    counts = np.bincount(q, minlength=256).astype(np.float64)
    freq = counts[counts > 0] / counts.sum()
    entropy_bits = float(-(freq * np.log2(freq)).sum())
    gzip_bits = 8.0 * len(zlib.compress(q.tobytes(), 9)) / q.size

    return {
        "w_spectral_entropy": float(np.mean(spectral)),
        "w_stable_rank": float(np.mean(stable)),
        "w_frobenius": float(np.sqrt(sum((k**2).sum() for k in kernels))),
        "w_entropy_bits": entropy_bits,
        "w_gzip_bits": gzip_bits,
    }


# ---------------------------------------------------------------------------
# residual-side functionals


def _rbf_gram(x: np.ndarray) -> np.ndarray:
    sq = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    positive = sq[sq > 0]
    # Median heuristic; degenerate (constant) inputs get a unit bandwidth,
    # where HSIC is zero anyway.
    bw = np.median(positive) if positive.size else 1.0
    return np.exp(-sq / bw)


def residual_hsic(residuals: np.ndarray, x: np.ndarray, n_permutations: int = 200,
                  seed: int = 0) -> tuple[float, float]:
    """HSIC between residuals and inputs, with a permutation p-value.

    The certificate of under-fitting: at the population optimum within the
    class, residual dependence on the retained inputs is exactly what remains
    of the structure the class could not express.  Residual *size* cannot
    distinguish a hard problem from a badly fit one; residual *dependence* can.
    """
    r = residuals.reshape(-1, 1)
    n = len(r)
    h = np.eye(n) - np.ones((n, n)) / n
    k = h @ _rbf_gram(x) @ h
    gram_r = _rbf_gram(r)
    stat = float((k * gram_r).sum() / (n - 1) ** 2)

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(n)
        null[i] = (k * gram_r[np.ix_(perm, perm)]).sum() / (n - 1) ** 2
    p = float((1.0 + (null >= stat).sum()) / (n_permutations + 1.0))
    return stat, p


def residual_autocorr(residuals: np.ndarray, max_lag: int = 5) -> float:
    """Mean |autocorrelation| of the residuals over the first ``max_lag`` lags.

    Only meaningful when the rows have a time order and the evaluation split is
    contiguous.  Structure here means the model is leaving temporally organised
    signal on the table -- the fingerprint asked about in the original proposal.
    """
    r = residuals - residuals.mean()
    denom = (r**2).sum()
    if denom <= 0.0:
        return 0.0
    acf = [np.abs((r[: -lag] * r[lag:]).sum() / denom) for lag in range(1, max_lag + 1)]
    return float(np.mean(acf))


# ---------------------------------------------------------------------------
# training + measurement in one pass


@dataclass
class InternalsResult:
    """Everything measured on one trained network."""

    metrics: dict[str, float] = field(default_factory=dict)


def train_and_measure(data: Dataset, config: TrainConfig, seed: int = 0,
                      n_probe: int = 256) -> InternalsResult:
    """Train one network for a FIXED epoch budget and measure its internals.

    No early stopping: the learning-curve statistics (final loss, area under
    the validation curve) are only comparable across ablation arms if every arm
    gets the same number of optimisation steps.  Early stopping would hand each
    arm a different budget and turn the learning-speed contrast into a stopping
    -time artefact -- the same confound that corrupted the first L1 sweep.
    """
    keras.utils.set_random_seed(seed)
    model = build_model(
        n_columns=data.n_columns,
        groups=data.groups,
        n_classes=data.n_classes,
        task=config.task,
        l1_gate=config.l1_gate,
        l2_dense=config.l2_dense,
        hidden_units=config.hidden_units,
        n_hidden_layers=config.n_hidden_layers,
        dropout=config.dropout,
        noise=config.noise,
        learning_rate=config.learning_rate,
        loss=config.loss,
        proximal=config.proximal,
    )

    y_train = data.y_train.astype("float32").reshape(-1, 1)
    y_val = data.y_val.astype("float32").reshape(-1, 1)
    history = model.fit(
        data.x_train, y_train,
        validation_data=(data.x_val, y_val),
        epochs=config.epochs, batch_size=config.batch_size,
        shuffle=True, verbose=0,
    )
    curve = np.asarray(history.history["val_loss"], dtype=np.float64)

    metrics: dict[str, float] = {
        "val_loss_final": float(curve[-5:].mean()),
        "val_loss_area": float(curve.mean()),
    }

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data.x_train), size=min(n_probe, len(data.x_train)), replace=False)
    metrics.update(activation_stats(model, data.x_train[idx], prefix="data"))

    # Grad-CAM-style relevance: which units the decision rests on, as opposed to
    # how many units fire. Output-conditioned, so it carries structure the
    # activation statistics above are blind to.
    relevance = unit_relevance(model, data.x_train[idx])
    metrics.update(relevance_structure(relevance, data.y_train[idx], prefix="relev_"))
    # Random probes ask about the learned function over the whole input space,
    # not just on the data manifold -- the two can diverge and the divergence is
    # itself informative.
    metrics.update(activation_stats(
        model, rng.standard_normal((n_probe, data.n_columns)).astype("float32"), prefix="rand"))
    metrics.update(weight_stats(model))

    probs = model.predict(data.x_test, verbose=0).reshape(-1)
    resid = data.y_test.astype(np.float64) - probs
    stat, p = residual_hsic(resid, data.x_test, seed=seed)
    metrics["resid_hsic"] = stat
    metrics["resid_hsic_p"] = p
    metrics["resid_acf"] = residual_autocorr(resid)

    return InternalsResult(metrics=metrics)
