"""Model-difference probes: what changes *inside* a network when a feature goes.

A gate value says how much a feature was used; a drop in test AUC says how much
the prediction suffered without it.  Neither says what the network did instead.
On a small, correlated dataset like this one the accuracy drop is close to
useless on its own: a feature with a near-perfect surrogate can be removed with
no measurable loss, and its importance reads as zero even though the network had
to rebuild its internal representation to cope.

The probes here train a full model and an ablated one and compare them at three
depths, so that "no accuracy change" and "no change" can be told apart:

* the predictive distribution (Jensen-Shannon divergence, delta AUC/F1);
* the logit scale the softmax discards -- free energy, Grathwohl et al. (2020);
* the penultimate representation itself -- linear CKA (Kornblith et al., 2019)
  and modern Hopfield energy (Ramsauer et al., 2020) against the training-set
  activations used as stored memories.

Caveats worth stating up front, because these numbers are easy to over-read:

* Two networks trained from the same seed still diverge for reasons unrelated to
  the ablation (dropout draws, batch order under a different loss surface).  A
  non-zero CKA gap is therefore an upper bound on the ablation's effect, not an
  estimate of it.  Compare a feature against the *other features'* probes rather
  than against zero.
* Free energy and Hopfield energy are compared as distributions, not per sample:
  the two models have different representation spaces, so their energies are not
  paired quantities.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace

import keras
import numpy as np
import pandas as pd
from scipy.special import logsumexp, rel_entr
from scipy.stats import wasserstein_distance
from sklearn.metrics import f1_score, roc_auc_score

from . import train as _train
from .data import Dataset
from .train import RunResult, TrainConfig


def free_energy(logits: np.ndarray) -> np.ndarray:
    """Free energy ``E(x) = -logsumexp_k f_k(x)`` of a classifier's logits.

    From Grathwohl et al. (2020), "Your classifier is secretly an energy based
    model and you should treat it like one".  The logits of a k-way classifier
    define an unnormalised joint density over ``(x, y)``; marginalising out the
    label gives this energy.  Low energy means the network reads the point as
    typical of what it was trained on.

    The reason to look at it here: the softmax is invariant to a constant shift
    of the logits, so the predictive distribution throws this quantity away
    entirely.  An ablation can leave every predicted probability alone and still
    move the energy, which is precisely the kind of internal change these probes
    exist to catch.

    A single-logit binary head is the k=2 case with the first logit pinned at
    zero, so ``E(x) = -logsumexp([0, z]) = -softplus(z)``.  Both cases are
    implemented: an ``(n, 1)`` or ``(n,)`` input takes the softplus branch, an
    ``(n, k)`` input the logsumexp branch.

    Shape trap: a 1-D array is read as ``n`` single-logit predictions, not as
    one k-class prediction.  Pass shape ``(1, k)`` for a single multiclass row.

    Args:
        logits: Pre-activation outputs.  Passing probabilities instead is wrong
            twice over -- softmax probabilities have already fixed the shift the
            energy measures, and their log differs from the logits by exactly
            the constant being measured.

    Returns:
        One energy per row, shape ``(n,)``.
    """
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    if z.ndim != 2:
        raise ValueError(f"logits must be 1-D or 2-D, got shape {z.shape}")

    if z.shape[1] == 1:
        # -softplus, via logaddexp so large |z| does not overflow.
        return -np.logaddexp(0.0, z[:, 0])
    return -logsumexp(z, axis=1)


def hopfield_energy(
    query: np.ndarray, memories: np.ndarray, beta: float = 1.0
) -> np.ndarray:
    """Modern Hopfield energy of ``query`` against a set of stored ``memories``.

    Equation (2) of Ramsauer et al. (2020), "Hopfield Networks is All You Need"::

        E = -lse(beta, memories @ query) + 0.5 * query @ query
            + beta**-1 * log(n_memories) + 0.5 * max_row_norm**2

    with ``lse(beta, z) = beta**-1 * log(sum(exp(beta * z)))``.  The last two
    terms are constants in ``query``; they exist to make ``E`` bounded below and
    are kept so the values sit on the paper's scale.

    Used here with the penultimate activations of the *training set* as stored
    memories: the energy of a test point is then a measure of how well the
    trained representation retrieves a stored pattern for it.  Ablating a feature
    that the network was leaning on shows up as test points falling into flatter
    parts of the energy landscape even when their class prediction is unchanged.

    Caveat when comparing two models: ``max_row_norm`` is a property of each
    model's own memories, so the constant offset differs between them.  A large
    shift can therefore reflect a change in the largest activation norm rather
    than in retrieval structure -- read it together with the CKA.

    Args:
        query: One query per row, shape ``(d,)`` or ``(n_queries, d)``.  A 1-D
            query is promoted to a single row, so the result has shape ``(1,)``.
        memories: Stored patterns, shape ``(n_memories, d)``.
        beta: Inverse temperature.  Large ``beta`` makes the energy approach the
            single best-matching memory; small ``beta`` averages over all of them.

    Returns:
        One energy per query row, shape ``(n_queries,)``.
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")

    q = np.atleast_2d(np.asarray(query, dtype=np.float64))
    x = np.asarray(memories, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"memories must be a non-empty 2-D array, got shape {x.shape}")
    if q.shape[1] != x.shape[1]:
        raise ValueError(
            f"query dimension {q.shape[1]} does not match memory dimension {x.shape[1]}"
        )

    # (n_queries, n_memories) similarities: this materialises the full matrix,
    # which is fine for a few hundred training rows but is the thing to chunk
    # if the memory set ever gets large.
    similarity = q @ x.T
    lse = logsumexp(beta * similarity, axis=1) / beta

    max_norm = float(np.linalg.norm(x, axis=1).max())
    return (
        -lse
        + 0.5 * np.einsum("ij,ij->i", q, q)
        + np.log(x.shape[0]) / beta
        + 0.5 * max_norm**2
    )


def cka(a: np.ndarray, b: np.ndarray) -> float:
    """Linear Centered Kernel Alignment between two representation matrices.

    Kornblith et al. (2019), "Similarity of Neural Network Representations
    Revisited".  Rows are examples and must be the *same* examples in the same
    order in both matrices; columns are neurons and need not correspond, nor even
    match in number.  That is the point: CKA is invariant to orthogonal rotation
    and to isotropic scaling of either representation, so it compares the
    geometry two networks impose on the data rather than their coordinates,
    which are arbitrary.

    Computed in the feature-space form ``||A'B||_F^2 / (||A'A||_F ||B'B||_F)`` on
    column-centred matrices, which is algebraically the Gram-matrix definition
    but costs ``O(n d^2)`` instead of ``O(n^2 d)``.

    Returns:
        A value in ``[0, 1]``; 1 means identical up to rotation and scale.  NaN
        if either representation is constant, since a matrix with no variance has
        no structure to align.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"both inputs must be 2-D, got {x.shape} and {y.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"both inputs must describe the same examples, got {x.shape[0]} and "
            f"{y.shape[0]} rows"
        )

    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)

    cross = float(np.linalg.norm(x.T @ y, ord="fro") ** 2)
    norm_x = float(np.linalg.norm(x.T @ x, ord="fro"))
    norm_y = float(np.linalg.norm(y.T @ y, ord="fro"))
    if norm_x < 1e-12 or norm_y < 1e-12:
        return float("nan")
    return cross / (norm_x * norm_y)


def ablate_feature(data: Dataset, feature_index: int) -> Dataset:
    """Copy of ``data`` with one feature's columns zeroed in every split.

    Zeroed, not dropped.  The two models have to share an architecture, a column
    count and a group vector or nothing measured afterwards is attributable to
    the ablation: dropping columns changes the input width, the first layer's
    parameter count and therefore the random initialisation, and the resulting
    networks are simply two different models.

    Zero is also the right constant rather than a convenient one.  The columns
    are standardised on the training split, so zero *is* the feature's training
    mean -- for a one-hot column, the category's base rate -- which is the value
    that carries no information about the individual row.

    A nominal feature spans several columns, so this selects through
    ``groups`` rather than taking a single column.
    """
    if not 0 <= feature_index < data.n_features:
        raise ValueError(
            f"feature_index must be in [0, {data.n_features}), got {feature_index}"
        )

    columns = np.flatnonzero(data.groups == feature_index)
    if columns.size == 0:
        raise ValueError(f"feature {feature_index} has no columns in the group vector")

    def blank(x: np.ndarray) -> np.ndarray:
        out = x.copy()
        out[:, columns] = 0.0
        return out

    return replace(
        data,
        x_train=blank(data.x_train),
        x_val=blank(data.x_val),
        x_test=blank(data.x_test),
    )


@dataclass(frozen=True)
class AblationResult:
    """One feature's full-versus-ablated comparison.

    ``delta_*`` are full minus ablated, so positive means removing the feature
    hurt.  ``js_divergence``, ``energy_shift`` and ``hopfield_shift`` are
    non-negative magnitudes of change with no sign to read.
    ``representation_cka`` runs the other way: 1 means the penultimate geometry
    survived the ablation intact, and lower values mean it was rebuilt.
    """

    feature: str
    feature_index: int
    n_columns: int
    seed: int
    delta_auc: float
    delta_f1: float
    js_divergence: float
    energy_shift: float
    hopfield_shift: float
    representation_cka: float
    full_metrics: dict[str, float]
    ablated_metrics: dict[str, float]

    def as_row(self) -> dict[str, float | str]:
        return {
            "feature": self.feature,
            "feature_index": self.feature_index,
            "n_columns": self.n_columns,
            "seed": self.seed,
            "delta_auc": self.delta_auc,
            "delta_f1": self.delta_f1,
            "js_divergence": self.js_divergence,
            "energy_shift": self.energy_shift,
            "hopfield_shift": self.hopfield_shift,
            "representation_cka": self.representation_cka,
        }


@dataclass(frozen=True)
class _Trace:
    """Everything read out of one trained model at inference time."""

    probs: np.ndarray
    energy: np.ndarray
    hopfield: np.ndarray
    reps: np.ndarray


@contextmanager
def _capture_model() -> Iterator[list[keras.Model]]:
    """Grab the model :func:`~deepfeatselect.train.train_one` builds.

    ``train_one`` returns gates and scores, not the fitted network, and its
    public API is not ours to change -- so the model is caught by wrapping the
    factory it calls.  The alternative, reimplementing the fit loop here with its
    early stopping, class weights and hierarchy projection, would drift out of
    sync with the trainer and make the two halves of a probe differ for reasons
    that have nothing to do with the ablated feature.

    Not thread-safe: it swaps a module attribute for the duration.  Probes run
    sequentially in one process for that reason.
    """
    built: list[keras.Model] = []
    original = _train.build_model

    def recording(*args, **kwargs):
        model = original(*args, **kwargs)
        built.append(model)
        return model

    _train.build_model = recording
    try:
        yield built
    finally:
        _train.build_model = original


def _train_capture(
    data: Dataset, config: TrainConfig, seed: int, verbose: int
) -> tuple[RunResult, keras.Model]:
    with _capture_model() as built:
        run = _train.train_one(data, config, seed=seed, verbose=verbose)
    if len(built) != 1:
        raise RuntimeError(
            f"expected train_one to build exactly one model, captured {len(built)}"
        )
    return run, built[0]


def _penultimate_layer(model: keras.Model) -> keras.layers.Layer:
    """The last hidden Dense layer, whose output is what the head sees.

    Dropout and GaussianNoise sit between it and the head but are identities at
    inference, so no separate handling is needed as long as representations are
    read through ``predict``.
    """
    for layer in reversed(model.layers[:-1]):
        if isinstance(layer, keras.layers.Dense):
            return layer
    raise ValueError("model has no hidden Dense layer to read representations from")


def _trace(model: keras.Model, data: Dataset, beta: float) -> _Trace:
    """Read predictions, energies and representations out of a trained model."""
    hidden = _penultimate_layer(model)
    extractor = keras.Model(inputs=model.inputs, outputs=hidden.output)

    reps = np.asarray(extractor.predict(data.x_test, verbose=0), dtype=np.float64)
    memories = np.asarray(extractor.predict(data.x_train, verbose=0), dtype=np.float64)

    # Logits have to be reconstructed from the head's weights because the output
    # layer bakes in its activation. Taking log(probs) instead would not do: for
    # a softmax head that recovers the logits only up to the additive constant
    # that free energy is entirely made of.
    head = model.layers[-1]
    weights = head.get_weights()
    kernel = np.asarray(weights[0], dtype=np.float64)
    bias = np.asarray(weights[1], dtype=np.float64) if len(weights) > 1 else 0.0
    logits = reps @ kernel + bias

    return _Trace(
        probs=np.asarray(model.predict(data.x_test, verbose=0), dtype=np.float64),
        energy=free_energy(logits),
        hopfield=hopfield_energy(reps, memories, beta=beta),
        reps=reps,
    )


def _predictive_distribution(probs: np.ndarray, task: str) -> np.ndarray:
    """Rows that sum to one, so binary and multiclass heads compare the same way."""
    if task == "binary":
        p = probs.reshape(-1, 1)
        return np.hstack([1.0 - p, p])
    return probs


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Mean per-sample Jensen-Shannon divergence in bits.

    Symmetric and bounded in ``[0, 1]`` for base-2 logs, unlike the KL either
    way round, which is what makes it readable as "how far apart are these two
    models' predictions" without a scale to calibrate first.
    """
    m = 0.5 * (p + q)
    # rel_entr handles the 0 log 0 = 0 convention that a bare log would turn into
    # a NaN, which matters because saturated sigmoids do produce exact zeros.
    kl_pm = rel_entr(p, m).sum(axis=1)
    kl_qm = rel_entr(q, m).sum(axis=1)
    return float(np.mean(0.5 * (kl_pm + kl_qm)) / np.log(2.0))


def _auc_f1(y_true: np.ndarray, probs: np.ndarray, task: str) -> tuple[float, float]:
    """AUC and F1, computed here so the multiclass case gets an AUC too.

    ``train._score`` omits AUC for multiclass; a one-vs-rest macro AUC is well
    defined as long as every class appears in the test split, and is reported as
    NaN when it does not rather than aborting the probe.
    """
    if task == "binary":
        p = probs.reshape(-1)
        y_pred = (p >= 0.5).astype(int)
        try:
            auc = float(roc_auc_score(y_true, p))
        except ValueError:
            auc = float("nan")
        return auc, float(f1_score(y_true, y_pred))

    y_pred = probs.argmax(axis=1)
    try:
        auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    return auc, float(f1_score(y_true, y_pred, average="macro"))


def _compare(
    full: _Trace, ablated: _Trace, y_test: np.ndarray, task: str
) -> dict[str, float]:
    full_auc, full_f1 = _auc_f1(y_test, full.probs, task)
    ablated_auc, ablated_f1 = _auc_f1(y_test, ablated.probs, task)

    return {
        "delta_auc": full_auc - ablated_auc,
        "delta_f1": full_f1 - ablated_f1,
        "js_divergence": _js_divergence(
            _predictive_distribution(full.probs, task),
            _predictive_distribution(ablated.probs, task),
        ),
        # 1-D Wasserstein between the two energy samples. The energies are not
        # paired across models -- the representations live in different spaces --
        # so this compares distributions, and its units are nats of energy.
        "energy_shift": float(wasserstein_distance(full.energy, ablated.energy)),
        "hopfield_shift": float(wasserstein_distance(full.hopfield, ablated.hopfield)),
        "representation_cka": cka(full.reps, ablated.reps),
    }


def _probe_against(
    data: Dataset,
    config: TrainConfig,
    full_run: RunResult,
    full_trace: _Trace,
    feature_index: int,
    seed: int,
    hopfield_beta: float,
    verbose: int,
) -> AblationResult:
    """Train the ablated half of a probe and compare it with a full model already
    trained.  Split out so :func:`loco_importance` pays for the full model once."""
    ablated_data = ablate_feature(data, feature_index)
    ablated_run, ablated_model = _train_capture(ablated_data, config, seed, verbose)
    ablated_trace = _trace(ablated_model, ablated_data, hopfield_beta)

    return AblationResult(
        feature=data.feature_names[feature_index],
        feature_index=feature_index,
        n_columns=int((data.groups == feature_index).sum()),
        seed=seed,
        full_metrics=dict(full_run.metrics),
        ablated_metrics=dict(ablated_run.metrics),
        **_compare(full_trace, ablated_trace, data.y_test, config.task),
    )


def ablation_probe(
    data: Dataset,
    config: TrainConfig,
    feature_index: int,
    seed: int = 0,
    hopfield_beta: float = 1.0,
    verbose: int = 0,
) -> AblationResult:
    """Train a full and an ablated model and report how far apart they ended up.

    Both models are trained by :func:`~deepfeatselect.train.train_one` at the
    same seed, so they share their initialisation and differ only in that one
    feature's columns are zero.  That is deliberate: it removes initialisation
    from the comparison, which would otherwise dominate the representation
    metrics.  It does not remove all of the training noise -- see the module
    docstring.

    Args:
        feature_index: Index into ``data.feature_names``, not a column index.
        hopfield_beta: Inverse temperature for :func:`hopfield_energy`.
    """
    full_run, full_model = _train_capture(data, config, seed, verbose)
    full_trace = _trace(full_model, data, hopfield_beta)
    return _probe_against(
        data, config, full_run, full_trace, feature_index, seed, hopfield_beta, verbose
    )


def loco_importance(
    data: Dataset,
    config: TrainConfig,
    seed: int = 0,
    features: Sequence[int] | None = None,
    hopfield_beta: float = 1.0,
    verbose: int = 0,
    progress: bool = True,
) -> pd.DataFrame:
    """Leave-one-covariate-out probe over every feature, one row per feature.

    Trains ``1 + n_features`` models: the full model once, then one ablated model
    per feature.  Retraining rather than zeroing a feature at *prediction* time
    is the whole point -- prediction-time zeroing measures how much a fixed
    network misses the feature, while retraining measures how much of the signal
    the rest of the inputs can recover, which is the question a feature-selection
    decision actually turns on.

    Rows are sorted by ``delta_auc`` descending; ``feature_index`` is kept so the
    original ordering is recoverable.  Read ``delta_auc`` next to
    ``representation_cka``: features whose removal costs no accuracy but still
    rearranges the representation are the redundant-but-used ones, and they are
    invisible to accuracy-based importance.
    """
    if features is None:
        features = range(data.n_features)
    indices = list(features)

    full_run, full_model = _train_capture(data, config, seed, verbose)
    full_trace = _trace(full_model, data, hopfield_beta)

    rows: list[dict[str, float | str]] = []
    for position, feature_index in enumerate(indices, 1):
        result = _probe_against(
            data,
            config,
            full_run,
            full_trace,
            feature_index,
            seed,
            hopfield_beta,
            verbose,
        )
        rows.append(result.as_row())
        if progress:
            print(
                f"  loco {position}/{len(indices)} {result.feature}: "
                f"delta_auc={result.delta_auc:+.4f} cka={result.representation_cka:.3f}"
            )

    table = pd.DataFrame(rows)
    if table.empty:
        # An empty selection gives a frame with no columns at all, so sorting on
        # a name that is not there raises rather than returning nothing.
        return table
    return table.sort_values("delta_auc", ascending=False).reset_index(drop=True)
