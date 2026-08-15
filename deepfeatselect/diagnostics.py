"""Post-training diagnostics: has this network actually learned?

Validation loss answers the question crudely -- it says how well the model
predicts, not whether it found structure or memorised. The literature has better
instruments, and all of them run after training with no extra fitting, which is
what makes them usable as a quality gate on an experiment that has already run.

Three families here, beyond the activation and weight statistics already in
``netstats``:

* **Heavy-tailed spectral analysis** (Martin & Mahoney). The eigenvalue spectrum
  of a well-trained layer's weight correlation matrix follows a power law, and
  the fitted exponent says which regime the layer is in: roughly 2-4 for a layer
  that has captured structure, above ~6 for one that is still close to random.
  It needs no data at all, only the weights, which makes it the cleanest
  available check that training did something.
* **Sharpness**. Flat minima generalise better than sharp ones, so the loss
  increase under a small random weight perturbation is a proxy for how brittle
  the solution is.
* **Neural collapse** (Papyan, Han & Donoho). Late in successful training the
  penultimate features of each class concentrate, and the ratio of within-class
  to between-class scatter falls. A high ratio means the representation never
  organised.

Each is reported per network so that a run can be excluded when the network did
not learn, rather than having its ablation scores averaged in regardless.
"""

from __future__ import annotations

import keras
import numpy as np


def power_law_alpha(kernel: np.ndarray, min_tail: int = 10) -> float:
    """Hill estimator of the power-law exponent of a weight matrix spectrum.

    Following the heavy-tailed self-regularisation picture: the eigenvalues of
    ``W^T W`` in a trained layer follow a power law, and the exponent indicates
    how much structure the layer has taken on. Values near 2 mean strongly
    heavy-tailed and well fitted; large values mean the spectrum is still close
    to the Marchenko-Pastur bulk a random matrix would give.

    Returns ``inf`` when the tail is too short to fit, which is itself a signal
    that the layer is too small to judge this way.
    """
    eig = np.linalg.svd(kernel, compute_uv=False) ** 2
    eig = np.sort(eig[eig > 0])[::-1]
    if len(eig) < min_tail + 2:
        return float("inf")

    # Hill estimator over the largest `tail` eigenvalues, with the tail chosen
    # as half the spectrum: long enough to fit, short enough to stay in the tail.
    tail = max(min_tail, len(eig) // 2)
    top, cutoff = eig[:tail], eig[tail]
    if cutoff <= 0:
        return float("inf")
    logs = np.log(top / cutoff)
    total = logs.sum()
    if total <= 0:
        return float("inf")
    return float(1.0 + tail / total)


def spectral_diagnostics(model: keras.Model) -> dict[str, float]:
    """Power-law exponent and effective rank, averaged over the dense layers."""
    kernels = [np.asarray(keras.ops.convert_to_numpy(l.kernel))
               for l in model.layers if l.name.startswith("fc_")]
    if not kernels:
        return {}

    alphas, eff_ranks = [], []
    for w in kernels:
        alphas.append(power_law_alpha(w))
        s = np.linalg.svd(w, compute_uv=False)
        p = s / (s.sum() + 1e-12)
        p = p[p > 0]
        # exp(spectral entropy): the number of singular directions actually used.
        eff_ranks.append(float(np.exp(-(p * np.log(p)).sum())))

    finite = [a for a in alphas if np.isfinite(a)]
    return {
        "alpha": float(np.mean(finite)) if finite else float("nan"),
        "effective_rank": float(np.mean(eff_ranks)),
    }


def sharpness(model: keras.Model, x: np.ndarray, y: np.ndarray,
              rho: float = 0.05, n_probes: int = 5, seed: int = 0) -> float:
    """Mean loss increase under a small random weight perturbation.

    Perturbation is scaled to each tensor's own norm, so layers with different
    magnitudes are stressed comparably. A flat minimum barely moves; a sharp one
    degrades quickly, which is the classic generalisation signal.
    """
    rng = np.random.default_rng(seed)
    weights = model.get_weights()
    base = float(model.evaluate(x, y, verbose=0)[0])

    losses = []
    for _ in range(n_probes):
        perturbed = []
        for w in weights:
            scale = rho * (np.linalg.norm(w) / np.sqrt(w.size) + 1e-12)
            perturbed.append(w + rng.normal(0.0, scale, size=w.shape))
        model.set_weights(perturbed)
        losses.append(float(model.evaluate(x, y, verbose=0)[0]))
    model.set_weights(weights)
    return float(np.mean(losses) - base)


def neural_collapse(model: keras.Model, x: np.ndarray, y: np.ndarray) -> float:
    """Within-class over between-class scatter of the penultimate features.

    Falls as a network's representation organises by class. A value near or
    above one means the classes are not separated in the representation at all,
    whatever the loss says.
    """
    hidden = [l for l in model.layers if l.name.startswith("fc_")]
    if not hidden:
        return float("nan")
    sub = keras.Model(model.inputs, hidden[-1].output)
    features = np.asarray(sub.predict(x, verbose=0))

    # Callers hand in whatever shape Keras wanted, commonly (n, 1); the mask
    # below has to be one-dimensional or it broadcasts against the feature axis.
    y = np.asarray(y).reshape(-1)
    classes = np.unique(y)
    if len(classes) < 2:
        return float("nan")
    overall = features.mean(axis=0)

    within, between = 0.0, 0.0
    for c in classes:
        block = features[y == c]
        if len(block) < 2:
            continue
        centre = block.mean(axis=0)
        within += ((block - centre) ** 2).sum()
        between += len(block) * ((centre - overall) ** 2).sum()
    if between <= 0:
        return float("inf")
    return float(within / between)


def learned_well(model: keras.Model, x: np.ndarray, y: np.ndarray,
                 val_loss: float, chance: float = float(np.log(2)),
                 seed: int = 0) -> dict[str, float]:
    """Every diagnostic at once, plus a single pass/fail verdict.

    ``verdict`` is 1.0 only when the model beats chance *and* its representation
    separates the classes. Loss alone can look acceptable while the penultimate
    features are unorganised, and the two failures want telling apart.
    """
    out: dict[str, float] = {"val_loss": val_loss}
    out.update(spectral_diagnostics(model))
    out["sharpness"] = sharpness(model, x, y, seed=seed)
    out["neural_collapse"] = neural_collapse(model, x, y)
    out["verdict"] = float(val_loss < chance and out["neural_collapse"] < 1.0)
    return out
