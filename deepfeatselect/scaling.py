"""A system whose difficulty scales with the number of features.

Built to test one claim: that a trained network finds structure which
subset-enumerating methods cannot reach once the feature count grows.

The system carries three kinds of column.

* ``interaction`` -- k features whose *joint* parity determines part of the
  target. No proper subset of them carries any information at all, which is the
  defining property of parity and the reason marginal statistics are blind to
  them by construction rather than by weakness. Finding them by enumeration
  costs C(d, k) tests.
* ``marginal`` -- a few ordinary causes, each individually informative. These
  are the control: any method that cannot find *these* is broken rather than
  merely limited.
* ``noise`` -- everything else, and the count that grows.

Sweeping d with k fixed changes only how much irrelevant material the
interaction is buried in. A method whose cost is linear in d and whose
representation captures arbitrary order should degrade slowly; one that must
enumerate subsets is already infeasible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScalingSystem:
    x: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    interaction: np.ndarray   # boolean mask
    marginal: np.ndarray
    irrelevant: np.ndarray

    @property
    def causal(self) -> np.ndarray:
        return self.interaction | self.marginal


def oblique_interaction(
    n: int = 4000,
    n_features: int = 20,
    k: int = 4,
    n_marginal: int = 2,
    noise: float = 0.15,
    frequency: float = 2.0,
    seed: int = 0,
) -> ScalingSystem:
    """Target driven by a smooth, oblique function of k continuous features.

    ``parity_interaction`` turns out to favour trees rather than test them.  Its
    features are binary and its structure is axis-aligned, which is exactly the
    geometry a split on ``x_i > 0`` represents natively -- so a forest solves it
    directly and the comparison says nothing about representing nonlinearity.

    Here the interaction is ``sin(frequency * w . x)`` for a random unit vector
    ``w`` with no zero components, so the decision surface is oblique to every
    coordinate axis and smooth.  Approximating it with axis-aligned splits costs
    a staircase whose steps multiply with k, while a single dense layer holds
    ``w`` in one weight vector.  If the architecture argument is about coupling
    form rather than feature count, this is where it shows.

    All k features are needed: the projection is a sum over them, so dropping
    one leaves the rest carrying a different projection, and the sine makes any
    strict subset weakly informative at best rather than exactly uninformative.
    """
    rng = np.random.default_rng(seed)
    if k + n_marginal > n_features:
        raise ValueError("k + n_marginal cannot exceed n_features")

    x = rng.standard_normal((n, n_features))

    # A direction tilted away from every axis, so no single coordinate is a
    # usable proxy for the projection.
    w = rng.uniform(0.6, 1.0, size=k) * rng.choice([-1.0, 1.0], size=k)
    w /= np.linalg.norm(w)
    projection = x[:, :k] @ w

    logit = 3.0 * np.sin(frequency * projection) + 1.0 * x[:, k:k + n_marginal].sum(axis=1)
    flip = rng.uniform(size=n) < noise
    logit = np.where(flip, -logit, logit)
    y = (logit > 0).astype(np.int64)

    names = ([f"inter_{i}" for i in range(k)]
             + [f"marg_{i}" for i in range(n_marginal)]
             + [f"noise_{i}" for i in range(n_features - k - n_marginal)])
    mask = lambda lo, hi: np.array(  # noqa: E731
        [lo <= i < hi for i in range(n_features)])
    interaction, marginal_mask = mask(0, k), mask(k, k + n_marginal)
    irrelevant = mask(k + n_marginal, n_features)

    order = rng.permutation(n_features)
    return ScalingSystem(
        x=x[:, order],
        y=y,
        feature_names=[names[i] for i in order],
        interaction=interaction[order],
        marginal=marginal_mask[order],
        irrelevant=irrelevant[order],
    )


def parity_interaction(
    n: int = 4000,
    n_features: int = 20,
    k: int = 4,
    n_marginal: int = 2,
    noise: float = 0.15,
    seed: int = 0,
) -> ScalingSystem:
    """Target driven by a k-way parity plus a few marginal causes.

    The parity term is written as a product of centred bits rather than a sum
    modulo two, so it stays differentiable-friendly and keeps the same property:
    flipping any one participant flips the term, while any strict subset of them
    is independent of it.

    ``noise`` is label noise, applied by flipping the sign of the logit for a
    fraction of rows. Without it the target is deterministic and the task
    becomes an interpolation problem rather than a detection one.
    """
    rng = np.random.default_rng(seed)
    if k + n_marginal > n_features:
        raise ValueError("k + n_marginal cannot exceed n_features")

    bits = rng.integers(0, 2, size=(n, n_features)).astype(np.float64)
    signed = 2.0 * bits - 1.0

    # Product of the first k signed bits: +1 for even parity, -1 for odd.
    parity = np.prod(signed[:, :k], axis=1)
    # Marginal causes enter additively and are individually detectable.
    marginal = signed[:, k:k + n_marginal].sum(axis=1)

    logit = 2.0 * parity + 1.0 * marginal
    flip = rng.uniform(size=n) < noise
    logit = np.where(flip, -logit, logit)
    y = (logit > 0).astype(np.int64)

    names = ([f"inter_{i}" for i in range(k)]
             + [f"marg_{i}" for i in range(n_marginal)]
             + [f"noise_{i}" for i in range(n_features - k - n_marginal)])

    mask = lambda lo, hi: np.array(  # noqa: E731
        [lo <= i < hi for i in range(n_features)])
    interaction = mask(0, k)
    marginal_mask = mask(k, k + n_marginal)
    irrelevant = mask(k + n_marginal, n_features)

    # Scatter the columns. Built in block order the interaction members sit
    # adjacent, which is invisible to any order-invariant method (dense
    # networks, forests, mutual information) but would hand a free result to
    # anything with a local receptive field -- a convolution with kernel >= k
    # would span them by construction rather than by finding them. Permuting
    # removes that trap without changing the system.
    order = rng.permutation(n_features)
    return ScalingSystem(
        x=signed[:, order],
        y=y,
        feature_names=[names[i] for i in order],
        interaction=interaction[order],
        marginal=marginal_mask[order],
        irrelevant=irrelevant[order],
    )
