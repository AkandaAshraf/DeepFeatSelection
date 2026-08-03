"""Simulated systems whose causal structure is known, so methods can be scored.

Feature-importance arguments on real data are unfalsifiable: nobody knows the
true parents of ``num`` in the Cleveland file, so any ranking can be defended.
Everything here is generated from a structural model that is written down, which
turns "is this ranking right?" into an arithmetic question.

Two families, aimed at two different failure modes.

Family A, :func:`nonlinear_scm`, is cross-sectional.  It contains a *child* of
the target, ``x_effect``, which is by construction the most predictive column in
the table and by construction has zero causal effect.  Any method that maximises
predictive accuracy is supposed to rank it first; that is the point.  It also
contains two proxies of an observed confounder, which are marginally predictive
and conditionally useless.

Family B, :func:`coupled_logistic`, :func:`rossler_lorenz`,
:func:`independent_logistic`, is dynamical, for convergent cross mapping and
other time-series causality tests.  :func:`redundancy_demo` is the counterexample
that motivates using them at all: a deterministic system where a genuine cause
has exactly zero leave-one-covariate-out importance.

Nothing here is scaled or split.  Callers that want to push a design matrix
through the gated network should standardise it themselves;
:attr:`SyntheticDataset.groups` gives the identity column-to-feature map that
:class:`~deepfeatselect.model.FeatureGate` expects, since every synthetic
feature is continuous and owns exactly one column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Every dynamical generator returns the same dict layout so a scoring harness can
# loop over them without special cases: "x" and "y" are the observed scalar
# series, "coupling_x_to_y"/"coupling_y_to_x" are the strengths that produced
# them, and "true_direction" is the label a causality test has to recover.
SystemData = dict[str, Any]

DIRECTION_NONE = "none"
DIRECTION_X_TO_Y = "x->y"
DIRECTION_Y_TO_X = "y->x"
DIRECTION_BOTH = "x<->y"


def _sigmoid(v: np.ndarray) -> np.ndarray:
    """Logistic function, written through ``tanh`` to avoid overflow warnings.

    ``1/(1+exp(-v))`` raises a RuntimeWarning for large negative ``v``; this form
    is algebraically identical and finite everywhere.
    """
    return 0.5 * (1.0 + np.tanh(0.5 * v))


def _direction(x_to_y: float, y_to_x: float) -> str:
    """Ground-truth label for a pair of coupling strengths."""
    if x_to_y and y_to_x:
        return DIRECTION_BOTH
    if x_to_y:
        return DIRECTION_X_TO_Y
    if y_to_x:
        return DIRECTION_Y_TO_X
    return DIRECTION_NONE


@dataclass(frozen=True)
class SyntheticDataset:
    """A design matrix with the causal role of every column recorded alongside it.

    ``x`` is ``(n_samples, n_features)`` and unscaled.  The three masks are
    length-``n_features`` boolean arrays and answer three different questions:

    * ``direct_causes`` -- parents of the target in the generating DAG.  This is
      what an intervention-oriented question wants: change one of these and the
      distribution of ``y`` changes.
    * ``markov_blanket`` -- parents, children, and the other parents of those
      children.  Conditional on this set, ``y`` is independent of everything
      else, so it is the answer to the *prediction* question: the smallest set
      that loses no predictive information.  It is a superset of the direct
      causes and the two are not interchangeable.
    * ``irrelevant`` -- independent of ``y`` both marginally and conditionally.

    Columns can belong to none of the three.  ``confounded`` marks the ones here
    that do: they are marginally associated with ``y`` through an observed common
    driver, but are neither causes nor blanket members.
    """

    x: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    direct_causes: np.ndarray
    markov_blanket: np.ndarray
    irrelevant: np.ndarray
    confounded: np.ndarray
    effects: np.ndarray

    @property
    def n_samples(self) -> int:
        return self.x.shape[0]

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def groups(self) -> np.ndarray:
        """Identity column-to-feature map, for feeding straight to ``FeatureGate``.

        Real data here needs a non-trivial map because one-hot encoding spreads a
        feature over several columns; synthetic features are all continuous, so
        the map is ``arange``.  Exposing it anyway keeps call sites identical.
        """
        return np.arange(self.x.shape[1], dtype=np.int32)

    def names(self, mask: np.ndarray) -> list[str]:
        """Feature names selected by a boolean mask, in column order."""
        return [name for name, keep in zip(self.feature_names, mask) if keep]


def nonlinear_scm(n: int = 2000, noise: float = 0.3, seed: int = 0) -> SyntheticDataset:
    """A nonlinear SCM whose Markov blanket is deliberately not its cause set.

    The graph::

        z ---> x_conf1        z is observed, and is itself a cause of y
        z ---> x_conf2
        z --------------\\
        x_cause1 --------+--> y --> x_effect
        x_cause2 -------/             ^
        x_cause1 --------------------/
        x_noise1..3                   (disconnected)

    ``y`` is Bernoulli with probability ``sigmoid(g(x_cause1, x_cause2, z))``,
    where ``g`` is nonlinear: ``x_cause2`` enters through its square and through
    an interaction with ``x_cause1``, and never on its own.  The interaction is
    linear in ``x_cause2``, but ``x_cause1`` is mean zero, so it averages away
    marginally and leaves the linear correlation with ``y`` close to zero --
    around 0.48 as an AUC -- while the mutual information is the largest of any
    cause.  Any method that hunts for monotone association will therefore miss a
    genuine cause, which is the second trap in the table.

    Why ``x_effect`` is the whole point
    -----------------------------------
    ``x_effect = 1.5*y + 0.5*sin(3*x_cause1) + noise`` is generated *from* ``y``.
    The arrow runs target -> feature, so ``x_effect`` is a child, not a parent:
    intervening on it cannot move ``y``, and its causal effect on the target is
    exactly zero.  Deleting it from a policy decision costs nothing.

    It is nonetheless *in* the Markov blanket, because the blanket is parents,
    children, and co-parents of children -- children are in it precisely because
    a noisy readout of the target is informative about the target.  Conditioning
    on ``x_effect`` genuinely reduces uncertainty about ``y``, so it is not
    "spurious" in any statistical sense; it is real information flowing the wrong
    way down the arrow.

    A predictive method is therefore *expected* to rank it first, and does.  The
    coefficient on ``y`` is 1.5 while the residual spread of ``x_effect`` is
    roughly 0.46, so the two classes are separated by more than three standard
    deviations in that single column.  Every true cause, by contrast, reaches
    ``y`` through a saturating sigmoid and a Bernoulli draw, and so carries much
    less information about the realised label than the label's own noisy copy
    does.  Gates, permutation importance, SHAP and mutual information will all
    put ``x_effect`` at the top.  That is not an estimator bug -- they answer the
    question they were asked, which is not the causal one.  This is the leakage
    failure mode in miniature: a quantity recorded *after* the outcome tops the
    importance table and is useless for intervention.

    ``x_conf1`` and ``x_conf2`` are the mirror-image trap.  Both are driven by
    ``z``, which is a cause of ``y``, so both are marginally predictive; neither
    is a cause and neither is in the blanket, because ``z`` itself is observed
    and screens them off.  They belong to none of the three masks, and a method
    that ranks them highly while ``z`` is available is reporting marginal
    association rather than conditional relevance.  They are also strongly
    associated with *each other* with no arrow between them, so a pairwise test
    run on that pair alone has a common cause to find and no edge.

    Args:
        n: Number of samples.
        noise: Standard deviation of the additive Gaussian noise on the
            structural equations.  ``0`` makes every non-root variable a
            deterministic function of its parents.
        seed: Seed for :func:`numpy.random.default_rng`.
    """
    rng = np.random.default_rng(seed)

    def eps() -> np.ndarray:
        return noise * rng.standard_normal(n)

    z = rng.standard_normal(n)
    # Both proxies are non-monotone or non-linear functions of z, so a method
    # cannot dismiss them by looking only at correlations. x_conf2 is centred
    # because E[z**2] = 1.
    x_conf1 = np.tanh(2.0 * z) + eps()
    x_conf2 = z**2 - 1.0 + eps()

    x_cause1 = rng.standard_normal(n)
    x_cause2 = rng.uniform(-2.0, 2.0, size=n)

    # 4/3 = E[U(-2,2)**2] and E[z**2] = 1; subtracting them keeps g centred and
    # the classes roughly balanced, which matters because a 90/10 split would let
    # a useless model score well and muddy any comparison built on this data.
    #
    # z enters through an odd term and an even term deliberately. With tanh(z)
    # alone, x_conf2 = z**2 - 1 would carry almost no marginal information about
    # y -- the even proxy of an odd signal averages out -- and the confounded
    # category would be indistinguishable from the noise columns. The quadratic
    # term gives each proxy something to track and makes both of them live traps.
    g = (
        1.5 * x_cause1
        + 1.2 * (x_cause2**2 - 4.0 / 3.0)
        + 0.6 * x_cause1 * x_cause2
        + 1.6 * np.tanh(z)
        + 0.6 * (z**2 - 1.0)
    )
    y = (rng.random(n) < _sigmoid(g)).astype(np.int64)

    # The sin term makes x_cause1 a co-parent of x_effect, so x_cause1 sits in
    # the blanket twice over: as a parent of y and as a spouse. Removing it would
    # not change the blanket, but it does mean the blanket cannot be read off the
    # marginal association of x_effect with y alone.
    x_effect = 1.5 * y + 0.5 * np.sin(3.0 * x_cause1) + eps()

    columns = {
        "z": z,
        "x_conf1": x_conf1,
        "x_conf2": x_conf2,
        "x_cause1": x_cause1,
        "x_cause2": x_cause2,
        "x_effect": x_effect,
        "x_noise1": rng.standard_normal(n),
        "x_noise2": rng.standard_normal(n),
        "x_noise3": rng.standard_normal(n),
    }
    feature_names = list(columns)
    x = np.column_stack([columns[name] for name in feature_names])

    def mask(*names: str) -> np.ndarray:
        chosen = set(names)
        return np.array([name in chosen for name in feature_names], dtype=bool)

    direct_causes = mask("z", "x_cause1", "x_cause2")
    effects = mask("x_effect")

    return SyntheticDataset(
        x=x,
        y=y,
        feature_names=feature_names,
        direct_causes=direct_causes,
        # parents | children | co-parents. The only co-parent of x_effect is
        # x_cause1, which is already a parent, so the union adds nothing new.
        markov_blanket=direct_causes | effects,
        irrelevant=mask("x_noise1", "x_noise2", "x_noise3"),
        confounded=mask("x_conf1", "x_conf2"),
        effects=effects,
    )


def coupled_logistic(
    n: int = 3000,
    r_x: float = 3.8,
    r_y: float = 3.5,
    coupling_x_to_y: float = 0.32,
    coupling_y_to_x: float = 0.0,
    burn_in: int = 500,
    seed: int = 0,
) -> SystemData:
    """Two logistic maps with adjustable one- or two-way coupling.

    The system of Sugihara et al. (2012), the standard convergent-cross-mapping
    testbed::

        x[t+1] = x[t] * (r_x - r_x*x[t] - coupling_y_to_x * y[t])
        y[t+1] = y[t] * (r_y - r_y*y[t] - coupling_x_to_y * x[t])

    Note the direction convention: ``coupling_x_to_y`` multiplies ``x`` inside
    ``y``'s equation, so it is the strength with which ``x`` drives ``y``.  The
    defaults give a purely unidirectional system, ``x -> y``, with ``x`` an
    autonomous chaotic map.

    The expected CCM signature is the reverse of most people's first guess.  If
    ``x`` drives ``y`` then ``y``'s trajectory carries the imprint of ``x``, so
    ``x`` is recoverable from a delay embedding of ``y`` and the cross-map skill
    ``rho(x | M_y)`` is high, while ``rho(y | M_x)`` stays low.  Reading the
    arrow off the higher skill without reversing it is the single most common way
    to get a CCM result backwards.

    Coupling strength has to stay moderate.  Too weak and the imprint is below
    the noise floor; strong enough for generalised synchrony and the response
    becomes a function of the driver, at which point cross mapping succeeds in
    both directions and the asymmetry that identifies the arrow is gone.  The
    default 0.32 sits between the two.

    A property of the default parameters worth knowing before using them: the
    response's own growth rate, ``r_y=3.5``, is inside the period-4 window, so
    with the coupling switched off ``y`` settles onto a four-point cycle.  It is
    aperiodic in the returned series only because the driver keeps knocking it
    off that cycle.  Setting ``coupling_x_to_y=0`` therefore does not give a
    matched negative control, it gives a nearly deterministic one -- use
    :func:`independent_logistic`, which moves ``r_y`` into the chaotic regime.

    The same fact makes the obvious way of building a ``y->x`` case wrong.
    Passing ``coupling_x_to_y=0, coupling_y_to_x=0.32`` and leaving the growth
    rates alone puts ``y`` in the role of driver, and that driver is the
    four-point cycle; it then entrains ``x`` onto a cycle of its own, so *both*
    returned series take four distinct values and the trajectory carries no
    information for any method to find.  The label is still honest -- the
    coupling really does run ``y->x`` -- but the data cannot support the
    inference, and every causality test trivially "passes".  Swap the growth
    rates instead, ``r_x=3.5, r_y=3.8``, which makes the driver the chaotic map
    and mirrors the default case exactly.

    Raises:
        ValueError: if the trajectory leaves ``[0, 1]``.  The map is only
            invariant on the unit interval for moderate ``r`` and coupling;
            outside that region it runs away to ``-inf`` within a few steps, and
            silently clipping would fabricate dynamics that were never simulated.
    """
    rng = np.random.default_rng(seed)
    total = burn_in + n

    x = np.empty(total, dtype=np.float64)
    y = np.empty(total, dtype=np.float64)
    # Starting well inside the interval keeps the very first step from ejecting
    # the state: x0 near 1 with a large coupling makes y1 negative, and the map
    # never recovers.
    x[0], y[0] = rng.uniform(0.1, 0.9, size=2)

    # Overflow is an expected outcome for extreme parameters, not something to
    # warn about; it is reported below as a ValueError instead.
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(total - 1):
            x[t + 1] = x[t] * (r_x - r_x * x[t] - coupling_y_to_x * y[t])
            y[t + 1] = y[t] * (r_y - r_y * y[t] - coupling_x_to_y * x[t])

    for name, series in (("x", x), ("y", y)):
        if not np.isfinite(series).all() or series.min() < 0.0 or series.max() > 1.0:
            raise ValueError(
                f"coupled_logistic diverged: series {name!r} left [0, 1] with "
                f"r_x={r_x}, r_y={r_y}, coupling_x_to_y={coupling_x_to_y}, "
                f"coupling_y_to_x={coupling_y_to_x}. Reduce the growth rates or "
                f"the coupling."
            )

    return {
        "system": "coupled_logistic",
        "x": x[burn_in:],
        "y": y[burn_in:],
        "r_x": r_x,
        "r_y": r_y,
        "coupling_x_to_y": coupling_x_to_y,
        "coupling_y_to_x": coupling_y_to_x,
        "true_direction": _direction(coupling_x_to_y, coupling_y_to_x),
        "burn_in": burn_in,
        "seed": seed,
    }


def independent_logistic(n: int = 3000, seed: int = 0) -> SystemData:
    """Two uncoupled chaotic logistic maps: the negative control.

    Both series are chaotic and heavily structured, so anything that mistakes
    "complicated" for "connected" fails here.  Different growth rates make the
    trajectories dynamically distinct as well as independent, which rules out the
    weaker failure of two identical systems being called coupled because their
    attractors match.

    Any method that reports a direction on this data is producing a false
    positive, so this is the run that calibrates a significance threshold.

    Note the growth rate: this is *not* ``coupled_logistic`` with the coupling
    set to zero.  That system's ``r_y=3.5`` sits in the period-4 window, so an
    uncoupled response collapses onto a four-point cycle taking four distinct
    values forever -- a control with almost no entropy, which any test passes.
    ``r_y=3.7`` is past the accumulation point and genuinely aperiodic.
    """
    data = coupled_logistic(
        n=n, r_x=3.8, r_y=3.7, coupling_x_to_y=0.0, coupling_y_to_x=0.0, seed=seed
    )
    data["system"] = "independent_logistic"
    return data


# Time units of transient discarded before sampling. The Rossler here is sped up
# by a factor of six, so 50 units is tens of revolutions on each attractor --
# long enough that the initial condition is forgotten.
ROSSLER_LORENZ_BURN_IN = 50.0


def _rossler_lorenz_derivatives(state: np.ndarray, coupling: float) -> np.ndarray:
    """Right-hand side of the coupled system, in the Quiroga et al. (2000) form."""
    x1, x2, x3, y1, y2, y3 = state
    return np.array(
        [
            -6.0 * (x2 + x3),
            6.0 * (x1 + 0.2 * x2),
            6.0 * (0.2 + x3 * (x1 - 5.7)),
            10.0 * (y2 - y1),
            # The only place the two systems touch: the Rossler's second
            # component forces the Lorenz, and nothing goes back.
            28.0 * y1 - y2 - y1 * y3 + coupling * x2**2,
            y1 * y2 - (8.0 / 3.0) * y3,
        ]
    )


def rossler_lorenz(
    n: int = 3000, coupling: float = 2.0, dt: float = 0.01, seed: int = 0
) -> SystemData:
    """Unidirectionally coupled Rossler -> Lorenz, integrated with RK4.

    The standard continuous-time CCM benchmark (Quiroga, Arnhold and Grassberger,
    2000).  A Rossler system running six times faster than usual drives a Lorenz
    system through ``coupling * x2**2`` added to the Lorenz's second equation;
    the Rossler is autonomous, so the ground-truth direction is always ``x->y``
    regardless of ``coupling``.  Setting ``coupling=0`` gives two independent
    chaotic attractors and is the continuous-time negative control.

    It is a harder test than :func:`coupled_logistic` because the two systems
    have different dimensions, different timescales and different attractor
    geometry, so a method cannot succeed by exploiting a shared functional form.

    The observed scalars are the components the coupling actually runs through:
    ``x`` is the Rossler's second component and ``y`` the Lorenz's.  Full
    three-dimensional trajectories are returned as well, for methods that need
    the true state rather than a delay embedding.

    A trap worth stating: ``dt=0.01`` oversamples both attractors badly, and
    consecutive samples are almost identical.  Nearest neighbours in a delay
    embedding are then *temporal* neighbours, and cross mapping scores high in
    both directions on autocorrelation alone.  Either subsample the returned
    series or use a Theiler exclusion window before believing any skill value.

    Raises:
        ValueError: if the integration produces non-finite values, which means
            ``dt`` is too large for the chosen coupling.
    """
    rng = np.random.default_rng(seed)
    burn_steps = int(round(ROSSLER_LORENZ_BURN_IN / dt))

    # Perturbed rather than fixed so repeated seeds explore different parts of
    # the attractor; the perturbation is small enough to stay in the basin.
    state = np.array([0.0, 0.0, 0.4, 0.0, 1.0, 1.05]) + 0.01 * rng.standard_normal(6)

    trajectory = np.empty((n, 6), dtype=np.float64)
    # Blow-up is a legitimate outcome for a too-large dt and is reported below as
    # a ValueError; the intermediate overflow warnings would only be noise.
    with np.errstate(over="ignore", invalid="ignore"):
        for step in range(burn_steps + n):
            k1 = _rossler_lorenz_derivatives(state, coupling)
            k2 = _rossler_lorenz_derivatives(state + 0.5 * dt * k1, coupling)
            k3 = _rossler_lorenz_derivatives(state + 0.5 * dt * k2, coupling)
            k4 = _rossler_lorenz_derivatives(state + dt * k3, coupling)
            state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if step >= burn_steps:
                trajectory[step - burn_steps] = state

    if not np.isfinite(trajectory).all():
        raise ValueError(
            f"rossler_lorenz integration diverged with coupling={coupling}, dt={dt}; "
            f"reduce dt"
        )

    return {
        "system": "rossler_lorenz",
        "x": trajectory[:, 1],
        "y": trajectory[:, 4],
        "rossler": trajectory[:, :3],
        "lorenz": trajectory[:, 3:],
        "coupling_x_to_y": coupling,
        "coupling_y_to_x": 0.0,
        "true_direction": DIRECTION_X_TO_Y,
        "dt": dt,
        "burn_in": burn_steps,
        "seed": seed,
    }


def parity_redundancy(
    n: int = 3000, n_bits: int = 8, k: int = 4, seed: int = 0
) -> SystemData:
    """Redundancy where the cheap route is a *sibling* of the target, not a child.

    ``k`` of ``n_bits`` independent bits determine the target by parity::

        y       = (b_0 + ... + b_{k-1}) mod 2
        summary = prod_i (1 - 2 b_i)          == (-1)^y

    ``summary`` is a deterministic function of the same bits that generate ``y``,
    so it is a sibling: caused by the parents of ``y``, not a cause of it and not
    an effect of it.  Conditional on the parity bits, ``y`` and ``summary`` are
    independent, so ``summary`` is *outside* the Markov blanket -- and yet it
    alone determines the target, which is the deterministic degeneracy that makes
    blanket-based reasoning insufficient here.

    The reason this family is worth having alongside ``redundancy_demo`` is the
    size of the computational gap at zero information gap.  Parity over ``k``
    inputs is the standard hard case for a small network: every bit must be
    combined before anything is predictable, and no subset is partially
    informative.  Reading ``summary`` is one threshold.  Shannon says the two
    routes are equivalent given the bits; the cost of taking them differs by as
    much as this construction can arrange.

    Bits ``k`` onward are irrelevant by construction.
    """
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n, n_bits))
    parity = bits[:, :k].sum(axis=1) % 2
    summary = np.prod(1 - 2 * bits[:, :k], axis=1).astype(np.float64)

    names = [f"bit_{i}" for i in range(n_bits)] + ["summary"]
    return {
        "system": "parity_redundancy",
        "x": np.column_stack([bits.astype(np.float64), summary]),
        "y": parity.astype(np.float64),
        "feature_names": names,
        "causes": tuple(f"bit_{i}" for i in range(k)),
        "sibling": ("summary",),
        "irrelevant": tuple(f"bit_{i}" for i in range(k, n_bits)),
        # Either every parity bit, or the single summary column.
        "minimal_sufficient_sets": (
            tuple(f"bit_{i}" for i in range(k)),
            ("summary",),
        ),
        "seed": seed,
    }


def manifold_redundancy(
    n: int = 3000, n_sensors: int = 6, noise: float = 0.0, seed: int = 0
) -> SystemData:
    """Many nonlinear readouts of one low-dimensional latent state.

    The shape most real redundancy takes: a two-dimensional latent ``(s1, s2)``
    observed through ``n_sensors`` distinct nonlinear projections::

        sensor_i = tanh(cos(theta_i) * s1 + sin(theta_i) * s2)
        y        = 1[s1 > 0]

    ``tanh`` is invertible, so any two sensors at different angles recover the
    latent and therefore every other sensor: each column is redundant given two
    others, and no column is redundant given only one.  This is co-expressed
    genes, metabolites in a pathway, or neighbouring voxels -- many windows onto
    one state.

    The first sensor sits at ``theta = 0``, so it reads ``s1`` directly and is
    individually sufficient for the target, while the rest need a partner.  That
    asymmetry is deliberate: it reproduces the situation where one assay happens
    to align with the clinical decision rule and therefore looks best, without
    being any closer to the biology than its neighbours.

    None of the sensors is a cause.  The cause is ``s1``, which is latent, so
    the honest ground truth for a causal question is that no observed column
    qualifies -- and any method that names one is answering a different question.
    """
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n, 2))

    # theta_0 = 0 aligns the first sensor with the target coordinate; the rest
    # are spread to stay pairwise independent.
    angles = np.linspace(0.0, np.pi * 0.8, n_sensors)
    sensors = np.tanh(latent @ np.vstack([np.cos(angles), np.sin(angles)]))
    if noise:
        sensors = sensors + noise * rng.standard_normal(sensors.shape)

    names = [f"sensor_{i}" for i in range(n_sensors)] + ["unrelated"]
    x = np.column_stack([sensors, rng.standard_normal(n)])

    return {
        "system": "manifold_redundancy",
        "x": x,
        "y": (latent[:, 0] > 0).astype(np.float64),
        "feature_names": names,
        "latent": latent,
        "causes": (),  # the cause is s1, which is not observed
        "aligned_sensor": ("sensor_0",),
        "irrelevant": ("unrelated",),
        "minimal_sufficient_sets": (("sensor_0",),),
        "seed": seed,
    }


def redundancy_demo(n: int = 3000, seed: int = 0) -> SystemData:
    """A genuine cause whose leave-one-out importance is exactly zero.

    A chaotic logistic map ``u`` drives everything::

        u[t+1] = 3.9 * u[t] * (1 - u[t])
        y[t]   = u[t+1]                     the target, a deterministic effect
        driver    = u[t]                    the genuine, unique cause
        proxy_cos = cos(2*pi*u[t])
        proxy_sin = sin(2*pi*u[t])
        unrelated = a second, independent logistic map

    ``u`` stays inside ``(0, 1)``, so ``atan2(proxy_sin, proxy_cos)/(2*pi)``
    returns it exactly.  Neither proxy alone identifies ``u`` -- each is
    two-to-one on the unit interval -- but the pair does, to machine precision.

    Identifying ``u`` is more than ``y`` needs, though, and ``proxy_cos`` alone
    is already sufficient for the target.  The logistic map is symmetric about
    ``u = 0.5`` -- ``y(u) = y(1-u)`` -- and ``cos(2*pi*u)`` is symmetric about
    the same point, so the cosine's two-to-one ambiguity is exactly the ambiguity
    ``y`` is blind to: ``y = 3.9*a*(1-a)`` with ``a = arccos(proxy_cos)/(2*pi)``
    recovers the target to machine precision whichever branch ``u`` was on.
    ``proxy_sin`` gets no such reprieve, because its ambiguity pairs ``u`` with
    ``0.5-u``, which the map does not identify.  The asymmetry between the two
    proxies is a real property of this system and not an implementation detail;
    it is why ``minimal_sufficient_sets`` lists ``proxy_cos`` on its own rather
    than the pair.

    Why removal-based importance fails here
    ---------------------------------------
    LOCO and ablation ask: how much worse is the *refit* model when this column
    is deleted?  Delete ``driver`` and the answer is nothing at all.  The refit
    model reconstructs ``y`` from the proxies and reproduces it exactly, so the
    measured importance of the system's only cause is zero.  Delete either proxy
    and the answer is also nothing, because ``driver`` is still there.  No single
    deletion changes the loss; the smallest deletion that does is the pair
    ``{driver, proxy_cos}``, since a set determines ``y`` exactly when it
    contains one of them.

    Permutation importance does not rescue the situation, it just makes the
    result arbitrary: whether ``driver`` scores high or zero depends on which of
    the equivalent representations the fit happened to latch onto, which is a
    property of the optimiser, not of the system.

    This is not small-sample noise and does not go away with more data.  It is a
    property of the joint distribution: the information is duplicated, so no
    method whose definition of importance is "the loss increase from removing
    it" can attribute anything to a variable that has a perfect substitute.
    Deterministic nonlinear systems are full of such substitutes, which is the
    argument for state-space methods -- CCM asks whether one variable's attractor
    can reconstruct another, a question redundant coordinates do not affect.

    Returns the design matrix, the target, the ground-truth driver by name and
    index, and ``minimal_sufficient_sets``: the inclusion-minimal column subsets
    that determine ``y``.  Importance is only well defined relative to one of
    those, which is the honest way to report it.  ``recoverable_from`` answers
    the different question of what it takes to reconstruct the *driver*, which
    genuinely needs both proxies.
    """
    rng = np.random.default_rng(seed)
    r = 3.9
    burn_in = 200

    def logistic(rate: float, start: float, length: int) -> np.ndarray:
        series = np.empty(length, dtype=np.float64)
        series[0] = start
        for t in range(length - 1):
            series[t + 1] = rate * series[t] * (1.0 - series[t])
        return series

    starts = rng.uniform(0.1, 0.9, size=2)
    # One extra step so y can be the successor of the last retained u.
    u = logistic(r, starts[0], burn_in + n + 1)[burn_in:]
    # The distractor is another chaotic map rather than white noise, so it cannot
    # be dismissed on the grounds of looking unstructured.
    unrelated = logistic(3.7, starts[1], burn_in + n)[burn_in:]

    driver = u[:-1]
    target = u[1:]

    feature_names = ["driver", "proxy_cos", "proxy_sin", "unrelated"]
    x = np.column_stack(
        [
            driver,
            np.cos(2.0 * np.pi * driver),
            np.sin(2.0 * np.pi * driver),
            unrelated,
        ]
    )

    return {
        "system": "redundancy_demo",
        "x": x,
        "y": target,
        "feature_names": feature_names,
        "driver": "driver",
        "driver_index": 0,
        "recoverable_from": ("proxy_cos", "proxy_sin"),
        # Inclusion-minimal, so ("proxy_cos", "proxy_sin") does not belong here:
        # it determines y but so does its proper subset ("proxy_cos",), because
        # the cosine's two-to-one ambiguity coincides with the logistic map's
        # own symmetry about u = 0.5. Listing the pair would overstate what a
        # method has to find and would score a correct answer wrong.
        "minimal_sufficient_sets": (("driver",), ("proxy_cos",)),
        "irrelevant": ("unrelated",),
        "seed": seed,
    }
