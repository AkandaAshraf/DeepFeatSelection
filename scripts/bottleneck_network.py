"""Direct versus indirect links: where pairwise cross mapping structurally fails.

PHASE 2 of the conditional-retrieval idea.  Phase 1 asked whether a bottleneck
can see a coupling at all; this asks whether it can tell a *direct* coupling
from a *mediated* one, which is the question pairwise CCM cannot represent.

WHY THE PAIRWISE FAILURE IS STRUCTURAL, NOT A TUNING PROBLEM
------------------------------------------------------------
In a chain ``A -> B -> C`` with no direct ``A -> C`` link, C's trajectory is
shaped by B, and B's is shaped by A, so C's attractor carries A's imprint
transitively.  Cross mapping A from C's manifold therefore succeeds, and CCM
reports ``A -> C``.  Nothing inside the pairwise formalism can subtract the
route through B, because B is not in the calculation.  The same holds for a
fork ``B <- A -> C``: B and C share a driver, each carries the other's imprint
through A, and the pair looks coupled from either side.  Partial cross mapping
(Leng et al., 2020) is the published repair.  The question here is whether the
compressibility principle gets the same discrimination for free.

THE TEST, AND WHAT IT PREDICTS
------------------------------
For a target V, retrieval of V's next state is attempted from bottlenecked
delay embeddings of four variable sets: ``{V}``, ``{V,U}``, ``{V,W}``,
``{V,U,W}``.  The prediction fixed before any number was looked at:

* a DIRECT parent improves retrieval beyond *every* set that excludes it -- both
  over ``{V}`` and over ``{V,W}``;
* an INDIRECT ancestor improves over ``{V}`` but adds NOTHING over a set that
  already contains the mediator, because one step of the dynamics is Markov in
  the true parents: ``C(t+1) = C(t)(r_C - r_C C(t) - c B(t))`` contains A only
  through B(t), so conditioning on B(t) screens A off exactly.

That screening is why the horizon is ONE step and not more.  At two steps ahead
C depends on B(t+1), which depends on A(t), and A would then genuinely add
information beyond {C,B} -- the mediator would stop screening and the signature
would be destroyed by the design rather than by the method.

STATISTICS FIXED A PRIORI (protocol rule 3; nothing below was chosen after
looking at a result)
--------------------------------------------------------------------------
* Growth rates 3.8 / 3.7 / 3.6 for A / B / C, coupling 0.16, burn-in 500, n=2000,
  3 seeds.  The coupling functional form and the burn-in convention are copied
  from ``coupled_logistic``; the growth rates extend the protocol reference's
  chaotic-band choice (3.8 / 3.7) with 3.6 for the third series, which is past
  the accumulation point at 3.5699 and is checked for aperiodicity below rather
  than assumed.
* Zero coupling must reproduce the control byte-for-byte: ``chain_logistic``,
  ``fork_logistic`` and ``independent_triple`` at coupling 0 are the same call
  into one core, and :func:`check_zero_coupling_is_control` asserts the three
  series are identical before any compute is spent.  This is the analogue of the
  protocol reference's check, and it exists because a run whose zero-coupling arm
  is a different dynamical system has no control.
* Embedding E=3, tau=1 for every series -- the protocol reference's fixed
  embedding, so the retrieval test and the headline CCM run at a matched
  embedding and the comparison is not confounded by the choice.  CCM is also run
  at ``optimal_embedding_dimension`` per series (the larger of the two, as in the
  reference) and reported as a robustness column.
* Horizon 1.  The target is the part of V's next embedded state that is not
  already in the input, which at horizon 1 is the scalar ``V(t+1)``; carrying the
  overlapping coordinates would add ``E-1`` dimensions of identity map to every
  arm equally and divide every gain by E.
* A SET IS A MASK, NOT A SLICE.  The input is always the full ``E*3 = 9``
  column joint embedding and the bottleneck is always E=3; a variable outside
  the set has its three columns set to zero, which after train-set
  standardisation is its own mean.  This is the ablation convention already used
  by ``sequence_causal.deprivation_score`` (``ablated[:, :, source] = 0``), and
  the smoke run is why it is used here rather than slicing: with slicing, the
  ``{V}`` arm is a 3-column problem and the ``{V,U,W}`` arm a 9-column one, so a
  gain confounds "U carries information" with "the operator now has six more
  dimensions to cope with".  That confound is not small.  Measured on the smoke
  run (chain, n=400, one seed, kept at
  ``ExpOutput/bottleneck_network/smoke_sliced_design/``), adding A to the
  retrieval of B -- where A's contribution to B's next state is real but tiny --
  moved the kNN error by a factor of 103, i.e. a gain of -102.0, entirely from
  dimension; the sliced design also put the true edge B->C at -0.60.  Under masking
  both arms present the same 9 columns to the same architecture with the same
  compression ratio, and for kNN a zeroed column contributes exactly zero to
  every pairwise distance, so the arms differ only in the information present.
* Bottleneck width = E = 3, the dimension of the state being retrieved, against a
  9-column input: every arm compresses by the same factor of three, so an added
  variable has to earn its place through the compression -- which is the
  principle under test -- and no arm is given more capacity than another.
* Retrieval statistic: relative mean-squared-error reduction,
  ``gain(S+U over S) = (mse(S) - mse(S+U)) / mse(S)``.  This is the protocol
  reference's deprivation score, reused rather than reinvented; it is
  dimensionless and comparable across arms of different difficulty.
    - marginal gain of U for target V:      gain of ``{V,U}``   over ``{V}``
    - conditional gain of U given W:        gain of ``{V,U,W}`` over ``{V,W}``
* "~ zero" is calibrated by the control, not by an epsilon: ``NULL(operator)`` is
  the MAXIMUM gain of either kind observed anywhere on ``independent_triple``,
  over all seeds, targets and sources.
* Edge verdict for the ordered pair ``U -> V`` (per system, per operator):
    - DIRECT   if  min over seeds of marginal gain > NULL  AND
                   min over seeds of conditional gain > NULL
    - INDIRECT if  min over seeds of marginal gain > NULL, conditional gain not
    - NONE     otherwise
  Taking the minimum over seeds and comparing it against the control's maximum is
  protocol rule 5 -- the weakest coupled arm must clear the strongest control
  arm, not the means.
* CCM edge verdict for ``U -> V``: ``DirectionResult.is_convergent()`` at the
  package defaults (min_rho 0.3, min_delta 0.05, alpha 0.05) in EVERY seed.  The
  same weakest-arm rule, and the package's own published criterion rather than a
  threshold invented here.  Orientation is read from the field named for the
  hypothesis, never from the cross-map label; see ``deepfeatselect.ccm``.
* Edge-recovery scoring runs over the chain and fork only: 12 ordered pairs, 4 of
  them direct edges (positives), 8 negatives of which 3 are the structural traps
  (chain A->C mediated, fork B->C and C->B confounded).  The control's own pairs
  are reported separately and are NOT scored, because the retrieval null is
  defined as the maximum of the control's own gains, so its verdicts are
  all-negative by construction and would inflate specificity for free.

GUARDS (protocol rule 4)
------------------------
* Learnability, per target: retrieval of V from ``{V}`` alone must beat the
  held-out mean by ``MIN_RETRIEVAL_R2``.  If it does not, that arm is
  unlearnable, its gains are a difference of two noise levels, and every verdict
  for that target is flagged and excluded from the score.
* Bottleneck pressure, per target: the same MLP with the narrow layer removed
  (full width throughout) is fitted on the full set and must reach
  ``MIN_FULLWIDTH_R2``.  A set that cannot be learned even without compression
  cannot support a statement about what compression cost it.

SPLITS (protocol rule 2)
------------------------
Three disjoint contiguous segments -- train / early-stop / test -- with an
embargo of ``EMBARGO`` rows at each seam.  A row at time t reads
``[t-(E-1)tau, t]`` and predicts ``t+horizon``, spanning four timesteps at the
settings above, so a ten-row embargo removes every shared timestep with margin.
Standardisation uses train-segment statistics only.  kNN does not use the
early-stop segment at all; it is left unused rather than folded into training so
that both operators are scored on identical test rows from identical training
rows.

CAPACITY.  The phase-1 script is not in this repository at the time of writing,
so the MLP's capacity is declared here rather than imported: one 64-unit ReLU
layer on each side of a linear bottleneck, Adam at 1e-3.  ``--units`` is exposed
so a later run can match phase 1 exactly if it differs.

DEVIATION WORTH STATING.  kNN has no bottleneck: it consumes the same masked
9-column state directly.  Only the MLP arm implements the compression literally,
so "does the bottleneck matter" is answered by the difference between the two
operators rather than by the kNN arm alone.

    python scripts/bottleneck_network.py --smoke
    python scripts/bottleneck_network.py
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

from deepfeatselect.ccm import ccm, optimal_embedding_dimension, time_delay_embed

# --- generator constants, extending the protocol reference -------------------
R_A, R_B, R_C = 3.8, 3.7, 3.6
BURN_IN = 500

# --- analysis constants, all fixed before any result was inspected -----------
EMBED_E = 3
EMBED_TAU = 1
HORIZON = 1
BOTTLENECK = EMBED_E
EMBARGO = 10
TRAIN_FRACTION = 0.60
VAL_FRACTION = 0.15
KNN_NEIGHBOURS = EMBED_E + 1  # the simplex convention in deepfeatselect.ccm

# Below this held-out R^2 the {V}-alone retrieval did not beat the mean and every
# gain built on it is a difference of two noise levels.  The permissive floor is
# the protocol reference's MIN_FORECAST_R2.
MIN_RETRIEVAL_R2 = 0.10
# Rule 4's "near-perfectly" for the no-bottleneck control.
MIN_FULLWIDTH_R2 = 0.90
# An uncoupled series that collapses onto a short cycle is a control with no
# entropy; the protocol reference measured exactly that trap at r=3.5 (4 unique
# values in 2000 samples).  Checked, not assumed.
MIN_UNIQUE_VALUES = 100

NAMES = ("A", "B", "C")
TRUE_EDGES: dict[str, frozenset[tuple[str, str]]] = {
    "chain": frozenset({("A", "B"), ("B", "C")}),
    "fork": frozenset({("A", "B"), ("A", "C")}),
    "independent": frozenset(),
}
# The pairs the motivation turns on, labelled a priori.
STRUCTURAL_TRAPS = {
    ("chain", "A", "C"): "indirect (mediated by B)",
    ("fork", "B", "C"): "spurious (common cause A)",
    ("fork", "C", "B"): "spurious (common cause A)",
}


# =============================================================================
# Generators
# =============================================================================
def _triple_logistic(
    n: int,
    a_to_b: float,
    b_to_c: float,
    a_to_c: float,
    system: str,
    burn_in: int = BURN_IN,
    seed: int = 0,
) -> dict[str, np.ndarray | float | str | int]:
    """Three logistic maps wired by the ``coupled_logistic`` coupling form.

    Every coupling enters exactly as it does in :func:`coupled_logistic` -- the
    driver's current value multiplies the response's own value inside the
    response's growth term::

        A[t+1] = A[t] * (r_A - r_A*A[t])
        B[t+1] = B[t] * (r_B - r_B*B[t] - a_to_b * A[t])
        C[t+1] = C[t] * (r_C - r_C*C[t] - b_to_c * B[t] - a_to_c * A[t])

    A is autonomous in every wiring, which makes it the internal control: no set
    of the other two can improve retrieval of A's next state, in any system.

    All three wirings share this one body and one ``default_rng`` call sequence,
    so setting every coupling to zero yields byte-identical series whichever
    wiring asked for it.  That identity is what makes ``independent_triple`` a
    control for the other two rather than merely a similar system, and it is
    asserted rather than trusted -- see :func:`check_zero_coupling_is_control`.

    Raises:
        ValueError: if any series leaves ``[0, 1]``, matching
            :func:`coupled_logistic`.  Clipping would fabricate dynamics that
            were never simulated.
    """
    rng = np.random.default_rng(seed)
    total = burn_in + n
    a = np.empty(total, dtype=np.float64)
    b = np.empty(total, dtype=np.float64)
    c = np.empty(total, dtype=np.float64)
    # Same starting box as coupled_logistic: well inside the interval, so the
    # first step cannot eject the state.
    a[0], b[0], c[0] = rng.uniform(0.1, 0.9, size=3)

    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(total - 1):
            a[t + 1] = a[t] * (R_A - R_A * a[t])
            b[t + 1] = b[t] * (R_B - R_B * b[t] - a_to_b * a[t])
            c[t + 1] = c[t] * (R_C - R_C * c[t] - b_to_c * b[t] - a_to_c * a[t])

    for name, series in (("A", a), ("B", b), ("C", c)):
        if not np.isfinite(series).all() or series.min() < 0.0 or series.max() > 1.0:
            raise ValueError(
                f"{system} diverged: series {name!r} left [0, 1] with "
                f"a_to_b={a_to_b}, b_to_c={b_to_c}, a_to_c={a_to_c}"
            )

    return {
        "system": system,
        "A": a[burn_in:],
        "B": b[burn_in:],
        "C": c[burn_in:],
        "a_to_b": a_to_b,
        "b_to_c": b_to_c,
        "a_to_c": a_to_c,
        "seed": seed,
    }


def chain_logistic(n: int = 2000, coupling: float = 0.16, seed: int = 0) -> dict:
    """``A -> B -> C`` with no direct ``A -> C`` link.

    The system that breaks pairwise cross mapping: C's attractor carries A's
    imprint through B, so a pairwise test has an A-C dependence to find and no
    A-C edge to find it with.
    """
    return _triple_logistic(n, coupling, coupling, 0.0, "chain", seed=seed)


def fork_logistic(n: int = 2000, coupling: float = 0.16, seed: int = 0) -> dict:
    """``B <- A -> C``: one driver, two responses, no link between the responses.

    The mirror-image trap.  B and C are dependent through their common driver,
    so the pair looks coupled from either side while neither drives the other.
    """
    return _triple_logistic(n, coupling, 0.0, coupling, "fork", seed=seed)


def independent_triple(n: int = 2000, seed: int = 0) -> dict:
    """Three uncoupled chaotic logistic maps: the negative control.

    Identical to either wiring with the coupling set to zero, by construction.
    """
    return _triple_logistic(n, 0.0, 0.0, 0.0, "independent", seed=seed)


SYSTEM_BUILDERS = {
    "chain": chain_logistic,
    "fork": fork_logistic,
    "independent": lambda n, coupling, seed: independent_triple(n=n, seed=seed),
}


# =============================================================================
# Wiring checks -- run before any compute is spent
# =============================================================================
def check_zero_coupling_is_control(n: int, seeds: int) -> bool:
    """Every wiring at coupling zero must be the control, exactly.

    Reported rather than swallowed: if it fails, the systems differ by more than
    their wiring and no comparison between them means anything.
    """
    ok = True
    for seed in range(seeds):
        control = independent_triple(n=n, seed=seed)
        for builder in (chain_logistic, fork_logistic):
            zero = builder(n=n, coupling=0.0, seed=seed)
            same = all(np.array_equal(zero[k], control[k]) for k in NAMES)
            ok = ok and same
            print(
                f"  seed {seed}: {builder.__name__}(coupling=0) == "
                f"independent_triple: {same}"
            )
    return ok


def check_wiring(n: int, seed: int = 0) -> bool:
    """Setting a coupling to zero must decouple exactly the series it feeds.

    The generators are new code, so their claimed wiring is a hypothesis until
    checked.  Each assertion below compares a trajectory against the control's:
    a series that should be decoupled must be byte-identical to the control's,
    and a series that should be driven must differ from it.  "Differs" is the
    weaker claim and is the one that can fail silently, so both directions are
    checked.
    """
    control = independent_triple(n=n, seed=seed)
    checks: list[tuple[str, bool]] = []

    def same(data: dict, name: str) -> bool:
        return bool(np.array_equal(data[name], control[name]))

    full_chain = chain_logistic(n=n, coupling=0.16, seed=seed)
    full_fork = fork_logistic(n=n, coupling=0.16, seed=seed)
    chain_no_bc = _triple_logistic(n, 0.16, 0.0, 0.0, "chain", seed=seed)
    chain_no_ab = _triple_logistic(n, 0.0, 0.16, 0.0, "chain", seed=seed)
    fork_no_ac = _triple_logistic(n, 0.16, 0.0, 0.0, "fork", seed=seed)

    checks += [
        ("A is autonomous in the chain", same(full_chain, "A")),
        ("A is autonomous in the fork", same(full_fork, "A")),
        ("chain drives B", not same(full_chain, "B")),
        ("chain drives C", not same(full_chain, "C")),
        ("fork drives B", not same(full_fork, "B")),
        ("fork drives C", not same(full_fork, "C")),
        ("b_to_c=0 decouples C", same(chain_no_bc, "C")),
        ("b_to_c=0 leaves B driven", not same(chain_no_bc, "B")),
        ("a_to_b=0 decouples B", same(chain_no_ab, "B")),
        ("a_to_b=0 leaves C driven through B", not same(chain_no_ab, "C")),
        ("fork a_to_c=0 decouples C", same(fork_no_ac, "C")),
        ("fork a_to_c=0 leaves B driven", not same(fork_no_ac, "B")),
    ]

    # A control that collapses onto a short cycle passes every causality test
    # trivially; r_C=3.6 is the growth rate this run adds, so it is the one that
    # most needs the check.
    for name in NAMES:
        unique = len(np.unique(control[name]))
        checks.append(
            (f"control {name} aperiodic ({unique} unique of {n})",
             unique >= min(MIN_UNIQUE_VALUES, n)),
        )

    for label, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")
    return all(passed for _, passed in checks)


# =============================================================================
# Design matrices, splits, operators
# =============================================================================
@dataclass(frozen=True)
class Design:
    """Aligned delay embeddings, one-step targets, and the three row segments."""

    embeddings: dict[str, np.ndarray]
    targets: dict[str, np.ndarray]
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def inputs(self, names: Sequence[str]) -> np.ndarray:
        """The full joint embedding with everything outside ``names`` zeroed.

        Masking rather than slicing keeps the input width, the architecture and
        the compression ratio identical across every set, so a gain is about
        information and not about how many columns the operator had to cope
        with.  Zero is the train-set mean after standardisation, and for a
        distance-based operator a constant column contributes nothing to any
        pairwise distance, which makes the mask exactly equivalent to a slice
        for kNN except that it does not inflate the dimension.
        """
        keep = set(names)
        return np.concatenate(
            [self.embeddings[name] if name in keep
             else np.zeros_like(self.embeddings[name])
             for name in NAMES],
            axis=1,
        )


def build_design(series: dict[str, np.ndarray]) -> Design:
    """Delay-embed every series, take one-step targets, split contiguously.

    Every series is embedded at the same E and tau and so shares one row index,
    which is what lets a set's design matrix be a plain concatenation and lets
    all four sets be scored on identical rows.

    Standardisation is per column using train-segment statistics only.  Doing it
    after the split rather than before is not cosmetic: the series are chaotic
    and their segment means differ, so whole-series scaling would leak the test
    segment's level into the training inputs.
    """
    embeddings: dict[str, np.ndarray] = {}
    times = np.empty(0, dtype=np.int64)
    for name, values in series.items():
        manifold, times = time_delay_embed(values, EMBED_E, EMBED_TAU)
        embeddings[name] = manifold

    length = len(next(iter(series.values())))
    keep = (times + HORIZON) < length
    times = times[keep]
    embeddings = {name: m[keep] for name, m in embeddings.items()}

    n_rows = len(times)
    train_end = int(TRAIN_FRACTION * n_rows)
    val_end = int((TRAIN_FRACTION + VAL_FRACTION) * n_rows)
    train = np.arange(0, train_end - EMBARGO)
    val = np.arange(train_end, val_end - EMBARGO)
    test = np.arange(val_end, n_rows)
    if len(train) < 50 or len(val) < 10 or len(test) < 20:
        raise ValueError(
            f"series too short: {n_rows} rows give train/val/test of "
            f"{len(train)}/{len(val)}/{len(test)}"
        )

    targets: dict[str, np.ndarray] = {}
    for name, values in series.items():
        manifold = embeddings[name]
        mean = manifold[train].mean(axis=0)
        std = manifold[train].std(axis=0) + 1e-12
        embeddings[name] = (manifold - mean) / std

        target = values[times + HORIZON]
        targets[name] = (target - target[train].mean()) / (target[train].std() + 1e-12)

    return Design(embeddings, targets, train, val, test)


@dataclass(frozen=True)
class Fit:
    """Held-out error with the variance it has to be judged against.

    Carrying the test-segment variance alongside the error makes "did this arm
    learn anything" arithmetic: the split is contiguous, so the test segment's
    variance is not exactly 1 even though the series were standardised.
    """

    mse: float
    target_variance: float

    @property
    def r2(self) -> float:
        return 1.0 - self.mse / (self.target_variance + 1e-12)

    @property
    def learned(self) -> bool:
        return self.r2 >= MIN_RETRIEVAL_R2


def knn_retrieval(x: np.ndarray, y: np.ndarray, design: Design) -> Fit:
    """Distance-weighted kNN regression: the cheap operator.

    Neighbour count is ``E+1``, the simplex convention already used by
    ``deepfeatselect.ccm``, and the weights decay with distance for the same
    reason they do there.  No bottleneck: this arm's penalty for a useless extra
    variable is the dilution of neighbour distances in a higher-dimensional
    input, which is a real penalty but not the same one the MLP pays.
    """
    model = KNeighborsRegressor(n_neighbors=KNN_NEIGHBOURS, weights="distance")
    model.fit(x[design.train], y[design.train])
    predicted = model.predict(x[design.test])
    return Fit(
        mse=float(np.mean((predicted - y[design.test]) ** 2)),
        target_variance=float(np.var(y[design.test])),
    )


def build_mlp(input_dim: int, bottleneck: int | None, units: int):
    """Encoder-bottleneck-decoder over a single state vector.

    ``bottleneck=None`` removes the narrow layer and leaves a plain two-hidden-
    layer network: that is the rule-4 control, and its job is to show the arm is
    learnable when nothing is being compressed away.
    """
    inputs = keras.layers.Input(shape=(input_dim,))
    h = keras.layers.Dense(units, activation="relu")(inputs)
    if bottleneck is not None:
        h = keras.layers.Dense(bottleneck, activation="linear", name="bottleneck")(h)
    h = keras.layers.Dense(units, activation="relu")(h)
    out = keras.layers.Dense(1)(h)
    model = keras.Model(inputs, out)
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-3))
    return model


def mlp_retrieval(
    x: np.ndarray, y: np.ndarray, design: Design, seed: int, args,
    bottleneck: int | None = BOTTLENECK,
) -> Fit:
    """Test-segment retrieval error from the three-segment embargoed split.

    The early-stop segment is selected on and therefore cannot also be the
    reported number; the test segment is touched by nothing but the final
    evaluation.  ``shuffle=True`` reorders batches within the training segment
    only and crosses no seam.

    The same seed is used for every set of a given (system, seed), so a gain is a
    difference between paired runs rather than between two unrelated restarts.
    """
    keras.utils.set_random_seed(seed)
    model = build_mlp(x.shape[1], bottleneck, args.units)
    stopper = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True
    )
    model.fit(
        x[design.train], y[design.train],
        validation_data=(x[design.val], y[design.val]),
        epochs=args.epochs, batch_size=args.batch, shuffle=True, verbose=0,
        callbacks=[stopper],
    )
    return Fit(
        mse=float(model.evaluate(x[design.test], y[design.test], verbose=0)),
        target_variance=float(np.var(y[design.test])),
    )


# =============================================================================
# Retrieval sweep
# =============================================================================
def variable_sets(target: str) -> list[tuple[str, ...]]:
    """The four sets scored for a target: it alone, plus each addition, plus both."""
    others = [name for name in NAMES if name != target]
    return [
        (target,),
        tuple(sorted((target, others[0]))),
        tuple(sorted((target, others[1]))),
        tuple(sorted((target, *others))),
    ]


def set_key(names: Iterable[str]) -> str:
    return "".join(sorted(names))


def retrieval_rows(design: Design, system: str, seed: int, args) -> list[dict]:
    """One row per (target, set, operator), plus the no-bottleneck guard row."""
    rows: list[dict] = []
    for target in NAMES:
        for names in variable_sets(target):
            x = design.inputs(names)
            y = design.targets[target]
            for operator, fit in (
                ("knn", knn_retrieval(x, y, design)),
                ("mlp", mlp_retrieval(x, y, design, seed, args)),
            ):
                rows.append({
                    "system": system, "seed": seed, "operator": operator,
                    "target": target, "set": set_key(names),
                    "n_columns": x.shape[1], "n_active": EMBED_E * len(names),
                    "mse": fit.mse, "r2": fit.r2,
                })
        # Rule 4: the same net without the narrow layer, on the largest set.
        full = mlp_retrieval(
            design.inputs(NAMES), design.targets[target], design, seed, args,
            bottleneck=None,
        )
        rows.append({
            "system": system, "seed": seed, "operator": "mlp_fullwidth",
            "target": target, "set": set_key(NAMES),
            "n_columns": EMBED_E * len(NAMES), "n_active": EMBED_E * len(NAMES),
            "mse": full.mse, "r2": full.r2,
        })
    return rows


def gain_rows(retrieval: pd.DataFrame) -> pd.DataFrame:
    """Marginal and conditional gains for every ordered pair, per seed.

    ``marginal`` is the gain of adding the source to the target alone;
    ``conditional`` is the gain of adding it to the target *plus the third
    series*.  A direct parent has to clear the null on both, because both sets it
    is being added to exclude it.  An ancestor whose route runs through the third
    series clears the first and not the second, which is the whole signature.
    """
    lookup = {
        (r.system, r.seed, r.operator, r.target, r.set): r.mse
        for r in retrieval.itertuples()
    }
    learnable = {
        (r.system, r.seed, r.operator, r.target): r.r2 >= MIN_RETRIEVAL_R2
        for r in retrieval.itertuples() if r.set == r.target
    }
    fullwidth = {
        (r.system, r.seed, r.target): r.r2
        for r in retrieval.itertuples() if r.operator == "mlp_fullwidth"
    }

    rows = []
    for (system, seed, operator), _ in retrieval.groupby(
        ["system", "seed", "operator"], sort=False
    ):
        if operator == "mlp_fullwidth":
            continue
        for target in NAMES:
            for source in NAMES:
                if source == target:
                    continue
                other = next(n for n in NAMES if n not in (target, source))
                base = lookup[(system, seed, operator, target, set_key([target]))]
                with_source = lookup[
                    (system, seed, operator, target, set_key([target, source]))]
                with_other = lookup[
                    (system, seed, operator, target, set_key([target, other]))]
                with_both = lookup[
                    (system, seed, operator, target, set_key(NAMES))]
                rows.append({
                    "system": system, "seed": seed, "operator": operator,
                    "target": target, "source": source, "other": other,
                    "mse_base": base, "mse_source": with_source,
                    "mse_other": with_other, "mse_both": with_both,
                    "marginal_gain": (base - with_source) / (base + 1e-30),
                    "conditional_gain": (with_other - with_both) / (with_other + 1e-30),
                    "learnable": learnable[(system, seed, operator, target)],
                    "fullwidth_r2": fullwidth[(system, seed, target)],
                })
    return pd.DataFrame(rows)


# =============================================================================
# CCM, the incumbent
# =============================================================================
def ccm_rows(series: dict[str, np.ndarray], system: str, seed: int, args) -> list[dict]:
    """Both directions of all three unordered pairs, at two embedding settings.

    The direction is read from the field named for the hypothesis
    (``x_causes_y`` / ``y_causes_x``), never from the cross-map label, which is
    the inversion the ccm module exists to prevent.
    """
    standardised = {
        name: (v - v.mean()) / (v.std() + 1e-12) for name, v in series.items()
    }
    rows = []
    for first, second in combinations(NAMES, 2):
        u, v = standardised[first], standardised[second]
        settings = {"E3": EMBED_E}
        e_opt = max(
            optimal_embedding_dimension(u, max_E=args.max_E),
            optimal_embedding_dimension(v, max_E=args.max_E),
        )
        settings["Eopt"] = e_opt
        for label, embedding in settings.items():
            start = time.time()
            result = ccm(u, v, E=embedding, seed=seed,
                         exclusion_radius=args.exclusion_radius)
            seconds = time.time() - start
            for source, target, direction in (
                (first, second, result.x_causes_y),
                (second, first, result.y_causes_x),
            ):
                rows.append({
                    "system": system, "seed": seed, "setting": label,
                    "E": embedding, "source": source, "target": target,
                    "rho": direction.rho_at_max_lib,
                    "delta_rho": direction.delta_rho,
                    "convergent": bool(direction.is_convergent()),
                    "seconds": seconds / 2.0,
                })
    return rows


# =============================================================================
# Verdicts and scoring
# =============================================================================
def retrieval_verdicts(gains: pd.DataFrame, nulls: dict[str, float]) -> pd.DataFrame:
    """Per (system, operator, ordered pair) verdict from the weakest seed.

    Protocol rule 5: the coupled arm's weakest seed has to clear the control's
    strongest observation, so the minimum over seeds is compared against the
    control's maximum.  A pair whose target was unlearnable in any seed is
    flagged and excluded rather than scored.
    """
    rows = []
    for (system, operator, source, target), block in gains.groupby(
        ["system", "operator", "source", "target"], sort=False
    ):
        null = nulls[operator]
        marginal = float(block.marginal_gain.min())
        conditional = float(block.conditional_gain.min())
        flagged = not bool(block.learnable.all())
        if marginal > null and conditional > null:
            verdict = "direct"
        elif marginal > null:
            verdict = "indirect"
        else:
            verdict = "none"
        rows.append({
            "system": system, "method": f"retrieval_{operator}",
            "source": source, "target": target,
            "marginal_min": marginal, "conditional_min": conditional,
            "null": null, "verdict": verdict, "flagged": flagged,
            "calls_edge": verdict == "direct" and not flagged,
        })
    return pd.DataFrame(rows)


def ccm_verdicts(ccm_frame: pd.DataFrame) -> pd.DataFrame:
    """An edge where every seed converges, per embedding setting."""
    rows = []
    for (system, setting, source, target), block in ccm_frame.groupby(
        ["system", "setting", "source", "target"], sort=False
    ):
        rows.append({
            "system": system, "method": f"ccm_{setting}",
            "source": source, "target": target,
            "rho_mean": float(block.rho.mean()),
            "rho_min": float(block.rho.min()),
            "n_convergent": int(block.convergent.sum()),
            "n_seeds": int(len(block)),
            "verdict": "direct" if bool(block.convergent.all()) else "none",
            "flagged": False,
            "calls_edge": bool(block.convergent.all()),
        })
    return pd.DataFrame(rows)


def score_edges(verdicts: pd.DataFrame, systems: Sequence[str]) -> pd.DataFrame:
    """Confusion counts against the generating DAG, direct edges as positives."""
    rows = []
    for method, block in verdicts.groupby("method", sort=False):
        block = block[block.system.isin(systems)]
        tp = fp = fn = tn = flagged = 0
        for row in block.itertuples():
            truth = (row.source, row.target) in TRUE_EDGES[row.system]
            if row.flagged:
                flagged += 1
                continue
            if truth and row.calls_edge:
                tp += 1
            elif truth:
                fn += 1
            elif row.calls_edge:
                fp += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if tp and np.isfinite(precision) and np.isfinite(recall) else 0.0)
        rows.append({
            "method": method, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "flagged": flagged, "precision": precision, "recall": recall, "F1": f1,
        })
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--coupling", type=float, default=0.16)
    parser.add_argument("--units", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-E", type=int, default=10)
    parser.add_argument("--exclusion-radius", type=int, default=0)
    parser.add_argument("--systems", nargs="+", default=list(SYSTEM_BUILDERS))
    parser.add_argument("--outdir", default="ExpOutput/bottleneck_network")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run that exercises every mechanism: n=400, "
                             "1 seed, chain only")
    args = parser.parse_args()

    if args.smoke:
        args.n, args.seeds, args.epochs, args.patience = 400, 1, 40, 8
        args.systems = ["chain"]
        args.outdir = f"{args.outdir}/smoke"
        print("SMOKE RUN: n=400, 1 seed, chain only, 40 epochs\n")

    print("=" * 96)
    print("WIRING CHECK 1: every wiring at coupling zero must BE the control")
    print("=" * 96)
    if not check_zero_coupling_is_control(args.n, args.seeds):
        print("  FAILED -- the systems differ by more than their wiring; aborting.")
        return 1

    print("\n" + "=" * 96)
    print("WIRING CHECK 2: a zeroed coupling must decouple exactly its own series")
    print("=" * 96)
    if not check_wiring(args.n):
        print("  FAILED -- the generators do not implement the stated graphs; aborting.")
        return 1

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    retrieval: list[dict] = []
    ccm_all: list[dict] = []
    for system in args.systems:
        builder = SYSTEM_BUILDERS[system]
        for seed in range(args.seeds):
            data = builder(n=args.n, coupling=args.coupling, seed=seed)
            series = {name: np.asarray(data[name]) for name in NAMES}

            start = time.time()
            ccm_all += ccm_rows(series, system, seed, args)
            ccm_seconds = time.time() - start

            start = time.time()
            design = build_design(series)
            retrieval += retrieval_rows(design, system, seed, args)
            retrieval_seconds = time.time() - start

            print(f"  {system}[{seed}]: ccm {ccm_seconds:.0f}s, "
                  f"retrieval {retrieval_seconds:.0f}s")

    retrieval_frame = pd.DataFrame(retrieval)
    ccm_frame = pd.DataFrame(ccm_all)
    gains = gain_rows(retrieval_frame)

    retrieval_frame.to_csv(outdir / "retrieval.csv", index=False)
    ccm_frame.to_csv(outdir / "ccm.csv", index=False)
    gains.to_csv(outdir / "gains.csv", index=False)

    fmt = ("display.float_format", "{:.4f}".format, "display.width", 250)

    # ---------------------------------------------------------------- guards
    print("\n" + "=" * 96)
    print("GUARD 1: did retrieval from the target alone beat the mean?")
    print("=" * 96)
    print(f"  Arms with R^2 < {MIN_RETRIEVAL_R2} on the {{V}}-alone set are flagged and")
    print("  excluded from scoring: their gains are differences of two noise levels.")
    alone = retrieval_frame[retrieval_frame["set"] == retrieval_frame["target"]]
    with pd.option_context(*fmt):
        print(alone.groupby(["system", "operator", "target"]).agg(
            r2=("r2", "mean"), mse=("mse", "mean"), n=("r2", "size"),
        ).to_string())
    failed = alone[alone.r2 < MIN_RETRIEVAL_R2]
    print(f"\n  flagged arms: {len(failed)} of {len(alone)}")
    for row in failed.itertuples():
        print(f"    {row.system}[{row.seed}] {row.operator} target {row.target}: "
              f"r2 {row.r2:+.4f}")

    print("\n" + "=" * 96)
    print("GUARD 2: the same MLP with no bottleneck (rule 4)")
    print("=" * 96)
    print(f"  A set that cannot be learned at full width ({MIN_FULLWIDTH_R2} R^2)")
    print("  cannot support a statement about what compressing it cost.")
    full = retrieval_frame[retrieval_frame.operator == "mlp_fullwidth"]
    with pd.option_context(*fmt):
        print(full.groupby(["system", "target"]).agg(
            r2=("r2", "mean"), worst=("r2", "min"), n=("r2", "size"),
        ).to_string())
    weak = full[full.r2 < MIN_FULLWIDTH_R2]
    if len(weak):
        print(f"\n  BELOW THRESHOLD: {len(weak)} of {len(full)} full-width fits")
        for row in weak.itertuples():
            print(f"    {row.system}[{row.seed}] target {row.target}: r2 {row.r2:+.4f}")
    else:
        print("\n  every full-width fit cleared the threshold")

    # ------------------------------------------------------------------ CCM
    print("\n" + "=" * 96)
    print("PAIRWISE CCM: does it produce the structural false positives?")
    print("=" * 96)
    print("  rho at the largest library, averaged over seeds; 'conv' counts the")
    print("  seeds where DirectionResult.is_convergent() holds at package defaults.")
    with pd.option_context(*fmt):
        print(ccm_frame.groupby(["system", "setting", "source", "target"]).agg(
            rho=("rho", "mean"), rho_min=("rho", "min"),
            delta=("delta_rho", "mean"), conv=("convergent", "sum"),
            n=("convergent", "size"), E=("E", "mean"),
        ).to_string())

    print("\n" + "=" * 96)
    print("TRANSITIVE IMPRINT: is the trap pair's rho elevated over the control's?")
    print("=" * 96)
    print("  A verdict of 'no edge' on a trap pair can mean two different things:")
    print("  the imprint is absent, or it is present and below the convergence")
    print("  criterion.  Those have different consequences for the motivation, so")
    print("  the same ordered pair is compared against the uncoupled control, which")
    print("  is the only arm where the imprint is known to be absent.")
    if "independent" in set(ccm_frame.system):
        control_rho = {
            (row.setting, row.source, row.target): row.rho
            for row in ccm_frame[ccm_frame.system == "independent"]
            .groupby(["setting", "source", "target"], as_index=False)
            .agg(rho=("rho", "max")).itertuples()
        }
        for (system, source, target), role in STRUCTURAL_TRAPS.items():
            if system not in args.systems:
                continue
            block = ccm_frame[(ccm_frame.system == system)
                              & (ccm_frame.source == source)
                              & (ccm_frame.target == target)]
            for setting, sub in block.groupby("setting"):
                worst_control = control_rho.get((setting, source, target), float("nan"))
                print(f"  {system:12s} {source}->{target} {setting:5s} {role:26s} "
                      f"rho min {sub.rho.min():+.3f} mean {sub.rho.mean():+.3f} "
                      f"| control max {worst_control:+.3f} "
                      f"| separates: {sub.rho.min() > worst_control}")
    else:
        print("  the control was not run; no comparison available.")

    ccm_v = ccm_verdicts(ccm_frame)
    retrieval_v = retrieval_verdicts(
        gains,
        # THE NULL: the largest gain of either kind anywhere on the control.
        nulls={
            operator: float(max(
                block.marginal_gain.max(), block.conditional_gain.max()
            )) if len(block) else float("nan")
            for operator, block in
            gains[gains.system == "independent"].groupby("operator", sort=False)
        } if (gains.system == "independent").any() else
        {operator: float("nan") for operator in gains.operator.unique()},
    )
    verdicts = pd.concat([ccm_v, retrieval_v], ignore_index=True)
    verdicts.to_csv(outdir / "verdicts.csv", index=False)

    print("\n" + "=" * 96)
    print("CONDITIONAL RETRIEVAL: gains per ordered pair")
    print("=" * 96)
    print("  marginal   = gain of {V,U} over {V}")
    print("  conditional= gain of {V,U,W} over {V,W}, W the third series")
    print("  Both are relative MSE reductions; the minimum over seeds is shown,")
    print("  which is the number the verdict uses.")
    with pd.option_context(*fmt):
        print(gains.groupby(["system", "operator", "source", "target"]).agg(
            marginal_mean=("marginal_gain", "mean"),
            marginal_min=("marginal_gain", "min"),
            conditional_mean=("conditional_gain", "mean"),
            conditional_min=("conditional_gain", "min"),
            n=("marginal_gain", "size"),
        ).to_string())

    print("\n" + "=" * 96)
    print("NULL CALIBRATION: the control's largest spurious gain")
    print("=" * 96)
    if (gains.system == "independent").any():
        for operator, block in gains[gains.system == "independent"].groupby("operator"):
            print(f"  {operator:5s} max marginal {block.marginal_gain.max():+.4f}  "
                  f"max conditional {block.conditional_gain.max():+.4f}  "
                  f"-> NULL {max(block.marginal_gain.max(), block.conditional_gain.max()):+.4f}")
    else:
        print("  independent_triple was not run; no null, verdicts are undefined.")

    # -------------------------------------------------------------- headline
    scored_systems = [s for s in args.systems if s != "independent"]
    print("\n" + "=" * 96)
    print("HEADLINE: EDGE RECOVERY")
    print("=" * 96)
    print(f"  Ordered pairs of {scored_systems}; positives are the direct edges of")
    print("  the generating DAG, negatives are everything else -- including the")
    print("  chain's mediated A->C and the fork's confounded B~C.")
    print("  The control's own pairs are excluded: the retrieval null is defined as")
    print("  the maximum of the control's gains, so its verdicts are all-negative by")
    print("  construction and would buy specificity for free.")
    scores = score_edges(verdicts, scored_systems)
    scores.to_csv(outdir / "edge_recovery.csv", index=False)
    with pd.option_context(*fmt):
        print(scores.to_string(index=False))

    print("\n" + "=" * 96)
    print("THE THREE STRUCTURAL TRAPS, PAIR BY PAIR")
    print("=" * 96)
    for (system, source, target), role in STRUCTURAL_TRAPS.items():
        if system not in args.systems:
            continue
        block = verdicts[(verdicts.system == system) & (verdicts.source == source)
                         & (verdicts.target == target)]
        calls = ", ".join(
            f"{row.method}={'EDGE' if row.calls_edge else 'no'}"
            for row in block.itertuples()
        )
        print(f"  {system:12s} {source}->{target}  {role:26s}  {calls}")

    print("\n  and the direct edges, for contrast:")
    for system in scored_systems:
        for source, target in sorted(TRUE_EDGES[system]):
            block = verdicts[(verdicts.system == system) & (verdicts.source == source)
                             & (verdicts.target == target)]
            calls = ", ".join(
                f"{row.method}={'EDGE' if row.calls_edge else 'no'}"
                for row in block.itertuples()
            )
            print(f"  {system:12s} {source}->{target}  {'direct':26s}  {calls}")

    print("\n" + "=" * 96)
    print("EVERY VERDICT")
    print("=" * 96)
    with pd.option_context(*fmt):
        print(verdicts.sort_values(["system", "method", "source", "target"])
              .to_string(index=False))

    print(f"\nwrote {outdir}/retrieval.csv, ccm.csv, gains.csv, verdicts.csv, "
          f"edge_recovery.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
