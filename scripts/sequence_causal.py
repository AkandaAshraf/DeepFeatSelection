"""Can a sequence model detect coupling where CCM has to be told the embedding?

CCM is the incumbent for causal detection in nonlinear dynamical systems, and
this project has measured three of its weaknesses directly:

* it needs an embedding dimension chosen by hand, and choosing wrongly is not a
  graceful failure -- at E=3 on Rossler-Lorenz it reported the arrow backwards
  with p=0.0099, confidently and incorrectly;
* it is pairwise, so a coupling mediated by a third series is not something it
  can represent;
* under strong coupling both directions converge and only their comparison
  resolves anything.

A recurrent model reading the multivariate trajectory has none of those
structural constraints: no embedding dimension to choose, all series visible at
once, and no pairwise restriction. Whether that translates into better detection
is the question here.

Scope, stated plainly: the target is **detection**, not orientation. Everything
in this project says orientation is unavailable observationally in the regime
where a driver's imprint lives inside its response, and nothing here is expected
to change that. What is being asked is whether the coupled pairs can be told
from the uncoupled ones.

Design: predict series B one step ahead from the recent history of all series,
then ablate A's channel. If A drives B, removing it should cost forecast
accuracy. CCM runs on the same data as the incumbent.

THREE THINGS THIS RUN ADDS OVER THE FIRST VERSION
-------------------------------------------------
1. A coupling-strength sweep.  Detection at strong coupling is trivial and
   cannot discriminate between methods; the informative regime is the one where
   detection starts to fail.  ``coupled_logistic`` already takes
   ``coupling_x_to_y``, so no generator was modified -- but see
   SWEEP GROWTH RATE below, because the obvious sweep is wired wrong.

2. A learned-or-not flag.  The series are standardised, so a one-step forecast
   MSE near the held-out variance means the model emitted the mean and learned
   nothing.  A deprivation score is a *difference* of two such models' errors,
   so when the full model never forecast, the score is a difference of two noise
   levels wearing a decimal point.  Those rows are flagged and excluded from
   every aggregate.

3. CCM at two embedding dimensions.  E=3 is the setting that produced the
   backwards call, so leaving it there would stack the comparison against the
   incumbent.  Every system is additionally run at the ``E`` that
   ``optimal_embedding_dimension`` picks, and both are reported.

SWEEP GROWTH RATE -- WHY THE SWEEP IS NOT AT THE GENERATOR DEFAULT
------------------------------------------------------------------
The zero-coupling end of the sweep has to reproduce the negative control,
otherwise the sweep is not a sweep of one variable.  At ``coupled_logistic``'s
default ``r_y=3.5`` it does not: that growth rate sits in the period-4 window,
so an uncoupled response collapses onto four distinct values forever (measured:
4 unique values in 2000 samples) and every causality test trivially passes on a
control with almost no entropy.  ``independent_logistic`` exists precisely
because of this and moves the response to ``r_y=3.7``.

So the sweep runs at ``r_x=3.8, r_y=3.7`` -- the control's growth rates -- with
only ``coupling_x_to_y`` varying.  At coupling 0 that call is byte-identical to
``independent_logistic``, which :func:`check_zero_coupling_is_control` asserts
before any model is fitted.  The consequence to keep in view when reading the
numbers: this is a *different* coupled system from the one the generator
defaults to, because its response is chaotic rather than period-4 when
undriven.  A driver imprinting on an already-chaotic response is the harder and
more honest case.

HELD OUT HAS TO MEAN HELD OUT
-----------------------------
The first version split the windows contiguously -- correctly, because they
overlap -- but then used one segment for three jobs: early stopping, weight
restoration, and the reported forecast error, which it took as the minimum
validation loss over epochs.  That number is a minimum selected over epochs on
its own data, and it equals the restored model's evaluation on that segment
exactly, so the held-out performance the protocol requires beside every
detection score was in fact selected on.  :func:`forecast_loss` now keeps a
third contiguous segment that nothing but the final evaluation touches, with an
embargo of ``lag`` windows at each seam so no test window shares a timestep, or
a training target, with the training segment.

The correction was checked before it was made rather than assumed to matter:
under the old split the ablated model's last-epoch loss ran 38% above its
per-epoch minimum against 4% for the unablated one, so the selection shrank the
deprivation numerator and made the score *conservative*.  Re-running the
weak-coupling rows with the clean split moves every individual number but leaves
the comparison that the run turns on unchanged.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import pandas as pd

from deepfeatselect.ccm import ccm, optimal_embedding_dimension
from deepfeatselect.synthetic import (
    coupled_logistic,
    independent_logistic,
    rossler_lorenz,
)

# Growth rates for the sweep. These are independent_logistic's, not
# coupled_logistic's defaults; see SWEEP GROWTH RATE in the module docstring.
SWEEP_R_X = 3.8
SWEEP_R_Y = 3.7

# Fraction of held-out target variance a forecast must explain before its
# deprivation score is treated as a measurement.  Fixed here rather than chosen
# after seeing the losses: a threshold picked downstream of the results is a
# free parameter for making a null look like a finding.  0.10 is deliberately
# permissive -- it excludes only models that essentially predicted the mean, not
# models that merely forecast poorly.
MIN_FORECAST_R2 = 0.10

# Contiguous three-way split: train / early-stop / test, in that temporal order.
# The early-stopping segment cannot double as the reported held-out score --
# see :func:`forecast_loss` -- so the last segment is reserved and touched by
# nothing but the final evaluation.
TRAIN_FRACTION = 0.60
VAL_FRACTION = 0.15


@dataclass(frozen=True)
class Fit:
    """Held-out forecast error together with what it has to be compared against.

    ``mse`` alone cannot say whether a model learned: the split is contiguous,
    so the held-out segment's variance is not exactly 1 even though the whole
    series was standardised.  Carrying ``target_variance`` alongside makes the
    "did it beat the mean" question arithmetic rather than a judgement call.
    """

    mse: float
    target_variance: float

    @property
    def r2(self) -> float:
        """Fraction of held-out variance explained; <= 0 means no better than the mean."""
        return 1.0 - self.mse / (self.target_variance + 1e-12)

    @property
    def learned(self) -> bool:
        return self.r2 >= MIN_FORECAST_R2


@dataclass(frozen=True)
class System:
    """One generated pair of series with its ground truth and its provenance."""

    label: str
    family: str
    coupling: float
    seed: int
    x: np.ndarray
    y: np.ndarray
    truth_x_to_y: bool
    truth_y_to_x: bool


def windows(series: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows of length ``lag`` and the next value of each channel.

    Shape (n_windows, lag, n_channels); the target is taken one step past the
    end of each window.
    """
    n, c = series.shape
    x = np.stack([series[i:i + lag] for i in range(n - lag)], axis=0)
    y = series[lag:]
    return x.astype("float32"), y.astype("float32")


def build_lstm(lag: int, channels: int, units: int, dropout: float):
    """Bidirectional LSTM over the window, predicting one channel ahead.

    Bidirectional over a *window* rather than over the whole series: within a
    short window both directions carry usable context, and nothing about the
    forecast target leaks backwards because the target sits outside the window.
    """
    inputs = keras.layers.Input(shape=(lag, channels))
    h = keras.layers.Bidirectional(keras.layers.LSTM(units))(inputs)
    if dropout:
        h = keras.layers.Dropout(dropout)(h)
    h = keras.layers.Dense(units, activation="relu")(h)
    out = keras.layers.Dense(1)(h)
    model = keras.Model(inputs, out)
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(3e-3))
    return model


def forecast_loss(x, y, seed, units, epochs, dropout, patience) -> Fit:
    """Test-segment one-step forecast error, from a contiguous three-way split.

    The split must not be shuffled: neighbouring windows overlap, so a random
    split would put near-duplicates on both sides and report a memorisation
    score as generalisation.  ``shuffle=True`` below reorders batches *within*
    the training segment only, which touches no segment boundary.

    Contiguity alone is not enough, and two further things were missing from the
    first version of this function:

    * A segment the model never selected on.  ``EarlyStopping`` restores the
      weights that minimise the validation loss, so that loss is a minimum taken
      over epochs of the very quantity it is chosen by -- measured, it equals the
      restored model's own evaluation on that segment exactly.  Reporting it as
      held-out performance overstates the forecast, and the protocol asks for
      a number that can be trusted standing next to a detection score.  Train,
      early-stop and test are therefore three disjoint contiguous segments and
      the reported error comes from the last.
    * An embargo at each seam.  Window ``i`` spans ``[i, i + lag)`` and its
      target sits at ``i + lag``, so without a gap the first ``lag`` windows of a
      segment share timesteps -- and training *targets* -- with the tail of the
      previous one.  Dropping ``lag`` windows before each boundary removes every
      shared sample.
    """
    keras.utils.set_random_seed(seed)
    lag, n = x.shape[1], len(x)
    train_end = int(TRAIN_FRACTION * n)
    val_end = int((TRAIN_FRACTION + VAL_FRACTION) * n)
    model = build_lstm(lag, x.shape[2], units, dropout)
    stopper = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True)
    model.fit(
        x[:train_end - lag], y[:train_end - lag],
        validation_data=(x[train_end:val_end - lag], y[train_end:val_end - lag]),
        epochs=epochs, batch_size=64, shuffle=True, verbose=0,
        callbacks=[stopper])
    # The variance is of the same test segment the loss was measured on, so the
    # two are directly comparable.
    return Fit(mse=float(model.evaluate(x[val_end:], y[val_end:], verbose=0)),
               target_variance=float(np.var(y[val_end:])))


def deprivation_score(series: np.ndarray, target: int, source: int,
                      args, seed: int) -> tuple[float, Fit, Fit]:
    """Rise in forecast error for ``target`` when ``source``'s channel is zeroed.

    Normalised by the full-model error, so systems with different scales stay
    comparable.  Both fits are returned, not just the score: the score is only
    interpretable if the unablated model forecast anything in the first place.
    """
    x, y_all = windows(series, args.lag)
    y = y_all[:, target:target + 1]
    base = forecast_loss(x, y, seed, args.units, args.epochs,
                         args.dropout, args.patience)
    ablated = x.copy()
    ablated[:, :, source] = 0.0
    without = forecast_loss(ablated, y, seed, args.units, args.epochs,
                            args.dropout, args.patience)
    return (without.mse - base.mse) / (base.mse + 1e-12), base, without


def check_zero_coupling_is_control(n: int, seeds: int) -> bool:
    """The sweep's zero-coupling end must be the negative control, exactly.

    This is a correctness check on the sweep's wiring, not on the generator.  If
    it fails, the sweep is varying more than the coupling and no comparison
    across coupling levels means anything -- so it is checked before any compute
    is spent, and reported rather than swallowed.
    """
    ok = True
    for seed in range(seeds):
        swept = coupled_logistic(n=n, r_x=SWEEP_R_X, r_y=SWEEP_R_Y,
                                 coupling_x_to_y=0.0, seed=seed)
        control = independent_logistic(n=n, seed=seed)
        same = (np.array_equal(swept["x"], control["x"])
                and np.array_equal(swept["y"], control["y"]))
        ok = ok and same
        print(f"  seed {seed}: zero-coupling arm == independent_logistic: {same}")
    return ok


def build_systems(args) -> list[System]:
    """Every (series pair, ground truth) the run scores, in the order it runs them."""
    systems: list[System] = []

    for coupling in args.couplings:
        for seed in range(args.seeds):
            data = coupled_logistic(n=args.n, r_x=SWEEP_R_X, r_y=SWEEP_R_Y,
                                    coupling_x_to_y=coupling, seed=seed)
            systems.append(System(
                label=f"logistic_c{coupling:.2f}[{seed}]",
                family=f"logistic_c{coupling:.2f}",
                coupling=coupling, seed=seed,
                x=np.asarray(data["x"]), y=np.asarray(data["y"]),
                truth_x_to_y=coupling > 0.0, truth_y_to_x=False))

    for coupling in args.rossler_couplings:
        for seed in range(args.seeds):
            # Generated at args.subsample times the wanted length so that the
            # analysed series is args.n points AFTER subsampling. dt=0.01
            # oversamples both attractors so badly that temporal neighbours are
            # manifold neighbours and cross mapping scores on autocorrelation
            # alone; subsampling is not optional here.
            data = rossler_lorenz(n=args.n * args.subsample, coupling=coupling,
                                  seed=seed)
            systems.append(System(
                label=f"rossler_c{coupling:.1f}[{seed}]",
                family=f"rossler_c{coupling:.1f}",
                coupling=coupling, seed=seed,
                x=np.asarray(data["x"])[::args.subsample],
                y=np.asarray(data["y"])[::args.subsample],
                truth_x_to_y=coupling > 0.0, truth_y_to_x=False))

    if args.rossler_short:
        for seed in range(args.seeds):
            # The configuration that produced the recorded backwards call:
            # n generated, then subsampled, so only n/subsample points survive.
            # Kept as its own arm because a finding recorded at 200 points is
            # not evidence about a run at 2000.
            data = rossler_lorenz(n=args.n, coupling=2.0, seed=seed)
            systems.append(System(
                label=f"rossler_short[{seed}]",
                family="rossler_short",
                coupling=2.0, seed=seed,
                x=np.asarray(data["x"])[::args.subsample],
                y=np.asarray(data["y"])[::args.subsample],
                truth_x_to_y=True, truth_y_to_x=False))

    return systems


def ccm_columns(series: np.ndarray, E: int, seed: int, prefix: str,
                truth_x_to_y: bool, exclusion_radius: int = 0) -> tuple[dict, float]:
    """One CCM run, flattened into columns, with its wall time.

    ``reversed`` is computed against the ground truth with the comparison fixed
    a priori: the call is backwards when the direction with the higher skill at
    the largest library is not the true one.  It is recorded for both embedding
    settings so "does choosing E properly rescue CCM" has an answer rather than
    an anecdote.

    ``exclusion_radius`` is the Theiler window.  Both :mod:`deepfeatselect.ccm`
    and :func:`rossler_lorenz` warn that oversampled continuous-time data lets
    cross mapping score on autocorrelation alone, so leaving it at 0 is a claim
    that needs checking rather than a default to inherit.  It was checked: on the
    subsampled Rossler-Lorenz arms, where the analysed series still has lag-1
    autocorrelation of 0.81 (driver) and 0.61 (response), raising the window to
    20 samples moves every rho by at most 0.003 and does not flip a single
    orientation call.  The backwards call on that system is therefore not an
    autocorrelation artefact.  Exposed anyway so the check is repeatable.
    """
    t0 = time.time()
    result = ccm(series[:, 0], series[:, 1], E=E, seed=seed,
                 exclusion_radius=exclusion_radius)
    seconds = time.time() - t0

    fwd = result.x_causes_y.rho_at_max_lib
    rev = result.y_causes_x.rho_at_max_lib
    higher_is_x_to_y = fwd > rev
    return {
        f"{prefix}_E": E,
        f"{prefix}_x_to_y": fwd,
        f"{prefix}_y_to_x": rev,
        f"{prefix}_conv_x_to_y": result.x_causes_y.is_convergent(),
        f"{prefix}_conv_y_to_x": result.y_causes_x.is_convergent(),
        # Only meaningful where a true arrow exists; nan on the uncoupled arms
        # keeps them out of the reversal rate instead of scoring a coin flip.
        f"{prefix}_reversed": (not higher_is_x_to_y) if truth_x_to_y else np.nan,
        f"{prefix}_seconds": seconds,
    }, seconds


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--lag", type=int, default=8)
    p.add_argument("--units", type=int, default=32)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--embedding", type=int, default=3,
                   help="fixed embedding dimension; the setting that produced "
                        "the backwards call, kept so the comparison is auditable")
    p.add_argument("--max-E", type=int, default=10)
    p.add_argument("--couplings", type=float, nargs="+",
                   default=[0.0, 0.08, 0.16, 0.24, 0.32])
    # nargs="*" so a logistic-only refinement run can pass an empty list rather
    # than paying for the continuous-time arms again.
    p.add_argument("--rossler-couplings", type=float, nargs="*",
                   default=[0.0, 2.0])
    p.add_argument("--rossler-short", action="store_true", default=True)
    p.add_argument("--no-rossler-short", dest="rossler_short",
                   action="store_false")
    p.add_argument("--subsample", type=int, default=10)
    p.add_argument("--exclusion-radius", type=int, default=0,
                   help="Theiler window for CCM; see ccm_columns for the "
                        "measurement showing 0 is safe on these arms")
    p.add_argument("--outdir", default="ExpOutput/sequence")
    args = p.parse_args()

    print("=" * 96)
    print("SWEEP WIRING CHECK: coupling 0 must be the negative control")
    print("=" * 96)
    if not check_zero_coupling_is_control(args.n, args.seeds):
        print("  FAILED -- the sweep varies more than the coupling; aborting.")
        return 1

    systems = build_systems(args)
    print(f"\n{len(systems)} systems, {4 * len(systems)} LSTM fits\n")

    rows = []
    for system in systems:
        xs, ys = system.x, system.y
        series = np.column_stack([
            (xs - xs.mean()) / (xs.std() + 1e-12),
            (ys - ys.mean()) / (ys.std() + 1e-12),
        ])
        seed = system.seed

        t0 = time.time()
        # Does removing x hurt the forecast of y, and vice versa?
        x_to_y, base_y, abl_y = deprivation_score(series, target=1, source=0,
                                                  args=args, seed=seed)
        y_to_x, base_x, abl_x = deprivation_score(series, target=0, source=1,
                                                  args=args, seed=seed)
        lstm_seconds = time.time() - t0

        # Chosen per series, then the larger taken: ccm embeds both manifolds at
        # one E, and under-embedding folds distinct states together, which is
        # exactly the failure that produced the backwards call. Over-embedding
        # only dilutes neighbours, so the max protects the weaker of the two.
        # Timed separately and reported separately: choosing E is a real cost of
        # running CCM at its optimal embedding, and folding it into neither
        # method's wall clock would understate what "CCM at E=opt" costs.
        t0 = time.time()
        e_opt_x = optimal_embedding_dimension(series[:, 0], max_E=args.max_E)
        e_opt_y = optimal_embedding_dimension(series[:, 1], max_E=args.max_E)
        e_select_seconds = time.time() - t0
        e_opt = max(e_opt_x, e_opt_y)

        fixed, _ = ccm_columns(series, args.embedding, seed, "ccm_fixed",
                               system.truth_x_to_y, args.exclusion_radius)
        chosen, _ = ccm_columns(series, e_opt, seed, "ccm_opt",
                                system.truth_x_to_y, args.exclusion_radius)

        row = {
            "system": system.label, "family": system.family,
            "coupling": system.coupling, "seed": seed,
            "n_points": len(xs),
            "truth_x_to_y": system.truth_x_to_y,
            "truth_y_to_x": system.truth_y_to_x,
            "lstm_x_to_y": x_to_y, "lstm_y_to_x": y_to_x,
            "base_loss_y": base_y.mse, "base_loss_x": base_x.mse,
            "ablated_loss_y": abl_y.mse, "ablated_loss_x": abl_x.mse,
            "val_var_y": base_y.target_variance,
            "val_var_x": base_x.target_variance,
            "forecast_r2_y": base_y.r2, "forecast_r2_x": base_x.r2,
            "learned_y": base_y.learned, "learned_x": base_x.learned,
            "learned_both": base_y.learned and base_x.learned,
            "e_opt_x": e_opt_x, "e_opt_y": e_opt_y,
            "lstm_seconds": lstm_seconds,
            "e_select_seconds": e_select_seconds,
        }
        row.update(fixed)
        row.update(chosen)
        rows.append(row)

        flag = "" if row["learned_both"] else "  [NOT LEARNED]"
        print(f"  {system.label}: lstm x->y {x_to_y:+.3f} y->x {y_to_x:+.3f} "
              f"(r2 {base_y.r2:+.2f}/{base_x.r2:+.2f}){flag}\n"
              f"      ccm E=3 {row['ccm_fixed_x_to_y']:+.3f}/{row['ccm_fixed_y_to_x']:+.3f}"
              f" | ccm E={e_opt} {row['ccm_opt_x_to_y']:+.3f}/{row['ccm_opt_y_to_x']:+.3f}"
              f" | {lstm_seconds:.0f}s vs "
              f"{row['ccm_fixed_seconds'] + row['ccm_opt_seconds']:.0f}s")

    frame = pd.DataFrame(rows)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outdir / "sequence_causal.csv", index=False)

    fmt = ("display.float_format", "{:.3f}".format, "display.width", 250)

    print("\n" + "=" * 96)
    print("FORECASTER SANITY: did the model learn anything before it was ablated?")
    print("=" * 96)
    print("  Series are standardised, so MSE near the held-out variance means the")
    print(f"  model emitted the mean. Rows with r2 < {MIN_FORECAST_R2} on either")
    print("  target are excluded from every aggregate below.")
    with pd.option_context(*fmt):
        print(frame.groupby("family").agg(
            base_loss_y=("base_loss_y", "mean"),
            base_loss_x=("base_loss_x", "mean"),
            r2_y=("forecast_r2_y", "mean"), r2_x=("forecast_r2_x", "mean"),
            learned=("learned_both", "sum"), n=("learned_both", "size"),
        ).to_string())

    failed = frame[~frame.learned_both]
    if len(failed):
        print(f"\n  EXCLUDED ({len(failed)} of {len(frame)}):")
        for _, r in failed.iterrows():
            print(f"    {r.system}: r2_y {r.forecast_r2_y:+.3f} "
                  f"r2_x {r.forecast_r2_x:+.3f}")
    else:
        print("\n  No row excluded: every forecaster beat its held-out mean.")

    usable = frame[frame.learned_both]

    print("\n" + "=" * 96)
    print("COUPLING SWEEP: detection score against coupling strength")
    print("=" * 96)
    with pd.option_context(*fmt):
        print(usable.groupby("family").agg(
            coupling=("coupling", "first"), n=("coupling", "size"),
            lstm_x_to_y=("lstm_x_to_y", "mean"), lstm_y_to_x=("lstm_y_to_x", "mean"),
            ccm3_x_to_y=("ccm_fixed_x_to_y", "mean"),
            ccm3_y_to_x=("ccm_fixed_y_to_x", "mean"),
            ccmE_x_to_y=("ccm_opt_x_to_y", "mean"),
            ccmE_y_to_x=("ccm_opt_y_to_x", "mean"),
            E_opt=("ccm_opt_E", "mean"),
        ).to_string())

    print("\n" + "=" * 96)
    print("DETECTION: COUPLED VERSUS UNCOUPLED")
    print("=" * 96)
    print("  The statistic is the larger of the two directions, fixed before the")
    print("  run and applied identically to the coupled and uncoupled arms. It is")
    print("  a detection statistic only -- orientation is reported separately and")
    print("  not claimed.")

    for label, control_family, coupled_families in (
        ("logistic", "logistic_c0.00",
         [f for f in usable.family.unique()
          if f.startswith("logistic_") and f != "logistic_c0.00"]),
        # rossler_short is deliberately absent: its control would have to be an
        # uncoupled run at the same 200 points, which is not in the grid, and a
        # 2000-point control cannot calibrate a 200-point arm. It appears in the
        # orientation table only, which is what it was kept for.
        ("rossler", "rossler_c0.0",
         [f for f in usable.family.unique() if f == "rossler_c2.0"]),
    ):
        control = usable[usable.family == control_family]
        if control.empty:
            print(f"\n  {label}: no usable control rows, skipped")
            continue
        print(f"\n  {label} (control {control_family}, "
              f"{len(control)} usable rows)")
        for method, cols in (("lstm", ("lstm_x_to_y", "lstm_y_to_x")),
                             ("ccm E=3", ("ccm_fixed_x_to_y", "ccm_fixed_y_to_x")),
                             ("ccm E=opt", ("ccm_opt_x_to_y", "ccm_opt_y_to_x"))):
            u = control[list(cols)].to_numpy().max(axis=1)
            print(f"    {method:10s} uncoupled max {u.mean():+.3f} "
                  f"(worst {u.max():+.3f})")
            for family in sorted(coupled_families):
                block = usable[usable.family == family]
                if block.empty:
                    continue
                c = block[list(cols)].to_numpy().max(axis=1)
                print(f"      vs {family:18s} {c.mean():+.3f} "
                      f"(min {c.min():+.3f})  separates: {c.min() > u.max()}")

    print("\n" + "=" * 96)
    print("EMBEDDING DIMENSION: does choosing E rescue CCM's orientation?")
    print("=" * 96)
    print("  'reversed' = the direction with the higher skill is not the true one.")
    print("  Only the coupled arms contribute; the uncoupled ones have no arrow.")
    oriented = usable[usable.truth_x_to_y]
    if len(oriented):
        with pd.option_context(*fmt):
            print(oriented.groupby("family").agg(
                n=("system", "size"),
                rev_E3=("ccm_fixed_reversed", "mean"),
                rev_Eopt=("ccm_opt_reversed", "mean"),
                e_opt_x=("e_opt_x", "mean"), e_opt_y=("e_opt_y", "mean"),
                conv_fwd_E3=("ccm_fixed_conv_x_to_y", "mean"),
                conv_rev_E3=("ccm_fixed_conv_y_to_x", "mean"),
                conv_fwd_Eopt=("ccm_opt_conv_x_to_y", "mean"),
                conv_rev_Eopt=("ccm_opt_conv_y_to_x", "mean"),
            ).to_string())

    print("\n" + "=" * 96)
    print("WALL CLOCK -- UPPER BOUNDS UNDER CONTENTION, NOT CLEAN MEASUREMENTS")
    print("=" * 96)
    print("  Another workflow may have been holding CPU during this run, so these")
    print("  are upper bounds on both methods and their ratio is not a benchmark.")
    with pd.option_context(*fmt):
        print(frame.groupby("family").agg(
            lstm_s=("lstm_seconds", "mean"),
            ccm_E3_s=("ccm_fixed_seconds", "mean"),
            ccm_Eopt_s=("ccm_opt_seconds", "mean"),
            E_select_s=("e_select_seconds", "mean"),
        ).to_string())
    print("\n  E_select_s is the cost of picking E and belongs to the CCM E=opt")
    print("  column, not to either method's cross-map time.")

    print(f"\nwrote {outdir}/sequence_causal.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
