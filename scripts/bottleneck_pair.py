"""Pairwise causal detection by the dimension deficit of the joint delay embedding.

THE IDEA UNDER TEST
-------------------
Compressibility as the causal signature, made geometric.  If X drives Y, the
joint attractor of the two delay-embedded systems occupies fewer dimensions
than dim(X) + dim(Y): generalised synchrony pulls Y's state toward a function
of X's history, so the concatenated state is retrievable through a bottleneck
narrower than the sum.  Independent systems need the full product geometry and
no such narrow retrieval exists.  A bottleneck autoencoder measures the deficit
directly: sweep the bottleneck width b and ask at which b the held-out joint
state becomes reconstructable.  CCM is the incumbent detector and a special
case of the same principle; PCA at the same widths is the linear autoencoder
and the cheap baseline the deep one must beat to earn its keep.

WHY THE PROTOCOL LOOKS THE WAY IT DOES
--------------------------------------
* The logistic sweep runs at ``r_x=3.8, r_y=3.7`` -- the growth rates of
  ``independent_logistic`` -- so that the coupling-zero arm is byte-identical
  to the negative control.  ``check_zero_coupling_is_control`` (imported from
  the verified protocol in ``scripts/sequence_causal.py``, together with the
  growth-rate and split-fraction constants) asserts that identity before any
  compute is spent.  At the generator default ``r_y=3.5`` the uncoupled
  response is a four-point cycle and the whole comparison is void; that
  mistake has been made once already and is not being made again.
* Rossler-Lorenz has its own generator-supported control: ``coupling=0`` gives
  two independent chaotic attractors from the same code path, so the arm and
  its control differ in exactly one variable by construction.  The series are
  subsampled ``::10`` because at dt=0.01 temporal neighbours are manifold
  neighbours and every state-space method scores on autocorrelation alone.
* Delay-embedded states overlap in time, so neighbouring rows are
  near-duplicates.  Train / early-stop / test are three disjoint contiguous
  segments (fractions 0.60 / 0.15 / 0.25, the reference protocol's) with an
  embargo of ``max(E_x, E_y)`` rows dropped before each seam -- at least the
  embedding span, so no test row shares a timestep with a training row.
  Standardisation uses train statistics only.
* Every arm trains the identical architecture (D -> 32 -> b -> 32 -> D, relu
  hidden layers, linear bottleneck and output, Adam 3e-3, same epoch budget and
  early-stopping patience) so arms differ only in data.

STATISTICS, FIXED A PRIORI -- DECLARED BEFORE ANY RESULT WAS LOOKED AT
----------------------------------------------------------------------
* ``r2(b)``: reconstruction R^2 of the joint state on the TEST segment at
  bottleneck width b, defined as ``1 - SSE / SST`` with SSE summed over all
  rows and columns and SST taken about the test segment's own per-column
  means.  Widths tested: ``b in {1, ..., min(D, 8)} union {D}``.
* ``knee``: the smallest TESTED b with ``r2(b) >= 0.90``.  If no tested width
  reaches 0.90 the knee is undefined (nan).  When D > 9 the widths 9..D-1 are
  untested; a knee that would fall there lands on D instead, which can only
  understate the deficit.  ``deficit = D - knee`` is the PRIMARY statistic.
* ``r2_half``: r2 at ``b = ceil(D/2)`` -- SECONDARY, continuous, reported
  alongside the knee; neither statistic gets promoted post hoc.  With both
  embedding dimensions capped at 8, ``ceil(D/2) <= 8`` is always a tested
  width.
* CCM statistic: ``max(rho(x->y), rho(y->x))`` at the largest library
  (``rho_at_max_lib``), the reference protocol's detection statistic, at both
  E=3 and E=opt.  No other transform -- in particular no ``max(a, 1-a)`` --
  is applied anywhere.
* Orientation of every statistic, fixed now: coupling raises the deficit,
  raises r2_half, and raises the CCM statistic.  Detection per coupling level
  is SEPARATION: the weakest coupled arm (min over seeds) must exceed the
  strongest control arm (max over seeds).  Overlap is a null result.
* Learnability guard (rule 4): an arm's numbers count only if its full-width
  autoencoder (b = D, no bottleneck pressure) reaches ``r2 >= 0.95`` on the
  test segment.  Arms below that are unlearnable and their detection numbers
  are flagged and excluded from the separation verdicts, not reported as
  detections or nulls.  PCA at b = D is the identity and needs no guard.
* Embedding: E per series from ``optimal_embedding_dimension`` capped at 8;
  joint dimension D = E_x + E_y; CCM's E=opt is ``max(E_x, E_y)`` (the max
  protects the weaker manifold, per the reference protocol).

PREDICTIONS ON RECORD (reported against, whatever they show)
------------------------------------------------------------
* PCA fails on the logistic maps: the manifold is curved, so the linear width
  stays near D even when coupled.  If PCA matches the AE, the deep method
  bought nothing and the report says so plainly.
* Whether the AE deficit reaches CCM's detection floor (c=0.01) is open.

Smoke test (``--smoke``): n=400, 1 seed, couplings {0.0, 0.32} plus both
Rossler arms; asserts the c=0 identity, full-width r2 > 0.95 on every arm, and
that AE and PCA r2(b) are monotone non-decreasing within a tolerance of 0.05
(an optimisation-noise allowance for the AE; the mechanics are what is being
asserted, not the science).
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import pandas as pd

from deepfeatselect.ccm import ccm, optimal_embedding_dimension, time_delay_embed
from deepfeatselect.synthetic import coupled_logistic, rossler_lorenz

# The verified protocol's constants and its wiring check, imported rather than
# retyped: a re-declared constant is a constant that can drift (rule 1).
from scripts.sequence_causal import (
    SWEEP_R_X,
    SWEEP_R_Y,
    TRAIN_FRACTION,
    VAL_FRACTION,
    check_zero_coupling_is_control,
)

# --- statistics fixed a priori; see the module docstring for the reasoning ---
KNEE_R2 = 0.90            # r2 threshold defining the knee (primary statistic)
GUARD_FULL_WIDTH_R2 = 0.95  # rule-4 learnability guard on the b=D arm
MAX_E = 8                 # cap on the per-series embedding dimension
MAX_BOTTLENECK = 8        # widths 1..min(D, 8), plus D itself
SMOKE_MONOTONE_TOL = 0.05  # smoke-test allowance for optimisation noise

STAT_COLUMNS = (
    "ae_deficit",
    "ae_r2_half",
    "pca_deficit",
    "pca_r2_half",
    "ccm3_stat",
    "ccmopt_stat",
)


@dataclass(frozen=True)
class Arm:
    """One generated pair of series with its provenance."""

    label: str
    family: str
    system: str
    coupling: float
    seed: int
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class Split:
    """Contiguous train / early-stop / test segments, train-standardised.

    Kept as one object so no call site can standardise with the wrong segment's
    statistics: the mean and scale are computed on ``train`` rows only and
    applied to all three.
    """

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def joint_embedding(x: np.ndarray, y: np.ndarray, e_x: int, e_y: int) -> np.ndarray:
    """Time-aligned concatenation of the two delay embeddings.

    ``time_delay_embed`` drops the first ``E-1`` samples of each series, so the
    two manifolds start at different times; trimming both to the later start
    makes row i of each refer to the same t, which is what "joint state" means.
    """
    manifold_x, times_x = time_delay_embed(x, e_x)
    manifold_y, times_y = time_delay_embed(y, e_y)
    start = max(times_x[0], times_y[0])
    return np.hstack([manifold_x[times_x >= start], manifold_y[times_y >= start]])


def contiguous_split(states: np.ndarray, embargo: int) -> Split:
    """Three disjoint contiguous segments with an embargo at each seam.

    A row spans ``embargo - 1`` timesteps of history at most (embargo is
    ``max(E_x, E_y)``, one more than the span), so dropping ``embargo`` rows
    before each boundary guarantees no two segments share a timestep.  A random
    split would put near-duplicate attractor states on both sides and report
    memorisation as reconstruction (rule 2).
    """
    n = len(states)
    train_end = int(TRAIN_FRACTION * n)
    val_end = int((TRAIN_FRACTION + VAL_FRACTION) * n)
    train = states[: train_end - embargo]
    mean = train.mean(axis=0)
    scale = train.std(axis=0) + 1e-12
    return Split(
        train=(train - mean) / scale,
        val=(states[train_end : val_end - embargo] - mean) / scale,
        test=(states[val_end:] - mean) / scale,
    )


def reconstruction_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Multivariate R^2 with SST about the evaluation segment's own column means.

    Summed over all entries rather than averaged per column so that columns
    with more held-out variance weigh more -- the quantity is "fraction of the
    joint state's held-out variance explained", matching the reference
    protocol's variance-explained reading of a forecast.
    """
    sse = float(((actual - predicted) ** 2).sum())
    sst = float(((actual - actual.mean(axis=0)) ** 2).sum())
    return 1.0 - sse / (sst + 1e-12)


def bottleneck_widths(d: int) -> list[int]:
    """The a-priori width grid: every width up to min(D, 8), plus D itself."""
    return sorted(set(range(1, min(d, MAX_BOTTLENECK) + 1)) | {d})


def autoencoder_r2(split: Split, width: int, seed: int, epochs: int, patience: int) -> float:
    """Held-out reconstruction R^2 of one bottleneck autoencoder.

    Architecture and budget are identical for every arm and width (only the
    bottleneck differs) so that arms differ only in data.  The seed is reset
    before every fit, so two arms at the same width start from identical
    weights.  Early stopping watches the middle segment; the returned number
    comes from the test segment, which nothing else touches.
    """
    keras.utils.set_random_seed(seed)
    d = split.train.shape[1]
    inputs = keras.layers.Input(shape=(d,))
    h = keras.layers.Dense(32, activation="relu")(inputs)
    code = keras.layers.Dense(width)(h)
    h = keras.layers.Dense(32, activation="relu")(code)
    outputs = keras.layers.Dense(d)(h)
    model = keras.Model(inputs, outputs)
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(3e-3))
    stopper = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    model.fit(
        split.train,
        split.train,
        validation_data=(split.val, split.val),
        epochs=epochs,
        batch_size=64,
        shuffle=True,
        verbose=0,
        callbacks=[stopper],
    )
    return reconstruction_r2(split.test, model.predict(split.test, verbose=0))


def pca_r2(split: Split, width: int) -> float:
    """Held-out reconstruction R^2 of rank-``width`` PCA fit on train only.

    This is the linear autoencoder: same widths, same train-only fitting, same
    held-out R^2, no iterative optimisation.  The early-stop segment is unused
    because there is nothing to stop.
    """
    mean = split.train.mean(axis=0)
    _, _, vt = np.linalg.svd(split.train - mean, full_matrices=False)
    components = vt[:width].T
    reconstructed = mean + (split.test - mean) @ components @ components.T
    return reconstruction_r2(split.test, reconstructed)


def knee_width(widths: list[int], r2s: list[float]) -> float:
    """Smallest tested width reaching KNEE_R2, or nan if none does."""
    for width, r2 in zip(widths, r2s):
        if r2 >= KNEE_R2:
            return float(width)
    return float("nan")


def build_arms(args: argparse.Namespace) -> list[Arm]:
    """Every (series pair, provenance) the run scores, in the order it runs them."""
    arms: list[Arm] = []
    for coupling in args.couplings:
        for seed in range(args.seeds):
            data = coupled_logistic(
                n=args.n, r_x=SWEEP_R_X, r_y=SWEEP_R_Y,
                coupling_x_to_y=coupling, seed=seed,
            )
            arms.append(Arm(
                label=f"logistic_c{coupling:.2f}[{seed}]",
                family=f"logistic_c{coupling:.2f}",
                system="logistic", coupling=coupling, seed=seed,
                x=np.asarray(data["x"]), y=np.asarray(data["y"]),
            ))
    for coupling in args.rossler_couplings:
        for seed in range(args.seeds):
            # Generated at subsample times the wanted length so the analysed
            # series is args.n points AFTER subsampling; see the module
            # docstring for why subsampling is not optional here.
            data = rossler_lorenz(n=args.n * args.subsample, coupling=coupling, seed=seed)
            arms.append(Arm(
                label=f"rossler_c{coupling:.1f}[{seed}]",
                family=f"rossler_c{coupling:.1f}",
                system="rossler", coupling=coupling, seed=seed,
                x=np.asarray(data["x"])[:: args.subsample],
                y=np.asarray(data["y"])[:: args.subsample],
            ))
    return arms


def ccm_stats(x: np.ndarray, y: np.ndarray, E: int, seed: int, prefix: str) -> dict:
    """One CCM run flattened into columns; the detection statistic is the max
    of the two directions' rho at the largest library, fixed a priori."""
    result = ccm(x, y, E=E, seed=seed)
    fwd = result.x_causes_y.rho_at_max_lib
    rev = result.y_causes_x.rho_at_max_lib
    return {
        f"{prefix}_E": E,
        f"{prefix}_x_to_y": fwd,
        f"{prefix}_y_to_x": rev,
        f"{prefix}_stat": max(fwd, rev),
        f"{prefix}_conv_x_to_y": result.x_causes_y.is_convergent(),
        f"{prefix}_conv_y_to_x": result.y_causes_x.is_convergent(),
    }


def analyse_arm(arm: Arm, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    """All statistics for one arm: AE curve, PCA curve, CCM, guard flag."""
    t0 = time.time()
    e_x = optimal_embedding_dimension(arm.x, max_E=MAX_E)
    e_y = optimal_embedding_dimension(arm.y, max_E=MAX_E)
    d = e_x + e_y
    states = joint_embedding(arm.x, arm.y, e_x, e_y)
    split = contiguous_split(states, embargo=max(e_x, e_y))
    widths = bottleneck_widths(d)

    ae_curve = [
        autoencoder_r2(split, b, arm.seed, args.epochs, args.patience) for b in widths
    ]
    ae_seconds = time.time() - t0
    pca_curve = [pca_r2(split, b) for b in widths]

    t0 = time.time()
    ccm_fixed = ccm_stats(arm.x, arm.y, E=3, seed=arm.seed, prefix="ccm3")
    ccm_opt = ccm_stats(arm.x, arm.y, E=max(e_x, e_y), seed=arm.seed, prefix="ccmopt")
    ccm_seconds = time.time() - t0

    half = math.ceil(d / 2)
    ae_knee = knee_width(widths, ae_curve)
    pca_knee = knee_width(widths, pca_curve)
    full_r2 = ae_curve[-1]  # widths are sorted, so the last is b = D

    row = {
        "label": arm.label, "family": arm.family, "system": arm.system,
        "coupling": arm.coupling, "seed": arm.seed,
        "e_x": e_x, "e_y": e_y, "d": d, "half_width": half,
        "n_train": len(split.train), "n_val": len(split.val), "n_test": len(split.test),
        "ae_full_r2": full_r2,
        "learned": full_r2 >= GUARD_FULL_WIDTH_R2,
        "ae_knee": ae_knee,
        "ae_deficit": d - ae_knee,
        "ae_r2_half": ae_curve[widths.index(half)],
        "pca_knee": pca_knee,
        "pca_deficit": d - pca_knee,
        "pca_r2_half": pca_curve[widths.index(half)],
        "ae_seconds": ae_seconds, "ccm_seconds": ccm_seconds,
    }
    row.update(ccm_fixed)
    row.update(ccm_opt)

    curve_rows = [
        {
            "label": arm.label, "family": arm.family, "system": arm.system,
            "coupling": arm.coupling, "seed": arm.seed, "d": d,
            "method": method, "b": b, "r2": r2,
        }
        for method, curve in (("ae", ae_curve), ("pca", pca_curve))
        for b, r2 in zip(widths, curve)
    ]
    return row, curve_rows


def separation_table(frame: pd.DataFrame, control_family: str) -> pd.DataFrame:
    """The a-priori verdict: weakest coupled arm versus strongest control arm.

    Guard-flagged rows are excluded from both sides; a cell with no unflagged
    rows reports nan and no verdict.  Every statistic is oriented so that
    coupling should raise it, so the one-sided comparison is the declared one.
    """
    usable = frame[frame.learned]
    control = usable[usable.family == control_family]
    rows = []
    for family in sorted(usable.family.unique()):
        if family == control_family:
            continue
        block = usable[usable.family == family]
        row: dict = {"family": family, "coupling": block.coupling.iloc[0],
                     "n_used": len(block), "n_control": len(control)}
        for stat in STAT_COLUMNS:
            weakest = block[stat].min() if len(block) else float("nan")
            strongest = control[stat].max() if len(control) else float("nan")
            row[f"{stat}_min"] = weakest
            row[f"{stat}_ctrl_max"] = strongest
            row[f"{stat}_sep"] = (
                bool(weakest > strongest)
                if np.isfinite(weakest) and np.isfinite(strongest)
                else None
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run_smoke_asserts(summary: pd.DataFrame, curves: pd.DataFrame) -> None:
    """Mechanics asserts (rule 6); failure raises before any full run is paid for."""
    bad_full = summary[summary.ae_full_r2 <= 0.95]
    assert bad_full.empty, (
        f"full-width r2 <= 0.95 on: {bad_full[['label', 'ae_full_r2']].to_dict('records')}"
    )
    for (label, method), block in curves.groupby(["label", "method"]):
        r2s = block.sort_values("b").r2.to_numpy()
        drops = np.diff(r2s)
        assert (drops >= -SMOKE_MONOTONE_TOL).all(), (
            f"{label} {method}: r2(b) not monotone within {SMOKE_MONOTONE_TOL}: {r2s}"
        )
    print("\nSMOKE ASSERTS PASSED: c=0 identity, full-width r2 > 0.95, monotone r2(b)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000,
                   help="points per analysed series, after burn-in and subsampling")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--couplings", type=float, nargs="+",
                   default=[0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32])
    p.add_argument("--rossler-couplings", type=float, nargs="*", default=[0.0, 2.0])
    p.add_argument("--subsample", type=int, default=10)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--outdir", default="ExpOutput/bottleneck_pair")
    p.add_argument("--smoke", action="store_true",
                   help="tiny run asserting the mechanics before the full run")
    args = p.parse_args()

    if args.smoke:
        args.n = 400
        args.seeds = 1
        args.couplings = [0.0, 0.32]
        args.rossler_couplings = [0.0, 2.0]
        args.outdir = str(Path(args.outdir) / "smoke")

    print("=" * 96)
    print("SWEEP WIRING CHECK: coupling 0 must be the negative control, byte-identical")
    print("=" * 96)
    if not check_zero_coupling_is_control(args.n, args.seeds):
        print("  FAILED -- the sweep varies more than the coupling; aborting.")
        return 1

    arms = build_arms(args)
    print(f"\n{len(arms)} arms\n")

    summary_rows: list[dict] = []
    curve_rows: list[dict] = []
    t_start = time.time()
    for arm in arms:
        row, curves_for_arm = analyse_arm(arm, args)
        summary_rows.append(row)
        curve_rows.extend(curves_for_arm)
        flag = "" if row["learned"] else "  [UNLEARNABLE -- FLAGGED]"
        print(
            f"  {arm.label}: D={row['d']} (E_x={row['e_x']}, E_y={row['e_y']}) "
            f"full r2 {row['ae_full_r2']:.3f} knee {row['ae_knee']:.0f} "
            f"deficit {row['ae_deficit']:.0f} r2@{row['half_width']} "
            f"{row['ae_r2_half']:.3f} | pca knee {row['pca_knee']:.0f} "
            f"r2@{row['half_width']} {row['pca_r2_half']:.3f} | "
            f"ccm3 {row['ccm3_stat']:+.3f} ccmE{row['ccmopt_E']} "
            f"{row['ccmopt_stat']:+.3f} | {row['ae_seconds']:.0f}s+"
            f"{row['ccm_seconds']:.0f}s{flag}"
        )

    summary = pd.DataFrame(summary_rows)
    curves = pd.DataFrame(curve_rows)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / "summary.csv", index=False)
    curves.to_csv(outdir / "curves.csv", index=False)

    fmt = ("display.float_format", "{:.3f}".format, "display.width", 250,
           "display.max_columns", 100)

    print("\n" + "=" * 96)
    print("LEARNABILITY GUARD (rule 4): full-width AE must reach r2 >= "
          f"{GUARD_FULL_WIDTH_R2}")
    print("=" * 96)
    failed = summary[~summary.learned]
    if len(failed):
        print(f"  FLAGGED ({len(failed)} of {len(summary)}) -- excluded from verdicts:")
        for _, r in failed.iterrows():
            print(f"    {r.label}: full-width r2 {r.ae_full_r2:.3f}")
    else:
        print("  No arm flagged: every full-width autoencoder reconstructed its arm.")

    print("\n" + "=" * 96)
    print("PER-FAMILY MEANS (guard-passing rows only)")
    print("=" * 96)
    usable = summary[summary.learned]
    with pd.option_context(*fmt):
        print(usable.groupby("family").agg(
            n=("label", "size"), D=("d", "mean"),
            ae_knee=("ae_knee", "mean"), ae_deficit=("ae_deficit", "mean"),
            ae_r2_half=("ae_r2_half", "mean"),
            pca_knee=("pca_knee", "mean"), pca_r2_half=("pca_r2_half", "mean"),
            ccm3=("ccm3_stat", "mean"), ccmopt=("ccmopt_stat", "mean"),
        ).to_string())

    print("\n" + "=" * 96)
    print("VERDICTS: separation (weakest coupled > strongest control), per statistic")
    print("=" * 96)
    for system, control_family in (("logistic", "logistic_c0.00"),
                                   ("rossler", "rossler_c0.0")):
        block = summary[summary.system == system]
        if block.empty or control_family not in set(block.family):
            print(f"\n  {system}: no control present, skipped")
            continue
        table = separation_table(block, control_family)
        if table.empty:
            print(f"\n  {system}: no coupled families usable")
            continue
        print(f"\n  {system} (control = {control_family})")
        with pd.option_context(*fmt):
            print(table.to_string(index=False))
        table.to_csv(outdir / f"verdicts_{system}.csv", index=False)

    print("\n" + "=" * 96)
    print("WALL CLOCK -- UPPER BOUND UNDER CONTENTION, NOT A CLEAN MEASUREMENT")
    print("=" * 96)
    print(f"  total {time.time() - t_start:.0f}s; per-arm AE and CCM seconds in "
          f"summary.csv")

    if args.smoke:
        run_smoke_asserts(summary, curves)

    print(f"\nwrote {outdir}/summary.csv, curves.csv, verdicts_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
