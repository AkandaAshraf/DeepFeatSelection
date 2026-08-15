"""Can the generative half of an autoencoder recreate a masked causal channel?

The discriminator, stated directly: with the causal variable in place,
recreation of the original data is better than without it.  On trajectories the
strong form of that is Takens' imprint -- if X drives Y, then X is recoverable
from Y's delay embedding alone, so a decoder given only Y's channels should be
able to *recreate the masked X channel*, and for an uncoupled pair it should
not.  That is cross-mapping implemented as a masked autoencoder: CCM's quantity,
estimated by a learned global map instead of local simplex averaging.

Phase 1 of the bottleneck work did not test this; it swept bottleneck widths and
its deficit turned out to be an artefact of the embedding-dimension selector
(E_y rose under coupling while the knee never moved).  The repair applied here:
the embedding dimension is FIXED per family -- identical for coupled, control
and noise arms -- so nothing varies between arms but the data.

Three training conditions per arm, each its own small model so the "without"
model does its best without X rather than being evaluated off-distribution:

* ``full``    input = [embed(x), embed(y)], reconstruct both.  Guard: this must
              be near-perfect or the arm is unlearnable.  Also provides the
              "with X" side of the inclusion contrast.
* ``mask_x``  input = Y channels only (X zeroed), reconstruct both.  The R^2 of
              the recreated X part is the cross-map readout; the R^2 of the Y
              part is the "without X" side of the inclusion contrast.
* ``mask_y``  the converse.

STATISTICS, FIXED BEFORE RUNNING:

* PRIMARY  ``xmap_r2`` = max over the two masked directions of the held-out R^2
  of the recreated masked part.  Both directions always computed; max is taken
  for every arm including controls, so the statistic is symmetric by
  construction and nothing is flipped post hoc.
* SECONDARY ``include_gain`` = max over targets of r2(target | full) minus
  r2(target | cause masked).  Reported alongside, never promoted.
* Verdict per coupling: separation -- weakest coupled arm must exceed the
  strongest control arm.
* Guards: full-condition reconstruction R^2 >= 0.95; a white-noise pair must
  show xmap_r2 near zero (leakage detector); the c=0 logistic arm must be
  byte-identical to independent_logistic (the r_y=3.7 protocol identity).

Baselines on identical data and splits: kNN cross-map (predict X embedding from
Y embedding, k=5) -- if the AE does not beat this, depth bought nothing -- and
CCM at fixed and optimal E as the incumbent.

    python scripts/masked_recreation.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

from deepfeatselect.ccm import ccm, optimal_embedding_dimension, time_delay_embed
from deepfeatselect.synthetic import (
    coupled_logistic,
    independent_logistic,
    rossler_lorenz,
)

TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2

# Fixed embedding dimension per family -- identical across coupled, control and
# noise arms.  This is the repair for the phase-1 confound.
E_LOGISTIC = 4
E_ROSSLER = 6

FULL_RECON_GUARD = 0.95


def embed_pair(x: np.ndarray, y: np.ndarray, E: int) -> tuple[np.ndarray, int]:
    """Time-aligned joint state [embed(x), embed(y)], and the per-series width."""
    mx, tx = time_delay_embed(x, E)
    my, ty = time_delay_embed(y, E)
    n = min(len(mx), len(my))
    return np.hstack([mx[:n], my[:n]]).astype("float32"), E


def contiguous_splits(n: int, embargo: int) -> tuple[slice, slice, slice]:
    """Train / early-stop / test as disjoint contiguous segments.

    Embedded states overlap in time, so an embargo of the embedding span is
    dropped at each seam; a shuffled split would put near-duplicate states on
    both sides and report memorisation as generalisation.
    """
    a = int(TRAIN_FRACTION * n)
    b = int((TRAIN_FRACTION + VAL_FRACTION) * n)
    return slice(0, a - embargo), slice(a, b - embargo), slice(b, n)


def part_r2(pred: np.ndarray, truth: np.ndarray, cols: slice) -> float:
    """R^2 of one channel's columns, against that part's own variance."""
    err = float(np.mean((pred[:, cols] - truth[:, cols]) ** 2))
    var = float(np.var(truth[:, cols]))
    return 1.0 - err / (var + 1e-12)


def fit_condition(z: np.ndarray, mask: slice | None, E: int, seed: int,
                  args) -> dict[str, float]:
    """One small autoencoder under one input condition; held-out part R^2s.

    The input under a mask condition has the masked channel zeroed at train AND
    test time, so the model is trained to do its best without that channel --
    evaluating the full model on masked input would instead measure
    off-distribution robustness, which is not the question.
    """
    keras.utils.set_random_seed(seed)
    D = z.shape[1]
    tr, va, te = contiguous_splits(len(z), embargo=E)

    mu, sd = z[tr].mean(axis=0), z[tr].std(axis=0) + 1e-12
    zs = (z - mu) / sd
    inp = zs.copy()
    if mask is not None:
        inp[:, mask] = 0.0

    model = keras.Sequential([
        keras.layers.Input(shape=(D,)),
        keras.layers.Dense(args.units, activation="tanh"),
        keras.layers.Dense(args.units, activation="tanh"),
        keras.layers.Dense(D),
    ])
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(3e-3))
    stopper = keras.callbacks.EarlyStopping(monitor="val_loss",
                                            patience=args.patience,
                                            restore_best_weights=True)
    model.fit(inp[tr], zs[tr], validation_data=(inp[va], zs[va]),
              epochs=args.epochs, batch_size=64, shuffle=True, verbose=0,
              callbacks=[stopper])
    pred = model.predict(inp[te], verbose=0)

    x_cols, y_cols = slice(0, E), slice(E, D)
    return {"r2_x_part": part_r2(pred, zs[te], x_cols),
            "r2_y_part": part_r2(pred, zs[te], y_cols),
            "r2_full": part_r2(pred, zs[te], slice(0, D))}


def knn_xmap(z: np.ndarray, E: int) -> dict[str, float]:
    """kNN recreation of each masked channel from the other, same splits.

    This is cross-mapping with a local estimator and the honest cheap baseline:
    the AE's global map has to beat it to justify training anything.
    """
    D = z.shape[1]
    tr, _, te = contiguous_splits(len(z), embargo=E)
    x_cols, y_cols = slice(0, E), slice(E, D)
    out = {}
    for name, src, dst in (("knn_x_from_y", y_cols, x_cols),
                           ("knn_y_from_x", x_cols, y_cols)):
        m = KNeighborsRegressor(n_neighbors=5)
        m.fit(z[tr, src], z[tr, dst])
        pred = m.predict(z[te, src])
        err = float(np.mean((pred - z[te, dst]) ** 2))
        out[name] = 1.0 - err / (float(np.var(z[te, dst])) + 1e-12)
    return out


def run_arm(label: str, family: str, coupling: float, x: np.ndarray,
            y: np.ndarray, E: int, seed: int, args) -> dict:
    z, E = embed_pair(np.asarray(x, dtype=np.float64),
                      np.asarray(y, dtype=np.float64), E)
    D = z.shape[1]
    t0 = time.time()

    full = fit_condition(z, None, E, seed, args)
    mask_x = fit_condition(z, slice(0, E), E, seed, args)
    mask_y = fit_condition(z, slice(E, D), E, seed, args)
    ae_seconds = time.time() - t0

    knn = knn_xmap(z, E)

    t0 = time.time()
    r3 = ccm(x, y, E=3, seed=seed)
    e_opt = max(optimal_embedding_dimension(x),
                optimal_embedding_dimension(y))
    ro = ccm(x, y, E=e_opt, seed=seed)
    ccm_seconds = time.time() - t0

    return {
        "label": label, "family": family, "coupling": coupling, "seed": seed,
        "E": E, "D": D,
        # Guard: the arm is only reportable if the unmasked model learned.
        "full_r2": full["r2_full"], "learned": full["r2_full"] >= FULL_RECON_GUARD,
        # Primary: recreate the masked channel.  Max of directions, always.
        "xmap_x_from_y": mask_x["r2_x_part"],
        "xmap_y_from_x": mask_y["r2_y_part"],
        "xmap_r2": max(mask_x["r2_x_part"], mask_y["r2_y_part"]),
        # Secondary: the inclusion contrast on the surviving channel.
        "include_gain_y": full["r2_y_part"] - mask_x["r2_y_part"],
        "include_gain_x": full["r2_x_part"] - mask_y["r2_x_part"],
        "include_gain": max(full["r2_y_part"] - mask_x["r2_y_part"],
                            full["r2_x_part"] - mask_y["r2_x_part"]),
        # Baselines.
        "knn_xmap_r2": max(knn["knn_x_from_y"], knn["knn_y_from_x"]),
        "ccm3_rho": max(r3.x_causes_y.rho_at_max_lib,
                        r3.y_causes_x.rho_at_max_lib),
        "ccmopt_rho": max(ro.x_causes_y.rho_at_max_lib,
                          ro.y_causes_x.rho_at_max_lib),
        "ccmopt_E": e_opt,
        "ae_seconds": ae_seconds, "ccm_seconds": ccm_seconds,
    }


def build_arms(args) -> list[tuple[str, str, float, np.ndarray, np.ndarray, int]]:
    arms = []
    for seed in range(args.seeds):
        for c in args.couplings:
            sys_ = coupled_logistic(n=args.n, r_x=3.8, r_y=3.7,
                                    coupling_x_to_y=c, seed=seed)
            if c == 0.0:
                # The protocol identity: c=0 must BE the negative control.
                ctrl = independent_logistic(n=args.n, seed=seed)
                assert (np.array_equal(sys_["x"], ctrl["x"])
                        and np.array_equal(sys_["y"], ctrl["y"])), \
                    "c=0 arm is not byte-identical to independent_logistic"
            arms.append((f"logistic_c{c:g}[{seed}]", "logistic", c,
                         np.asarray(sys_["x"]), np.asarray(sys_["y"]),
                         E_LOGISTIC))
        for c in (0.0, 2.0):
            rl = rossler_lorenz(n=args.n * args.subsample // 1, coupling=c,
                                seed=seed)
            arms.append((f"rossler_c{c:g}[{seed}]", "rossler", c,
                         np.asarray(rl["x"])[::args.subsample],
                         np.asarray(rl["y"])[::args.subsample],
                         E_ROSSLER))
        rng = np.random.default_rng(seed + 990)
        arms.append((f"white_noise[{seed}]", "noise", 0.0,
                     rng.standard_normal(args.n), rng.standard_normal(args.n),
                     E_LOGISTIC))
    return arms


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--couplings", type=float, nargs="+",
                   default=[0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--units", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--subsample", type=int, default=10)
    p.add_argument("--outdir", default="ExpOutput/masked_ae")
    args = p.parse_args()

    rows = []
    for label, family, coupling, x, y, E in build_arms(args):
        row = run_arm(label, family, coupling, x, y, E,
                      seed=int(label.split("[")[1][0]), args=args)
        rows.append(row)
        print(f"  {label:<22} xmap {row['xmap_r2']:+.3f}  include "
              f"{row['include_gain']:+.3f}  knn {row['knn_xmap_r2']:+.3f}  "
              f"ccm {row['ccmopt_rho']:+.3f}  full {row['full_r2']:.3f}  "
              f"({row['ae_seconds']:.0f}s)")

    frame = pd.DataFrame(rows)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outdir / "masked_recreation.csv", index=False)

    print("\n" + "=" * 100)
    print("SEPARATION: weakest coupled arm must exceed strongest control arm")
    print("=" * 100)
    for family, ctrl_mask in (("logistic", (frame.family == "logistic")
                               & (frame.coupling == 0.0)),
                              ("rossler", (frame.family == "rossler")
                               & (frame.coupling == 0.0))):
        ctrl = frame[ctrl_mask]
        if ctrl.empty:
            continue
        print(f"\n  {family}  (controls: n={len(ctrl)}; noise arms shown last)")
        for stat in ("xmap_r2", "include_gain", "knn_xmap_r2", "ccmopt_rho"):
            ceiling = ctrl[stat].max()
            cells = []
            fam = frame[(frame.family == family) & (frame.coupling > 0)]
            for c in sorted(fam.coupling.unique()):
                vals = fam[fam.coupling == c][stat]
                cells.append(f"c={c:g}:{'SEP' if vals.min() > ceiling else '---'}")
            print(f"    {stat:<14} ctrl_max {ceiling:+.3f}   " + "  ".join(cells))
        bad = frame[(frame.family == family) & ~frame.learned]
        if len(bad):
            print(f"    UNLEARNABLE ARMS (excluded from interpretation): "
                  f"{list(bad.label)}")

    noise = frame[frame.family == "noise"]
    print("\n  LEAKAGE CONTROL (white noise; xmap_r2 must be ~0):")
    print(f"    xmap_r2 {noise.xmap_r2.min():+.3f} .. {noise.xmap_r2.max():+.3f}"
          f"   knn {noise.knn_xmap_r2.max():+.3f}   ccm {noise.ccmopt_rho.max():+.3f}")

    print(f"\nwrote {outdir}/masked_recreation.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
