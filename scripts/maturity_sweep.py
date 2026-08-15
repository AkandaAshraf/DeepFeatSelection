"""Does training maturity kill the LOCO gain, as Takens says it must?

The hypothesis under test: a fully trained model extracts the
cause from ANY of its effects -- Takens guarantees a single child's delay
embedding reconstructs the driver -- so when a cause has several children its
imprint is redundant across them, and a mature model asked to recreate the
cause without one child simply reads the same imprint from another.  The LOCO
difference collapses exactly when training succeeds.  An immature model is a
restricted function class (Proposition 2), which is the only reason the gain
was ever visible: the method worked because of limited training, not despite
it.

DESIGN: one model per LOCO fit, trained ONCE to the last grid
epoch, with its held-out recreation r2 CHECKPOINTED at every grid epoch along
the way.  Every maturity level is therefore the same model earlier or later in
its own trajectory -- no retraining, no restart variance between levels -- and
afterwards the best epoch is correlated against observables (fraction of the
r2 plateau, gradient steps) to find what optimal maturity tracks, since a raw
epoch count cannot transfer between datasets of different sizes.

Predictions, fixed before running:

1. NON-MONOTONE: LOCO edge-detection AUROC rises out of noise, peaks at
   partial maturity, decays as the model matures.  The ghost floor falls
   monotonically (optimisation noise only shrinks).  A curve that only rises
   or only falls refutes the hypothesis.
2. REDUNDANCY-ORDERED COLLAPSE (the mechanism): gains on edges from HUB
   parents (out-degree >= 6 among varying variables) collapse with maturity
   FASTER than gains on edges from single-child parents.  A uniform collapse
   would implicate plain overfitting, not the Takens mechanism.

Early stopping is deliberately absent: maturity is the independent variable.
Orientation is the protocol's: the score for u -> v is the gain of v in
recreating u.

    python scripts/maturity_sweep.py --epochs-grid 2 5 10 25 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from chamber_detect import (  # noqa: E402
    ACTUATORS, contiguous_splits, embed_columns, load_experiment, r2_of,
    var_cols, varying_variables, E,
)


class _CheckpointR2(keras.callbacks.Callback):
    """Record held-out recreation r2 at chosen epochs of ONE training run."""

    def __init__(self, src_test: np.ndarray, dst_test: np.ndarray,
                 grid: list[int]):
        super().__init__()
        self.src_test, self.dst_test = src_test, dst_test
        self.grid = set(grid)
        self.r2_at: dict[int, float] = {}

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) in self.grid:
            pred = self.model.predict(self.src_test, verbose=0)
            self.r2_at[epoch + 1] = r2_of(pred, self.dst_test)


def fit_checkpointed(src, dst, splits, seed: int, units: int,
                     grid: list[int]) -> dict[int, float]:
    """One training run; r2 at every grid epoch of its own trajectory."""
    tr, va, te = splits
    keras.utils.set_random_seed(seed)
    m = keras.Sequential([
        keras.layers.Input(shape=(src.shape[1],)),
        keras.layers.Dense(units, activation="tanh"),
        keras.layers.Dense(units, activation="tanh"),
        keras.layers.Dense(dst.shape[1]),
    ])
    m.compile(loss="mse", optimizer=keras.optimizers.Adam(3e-3))
    cb = _CheckpointR2(src[te], dst[te], grid)
    m.fit(src[tr], dst[tr], epochs=max(grid), batch_size=64, shuffle=True,
          verbose=0, callbacks=[cb])
    return cb.r2_at


def loco_trajectories(zs: np.ndarray, V: int, grid: list[int], seed: int,
                      units: int) -> tuple[dict, dict, dict]:
    """LOCO gains per epoch from checkpointed runs.

    Returns ``gains[e][v, u]`` (gain of v recreating u at epoch e),
    ``full_r2[e]`` per target, and ``ghosts[e]`` -- the ghost-source null
    draws, one per target, measured by the same machinery.
    """
    splits = contiguous_splits(len(zs), embargo=E)
    rng = np.random.default_rng(seed + 7331)
    gains = {e: np.full((V, V), np.nan) for e in grid}
    full_r2 = {e: np.empty(V) for e in grid}
    ghosts = {e: [] for e in grid}

    for u in range(V):
        others = [v for v in range(V) if v != u]
        donor = int(rng.choice(others))
        shift = int(rng.integers(len(zs) // 4, 3 * len(zs) // 4))
        ghost = np.roll(zs[:, var_cols(donor)], shift, axis=0)
        dst = zs[:, var_cols(u)]

        def run(sources, extra=None) -> dict[int, float]:
            parts = [zs[:, var_cols(v)] for v in sources]
            if extra is not None:
                parts.append(extra)
            return fit_checkpointed(np.hstack(parts), dst, splits, seed,
                                    units, grid)

        base = run(others, extra=ghost)
        no_ghost = run(others, extra=None)
        for e in grid:
            full_r2[e][u] = base[e]
            ghosts[e].append(base[e] - no_ghost[e])
        for v in others:
            without = run([w for w in others if w != v], extra=ghost)
            for e in grid:
                gains[e][v, u] = base[e] - without[e]

    return gains, full_r2, ghosts


def score_epoch(gains_e: np.ndarray, truth: np.ndarray, names: list[str],
                floor: float) -> tuple[float, float, int]:
    """AUROC, share of true edges above the floor, actuator false positives."""
    V = len(names)
    pos, neg, act_fp = [], [], 0
    for i in range(V):
        for j in range(V):
            if i == j:
                continue
            score = gains_e[j, i]   # protocol orientation: u->v = gains[v,u]
            (pos if truth[i, j] else neg).append(score)
            if not truth[i, j] and names[j] in ACTUATORS and score > floor:
                act_fp += 1
    ranks = pd.Series(pos + neg).rank().to_numpy()
    auroc = ((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
             / (len(pos) * len(neg)))
    return float(auroc), float(np.mean([s > floor for s in pos])), act_fp


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="actuators_random_walk_2")
    p.add_argument("--root", default="Data/causalchamber")
    p.add_argument("--epochs-grid", type=int, nargs="+",
                   default=[2, 5, 10, 25, 50])
    p.add_argument("--units", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--difference", action="store_true",
                   help="First-difference every series before embedding. The "
                        "chamber's actuators are random walks; on levels, a "
                        "circular-shift ghost still shares their drift and the "
                        "null floor rises with maturity instead of falling. "
                        "Differencing makes the walks stationary increments, "
                        "which restores the ghost null's validity and gives "
                        "recreation a stationary target.")
    p.add_argument("--outdir", default="ExpOutput/maturity")
    args = p.parse_args()
    grid = sorted(args.epochs_grid)

    frame, graph = load_experiment(args.experiment, args.root)
    names = varying_variables(frame, graph)
    if args.difference:
        frame = frame.copy()
        frame[names] = frame[names].diff()
        frame = frame.iloc[1:]
        print("series first-differenced before embedding")
    V = len(names)
    z = embed_columns(frame, names)
    splits = contiguous_splits(len(z), embargo=E)
    mu, sd = z[splits[0]].mean(axis=0), z[splits[0]].std(axis=0) + 1e-12
    zs = ((z - mu) / sd).astype("float32")
    batches_per_epoch = int(np.ceil(splits[0].stop / 64))

    truth = np.array([[bool(graph.loc[u, v]) for v in names] for u in names])
    out_degree = {u: int(truth[i].sum()) for i, u in enumerate(names)}
    print(f"{args.experiment}: {V} varying variables, {int(truth.sum())} "
          f"true edges, {batches_per_epoch} batches/epoch")

    t0 = time.time()
    gains, full_r2, ghosts = loco_trajectories(zs, V, grid, args.seed,
                                               args.units)
    print(f"LOCO trajectories in {time.time()-t0:.0f}s "
          f"({V * (V + 1)} checkpointed runs)")

    plateau = float(full_r2[grid[-1]].mean())
    rows, edge_rows = [], []
    for e in grid:
        floor = max(ghosts[e])
        auroc, above, act_fp = score_epoch(gains[e], truth, names, floor)
        rows.append({"epochs": e, "steps": e * batches_per_epoch,
                     "auroc": auroc, "floor": floor,
                     "edges_above_floor": above, "actuator_fp": act_fp,
                     "mean_full_r2": float(full_r2[e].mean()),
                     "r2_fraction_of_plateau":
                         float(full_r2[e].mean()) / (plateau + 1e-12)})
        for i in range(V):
            for j in range(V):
                if truth[i, j]:
                    edge_rows.append({"epochs": e, "parent": names[i],
                                      "child": names[j],
                                      "parent_out_degree": out_degree[names[i]],
                                      "gain": gains[e][j, i]})

    curve = pd.DataFrame(rows)
    edges = pd.DataFrame(edge_rows)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(outdir / "maturity_curve.csv", index=False)
    edges.to_csv(outdir / "edge_gains.csv", index=False)

    print("\n" + "=" * 88)
    print("PREDICTION 1: NON-MONOTONE AUROC, MONOTONE FLOOR")
    print("=" * 88)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(curve.to_string(index=False))

    print("\n" + "=" * 88)
    print("PREDICTION 2: HUB-PARENT GAINS COLLAPSE FIRST")
    print("=" * 88)
    edges["group"] = np.where(edges.parent_out_degree >= 6, "hub_parent",
                              "few_children")
    pivot = edges.pivot_table(index="epochs", columns="group", values="gain",
                              aggfunc="mean")
    if {"hub_parent", "few_children"} <= set(pivot.columns):
        pivot["hub_over_few"] = (pivot.hub_parent
                                 / pivot.few_children.replace(0, np.nan))
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(pivot.to_string())

    best = curve.loc[curve.auroc.idxmax()]
    print("\nWHAT THE BEST EPOCH CORRELATES TO (for the transferable rule):")
    print(f"  best epochs {int(best.epochs)}  =  {int(best.steps)} steps  =  "
          f"r2 fraction {best.r2_fraction_of_plateau:.2f} of plateau")
    print(f"\nwrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
