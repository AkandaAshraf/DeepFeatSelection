"""Calibrate the maturity stopping rule on synthetic data with a hub.

Phase A of the maturity experiment.  The chamber taught us that LOCO gains
collapse when the model matures enough to re-route a redundant imprint
(Takens: any child reconstructs its driver).  If partial training is to be a
TOOL feature rather than a tuned hyperparameter, the stopping point must be
chosen without real ground truth.  Here it is chosen on synthetic data, where
truth is free, and expressed in three transferable currencies:

* best epoch count (expected NOT to transfer: epochs scale with rows),
* best gradient-step count (steps = epochs x batches/epoch),
* best fraction of the r2 plateau (stop when held-out recreation r2 first
  reaches this fraction of its 50-epoch value).

Phase B (``maturity_sweep.py`` on the chamber) then reports which rule lands
nearest the chamber's actual peak -- scored after the fact, never used for
selection.

The synthetic network is NOT one of the sparse random DAGs that flattered the
method in August.  It is built to mirror the chamber's shape: one hub parent
with six children (imprint redundant six ways, the regime that kills mature
LOCO) plus two single-child parents (unique imprint, the control stratum).

    python scripts/maturity_synthetic.py --epochs-grid 2 5 10 25 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from chamber_detect import contiguous_splits, E  # noqa: E402
from maturity_sweep import loco_trajectories  # noqa: E402
from network_scale import simulate  # noqa: E402
from deepfeatselect.ccm import time_delay_embed  # noqa: E402

# Chamber-shaped graph: node 0 is the hub (six children), 1 and 2 have one
# child each.  In-degree stays <= 2, inside the divergence-safe regime.
HUB_EDGES = [(0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
             (1, 9), (2, 4)]
N_NODES = 10


def embed_at_protocol_E(x: np.ndarray) -> np.ndarray:
    """Per-node embeddings at the chamber protocol's E, so maturity currencies
    transfer without an embedding-dimension confound."""
    mats = [time_delay_embed(x[:, j], E)[0] for j in range(x.shape[1])]
    n = min(len(m) for m in mats)
    return np.hstack([m[:n] for m in mats])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--coupling", type=float, default=0.3)
    p.add_argument("--epochs-grid", type=int, nargs="+",
                   default=[2, 5, 10, 25, 50])
    p.add_argument("--units", type=int, default=32)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--outdir", default="ExpOutput/maturity_synth")
    args = p.parse_args()

    truth = np.zeros((N_NODES, N_NODES), dtype=bool)
    for i, j in HUB_EDGES:
        truth[i, j] = True
    out_degree = truth.sum(axis=1)

    rows = []
    for seed in range(args.seeds):
        x = simulate(args.n, HUB_EDGES, N_NODES, args.coupling, seed)
        z = embed_at_protocol_E(x)
        splits = contiguous_splits(len(z), embargo=E)
        mu, sd = z[splits[0]].mean(axis=0), z[splits[0]].std(axis=0) + 1e-12
        zs = ((z - mu) / sd).astype("float32")
        batches_per_epoch = int(np.ceil((splits[0].stop) / 64))

        t0 = time.time()
        all_gains, all_r2, all_ghosts = loco_trajectories(
            zs, N_NODES, sorted(args.epochs_grid), seed, args.units)
        traj_seconds = time.time() - t0
        plateau = float(all_r2[max(args.epochs_grid)].mean())
        for epochs in sorted(args.epochs_grid):
            gains, full_r2, ghosts = all_gains[epochs], all_r2[epochs], all_ghosts[epochs]
            floor = max(ghosts)
            pos, neg = [], []
            for i in range(N_NODES):
                for j in range(N_NODES):
                    if i == j:
                        continue
                    (pos if truth[i, j] else neg).append(gains[j, i])
            ranks = pd.Series(pos + neg).rank().to_numpy()
            auroc = ((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                     / (len(pos) * len(neg)))
            hub_gain = float(np.mean(
                [gains[j, i] for i, j in HUB_EDGES if out_degree[i] >= 6]))
            few_gain = float(np.mean(
                [gains[j, i] for i, j in HUB_EDGES if out_degree[i] < 6]))
            rows.append({"seed": seed, "epochs": epochs,
                         "steps": epochs * batches_per_epoch,
                         "auroc": float(auroc), "floor": floor,
                         "mean_full_r2": float(full_r2.mean()),
                         "hub_gain": hub_gain, "few_gain": few_gain,
                         "r2_fraction_of_plateau":
                             float(full_r2.mean()) / (plateau + 1e-12),
                         "seconds": traj_seconds})
            r = rows[-1]
            print(f"  seed {seed} epochs={epochs:>3}: auroc {r['auroc']:.3f}  "
                  f"floor {r['floor']:+.3f}  r2 {r['mean_full_r2']:.3f}  "
                  f"hub {r['hub_gain']:+.4f}  few {r['few_gain']:+.4f}  "
                  f"({r['seconds']:.0f}s)")

    frame = pd.DataFrame(rows)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outdir / "synthetic_curve.csv", index=False)

    print("\n" + "=" * 88)
    print("SYNTHETIC MATURITY CURVE (mean over seeds)")
    print("=" * 88)
    mean = frame.groupby("epochs").mean(numeric_only=True)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(mean[["steps", "auroc", "floor", "mean_full_r2",
                    "hub_gain", "few_gain"]].to_string())

    best_epochs = int(mean.auroc.idxmax())
    plateau = mean.mean_full_r2.iloc[-1]
    frac = mean.loc[best_epochs, "mean_full_r2"] / (plateau + 1e-12)
    print(f"\nCALIBRATED RULES (to be applied blind to the chamber):")
    print(f"  rule A  epochs        = {best_epochs}")
    print(f"  rule B  gradient steps = {mean.loc[best_epochs, 'steps']:.0f}")
    print(f"  rule C  stop at r2 fraction of plateau = {frac:.2f}")
    print(f"\nwrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
