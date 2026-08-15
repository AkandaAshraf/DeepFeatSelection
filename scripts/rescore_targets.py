"""Re-score every saved detection result against the target it should have used.

Every detection number in the scaling and channel experiments was computed as
*interaction members versus irrelevant features*. The role-share check showed
that at high coupling frequency the models put 0.675 of their attribution mass
on the two MARGINAL causes -- a 6.75x enrichment over their 0.10 share of the
columns -- while irrelevant features got 0.272 against a 0.70 null.

The marginal causes are genuine causes. They are individually informative and
therefore learnable when the sine interaction is not. So a model that abandons
the unlearnable interaction and concentrates on them has behaved correctly, and
scoring it *only* on interaction-versus-irrelevant measures it on the one target
it rightly gave up on, crediting nothing for the part it got right.

Three targets are reported here, all against the irrelevant columns:

* ``interaction`` -- the original, and the harshest;
* ``marginal``    -- the part that stays learnable at every frequency;
* ``causal``      -- interaction and marginal together, which is the honest
  question "does this method find the features that actually drive the target".

Probe channels are re-scored from saved per-feature deltas, so nothing is
retrained. Baselines are recomputed from scratch because their per-feature
scores were never saved -- but they cost seconds, and the systems are seeded so
the data regenerates identically.

    python scripts/rescore_targets.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from deepfeatselect.scaling import oblique_interaction

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scaling_benchmark import (  # noqa: E402
    score_forest, score_mutual_info, score_permutation,
)

TARGETS = {
    "interaction": lambda s: (s.interaction, s.irrelevant),
    "marginal": lambda s: (s.marginal, s.irrelevant),
    "causal": lambda s: (s.interaction | s.marginal, s.irrelevant),
}


def auroc(scores: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> float:
    keep = positive | negative
    y = positive[keep].astype(int)
    v = np.asarray(scores)[keep]
    if len(np.unique(y)) < 2 or np.allclose(v, v[0]) or not np.isfinite(v).all():
        return np.nan
    a = roc_auc_score(y, v)
    return max(a, 1 - a)


def role_masks(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the three masks from the saved role column, in feature order."""
    block = frame.sort_values("feature")
    role = block.role.to_numpy()
    return role == "interaction", role == "marginal", role == "irrelevant"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--deltas", default="ExpOutput/diagnostics/diagnostic_deltas.csv")
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--d", type=int, default=20)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--outdir", default="ExpOutput/rescored")
    args = p.parse_args()

    frame = pd.read_csv(args.deltas)
    if "labels" in frame.columns:
        frame = frame[frame.labels == "real"]
    channels = [c for c in frame.columns if c.startswith("d_")]

    rows = []
    for (freq, seed), block in frame.groupby(["freq", "seed"]):
        inter, marg, irrel = role_masks(block)
        ordered = block.sort_values("feature")

        for name, pick in TARGETS.items():
            class Masks:  # tiny holder so TARGETS can stay declarative
                interaction, marginal, irrelevant = inter, marg, irrel
            positive, negative = pick(Masks)

            for ch in channels:
                # Magnitude, not signed: a channel's direction is not known in
                # advance and orienting it after the fact fits the answer.
                rows.append({
                    "freq": freq, "seed": seed, "target": name,
                    "method": ch.replace("d_", "probe_"),
                    "auroc": auroc(np.abs(ordered[ch].to_numpy()), positive, negative),
                })

    # Baselines were never saved per-feature; the systems are seeded so the
    # identical data regenerates, and these cost seconds rather than trainings.
    for freq in sorted(frame.freq.unique()):
        for seed in sorted(frame.seed.unique()):
            system = oblique_interaction(n=args.n, n_features=args.d, k=args.k,
                                         frequency=float(freq), seed=int(seed))
            baselines = {
                "mutual_info": score_mutual_info(system.x, system.y, seed),
                "random_forest": score_forest(system.x, system.y, seed),
                "permutation": score_permutation(system.x, system.y, seed),
            }
            for name, pick in TARGETS.items():
                positive, negative = pick(system)
                for method, scores in baselines.items():
                    rows.append({"freq": freq, "seed": seed, "target": name,
                                 "method": method,
                                 "auroc": auroc(scores, positive, negative)})
        print(f"  baselines recomputed for freq={freq:g}")

    table = pd.DataFrame(rows)
    summary = (table.groupby(["target", "method", "freq"]).auroc.mean()
               .reset_index())

    print("\n" + "=" * 92)
    print("DETECTION AUROC UNDER EACH TARGET  (chance 0.5)")
    print("=" * 92)
    interesting = ["mutual_info", "random_forest", "permutation",
                   "probe_val_loss", "probe_neural_collapse",
                   "probe_act_pr_norm_data", "probe_act_distinct_frac_data"]
    for target in ("interaction", "marginal", "causal"):
        block = summary[summary.target == target]
        pivot = block.pivot(index="method", columns="freq", values="auroc")
        pivot = pivot.reindex([m for m in interesting if m in pivot.index])
        print(f"\n--- target: {target} "
              f"({'the original, harshest' if target == 'interaction' else ''}"
              f"{'the learnable part' if target == 'marginal' else ''}"
              f"{'the honest question' if target == 'causal' else ''}) ---")
        with pd.option_context("display.float_format", "{:.3f}".format,
                               "display.width", 200):
            print(pivot.to_string())

    print("\n" + "=" * 92)
    print("WHAT THE CORRECTION CHANGES  (causal minus interaction)")
    print("=" * 92)
    wide = summary.pivot_table(index=["method", "freq"], columns="target",
                               values="auroc").reset_index()
    wide["gain"] = wide["causal"] - wide["interaction"]
    gain = wide[wide.method.isin(interesting)].pivot(
        index="method", columns="freq", values="gain")
    gain = gain.reindex([m for m in interesting if m in gain.index])
    with pd.option_context("display.float_format", "{:+.3f}".format,
                           "display.width", 200):
        print(gain.to_string())

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(outdir / "rescored_raw.csv", index=False)
    summary.to_csv(outdir / "rescored_summary.csv", index=False)
    print(f"\nwrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
