"""The three confirmed channels against the standard algorithms.

Every scaling comparison so far scored the deep probe by how much validation
loss rose under ablation. The shuffled-label control showed that channel is
empty at high nonlinearity -- 0.522 real against 0.545 shuffled -- while three
representation-geometry channels carry real signal that collapses to chance
when the labels are destroyed:

    neural collapse            0.781 real / 0.504 shuffled
    participation ratio        0.732 / 0.549
    distinct activation sets   0.728 / 0.554

So the probe has been judged on its weakest reading. This is the rerun with the
channels that survived the control, against mutual information, random forest
and permutation importance on the same systems.

Two ways of combining them:

* **unsupervised** -- z-score the absolute delta of each channel within a run
  and average. Magnitude rather than signed direction, because a channel's sign
  is not known before the fact and orienting it post hoc would be fitting the
  answer. Needs no labels, so it is what could actually be deployed.
* **supervised** -- a logistic model fitted on one nonlinearity level and
  tested on another. Transfer across conditions rather than within, since a
  within-condition fit can memorise which column index is which.

    python scripts/channel_vs_baselines.py --frequencies 2 4 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from deepfeatselect.diagnostics import neural_collapse
from deepfeatselect.model import build_model
from deepfeatselect.netstats import activation_stats
from deepfeatselect.scaling import oblique_interaction

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scaling_benchmark import (  # noqa: E402
    score_forest, score_mutual_info, score_permutation,
)

CHANNELS = ["neural_collapse", "pr_norm", "distinct_frac"]


def train_and_measure(x_tr, y_tr, x_va, y_va, args, seed) -> dict[str, float]:
    keras.utils.set_random_seed(seed)
    model = build_model(
        n_columns=x_tr.shape[1], groups=np.arange(x_tr.shape[1]), n_classes=2,
        task="binary", l1_gate=0.0, dropout=args.dropout, noise=0.0,
        hidden_units=args.width, n_hidden_layers=2, l2_dense=1e-3,
        learning_rate=3e-3,
    )
    stopper = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True)
    model.fit(x_tr, y_tr.astype("float32").reshape(-1, 1),
              validation_data=(x_va, y_va.astype("float32").reshape(-1, 1)),
              epochs=args.epochs, batch_size=128, shuffle=True, verbose=0,
              callbacks=[stopper])

    probe = x_va[:min(256, len(x_va))]
    acts = activation_stats(model, probe, prefix="d")
    return {
        "neural_collapse": neural_collapse(model, x_va, y_va),
        "pr_norm": acts["act_pr_norm_d"],
        "distinct_frac": acts["act_distinct_frac_d"],
    }


def probe_scores(x, y, args, seed) -> pd.DataFrame:
    """Absolute change in each channel when a feature is zeroed."""
    x_tr, x_va, y_tr, y_va = train_test_split(
        x, y, test_size=0.3, random_state=seed, stratify=y)
    base = train_and_measure(x_tr, y_tr, x_va, y_va, args, seed)

    rows = []
    for j in range(x.shape[1]):
        tr, va = x_tr.copy(), x_va.copy()
        tr[:, j] = 0.0
        va[:, j] = 0.0
        stats = train_and_measure(tr, y_tr, va, y_va, args, seed)
        rows.append({c: abs(stats[c] - base[c]) for c in CHANNELS})
    return pd.DataFrame(rows)


def combine(frame: pd.DataFrame) -> np.ndarray:
    """Unsupervised ensemble: mean of within-run z-scores of |delta|.

    Magnitude, not signed direction. A channel's sign is not known in advance
    and choosing it to maximise the result would be fitting the answer, which is
    the mistake the effect-first metric caught earlier in this project.
    """
    z = frame[CHANNELS].apply(
        lambda s: (s - s.mean()) / (s.std() + 1e-12), axis=0)
    return z.mean(axis=1).to_numpy()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=20)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--frequencies", type=float, nargs="+", default=[2.0, 4.0, 8.0])
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--outdir", default="ExpOutput/channels")
    args = p.parse_args()

    rows, features = [], []
    for freq in args.frequencies:
        for seed in range(args.seeds):
            system = oblique_interaction(n=args.n, n_features=args.d, k=args.k,
                                         frequency=freq, seed=seed)
            x, y = system.x, system.y
            positive, negative = system.interaction, system.irrelevant
            keep = positive | negative
            truth = positive[keep].astype(int)

            def auc(scores):
                v = scores[keep]
                if np.allclose(v, v[0]) or not np.isfinite(v).all():
                    return np.nan
                a = roc_auc_score(truth, v)
                return max(a, 1 - a)

            probe = probe_scores(x, y, args, seed)
            entry = {"freq": freq, "seed": seed}
            for channel in CHANNELS:
                entry[f"probe_{channel}"] = auc(probe[channel].to_numpy())
            entry["probe_combined"] = auc(combine(probe))
            entry["mutual_info"] = auc(score_mutual_info(x, y, seed))
            entry["random_forest"] = auc(score_forest(x, y, seed))
            entry["permutation"] = auc(score_permutation(x, y, seed))
            rows.append(entry)

            # Kept for the supervised transfer test below.
            block = probe.copy()
            block["freq"], block["seed"] = freq, seed
            block["is_interaction"] = positive.astype(int)
            block["keep"] = keep
            features.append(block)
            print(f"  freq={freq} seed={seed} done")

    table = pd.DataFrame(rows)
    summary = table.groupby("freq").mean(numeric_only=True).drop(columns=["seed"])

    print("\n" + "=" * 96)
    print("CONFIRMED CHANNELS VERSUS THE STANDARD ALGORITHMS")
    print("=" * 96)
    print("  detection AUROC, interaction members against irrelevant (0.5 = chance)\n")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 220):
        print(summary.to_string())

    # Supervised combination, tested across nonlinearity levels rather than
    # within one: a within-condition fit can learn column identity instead.
    pooled = pd.concat(features, ignore_index=True)
    pooled = pooled[pooled.keep]
    if len(args.frequencies) > 1:
        print("\n" + "=" * 96)
        print("SUPERVISED COMBINATION, TRAINED ON ONE FREQUENCY AND TESTED ON ANOTHER")
        print("=" * 96)
        for train_f in args.frequencies:
            for test_f in args.frequencies:
                if train_f == test_f:
                    continue
                tr = pooled[pooled.freq == train_f]
                te = pooled[pooled.freq == test_f]
                if tr.is_interaction.nunique() < 2 or te.is_interaction.nunique() < 2:
                    continue
                model = make_pipeline(StandardScaler(),
                                      LogisticRegression(C=1.0, max_iter=2000))
                model.fit(tr[CHANNELS], tr.is_interaction)
                scores = model.predict_proba(te[CHANNELS])[:, 1]
                a = roc_auc_score(te.is_interaction, scores)
                print(f"  f={train_f:g} -> f={test_f:g}: AUROC {max(a, 1 - a):.3f}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(outdir / "channel_vs_baselines.csv", index=False)
    pooled.to_csv(outdir / "channel_features.csv", index=False)
    print(f"\nwrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
