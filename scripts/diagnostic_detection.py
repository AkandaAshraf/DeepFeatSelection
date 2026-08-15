"""Do internal diagnostics detect an interaction better than validation loss?

Two questions, both answered from the same runs.

**Did the networks learn?** The oblique sweep reported validation losses of 0.49
to 0.63 against a noise floor near 0.42 -- better than the chance level of 0.69,
but far from the floor, so "partially" was as much as could be said. Spectral,
sharpness and neural-collapse diagnostics give a sharper answer, and every arm
gets a pass/fail rather than being averaged in regardless.

**Is loss the best deprivation channel?** Every scaling result so far scored an
ablation by how much validation loss rose. But a network deprived of a feature
it needs may reorganise its representation before its loss moves, which is what
the companion work on deprivation signatures found: a feature invisible in the
loss showed up at |z| = 3.4 in participation ratio. Here each diagnostic is used
as its own detection channel and scored against ground truth.

    python scripts/diagnostic_detection.py --frequencies 2 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from deepfeatselect.diagnostics import learned_well
from deepfeatselect.model import build_model
from deepfeatselect.netstats import activation_stats, weight_stats
from deepfeatselect.scaling import oblique_interaction

CHANCE = float(np.log(2))


def train_one(x_tr, y_tr, x_va, y_va, width, epochs, dropout, patience, seed):
    keras.utils.set_random_seed(seed)
    model = build_model(
        n_columns=x_tr.shape[1], groups=np.arange(x_tr.shape[1]), n_classes=2,
        task="binary", l1_gate=0.0, dropout=dropout, noise=0.0,
        hidden_units=width, n_hidden_layers=2, l2_dense=1e-3, learning_rate=3e-3,
    )
    stopper = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True)
    history = model.fit(
        x_tr, y_tr.astype("float32").reshape(-1, 1),
        validation_data=(x_va, y_va.astype("float32").reshape(-1, 1)),
        epochs=epochs, batch_size=128, shuffle=True, verbose=0, callbacks=[stopper])
    return model, float(np.min(history.history["val_loss"]))


def measure(model, val_loss, x_va, y_va, seed) -> dict[str, float]:
    """Every diagnostic channel for one trained network."""
    out = learned_well(model, x_va, y_va.astype("float32").reshape(-1, 1),
                       val_loss, seed=seed)
    out.update(weight_stats(model))
    probe = x_va[:min(256, len(x_va))]
    out.update(activation_stats(model, probe, prefix="data"))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=20)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--frequencies", type=float, nargs="+", default=[2.0, 8.0])
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--label-modes", nargs="+", default=["real", "shuffled"],
                   choices=["real", "shuffled"],
                   help="shuffled is the control: same inputs, no signal")
    p.add_argument("--outdir", default="ExpOutput/diagnostics")
    args = p.parse_args()

    rows, quality = [], []
    for freq in args.frequencies:
      for label_mode in args.label_modes:
        for seed in range(args.seeds):
            system = oblique_interaction(n=args.n, n_features=args.d, k=args.k,
                                         frequency=freq, seed=seed)
            labels = system.y
            if label_mode == "shuffled":
                # The control that decides whether a diagnostic is reading
                # learned structure or the input distribution. Permuting labels
                # leaves x untouched -- same marginals, same covariance, same
                # everything the inputs could contribute -- while removing all
                # information about the target. A channel that still separates
                # interaction from irrelevant here is not detecting causality;
                # it is responding to the act of zeroing a column.
                labels = np.random.default_rng(seed + 7919).permutation(labels)
            x_tr, x_va, y_tr, y_va = train_test_split(
                system.x, labels, test_size=0.3, random_state=seed,
                stratify=labels)

            full_model, full_loss = train_one(
                x_tr, y_tr, x_va, y_va, args.width, args.epochs,
                args.dropout, args.patience, seed)
            base = measure(full_model, full_loss, x_va, y_va, seed)
            quality.append({"freq": freq, "labels": label_mode, "seed": seed,
                            "arm": "full", **base})

            for j in range(args.d):
                tr, va = x_tr.copy(), x_va.copy()
                tr[:, j] = 0.0
                va[:, j] = 0.0
                model, loss = train_one(tr, y_tr, va, y_va, args.width,
                                        args.epochs, args.dropout, args.patience, seed)
                stats = measure(model, loss, va, y_va, seed)
                row = {"freq": freq, "labels": label_mode, "seed": seed, "feature": j,
                       "role": ("interaction" if system.interaction[j] else
                                "marginal" if system.marginal[j] else "irrelevant")}
                # Paired delta against the full model from the same seed.
                row.update({f"d_{k}": stats[k] - base[k] for k in stats
                            if k in base and np.isfinite(stats[k]) and np.isfinite(base[k])})
                rows.append(row)
            print(f"  freq={freq} labels={label_mode} seed={seed} done "
                  f"(full val_loss {full_loss:.3f}, collapse {base['neural_collapse']:.3f})")

    frame = pd.DataFrame(rows)
    qframe = pd.DataFrame(quality)

    print("\n" + "=" * 92)
    print("DID THE NETWORKS LEARN?  (full models only)")
    print("=" * 92)
    cols = ["freq", "seed", "val_loss", "alpha", "effective_rank",
            "sharpness", "neural_collapse", "verdict"]
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(qframe[cols].to_string(index=False))
    print(f"\n  chance loss {CHANCE:.3f}; alpha 2-4 indicates a layer that has")
    print("  captured structure, large alpha means still near random;")
    print("  neural_collapse below 1 means the classes separated in the features.")

    channels = [c for c in frame.columns if c.startswith("d_")]
    print("\n" + "=" * 92)
    print("EACH DIAGNOSTIC AS A DETECTION CHANNEL (AUROC, chance 0.5)")
    print("=" * 92)

    scores = []
    for freq in args.frequencies:
        for mode in args.label_modes:
            block = frame[(frame.freq == freq) & (frame.labels == mode)]
            keep = block[block.role.isin({"interaction", "irrelevant"})]
            if keep.empty:
                continue
            y = (keep.role == "interaction").astype(int)
            for ch in channels:
                v = keep[ch].to_numpy()
                if np.allclose(v, v[0]) or not np.isfinite(v).all():
                    continue
                auc = roc_auc_score(y, v)
                scores.append({"freq": freq, "labels": mode, "channel": ch,
                               "auroc": max(auc, 1 - auc), "raw": auc})
    table = pd.DataFrame(scores)
    if not table.empty:
        pivot = table.pivot(index="channel", columns=["freq", "labels"],
                            values="auroc").sort_index(axis=1)
        with pd.option_context("display.float_format", "{:.3f}".format,
                               "display.width", 220):
            print(pivot.to_string())

        if {"real", "shuffled"} <= set(args.label_modes):
            print("\n" + "=" * 92)
            print("IS THE SIGNAL REAL?  real minus shuffled, per channel")
            print("=" * 92)
            print("  A channel reading learned structure collapses to ~0.5 once the")
            print("  labels carry no information. One that holds up under shuffling")
            print("  is responding to the ablation itself, not to what was learned.\n")
            gap = pd.DataFrame(index=pivot.index)
            for f in args.frequencies:
                if (f, "real") in pivot.columns and (f, "shuffled") in pivot.columns:
                    gap[f"f{f:g}_real"] = pivot[(f, "real")]
                    gap[f"f{f:g}_shuf"] = pivot[(f, "shuffled")]
                    gap[f"f{f:g}_gap"] = pivot[(f, "real")] - pivot[(f, "shuffled")]
            gap_cols = [c for c in gap.columns if c.endswith("_gap")]
            if gap_cols:
                gap = gap.sort_values(gap_cols[-1], ascending=False)
                with pd.option_context("display.float_format", "{:.3f}".format,
                                       "display.width", 220):
                    print(gap.to_string())

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outdir / "diagnostic_deltas.csv", index=False)
    qframe.to_csv(outdir / "network_quality.csv", index=False)
    if not table.empty:
        table.to_csv(outdir / "diagnostic_auroc.csv", index=False)
    print(f"\nwrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
