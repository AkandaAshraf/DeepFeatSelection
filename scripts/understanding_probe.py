"""Does ablation-profile shape tell understanding from memorisation?

Every scaling run in this project recorded validation loss alone, which cannot
separate a network that memorised its rows from one that never fitted anything.
Both report a loss at chance. This adds training loss, and adds the measurement
that motivated the whole idea: the *shape* of the ablation profile.

The claim under test: a model that represents an irreducible k-way interaction
must lose the function when any one member is removed, so its profile of
ablation deltas is peaked -- k large, the rest flat. A model doing lookup has
every column in the key, so its profile is uniform. Peakedness therefore
indicates understanding without needing to know which features matter.

Three arms per configuration:

* **real** -- the system as generated;
* **shuffled** -- labels permuted, so there is nothing to understand and any
  structure the model finds is memorisation. This is the reference profile;
* the contrast between them is the result.

If the claim holds, peakedness is high on real data where the model learned,
falls where it did not, and is at floor on shuffled labels regardless.

    python scripts/understanding_probe.py --frequencies 2 4 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from deepfeatselect.model import build_model
from deepfeatselect.scaling import oblique_interaction
from deepfeatselect.understanding import memorisation_gap, profile_shape

CHANCE = float(np.log(2))


def train_arm(x_tr, y_tr, x_va, y_va, args, seed) -> dict[str, float]:
    """Losses and held-out AUC for one arm, with training loss measured properly.

    The training loss reported by ``fit`` is not comparable to validation loss:
    Keras computes it with dropout ACTIVE and averaged over batches as the
    weights change, while validation loss is computed with dropout OFF on the
    epoch's final weights. Differencing the two measures the dropout asymmetry
    as much as it measures generalisation, and it understates the gap -- which
    is exactly the wrong direction when the quantity of interest is whether the
    model memorised.

    Since ``restore_best_weights`` leaves the model at the selected epoch, the
    fix is to re-evaluate the training set in inference mode. Both numbers are
    returned so the size of the artefact stays visible rather than being
    silently corrected away.
    """
    keras.utils.set_random_seed(seed)
    model = build_model(
        n_columns=x_tr.shape[1], groups=np.arange(x_tr.shape[1]), n_classes=2,
        task="binary", l1_gate=0.0, dropout=args.dropout, noise=0.0,
        hidden_units=args.width, n_hidden_layers=2, l2_dense=1e-3,
        learning_rate=3e-3,
    )
    stopper = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True)
    history = model.fit(
        x_tr, y_tr.astype("float32").reshape(-1, 1),
        validation_data=(x_va, y_va.astype("float32").reshape(-1, 1)),
        epochs=args.epochs, batch_size=128, shuffle=True, verbose=0,
        callbacks=[stopper])

    val_curve = history.history["val_loss"]
    best = int(np.argmin(val_curve))

    # Inference mode: dropout off, same weights the validation loss was measured
    # on, so the two are finally on the same footing.
    evaluated = model.evaluate(x_tr, y_tr.astype("float32").reshape(-1, 1),
                               verbose=0, return_dict=True)
    probs = model.predict(x_va, verbose=0).reshape(-1)
    auc = (roc_auc_score(y_va, probs) if len(np.unique(y_va)) > 1 else np.nan)

    return {
        "val_loss": float(val_curve[best]),
        "train_loss": float(evaluated["loss"]),
        "train_loss_dropout_on": float(history.history["loss"][best]),
        "held_out_auc": float(auc),
    }


def run_config(freq: float, seed: int, label_mode: str, args) -> dict:
    system = oblique_interaction(n=args.n, n_features=args.d, k=args.k,
                                 frequency=freq, seed=seed)
    labels = system.y
    if label_mode == "shuffled":
        labels = np.random.default_rng(seed + 4241).permutation(labels)

    x_tr, x_va, y_tr, y_va = train_test_split(
        system.x, labels, test_size=0.3, random_state=seed, stratify=labels)

    base = train_arm(x_tr, y_tr, x_va, y_va, args, seed)
    quality = memorisation_gap(base["train_loss"], base["val_loss"])
    # How much the dropout asymmetry alone was contributing, before the fix.
    quality["artefact_size"] = base["train_loss_dropout_on"] - base["train_loss"]
    auc = base["held_out_auc"]

    deltas = np.empty(args.d)
    for j in range(args.d):
        tr, va = x_tr.copy(), x_va.copy()
        tr[:, j] = 0.0
        va[:, j] = 0.0
        deltas[j] = train_arm(tr, y_tr, va, y_va, args, seed)["val_loss"] - base["val_loss"]

    shape = profile_shape(deltas)

    # Detection is reported alongside, so profile shape can be checked against
    # whether the probe actually found anything.
    keep = system.interaction | system.irrelevant
    truth = system.interaction[keep].astype(int)
    v = np.abs(deltas)[keep]
    # Orientation is fixed a priori: the hypothesis is that true members have
    # LARGER ablation deltas, so |delta| is the score and there is nothing to
    # flip. An earlier version took max(a, 1-a) "defensively", which silently
    # fitted the sign to the answer -- it inflated the shuffled-label null from
    # 0.500 to a mean of 0.637 and produced a 0.804 on permuted labels, where by
    # construction there is nothing to detect. Any run reporting detection near
    # 0.6 under that version was reporting the null.
    detection = np.nan
    if not np.allclose(v, v[0]):
        detection = float(roc_auc_score(truth, v))

    # How much of the profile's mass sits on the true members: the ground-truth
    # version of peakedness, for calibrating the blind version against.
    on_target = float(np.abs(deltas)[system.interaction].sum()
                      / (np.abs(deltas).sum() + 1e-12))

    return {"freq": freq, "seed": seed, "labels": label_mode, "held_out_auc": auc,
            "detection_auroc": detection, "on_target_share": on_target,
            **quality, **shape}


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
    p.add_argument("--label-modes", nargs="+", default=["real", "shuffled"])
    p.add_argument("--outdir", default="ExpOutput/understanding")
    args = p.parse_args()

    rows = []
    for freq in args.frequencies:
        for mode in args.label_modes:
            for seed in range(args.seeds):
                row = run_config(freq, seed, mode, args)
                rows.append(row)
                print(f"  freq={freq:g} {mode} seed={seed}: {row['regime']:<10} "
                      f"train {row['train_loss']:.3f} val {row['val_loss']:.3f} "
                      f"gini {row['gini']:.3f} top4 {row['top4_share']:.3f}")

    frame = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print("MEMORISATION VERSUS UNDERSTANDING")
    print("=" * 100)
    print(f"  chance loss {CHANCE:.3f}. 'memorised' = train well below chance while")
    print("  validation sits at it; 'failed' = neither moved.\n")
    cols = ["freq", "labels", "seed", "train_loss", "val_loss", "memorisation_gap",
            "artefact_size",
            "regime", "held_out_auc"]
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 220):
        print(frame[cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("ABLATION PROFILE SHAPE  (blind: needs no ground truth)")
    print("=" * 100)
    shape_cols = ["freq", "labels", "seed", "gini", "kurtosis", "top4_share",
                  "participation", "on_target_share", "detection_auroc"]
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 220):
        print(frame[shape_cols].to_string(index=False))

    print("\n  top4_share is the fraction of ablation mass on the four largest")
    print(f"  features; with {args.d} features and no structure it would be about "
          f"{4 / args.d:.2f}.")
    print("  on_target_share is the same quantity restricted to the TRUE members,")
    print("  so top4_share ~ on_target_share means the peak is in the right place.")

    real = frame[frame.labels == "real"]
    shuf = frame[frame.labels == "shuffled"]
    if not shuf.empty:
        print("\n" + "=" * 100)
        print("DOES PEAKEDNESS TRACK UNDERSTANDING?")
        print("=" * 100)
        summary = pd.concat([
            real.groupby("freq")[["gini", "top4_share", "held_out_auc",
                                  "detection_auroc"]].mean().add_prefix("real_"),
            shuf.groupby("freq")[["gini", "top4_share"]].mean().add_prefix("shuf_"),
        ], axis=1)
        with pd.option_context("display.float_format", "{:.3f}".format,
                               "display.width", 220):
            print(summary.to_string())

        if len(real) > 2 and real.held_out_auc.notna().all():
            for stat in ("gini", "top4_share", "participation"):
                r = np.corrcoef(real[stat], real.held_out_auc)[0, 1]
                print(f"  corr({stat}, held-out AUC) over real runs = {r:+.3f}")
            print("\n  A strong positive correlation would mean profile shape reports")
            print("  learning quality without labels, which is the claim.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outdir / "understanding.csv", index=False)
    print(f"\nwrote {outdir}/understanding.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
