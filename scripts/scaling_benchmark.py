"""Where does a trained network beat subset enumeration?

Marginal statistics cost O(d) and see only first-order structure. Exhaustive
subset methods see any order and cost C(d, k) or 2^d. A network sees any order
in one fitted function, and probing it by ablation costs d trainings -- linear
in the feature count.

That is an argument, not a result. This measures the crossover: a k-way parity
buried in a growing pile of noise features, scored by detection AUROC for every
method as d increases. The interaction members are the interesting target,
because no proper subset of a parity carries information and marginal methods
are therefore blind to them by construction.

The marginal causes are the control. Any method that fails to find *those* is
broken rather than limited, and its interaction score means nothing.

    python scripts/scaling_benchmark.py --dims 10 20 40 --seeds 3
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from deepfeatselect.model import build_model
from deepfeatselect.scaling import oblique_interaction, parity_interaction


def detection_auc(scores: np.ndarray, positive: np.ndarray,
                  negative: np.ndarray) -> float:
    """AUROC at ranking one masked group above another."""
    keep = positive | negative
    y = positive[keep].astype(int)
    v = scores[keep]
    if len(np.unique(y)) < 2 or np.allclose(v, v[0]):
        return np.nan
    return float(roc_auc_score(y, v))


# --------------------------------------------------------------------------
# baselines

def score_mutual_info(x, y, seed):
    return mutual_info_classif(x, y, random_state=seed)


def score_forest(x, y, seed):
    model = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1)
    model.fit(x, y)
    return model.feature_importances_


def score_permutation(x, y, seed):
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.3, random_state=seed, stratify=y)
    model = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    model.fit(x_tr, y_tr)
    result = permutation_importance(model, x_te, y_te, n_repeats=10,
                                    random_state=seed, n_jobs=-1)
    return result.importances_mean


# --------------------------------------------------------------------------
# the deep probe

def _dense_model(d: int, width: int, dropout: float):
    return build_model(
        n_columns=d, groups=np.arange(d), n_classes=2, task="binary",
        l1_gate=0.0, dropout=dropout, noise=0.0,
        hidden_units=width, n_hidden_layers=2, l2_dense=1e-3,
        learning_rate=3e-3,
    )


def _conv_model(d: int, width: int, dropout: float, kernel: int = 3):
    """Convolutional trunk with a dense head.

    Weight sharing across the feature axis makes the trunk cheap, and the dense
    head restores the global mixing the scaling argument depends on. Note the
    trunk's receptive field is local: with the interaction members scattered by
    the generator, a single conv layer cannot span them and the head has to do
    the combining. That is the point of the comparison.
    """
    inputs = keras.layers.Input(shape=(d,))
    h = keras.layers.Reshape((d, 1))(inputs)
    h = keras.layers.Conv1D(width // 2, kernel, padding="same", activation="relu")(h)
    h = keras.layers.Conv1D(width // 2, kernel, padding="same", activation="relu")(h)
    h = keras.layers.Flatten()(h)
    if dropout:
        h = keras.layers.Dropout(dropout)(h)
    h = keras.layers.Dense(width, activation="relu",
                           kernel_regularizer=keras.regularizers.L2(1e-3))(h)
    out = keras.layers.Dense(1, activation="sigmoid", name="output")(h)
    model = keras.Model(inputs, out)
    model.compile(loss=keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(3e-3),
                  metrics=[keras.metrics.AUC(name="auc")])
    return model


def _train(x_tr, y_tr, x_va, y_va, width, epochs, seed, arch, dropout, patience):
    """Train one arm with early stopping, returning its best validation loss.

    Early stopping is both the regularisation the previous run lacked -- where
    validation loss reached 2.0 against a chance level of 0.69 -- and the main
    saving, since most arms converge long before the epoch cap.
    """
    keras.utils.set_random_seed(seed)
    d = x_tr.shape[1]
    model = (_dense_model(d, width, dropout) if arch == "dense"
             else _conv_model(d, width, dropout))
    stopper = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min",
        restore_best_weights=True)
    history = model.fit(
        x_tr, y_tr.astype("float32").reshape(-1, 1),
        validation_data=(x_va, y_va.astype("float32").reshape(-1, 1)),
        epochs=epochs, batch_size=128, shuffle=True, verbose=0,
        callbacks=[stopper],
    )
    return float(np.min(history.history["val_loss"]))


def score_deprivation(x, y, seed, width, epochs, arch="dense",
                      dropout=0.2, patience=15):
    """Loss increase when each feature is zeroed, one network per feature.

    Linear in the feature count, and the network has already learned whatever
    joint structure exists, so no subset enumeration is needed to notice that a
    feature participates in one.
    """
    x_tr, x_va, y_tr, y_va = train_test_split(
        x, y, test_size=0.3, random_state=seed, stratify=y)
    base = _train(x_tr, y_tr, x_va, y_va, width, epochs, seed, arch, dropout, patience)

    scores = np.empty(x.shape[1])
    for j in range(x.shape[1]):
        tr, va = x_tr.copy(), x_va.copy()
        tr[:, j] = 0.0
        va[:, j] = 0.0
        scores[j] = _train(tr, y_tr, va, y_va, width, epochs, seed,
                           arch, dropout, patience) - base
    return scores, base


# --------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dims", type=int, nargs="+", default=[10, 20, 40])
    p.add_argument("--system", choices=["parity", "oblique"], default="parity",
                   help="parity is axis-aligned and suits trees natively; "
                        "oblique is smooth and tilted to every axis")
    p.add_argument("--frequencies", type=float, nargs="+", default=[2.0],
                   help="oblique only: how oscillatory the coupling is. A tree "
                        "approximates a smooth oblique surface with a staircase, "
                        "so the splits it needs grow with this, while a dense "
                        "layer holds the same projection vector regardless.")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--n-marginal", type=int, default=2)
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--archs", nargs="+", default=["dense", "conv"],
                   choices=["dense", "conv"])
    p.add_argument("--skip-deep", action="store_true")
    p.add_argument("--outdir", default="ExpOutput/scaling")
    args = p.parse_args()

    rows = []
    for d in args.dims:
      for freq in args.frequencies:
        for seed in range(args.seeds):
            generator = (parity_interaction if args.system == "parity"
                         else oblique_interaction)
            kwargs = {"frequency": freq} if args.system == "oblique" else {}
            system = generator(n=args.n, n_features=d, k=args.k,
                               n_marginal=args.n_marginal, seed=seed, **kwargs)
            x, y = system.x, system.y

            cheap = {
                "mutual_info": score_mutual_info,
                "random_forest": score_forest,
                "permutation": score_permutation,
            }
            for name, fn in cheap.items():
                t0 = time.time()
                scores = fn(x, y, seed)
                rows.append({
                    "d": d, "freq": freq, "seed": seed, "method": name,
                    "auc_interaction": detection_auc(scores, system.interaction,
                                                     system.irrelevant),
                    "auc_marginal": detection_auc(scores, system.marginal,
                                                  system.irrelevant),
                    "seconds": time.time() - t0, "val_loss": np.nan,
                })

            if not args.skip_deep:
                for arch in args.archs:
                    t0 = time.time()
                    scores, base = score_deprivation(
                        x, y, seed, args.width, args.epochs, arch=arch,
                        dropout=args.dropout, patience=args.patience)
                    rows.append({
                        "d": d, "freq": freq, "seed": seed, "method": f"probe_{arch}",
                        "auc_interaction": detection_auc(scores, system.interaction,
                                                         system.irrelevant),
                        "auc_marginal": detection_auc(scores, system.marginal,
                                                      system.irrelevant),
                        "seconds": time.time() - t0, "val_loss": base,
                    })
            print(f"  d={d} freq={freq} seed={seed} done")

    frame = pd.DataFrame(rows)
    axis = "freq" if len(args.frequencies) > 1 else "d"
    summary = (frame.groupby(["method", axis])
               .agg(auc_interaction=("auc_interaction", "mean"),
                    auc_marginal=("auc_marginal", "mean"),
                    seconds=("seconds", "mean"),
                    val_loss=("val_loss", "mean"))
               .reset_index())

    print("\n" + "=" * 92)
    sweeping = "COUPLING NONLINEARITY" if axis == "freq" else "FEATURE COUNT"
    print(f"DETECTING A {args.k}-WAY {args.system.upper()} INTERACTION "
          f"AS {sweeping} GROWS")
    print("=" * 92)
    print("  auc_interaction: finding the interaction members (0.5 = chance)")
    print("  auc_marginal   : finding the ordinary causes -- the control\n")

    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(summary.pivot(index=axis, columns="method",
                            values="auc_interaction").to_string())
        print("\nmarginal causes (control):")
        print(summary.pivot(index=axis, columns="method",
                            values="auc_marginal").to_string())
        print("\nseconds per run:")
        print(summary.pivot(index=axis, columns="method", values="seconds").to_string())

    if not args.skip_deep:
        chance = float(np.log(2))
        print(f"\nnetwork fit -- chance is {chance:.3f}; any arm above it has learned")
        print("nothing and its AUC is uninformative whatever the number says:")
        deep = summary[summary.method.str.startswith("probe_")]
        for _, r in deep.iterrows():
            flag = "  FAILED TO LEARN" if r.val_loss >= chance else ""
            print(f"  {r.method:<14} {axis}={r[axis]:<5g}  val_loss {r.val_loss:.4f}{flag}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outdir / "scaling_raw.csv", index=False)
    summary.to_csv(outdir / "scaling_summary.csv", index=False)
    print(f"\nwrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
