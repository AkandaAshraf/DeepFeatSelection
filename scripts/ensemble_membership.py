"""Ensemble consensus at scale: many independent minima, brute-force analysis.

The design principle: train MULTIPLE models and
do a brute-force analysis across them to find causal links -- because a single
model starts from somewhere arbitrary and we do not even know what minimum it
reaches. At V=1000 the single masked AE fell into the family-prior minimum
and measured typicality. The ensemble hypothesis: independently initialised
models fall into DIFFERENT minima, so

* quantities determined by the DATA (a member's recreatability from the web
  it is coupled to) are STABLE across models, while
* artifacts of the OPTIMISATION (which channels get memorised, which spurious
  routes form, how the family prior is expressed) VARY across models.

Consensus mean is then the membership signal and cross-model variance is an
artifact alarm. Scored per variable across M models:

    consensus  = mean_m r2_m(X)
    stability  = std_m  r2_m(X)
    tstat      = consensus / (stability + eps)   -- the brute-force statistic

All three are evaluated against membership truth (PR-AUC primary), alongside
the single best model, so the ensemble's contribution is isolated. The ghost
channel and the per-loner periodicity proxy ride along: if the ghost's high
single-model score is memorisation/clock-lookup (init-dependent), its
cross-model variance should be large and its consensus rank should fall.

Weights of every ensemble member are saved; the scorer is vectorised
(variables scored in chunks per forward pass) so readouts stay cheap.

    python scripts/ensemble_membership.py --v 1000 --members 100 --n-models 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from bottleneck_membership import (MaskedAE, build_system, build_system_hetero, splits_for, E)  # noqa: E402
from network_scale import random_dag, simulate  # noqa: E402
from chamber_detect import r2_of  # noqa: E402
from deepfeatselect.ccm import time_delay_embed  # noqa: E402


def scores_vectorised(model: MaskedAE, zs: np.ndarray, te: slice, v: int,
                      chunk: int = 64) -> np.ndarray:
    """Per-variable masked-recreation r2, ``chunk`` variables per forward pass."""
    base = zs[te]
    rows = len(base)
    out = np.empty(v)
    for start in range(0, v, chunk):
        xs = list(range(start, min(start + chunk, v)))
        stacked = np.tile(base, (len(xs), 1))
        for k, x in enumerate(xs):
            stacked[k * rows:(k + 1) * rows, x * E:(x + 1) * E] = 0.0
        pred = model.predict(stacked, verbose=0, batch_size=4096)
        for k, x in enumerate(xs):
            out[x] = r2_of(pred[k * rows:(k + 1) * rows, x * E:(x + 1) * E],
                           base[:, x * E:(x + 1) * E])
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--v", type=int, default=1000)
    p.add_argument("--members", type=int, default=100)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--coupling", type=float, default=0.3)
    p.add_argument("--bottleneck", type=int, default=128)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--n-models", type=int, default=8)
    p.add_argument("--system-seed", type=int, default=0)
    p.add_argument("--hetero", action="store_true")
    p.add_argument("--outdir", default="ExpOutput/ensemble")
    args = p.parse_args()

    # ONE system; the ensemble varies ONLY the model initialisation and batch
    # order, so cross-model variance is purely optimisation variance.
    builder = build_system_hetero if args.hetero else build_system
    x, truth = builder(args.v, args.members, args.n, args.coupling,
                       args.system_seed)
    mats = [time_delay_embed(x[:, j], E)[0] for j in range(args.v)]
    n = min(len(m) for m in mats)
    joint = np.hstack([m[:n] for m in mats]).astype("float64")
    rng = np.random.default_rng(args.system_seed + 7331)
    donor = int(rng.integers(0, args.members))
    ghost = np.roll(joint[:, donor * E:(donor + 1) * E],
                    int(rng.integers(n // 4, 3 * n // 4)), axis=0)
    joint = np.hstack([joint, ghost])
    v_all = args.v + 1
    truth_all = np.append(truth, False)
    tr, va, te = splits_for(n)
    mu, sd = joint[tr].mean(0), joint[tr].std(0) + 1e-12
    zs = ((joint - mu) / sd).astype("float32")
    print(f"system: V={v_all} (ghost incl.), D={zs.shape[1]}, "
          f"{int(truth_all.sum())} members, baseline {truth_all.mean():.3f}")

    outdir = Path(args.outdir)
    (outdir / "models").mkdir(parents=True, exist_ok=True)
    all_scores = np.empty((args.n_models, v_all))
    for m in range(args.n_models):
        t0 = time.time()
        keras.utils.set_random_seed(1000 + m)   # the ONLY thing that varies
        model = MaskedAE(v_all, E, args.bottleneck, mask_mode="zero",
                         loss_on_masked_only=True)
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(3e-3))
        model.fit(zs[tr], zs[tr], validation_data=(zs[va], zs[va]),
                  epochs=args.epochs, batch_size=64, shuffle=True, verbose=0)
        model.save_weights(outdir / "models" / f"m{m}.weights.h5")
        all_scores[m] = scores_vectorised(model, zs, te, v_all)
        ap = average_precision_score(truth_all, all_scores[m])
        print(f"  model {m}: single-model ap {ap:.3f}  "
              f"ghost {all_scores[m][-1]:+.3f}  ({time.time()-t0:.0f}s)")

    consensus = all_scores.mean(axis=0)
    stability = all_scores.std(axis=0)
    tstat = consensus / (stability + 1e-6)

    np.save(outdir / "all_scores.npy", all_scores)
    rows = []
    for name, s in (("single_best", None), ("consensus", consensus),
                    ("neg_stability", -stability), ("tstat", tstat)):
        if name == "single_best":
            aps = [average_precision_score(truth_all, all_scores[m])
                   for m in range(args.n_models)]
            best = int(np.argmax(aps))
            s = all_scores[best]
        rows.append({"statistic": name,
                     "prauc": average_precision_score(truth_all, s),
                     "auroc": roc_auc_score(truth_all, s),
                     "ghost_value": float(s[-1]),
                     "ghost_rank": int((np.argsort(-s) == v_all - 1
                                        ).argmax()) + 1})
    table = pd.DataFrame(rows)
    table.to_csv(outdir / "ensemble_eval.csv", index=False)

    print("\n" + "=" * 84)
    print(f"ENSEMBLE ANALYSIS ({args.n_models} independent minima, "
          f"baseline {truth_all.mean():.3f})")
    print("=" * 84)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))

    # THE LAW-OF-LARGE-NUMBERS CURVE, made measurable:
    # consensus AP as a function of ensemble size M, by prefix averaging.
    # Rising -> optimisation variance is averaging out and the required M for
    # a target precision is estimable (M ~ 1/SNR^2). Flat at chance -> the
    # artifact is a SHARED bias of the loss landscape and no M rescues it;
    # LLN removes variance, never bias.
    print("\nconsensus AP by ensemble size (prefix means):")
    for m_size in range(1, args.n_models + 1):
        cm = all_scores[:m_size].mean(axis=0)
        print(f"  M={m_size}: ap {average_precision_score(truth_all, cm):.3f}"
              f"  auroc {roc_auc_score(truth_all, cm):.3f}"
              f"  ghost {cm[-1]:+.3f}")

    member_stab = stability[truth_all]
    lone_stab = stability[args.members:args.v]
    print(f"\ncross-model std: members {member_stab.mean():.4f}  "
          f"loners {stability[args.members:args.v].mean():.4f}  "
          f"ghost {stability[-1]:.4f}")
    order = np.argsort(-consensus)
    for k in (10, 20, 50):
        hits = int(truth_all[order[:k]].sum())
        print(f"consensus precision@{k}: {hits}/{k}")

    # Clock-leak probe: periodic loners identifiable by few unique states.
    lone_series = x[:, args.members:]
    uniq = np.array([len(np.unique(np.round(lone_series[-500:, j], 6)))
                     for j in range(args.v - args.members)])
    few = uniq < 50
    ls = consensus[args.members:args.v]
    print(f"\nperiodic loners: {int(few.sum())}; consensus score "
          f"{ls[few].mean():+.3f} vs chaotic {ls[~few].mean():+.3f}")
    print(f"wrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
