"""Excess-over-self-predictability: the complexity-corrected membership score.

The ensemble run diagnosed the V=1000 failure in one sentence: masked
recreation conflates "predictable from others" with "predictable at all",
ranking channels by complexity (periodic loners 0.76 > ghost 0.74 > chaotic
loners 0.29 > members 0.25). The correction, identified there and tested
here: membership is the EXCESS of what the system knows about X over what X
knows about itself,

    excess(X) = r2( x_{t+1} | own lags + system code ) - r2( x_{t+1} | own lags )

A periodic channel's own lags explain it entirely -> excess 0. An autonomous
chaotic map is deterministic in its own lag -> excess 0. A member's own lags
cannot explain its drive term, but the code carries its parents' states ->
excess > 0. The complexity confound cancels by construction. This is
group-conditioning (all others vs none), which the removal-sets result showed
survives redundancy where single-source differences collapse; and it is the
quantity CCM's convergence criterion implicitly controls for.

EVERYTHING here is a readout on the SAVED ensemble encoders -- no training.
Per saved model: encode the joint state to its 128-dim code, then per
variable fit two ridge regressions (seconds each). Consensus over the 8
encoders per the ensemble design. A raw-ridge arm (own lags + ALL
other channels' raw dims, no AE) rides along: if it matches the code arm,
the encoder adds nothing and the readout alone was the fix.

    python scripts/excess_membership.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from bottleneck_membership import (MaskedAE, build_system, build_system_hetero, splits_for, E)  # noqa: E402
from deepfeatselect.ccm import time_delay_embed  # noqa: E402


def poly_own(own: np.ndarray, degree: int = 2) -> np.ndarray:
    """Own lags plus squares and pairwise products.

    The first run used LINEAR own-lag baselines and the self r2 came out 0.62
    where theory demands ~1.0 for a deterministic map -- a linear fit of a
    quadratic map. That single flaw resurrected the clock confound (the code
    completed what the too-weak baseline could not) and inverted members. With
    quadratic features the logistic family is represented EXACTLY, so the
    self-baseline absorbs all own-dynamics, periodic or chaotic.
    """
    cols = [own]
    e = own.shape[1]
    for i in range(e):
        for j in range(i, e):
            cols.append((own[:, i] * own[:, j])[:, None])
    if degree >= 3:
        for i in range(e):
            for j in range(i, e):
                for k in range(j, e):
                    cols.append((own[:, i] * own[:, j] * own[:, k])[:, None])
    return np.hstack(cols)


def r2_clamped(pred: np.ndarray, truth: np.ndarray) -> float:
    err = float(np.mean((pred - truth) ** 2))
    return max(0.0, 1.0 - err / (float(np.var(truth)) + 1e-12))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--v", type=int, default=1000)
    p.add_argument("--members", type=int, default=100)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--coupling", type=float, default=0.3)
    p.add_argument("--bottleneck", type=int, default=128)
    p.add_argument("--system-seed", type=int, default=0)
    p.add_argument("--hetero", action="store_true")
    p.add_argument("--poly-degree", type=int, default=2)
    p.add_argument("--models-dir", default="ExpOutput/ensemble/models")
    p.add_argument("--outdir", default="ExpOutput/excess")
    args = p.parse_args()

    # Rebuild the exact system the ensemble trained on.
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
    base = float(truth_all.mean())
    print(f"system rebuilt: V={v_all}, {int(truth_all.sum())} members, "
          f"baseline {base:.3f}")

    # Targets: next-step value of each channel's leading coordinate.
    # x_{t+1} approximated by the leading embedding coordinate at t+1.
    lead = zs[:, [j * E for j in range(v_all)]]          # (n, v_all)
    tr_idx = np.arange(tr.start, tr.stop - 1)
    te_idx = np.arange(te.start, n - 1)

    def fit_pair(own_cols: np.ndarray, extra: np.ndarray | None,
                 target: np.ndarray, alpha: float = 1.0) -> float:
        own_p = poly_own(own_cols, args.poly_degree)
        src_tr = own_p[tr_idx]
        src_te = own_p[te_idx]
        if extra is not None:
            src_tr = np.hstack([src_tr, extra[tr_idx]])
            src_te = np.hstack([src_te, extra[te_idx]])
        m = Ridge(alpha=alpha)
        m.fit(src_tr, target[tr_idx + 1])
        return r2_clamped(m.predict(src_te), target[te_idx + 1])

    model_files = sorted(Path(args.models_dir).glob("m*.weights.h5"))
    print(f"{len(model_files)} saved encoders found")
    excess_all, joint_all, self_r2 = [], [], np.empty(v_all)

    # Self baseline is model-independent: compute once.
    t0 = time.time()
    for q in range(v_all):
        own = zs[:, q * E:(q + 1) * E]
        self_r2[q] = fit_pair(own, None, lead[:, q])
    print(f"self baselines in {time.time()-t0:.0f}s "
          f"(members {self_r2[truth_all].mean():.3f}, "
          f"loners {self_r2[args.members:args.v].mean():.3f}, "
          f"ghost {self_r2[-1]:.3f})")

    for f in model_files:
        model = MaskedAE(v_all, E, args.bottleneck, mask_mode="zero",
                         loss_on_masked_only=True)
        model(zs[:2])
        model.load_weights(f)
        code = model.encoder.predict(zs, verbose=0, batch_size=4096)
        ex = np.empty(v_all)
        jr = np.empty(v_all)
        for q in range(v_all):
            own = zs[:, q * E:(q + 1) * E]
            jr[q] = fit_pair(own, code, lead[:, q])
            ex[q] = jr[q] - self_r2[q]
        excess_all.append(ex)
        joint_all.append(jr)
        print(f"  {f.name}: excess ap "
              f"{average_precision_score(truth_all, ex):.3f}")

    excess = np.mean(excess_all, axis=0)

    # No-AE control: ridge on own lags + ALL raw other dims.
    t0 = time.time()
    raw_ex = np.empty(v_all)
    for q in range(v_all):
        own = zs[:, q * E:(q + 1) * E]
        others = np.delete(zs, np.s_[q * E:(q + 1) * E], axis=1)
        raw_ex[q] = fit_pair(own, others, lead[:, q],
                             alpha=100.0) - self_r2[q]
    print(f"raw-ridge control in {time.time()-t0:.0f}s")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "excess_consensus.npy", excess)
    rows = []
    for name, s in (("excess_consensus", excess),
                    ("excess_single_first", excess_all[0]),
                    ("raw_ridge_excess", raw_ex)):
        rows.append({"statistic": name,
                     "prauc": average_precision_score(truth_all, s),
                     "auroc": roc_auc_score(truth_all, s),
                     "ghost_value": float(s[-1]),
                     "member_mean": float(s[truth_all].mean()),
                     "lone_mean": float(s[args.members:args.v].mean())})
    table = pd.DataFrame(rows)
    table.to_csv(outdir / "excess_eval.csv", index=False)
    print("\n" + "=" * 88)
    print(f"EXCESS-OVER-SELF MEMBERSHIP (baseline {base:.3f})")
    print("=" * 88)
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 140):
        print(table.to_string(index=False))

    lone_series = x[:, args.members:]
    uniq = np.array([len(np.unique(np.round(lone_series[-500:, j], 6)))
                     for j in range(args.v - args.members)])
    few = uniq < 50
    ls = excess[args.members:args.v]
    print(f"\nperiodic loners: excess {ls[few].mean():+.4f}   "
          f"chaotic loners: {ls[~few].mean():+.4f}   "
          f"(the confound is gone if both are ~0)")
    order = np.argsort(-excess)
    for k in (10, 20, 50):
        print(f"precision@{k}: {int(truth_all[order[:k]].sum())}/{k}")
    print(f"wrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
