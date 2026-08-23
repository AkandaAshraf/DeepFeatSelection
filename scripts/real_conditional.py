"""Conditional outflow on a real physical system with structural ground truth.

Pre-registration: paper/real_conditional_protocol.md, committed before any
statistic was computed on these datasets. The fitness screen that selected
them ran first and is reported in paper/dataset_fitness_protocol.md.

A1 and C1 differ only in the conditioning set. Both qualifying datasets are
tested and both are reported.

    python scripts/real_conditional.py
"""

from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import source_outflow_gate as G  # noqa: E402
from error_metrics import _ridge_pred, auc, score_all  # noqa: E402

OUT = Path("ExpOutput/real_conditional")
SETTABLE = ["hatch", "pot_1", "pot_2", "load_in", "load_out"]
MEASURED = ["current_in", "current_out", "rpm_in", "rpm_out",
            "pressure_upwind", "pressure_downwind", "pressure_ambient",
            "pressure_intake", "mic", "signal_1", "signal_2"]
M = 10
EPOCHS, N_PERM = 25, 10000
DATASETS = [
    ("wt_intake_impulse_v1",
     "Data/causalchamber/wt_intake_impulse_v1/**/*.csv", "PRIMARY"),
    ("wt_walks_v1", "Data/causalchamber/wt_walks_v1/*.csv", "SECOND"),
]
MIN_N = 2000


def run_one(x, n_src, seed):
    """A1 and C1 per channel on one run, with the ghost appended."""
    xg = np.concatenate([x, np.roll(x[:, [0]], x.shape[0] // 3, axis=0)],
                        axis=1)
    V = xg.shape[1]
    G.BOTTLENECK, G.SEED = 4 * V, seed          # b = 4V, as declared
    emb = G.embed(xg)
    m = emb.shape[0]
    a, b = int(0.6 * m), int(0.8 * m)
    tr, tr_i, te_i = slice(0, a), np.arange(0, a - 1), np.arange(b, m - 1)
    mu, sd = emb[tr].mean(0), emb[tr].std(0) + 1e-12
    zs = np.clip(np.nan_to_num((emb - mu) / sd), -20, 20).astype(np.float32)
    feats = [G.poly2(zs[:, q * G.E:(q + 1) * G.E]) for q in range(V)]
    net = G.train_ae(zs, V, tr, EPOCHS, seed)
    span = G.E - 1
    ti, si = tr_i[tr_i >= span], te_i[te_i >= span]

    rows = []
    for q in range(V):
        zq = G.codes_with_mask(net, zs, V, q=q)
        zh = G.code_history(zq)
        ytr, yte = zq[ti + 1], zq[si + 1]
        aug = np.hstack([zh, feats[q]])
        a1 = (score_all(_ridge_pred(aug[ti], ytr, aug[si]), yte)["A1_r2"]
              - score_all(_ridge_pred(zh[ti], ytr, zh[si]), yte)["A1_r2"])
        oth = np.hstack([zh] + [feats[j] for j in range(V) if j != q])
        both = np.hstack([oth, feats[q]])
        c1 = (score_all(_ridge_pred(both[ti], ytr, both[si]), yte)["A1_r2"]
              - score_all(_ridge_pred(oth[ti], ytr, oth[si]), yte)["A1_r2"])
        role = ("source" if q < n_src
                else "ghost" if q == V - 1 else "sensor")
        rows.append({"channel": q, "role": role, "A1": a1, "C1": c1, "V": V})
    return pd.DataFrame(rows)


def perm_p(src, sen, observed, rng):
    """P(AUC of a random relabelling >= observed), roles shuffled."""
    pool = np.concatenate([src, sen])
    k = len(src)
    hits = 0
    for _ in range(N_PERM):
        rng.shuffle(pool)
        if auc(pool[:k], pool[k:]) >= observed:
            hits += 1
    return (hits + 1) / (N_PERM + 1)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device: {G.DEV}   decimation m={M}   b = 4V   epochs={EPOCHS}\n")
    t0, summary = time.time(), []

    for name, pat, tag in DATASETS:
        files = sorted(glob.glob(pat, recursive=True))
        parts, used = [], 0
        for i, f in enumerate(files):
            d = pd.read_csv(f)
            if not set(SETTABLE + MEASURED).issubset(d.columns):
                continue
            x = d[SETTABLE + MEASURED].to_numpy(float)[::M]
            if len(x) < MIN_N:
                continue
            live = np.where(x[:, :len(SETTABLE)].std(0) > 1e-9)[0]
            if len(live) == 0:
                continue
            cols = np.concatenate(
                [live, np.arange(len(SETTABLE), x.shape[1])])
            r = run_one(x[:, cols], len(live), seed=i)
            r["run"] = i
            r["sources"] = ",".join(SETTABLE[j] for j in live)
            parts.append(r)
            used += 1
            print(f"  {name} run {i}: n={len(x)} V={int(r.V.iloc[0])} "
                  f"sources={r.sources.iloc[0]}  ({time.time()-t0:.0f}s)",
                  flush=True)
        if not parts:
            print(f"  {name}: no usable run\n")
            continue
        d = pd.concat(parts, ignore_index=True)
        d.to_csv(OUT / f"channels_{name}.csv", index=False)

        rng = np.random.default_rng(0)
        row = {"dataset": name, "tag": tag, "runs": used}
        print(f"\n{tag}  {name}   ({used} runs, "
              f"{int((d.role=='source').sum())} source channels, "
              f"{int((d.role=='sensor').sum())} sensor channels)")
        for st in ("A1", "C1"):
            s = d[d.role == "source"][st].values
            n = d[d.role == "sensor"][st].values
            g = float(d[d.role == "ghost"][st].median())
            u = auc(s, n)
            row[f"{st}_auc"] = u
            row[f"{st}_src"] = float(np.median(s))
            row[f"{st}_sen"] = float(np.median(n))
            row[f"{st}_ghost"] = g
            row[f"{st}_clean"] = bool(g < np.percentile(s, 5))
            print(f"   {st}  AUC {u:.3f}   source {np.median(s):+.5f}   "
                  f"sensor {np.median(n):+.5f}   ghost {g:+.5f}   "
                  f"{'clean' if row[f'{st}_clean'] else 'DIRTY'}")
        row["C1_p"] = perm_p(d[d.role == "source"].C1.values,
                             d[d.role == "sensor"].C1.values,
                             row["C1_auc"], rng)
        print(f"   R1 C1 above A1? {row['C1_auc']:.3f} vs {row['A1_auc']:.3f}"
              f"  -> {'YES' if row['C1_auc'] > row['A1_auc'] else 'NO'}")
        print(f"   R2 C1 sensors at or below ghost? "
              f"{row['C1_sen']:+.5f} vs {row['C1_ghost']:+.5f}  -> "
              f"{'YES' if row['C1_sen'] <= row['C1_ghost'] else 'NO'}")
        print(f"   R5 permutation p for C1 (reported, not a decision): "
              f"{row['C1_p']:.4f}\n")
        summary.append(row)

    s = pd.DataFrame(summary)
    s.to_csv(OUT / "summary.csv", index=False)
    print(f"({(time.time()-t0)/60:.1f} min)\n")

    print("VERDICT (rule fixed before running)")
    r1 = [bool(r.C1_auc > r.A1_auc) for _, r in s.iterrows()]
    r3 = [bool(r.A1_clean and r.C1_clean) for _, r in s.iterrows()]
    for (_, r), a, b_ in zip(s.iterrows(), r1, r3):
        print(f"   {r.dataset:22s} R1 {'PASS' if a else 'FAIL'}   "
              f"R3 {'PASS' if b_ else 'FAIL'}")
    if all(r1) and all(r3):
        print("   -> CONFIRMED ON REAL DATA. Conditioning detects "
              "experimenter-set\n      variables in a physical system whose "
              "ground truth is structural.")
    elif any(r1):
        print("   -> PARTIAL. R1 holds on one dataset and not the other. NO "
              "claim of\n      transfer is made.")
    else:
        print("   -> FAILS. The synthetic confirmation does not transfer. The "
              "statistic\n      joins the iEEG result, and the "
              "source-detection line CLOSES.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
