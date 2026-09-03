"""Sinks per source as a continuous axis: marginal (A1) and conditional (C1)
outflow on identical runs.

Pre-registration: paper/ratio_sweep_protocol.md, committed before this was
written or run. The per-cell machinery is conditional_outflow.cell with the
shape parameterised; A1 and C1 differ only in the conditioning set.

POST-RUN FIXES (2026-09-03), made after the run of that date and disclosed in
the protocol's result section. The verdict that run printed is preserved in
ExpOutput/ratio_sweep_run.log and was WRONG for two reasons fixed here:
  1. a tie in A1_auc - C1_auc (both at the 1.000 ceiling) was counted as a
     sign change, i.e. a crossover;
  2. the verdict never consulted T2, although the declared rule requires
     T1, T2 and T3 together.
Also fixed: per-run frames are now written to disk (the run saved only six
aggregate rows), the docstring pointed at the wrong protocol, and
COPY_NOISE was undefined on a dead path.

    python scripts/ratio_sweep.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import source_outflow_gate as G  # noqa: E402
from error_metrics import _ridge_pred, score_all  # noqa: E402

OUT = Path("ExpOutput/ratio_sweep")
COUPLING = 0.50
ALPHA = 0.05
V_TOTAL = 25                      # + ghost = 26, so b = 4V = 104 everywhere
EPOCHS = 25                       # as in every prior outflow gate
COPY_NOISE = 0.02                 # conditional_outflow's value; copies=0 here
CAL_SEEDS, TEST_SEEDS = range(900, 920), range(1000, 1020)
# (n_src, n_sink) -> sinks per source; n_iso pads to V_TOTAL
RATIOS = [(12, 12), (6, 12), (4, 12), (3, 15), (2, 16), (1, 11)]


def system(coupling, seed, copies, n=4000, n_src=3, n_sink=6, n_iso=6):
    """coupled() with near-duplicate copies of each source appended.

    A copy is neither a source nor driven, so it is excluded from both truth
    sets - the boundary map's convention.
    """
    rng = np.random.default_rng(seed)
    V = n_src + n_sink + n_iso
    x = np.zeros((n, V))
    x[0] = rng.uniform(0.2, 0.8, V)
    r_src = rng.uniform(3.7, 3.9, n_src)
    r_iso = rng.uniform(3.7, 3.9, n_iso)
    r_snk = rng.uniform(3.5, 3.7, n_sink)
    parent = rng.integers(0, n_src, n_sink)
    for t in range(n - 1):
        s = x[t, :n_src]
        x[t + 1, :n_src] = np.clip(r_src * s * (1 - s), 0, 1)
        k = x[t, n_src:n_src + n_sink]
        x[t + 1, n_src:n_src + n_sink] = np.clip(
            r_snk * k * (1 - k) + coupling * x[t, parent] * (1 - k), 0, 1)
        i = x[t, n_src + n_sink:]
        x[t + 1, n_src + n_sink:] = np.clip(r_iso * i * (1 - i), 0, 1)
    x += 0.01 * rng.standard_normal((n, V))
    role = ["source"] * n_src + ["sink"] * n_sink + ["isolated"] * n_iso
    if copies:
        dup = [x[:, j] + COPY_NOISE * rng.standard_normal(n)
               for j in range(n_src) for _ in range(copies)]
        x = np.concatenate([x, np.stack(dup, axis=1)], axis=1)
        role += ["copy"] * len(dup)
    return x, np.array(role)


def cell_shaped(coupling, seed, shape):
    G.SEED = seed          # BOTTLENECK is set by run() from V
    x, role = system(coupling, seed, 0, **shape)
    xg = np.concatenate([x, np.roll(x[:, [0]], x.shape[0] // 3, axis=0)],
                        axis=1)
    role = np.append(role, "ghost")
    V = xg.shape[1]
    emb = G.embed(xg)
    m = emb.shape[0]
    a, b = int(0.6 * m), int(0.8 * m)
    tr, tr_i, te_i = slice(0, a), np.arange(0, a - 1), np.arange(b, m - 1)
    mu, sd = emb[tr].mean(0), emb[tr].std(0) + 1e-12
    zs = np.clip((emb - mu) / sd, -20, 20).astype(np.float32)
    feats = [G.poly2(zs[:, q * G.E:(q + 1) * G.E]) for q in range(V)]

    net = G.train_ae(zs, V, tr, EPOCHS, seed)
    span = G.E - 1
    ti, si = tr_i[tr_i >= span], te_i[te_i >= span]

    rows = []
    for q in range(V):
        zq = G.codes_with_mask(net, zs, V, q=q)
        zh = G.code_history(zq)
        ytr, yte = zq[ti + 1], zq[si + 1]

        # A1 MARGINAL: baseline is the code's own history
        a1 = (score_all(_ridge_pred(np.hstack([zh, feats[q]])[ti], ytr,
                                    np.hstack([zh, feats[q]])[si]),
                        yte)["A1_r2"]
              - score_all(_ridge_pred(zh[ti], ytr, zh[si]), yte)["A1_r2"])

        # C1 CONDITIONAL: baseline also holds every OTHER channel
        oth = np.hstack([zh] + [feats[j] for j in range(V) if j != q])
        both = np.hstack([oth, feats[q]])
        c1 = (score_all(_ridge_pred(both[ti], ytr, both[si]), yte)["A1_r2"]
              - score_all(_ridge_pred(oth[ti], ytr, oth[si]), yte)["A1_r2"])

        rows.append({"coupling": coupling, "seed": seed,
                     "channel": q, "role": role[q], "A1": a1, "C1": c1})
    return pd.DataFrame(rows)


def auc(pos, neg):
    """Identical to error_metrics.auc; see scripts/test_auc_identical.py."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def run(n_src, n_sink, seed):
    n_iso = V_TOTAL - n_src - n_sink
    assert n_iso >= 0, (n_src, n_sink)
    shape = dict(n_src=n_src, n_sink=n_sink, n_iso=n_iso)
    V = V_TOTAL + 1
    G.BOTTLENECK, G.SEED = 4 * V, seed
    d = cell_shaped(COUPLING, seed, shape)
    s = d[d.role == "source"]
    k = d[d.role == "sink"]
    return {"seed": seed, "n_src": n_src, "n_sink": n_sink, "n_iso": n_iso,
            "A1_src": float(s.A1.median()), "A1_sink": float(k.A1.median()),
            "C1_src": float(s.C1.median()), "C1_sink": float(k.C1.median()),
            "A1_auc": auc(s.A1.values, k.A1.values),
            "C1_auc": auc(s.C1.values, k.C1.values)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"ratio sweep: V={V_TOTAL}+ghost, b=4V, coupling={COUPLING}, "
          f"alpha={ALPHA}\n")
    rows, t0 = [], time.time()
    for n_src, n_sink in RATIOS:
        ratio = n_sink / n_src
        cal = pd.DataFrame([run(n_src, n_sink, s) for s in CAL_SEEDS])
        test = pd.DataFrame([run(n_src, n_sink, s) for s in TEST_SEEDS])
        pd.concat([cal.assign(set="cal"), test.assign(set="test")]).to_csv(
            OUT / f"runs_{n_src}_{n_sink}.csv", index=False)
        barA = float(np.quantile(cal.A1_sink, 1 - ALPHA))
        barC = float(np.quantile(cal.C1_sink, 1 - ALPHA))
        rows.append({
            "n_src": n_src, "n_sink": n_sink, "ratio": ratio,
            "A1_bar": barA, "C1_bar": barC,
            "A1_sens": float((test.A1_src > barA).mean()),
            "C1_sens": float((test.C1_src > barC).mean()),
            "A1_auc": float(test.A1_auc.median()),
            "C1_auc": float(test.C1_auc.median())})
        r = rows[-1]
        print(f"  {n_src:>2} src / {n_sink:>2} sink  ratio {ratio:4.1f}   "
              f"A1 sens {r['A1_sens']:.2f} auc {r['A1_auc']:.3f}   "
              f"C1 sens {r['C1_sens']:.2f} auc {r['C1_auc']:.3f}   "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

    d = pd.DataFrame(rows).sort_values("ratio").reset_index(drop=True)
    d.to_csv(OUT / "results.csv", index=False)
    print(f"\n({(time.time()-t0)/60:.1f} min)\n")
    print(d.round(3).to_string(index=False))

    rho_a = d[["ratio", "A1_sens"]].corr(method="spearman").iloc[0, 1]
    rho_c = d[["ratio", "C1_auc"]].corr(method="spearman").iloc[0, 1]
    print(f"\nT1  marginal sensitivity vs ratio: Spearman {rho_a:+.3f}  "
          f"-> {'RISES' if rho_a > 0 else 'does NOT rise'}")
    print(f"T2  conditional AUC vs ratio:       Spearman {rho_c:+.3f}  "
          f"-> {'FALLS' if rho_c < 0 else 'does NOT fall'}")

    # T3: crossover in AUC, the metric both variants share
    diff = d.A1_auc - d.C1_auc
    signs = np.sign(diff.values)
    # a STRICT sign change; a tie (both variants at the same value, e.g. the
    # 1.000 ceiling) is a meeting, not a crossing
    cross = [(d.ratio.iloc[i], d.ratio.iloc[i + 1])
             for i in range(len(d) - 1) if signs[i] * signs[i + 1] < 0]
    tie = [d.ratio.iloc[i] for i in range(len(d)) if signs[i] == 0]
    print(f"\nT3  DECISIVE: A1_auc - C1_auc across the sweep: "
          + "  ".join(f"{r:.1f}:{v:+.3f}" for r, v in zip(d.ratio, diff)))
    if tie:
        print("    ties (both variants equal) at ratio "
              + ", ".join(f"{t:.1f}" for t in tie) + " - not crossings")

    print("\nVERDICT (rule fixed before running; T1, T2 and T3 all consulted)")
    if rho_a <= 0:
        print("   -> MECHANISM WRONG (T5). Marginal sensitivity does not "
              "rise with the ratio;\n      today's shape explanation is "
              "not a ratio effect and needs re-examination.")
    elif rho_c >= 0:
        print("   -> T2 FAILS: conditional AUC does not fall with the ratio. "
              "The declared rule has\n      no branch for this outcome, so "
              "no crossover verdict is available; the protocol's\n      "
              "mechanism (conditioning loses as sinks per source rise) is "
              "contradicted.")
    elif cross:
        lo, hi = cross[0]
        print(f"   -> CROSSOVER FOUND between ratio {lo:.1f} and {hi:.1f} "
              f"(grid spacing is the\n      precision). Below it prefer the "
              "conditional variant, above it the marginal\n      one - on "
              "one synthetic family at one coupling.")
    else:
        print("   -> DIRECTION ONLY. The curves do not cross within the "
              f"tested range\n      ({d.ratio.min():.1f} to "
              f"{d.ratio.max():.1f} sinks per source); direction confirmed, "
              "crossover outside range.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
