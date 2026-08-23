"""Conditional outflow as the primary hypothesis, on fresh seeds.

Pre-registration: paper/conditional_outflow_protocol.md, committed before
this was written or run.

A1 and C1 differ ONLY in the conditioning set - same autoencoder, same codes,
same data, same ridge, same squared-error scoring - so any difference between
them is attributable to conditioning and to nothing else. The redundancy axis
is part of the design because it tests the risk C1 carries: if a source is
recoverable from its own sinks, conditioning should zero the source too.

    python scripts/conditional_outflow.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import source_outflow_gate as G  # noqa: E402
from error_metrics import _ridge_pred, auc, score_all  # noqa: E402

OUT = Path("ExpOutput/conditional_outflow")
COUPLINGS = (0.30, 0.50, 0.70)
COPIES = (0, 1, 2)
SEEDS = tuple(range(5, 15))          # fresh; 0-4 are spent
B, EPOCHS, COPY_NOISE = 64, 25, 0.02


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


def cell(coupling, seed, copies):
    G.BOTTLENECK, G.SEED = B, seed
    x, role = system(coupling, seed, copies)
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

        rows.append({"coupling": coupling, "copies": copies, "seed": seed,
                     "channel": q, "role": role[q], "A1": a1, "C1": c1})
    return pd.DataFrame(rows)


def aucs(d, stat):
    return [auc(d[(d.coupling == c) & (d.role == "source")][stat].values,
               d[(d.coupling == c) & (d.role == "sink")][stat].values)
            for c in COUPLINGS]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device: {G.DEV}  b={B}  couplings {COUPLINGS}  copies {COPIES}")
    print(f"seeds {SEEDS[0]}-{SEEDS[-1]} (fresh; 0-4 are spent)\n")
    t0, parts = time.time(), []
    for k in COPIES:
        for c in COUPLINGS:
            for s in SEEDS:
                parts.append(cell(c, s, k))
            print(f"  k={k} c={c} done  ({(time.time()-t0)/60:.1f} min)",
                  flush=True)
    d = pd.concat(parts, ignore_index=True)
    d.to_csv(OUT / "channels.csv", index=False)
    print(f"\n({(time.time()-t0)/60:.1f} min)\n")

    d0 = d[d.copies == 0]
    A, C = aucs(d0, "A1"), aucs(d0, "C1")

    print("SOURCE vs SINK AUC at k=0        " +
          "".join(f"{c:>9}" for c in COUPLINGS))
    print("   A1 marginal (current)        " + "".join(f"{v:>9.3f}" for v in A))
    print("   C1 conditional               " + "".join(f"{v:>9.3f}" for v in C))
    print("   margin C1-A1                 " +
          "".join(f"{c_-a_:>+9.3f}" for a_, c_ in zip(A, C)))

    print(f"\nP1  direction only: does A1 degrade from 0.30 to 0.70?")
    p1 = A[2] < A[0]
    print(f"   {A[0]:.3f} -> {A[2]:.3f}   "
          f"{'REPRODUCES' if p1 else 'DOES NOT - VOID'}")
    if not p1:
        return 0

    print("\nP2  PRIMARY: C1 above A1 at all three, and the margin grows?")
    above = all(c_ > a_ for a_, c_ in zip(A, C))
    grows = (C[2] - A[2]) > (C[0] - A[0])
    print(f"   above at all three: {'yes' if above else 'NO'}   "
          f"margin {C[0]-A[0]:+.3f} -> {C[2]-A[2]:+.3f}: "
          f"{'grows' if grows else 'DOES NOT GROW'}")
    p2 = above and grows

    print("\nP3  does C1 place sinks on the ISOLATED reference?")
    p3 = True
    for c in COUPLINGS:
        g = d0[d0.coupling == c]
        sk, iso = g[g.role == "sink"].C1, g[g.role == "isolated"].C1
        iqr = float(iso.quantile(0.75) - iso.quantile(0.25))
        ok = abs(float(sk.median() - iso.median())) <= iqr
        p3 &= ok
        print(f"   c={c}  sink {sk.median():+.5f}  isolated {iso.median():+.5f}"
              f"  |diff| {abs(sk.median()-iso.median()):.5f} vs IQR {iqr:.5f}"
              f"   {'PASS' if ok else 'FAIL'}")

    print("\nP4  THE DECLARED RISK: does C1 fall as copies are added?")
    print("   k    " + "".join(f"{c:>9}" for c in COUPLINGS) + "     (C1 AUC)")
    ck = {}
    for k in COPIES:
        ck[k] = aucs(d[d.copies == k], "C1")
        ak = aucs(d[d.copies == k], "A1")
        print(f"   {k}    " + "".join(f"{v:>9.3f}" for v in ck[k]) +
              "     A1: " + "".join(f"{v:>7.3f}" for v in ak))
    fell = np.mean(ck[2]) < np.mean(ck[0])
    worse = np.mean(ck[2]) < np.mean(aucs(d[d.copies == 2], "A1"))
    print(f"   k=2 mean {np.mean(ck[2]):.3f} vs k=0 mean {np.mean(ck[0]):.3f}"
          f"  -> {'FALLS, risk realised' if fell else 'does not fall'}")
    if fell:
        print(f"   below A1 at k=2? {'YES - an operating limit' if worse else 'no'}")

    print("\nP5  ghost clean on each statistic's own scale?")
    p5 = True
    for st in ("A1", "C1"):
        gh = float(d[d.role == "ghost"][st].median())
        p5_ = gh < float(np.percentile(d[d.role == "source"][st], 5))
        p5 &= p5_
        print(f"   {st}  ghost {gh:+.5f}   {'clean' if p5_ else 'DIRTY'}")

    print("\nVERDICT (rule fixed before running)")
    if p2 and p3 and p5 and not fell:
        print("   -> CONFIRMED. Conditioning removes the sink-proxy confound. "
              "The source-\n      detection line reopens on a measurement, "
              "not a threshold.")
    elif p2 and p3 and p5:
        print("   -> CONFIRMED WITH LIMIT. Conditioning works at k=0 and "
              "degrades as sources\n      become recoverable from their "
              "copies. The envelope is stated above.")
    else:
        print("   -> NOT CONFIRMED. The sweep's column does not survive fresh "
              "seeds and a\n      declared bar. The line stays shut.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
