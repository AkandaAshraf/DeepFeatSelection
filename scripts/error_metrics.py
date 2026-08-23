"""Does the error metric decide what MACE can see?

Pre-registration: paper/error_metric_protocol.md, committed before this was
written or run.

Every statistic in this project is a difference of squared-error R2 from a
linear ridge probe, and that choice has never been varied. Three threshold
fixes for the sink-proxy confound have failed; this asks whether the proxy
and the source signal differ in KIND, which a threshold cannot reach but a
different measurement might.

One autoencoder per cell, every metric computed from it, so the comparison is
on identical codes and identical data.

    python scripts/error_metrics.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import source_outflow_gate as G  # noqa: E402
from source_outflow_coupling import coupled  # noqa: E402

OUT = Path("ExpOutput/error_metrics")
COUPLINGS = (0.30, 0.50, 0.70)
SEEDS = (0, 1, 2, 3, 4)
B, EPOCHS = 64, 25
RFF_D = 128
METRICS = ["A1_r2", "A2_nmae", "A3_spearman", "A4_gauss_nats",
           "A5_rff_r2", "B1_recon_mse", "B2_recon_mae", "C1_conditional"]


# ------------------------------------------------------------ scoring
def _ridge_pred(Xtr, ytr, Xte):
    Xt = torch.as_tensor(Xtr, dtype=torch.float64, device=G.DEV)
    yt = torch.as_tensor(ytr, dtype=torch.float64, device=G.DEV)
    Xe = torch.as_tensor(Xte, dtype=torch.float64, device=G.DEV)
    A = Xt.T @ Xt + G.ALPHA * torch.eye(Xt.shape[1], device=G.DEV,
                                        dtype=torch.float64)
    return (Xe @ torch.linalg.solve(A, Xt.T @ yt)).cpu().numpy()


def _rankdata(a):
    order = a.argsort(0)
    r = np.empty_like(order, dtype=np.float64)
    idx = np.arange(a.shape[0])[:, None] * np.ones((1, a.shape[1]))
    np.put_along_axis(r, order, idx, axis=0)
    return r


def score_all(pred, y):
    """Every FAMILY A score for one fit, as a dict. y is (n, d)."""
    resid = y - pred
    var = y.var(0) + 1e-12
    mse = (resid ** 2).mean(0)
    mae = np.abs(resid).mean(0)
    mae0 = np.abs(y - y.mean(0)).mean(0) + 1e-12
    pr, yr = _rankdata(pred), _rankdata(y)
    prc = pr - pr.mean(0)
    yrc = yr - yr.mean(0)
    sp = ((prc * yrc).sum(0)
          / (np.sqrt((prc ** 2).sum(0) * (yrc ** 2).sum(0)) + 1e-12))
    return {
        "A1_r2": float(np.clip(1 - mse / var, 0, None).mean()),
        "A2_nmae": float(np.clip(1 - mae / mae0, 0, None).mean()),
        "A3_spearman": float(np.nan_to_num(sp).mean()),
        "A4_gauss_nats": float((0.5 * np.log(var / (mse + 1e-12))
                                ).clip(0, None).mean()),
    }


def rff(a, seed, d=RFF_D):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((a.shape[1], d)) * 1.0
    b = rng.uniform(0, 2 * np.pi, d)
    return np.hstack([a, np.sqrt(2.0 / d) * np.cos(a @ W + b)])


# ------------------------------------------------------------ one cell
def cell(coupling, seed):
    G.BOTTLENECK, G.SEED = B, seed
    x, role = coupled(coupling=coupling, seed=seed)
    # ghost channel appended exactly as in the deployed gate
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
    allf = np.hstack(feats)

    rows = []
    zt_full = torch.as_tensor(zs, device=G.DEV)
    with torch.no_grad():
        recon_full = net(zt_full).cpu().numpy()
    err_full_sq = (recon_full - zs) ** 2
    err_full_ab = np.abs(recon_full - zs)

    for q in range(V):
        zq = G.codes_with_mask(net, zs, V, q=q)
        zh = G.code_history(zq)
        y_tr, y_te = zq[ti + 1], zq[si + 1]

        base = score_all(_ridge_pred(zh[ti], y_tr, zh[si]), y_te)
        aug = np.hstack([zh, feats[q]])
        with_q = score_all(_ridge_pred(aug[ti], y_tr, aug[si]), y_te)
        r = {k: with_q[k] - base[k] for k in base}

        # A5 nonlinear readout
        zb, za = rff(zh, seed), rff(aug, seed)
        r["A5_rff_r2"] = (
            score_all(_ridge_pred(za[ti], y_tr, za[si]), y_te)["A1_r2"]
            - score_all(_ridge_pred(zb[ti], y_tr, zb[si]), y_te)["A1_r2"])

        # C1 conditional: baseline already contains every OTHER channel
        others = np.hstack([zh] + [feats[j] for j in range(V) if j != q])
        r["C1_conditional"] = (
            score_all(_ridge_pred(np.hstack([others, feats[q]])[ti], y_tr,
                                  np.hstack([others, feats[q]])[si]),
                      y_te)["A1_r2"]
            - score_all(_ridge_pred(others[ti], y_tr, others[si]),
                        y_te)["A1_r2"])

        # FAMILY B: the network's own reconstruction error for OTHER channels
        zt = zt_full.clone()
        zt[:, q * G.E:(q + 1) * G.E] = 0.0
        with torch.no_grad():
            rec = net(zt).cpu().numpy()
        keep = np.ones(V * G.E, bool)
        keep[q * G.E:(q + 1) * G.E] = False
        r["B1_recon_mse"] = float(
            (((rec - zs) ** 2)[si][:, keep] - err_full_sq[si][:, keep]).mean())
        r["B2_recon_mae"] = float(
            ((np.abs(rec - zs))[si][:, keep]
             - err_full_ab[si][:, keep]).mean())

        r.update(coupling=coupling, seed=seed, channel=q, role=role[q])
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------ analysis
def auc(pos, neg):
    """P(a random positive outranks a random negative), ties at 0.5."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device: {G.DEV}  b={B}  epochs={EPOCHS}  "
          f"couplings {COUPLINGS}  seeds {SEEDS}")
    print(f"{len(METRICS)} metrics, one autoencoder per cell\n")
    t0, parts = time.time(), []
    for c in COUPLINGS:
        for s in SEEDS:
            t = time.time()
            parts.append(cell(c, s))
            print(f"  c={c} s={s}  ({time.time()-t:.0f}s)", flush=True)
    d = pd.concat(parts, ignore_index=True)
    d.to_csv(OUT / "channels.csv", index=False)
    print(f"\n({(time.time()-t0)/60:.1f} min)\n")

    # ---- M6 first: a metric with a dirty ghost is out, whatever its AUC --
    print("M6  ghost cleanliness, per metric on its own scale")
    alive = []
    for mt in METRICS:
        src = d[d.role == "source"][mt].values
        gh = d[d.role == "ghost"][mt].values
        p5 = float(np.percentile(src, 5))
        clean = float(np.median(gh)) < p5
        alive.append(clean)
        print(f"   {mt:16s} ghost {np.median(gh):+.5f}  "
              f"source p5 {p5:+.5f}   {'clean' if clean else 'DIRTY - excluded'}")
    keep = [m for m, k in zip(METRICS, alive) if k]

    # ---- AUC table -------------------------------------------------------
    print("\nSOURCE vs SINK AUC")
    hdr = "   " + f"{'metric':16s}" + "".join(f"{c:>9}" for c in COUPLINGS)
    print(hdr)
    tab = {}
    for mt in METRICS:
        row = []
        for c in COUPLINGS:
            g = d[d.coupling == c]
            row.append(auc(g[g.role == "source"][mt].values,
                           g[g.role == "sink"][mt].values))
        tab[mt] = row
        mark = "" if mt in keep else "   (ghost-excluded)"
        print(f"   {mt:16s}" + "".join(f"{v:>9.3f}" for v in row) + mark)
    pd.DataFrame(tab, index=[str(c) for c in COUPLINGS]).T.to_csv(
        OUT / "auc_source_vs_sink.csv")

    print("\nSOURCE vs ISOLATED AUC")
    print(hdr)
    for mt in METRICS:
        row = [auc(d[(d.coupling == c) & (d.role == "source")][mt].values,
                   d[(d.coupling == c) & (d.role == "isolated")][mt].values)
               for c in COUPLINGS]
        print(f"   {mt:16s}" + "".join(f"{v:>9.3f}" for v in row))

    # ---- M1 --------------------------------------------------------------
    a1 = tab["A1_r2"]
    print(f"\nM1  A1 reproduces? AUC {a1[1]:.3f} at 0.50, {a1[2]:.3f} at 0.70")
    m1 = a1[1] >= 0.80 and a1[2] < a1[1]
    print(f"   -> {'REPRODUCES' if m1 else 'DOES NOT REPRODUCE - VOID'}")
    if not m1:
        return 0

    # ---- M3, M5 ----------------------------------------------------------
    gap = max(abs(tab["A4_gauss_nats"][i] - a1[i]) for i in range(3))
    print(f"\nM3  A4 within 0.02 of A1? max difference {gap:.3f}  "
          f"-> {'as declared' if gap <= 0.02 else 'NOT degenerate'}")

    print("\nM5  C1 conditional: does it drive sinks to the ghost?")
    for c in COUPLINGS:
        g = d[d.coupling == c]
        print(f"   c={c}  source {g[g.role=='source'].C1_conditional.median():+.5f}"
              f"   sink {g[g.role=='sink'].C1_conditional.median():+.5f}"
              f"   ghost {g[g.role=='ghost'].C1_conditional.median():+.5f}")

    # ---- M2 DECISIVE ------------------------------------------------------
    print("\nM2  DECISIVE: any surviving metric with source-vs-sink AUC "
          ">= 0.90 at coupling 0.70?")
    best, best_v = None, -1.0
    for mt in keep:
        if tab[mt][2] > best_v:
            best, best_v = mt, tab[mt][2]
    print(f"   best surviving metric at 0.70: {best} at {best_v:.3f}")
    print(f"   baseline A1_r2 at 0.70: {a1[2]:.3f}")

    print("\nVERDICT (rule fixed before running)")
    if best_v >= 0.90:
        print(f"   -> {best} SEPARATES where R2 does not. The confound is a "
              "property of\n      squared error, not of the information. "
              "Reported with every other\n      metric beside it, as declared.")
    else:
        print("   -> M7. NO metric reaches 0.90. The sink proxy is "
              "informational, not an\n      artefact of squared error: no "
              "error measure distinguishes removing\n      information that "
              "mattered from removing a copy of it.\n      THE OUTFLOW LINE "
              "CLOSES on that ground, as declared in advance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
