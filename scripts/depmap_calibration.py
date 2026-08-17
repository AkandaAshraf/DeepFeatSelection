"""Does observational redundancy predict interventional equivalence?
Phase 1: the pan-cancer calibration curve, with controls.

Pre-registration: paper/depmap_protocol.md (declared 2026-08-16, addendum
A1-A6, A5 gate closed 2026-08-17 with no published forward calibration
found). Release pinned: DepMap 24Q4 Public, figshare article 27993248.

DECLARED IMPLEMENTATION CONSTANTS (fixed before the expression matrix was
opened; marginals may set percentile-based floors but every rule is written
here first):

  join        on ModelID present in Model.csv, expression and Chronos
  expression  keep genes with mean log2(TPM+1) >= 1.0 and std >= the 20th
              percentile of std among those expressed genes
  dependency  a gene "has a phenotype" if Chronos <= -0.5 in >= 3 lines (A1);
              pairs where NEITHER gene has a phenotype form the declared
              phenotype-free stratum, reported separately, never pooled
  pan-ess.    for the without-pan-essential arm, drop genes whose Chronos
              std is below the 20th percentile (near-flat everywhere)
  lineage     one-hot OncotreeLineage with >= 8 lines; smaller lineages pool
              into 'other'; regressed out of BOTH matrices (confound 1);
              curves reported corrected AND uncorrected
  r_obs       squared Pearson correlation of expression residuals
  e_int       signed Pearson correlation of Chronos residuals
  tau         0.5 (headline: P(e_int > 0.5 | r_obs > 0.8))
  bins        r_obs: 20 bins of width 0.05; e_int: 41 bins of width 0.05
              from -1.025 to +1.025
  paralogs    proxy flag (A2): shared alphabetic symbol root of length >= 3
              after stripping trailing digits/single letters (e.g. RPL13,
              RPL13A share root RPL); curves reported with the flag in and
              out; cytoband flag pending a location file and said so
  ghost       Chronos residual matrix row-permuted with seed 0, identical
              pipeline; its curve must be flat (void condition)
  CIs         bootstrap over GENES, B=100, seed 0 (A4); pair-level
              resampling nowhere
  GPU         torch CUDA; chunked so no full pair matrix is materialised

Wall-clock is reported per stage.

    python scripts/depmap_calibration.py            # phases A+B (~minutes)
    python scripts/depmap_calibration.py --boot     # phase C bootstrap
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DATA = Path("Data/depmap_24q4")
OUT = Path("ExpOutput/depmap_calibration")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
TAU = 0.5
R_EDGES = np.linspace(0, 1, 21)            # r_obs bins, width 0.05
E_EDGES = np.linspace(-1.025, 1.025, 42)   # e_int bins, width 0.05
EXPR_MEAN_MIN = 1.0
STD_PCTL = 20
DEP_THRESH, DEP_MIN_LINES = -0.5, 3
LIN_MIN = 8
CHUNK = 2048
BOOT_B = 100


def log(msg: str) -> None:
    print(msg, flush=True)


def symbol_root(sym: str) -> str:
    s = re.sub(r"\d+$", "", sym)
    s = re.sub(r"[A-Z]$", "", s) if len(s) > 3 else s
    return s if len(s) >= 3 else sym


def load_join():
    t0 = time.time()
    model = pd.read_csv(DATA / "Model.csv", low_memory=False)
    expr = pd.read_csv(DATA / "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
                       index_col=0)
    chron = pd.read_csv(DATA / "CRISPRGeneEffect.csv", index_col=0)
    # column names are "SYMBOL (ENTREZ)"; reduce to SYMBOL, drop duplicates
    for df in (expr, chron):
        df.columns = [c.split(" (")[0] for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
    expr = expr.loc[:, ~expr.columns.duplicated()]
    chron = chron.loc[:, ~chron.columns.duplicated()]
    lines = sorted(set(expr.index) & set(chron.index) & set(model.ModelID))
    genes = sorted(set(expr.columns) & set(chron.columns))
    expr = expr.loc[lines, genes]
    chron = chron.loc[lines, genes]
    meta = model.set_index("ModelID").loc[lines]
    log(f"joined: {len(lines)} lines x {len(genes)} genes "
        f"({time.time()-t0:.0f}s)")
    return expr, chron, meta


def residualise(mat: np.ndarray, dummies: np.ndarray) -> np.ndarray:
    """Regress lineage dummies out of every gene column, on GPU."""
    X = torch.as_tensor(dummies, dtype=torch.float32, device=DEV)
    Y = torch.as_tensor(mat, dtype=torch.float32, device=DEV)
    beta = torch.linalg.lstsq(X, Y).solution
    R = (Y - X @ beta).cpu().numpy()
    return R


def zscore(mat: np.ndarray) -> np.ndarray:
    mu = mat.mean(0, keepdims=True)
    sd = mat.std(0, keepdims=True) + 1e-12
    return ((mat - mu) / sd).astype(np.float32)


def joint_hist(ez: np.ndarray, kz: np.ndarray, keep: np.ndarray,
               weights: np.ndarray | None = None,
               gene_perm: np.ndarray | None = None) -> np.ndarray:
    """2-D histogram over gene pairs: r_obs bin x e_int bin.

    ez, kz: (n_lines, G) z-scored residuals. keep: boolean mask (G,) selecting
    the gene universe. weights: per-gene multiplicities for bootstrap (None ->
    ones). Chunked so no G x G matrix is materialised. Counts each unordered
    pair twice; callers divide by 2.
    """
    idx = np.where(keep)[0]
    # A7 ghost: relabel the Chronos gene columns by a fixed permutation of the
    # kept set, so pair (A,B) in expression is scored against (sigma(A),
    # sigma(B)) in knockouts. Row permutation is a no-op for within-matrix
    # correlations and was retired as vacuous; see the protocol.
    k_idx = idx if gene_perm is None else idx[gene_perm]
    E = torch.as_tensor(ez[:, idx], device=DEV)
    K = torch.as_tensor(kz[:, k_idx], device=DEV)
    n = E.shape[0]
    G = E.shape[1]
    w = (torch.ones(G, device=DEV) if weights is None
         else torch.as_tensor(weights[idx], dtype=torch.float32, device=DEV))
    re_edges = torch.as_tensor(R_EDGES, device=DEV)
    ee_edges = torch.as_tensor(E_EDGES, device=DEV)
    hist = torch.zeros((len(R_EDGES) - 1) * (len(E_EDGES) - 1), device=DEV)
    for s in range(0, G, CHUNK):
        e_blk = E[:, s:s + CHUNK]
        r = (e_blk.T @ E) / (n - 1)          # (c, G) expression corr
        q = (K[:, s:s + CHUNK].T @ K) / (n - 1)
        robs = (r ** 2).clamp(0, 0.999999)
        a = torch.bucketize(robs, re_edges) - 1
        b = torch.bucketize(q.clamp(-1.024, 1.024), ee_edges) - 1
        flat = (a.clamp(0, len(R_EDGES) - 2) * (len(E_EDGES) - 1)
                + b.clamp(0, len(E_EDGES) - 2))
        wpair = w[s:s + CHUNK, None] * w[None, :]
        # zero out self-pairs
        c = e_blk.shape[1]
        rows = torch.arange(s, s + c, device=DEV)
        wpair[torch.arange(c, device=DEV), rows] = 0.0
        hist.scatter_add_(0, flat.reshape(-1), wpair.reshape(-1))
    return hist.reshape(len(R_EDGES) - 1, len(E_EDGES) - 1).cpu().numpy() / 2


def prox_hist(ez: np.ndarray, k_raw: np.ndarray, keep: np.ndarray,
              gene_perm: np.ndarray | None = None) -> np.ndarray:
    """r_obs bin x e_prox bin histogram (A8).

    e_prox = 2<ka,kb>/(<ka,ka>+<kb,kb>) on RAW uncentred Chronos, so the
    shared-mean component of uniform co-essentiality is kept as signal.
    """
    idx = np.where(keep)[0]
    k_idx = idx if gene_perm is None else idx[gene_perm]
    E = torch.as_tensor(ez[:, idx], device=DEV)
    K = torch.as_tensor(k_raw[:, k_idx], dtype=torch.float32, device=DEV)
    n = E.shape[0]
    G = E.shape[1]
    norms = (K * K).sum(0)
    re_edges = torch.as_tensor(R_EDGES, device=DEV)
    ee_edges = torch.as_tensor(E_EDGES, device=DEV)
    hist = torch.zeros((len(R_EDGES) - 1) * (len(E_EDGES) - 1), device=DEV)
    for s0 in range(0, G, CHUNK):
        e_blk = E[:, s0:s0 + CHUNK]
        r = (e_blk.T @ E) / (n - 1)
        dot = K[:, s0:s0 + CHUNK].T @ K
        prox = (2 * dot / (norms[s0:s0 + CHUNK, None] + norms[None, :] + 1e-12))
        robs = (r ** 2).clamp(0, 0.999999)
        a = torch.bucketize(robs, re_edges) - 1
        b = torch.bucketize(prox.clamp(-1.024, 1.024), ee_edges) - 1
        flat = (a.clamp(0, len(R_EDGES) - 2) * (len(E_EDGES) - 1)
                + b.clamp(0, len(E_EDGES) - 2))
        w = torch.ones_like(prox)
        c = e_blk.shape[1]
        w[torch.arange(c, device=DEV), torch.arange(s0, s0 + c, device=DEV)] = 0
        hist.scatter_add_(0, flat.reshape(-1), w.reshape(-1))
    return hist.reshape(len(R_EDGES) - 1, len(E_EDGES) - 1).cpu().numpy() / 2


def curve_from_hist(h: np.ndarray, tau: float = None) -> pd.DataFrame:
    if tau is None:
        tau = TAU
    e_centers = (E_EDGES[:-1] + E_EDGES[1:]) / 2
    rows = []
    for i in range(h.shape[0]):
        tot = h[i].sum()
        hi = h[i][e_centers > tau].sum()
        rows.append({"r_lo": R_EDGES[i], "r_hi": R_EDGES[i + 1],
                     "pairs": tot, "p_equiv": hi / tot if tot else np.nan,
                     "mean_e": (h[i] * e_centers).sum() / tot if tot else np.nan})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"device: {DEV}")

    expr, chron, meta = load_join()
    genes = np.array(expr.columns)

    # ---- filters (rules in header) ---------------------------------------
    e_np = expr.to_numpy(np.float64)
    k_np = chron.to_numpy(np.float64)
    nan_frac = np.isnan(k_np).mean(0)
    k_np = np.where(np.isnan(k_np), np.nanmean(k_np, axis=0, keepdims=True), k_np)
    expressed = e_np.mean(0) >= EXPR_MEAN_MIN
    std_floor = np.percentile(e_np[:, expressed].std(0), STD_PCTL)
    keep_e = expressed & (e_np.std(0) >= std_floor) & (nan_frac < 0.1)
    kd_std_floor = np.percentile(k_np.std(0), STD_PCTL)
    pan_flat = k_np.std(0) < kd_std_floor
    has_dep = (k_np <= DEP_THRESH).sum(0) >= DEP_MIN_LINES
    keep = keep_e
    log(f"filters: expressed+variable {keep_e.sum()}, of which "
        f"has-phenotype {int((keep_e & has_dep).sum())}, "
        f"pan-flat {int((keep_e & pan_flat).sum())}")

    # ---- lineage dummies --------------------------------------------------
    lin = meta.OncotreeLineage.fillna("other")
    counts = lin.value_counts()
    lin = lin.where(lin.map(counts) >= LIN_MIN, "other")
    dummies = pd.get_dummies(lin, drop_first=False).to_numpy(np.float32)
    dummies = np.hstack([np.ones((len(lin), 1), np.float32), dummies])
    log(f"lineages kept: {int((counts >= LIN_MIN).sum())} "
        f"(+other), lines {len(lin)}")

    # ---- residualise + z-score -------------------------------------------
    t0 = time.time()
    ez_c = zscore(residualise(e_np, dummies))
    kz_c = zscore(residualise(k_np, dummies))
    ez_u = zscore(e_np)
    kz_u = zscore(k_np)
    log(f"residualised ({time.time()-t0:.0f}s)")

    # paralog proxy flag needs pair-level knowledge; handled as a stratum by
    # restricting the gene universe: curves excluding whole families vs all.
    roots = np.array([symbol_root(s) for s in genes])
    fam_sizes = pd.Series(roots).value_counts()
    in_family = pd.Series(roots).map(fam_sizes).to_numpy() > 1

    strata = {
        "corrected": (ez_c, kz_c, keep),
        "uncorrected": (ez_u, kz_u, keep),
        "corrected_hasdep": (ez_c, kz_c, keep & has_dep),
        "corrected_nopanflat": (ez_c, kz_c, keep & ~pan_flat),
        "corrected_nofamily": (ez_c, kz_c, keep & ~in_family),
    }

    results = {}
    for name, (ez, kz, mask) in strata.items():
        t0 = time.time()
        h = joint_hist(ez, kz, mask)
        c = curve_from_hist(h)
        c.to_csv(OUT / f"curve_{name}.csv", index=False)
        np.save(OUT / f"hist_{name}.npy", h)
        top = c[c.r_lo >= 0.8]
        p_top = (top.pairs * top.p_equiv).sum() / max(top.pairs.sum(), 1)
        results[name] = dict(
            genes=int(mask.sum()), pairs=float(h.sum()),
            p_equiv_r80=float(p_top), seconds=round(time.time() - t0, 1))
        log(f"[{name:22s}] genes={mask.sum():5d} "
            f"P(e>{TAU}|r>0.8)={p_top:.3f}  ({results[name]['seconds']}s)")

    # ---- ghost (A7): permute Chronos GENE labels; must be flat ------------
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(int(keep.sum()))
    h_ghost = joint_hist(ez_c, kz_c, keep, gene_perm=perm)
    curve_from_hist(h_ghost).to_csv(OUT / "curve_ghost.csv", index=False)
    gtop = curve_from_hist(h_ghost)
    gtop = gtop[gtop.r_lo >= 0.8]
    g_p = (gtop.pairs * gtop.p_equiv).sum() / max(gtop.pairs.sum(), 1)
    results["ghost"] = dict(p_equiv_r80=float(g_p))
    log(f"[ghost                 ] P(e>{TAU}|r>0.8)={g_p:.4f}  (must be ~0)")

    # ---- reverse conditional (prediction 3) -------------------------------
    h = np.load(OUT / "hist_corrected.npy")
    e_centers = (E_EDGES[:-1] + E_EDGES[1:]) / 2
    r_centers = (R_EDGES[:-1] + R_EDGES[1:]) / 2
    hi_e = h[:, e_centers > TAU]
    p_rev = hi_e[r_centers > 0.8, :].sum() / max(hi_e.sum(), 1)
    fwd = results["corrected"]["p_equiv_r80"]
    results["reverse_conditional"] = dict(
        p_robs80_given_e=float(p_rev), p_e_given_robs80=fwd)
    log(f"asymmetry: P(r>0.8|e>{TAU})={p_rev:.3f} vs "
        f"P(e>{TAU}|r>0.8)={fwd:.3f}")

    # ---- A8: the e_prox axis over the declared strata ---------------------
    TAU_PROX = 0.8
    prox_strata = {
        "prox_all": keep,
        "prox_hasdep": keep & has_dep,
        "prox_panflat": keep & pan_flat,
        "prox_nopanflat": keep & ~pan_flat,
    }
    for name, mask in prox_strata.items():
        t0 = time.time()
        hp = prox_hist(ez_c, k_np, mask)
        cp = curve_from_hist(hp, tau=TAU_PROX)
        cp.to_csv(OUT / f"curve_{name}.csv", index=False)
        np.save(OUT / f"hist_{name}.npy", hp)
        tp = cp[cp.r_lo >= 0.8]
        pt = (tp.pairs * tp.p_equiv).sum() / max(tp.pairs.sum(), 1)
        results[name] = dict(genes=int(mask.sum()),
                             p_prox_r80=float(pt),
                             seconds=round(time.time() - t0, 1))
        log(f"[{name:22s}] genes={mask.sum():5d} "
            f"P(prox>{TAU_PROX}|r>0.8)={pt:.3f}  ({results[name]['seconds']}s)")
    hpg = prox_hist(ez_c, k_np, keep, gene_perm=perm)
    cpg = curve_from_hist(hpg, tau=TAU_PROX)
    cpg.to_csv(OUT / "curve_prox_ghost.csv", index=False)
    gt = cpg[cpg.r_lo >= 0.8]
    results["prox_ghost"] = dict(p_prox_r80=float(
        (gt.pairs * gt.p_equiv).sum() / max(gt.pairs.sum(), 1)))
    log(f"[prox ghost            ] P(prox>{TAU_PROX}|r>0.8)="
        f"{results['prox_ghost']['p_prox_r80']:.4f} "
        f"(flat-at-base-rate required)")

    # positive control on the prox axis (A8 gate)
    # computed below with the CORUM block

    # ---- sensitivity arm (A7): Spearman e_int, labelled as such -----------
    def rank_z(mat: np.ndarray) -> np.ndarray:
        r = mat.argsort(0).argsort(0).astype(np.float64)
        return zscore(r)

    kz_sp = rank_z(residualise(k_np, dummies))
    h_sp = joint_hist(ez_c, kz_sp, keep)
    curve_from_hist(h_sp).to_csv(OUT / "curve_spearman.csv", index=False)
    csp = curve_from_hist(h_sp)
    tsp = csp[csp.r_lo >= 0.8]
    results["spearman_sensitivity"] = dict(p_equiv_r80=float(
        (tsp.pairs * tsp.p_equiv).sum() / max(tsp.pairs.sum(), 1)),
        total_hits=float((csp.pairs * csp.p_equiv).sum()))
    log(f"[spearman sensitivity  ] total e>{TAU} hits="
        f"{results['spearman_sensitivity']['total_hits']:.0f} "
        f"(Pearson: {(curve_from_hist(np.load(OUT / 'hist_corrected.npy')).pipe(lambda c: (c.pairs*c.p_equiv).sum())):.0f})")

    # ---- positive control (A3): curated complexes on UNFILTERED matrices --
    from depmap_audit import COMPLEXES, CONTROLS
    gene_pos = {g: i for i, g in enumerate(genes)}
    pos_pairs = [(a, b) for mem in COMPLEXES.values()
                 for i, a in enumerate(mem) for b in mem[i + 1:]
                 if a in gene_pos and b in gene_pos]
    neg_pool = [g for g in CONTROLS if g in gene_pos]

    def pair_vals(pairs, ez, kz):
        out = []
        for a, b in pairs:
            ia, ib = gene_pos[a], gene_pos[b]
            r = float(np.corrcoef(ez[:, ia], ez[:, ib])[0, 1]) ** 2
            e = float(np.corrcoef(kz[:, ia], kz[:, ib])[0, 1])
            ka, kb = k_np[:, ia], k_np[:, ib]
            prox = float(2 * np.dot(ka, kb)
                         / (np.dot(ka, ka) + np.dot(kb, kb) + 1e-12))
            out.append((a, b, r, e, prox))
        return pd.DataFrame(out,
                            columns=["a", "b", "r_obs", "e_int", "e_prox"])

    corum = pair_vals(pos_pairs, ez_u, kz_u)
    corum.to_csv(OUT / "control_corum.csv", index=False)
    ok_pos = float((corum.e_int > TAU).mean())
    ok_prox = float((corum.e_prox > 0.8).mean())
    log(f"[positive control      ] {len(corum)} complex pairs: "
        f"P(e_sel>{TAU})={ok_pos:.2f}  P(e_prox>0.8)={ok_prox:.2f}  "
        f"median prox={corum.e_prox.median():.2f}")
    results["control_corum"] = dict(n=len(corum), p_sel=ok_pos,
                                    p_prox=ok_prox,
                                    median_prox=float(corum.e_prox.median()))

    # ---- negative control: matched random pairs ---------------------------
    nrng = np.random.default_rng(SEED + 9)
    keep_idx = np.where(keep)[0]
    rnd = [(genes[i], genes[j]) for i, j in zip(
        nrng.choice(keep_idx, 2000), nrng.choice(keep_idx, 2000)) if i != j]
    negc = pair_vals(rnd, ez_c, kz_c)
    negc.to_csv(OUT / "control_negative.csv", index=False)
    log(f"[negative control      ] {len(negc)} random pairs: "
        f"P(e>{TAU})={float((negc.e_int > TAU).mean()):.4f}  "
        f"median e={negc.e_int.median():+.3f}")
    results["control_negative"] = dict(
        n=len(negc), p_equiv=float((negc.e_int > TAU).mean()))

    json.dump(results, open(OUT / "summary.json", "w"), indent=1)

    # ---- phase C: gene bootstrap ------------------------------------------
    if args.boot:
        t0 = time.time()
        curves = []
        for b in range(BOOT_B):
            brng = np.random.default_rng(SEED + 1 + b)
            w = np.bincount(brng.integers(0, len(genes), len(genes)),
                            minlength=len(genes)).astype(np.float32)
            hb = joint_hist(ez_c, kz_c, keep, weights=w)
            cb = curve_from_hist(hb)
            cb["boot"] = b
            curves.append(cb)
            if (b + 1) % 10 == 0:
                log(f"  bootstrap {b+1}/{BOOT_B} "
                    f"({(time.time()-t0)/60:.1f} min)")
        pd.concat(curves).to_csv(OUT / "bootstrap_curves.csv", index=False)
        log(f"bootstrap done ({(time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
