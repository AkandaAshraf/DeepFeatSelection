"""DepMap phase 2: the three declared arms that finish the calibration study.

All three reuse the phase-1 measures unchanged (imported, not reimplemented),
so nothing here can drift from the numbers already published.

  A. TNBC arm (depmap_protocol.md "TNBC case study, and its declared
     weakness"). Triple-negative is defined by LOW ESR1/PGR/ERBB2 expression
     among breast lines, thresholds taken from the marginal expression
     distributions of the breast lines only, with no dependency data
     consulted. Declared underpowered in advance; reported as such.

  B. Many-to-one arm. The pairwise question asked whether ONE partner is
     interventionally equivalent. This asks whether a gene whose expression
     is predictable from a PANEL of others -- r_obs(A | rest), the honest
     form of "this gene is redundant" -- is likelier to have an equivalent
     partner than best-pair redundancy alone implies.

  C. Real paralog flag (A2). The published curves used an alphabetic
     symbol-root proxy. This replaces it with HGNC gene groups and re-reports
     the in-family / out-of-family split, so the proxy's error is measured
     rather than assumed.

    python scripts/depmap_arms.py            # all three
    python scripts/depmap_arms.py --arm b
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from depmap_calibration import (  # noqa: E402  measures reused verbatim
    CHUNK, DEV, E_EDGES, EXPR_MEAN_MIN, OUT, R_EDGES, STD_PCTL, TAU,
    curve_from_hist, load_join, log, prox_hist, residualise, symbol_root,
    zscore,
)

HGNC_URL = ("https://storage.googleapis.com/public-download-files/hgnc/"
            "tsv/tsv/hgnc_complete_set.txt")
HGNC_LOCAL = Path("Data/depmap_24q4/hgnc_complete_set.txt")
TOPK = 10          # panel size for r_obs(A | panel), declared here
RIDGE = 1.0


def filters(e_np: np.ndarray, k_np: np.ndarray):
    """The phase-1 gene filter, applied to whatever line subset is passed."""
    expressed = e_np.mean(0) >= EXPR_MEAN_MIN
    if expressed.sum() == 0:
        return expressed
    std_floor = np.percentile(e_np[:, expressed].std(0), STD_PCTL)
    return expressed & (e_np.std(0) >= std_floor)


def lineage_dummies(meta: pd.DataFrame, min_lines: int = 8) -> np.ndarray:
    lin = meta.OncotreeLineage.fillna("other")
    counts = lin.value_counts()
    lin = lin.where(lin.map(counts) >= min_lines, "other")
    d = pd.get_dummies(lin, drop_first=False).to_numpy(np.float32)
    return np.hstack([np.ones((len(lin), 1), np.float32), d])


# ---------------------------------------------------------------------------
# Arm A: TNBC
# ---------------------------------------------------------------------------

def arm_tnbc(expr, chron, meta, genes) -> pd.DataFrame:
    log("=" * 70)
    log("ARM A: TNBC case study (declared underpowered before running)")
    breast = meta.index[meta.OncotreeLineage.fillna("") == "Breast"]
    log(f"breast lines: {len(breast)}")
    if len(breast) < 20:
        log("  too few breast lines; arm reported as not runnable")
        return pd.DataFrame()

    # thresholds from the marginal expression of breast lines ONLY.
    marks = ["ESR1", "PGR", "ERBB2"]
    missing = [m for m in marks if m not in expr.columns]
    if missing:
        log(f"  receptor genes missing: {missing}; arm not runnable")
        return pd.DataFrame()
    sub = expr.loc[breast, marks]
    # Definition, and why it is not the tertile rule first tried. A lower-
    # tertile cut on all three receptors recovered only 4 of the 19 documented
    # triple-negative lines in this panel: the definition failed its positive
    # control, so per the standing rule the definition was replaced, not the
    # biology reinterpreted. ESR1/PGR use the conventional not-expressed line
    # (log2(TPM+1) <= 1, i.e. about 1 TPM) and ERBB2 the breast-line median,
    # since HER2+ lines carry the amplified mode. This recovers 10 of 19 known
    # TNBC lines and admits 0 of the 8 documented receptor-positive lines:
    # conservative, uncontaminated, and fixed by cell-line literature alone
    # with no dependency data consulted.
    cut = pd.Series({"ESR1": 1.0, "PGR": 1.0,
                     "ERBB2": float(sub.ERBB2.median())})
    log("  receptor cutoffs: "
        + ", ".join(f"{m}<={cut[m]:.2f}" for m in marks))
    is_tn = (sub <= cut).all(axis=1)
    tn_lines = list(sub.index[is_tn])
    log(f"  triple-negative lines: {len(tn_lines)} of {len(breast)} breast")
    if len(tn_lines) < 15:
        log("  n < 15: arm reported as not supporting a claim (declared rule)")
        return pd.DataFrame()

    rows = []
    for name, lines in (("tnbc", tn_lines), ("breast_all", list(breast))):
        e = expr.loc[lines].to_numpy(np.float64)
        k = chron.loc[lines].to_numpy(np.float64)
        k = np.where(np.isnan(k), np.nanmean(k, axis=0, keepdims=True), k)
        keep = filters(e, k)
        # lineage correction is meaningless inside one lineage; centre only
        ez = zscore(e - e.mean(0, keepdims=True))
        curve = curve_from_hist(prox_hist(ez, k, keep), tau=0.8)
        curve["arm"] = name
        curve["n_lines"] = len(lines)
        curve.to_csv(OUT / f"curve_{name}.csv", index=False)
        tot, hi = curve.pairs.sum(), (curve.pairs * curve.p_equiv).sum()
        base = hi / tot
        top = curve[curve.r_lo >= 0.60]
        ceil = ((top.pairs * top.p_equiv).sum() / top.pairs.sum()
                if top.pairs.sum() else np.nan)
        log(f"  [{name}] lines={len(lines)} genes={int(keep.sum())} "
            f"base={base:.4f} ceiling(r>=0.6)={ceil:.4f} "
            f"pairs>=0.6: {int(top.pairs.sum())}")
        rows.append({"arm": name, "n_lines": len(lines),
                     "n_genes": int(keep.sum()), "base": base,
                     "ceiling": ceil, "pairs_high": int(top.pairs.sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Arm B: many-to-one
# ---------------------------------------------------------------------------

def arm_many_to_one(ez: np.ndarray, k_raw: np.ndarray,
                    keep: np.ndarray, genes: np.ndarray) -> pd.DataFrame:
    """Per gene: best-pair r_obs, panel R^2 from top-K partners, and whether
    any partner is interventionally equivalent (e_prox > TAU_PROX)."""
    log("=" * 70)
    log(f"ARM B: many-to-one redundancy (panel of top {TOPK})")
    idx = np.where(keep)[0]
    E = torch.as_tensor(ez[:, idx], device=DEV)
    K = torch.as_tensor(k_raw[:, idx], dtype=torch.float32, device=DEV)
    n, G = E.shape
    norms = (K * K).sum(0)
    best_r2 = torch.zeros(G, device=DEV)
    has_equiv = torch.zeros(G, device=DEV)
    top_idx = torch.zeros((G, TOPK), dtype=torch.long, device=DEV)
    t0 = time.time()
    for s in range(0, G, CHUNK):
        c = min(CHUNK, G - s)
        r = (E[:, s:s + c].T @ E) / (n - 1)
        rows = torch.arange(c, device=DEV)
        cols = torch.arange(s, s + c, device=DEV)
        r2 = (r ** 2).clamp(0, 0.999999)
        r2[rows, cols] = -1.0                     # exclude self
        vals, order = torch.topk(r2, TOPK, dim=1)
        best_r2[s:s + c] = vals[:, 0]
        top_idx[s:s + c] = order
        dot = K[:, s:s + c].T @ K
        prox = 2 * dot / (norms[s:s + c, None] + norms[None, :] + 1e-12)
        prox[rows, cols] = -1.0
        has_equiv[s:s + c] = (prox > TAU).any(1).float()
    log(f"  pairwise pass ({time.time()-t0:.0f}s)")

    # panel R^2: ridge regression of each gene on its top-K partners
    panel_r2 = torch.zeros(G, device=DEV)
    eye = torch.eye(TOPK, device=DEV) * RIDGE
    t0 = time.time()
    for s in range(0, G, 512):
        c = min(512, G - s)
        P = E[:, top_idx[s:s + c]].permute(1, 0, 2)        # (c, n, K)
        y = E[:, s:s + c].T.unsqueeze(2)                   # (c, n, 1)
        A = P.transpose(1, 2) @ P + eye
        w = torch.linalg.solve(A, P.transpose(1, 2) @ y)
        resid = y - P @ w
        panel_r2[s:s + c] = (1 - resid.var(1).squeeze(1)
                             / (y.var(1).squeeze(1) + 1e-12)).clamp(0, 1)
    log(f"  panel pass ({time.time()-t0:.0f}s)")

    df = pd.DataFrame({
        "gene": genes[idx],
        "best_pair_r2": best_r2.cpu().numpy(),
        "panel_r2": panel_r2.cpu().numpy(),
        "has_equiv_partner": has_equiv.cpu().numpy().astype(bool),
    })
    df.to_csv(OUT / "many_to_one_genes.csv", index=False)

    rows = []
    for axis in ("best_pair_r2", "panel_r2"):
        for lo, hi in zip(R_EDGES[:-1], R_EDGES[1:]):
            m = (df[axis] >= lo) & (df[axis] < hi)
            if m.sum() == 0:
                continue
            rows.append({"axis": axis, "r_lo": lo, "genes": int(m.sum()),
                         "p_has_equiv": float(df.has_equiv_partner[m].mean())})
    curve = pd.DataFrame(rows)
    curve.to_csv(OUT / "curve_many_to_one.csv", index=False)

    base = float(df.has_equiv_partner.mean())
    log(f"  base P(gene has an equivalent partner) = {base:.4f}")
    for axis in ("best_pair_r2", "panel_r2"):
        hi = df[df[axis] >= 0.6]
        auc = _auc(df[axis].to_numpy(), df.has_equiv_partner.to_numpy())
        log(f"  [{axis}] AUC={auc:.3f}  "
            f"P(equiv | axis>=0.6)={float(hi.has_equiv_partner.mean()):.4f} "
            f"(n={len(hi)})")
    return curve


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(score)
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    pos, neg = label.sum(), (~label).sum()
    if pos == 0 or neg == 0:
        return np.nan
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


# ---------------------------------------------------------------------------
# Arm C: real paralogs
# ---------------------------------------------------------------------------

def load_hgnc_families(genes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (in_family_real, in_family_proxy) boolean arrays over genes."""
    if not HGNC_LOCAL.exists():
        HGNC_LOCAL.parent.mkdir(parents=True, exist_ok=True)
        log("  downloading HGNC complete set")
        req = urllib.request.Request(HGNC_URL,
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as r, \
                open(HGNC_LOCAL, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    hg = pd.read_csv(HGNC_LOCAL, sep="\t", low_memory=False,
                     usecols=lambda c: c in ("symbol", "gene_group"))
    hg = hg.dropna(subset=["gene_group"])
    sym2grp = dict(zip(hg.symbol, hg.gene_group))
    grp_counts = pd.Series([g for s, g in sym2grp.items()
                            if s in set(genes)]).value_counts()
    real = np.array([sym2grp.get(s) is not None
                     and grp_counts.get(sym2grp.get(s), 0) > 1 for s in genes])
    roots = np.array([symbol_root(s) for s in genes])
    fam = pd.Series(roots).value_counts()
    proxy = pd.Series(roots).map(fam).to_numpy() > 1
    return real, proxy


def arm_paralogs(ez, k_raw, keep, genes) -> pd.DataFrame:
    log("=" * 70)
    log("ARM C: real gene families (HGNC) vs the published symbol-root proxy")
    real, proxy = load_hgnc_families(genes)
    agree = (real == proxy).mean()
    log(f"  genes in a real family: {int(real.sum())}; proxy: "
        f"{int(proxy.sum())}; agreement {agree:.3f}; "
        f"proxy false-positives {int((proxy & ~real).sum())}, "
        f"missed {int((real & ~proxy).sum())}")
    rows = []
    for name, mask in (("real_nofamily", keep & ~real),
                       ("real_familyonly", keep & real),
                       ("proxy_nofamily", keep & ~proxy)):
        curve = curve_from_hist(prox_hist(ez, k_raw, mask), tau=0.8)
        curve["arm"] = name
        curve.to_csv(OUT / f"curve_{name}.csv", index=False)
        tot = curve.pairs.sum()
        base = (curve.pairs * curve.p_equiv).sum() / tot
        top = curve[curve.r_lo >= 0.60]
        ceil = ((top.pairs * top.p_equiv).sum() / top.pairs.sum()
                if top.pairs.sum() else np.nan)
        log(f"  [{name}] genes={int(mask.sum())} base={base:.5f} "
            f"ceiling(r>=0.6)={ceil:.4f} pairs>=0.6={int(top.pairs.sum())}")
        rows.append({"arm": name, "n_genes": int(mask.sum()),
                     "base": base, "ceiling": ceil})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["a", "b", "c", "all"], default="all")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"device: {DEV}")

    expr, chron, meta = load_join()
    genes = np.array(expr.columns)
    e_np = expr.to_numpy(np.float64)
    k_np = chron.to_numpy(np.float64)
    k_np = np.where(np.isnan(k_np), np.nanmean(k_np, axis=0, keepdims=True),
                    k_np)
    keep = filters(e_np, k_np)
    dummies = lineage_dummies(meta)
    ez_c = zscore(residualise(e_np, dummies))
    log(f"universe: {int(keep.sum())} genes, {len(meta)} lines")

    out = {}
    if args.arm in ("a", "all"):
        out["a"] = arm_tnbc(expr, chron, meta, genes)
    if args.arm in ("b", "all"):
        out["b"] = arm_many_to_one(ez_c, k_np, keep, genes)
    if args.arm in ("c", "all"):
        out["c"] = arm_paralogs(ez_c, k_np, keep, genes)
    log("=" * 70)
    log("arms complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
