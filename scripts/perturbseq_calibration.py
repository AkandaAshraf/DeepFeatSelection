"""Does the redundancy calibration survive a richer phenotype?

Pre-registration: paper/perturbseq_protocol.md, committed before the data was
read.

DepMap defines "same effect" as "same effect on proliferation" - one number
per gene per cell line. This replaces it with the whole transcriptomic
response to each CRISPRi knockdown (Replogle et al. 2022, genome-scale
Perturb-seq in K562), holding the observational axis fixed at the DepMap
definition so that only the phenotype changes.

    python scripts/perturbseq_calibration.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from depmap_calibration import (DEV, R_EDGES, load_join, log,  # noqa: E402
                                residualise, zscore)

PS = Path("Data/perturbseq/K562_gwps_normalized_bulk.h5ad")
OUT = Path("ExpOutput/perturbseq")
CHUNK = 1024
TAU_SWEEP = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
DEPMAP_BASE = 0.0042        # the base rate the threshold is calibrated to


def load_perturbseq():
    """Pseudo-bulk response profiles: perturbations x measured genes."""
    import anndata
    a = anndata.read_h5ad(PS)
    log(f"perturb-seq: {a.shape[0]} perturbations x {a.shape[1]} genes")
    # obs index is "<n>_<SYMBOL>_<protospacer>_<ENSG>"; the target symbol is
    # the second underscore-delimited field
    targets = np.array([str(i).split("_")[1] for i in a.obs.index])
    # var_names are Ensembl ids; symbols live in var["gene_name"]
    genes = np.array([str(g) for g in a.var["gene_name"]])
    X = np.asarray(a.X, dtype=np.float32)
    log(f"  unique targets {len(set(targets))}, "
        f"measured symbols {len(set(genes))}")
    return X, targets, genes


def prox_hist_cosine(ez, R, keep_e, keep_p, tau, gene_perm=None):
    """r_obs bin x equivalence, with on-target transcripts excluded.

    ez  (lines, G)     z-scored lineage-corrected DepMap expression
    R   (G, M)         perturbation response of each shortlisted gene
    """
    idx = np.where(keep_e & keep_p)[0]
    E = torch.as_tensor(ez[:, idx], device=DEV)
    Rt = torch.as_tensor(R[idx], dtype=torch.float32, device=DEV)
    n, G = E.shape
    p_idx = torch.as_tensor(
        np.arange(len(idx)) if gene_perm is None else gene_perm,
        device=DEV, dtype=torch.long)
    Rp = Rt[p_idx]
    # column position of each gene's OWN transcript, for on-target removal
    own = torch.as_tensor(OWN_COL[idx], device=DEV, dtype=torch.long)
    re_edges = torch.as_tensor(R_EDGES, device=DEV)
    nb = len(R_EDGES) - 1
    tot = torch.zeros(nb, device=DEV, dtype=torch.float64)
    hi = torch.zeros(nb, device=DEV, dtype=torch.float64)

    for s in range(0, G, CHUNK):
        c = min(CHUNK, G - s)
        eb = E[:, s:s + c]
        r = (eb.T @ E) / (n - 1)
        robs = (r ** 2).clamp(0, 0.999999)

        A = Rp[s:s + c].clone()                       # (c, M)
        # DECLARED CONTAMINATION CONTROL: zero each pair's own transcripts in
        # BOTH vectors before similarity. Row i's own column is zeroed for the
        # whole row; the partner's own column is zeroed per pair below.
        rows = torch.arange(c, device=DEV)
        A[rows, own[s:s + c]] = 0.0
        B = Rp.clone()
        # zero every shortlisted gene's own transcript in B as well: this
        # removes both members' on-target signal from every comparison
        B[torch.arange(G, device=DEV), own] = 0.0

        an = A / (A.norm(dim=1, keepdim=True) + 1e-8)
        bn = B / (B.norm(dim=1, keepdim=True) + 1e-8)
        cos = an @ bn.T                                # (c, G)
        cos[rows, torch.arange(s, s + c, device=DEV)] = -1.0   # self-pairs

        a_bin = (torch.bucketize(robs, re_edges) - 1).clamp(0, nb - 1)
        flat = a_bin.reshape(-1)
        tot.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float64))
        hi.scatter_add_(0, flat, (cos > tau).reshape(-1).double())
    return (tot / 2).cpu().numpy(), (hi / 2).cpu().numpy()


def curve(tot, hi):
    return pd.DataFrame({
        "r_lo": R_EDGES[:-1], "pairs": tot,
        "p_equiv": np.where(tot > 0, hi / np.maximum(tot, 1), np.nan)})


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"device: {DEV}")
    X, targets, ps_genes = load_perturbseq()

    expr, chron, meta = load_join()
    dm_genes = np.array(expr.columns)

    # one response profile per perturbed gene present in both datasets
    order = {g: i for i, g in enumerate(targets)}
    shared = np.array([g for g in dm_genes if g in order])
    log(f"genes in both DepMap and Perturb-seq: {len(shared)}")
    R = X[[order[g] for g in shared]]
    keep_rows = ~(np.all(R == 0, axis=1) | (R.std(axis=1) < 1e-9))
    log(f"  usable perturbation profiles: {int(keep_rows.sum())} "
        f"(excluded {int((~keep_rows).sum())})")

    # DepMap observational axis, exactly as published
    e_np = expr[shared].to_numpy(np.float64)
    lin = meta.OncotreeLineage.fillna("other")
    lin = lin.where(lin.map(lin.value_counts()) >= 8, "other")
    D = pd.get_dummies(lin).to_numpy(np.float32)
    D = np.hstack([np.ones((len(lin), 1), np.float32), D])
    ez = zscore(residualise(e_np, D))
    keep_e = (e_np.mean(0) >= 1.0)
    floor = np.percentile(e_np[:, keep_e].std(0), 20)
    keep_e = keep_e & (e_np.std(0) >= floor)
    log(f"  expression filter keeps {int(keep_e.sum())}")

    # on-target column for each shortlisted gene
    pos = {g: i for i, g in enumerate(ps_genes)}
    global OWN_COL
    OWN_COL = np.array([pos.get(g, 0) for g in shared])
    has_own = np.array([g in pos for g in shared])
    log(f"  own transcript measured for {int(has_own.sum())}/{len(shared)}")

    keep = keep_e & keep_rows
    log(f"  final universe: {int(keep.sum())} genes "
        f"-> {int(keep.sum())*(int(keep.sum())-1)//2:,} pairs\n")

    t0 = time.time()
    rows = []
    for tau in TAU_SWEEP:
        tot, hi = prox_hist_cosine(ez, R, keep_e, keep_rows, tau)
        c = curve(tot, hi)
        c["tau"] = tau
        c.to_csv(OUT / f"curve_tau{tau}.csv", index=False)
        base = float(hi.sum() / max(tot.sum(), 1))
        top = c[(c.r_lo >= 0.60 - 1e-9) & (c.r_lo < 0.70 - 1e-9)]
        ceil = (float((top.pairs * top.p_equiv).sum() / top.pairs.sum())
                if top.pairs.sum() else float("nan"))
        rows.append({"tau": tau, "base": base, "ceiling": ceil,
                     "lift": ceil / base if base > 0 else np.nan,
                     "pairs_high": float(top.pairs.sum())})
        log(f"  tau={tau}: base {base*100:.3f}%  ceiling {ceil*100:.1f}%  "
            f"lift {ceil/base if base>0 else float('nan'):.1f}x")
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "tau_sweep.csv", index=False)
    log(f"\nsweep done ({time.time()-t0:.0f}s)")

    # declared calibration: quote at the tau whose base rate matches DepMap
    s["gap"] = (s.base - DEPMAP_BASE).abs()
    pick = s.loc[s.gap.idxmin()]
    log(f"\nDECLARED CALIBRATION: tau={pick.tau} puts the base rate at "
        f"{pick.base*100:.3f}% vs DepMap's {DEPMAP_BASE*100:.2f}%")
    log(f"  PERTURB-SEQ ceiling {pick.ceiling*100:.1f}%   "
        f"lift {pick.lift:.1f}x   (DepMap: 17.0%, 39.3x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
