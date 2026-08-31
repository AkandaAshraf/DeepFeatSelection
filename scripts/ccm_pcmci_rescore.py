"""Corrected scoring of the CCM/PCMCI baseline, from the saved matrices.

Two defects in scripts/ccm_pcmci_baseline.py's scoring, found when PCMCI
returned a membership AUROC of 0.008 -- near-perfect INVERSION, which is the
signature of a transposed score matrix rather than of a method failing:

  1. TRANSPOSE BUG. tigramite's val_matrix[i, j, tau] is the dependence of j
     at lag 0 on i at lag -tau, i.e. it is ALREADY evidence for i -> j.
     scripts/chamber_detect.py returns val.max(axis=2) with no transpose and
     is correct; ccm_pcmci_baseline.py added a .T and inverted every PCMCI
     edge. CCM is unaffected (its direction convention was checked against
     deepfeatselect.ccm's documented semantics and is right).

  2. DILUTED H3. The declared H3 compared ALL source->driven pairs (5 x 25 =
     125) against non-edges, but only 25 of those 125 are true edges -- each
     driven channel has exactly one parent. The positive set was 80%
     non-edges, so H3 was pinned near 0.5 by construction and could not have
     detected anything. The protocol flagged this as "a weaker, conservative
     version of H3"; it is weaker to the point of being uninformative, so it
     is replaced here with the true-edge version.

The true parent structure is recovered by replicating make_system's rng
sequence exactly (default_rng(seed), then uniform(V), uniform(V), then
integers(0, n_src, n_drv)), verified against the generated system.

No re-run of CCM (18.8 min) is needed: both matrices were saved.

    python scripts/ccm_pcmci_rescore.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from error_metrics import auc  # noqa: E402

IN = Path("ExpOutput/ccm_pcmci_baseline/matrices.npz")
OUT = Path("ExpOutput/ccm_pcmci_baseline")
N, V, COUPLING, REDUNDANCY, SEED = 4000, 30, 0.20, 0, 0


def true_parents(V: int, seed: int) -> tuple[np.ndarray, int]:
    """Recover make_system's parent[] by replaying its rng draws in order."""
    rng = np.random.default_rng(seed)
    n_src = max(3, V // 6)
    n_drv = V - n_src
    rng.uniform(0.2, 0.8, V)      # x[0]
    rng.uniform(3.6, 3.9, V)      # r
    return rng.integers(0, n_src, n_drv), n_src


def membership(mat: np.ndarray) -> np.ndarray:
    m = mat.copy()
    np.fill_diagonal(m, np.nan)
    return np.nanmax(m, axis=0)


def main() -> int:
    z = np.load(IN)
    ccm_mat = z["ccm"]
    pcmci_mat = z["pcmci"].T        # undo the transpose bug
    is_driven, is_source = z["is_driven"], z["is_source"]
    parent, n_src = true_parents(V, SEED)
    src_idx = np.where(is_source)[0]
    drv_idx = np.where(is_driven)[0]
    assert len(src_idx) == n_src and len(drv_idx) == len(parent)

    print(f"V={V} n={N} seed={SEED}: {n_src} sources, {len(drv_idx)} driven")
    print(f"true edges: {len(parent)} (one parent per driven channel)\n")

    # true edge mask: source src_idx[parent[k]] -> driven drv_idx[k]
    true_edge = np.zeros_like(ccm_mat, dtype=bool)
    for k, p in enumerate(parent):
        true_edge[src_idx[p], drv_idx[k]] = True

    rows = []
    for name, mat in [("CCM", ccm_mat), ("PCMCI", pcmci_mat)]:
        score = membership(mat)
        m_auc = auc(score[is_driven], score[is_source])

        off = ~np.eye(len(mat), dtype=bool)
        pos = mat[true_edge & off]
        neg = mat[(~true_edge) & off]
        pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
        e_auc = auc(pos, neg)

        rows.append({"method": name, "membership_auroc": m_auc,
                     "true_edge_auroc": e_auc,
                     "n_true_edges": int(true_edge.sum())})
        print(f"{name:6s} membership AUROC (driven vs source): {m_auc:.3f}")
        print(f"{name:6s} H3 true-edge AUROC (real edges vs non-edges): "
              f"{e_auc:.3f}  {'functions' if e_auc >= 0.7 else 'FAILS to function here'}\n")

    mace_npz = Path(f"ExpOutput/boundary_map/raw_n{N}_V{V}_c{COUPLING}"
                    f"_r{REDUNDANCY}_s{SEED}.npz")
    m = np.load(mace_npz)
    mace_auc = auc(m["excess"][m["is_driven"]], m["excess"][m["is_source"]])
    print(f"MACE   membership AUROC (same system, same seed): {mace_auc:.3f}")
    print("       (MACE scores membership directly; it produces no edge "
          "matrix, so H3 does not apply to it.)")
    rows.append({"method": "MACE", "membership_auroc": mace_auc,
                 "true_edge_auroc": np.nan, "n_true_edges": np.nan})

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "results_corrected.csv", index=False)

    print("\nVERDICT (H1 fixed before running)")
    base = d[d.method != "MACE"].membership_auroc
    print(f"   H1: MACE ({mace_auc:.3f}) exceeds BOTH baselines: "
          f"{'YES' if all(mace_auc > b for b in base) else 'NO'}")
    print(f"   CCM ties MACE at {d.iloc[0].membership_auroc:.3f} on this "
          f"cell; the honest reading is that at V={V}, where pairwise CCM is")
    print("   still affordable, CCM is not worse at the membership task. "
          "MACE's claim is\n   about scale, not about accuracy at widths a "
          "pairwise method can reach.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
