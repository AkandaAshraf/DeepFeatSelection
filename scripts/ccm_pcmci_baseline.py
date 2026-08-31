"""CCM and PCMCI against MACE, on MACE's own boundary-map ground truth.

Pre-registration: paper/ccm_pcmci_baseline_protocol.md, committed before any
score was computed. Prompted by an adversarial review finding that no
baseline in the paper is compared to MACE directly on the same task.

System: boundary_map.make_system(n=4000, V=30, coupling=0.20, redundancy=0,
seed=0) -- byte-identical to the system behind the published V=30 row of
Table 4 (precision 1.00, recall 0.88).

Each pairwise method produces a directed score matrix; a channel's
membership score is the max incoming edge score over all other channels,
matching the aggregation already used in scripts/chamber_detect.py. Ranked
by AUROC (threshold-free -- neither baseline has a ghost panel to calibrate
a cutoff from).

    python scripts/ccm_pcmci_baseline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from boundary_map import make_system  # noqa: E402
from deepfeatselect.ccm import ccm  # noqa: E402

OUT = Path("ExpOutput/ccm_pcmci_baseline")
N, V, COUPLING, REDUNDANCY, SEED = 4000, 30, 0.20, 0, 0
TAU_MAX = 3


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def ccm_matrix(x: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    """score[p, q] = evidence p -> q. One ccm() call per UNORDERED pair,
    reading both directions off it -- matching chamber_detect.py's cost
    model of C(V,2) calls rather than V*(V-1)."""
    t0 = time.time()
    n = x.shape[1]
    out = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            r = ccm(x[:, i], x[:, j], E=3, seed=seed)
            out[i, j] = r.x_causes_y.rho_at_max_lib  # i -> j
            out[j, i] = r.y_causes_x.rho_at_max_lib  # j -> i
    return out, time.time() - t0


def pcmci_matrix(x: np.ndarray) -> tuple[np.ndarray, float]:
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    import tigramite.data_processing as pp

    t0 = time.time()
    z = (x - x.mean(0)) / (x.std(0) + 1e-12)
    names = [str(i) for i in range(x.shape[1])]
    dataframe = pp.DataFrame(z, var_names=names)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    result = pcmci.run_pcmci(tau_max=TAU_MAX, pc_alpha=None)
    val = np.abs(result["val_matrix"])[:, :, 1:]   # drop lag 0
    return val.max(axis=2).T, time.time() - t0     # transpose: [p, q] = p->q


def membership_scores(mat: np.ndarray, is_source: np.ndarray) -> np.ndarray:
    """score(q) = max incoming edge from any OTHER channel."""
    m = mat.copy()
    np.fill_diagonal(m, np.nan)
    return np.nanmax(m, axis=0)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"CCM/PCMCI baseline: V={V} n={N} coupling={COUPLING} seed={SEED}\n")

    x, is_driven, is_source = make_system(
        n=N, V=V, coupling=COUPLING, redundancy=REDUNDANCY, seed=SEED)
    print(f"system: {is_source.sum()} sources, {is_driven.sum()} driven, "
          f"{x.shape[0]} samples\n")

    print("running CCM ...", flush=True)
    ccm_mat, ccm_t = ccm_matrix(x, seed=SEED)
    print(f"  done in {ccm_t/60:.1f} min\n", flush=True)

    print("running PCMCI ...", flush=True)
    pcmci_mat, pcmci_t = pcmci_matrix(x)
    print(f"  done in {pcmci_t/60:.1f} min\n", flush=True)

    np.savez(OUT / "matrices.npz", ccm=ccm_mat, pcmci=pcmci_mat,
             is_driven=is_driven, is_source=is_source)

    rows = []
    for name, mat in [("CCM", ccm_mat), ("PCMCI", pcmci_mat)]:
        score = membership_scores(mat, is_source)
        auc = auroc(score[is_driven], score[is_source])
        rows.append({"method": name, "auroc": auc, "runtime_min":
                     (ccm_t if name == "CCM" else pcmci_t) / 60})
        print(f"{name:8s} membership AUROC (driven vs source): {auc:.3f}")

    # H3: restrict to DIRECT parent->child edges vs all non-edges
    n_src = int(is_source.sum())
    # boundary_map.make_system: driven channel k (0-indexed among driven)
    # has parent = rng.integers(0, n_src, n_drv) -- but the seed sequence
    # used to build parent[] is internal to make_system and not returned,
    # so H3 is evaluated on ALL source->driven pairs restricted to sources
    # (a superset of true edges, since every driven channel has exactly one
    # parent among the sources) -- this is declared here rather than
    # silently narrowed, and is a weaker, conservative version of H3.
    src_idx = np.where(is_source)[0]
    drv_idx = np.where(is_driven)[0]
    for name, mat in [("CCM", ccm_mat), ("PCMCI", pcmci_mat)]:
        sub = mat[np.ix_(src_idx, drv_idx)]  # source -> driven edges only
        pos = sub[~np.isnan(sub)]
        # negatives: driven -> driven and driven -> source (never true edges)
        other = mat[np.ix_(drv_idx, np.concatenate([src_idx, drv_idx]))]
        other = other[~np.isnan(other)]
        h3 = auroc(pos, other)
        print(f"{name:8s} H3 (source->driven edges vs all else): {h3:.3f}"
              f"  {'PASS' if h3 >= 0.7 else 'FAILS to function here'}")
        rows[-1]["h3_source_to_driven_auroc"] = h3

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "results.csv", index=False)

    # MACE's own number on the identical system, re-derived from the saved
    # per-channel excess (not the pooled boundary_map.csv precision/recall,
    # which needs a threshold; AUROC is threshold-free like the baselines)
    mace_npz = Path(
        f"ExpOutput/boundary_map/raw_n{N}_V{V}_c{COUPLING}_r{REDUNDANCY}"
        f"_s{SEED}.npz")
    if mace_npz.exists():
        m = np.load(mace_npz)
        mace_auc = auroc(m["excess"][m["is_driven"]],
                         m["excess"][m["is_source"]])
        print(f"\nMACE     membership AUROC (same system, same seed): "
              f"{mace_auc:.3f}")
        pd.DataFrame([{"method": "MACE", "auroc": mace_auc}]).to_csv(
            OUT / "mace_reference.csv", index=False)
    else:
        print(f"\nMACE reference file not found at {mace_npz} -- "
              f"boundary map must be re-run to produce it for this cell.")
        mace_auc = None

    print("\nVERDICT (rule fixed before running)")
    if mace_auc is not None:
        beats = all(mace_auc > r for r in d.auroc)
        print(f"   H1 MACE ({mace_auc:.3f}) exceeds both baselines: "
              f"{'YES' if beats else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
