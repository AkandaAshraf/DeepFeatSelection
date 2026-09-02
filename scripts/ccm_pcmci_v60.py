"""CCM and PCMCI against MACE at V=60, where MACE's own recall has collapsed.

Pre-registration: paper/ccm_pcmci_v60_protocol.md, committed before any
score at this width. Companion to the V=30 run, whose result section said
the informative comparison is here and that it had not been run.

At V=60, coupling 0.20, the boundary map gives MACE recall 0.18/0.18/0.14 at
precision 1.00 across three seeds. If a pairwise method holds up at a width
where MACE misses four driven channels in five, the paper's positioning
needs qualifying.

Both corrections from the V=30 run are carried in from the start:
  - PCMCI's val_matrix is used WITHOUT a transpose (val_matrix[i,j,tau] is
    already evidence i -> j; the V=30 script's .T inverted every edge).
  - H3 is the true-edge version, recovering parent[] by replaying
    make_system's rng draws, not the diluted all-source-to-driven version.

Expect four to five hours: ~77 min of CCM per seed (1,770 pairs at the
2.59 s/pair realised at V=30), plus PCMCI, whose V=60 cost is unmeasured.

    python scripts/ccm_pcmci_v60.py
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

# LEAN IMPORTS. `boundary_map` pulls torch (+351 MB) and the `deepfeatselect`
# package __init__ pulls keras/tensorflow (+213 MB); the computation here
# needs neither. The first V=60 attempt died on a 1 MB allocation, so that
# 564 MB is worth not paying. Systems come pre-generated from
# scripts/v60_gen_systems.py, and ccm.py is loaded directly by path so the
# package __init__ never executes.
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "_ccm_direct", str(Path(__file__).parent.parent / "deepfeatselect"
                       / "ccm.py"))
_ccm_mod = _ilu.module_from_spec(_spec)
sys.modules["_ccm_direct"] = _ccm_mod       # dataclasses need this registered
_spec.loader.exec_module(_ccm_mod)
ccm = _ccm_mod.ccm


def auc(pos, neg):
    """P(a random positive outranks a random negative), ties at 0.5.

    Inlined verbatim from scripts/error_metrics.py rather than imported:
    that module imports torch and source_outflow_gate, costing ~330 MB for
    six lines of numpy. scripts/test_auc_identical.py asserts this stays
    byte-identical in behaviour to the original.
    """
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)

class _PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


_GPMI = ctypes.windll.psapi.GetProcessMemoryInfo
_GPMI.argtypes = [wt.HANDLE, ctypes.POINTER(_PMC), wt.DWORD]
_GPMI.restype = wt.BOOL
_CURPROC = ctypes.windll.kernel32.GetCurrentProcess
_CURPROC.restype = wt.HANDLE


def mem_mb() -> tuple[float, float]:
    """(current, peak) working set in MB for this process.

    argtypes must be declared: without them the 64-bit process handle is
    truncated to int and the call silently returns zeros.
    """
    try:
        c = _PMC()
        c.cb = ctypes.sizeof(_PMC)
        if not _GPMI(_CURPROC(), ctypes.byref(c), c.cb):
            return float("nan"), float("nan")
        return c.WorkingSetSize / 2**20, c.PeakWorkingSetSize / 2**20
    except Exception:
        return float("nan"), float("nan")


OUT = Path("ExpOutput/ccm_pcmci_v60")
N, V, COUPLING, REDUNDANCY = 4000, 60, 0.20, 0
SEEDS = (0, 1, 2)
TAU_MAX = 3


def true_parents(V: int, seed: int) -> tuple[np.ndarray, int]:
    """Replay make_system's rng draws in order to recover parent[]."""
    rng = np.random.default_rng(seed)
    n_src = max(3, V // 6)
    n_drv = V - n_src
    rng.uniform(0.2, 0.8, V)      # x[0]
    rng.uniform(3.6, 3.9, V)      # r
    return rng.integers(0, n_src, n_drv), n_src


def ccm_matrix(x: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    """score[p, q] = evidence p -> q. One ccm() call per unordered pair."""
    t0 = time.time()
    n = x.shape[1]
    out = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            r = ccm(x[:, i], x[:, j], E=3, seed=seed)
            out[i, j] = r.x_causes_y.rho_at_max_lib
            out[j, i] = r.y_causes_x.rho_at_max_lib
        if i % 10 == 0:
            print(f"    ccm row {i}/{n}  ({time.time()-t0:.0f}s)", flush=True)
    return out, time.time() - t0


def pcmci_matrix(x: np.ndarray) -> tuple[np.ndarray, float]:
    """score[p, q] = evidence p -> q. NO transpose: val_matrix[i,j,tau] is
    the dependence of j at lag 0 on i at lag -tau, i.e. already i -> j."""
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    import tigramite.data_processing as pp

    t0 = time.time()
    # float32 halves the conditioning arrays tigramite builds internally --
    # the OOM at V=60 was inside one of them. ParCorr is a partial
    # correlation at n=4000; float32 precision is ample for it.
    z = ((x - x.mean(0)) / (x.std(0) + 1e-12)).astype(np.float32)
    df = pp.DataFrame(z, var_names=[str(i) for i in range(x.shape[1])])
    pcmci = PCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0)
    res = pcmci.run_pcmci(tau_max=TAU_MAX, pc_alpha=None)
    val = np.abs(res["val_matrix"])[:, :, 1:]      # drop lag 0
    return val.max(axis=2), time.time() - t0


def membership(mat: np.ndarray) -> np.ndarray:
    m = mat.copy()
    np.fill_diagonal(m, np.nan)
    return np.nanmax(m, axis=0)


def score_cell(mat, is_driven, is_source, true_edge):
    s = membership(mat)
    m_auc = auc(s[is_driven], s[is_source])
    off = ~np.eye(len(mat), dtype=bool)
    pos = mat[true_edge & off]
    neg = mat[(~true_edge) & off]
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    return m_auc, auc(pos, neg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None,
                    help="run ONE seed in this process (memory isolation); "
                         "omit to run all seeds in one process")
    args = ap.parse_args()
    seeds = (args.seed,) if args.seed is not None else SEEDS

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"V={V} n={N} coupling={COUPLING} seeds={seeds}")
    print("MACE reference recall at this cell: 0.18 / 0.18 / 0.14\n")
    rows, t_start = [], time.time()

    for seed in seeds:
        print(f"=== seed {seed} ===", flush=True)
        sysf = OUT / f"system_s{seed}.npz"
        if not sysf.exists():
            print(f"  missing {sysf} - run scripts/v60_gen_systems.py first")
            return 1
        _d = np.load(sysf)
        x = _d["x"]
        is_driven, is_source = _d["is_driven"], _d["is_source"]
        true_edge = _d["true_edge"]
        print(f"  system loaded, sha={_d['sha']}  "
              f"(mem {mem_mb()[0]:.0f} MB)", flush=True)

        # CCM is the memory-light, expensive part and carries H1 (the
        # decisive MACE-vs-CCM prediction). Save it the instant it finishes,
        # so a later PCMCI failure never discards 90 minutes of it.
        cm, ct = ccm_matrix(x, seed)
        print(f"  CCM   {ct/60:.1f} min", flush=True)
        np.savez_compressed(OUT / f"ccm_s{seed}.npz", ccm=cm,
                            is_driven=is_driven, is_source=is_source,
                            true_edge=true_edge)
        methods = [("CCM", cm, ct)]
        print(f"  mem after CCM: {mem_mb()[0]:.0f} MB "
              f"(peak {mem_mb()[1]:.0f} MB)", flush=True)
        gc.collect()   # return CCM's per-pair KD-trees before PCMCI's peak

        # PCMCI is secondary (H2 has no prediction) and is the part that
        # OOM'd at V=60 on this loaded machine. Best-effort: a MemoryError
        # records it as unavailable for this seed and the run continues, so
        # the decisive comparison still completes.
        try:
            pm, pt = pcmci_matrix(x)
            print(f"  PCMCI {pt/60:.1f} min", flush=True)
            np.savez_compressed(OUT / f"pcmci_s{seed}.npz", pcmci=pm)
            methods.append(("PCMCI", pm, pt))
        except MemoryError:
            print("  PCMCI OOM - recorded as unavailable, continuing",
                  flush=True)
            rows.append({"seed": seed, "method": "PCMCI",
                         "membership_auroc": np.nan,
                         "true_edge_auroc": np.nan, "minutes": np.nan})

        for name, mat, t in methods:
            m_auc, e_auc = score_cell(mat, is_driven, is_source, true_edge)
            rows.append({"seed": seed, "method": name,
                         "membership_auroc": m_auc, "true_edge_auroc": e_auc,
                         "minutes": t / 60})
            print(f"  {name:6s} membership {m_auc:.3f}   "
                  f"true-edge {e_auc:.3f}", flush=True)
        tag = f"_s{seed}" if args.seed is not None else ""
        pd.DataFrame(rows).to_csv(OUT / f"results{tag}.csv", index=False)
        print(f"  mem peak this seed: {mem_mb()[1]:.0f} MB", flush=True)

        m = np.load(f"ExpOutput/boundary_map/raw_n{N}_V{V}_c{COUPLING}"
                    f"_r{REDUNDANCY}_s{seed}.npz")
        mace_auc = auc(m["excess"][m["is_driven"]],
                       m["excess"][m["is_source"]])
        rows.append({"seed": seed, "method": "MACE",
                     "membership_auroc": mace_auc,
                     "true_edge_auroc": np.nan, "minutes": 0.0})
        print(f"  MACE   membership {mace_auc:.3f}\n", flush=True)

    d = pd.DataFrame(rows)
    print(f"({(time.time()-t_start)/60:.1f} min total, "
          f"peak {mem_mb()[1]:.0f} MB)\n")
    if args.seed is not None:
        print("single-seed run complete; aggregate with the driver once all "
              "seeds are done.")
        return 0
    d.to_csv(OUT / "results.csv", index=False)

    g = d.groupby("method").membership_auroc.agg(["median", "min", "max"])
    print("MEMBERSHIP AUROC over three seeds")
    print("   " + g.round(3).to_string().replace("\n", "\n   "))
    e = d.groupby("method").true_edge_auroc.median()
    print("\nTRUE-EDGE AUROC (median)")
    for k in ("CCM", "PCMCI"):
        print(f"   {k:6s} {e.loc[k]:.3f}  "
              f"{'functions' if e.loc[k] >= 0.7 else 'FAILS to function'}")

    mace, ccm_m = g.loc["MACE", "median"], g.loc["CCM", "median"]
    print("\nVERDICT (rule fixed before running)")
    both_ceiling = mace > 0.99 and ccm_m > 0.99
    h3_ok = all(e.loc[k] >= 0.7 for k in ("CCM", "PCMCI"))
    if both_ceiling or not h3_ok:
        print("   -> NOT INFORMATIVE: "
              + ("both methods at ceiling again" if both_ceiling
                 else "a baseline failed H3"))
    elif mace > ccm_m:
        print(f"   -> H1 HOLDS. MACE {mace:.3f} > CCM {ccm_m:.3f} at the "
              "width where MACE's\n      own recall has collapsed. The scale "
              "claim is supported where it matters.")
    else:
        print(f"   -> BASELINE AHEAD. CCM {ccm_m:.3f} >= MACE {mace:.3f}. "
              "Table 12's membership\n      row must be amended: CCM is "
              "competitive to at least V=60, and MACE's\n      advantage is "
              "affordability past that width, not accuracy at it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
