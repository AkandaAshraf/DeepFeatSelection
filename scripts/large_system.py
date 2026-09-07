"""Large systems on one 8 GB GPU: staged widths, hard caps, no shrinking.

Pre-registration: paper/large_system_protocol.md, committed before this was
written. EXPLORATORY. No verdict word is licensed or used.

Two questions kept apart: STATISTICAL (does ranking still separate sources
from driven channels; does the deployed rule still surface anything) and
COMPUTE (wall-clock, GPU memory, host RAM per arm as V grows). A width that
cannot run here is HARDWARE-LIMITED, reported as such, never as tested, and
never rescued by a smaller model, fewer epochs, shorter data or a dropped arm.

Arms: FLAT, HIER-CLUST-TRAIN, HIER-RAND-SIZED, HIER-TRUE (C01 machinery,
imported not re-implemented), SELFR2 and LAGCORR (cheap baselines).

    python scripts/large_system.py
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
from boundary_map import (DEV, DONOR_R2, E, MIN_DONORS, N_GHOSTS,  # noqa: E402
                          embed, make_system, poly3, ridge_r2)
import hierarchy_repair as HR  # noqa: E402

OUT = Path("ExpOutput/large_system")
N, COUPLING = 4000, 0.20
WIDTHS = (120, 240, 500, 1000)
SEEDS = (100, 101, 102)
ARMS = ["FLAT", "HIER-CLUST-TRAIN", "HIER-RAND-SIZED", "HIER-TRUE",
       "SELFR2", "LAGCORR"]
HIER = ("HIER-CLUST-TRAIN", "HIER-RAND-SIZED", "HIER-TRUE")

CAP_GPU_MB = 7000.0
CAP_RSS_MB = 2500.0
CAP_FREE_MB = 1000.0
CAP_CELL_SEC = 60 * 60
CAP_TOTAL_SEC = 6 * 60 * 60
SHARE_DENOM_EPS = HR.SHARE_DENOM_EPS


# ------------------------------------------------------------- resources

class _PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


class _MEMSTAT(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64)]


def host_rss_mb() -> float:
    if os.name != "nt":
        return float("nan")
    k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    fn = psapi.GetProcessMemoryInfo
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC), ctypes.c_uint32]
    fn.restype = ctypes.c_int
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    if not fn(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return float("nan")
    return pmc.WorkingSetSize / 2 ** 20


def sys_free_mb() -> float:
    if os.name != "nt":
        return float("nan")
    s = _MEMSTAT()
    s.dwLength = ctypes.sizeof(_MEMSTAT)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPhys / 2 ** 20


def gpu_reset() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def gpu_peak_mb() -> float:
    if not torch.cuda.is_available():
        return float("nan")
    torch.cuda.synchronize()
    # reserved is the true footprint on the card and already contains
    # allocated; summing them would double-count and breach the cap at
    # roughly half the real usage.
    return torch.cuda.max_memory_reserved() / 2 ** 20


class CapBreach(RuntimeError):
    def __init__(self, cap: str, value: float, limit: float, arm: str):
        super().__init__(f"{cap}={value:.0f} > {limit:.0f} in {arm}")
        self.cap, self.value, self.limit, self.arm = cap, value, limit, arm


# --------------------------------------------------------------- pieces

def ghost_rule(zs, lead, own, base, sys_code, self_r2, tr_i, te_i, seed):
    """The DEPLOYED threshold, replicated from boundary_map unchanged: 30
    donor ghosts from channels with self-R2 > DONOR_R2 (fallback to all if
    fewer than MIN_DONORS), circular shift in [m/4, 3m/4], threshold at the
    panel maximum clamped at zero. Its miscalibration is already on record;
    it is used here because it is the rule that ships."""
    Vt = len(base)
    mrows = zs.shape[0]

    def excess_of(f, target):
        b0 = ridge_r2(f[tr_i], target[tr_i + 1], f[te_i], target[te_i + 1])
        return ridge_r2(np.hstack([f[tr_i], sys_code[tr_i]]), target[tr_i + 1],
                        np.hstack([f[te_i], sys_code[te_i]]),
                        target[te_i + 1]) - b0

    rng = np.random.default_rng(seed + 4242)
    qual = np.where(self_r2 > DONOR_R2)[0]
    fallback = len(qual) < MIN_DONORS
    pool = np.arange(Vt) if fallback else qual
    donors = rng.choice(pool, size=min(N_GHOSTS, len(pool)), replace=False)
    ghosts = []
    for d in donors:
        s = int(rng.integers(mrows // 4, 3 * mrows // 4))
        gz = np.roll(zs[:, d * E:(d + 1) * E], s, axis=0)
        ghosts.append(excess_of(poly3(gz), np.roll(lead[:, d], s)))
    ghosts = np.array(ghosts)
    return max(0.0, float(ghosts.max())), ghosts, bool(fallback)


def lagcorr_scores(x_raw: np.ndarray, raw_cutoff: int) -> np.ndarray:
    """Cheap linear baseline, TRAINING ROWS ONLY: for target q, the largest
    |corr(x_j(t), x_q(t+1))| over j != q, minus |corr(x_q(t), x_q(t+1))|.
    Higher = more driven."""
    xt = x_raw[:raw_cutoff]
    A = xt[:-1]                          # x(t)
    B = xt[1:]                           # x(t+1)
    A = (A - A.mean(0)) / (A.std(0) + 1e-12)
    B = (B - B.mean(0)) / (B.std(0) + 1e-12)
    C = np.abs(A.T @ B) / len(A)         # C[j, q] = |corr(x_j(t), x_q(t+1))|
    self_c = np.diag(C).copy()
    np.fill_diagonal(C, -np.inf)
    return C.max(0) - self_c


def rule_metrics(score_driven, thr, is_driven, is_source):
    flagged = score_driven > thr
    tp = int((flagged & is_driven).sum())
    return dict(
        recall_rule=tp / max(int(is_driven.sum()), 1),
        source_fp_rule=float((flagged & is_source).sum()
                             / max(int(is_source.sum()), 1)),
        n_flagged=int(flagged.sum()))


# -------------------------------------------------------------- bundles

def _quarantine(p: Path, why: str) -> None:
    q = OUT / "quarantine"
    q.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(p, q / f"{p.name}.{int(time.time())}")
        print(f"   QUARANTINED {p.name}: {why}")
    except OSError as exc:
        print(f"   could not quarantine {p.name}: {exc}")


def bpath(V, seed) -> Path:
    return OUT / f"cell_V{V}_s{seed}.json"


def save_bundle(V, seed, rows, status, failures, extra) -> None:
    payload = dict(V=V, seed=seed, rows=rows, status=status,
                   failures=failures, extra=extra)
    tmp = bpath(V, seed).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, default=float))
    os.replace(tmp, bpath(V, seed))


def load_bundle(V, seed):
    p = bpath(V, seed)
    if not p.exists():
        return None
    try:
        b = json.loads(p.read_text())
    except (OSError, ValueError):
        _quarantine(p, "unreadable")
        return None
    ok = (b.get("V") == V and b.get("seed") == seed
          and set(b.get("status", {})) == set(ARMS)
          and all(v in ("ok", "infeasible") for v in b["status"].values())
          and (OUT / f"raw_V{V}_s{seed}.npz").exists())
    if not ok:
        _quarantine(p, "incomplete or wrong cell")
        return None
    return b


# ---------------------------------------------------------------- cell

def cell(V: int, seed: int) -> tuple[list[dict], dict, list[dict], dict]:
    t_cell = time.time()
    x, is_driven, is_source = make_system(N, V, COUPLING, 0, seed)
    Vt = x.shape[1]
    n_src = max(3, V // 6)
    rng_g = np.random.default_rng(seed)
    _ = rng_g.uniform(0.2, 0.8, V)
    _ = rng_g.uniform(3.6, 3.9, V)
    parent = rng_g.integers(0, n_src, V - n_src)

    emb = embed(x)
    mrows = emb.shape[0]
    a, bnd = int(0.6 * mrows), int(0.8 * mrows)
    raw_cutoff = a
    tr = slice(0, a)
    tr_i, te_i = np.arange(0, a - 1), np.arange(bnd, mrows - 1)
    mu, sd = emb[tr].mean(0), emb[tr].std(0) + 1e-12            # train-only
    zs = np.clip(np.nan_to_num((emb - mu) / sd), -20, 20).astype(np.float32)
    lead = zs[:, [j * E for j in range(Vt)]]
    own = [poly3(zs[:, q * E:(q + 1) * E]) for q in range(Vt)]
    base = np.array([ridge_r2(own[q][tr_i], lead[tr_i + 1, q],
                              own[q][te_i], lead[te_i + 1, q])
                     for q in range(Vt)])
    prevalence = float(is_source.mean())

    rows, status, failures = [], {}, []
    raw = {"is_driven": is_driven, "is_source": is_source, "parent": parent,
           "raw_cutoff": raw_cutoff, "self_r2": base}
    extra = {"prevalence": prevalence, "n_train_rows": int(len(tr_i)),
             "sys_code_width": 2 * V,
             "capacity_ratio": float(len(tr_i) / (2 * V + own[0].shape[1]))}

    def record(arm, score_driven, secs, thr=None, **more):
        r = dict(V=V, seed=seed, arm=arm, prevalence=prevalence,
                 ap_source=float(average_precision_score(is_source,
                                                         -score_driven)),
                 secs=secs, gpu_peak_mb=gpu_peak_mb(), host_rss_mb=host_rss_mb(),
                 **more)
        if thr is not None:
            r.update(rule_metrics(score_driven, thr, is_driven, is_source))
        rows.append(r)
        raw[arm] = score_driven
        status[arm] = "ok"
        if r["gpu_peak_mb"] > CAP_GPU_MB:
            raise CapBreach("gpu_peak_mb", r["gpu_peak_mb"], CAP_GPU_MB, arm)
        if r["host_rss_mb"] > CAP_RSS_MB:
            raise CapBreach("host_rss_mb", r["host_rss_mb"], CAP_RSS_MB, arm)
        if time.time() - t_cell > CAP_CELL_SEC:
            raise CapBreach("cell_sec", time.time() - t_cell, CAP_CELL_SEC, arm)

    def infeasible(arm, exc):
        failures.append(dict(V=V, seed=seed, arm=arm, kind=type(exc).__name__,
                             detail=str(exc)[:300]))
        status[arm] = "infeasible"
        torch.cuda.empty_cache()

    # ---- baselines first: cheap, CPU, always feasible
    t0 = time.time()
    record("SELFR2", -base, time.time() - t0)      # low self-R2 = driven
    t0 = time.time()
    lc = lagcorr_scores(x, raw_cutoff)
    record("LAGCORR", lc, time.time() - t0)

    # ---- FLAT
    sys_code = None
    gpu_reset()
    t0 = time.time()
    try:
        _, sys_code = HR.train_code(zs, tr, zs.shape[1], 2 * V, seed * 100)
        flat = np.array([
            ridge_r2(np.hstack([own[q][tr_i], sys_code[tr_i]]),
                     lead[tr_i + 1, q],
                     np.hstack([own[q][te_i], sys_code[te_i]]),
                     lead[te_i + 1, q]) - base[q] for q in range(Vt)])
        thr, ghosts, fb = ghost_rule(zs, lead, own, base, sys_code, base,
                                     tr_i, te_i, seed)
        raw["ghosts"] = ghosts
        record("FLAT", flat, time.time() - t0, thr=thr, ghost_thr=thr,
               ghost_max=float(ghosts.max()), ghost_med=float(np.median(ghosts)),
               donor_fallback=fb)
    except torch.cuda.OutOfMemoryError as exc:
        infeasible("FLAT", exc)
        sys_code = None

    # ---- hierarchy arms need the system code
    if sys_code is None:
        for arm in HIER:
            failures.append(dict(V=V, seed=seed, arm=arm, kind="Dependency",
                                 detail="FLAT infeasible; system code absent"))
            status[arm] = "infeasible"
    else:
        m = max(2, V // 6)
        lab_clust = HR.cluster_train_only(x, raw_cutoff, m)
        sizes = np.bincount(lab_clust, minlength=m)
        parts = {
            "HIER-CLUST-TRAIN": lab_clust,
            "HIER-RAND-SIZED": HR.sized_random(sizes, Vt,
                                               np.random.default_rng(seed + 41)),
            "HIER-TRUE": HR.true_modules(Vt, m, parent, n_src),
        }
        for arm, lab in parts.items():
            gpu_reset()
            t0 = time.time()
            try:
                e2, e3, widths = HR.module_readout(lab, zs, own, lead, base,
                                                   tr, tr_i, te_i, sys_code,
                                                   Vt, seed)
                tot2, tot3 = float(e2[is_driven].mean()), float(e3[is_driven].mean())
                denom = tot2 + tot3
                excl = abs(denom) < SHARE_DENOM_EPS
                sz = HR.size_stats(lab, Vt, is_driven)
                raw[f"{arm}_lab"] = lab
                raw[f"{arm}_e2"] = e2
                raw[f"{arm}_e3"] = e3
                record(arm, e2 + e3, time.time() - t0,
                       mod_share=(np.nan if excl else tot2 / denom),
                       share_excluded=excl, mean_e2_driven=tot2,
                       mean_e3_driven=tot3,
                       target_weighted=sz["target_weighted"],
                       driven_weighted=sz["driven_weighted"],
                       n_modules=int(len(sz["sizes"])),
                       max_width=int(widths.max()))
            except torch.cuda.OutOfMemoryError as exc:
                infeasible(arm, exc)

    raw["status"] = json.dumps(status)
    np.savez_compressed(OUT / f"raw_V{V}_s{seed}.npz", **raw)
    extra["cell_secs"] = time.time() - t_cell
    return rows, status, failures, extra


# ---------------------------------------------------------- preflight

def provenance_preflight() -> bool:
    print("PROVENANCE PREFLIGHT (CPU only)")
    ok = True
    # 1. train-only clustering invariance, re-run at V=120 on a fresh seed
    inv_ok, trials = HR.invariance_test(V=120, noise=0.0, seed=100, n_trials=3)
    print(f"   clustering invariance V=120: "
          f"{'PASS' if inv_ok else 'FAIL'} "
          f"(ARI {[round(t['ari'], 6) for t in trials]})")
    ok &= inv_ok
    # 2. standardisation statistics from training rows only
    x, _, _ = make_system(N, 30, COUPLING, 0, 100)
    emb = embed(x)
    a = int(0.6 * emb.shape[0])
    mu_tr = emb[:a].mean(0)
    xp = x.copy()
    xp[a + 50:] += 100.0                      # perturb far past the cutoff
    embp = embed(xp)
    same = np.allclose(embp[:a].mean(0), mu_tr)
    print(f"   standardisation uses train rows only: {'PASS' if same else 'FAIL'}")
    ok &= same
    # 3. LAGCORR uses training rows only
    lc1 = lagcorr_scores(x, a)
    lc2 = lagcorr_scores(xp, a)
    same2 = np.allclose(lc1, lc2)
    print(f"   LAGCORR invariant to held-out rows: {'PASS' if same2 else 'FAIL'}")
    ok &= same2
    # 4. resource helpers return finite numbers
    rss, free = host_rss_mb(), sys_free_mb()
    fin = np.isfinite(rss) and np.isfinite(free) and rss > 0 and free > 0
    print(f"   memory helpers: rss {rss:.0f} MB, system free {free:.0f} MB "
          f"{'PASS' if fin else 'FAIL'}")
    ok &= bool(fin)
    print(f"   -> {'ALL PASS' if ok else 'FAILED'}\n")
    return ok


# ------------------------------------------------------------------ main

def _run() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not provenance_preflight():
        print("Not proceeding: provenance preflight failed.")
        return 1
    if not HR.acquire_lock(what="large_system"):
        return 1
    print(f"device: {DEV}   caps: GPU {CAP_GPU_MB:.0f} MB, RSS {CAP_RSS_MB:.0f} "
          f"MB, free>={CAP_FREE_MB:.0f} MB, {CAP_CELL_SEC//60} min/cell\n")

    all_rows, all_fail, limits, extras = [], [], [], []
    t_all = time.time()
    for V in WIDTHS:
        free = sys_free_mb()
        if free < CAP_FREE_MB:
            msg = f"system free RAM {free:.0f} MB < {CAP_FREE_MB:.0f} MB at start of V={V}"
            print(f"HARDWARE-LIMITED at V={V} and above: {msg}")
            limits.append(dict(V=V, reason=msg))
            break
        stage_ok = True
        for seed in SEEDS:
            cached = load_bundle(V, seed)
            if cached is not None:
                all_rows += cached["rows"]
                all_fail += cached["failures"]
                extras.append(dict(V=V, seed=seed, **cached["extra"]))
                print(f"  V={V:4d} seed={seed}  RESUMED", flush=True)
                continue
            if time.time() - t_all > CAP_TOTAL_SEC:
                limits.append(dict(V=V, reason="total wall-clock cap"))
                stage_ok = False
                break
            try:
                rows, status, fails, extra = cell(V, seed)
            except CapBreach as exc:
                msg = f"{exc.cap} {exc.value:.0f} > {exc.limit:.0f} in {exc.arm} (seed {seed})"
                print(f"HARDWARE-LIMITED at V={V} and above: {msg}")
                limits.append(dict(V=V, reason=msg))
                all_fail.append(dict(V=V, seed=seed, arm=exc.arm,
                                     kind="CapBreach", detail=msg))
                stage_ok = False
                break
            save_bundle(V, seed, rows, status, fails, extra)
            all_rows += rows
            all_fail += fails
            extras.append(dict(V=V, seed=seed, **extra))
            st = " ".join(f"{a.split('-')[-1][:5]}={'ok' if status[a]=='ok' else 'X'}"
                          for a in ARMS)
            print(f"  V={V:4d} seed={seed}  {st}  {extra['cell_secs']/60:.1f}m  "
                  f"rss {host_rss_mb():.0f}MB free {sys_free_mb():.0f}MB",
                  flush=True)
            pd.DataFrame(all_rows).to_csv(OUT / "cells.csv", index=False)
            pd.DataFrame(all_fail).to_csv(OUT / "failures.csv", index=False)
            pd.DataFrame(limits).to_csv(OUT / "hardware_limits.csv", index=False)
            pd.DataFrame(extras).to_csv(OUT / "capacity.csv", index=False)
        if not stage_ok:
            break

    pd.DataFrame(all_rows).to_csv(OUT / "cells.csv", index=False)
    pd.DataFrame(all_fail).to_csv(OUT / "failures.csv", index=False)
    pd.DataFrame(limits).to_csv(OUT / "hardware_limits.csv", index=False)
    pd.DataFrame(extras).to_csv(OUT / "capacity.csv", index=False)
    if not all_rows:
        print("No results.")
        return 1
    report(pd.DataFrame(all_rows), pd.DataFrame(all_fail),
           pd.DataFrame(limits), pd.DataFrame(extras))
    return 0


def report(d, f, lim, ex) -> None:
    print(f"\n({len(d) // len(ARMS)} cells; {len(f)} arm failures; "
          f"{len(lim)} hardware limits)\n")
    print("HARDWARE-LIMITED WIDTHS")
    print("   " + (lim.to_string(index=False).replace("\n", "\n   ")
                   if len(lim) else "none within the declared caps"))
    print("\nUNEQUAL CAPACITY (train rows / ridge feature width), by V")
    print("   " + ex.groupby("V").capacity_ratio.median().round(2).to_string()
          .replace("\n", "\n   "))
    print("\nSOURCE AVERAGE PRECISION (chance = prevalence)")
    piv = d.pivot_table(index="V", columns="arm", values="ap_source")
    piv["prevalence"] = d.groupby("V").prevalence.first()
    print("   " + piv.reindex(columns=[a for a in ARMS if a in piv] + ["prevalence"])
          .round(3).to_string().replace("\n", "\n   "))
    dr = d[d.arm == "FLAT"]
    print("\nDRIVEN RECALL AT THE DEPLOYED RULE (FLAT only: the rule is FLAT's own ghost panel), and source FP at that rule")
    print("   " + dr.pivot_table(index="V", columns="arm",
                                 values=["recall_rule", "source_fp_rule"])
          .round(3).to_string().replace("\n", "\n   "))
    h = d[d.arm.isin(HIER)]
    if len(h):
        print("\nMODULE SHARE (hierarchy arms; NaN = predefined exclusion)")
        print("   " + h.pivot_table(index="V", columns="arm", values="mod_share")
              .round(3).to_string().replace("\n", "\n   "))
        p = h.pivot_table(index=["V", "seed"], columns="arm", values="mod_share")
        if {"HIER-CLUST-TRAIN", "HIER-RAND-SIZED"} <= set(p.columns):
            c = p.dropna(subset=["HIER-CLUST-TRAIN", "HIER-RAND-SIZED"])
            for V, g in c.groupby(level="V"):
                n_up = int((g["HIER-CLUST-TRAIN"] > g["HIER-RAND-SIZED"]).sum())
                print(f"   V={V}: CLUST-TRAIN > RAND-SIZED in {n_up} of {len(g)} "
                      f"comparable seeds")
        print("\n   target-weighted module width")
        print("   " + h.pivot_table(index="V", columns="arm",
                                    values="target_weighted").round(2)
              .to_string().replace("\n", "\n   "))
    print("\nCOMPUTE per arm (median over seeds)")
    print("   " + d.pivot_table(index="V", columns="arm",
                                values=["secs", "gpu_peak_mb", "host_rss_mb"],
                                aggfunc="median").round(0)
          .to_string().replace("\n", "\n   "))
    print("\nDescriptive only. Three seeds are not a rate. No verdict is drawn.")


def main() -> int:
    try:
        return _run()
    finally:
        HR.release_lock()


if __name__ == "__main__":
    raise SystemExit(main())
