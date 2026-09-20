"""Stage B arms for bounded-code parent screening.

Pre-registration: paper/parent_screening_protocol.md (amendment 9b1ec46 plus
the Stage B clarification amendment committed before this file's first
scientific use). A SEPARATE module from parent_screening.py on purpose: that
file's shared machinery was independently reviewed and is kept stable; this
file adds the six required screening arms, the decision gate, the resource
guard and the per-arm preflight, so a reviewer can diff only what is new.

Arms (all receive ONLY the observed series and V; truth lives in the
evaluator, never here):

  1 RANDOM               k uniform-without-replacement candidates per target
  2 LAGCORR              max |lagged correlation|, lags 1-3, two variants
                         (raw; train-residualised), the DEPLOYED variant
                         chosen on VALIDATION rows, both reported
  3 LASSO                sparse fit on a fixed univariate lag+hinge-spline
                         dictionary, target residualised on own history,
                         penalty tuned on an embargoed inner split
  4 PCA-GROUP            train-clustered groups, PCA code, same b_G
  5 LEARNED-SIZEDRAND    size-matched random partition, learned encoders
  6 LEARNED-CLUSTERED    train-clustered groups, learned encoders (the
                         candidate of interest)

Arms 4-6 apply the REGISTERED unresolved rule (max group gain <= 0.01 -> the
target keeps all V-1 candidates and fails G4 by construction). It is not
tuned, relaxed or bypassed anywhere in this file.

    python scripts/parent_screening_arms.py --preflight
    python scripts/parent_screening_arms.py --engineering-diag
    python scripts/parent_screening_arms.py --timing       # bounded, locked
    python scripts/parent_screening_arms.py --run-pilot    # needs clearance
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import numbers
import os
import subprocess
import sys
import time
import warnings
from ctypes import wintypes
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import parent_screening as PS  # noqa: E402
from boundary_map import poly3  # noqa: E402

DEV = PS.DEV
HERE = Path(__file__).parent
OUT = Path("ExpOutput/parent_screening")
CLEARANCE = Path("ExpOutput/parent_screening_pilot_cleared")
LOCK = Path(".agent-lock")
GUARDED_FILES = ("scripts/parent_screening.py", "scripts/parent_screening_arms.py",
                 "scripts/parent_screening_split_test.py",
                 "paper/parent_screening_protocol.md")

ARMS = ["RANDOM", "LAGCORR", "LASSO", "PCA-GROUP", "LEARNED-SIZEDRAND",
        "LEARNED-CLUSTERED"]
GROUP_ARMS = ("PCA-GROUP", "LEARNED-SIZEDRAND", "LEARNED-CLUSTERED")
FAMILIES = ("family1", "family2")
LAGS = (1, 2, 3)
LASSO_GRID_REL = (10 ** -1.0, 10 ** -1.5, 10 ** -2.0, 10 ** -2.5, 10 ** -3.0)
LASSO_MAX_ITER, LASSO_TOL = 2000, 1e-4

CAP_RUNTIME_SEC = 3600            # Stage B total, per the protocol
CAP_GPU_GIB = 6.0
CAP_RSS_MB = 3 * 1024
CAP_FREE_MB = 2 * 1024
CAP_DISK_MB = 100

FROZEN_SEEDS = {
    "family1": (20001, 20002, 20003, 20004, 20005, 20006),
    "family2": (21001, 21002, 21003, 21004, 21005, 21006),
}
METRIC_KEYS = ("recall", "coverage", "macro_recall", "candidate_fraction",
               "unresolved_fraction")


# ================================================================
# Windows process introspection (private DLL handles, no global side effects)
# ================================================================

_k32 = ctypes.WinDLL("kernel32")
_psapi = ctypes.WinDLL("psapi")
_INVALID = ctypes.c_void_p(-1).value


class _PE32(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_wchar * 260)]


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


class _MEMSTAT(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64)]


_k32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
_k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
_k32.Process32FirstW.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PE32)]
_k32.Process32FirstW.restype = wintypes.BOOL
_k32.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PE32)]
_k32.Process32NextW.restype = wintypes.BOOL
_k32.CloseHandle.argtypes = [ctypes.c_void_p]
_k32.OpenProcess.restype = ctypes.c_void_p
_k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_k32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MEMSTAT)]
_k32.GlobalMemoryStatusEx.restype = wintypes.BOOL
_psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC),
                                        wintypes.DWORD]
_psapi.GetProcessMemoryInfo.restype = wintypes.BOOL


def descendant_pids(root_pid: int) -> list[int]:
    """Every descendant of root_pid, from a ToolHelp32 process snapshot."""
    snap = _k32.CreateToolhelp32Snapshot(0x2, 0)          # TH32CS_SNAPPROCESS
    if snap in (None, _INVALID):
        raise OSError("CreateToolhelp32Snapshot failed")
    kids: dict[int, list[int]] = {}
    try:
        pe = _PE32()
        pe.dwSize = ctypes.sizeof(pe)
        ok = _k32.Process32FirstW(snap, ctypes.byref(pe))
        while ok:
            kids.setdefault(int(pe.th32ParentProcessID), []).append(
                int(pe.th32ProcessID))
            ok = _k32.Process32NextW(snap, ctypes.byref(pe))
    finally:
        _k32.CloseHandle(snap)
    out, stack = [], [root_pid]
    while stack:
        for c in kids.get(stack.pop(), []):
            if c not in out:
                out.append(c)
                stack.append(c)
    return out


def rss_mb(pid: int) -> float:
    """Working set of one process in MiB; 0.0 if it cannot be opened (already
    exited)."""
    h = _k32.OpenProcess(0x1000, False, pid)          # QUERY_LIMITED_INFORMATION
    if not h:
        return 0.0
    try:
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(pmc)
        if not _psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return 0.0
        return pmc.WorkingSetSize / 2 ** 20
    finally:
        _k32.CloseHandle(h)


def free_mb() -> float:
    s = _MEMSTAT()
    s.dwLength = ctypes.sizeof(s)
    _k32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPhys / 2 ** 20


# ================================================================
# Resource guard: monotonic clock, threaded through the loops, tree RSS
# ================================================================

class CapBreach(RuntimeError):
    pass


class ResourceGuard:
    """Every declared cap, checked INSIDE the work loops (per target, per
    group, per encoder epoch), not only between arms. Monotonic clock. RSS is
    the WHOLE PROCESS TREE: this process plus every descendant, measured,
    not assumed from the fact that Stage B spawns none. check_light is cheap
    (clock, free RAM, GPU peak) and runs at loop granularity; check_full
    (tree RSS, disk) is rate-limited to once per 10 s."""

    def __init__(self, runtime_cap_sec: float = CAP_RUNTIME_SEC):
        self.t0 = time.perf_counter()
        self.runtime_cap = runtime_cap_sec
        self.peak_tree_rss = 0.0
        self.min_free = float("inf")
        self.n_light = self.n_full = 0
        self._last_full = -1e9
        if DEV == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def tree_rss_mb(self) -> float:
        return rss_mb(os.getpid()) + sum(rss_mb(p) for p in
                                         descendant_pids(os.getpid()))

    def gpu_peak_gib(self) -> float:
        return (torch.cuda.max_memory_allocated() / 2 ** 30
                if DEV == "cuda" else 0.0)

    def check_light(self, where: str):
        self.n_light += 1
        el = self.elapsed()
        if el > self.runtime_cap:
            raise CapBreach(f"runtime {el:.1f}s > {self.runtime_cap:.1f}s "
                            f"at {where}")
        fr = free_mb()
        self.min_free = min(self.min_free, fr)
        if fr < CAP_FREE_MB:
            raise CapBreach(f"free RAM {fr:.0f}MB < {CAP_FREE_MB}MB at {where}")
        gpu = self.gpu_peak_gib()
        if gpu > CAP_GPU_GIB:
            raise CapBreach(f"GPU {gpu:.2f}GiB > {CAP_GPU_GIB}GiB at {where}")
        if el - self._last_full >= 10.0:
            self.check_full(where)

    def check_full(self, where: str):
        self.n_full += 1
        self._last_full = self.elapsed()
        tree = self.tree_rss_mb()
        self.peak_tree_rss = max(self.peak_tree_rss, tree)
        if tree > CAP_RSS_MB:
            raise CapBreach(f"process-TREE RSS {tree:.0f}MB > {CAP_RSS_MB}MB "
                            f"at {where}")
        disk = (sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
                / 2 ** 20) if OUT.exists() else 0.0
        if disk > CAP_DISK_MB:
            raise CapBreach(f"disk {disk:.1f}MiB > {CAP_DISK_MB}MiB at {where}")

    def snapshot(self) -> dict:
        return dict(elapsed_sec=self.elapsed(),
                    peak_tree_rss_mb=self.peak_tree_rss,
                    min_free_mb=self.min_free, gpu_peak_gib=self.gpu_peak_gib(),
                    n_light_checks=self.n_light, n_full_checks=self.n_full)


# ================================================================
# Shared preprocessing (standardise from the TRAIN raw span only)
# ================================================================

def prep_system(x_obs: np.ndarray) -> dict:
    n, V = x_obs.shape
    tr, va, te, m, embargo = PS.splits_for(n)
    raw_end = PS.MAX_DELAY + int(tr[-1]) + 2      # exclusive end of train's raw support
    mu = x_obs[:raw_end].mean(0)
    sd = x_obs[:raw_end].std(0) + 1e-12
    xs = np.nan_to_num((x_obs - mu) / sd)
    own_raw, target = [], []
    for q in range(V):
        o, t_, _ = PS.own_lag_window(xs, q)
        own_raw.append(o)
        target.append(t_)
    feats = np.stack([poly3(o) for o in own_raw])            # (V, m, 19)
    tgt = np.stack(target)
    return dict(
        V=V, n=n, m=m, tr=tr, va=va, te=te, xs=xs, raw_end=raw_end,
        own_raw=own_raw, target=tgt,
        feats_t=torch.as_tensor(feats, dtype=torch.float64, device=DEV),
        target_t=torch.as_tensor(tgt, dtype=torch.float64, device=DEV),
        tr_t=torch.as_tensor(tr, device=DEV),
        va_t=torch.as_tensor(va, device=DEV))


def _own_r2(prep, q):
    F, y = prep["feats_t"][q], prep["target_t"][q]
    tr, va = prep["tr_t"], prep["va_t"]
    return PS.ridge_r2_val(F[tr], y[tr], F[va], y[va])[0]


def _ridge_w(F, y):
    a, _ = PS.ridge_select_alpha(F, y)
    return PS.ridge_fit_predict(F, y, a)


# ================================================================
# Arm 1: RANDOM
# ================================================================

def screen_random(V: int, k: int, seed: int):
    C = {}
    for q in range(V):
        rng = np.random.default_rng([seed, q])
        others = np.array([j for j in range(V) if j != q])
        C[q] = sorted(rng.choice(others, size=min(k, V - 1),
                                 replace=False).tolist())
    return C, {q: False for q in range(V)}, {}


# ================================================================
# Arm 2: LAGCORR, two variants, validation-chosen
# ================================================================

def _lag_matrices(prep):
    xs, md, m = prep["xs"], PS.MAX_DELAY, prep["m"]
    L = {d: xs[md + 1 - d: md + 1 - d + m] for d in LAGS}     # (m, V) each
    T = xs[md + 1: md + 1 + m]
    return L, T


def _abs_corr(Ld_tr, Tq_tr):
    a = (Ld_tr - Ld_tr.mean(0)) / (Ld_tr.std(0) + 1e-12)
    b = (Tq_tr - Tq_tr.mean(0)) / (Tq_tr.std(0) + 1e-12)
    return np.abs(a.T @ b) / len(a)


def _own_residual_train(prep, guard=None):
    V, tr_t = prep["V"], prep["tr_t"]
    res = np.empty((V, len(prep["tr"])))
    for q in range(V):
        if guard:
            guard.check_light("lagcorr residual")
        F, y = prep["feats_t"][q][tr_t], prep["target_t"][q][tr_t]
        w = _ridge_w(F, y)
        res[q] = (y - PS.ridge_predict(F, w)).cpu().numpy()
    return res


def screen_lagcorr(prep, k: int, guard=None):
    """Both variants, the validation proxy per variant, and the DEPLOYED
    choice. The proxy is each variant's mean validation R2 improvement when a
    target's top-k candidates (each contributing its single best-lag column)
    are appended to that target's own-history fit. Every fit uses TRAIN rows
    only; validation rows only score."""
    V, tr = prep["V"], prep["tr"]
    L, T = _lag_matrices(prep)
    resid = _own_residual_train(prep, guard)
    results = {}
    for variant in ("raw", "resid"):
        Tq = T[tr] if variant == "raw" else resid.T
        per_lag = np.stack([_abs_corr(L[d][tr], Tq) for d in LAGS])
        score = per_lag.max(0)
        best_lag = np.array(LAGS)[per_lag.argmax(0)]
        np.fill_diagonal(score, -np.inf)
        C = {}
        for q in range(V):
            order = np.lexsort((np.arange(V), -score[:, q]))
            C[q] = sorted(order[:k].tolist())
        results[variant] = dict(C=C, best_lag=best_lag)
    proxy = {}
    tr_t, va_t = prep["tr_t"], prep["va_t"]
    for variant, r in results.items():
        gains = []
        for q in range(V):
            if guard:
                guard.check_light("lagcorr proxy")
            cols = np.stack([L[int(r["best_lag"][j, q])][:, j]
                             for j in r["C"][q]], axis=1)
            Lc = torch.as_tensor(cols, dtype=torch.float64, device=DEV)
            F, y = prep["feats_t"][q], prep["target_t"][q]
            X = torch.cat([F, Lc], dim=1)
            r_full = PS.ridge_r2_val(X[tr_t], y[tr_t], X[va_t], y[va_t])[0]
            gains.append(r_full - _own_r2(prep, q))
        proxy[variant] = float(np.mean(gains))
    deployed = "resid" if proxy["resid"] > proxy["raw"] else "raw"
    return results, proxy, deployed


# ================================================================
# Arm 3: LASSO on a fixed univariate lag + hinge-spline dictionary
# ================================================================

def _dictionary(prep):
    """(m, 7V): per variable j, [lag1, lag2, lag3, hinge(lag1) at the
    20/40/60/80% TRAIN quantiles]; every column standardised with OUTER-TRAIN
    statistics (temporal preprocessing, as registered). Block for variable j
    is columns 7j .. 7j+6."""
    L, _ = _lag_matrices(prep)
    tr, V = prep["tr"], prep["V"]
    blocks = []
    for j in range(V):
        l1 = L[1][:, j]
        knots = np.quantile(l1[tr], [0.2, 0.4, 0.6, 0.8])
        cols = [L[1][:, j], L[2][:, j], L[3][:, j]] + \
               [np.maximum(0.0, l1 - kn) for kn in knots]
        blocks.append(np.stack(cols, axis=1))
    D = np.concatenate(blocks, axis=1)
    mu, sd = D[tr].mean(0), D[tr].std(0) + 1e-12
    return (D - mu) / sd


def _fit_lasso(X, r, rel_alpha, a_max, warm=None):
    """One sklearn Lasso fit; returns (model, n_nonconverged)."""
    from sklearn.linear_model import Lasso
    from sklearn.exceptions import ConvergenceWarning
    model = warm or Lasso(alpha=a_max * rel_alpha, fit_intercept=False,
                          max_iter=LASSO_MAX_ITER, tol=LASSO_TOL,
                          warm_start=True)
    model.set_params(alpha=a_max * rel_alpha)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.fit(X, r)
    return model, int(any(issubclass(x.category, ConvergenceWarning)
                          for x in w))


def lasso_tuning_fit(Xfit, rfit_raw, Xval, rval_raw):
    """The ONLY function that both fits and scores during tuning. It is handed
    inner-fit rows for every fitted quantity (centring, scale, alpha_max,
    coefficients) and inner-validation rows for scoring only, transformed with
    the FIXED inner-fit statistics. Returns (best_relative_alpha, stats)."""
    mu, sd = float(rfit_raw.mean()), float(rfit_raw.std()) + 1e-12
    rfit, rval = (rfit_raw - mu) / sd, (rval_raw - mu) / sd
    a_max = float(np.max(np.abs(Xfit.T @ rfit)) / len(rfit)) + 1e-12
    model, bad, errs = None, 0, []
    for g in LASSO_GRID_REL:
        model, b = _fit_lasso(Xfit, rfit, g, a_max, model)
        bad += b
        errs.append(float(np.mean((Xval @ model.coef_ - rval) ** 2)))
    return LASSO_GRID_REL[int(np.argmin(errs))], dict(
        mu=mu, sd=sd, a_max_fit=a_max, nonconverged=bad)


def lasso_target(prep, D, q):
    """One target's LASSO candidate scores, tuned WITHOUT inner-validation
    leakage. TUNING: the own-history residualiser, the residual's centring and
    scale, alpha_max and every coefficient come from inner-fit rows only;
    inner-validation rows are transformed with those fixed quantities and
    used only to score each grid point. FINAL (all outer-train rows): the
    residualiser, centring, scale and alpha_max are refit on the full
    outer-train block and the chosen RELATIVE alpha is applied to the
    full-train alpha_max."""
    tr, tr_t = prep["tr"], prep["tr_t"]
    itr, iva = PS.internal_val_split(len(tr))
    itr_t = torch.as_tensor(itr, device=DEV)
    iva_t = torch.as_tensor(iva, device=DEV)
    Ftr, ytr = prep["feats_t"][q][tr_t], prep["target_t"][q][tr_t]
    keep = np.ones(D.shape[1], bool)
    keep[7 * q: 7 * q + 7] = False                            # drop own block
    X = D[tr][:, keep]

    w_fit = _ridge_w(Ftr[itr_t], ytr[itr_t])
    r_fit = (ytr[itr_t] - PS.ridge_predict(Ftr[itr_t], w_fit)).cpu().numpy()
    r_val = (ytr[iva_t] - PS.ridge_predict(Ftr[iva_t], w_fit)).cpu().numpy()
    best_rel, stats = lasso_tuning_fit(X[itr], r_fit, X[iva], r_val)

    w_full = _ridge_w(Ftr, ytr)
    r_full = (ytr - PS.ridge_predict(Ftr, w_full)).cpu().numpy()
    mu_f, sd_f = float(r_full.mean()), float(r_full.std()) + 1e-12
    r_full_s = (r_full - mu_f) / sd_f
    a_max_full = float(np.max(np.abs(X.T @ r_full_s)) / len(r_full_s)) + 1e-12
    model, bad = _fit_lasso(X, r_full_s, best_rel, a_max_full)
    blk = np.abs(model.coef_).reshape(-1, 7).sum(1)           # per candidate
    stats.update(best_rel=best_rel, nonconverged=stats["nonconverged"] + bad,
                 w_fit=w_fit.cpu().numpy())
    return blk, stats


def screen_lasso(prep, k: int, guard=None, max_targets: int | None = None):
    V = prep["V"]
    D = _dictionary(prep)
    C, alpha_rel, bad = {}, {}, 0
    targets = range(V) if max_targets is None else range(min(max_targets, V))
    for q in targets:
        if guard:
            guard.check_light("lasso target")
        blk, info = lasso_target(prep, D, q)
        cand = np.array([j for j in range(V) if j != q])
        order = np.lexsort((cand, -blk))
        C[q] = sorted(int(cand[i]) for i in order[:k])
        alpha_rel[q] = info["best_rel"]
        bad += info["nonconverged"]
    return C, {q: False for q in C}, dict(alpha_rel=alpha_rel,
                                          nonconverged_fits=bad)


# ================================================================
# Arms 4-6: group screens (learned / PCA codes), one shared scoring path
# ================================================================

def _pca_codes(z, tr, members, b):
    mu = z[tr].mean(0)
    _, _, Vh = np.linalg.svd(z[tr] - mu, full_matrices=False)
    comps = Vh[:b]
    full = (z - mu) @ comps.T
    excl = []
    for pos in range(len(members)):
        ze = z.copy()
        ze[:, pos * PS.E:(pos + 1) * PS.E] = 0.0
        excl.append((ze - mu) @ comps.T)
    return full, excl


def screen_groups(prep, labels, kind: str, k: int, seed_base: int, guard=None):
    """kind in {'learned','pca'}. One encoder (or PCA basis) per group;
    per-(target,group) ridge readouts; own-history R2 computed ONCE per
    target; codes for non-member targets computed ONCE per group. The full
    per-target gain table is returned in diag so the abstention decision can
    be audited and the invariance check can compare gains, not only sets."""
    V, tr_t, va_t = prep["V"], prep["tr_t"], prep["va_t"]
    groups = {g: sorted(np.where(labels == g)[0].tolist())
              for g in sorted(set(labels.tolist()))}
    code_full, code_excl, pos_of = {}, {}, {}
    as_t = lambda a: torch.as_tensor(a, dtype=torch.float64,       # noqa: E731
                                     device=DEV)
    for gid, members in groups.items():
        if guard:
            guard.check_light("group encoder")
        z = np.concatenate([prep["own_raw"][j] for j in members],
                           axis=1).astype(np.float32)
        b = PS.group_bottleneck(len(members))
        pos_of[gid] = {j: i for i, j in enumerate(members)}
        if kind == "learned":
            net = PS.train_group_encoder(z, prep["tr"], len(members),
                                         seed=seed_base + gid, guard=guard)
            full = PS.group_code(net, z, None)
            excl = [PS.group_code(net, z, p) for p in range(len(members))]
        else:
            full, excl = _pca_codes(z, prep["tr"], members, b)
        code_full[gid] = as_t(full)
        code_excl[gid] = [as_t(e) for e in excl]
    gains, C, unresolved = {}, {}, {}
    alpha_hits = alpha_total = 0
    for q in range(V):
        if guard:
            guard.check_light("group scoring")
        F, y = prep["feats_t"][q], prep["target_t"][q]
        r2_own = PS.ridge_r2_val(F[tr_t], y[tr_t], F[va_t], y[va_t])[0]
        gains[q] = {}
        for gid, members in groups.items():
            if all(j == q for j in members):
                continue                # only q itself: nothing to rank
            code = (code_excl[gid][pos_of[gid][q]] if q in pos_of[gid]
                    else code_full[gid])
            X = torch.cat([F, code], dim=1)
            r2_full, _, hit = PS.ridge_r2_val(X[tr_t], y[tr_t],
                                              X[va_t], y[va_t])
            gains[q][gid] = r2_full - r2_own
            alpha_hits += int(hit)
            alpha_total += 1
        C[q], unresolved[q] = PS.build_candidate_set(q, gains[q], groups, k)
    diag = dict(n_groups=len(groups),
                sizes=sorted(len(v) for v in groups.values()),
                alpha_boundary_rate=alpha_hits / max(alpha_total, 1),
                init_seeds=[seed_base + g for g in groups],
                gains=gains,
                max_gain={q: (max(g.values()) if g else float("nan"))
                          for q, g in gains.items()})
    return C, unresolved, diag


def clustered_labels(prep):
    return PS.cluster_size_capped(prep["xs"][:prep["raw_end"]])


def sized_random_labels(labels, V, seed):
    return PS.sized_random(np.bincount(labels), V, np.random.default_rng(seed))


# ================================================================
# Running every arm on one system
# ================================================================

def run_all_arms(x_obs: np.ndarray, seed: int, guard=None,
                 timings: dict | None = None, lasso_targets=None):
    prep = prep_system(x_obs)
    V = prep["V"]
    k = PS.k_for(V)
    out = {}

    def timed(name, fn):
        t0 = time.perf_counter()
        res = fn()
        if timings is not None:
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - t0
        return res

    out["RANDOM"] = timed("RANDOM", lambda: screen_random(V, k, seed))
    lc, proxy, deployed = timed("LAGCORR",
                                lambda: screen_lagcorr(prep, k, guard))
    out["LAGCORR"] = (lc[deployed]["C"], {q: False for q in range(V)},
                      dict(proxy=proxy, deployed=deployed,
                           C_raw=lc["raw"]["C"], C_resid=lc["resid"]["C"]))
    out["LASSO"] = timed("LASSO", lambda: screen_lasso(
        prep, k, guard, max_targets=lasso_targets))
    labels = timed("CLUSTER", lambda: clustered_labels(prep))
    out["PCA-GROUP"] = timed("PCA-GROUP", lambda: screen_groups(
        prep, labels, "pca", k, seed * 1000, guard))
    rl = sized_random_labels(labels, V, seed + 777)
    out["LEARNED-SIZEDRAND"] = timed("LEARNED-SIZEDRAND",
        lambda: screen_groups(prep, rl, "learned", k, seed * 1000, guard))
    out["LEARNED-CLUSTERED"] = timed("LEARNED-CLUSTERED",
        lambda: screen_groups(prep, labels, "learned", k, seed * 1000, guard))
    return out, prep, labels


# ================================================================
# Metrics and the machine-checkable decision gate (G1-G4)
# ================================================================

def arm_metrics(C, unresolved, parent, V, k):
    non_root = [q for q in parent if parent[q]]
    resolved = [q for q in non_root if not unresolved.get(q, False)]
    r_num = sum(len(set(C[q]) & {j for j, _ in parent[q]}) for q in resolved)
    r_den = sum(len(parent[q]) for q in resolved)
    return dict(
        recall=PS.retained_parent_recall(C, parent),
        coverage=PS.complete_target_coverage(C, parent),
        macro_recall=float(np.mean([
            len(set(C[q]) & {j for j, _ in parent[q]}) / len(parent[q])
            for q in non_root])),
        candidate_fraction=PS.candidate_fraction(C, V, non_root),
        unresolved_fraction=PS.unresolved_fraction(unresolved, non_root),
        budget_ok=PS.budget_ok(C, k, V, non_root),
        n_roots=V - len(non_root),
        recall_resolved_only=(r_num / r_den if r_den else None),
        n_resolved=len(resolved))


def _finite_unit(v) -> bool:
    return (isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_))
            and bool(np.isfinite(v)) and 0.0 <= float(v) <= 1.0)


def validate_pilot_inputs(per_family) -> list[str]:
    """Every reason the gate must REFUSE to evaluate. Empty list = complete.
    Required: exactly the two families; exactly the six arms in each; exactly
    the six registered seeds per arm, unique; every metric a finite number in
    [0,1]; budget_ok a real bool."""
    if not isinstance(per_family, dict) or set(per_family) != set(FAMILIES):
        got = sorted(per_family) if isinstance(per_family, dict) else per_family
        return [f"families must be exactly {FAMILIES}, got {got}"]
    problems = []
    for fam in FAMILIES:
        arms = per_family[fam]
        if not isinstance(arms, dict) or set(arms) != set(ARMS):
            problems.append(f"{fam}: arms must be exactly {ARMS}, got "
                            f"{sorted(arms) if isinstance(arms, dict) else arms}")
            continue
        want = sorted(FROZEN_SEEDS[fam])
        for a in ARMS:
            rows = arms[a]
            if not isinstance(rows, list) or not all(
                    isinstance(r, dict) for r in rows):
                problems.append(f"{fam}/{a}: cells must be a list of dicts")
                continue
            seeds = [r.get("seed") for r in rows]
            try:
                seeds_sorted = sorted(seeds)
            except TypeError:
                seeds_sorted = None
            if seeds_sorted != want:
                problems.append(f"{fam}/{a}: seeds {seeds} != registered {want}")
            for r in rows:
                bad = [kx for kx in METRIC_KEYS if not _finite_unit(r.get(kx))]
                if bad:
                    problems.append(f"{fam}/{a}/{r.get('seed')}: missing, "
                                    f"non-finite or outside [0,1]: {bad}")
                if not isinstance(r.get("budget_ok"), (bool, np.bool_)):
                    problems.append(f"{fam}/{a}/{r.get('seed')}: budget_ok "
                                    f"missing or not a bool")
    return problems


def evaluate_gate(per_family, verbose: bool = True) -> bool:
    """True iff the input is COMPLETE and G1-G4 hold in BOTH families."""
    problems = validate_pilot_inputs(per_family)
    if problems:
        if verbose:
            print("  GATE REFUSES INCOMPLETE INPUT -> FAIL")
            for p in problems:
                print("    -", p)
        return False
    ok_all = True
    for fam in FAMILIES:
        arms = per_family[fam]
        mean = lambda arm, key: float(np.mean([r[key] for r in arms[arm]]))  # noqa: E731
        a6 = "LEARNED-CLUSTERED"
        g1 = mean(a6, "recall") >= 0.90
        g2 = mean(a6, "coverage") >= 0.80
        margins = {a: mean(a6, "recall") - mean(a, "recall")
                   for a in arms if a != a6}
        g3 = all(v >= 0.05 for v in margins.values())
        g4 = all(bool(r["budget_ok"]) for r in arms[a6])
        if verbose:
            print(f"  {fam}:")
            print(f"    G1 recall {mean(a6,'recall'):.3f} >= 0.90        "
                  f"-> {'PASS' if g1 else 'FAIL'}")
            print(f"    G2 coverage {mean(a6,'coverage'):.3f} >= 0.80      "
                  f"-> {'PASS' if g2 else 'FAIL'}")
            print(f"    G3 margin over every baseline >= 0.05   "
                  f"-> {'PASS' if g3 else 'FAIL'}")
            for a, v in sorted(margins.items()):
                print(f"        vs {a:18s} {v:+.3f}")
            n_ok = sum(bool(r["budget_ok"]) for r in arms[a6])
            print(f"    G4 fixed budget in every seed ({n_ok}/6 seeds ok) "
                  f"-> {'PASS' if g4 else 'FAIL'}")
        ok_all &= g1 and g2 and g3 and g4
    if verbose:
        print(f"  GATE (both families, all of G1-G4): "
              f"{'PASS' if ok_all else 'FAIL'}")
    return ok_all


# ================================================================
# Preflight checks (each targets a specific reviewed failure mode)
# ================================================================

def _synthetic_pilot(a6_recall=0.95, a6_cov=0.9, baseline_recall=0.5,
                     a6_budget=True, seeds=None, drop=None, nan_arm=None):
    pf = {}
    for fam in FAMILIES:
        sd = list(seeds[fam]) if seeds else list(FROZEN_SEEDS[fam])
        arms = {}
        for a in ARMS:
            rec = a6_recall if a == "LEARNED-CLUSTERED" else baseline_recall
            cov = a6_cov if a == "LEARNED-CLUSTERED" else 0.3
            bud = a6_budget if a == "LEARNED-CLUSTERED" else True
            arms[a] = [dict(seed=s, recall=rec, coverage=cov, macro_recall=rec,
                            candidate_fraction=0.1, unresolved_fraction=0.0,
                            budget_ok=bud) for s in sd]
        pf[fam] = arms
    if drop == "family":
        del pf["family2"]
    if drop == "arm":
        del pf["family1"]["LASSO"]
    if drop == "cells":
        pf["family1"] = {"LEARNED-CLUSTERED":
                         pf["family1"]["LEARNED-CLUSTERED"][:1]}
    if nan_arm:
        pf["family1"][nan_arm][0]["recall"] = float("nan")
    return pf


def gate_selftest() -> bool:
    one_cell = {"family1": {"LEARNED-CLUSTERED": [dict(
        seed=20001, recall=1.0, coverage=1.0, macro_recall=1.0,
        candidate_fraction=0.1, unresolved_fraction=0.0, budget_ok=True)]}}
    out_of_range = _synthetic_pilot()
    out_of_range["family2"]["RANDOM"][0]["coverage"] = 5.0
    cases = [
        ("empty input {}", {}, False),
        ("None input", None, False),
        ("reviewer repro: one family, one arm-6 cell, perfect", one_cell, False),
        ("missing a whole family", _synthetic_pilot(drop="family"), False),
        ("missing a baseline arm", _synthetic_pilot(drop="arm"), False),
        ("one cell only in a family", _synthetic_pilot(drop="cells"), False),
        ("duplicated seed",
         _synthetic_pilot(seeds={"family1": (20001,) * 6,
                                 "family2": FROZEN_SEEDS["family2"]}), False),
        ("unregistered seeds",
         _synthetic_pilot(seeds={"family1": (1, 2, 3, 4, 5, 6),
                                 "family2": FROZEN_SEEDS["family2"]}), False),
        ("non-finite metric", _synthetic_pilot(nan_arm="LASSO"), False),
        ("metric outside [0,1]", out_of_range, False),
        ("all-fallback arm 6: perfect recall but budget violated",
         _synthetic_pilot(a6_recall=1.0, a6_budget=False), False),
        ("baseline within 0.05 of arm 6",
         _synthetic_pilot(a6_recall=0.95, baseline_recall=0.93), False),
        ("arm 6 below the 0.90 recall bar",
         _synthetic_pilot(a6_recall=0.85), False),
        ("complete, arm 6 clears every gate", _synthetic_pilot(), True),
    ]
    ok = True
    for name, inp, want in cases:
        got = evaluate_gate(inp, verbose=False)
        good = got == want
        ok &= good
        print(f"    {'OK ' if good else 'BAD'} {name:58s} gate={got} "
              f"(want {want})")
    return ok


def tree_rss_selftest() -> bool:
    """A child process holding ~200MB must show up in the guard's tree RSS,
    proving the guard measures the tree rather than asserting it."""
    g = ResourceGuard()
    own = rss_mb(os.getpid())
    child = subprocess.Popen(
        [sys.executable, "-c",
         "import numpy as np,time; a=np.ones(25_000_000); time.sleep(25)"])
    try:
        time.sleep(4.0)
        tree = g.tree_rss_mb()
        kids = descendant_pids(os.getpid())
        ok = (child.pid in kids) and (tree - own > 150)
        print(f"    child pid seen: {child.pid in kids}; tree - own = "
              f"{tree - own:.0f}MB (need >150)   -> {'PASS' if ok else 'FAIL'}")
        return ok
    finally:
        child.kill()


class _CountingGuard(ResourceGuard):
    """Test double: counts check_light calls and breaches on the stop_at-th."""

    def __init__(self, stop_at: float = float("inf")):
        super().__init__(runtime_cap_sec=1e9)
        self.stop_at, self.calls = stop_at, 0

    def check_light(self, where: str):
        self.calls += 1
        if self.calls >= self.stop_at:
            raise CapBreach(f"synthetic stop after {self.calls} checks at "
                            f"{where}")


def guard_granularity_selftest() -> bool:
    """The guard must be polled INSIDE every long loop, once per iteration,
    and a breach must stop the arm mid-loop. For each loop: (a) a never-
    stopping counting guard shows the loop polls more than 3 times; (b) a
    guard that breaches on its 3rd poll stops the arm after exactly 3 polls,
    at the expected label (so a breach at iteration 3 of many cannot be
    deferred to the end of the arm). Then a real blown wall-clock cap raises
    at the first poll of the lasso loop."""
    x = PS.family1_generate(14, 1200, seed=9900)["x_obs"]
    prep = prep_system(x)
    labels = clustered_labels(prep)
    k = PS.k_for(14)
    n_groups = len(set(labels.tolist()))
    z_small = np.zeros((len(prep["xs"]) - 4, 6), np.float32)
    loops = [
        ("lasso target", 3, lambda g: screen_lasso(prep, k, g)),
        ("lagcorr residual", 3, lambda g: screen_lagcorr(prep, k, g)),
        ("lagcorr proxy", 14 + 3,
         lambda g: screen_lagcorr(prep, k, g)),
        ("encoder epoch", 3, lambda g: PS.train_group_encoder(
            z_small, prep["tr"], 2, seed=0, guard=g)),
        ("group scoring", n_groups + 3,
         lambda g: screen_groups(prep, labels, "pca", k, 0, g)),
    ]
    ok = True
    for label, stop_at, run in loops:
        free = _CountingGuard()
        run(free)
        stopper = _CountingGuard(stop_at)
        try:
            run(stopper)
            raised, msg = False, "no breach raised"
        except CapBreach as e:
            raised, msg = True, str(e)
        good = (free.calls > stop_at and raised and stopper.calls == stop_at
                and label in msg)
        ok &= good
        print(f"    {'OK ' if good else 'BAD'} '{label}': loop polls "
              f"{free.calls}x unstopped; breach at poll {stopper.calls}/"
              f"{stop_at} raised mid-arm at '{label}': {raised and label in msg}")
    g = ResourceGuard(runtime_cap_sec=0.0)
    time.sleep(0.05)
    try:
        screen_lasso(prep, k, g)
        good, msg = False, "no breach raised"
    except CapBreach as e:
        good, msg = "runtime" in str(e) and "lasso target" in str(e), str(e)
    ok &= good
    print(f"    {'OK ' if good else 'BAD'} real perf_counter wall-clock cap "
          f"breaches at the first lasso poll: {msg[:60]}")
    return ok


def scoring_equivalence_selftest() -> bool:
    """The arms module scores groups with its own tensor loop (own R2 once per
    target, codes cached per group). That loop must reproduce, to 1e-9, the
    reference path Stage A validated: PS.score_target_against_group on numpy.
    PCA arm (no training) for every (target, group); the learned arm with the
    same seeds for every pair on a small system."""
    x = PS.family1_generate(14, 1200, seed=9900)["x_obs"]
    prep = prep_system(x)
    labels = clustered_labels(prep)
    k = PS.k_for(14)
    groups = {g: sorted(np.where(labels == g)[0].tolist())
              for g in sorted(set(labels.tolist()))}
    worst = {}
    for kind in ("pca", "learned"):
        _, _, diag = screen_groups(prep, labels, kind, k, 0)
        codes = {}
        for gid, members in groups.items():
            z = np.concatenate([prep["own_raw"][j] for j in members],
                               axis=1).astype(np.float32)
            if kind == "pca":
                codes[gid] = _pca_codes(z, prep["tr"], members,
                                        PS.group_bottleneck(len(members)))
            else:
                net = PS.train_group_encoder(z, prep["tr"], len(members),
                                             seed=0 + gid)
                codes[gid] = (PS.group_code(net, z, None),
                              [PS.group_code(net, z, p)
                               for p in range(len(members))])
        w = 0.0
        for q in range(prep["V"]):
            for gid, members in groups.items():
                if all(j == q for j in members):
                    continue
                full, excl = codes[gid]
                c = excl[members.index(q)] if q in members else full
                ref, _, _ = PS.score_target_against_group(
                    poly3(prep["own_raw"][q]), prep["target"][q], c,
                    prep["tr"], prep["va"])
                w = max(w, abs(ref - diag["gains"][q][gid]))
        worst[kind] = w
    ok = all(v < 1e-9 for v in worst.values())
    print(f"    max |arms gain - reference gain|: pca {worst['pca']:.2e}, "
          f"learned {worst['learned']:.2e}   -> {'PASS' if ok else 'FAIL'}")
    return ok


def lasso_leak_selftest() -> bool:
    """Perturb ONLY the inner-validation labels of one target's train block.
    Every FITTED tuning quantity (residualiser weights, centring, scale,
    alpha_max) must be bit-identical -- inner-val labels never enter a fit.
    A deliberately leaky reference (statistics from ALL outer-train rows)
    must change under the same perturbation, proving the test can fail."""
    out = PS.family1_generate(14, 1200, seed=9900)
    prep = prep_system(out["x_obs"])
    D = _dictionary(prep)
    q = 3
    _, sa = lasso_target(prep, D, q)
    itr, iva = PS.internal_val_split(len(prep["tr"]))
    rows = prep["tr"][iva]
    prep2 = dict(prep)
    t2 = prep["target_t"].clone()
    t2[q, rows] += torch.as_tensor(
        np.random.default_rng(1).standard_normal(len(rows)) * 50.0,
        dtype=torch.float64, device=DEV)
    prep2["target_t"] = t2
    _, sb = lasso_target(prep2, D, q)
    same = (sa["mu"] == sb["mu"] and sa["sd"] == sb["sd"]
            and sa["a_max_fit"] == sb["a_max_fit"]
            and np.array_equal(sa["w_fit"], sb["w_fit"]))
    tr_t = prep["tr_t"]

    def leaky_stats(p):
        F, y = p["feats_t"][q][tr_t], p["target_t"][q][tr_t]
        r = (y - PS.ridge_predict(F, _ridge_w(F, y))).cpu().numpy()
        return float(r.mean()), float(r.std())
    leaky_differs = leaky_stats(prep) != leaky_stats(prep2)
    print(f"    fit quantities identical after perturbing inner-val labels "
          f"only: {same}")
    print(f"    leaky all-outer-train reference DOES change (test can fail): "
          f"{leaky_differs}")
    print(f"    -> {'PASS' if same and leaky_differs else 'FAIL'}")
    return same and leaky_differs


def check_invariance(seed: int = 9700, V: int = 16, n: int = 1200):
    """FINAL production arm paths, every arm: perturb the last raw rows (test
    block) and require identical partitions, shortlists, per-target gains,
    alpha choices and deployed-variant choice. A control run on the SAME input
    must first reproduce itself exactly, otherwise nondeterminism would be
    indistinguishable from leakage. Gains are compared, not only sets:
    an abstained target's set is 'all others' either way and would hide a
    perturbation."""
    x = PS.family1_generate(V, n, seed)["x_obs"]
    xp = x.copy()
    rows = np.arange(len(x) - 60, len(x))
    xp[rows] += np.random.default_rng(5).standard_normal(
        (len(rows), x.shape[1])) * 10.0
    xs_ = x.copy()                                   # SENSITIVITY control: the
    tr_rows = np.arange(150, 210)                    # same perturbation inside
    xs_[tr_rows] += np.random.default_rng(5).standard_normal(   # the TRAIN span
        (len(tr_rows), x.shape[1])) * 10.0
    r0, _, l0 = run_all_arms(x, seed)
    r1, _, l1 = run_all_arms(x, seed)
    r2, _, l2 = run_all_arms(xp, seed)
    r3, _, l3 = run_all_arms(xs_, seed)

    def compare(a, la, b, lb):
        res = {"partition labels": bool(np.array_equal(la, lb))}
        for arm in ARMS:
            res[f"{arm} shortlists"] = bool(a[arm][0] == b[arm][0])
        for arm in GROUP_ARMS:
            res[f"{arm} per-target gains"] = bool(
                a[arm][2]["gains"] == b[arm][2]["gains"])
            res[f"{arm} alpha boundary rate"] = bool(
                a[arm][2]["alpha_boundary_rate"]
                == b[arm][2]["alpha_boundary_rate"])
        res["LAGCORR deployed + proxy"] = bool(
            a["LAGCORR"][2]["deployed"] == b["LAGCORR"][2]["deployed"]
            and a["LAGCORR"][2]["proxy"] == b["LAGCORR"][2]["proxy"])
        res["LASSO chosen alphas"] = bool(
            a["LASSO"][2]["alpha_rel"] == b["LASSO"][2]["alpha_rel"])
        return res

    return (compare(r0, l0, r1, l1), compare(r0, l0, r2, l2),
            compare(r0, l0, r3, l3))


def preflight() -> bool:
    _take_lock("parent_screening arms preflight (V<=16, small encoders)")
    try:
        return _preflight()
    finally:
        LOCK.unlink(missing_ok=True)


def _preflight() -> bool:
    ok = True
    fr = free_mb()
    print(f"[env] available RAM {fr:.0f} MB vs the registered floor "
          f"{CAP_FREE_MB} MB -> "
          f"{'ok' if fr >= CAP_FREE_MB else 'BELOW FLOOR: a guarded pilot would breach at its first poll (informational, not a code failure)'}")
    print("[a] gate refuses incomplete/fallback input (incl. both repro cases)")
    ok &= gate_selftest()
    print("[b] resource guard measures the process TREE, not a docstring")
    ok &= tree_rss_selftest()
    print("[c] guard breaches INSIDE every long loop, not after the arm")
    ok &= guard_granularity_selftest()
    print("[d] lasso tuning never fits on inner-validation labels")
    ok &= lasso_leak_selftest()
    print("[e] scoring fast path == the reviewed reference scoring path")
    ok &= scoring_equivalence_selftest()
    print("[f] per-arm test-block invariance through the FINAL production "
          "arm paths (same-input determinism control; train-span "
          "sensitivity control proves the check can fail)")
    control, perturbed, sens = check_invariance()
    for name in control:
        print(f"    {name:36s} control {'identical' if control[name] else 'DIFFERS':9s}"
              f" | test-block perturbed {'identical' if perturbed[name] else 'CHANGED':9s}"
              f" | train-span perturbed {'identical' if sens[name] else 'changed'}")
    n_sens = sum(not v for v in sens.values())
    print(f"    sensitivity: a TRAIN-span perturbation changes {n_sens} of "
          f"{len(sens)} comparisons (must be > 0)")
    ok &= all(control.values()) and all(perturbed.values()) and n_sens > 0
    print(f"\nPREFLIGHT {'PASSED' if ok else 'FAILED'}")
    return ok


# ================================================================
# Engineering diagnostic: the frozen abstention rule's scale (NOT science)
# ================================================================

def engineering_gain_diag() -> Path:
    """Records, on engineering seeds excluded from every scientific result,
    the distribution of per-target MAX group gain for the learned clustered
    arm against the registered 0.01 abstention threshold. Encoder seed = group
    id. This is disclosure, not tuning: the threshold is not changed."""
    OUT.mkdir(parents=True, exist_ok=True)
    V, n = 24, 1200
    rec = dict(purpose="scale of max group gain vs the frozen 0.01 unresolved "
                       "rule; engineering seeds, excluded from every result",
               tau=PS.UNRESOLVED_GAIN, V=V, n=n, encoder_seed="group id",
               families={})
    for fam, gen, seed in (("family1", PS.family1_generate, 9800),
                           ("family2", PS.family2_generate, 9801)):
        out = gen(V, n, seed)
        prep = prep_system(out["x_obs"])
        labels = clustered_labels(prep)
        k = PS.k_for(V)
        _, unres, diag = screen_groups(prep, labels, "learned", k, 0)
        parent = out["parent"]
        nonroot = [q for q in parent if parent[q]]
        roots = [q for q in parent if not parent[q]]
        stat = lambda qs: dict(  # noqa: E731
            n=len(qs),
            median=float(np.median([diag["max_gain"][q] for q in qs])),
            p90=float(np.quantile([diag["max_gain"][q] for q in qs], 0.9)),
            max=float(max(diag["max_gain"][q] for q in qs)),
            frac_above_tau=float(np.mean(
                [diag["max_gain"][q] > PS.UNRESOLVED_GAIN for q in qs])))
        rec["families"][fam] = dict(
            seed=seed, nonroot=stat(nonroot), root=stat(roots),
            unresolved_fraction_nonroot=float(np.mean(
                [unres[q] for q in nonroot])),
            per_target=[dict(q=q, root=not parent[q],
                             max_gain=diag["max_gain"][q]) for q in parent])
    path = OUT / "engineering_gain_scale.json"
    path.write_text(json.dumps(rec, indent=1))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"wrote {path}  sha256={digest}")
    for fam, r in rec["families"].items():
        print(f"  {fam} seed {r['seed']}: non-root median "
              f"{r['nonroot']['median']:+.4f} p90 {r['nonroot']['p90']:+.4f} "
              f"max {r['nonroot']['max']:+.4f} frac>tau "
              f"{r['nonroot']['frac_above_tau']:.2f} | root median "
              f"{r['root']['median']:+.4f} | unresolved non-root "
              f"{r['unresolved_fraction_nonroot']:.2f}")
    return path


# ================================================================
# Bounded engineering timing (NOT a hidden pilot)
# ================================================================

def _take_lock(what: str):
    if LOCK.exists():
        raise SystemExit(f".agent-lock exists ({LOCK.read_text().strip()!r}); "
                         f"not starting")
    LOCK.write_text(f"parent_screening | {what} | "
                    f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")


def timing_run(system_cap_sec: float = 600, lasso_subset: int = 12) -> dict:
    """One V=240 engineering system per family (seeds 9700/9701), each capped
    at system_cap_sec by the guard. LASSO runs on `lasso_subset` targets and
    is extrapolated linearly (per-target cost is uniform); every other arm
    runs in full. Records actual elapsed and a conservative extrapolation to
    the 12-system pilot against the 1h cap."""
    OUT.mkdir(parents=True, exist_ok=True)
    _take_lock("parent_screening engineering timing (2 systems, V=240)")
    rec = {}
    try:
        for fam, gen, seed in (("family1", PS.family1_generate, 9700),
                               ("family2", PS.family2_generate, 9701)):
            guard = ResourceGuard(system_cap_sec)
            t0 = time.perf_counter()
            out = gen(240, 4000, seed)
            t_gen = time.perf_counter() - t0
            timings, err = {}, None
            try:
                run_all_arms(out["x_obs"], seed, guard, timings,
                             lasso_targets=lasso_subset)
            except CapBreach as e:
                err = str(e)
            lasso_full = timings.get("LASSO", 0.0) * 240 / lasso_subset
            per_sys = t_gen + lasso_full + sum(
                v for kx, v in timings.items() if kx != "LASSO")
            rec[fam] = dict(gen_sec=t_gen, timings=timings,
                            lasso_full_extrapolated_sec=lasso_full,
                            breach=err, per_system_sec=per_sys,
                            **guard.snapshot())
        pilot_total = 6 * sum(r["per_system_sec"] for r in rec.values())
        rec["pilot_extrapolated_sec"] = pilot_total
        rec["pilot_cap_sec"] = CAP_RUNTIME_SEC
        rec["fits_cap"] = bool(pilot_total <= CAP_RUNTIME_SEC and not any(
            r["breach"] for r in rec.values() if isinstance(r, dict)
            and "breach" in r))
        (OUT / "engineering_timing.json").write_text(json.dumps(
            rec, indent=1, default=float))
        return rec
    finally:
        LOCK.unlink(missing_ok=True)


# ================================================================
# The pilot runner: clearance-bound, resumable, truth opened last
# ================================================================

def _src_hash() -> str:
    h = hashlib.sha256()
    for f in ("parent_screening.py", "parent_screening_arms.py"):
        h.update((HERE / f).read_bytes())
    return h.hexdigest()[:16]


def config_hash() -> str:
    cfg = dict(seeds=FROZEN_SEEDS, arms=ARMS, alpha_grid=PS.ALPHA_GRID,
               lasso_grid=LASSO_GRID_REL, lags=LAGS, group_cap=PS.GROUP_CAP,
               E=PS.E, unresolved=PS.UNRESOLVED_GAIN, V=240, n=4000,
               src=_src_hash())
    return hashlib.sha256(json.dumps(cfg, sort_keys=True,
                                     default=str).encode()).hexdigest()[:16]


def clearance_state() -> tuple[bool, str]:
    """The pilot may start only if the reviewer's clearance file reads
    'cleared <hash>' with <hash> a prefix of the CURRENT git HEAD, and every
    guarded script/protocol is committed (so the code that runs is the code
    that was reviewed)."""
    if not CLEARANCE.exists():
        return False, f"{CLEARANCE} does not exist"
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    toks = CLEARANCE.read_text().split()
    if len(toks) < 2 or toks[0] != "cleared" or len(toks[1]) < 7 \
            or not head.startswith(toks[1]):
        return False, (f"clearance file must read 'cleared <commit-hash>' "
                       f"matching HEAD {head[:12]}")
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *GUARDED_FILES], capture_output=True,
                           text=True).stdout.strip()
    if dirty:
        return False, f"guarded files have uncommitted changes:\n{dirty}"
    return True, head


def _payload(res, V):
    """Compact JSON of one system's shortlists: an unresolved target is stored
    as null (meaning 'all V-1 others'), never expanded on disk."""
    pl = {}
    for a in ARMS:
        C, unres, diag = res[a]
        pl[a] = dict(
            C={str(q): (None if unres.get(q, False) else c)
               for q, c in C.items()},
            unresolved=[q for q, u in unres.items() if u])
        if a in GROUP_ARMS:
            pl[a]["max_gain"] = {str(q): g for q, g in diag["max_gain"].items()}
            pl[a]["diag"] = {kx: diag[kx] for kx in
                             ("n_groups", "sizes", "alpha_boundary_rate")}
        if a == "LAGCORR":
            pl[a]["diag"] = dict(proxy=diag["proxy"], deployed=diag["deployed"])
        if a == "LASSO":
            pl[a]["diag"] = dict(nonconverged_fits=diag["nonconverged_fits"])
    return pl


def _restore(payload_arm, V):
    C, unres = {}, {}
    for q_s, c in payload_arm["C"].items():
        q = int(q_s)
        unres[q] = c is None
        C[q] = sorted(j for j in range(V) if j != q) if c is None else c
    return C, unres


def run_pilot():
    ok, info = clearance_state()
    if not ok:
        raise SystemExit("pilot NOT cleared: " + info)
    OUT.mkdir(parents=True, exist_ok=True)
    ch = config_hash()
    spent = 0.0
    for f in OUT.glob("done_*.json"):
        d = json.loads(f.read_text())
        if d.get("config") == ch:
            spent += d.get("system_sec", 0.0)
    _take_lock("parent_screening Stage B pilot (12 systems, V=240)")
    guard = ResourceGuard(CAP_RUNTIME_SEC - spent)
    try:
        for fam, gen in (("family1", PS.family1_generate),
                         ("family2", PS.family2_generate)):
            for seed in FROZEN_SEEDS[fam]:
                done = OUT / f"done_{fam}_{seed}.json"
                if done.exists() and json.loads(done.read_text()
                                                ).get("config") == ch:
                    print(f"  {fam} {seed}: RESUMED", flush=True)
                    continue
                guard.check_light(f"generate {fam} {seed}")
                t_sys = time.perf_counter()
                out = gen(240, 4000, seed)
                timings = {}
                res, prep, labels = run_all_arms(out["x_obs"], seed, guard,
                                                 timings)
                (OUT / f"shortlists_{fam}_{seed}.json").write_text(
                    json.dumps(_payload(res, 240)))
                # truth is written only AFTER this system's shortlists are on disk
                (OUT / f"evaluator_{fam}_{seed}.json").write_text(json.dumps(
                    {str(q): out["parent"][q] for q in out["parent"]}))
                sys_sec = time.perf_counter() - t_sys
                done.write_text(json.dumps(dict(
                    config=ch, head=info, family=fam, seed=seed,
                    system_sec=sys_sec, timings=timings, **guard.snapshot()),
                    default=float))
                print(f"  {fam} {seed}: shortlists frozen, {sys_sec:.0f}s "
                      f"(run total {guard.elapsed():.0f}s)", flush=True)
        per_family = {fam: {a: [] for a in ARMS} for fam in FAMILIES}
        k = PS.k_for(240)
        for fam in FAMILIES:
            for seed in FROZEN_SEEDS[fam]:
                truth = {int(q): [tuple(e) for e in v] for q, v in json.loads(
                    (OUT / f"evaluator_{fam}_{seed}.json").read_text()).items()}
                pl = json.loads((OUT / f"shortlists_{fam}_{seed}.json"
                                 ).read_text())
                for a in ARMS:
                    C, unres = _restore(pl[a], 240)
                    m = arm_metrics(C, unres, truth, 240, k)
                    m["seed"] = seed
                    per_family[fam][a].append(m)
        (OUT / "pilot_metrics.json").write_text(json.dumps(
            per_family, indent=1, default=lambda o: bool(o)
            if isinstance(o, np.bool_) else float(o)))
        print("\nSTAGE B GATE")
        passed = evaluate_gate(per_family)
        print("\nDESCRIPTIVE (not gated): mean over seeds")
        for fam in FAMILIES:
            for a in ARMS:
                rows = per_family[fam][a]
                rr = [r["recall_resolved_only"] for r in rows
                      if r["recall_resolved_only"] is not None]
                print(f"  {fam} {a:18s} recall {np.mean([r['recall'] for r in rows]):.3f}"
                      f"  coverage {np.mean([r['coverage'] for r in rows]):.3f}"
                      f"  cand_frac {np.mean([r['candidate_fraction'] for r in rows]):.3f}"
                      f"  unresolved {np.mean([r['unresolved_fraction'] for r in rows]):.3f}"
                      f"  resolved-only recall "
                      f"{(np.mean(rr) if rr else float('nan')):.3f}"
                      f"  budget_ok {sum(bool(r['budget_ok']) for r in rows)}/6")
        return passed
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    if "--preflight" in sys.argv:
        raise SystemExit(0 if preflight() else 1)
    if "--engineering-diag" in sys.argv:
        _take_lock("parent_screening engineering gain diagnostic (V=24)")
        try:
            engineering_gain_diag()
        finally:
            LOCK.unlink(missing_ok=True)
        raise SystemExit(0)
    if "--timing" in sys.argv:
        r = timing_run()
        print(json.dumps(r, indent=1, default=float))
        raise SystemExit(0)
    if "--run-pilot" in sys.argv:
        raise SystemExit(0 if run_pilot() else 2)
    print(__doc__)
