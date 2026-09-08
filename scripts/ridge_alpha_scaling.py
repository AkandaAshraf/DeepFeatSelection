"""Does selecting alpha on validation recover a planted signal at the real
pipeline's own (n_train, p) shapes?

Pre-registration: paper/ridge_alpha_scaling_protocol.md, committed before
this was written. CPU-only, no encoder, no dataset touched. Tests a
CANDIDATE mechanism the 2026-09-08 diagnosis found on synthetic noise --
this run does not, by itself, say anything about the real learned system
code (see the protocol's own repeated hedge on this point).

    python scripts/ridge_alpha_scaling.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import boundary_map as BM  # noqa: E402  -- for the equivalence check only
# Reuse large_system.py's own tested Windows RSS helper rather than a new
# ctypes struct -- same repo, already correct, no reason to duplicate it.
# Pure-definition import: no directory created, no GPU touched, no dataset
# read at import time (checked against its own top-level code).
import large_system as _LS  # noqa: E402

OUT = Path("ExpOutput/ridge_alpha_scaling")
DEV = "cpu"                      # forced, per protocol -- no GPU work at all

S_SIGNAL = 0.15
CEILING_ANALYTIC = S_SIGNAL ** 2          # 0.0225, derived not measured
N_TRAIN, N_VAL, N_TEST = 2397, 799, 799
P_GRID = (20, 100, 500, 1000, 2000)
SEEDS = (0, 1, 2, 3, 4)
VAL_GRID = (0.1, 1, 10, 100, 1000, 10000, 100000)

CAP_RUNTIME_SEC = 300
CAP_RSS_MB = 500


def rss_mb() -> float:
    return _LS.host_rss_mb()


class CapBreach(RuntimeError):
    pass


def ridge_r2_alpha(Xtr, ytr, Xte, yte, alpha: float) -> float:
    """Identical formula to boundary_map.ridge_r2, alpha as a parameter
    instead of a module constant. Verified equivalent at alpha=1.0 by
    the pre-flight check below before any declared cell runs."""
    Xt = torch.as_tensor(Xtr, dtype=torch.float64, device=DEV)
    yt = torch.as_tensor(ytr, dtype=torch.float64, device=DEV)
    Xe = torch.as_tensor(Xte, dtype=torch.float64, device=DEV)
    ye = torch.as_tensor(yte, dtype=torch.float64, device=DEV)
    A = Xt.T @ Xt + alpha * torch.eye(Xt.shape[1], device=DEV,
                                      dtype=torch.float64)
    w = torch.linalg.solve(A, Xt.T @ yt)
    err = float(((Xe @ w - ye) ** 2).mean())
    return max(0.0, 1.0 - err / (float(ye.var()) + 1e-12))


def preflight() -> bool:
    """The local formula at alpha=1.0 must match boundary_map.ridge_r2's
    own output to within 1e-9, on one fixed synthetic draw. A FAIL stops
    the run before any declared cell is touched."""
    rng = np.random.default_rng(12345)
    n, p = 300, 15
    X = rng.standard_normal((n, p)).astype(np.float64)
    y = rng.standard_normal(n).astype(np.float64)
    a, b = 200, 250
    r_local = ridge_r2_alpha(X[:a], y[:a], X[b:], y[b:], alpha=1.0)
    r_real = BM.ridge_r2(X[:a], y[:a], X[b:], y[b:])
    ok = abs(r_local - r_real) < 1e-9
    print(f"PREFLIGHT  local(alpha=1.0)={r_local:.12f}  "
          f"boundary_map.ridge_r2={r_real:.12f}  "
          f"diff={abs(r_local - r_real):.2e}  -> {'PASS' if ok else 'FAIL'}")
    return ok


def make_draw(p: int, seed: int):
    """One synthetic draw: y = s*x0 + sqrt(1-s^2)*noise, x0 the one
    informative column, p-1 pure-noise columns, matched to N_TRAIN."""
    rng = np.random.default_rng(1000 * p + seed)
    n = N_TRAIN + N_VAL + N_TEST
    x0 = rng.standard_normal(n)
    noise_cols = rng.standard_normal((n, max(p - 1, 0)))
    y = (S_SIGNAL * x0
        + np.sqrt(1 - S_SIGNAL ** 2) * rng.standard_normal(n))
    X = (np.hstack([x0[:, None], noise_cols]) if p > 1
        else x0[:, None]).astype(np.float64)
    tr = slice(0, N_TRAIN)
    va = slice(N_TRAIN, N_TRAIN + N_VAL)
    te = slice(N_TRAIN + N_VAL, n)
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def alpha_val_select(Xtr, ytr, Xva, yva) -> float:
    """Lowest squared error on VALIDATION only, over VAL_GRID. Test is
    never touched by this function."""
    best_alpha, best_err = VAL_GRID[0], float("inf")
    for a in VAL_GRID:
        Xt = torch.as_tensor(Xtr, dtype=torch.float64, device=DEV)
        yt = torch.as_tensor(ytr, dtype=torch.float64, device=DEV)
        Xv = torch.as_tensor(Xva, dtype=torch.float64, device=DEV)
        yv = torch.as_tensor(yva, dtype=torch.float64, device=DEV)
        A = Xt.T @ Xt + a * torch.eye(Xt.shape[1], device=DEV,
                                      dtype=torch.float64)
        w = torch.linalg.solve(A, Xt.T @ yt)
        err = float(((Xv @ w - yv) ** 2).mean())
        if err < best_err:
            best_err, best_alpha = err, a
    return best_alpha


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEV} (forced)   analytic ceiling s^2 = "
          f"{CEILING_ANALYTIC:.4f}\n")

    if not preflight():
        print("\nPREFLIGHT FAILED -- stopping before any declared cell.")
        return 1

    rows = []

    def check_caps(where: str):
        el = time.time() - t0
        rss = rss_mb()
        if el > CAP_RUNTIME_SEC:
            raise CapBreach(f"runtime {el:.1f}s > {CAP_RUNTIME_SEC}s at {where}")
        if rss > CAP_RSS_MB:
            raise CapBreach(f"RSS {rss:.1f}MB > {CAP_RSS_MB}MB at {where}")

    # p=1 empirical control, per seed, descriptive only, not an arm
    for seed in SEEDS:
        Xtr, ytr, _, _, Xte, yte = make_draw(1, seed)
        r2 = ridge_r2_alpha(Xtr, ytr, Xte, yte, alpha=1.0)
        rows.append(dict(p=1, seed=seed, arm="P1-CONTROL", alpha_used=1.0,
                         r2=r2))
        check_caps(f"p=1 seed={seed}")

    for p in P_GRID:
        for seed in SEEDS:
            Xtr, ytr, Xva, yva, Xte, yte = make_draw(p, seed)

            r2 = ridge_r2_alpha(Xtr, ytr, Xte, yte, alpha=1.0)
            rows.append(dict(p=p, seed=seed, arm="ALPHA-FIXED",
                             alpha_used=1.0, r2=r2))

            r2 = ridge_r2_alpha(Xtr, ytr, Xte, yte, alpha=float(p))
            rows.append(dict(p=p, seed=seed, arm="ALPHA-SCALED",
                             alpha_used=float(p), r2=r2))

            a_sel = alpha_val_select(Xtr, ytr, Xva, yva)
            r2 = ridge_r2_alpha(Xtr, ytr, Xte, yte, alpha=a_sel)
            rows.append(dict(p=p, seed=seed, arm="ALPHA-VAL",
                             alpha_used=a_sel, r2=r2))

            check_caps(f"p={p} seed={seed}")
        print(f"  p={p:<5} done   ({(time.time()-t0):.1f}s, "
              f"rss {rss_mb():.0f}MB)", flush=True)

    d = pd.DataFrame(rows)
    csv_path = OUT / "cells.csv"
    d.to_csv(csv_path, index=False)
    size_kb = csv_path.stat().st_size / 1024
    print(f"\nwritten: {csv_path} ({size_kb:.1f} KB)")

    print("\nMEAN TEST R2 by p, arm (declared cells only, p=1 separate)")
    piv = d[d.p != 1].pivot_table(index="p", columns="arm", values="r2",
                                  aggfunc="mean")
    print(piv[["ALPHA-FIXED", "ALPHA-SCALED", "ALPHA-VAL"]].round(4)
          .to_string())

    ctrl = d[d.arm == "P1-CONTROL"]
    print(f"\np=1 empirical control: mean R2 {ctrl.r2.mean():.4f}  "
          f"(analytic ceiling {CEILING_ANALYTIC:.4f})")

    at2000 = d[(d.p == 2000)]
    fixed_p100 = d[(d.p == 100) & (d.arm == "ALPHA-FIXED")].r2.mean()
    val_2000 = at2000[at2000.arm == "ALPHA-VAL"].r2.mean()
    scaled_2000 = at2000[at2000.arm == "ALPHA-SCALED"].r2.mean()

    a4 = bool(fixed_p100 == 0.0)
    a1 = bool(val_2000 >= 0.5 * CEILING_ANALYTIC)
    a2 = bool(scaled_2000 > 0.0)

    print(f"\nA4 GUARD   ALPHA-FIXED mean R2 at p=100: {fixed_p100:.4f}"
          f"   -> {'HOLDS' if a4 else 'FAILS'}")
    print(f"A1 DECISIVE   ALPHA-VAL mean R2 at p=2000: {val_2000:.4f}"
          f"  vs bar {0.5*CEILING_ANALYTIC:.5f}"
          f"   -> {'HOLDS' if a1 else 'FAILS'}")
    print(f"A2   ALPHA-SCALED mean R2 at p=2000: {scaled_2000:.4f}"
          f"   -> {'HOLDS' if a2 else 'FAILS'}")

    print(f"\nTotal runtime: {(time.time()-t0):.1f}s  "
          f"(cap {CAP_RUNTIME_SEC}s)   peak-ish RSS {rss_mb():.0f}MB "
          f"(cap {CAP_RSS_MB}MB)")

    print("\nVERDICT (rule fixed before running; no adoption possible "
          "either way)")
    if not a4:
        print("   -> UNINFORMATIVE. The instrument, not the mechanism, "
              "is what was measured.")
    elif a1:
        print("   -> A CANDIDATE THAT SURVIVED THIS TEST. Worth a future,")
        print("      separately pre-registered test on the real system "
              "code. Nothing")
        print("      here is licensed beyond that pointer -- see the "
              "protocol's own text.")
    else:
        print("   -> A CANDIDATE THAT DID NOT SURVIVE. Selecting alpha "
              "properly does not")
        print("      recover the signal even on controlled synthetic "
              "noise at the real")
        print("      pipeline's own shape. Materially weakens this "
              "candidate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
