"""CPU regressions for the hierarchy-repair resume and lock guards.

Covers the cases named in review: malformed bundle, corrupted NPZ, wrong
cell, failed guard, duplicate arm, foreign lock, and all-complete summary
rebuild. No torch and no GPU.

    python scripts/test_repair_guards.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import hierarchy_repair as H  # noqa: E402

TMP = Path("ExpOutput/_guard_tests")
OK, BAD = [], []


def chk(label, cond):
    (OK if cond else BAD).append(label)
    print(f"   {'PASS' if cond else 'FAIL'}  {label}")


def reset():
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    H.OUT = TMP
    H.LOCK = str(TMP / "lock")
    H._LOCK_TOKEN = None


def good_cell(V=30, nz=0.0, s=0, n=30, arms=None, guard=None, npz=True):
    arms = arms or H.ARMS
    rows = [dict(V=V, noise=nz, seed=s, arm=a, ap_source=0.5) for a in arms]
    g = guard or dict(V=V, noise=nz, seed=s, archive_found=True, ap_diff=0.0,
                      arr_max_abs_diff=0.0, ap_ok=True, arr_ok=True,
                      reason="")
    extra = dict(guard=g, sizes=[dict(arm=a, unweighted_mean=6.0)
                                 for a in H.ARMS[1:]],
                 graded=[dict(arm="HIER-TRUE", module=0)], d2_ari=0.7)
    if npz:
        payload = {"FLAT": np.zeros(n), "is_driven": np.ones(n, bool),
                   "is_source": np.zeros(n, bool),
                   "parent": np.zeros(n - 5, int), "raw_cutoff": 100}
        for a in H.ARMS[1:]:
            payload[f"{a}_lab"] = np.zeros(n, int)
        np.savez_compressed(TMP / f"raw_V{V}_nz{nz}_s{s}.npz", **payload)
    H.save_bundle(V, nz, s, rows, extra)


def main() -> int:
    print("Hierarchy-repair guard regressions (CPU only)\n")

    print("BUNDLE VALIDATION")
    reset(); good_cell()
    chk("valid bundle accepted", H.load_bundle(30, 0.0, 0) is not None)

    reset(); good_cell()
    H.bundle_path(30, 0.0, 0).write_text("{ not json")
    chk("malformed bundle rejected", H.load_bundle(30, 0.0, 0) is None)
    chk("malformed bundle quarantined",
        any((TMP / "quarantine").glob("cell_*")))

    reset(); good_cell()
    (TMP / "raw_V30_nz0.0_s0.npz").write_bytes(b"not an npz at all")
    chk("corrupted NPZ rejected", H.load_bundle(30, 0.0, 0) is None)

    reset(); good_cell()
    b = json.loads(H.bundle_path(30, 0.0, 0).read_text())
    b["V"] = 60
    H.bundle_path(30, 0.0, 0).write_text(json.dumps(b))
    chk("wrong-cell bundle rejected", H.load_bundle(30, 0.0, 0) is None)

    reset()
    good_cell(guard=dict(V=30, noise=0.0, seed=0, archive_found=True,
                         ap_diff=0.9, arr_max_abs_diff=0.9, ap_ok=False,
                         arr_ok=True, reason="AP mismatch"))
    chk("failed-guard bundle rejected", H.load_bundle(30, 0.0, 0) is None)

    reset()
    good_cell(guard=dict(V=30, noise=0.0, seed=0, archive_found=True,
                         ap_diff=float("nan"), arr_max_abs_diff=0.0,
                         ap_ok=True, arr_ok=True, reason=""))
    chk("nonfinite-guard bundle rejected", H.load_bundle(30, 0.0, 0) is None)

    reset(); good_cell(arms=H.ARMS + ["FLAT"])
    chk("duplicate-arm bundle rejected", H.load_bundle(30, 0.0, 0) is None)

    reset(); good_cell(arms=H.ARMS[:-1])
    chk("missing-arm bundle rejected", H.load_bundle(30, 0.0, 0) is None)

    reset(); good_cell(npz=False)
    chk("bundle without NPZ rejected", H.load_bundle(30, 0.0, 0) is None)

    reset(); good_cell()
    with np.load(TMP / "raw_V30_nz0.0_s0.npz") as z:
        d = {k: z[k] for k in z.files}
    d["HIER-TRUE_lab"] = np.zeros(7, int)          # wrong length
    np.savez_compressed(TMP / "raw_V30_nz0.0_s0.npz", **d)
    chk("NPZ shape mismatch rejected", H.load_bundle(30, 0.0, 0) is None)

    print("\nLOCK")
    reset()
    chk("acquire succeeds on a free lock", H.acquire_lock() is True)
    chk("owner re-acquire succeeds", H.acquire_lock() is True)
    H.release_lock()
    chk("release removes our own lock", not Path(H.LOCK).exists())

    reset()
    child = subprocess.Popen([sys.executable, "-c", "import time;"
                              "time.sleep(30)"])
    Path(H.LOCK).write_text(json.dumps(
        {"pid": child.pid, "token": "other", "what": "foreign",
         "started": time.time()}))
    chk("live foreign lock refused", H.acquire_lock() is False)
    chk("live foreign lock NOT removed", Path(H.LOCK).exists())
    child.terminate(); child.wait(); time.sleep(0.3)
    chk("dead foreign lock ALSO refused (no auto-unlink race)",
        H.acquire_lock() is False)
    chk("dead foreign lock still present", Path(H.LOCK).exists())

    Path(H.LOCK).write_text("{ corrupt")
    chk("unreadable lock refused", H.acquire_lock() is False)

    reset()
    H.acquire_lock()
    Path(H.LOCK).write_text(json.dumps({"pid": os.getpid(),
                                        "token": "someone-else"}))
    H.release_lock()
    chk("release refuses when the token is not ours",
        Path(H.LOCK).exists())

    reset()
    saved_run = H._run
    H._run = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    H.acquire_lock()
    try:
        H.main()
    except RuntimeError:
        pass
    chk("lock released after an exception", not Path(H.LOCK).exists())
    H._run = saved_run

    print("\nALL-COMPLETE RESUME")
    reset()
    H.WIDTHS, H.NOISES, H.SEEDS = (30,), (0.0,), (0, 1)
    H.run_invariance_precheck = lambda: True
    calls = {"n": 0}

    def stub(V, nz, s):
        calls["n"] += 1
        rows = [dict(V=V, noise=nz, seed=s, arm=a, ap_source=0.5,
                     mod_share=0.6 if a != "FLAT" else float("nan"),
                     share_excluded=False, mean_e2_driven=0.01,
                     mean_e3_driven=0.01, ap_driven=0.9, secs=1.0)
                for a in H.ARMS]
        g = dict(V=V, noise=nz, seed=s, archive_found=True, ap_diff=0.0,
                 arr_max_abs_diff=0.0, ap_ok=True, arr_ok=True, reason="")
        extra = dict(guard=g,
                     sizes=[dict(V=V, noise=nz, seed=s, arm=a,
                                 unweighted_mean=6.0, target_weighted=7.0,
                                 driven_weighted=7.0, n_modules=5,
                                 min_width=4, max_width=12)
                            for a in H.ARMS[1:]],
                     graded=[dict(V=V, noise=nz, seed=s, arm=a, module=i,
                                  n_driven=3, frac_parent_in=0.2 * i,
                                  share=0.1 * i, mod_e2=0.01, mod_e3=0.01,
                                  mod_denom=0.02, excluded=False)
                             for a in H.ARMS[1:] for i in range(4)],
                     d2_ari=0.7)
        n = 30
        payload = {"FLAT": np.zeros(n), "is_driven": np.ones(n, bool),
                   "is_source": np.zeros(n, bool),
                   "parent": np.zeros(n - 5, int), "raw_cutoff": 100}
        for a in H.ARMS[1:]:
            payload[f"{a}_lab"] = np.zeros(n, int)
        np.savez_compressed(TMP / f"raw_V{V}_nz{nz}_s{s}.npz", **payload)
        return rows, extra

    H.cell = stub
    H.main()
    first = calls["n"]
    (TMP / "flat_guard_summary.csv").unlink(missing_ok=True)
    rc = H.main()
    chk("all-complete resume retrains nothing", calls["n"] == first)
    chk("all-complete resume returns 0", rc == 0)
    chk("all-complete resume rebuilds flat_guard_summary.csv",
        (TMP / "flat_guard_summary.csv").exists())
    if (TMP / "flat_guard_summary.csv").exists():
        chk("summary covers every completed cell",
            len(pd.read_csv(TMP / "flat_guard_summary.csv")) == first)
    chk("cells.csv covers every completed cell",
        len(pd.read_csv(TMP / "cells.csv")) == first * len(H.ARMS))

    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{len(OK)} passed, {len(BAD)} failed")
    for b in BAD:
        print("  FAILED:", b)
    return 1 if BAD else 0


if __name__ == "__main__":
    raise SystemExit(main())
