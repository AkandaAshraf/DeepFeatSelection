"""Confirmatory replication of P-S1 on the 10 held-out ds003876 subjects.

Pre-registration: paper/ieeg_replication_protocol.md, committed at 93a95be
before any held-out recording was downloaded.

This is a thin wrapper over the discovery analysis rather than a copy: the
pipeline, gate and tests are imported unchanged from ieeg_soz_confirmatory,
so the replication cannot silently diverge from the study it replicates.
Only the cohort, the file naming and the output directory differ.

    python scripts/ieeg_replication.py --stage scan
    python scripts/ieeg_replication.py --stage test

Labels are opened only in the test stage, after every gate verdict exists.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import ieeg_soz_confirmatory as C  # noqa: E402
from ieeg_gate import DATA, DEV, arm, good_channels, norm_label, read_edf  # noqa: E402

# Cohort fixed in the pre-registration; no subject may be added or removed.
HELDOUT = ["jh103", "jh105", "pt1", "pt2", "pt3",
           "umf001", "umf002", "umf003", "umf004", "umf005"]
TASK = "interictalawake"          # the direct match to the discovery task
OUT = Path("ExpOutput/ieeg_replication")
ARMS = ("bipolar", "laplacian", "bipolar_skip")   # raw excluded by criterion


def edf_path(sub: str) -> Path:
    return DATA / f"sub-{sub}_{TASK}_run-01_ieeg.edf"


def scan() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"cohort: {len(HELDOUT)} held-out subjects, task={TASK}\n"
          f"device: {DEV}\narms: {ARMS}")
    for sub in HELDOUT:
        dest = OUT / f"excess_{sub}.csv"
        todo = list(ARMS)
        if dest.exists():
            have = set(pd.read_csv(dest).arm.unique())
            todo = [a for a in ARMS if a not in have]
            if not todo:
                print(f"[{sub}] complete, skipping")
                continue
        p = edf_path(sub)
        if not p.exists():
            print(f"[{sub}] MISSING {p.name} - excluded and reported")
            continue
        t0 = time.time()
        try:
            x, labels, fs = read_edf(p)
            idx = good_channels(sub, labels)
            x = x[:, idx]
            labels_n = [norm_label(labels[i]) for i in idx]
            rows = []
            for mode in todo:
                xp, names, _ = C.preprocess_labelled(x, labels_n, fs, mode)
                if xp.shape[1] < 8:
                    print(f"[{sub}] {mode}: {xp.shape[1]} channels, skipped")
                    continue
                r = arm(xp, mode, return_channels=True)
                for nm, ex, sr in zip(names, r["excess"], r["self_r2"]):
                    rows.append({"sub": sub, "arm": mode, "channel": nm,
                                 "excess": ex, "self_r2": sr,
                                 "threshold": max(0.0, r["g_max"]),
                                 "g_med": r["g_med"], "self_f90": r["self_f90"],
                                 "donor_fallback": r["donor_fallback"]})
            if not rows:
                print(f"[{sub}] no usable arm")
                continue
            new = pd.DataFrame(rows)
            if dest.exists():
                new = pd.concat([pd.read_csv(dest), new], ignore_index=True)
            new.to_csv(dest, index=False)
            print(f"[{sub}] {len(rows)} rows, {x.shape[1]} channels, "
                  f"fs={fs:.0f} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as exc:
            print(f"[{sub}] FAILED: {str(exc)[:110]}", flush=True)
    return 0


def gate_report(rows: pd.DataFrame) -> set[str]:
    """G3 stationarity per subject on the primary montage. No label read."""
    print("\n" + "=" * 74)
    print("GATE (G3 ghost-panel median <= 0.005 on the primary montage)")
    keep = set()
    for sub, d in rows.groupby("sub"):
        bp = d[d.arm == "bipolar"]
        if bp.empty:
            print(f"  {sub:9s} no primary montage -> EXCLUDED")
            continue
        g = float(bp.g_med.iloc[0])
        ok = g <= 0.005
        print(f"  {sub:9s} ghost median {g:+.4f}   self-R2>0.9 "
              f"{float(bp.self_f90.iloc[0]):.2f}   "
              f"{'PASS' if ok else 'FAIL -> EXCLUDED'}")
        if ok:
            keep.add(sub)
    print(f"  -> {len(keep)}/{rows['sub'].nunique()} subjects pass the gate")
    return keep


def test() -> int:
    files = sorted(OUT.glob("excess_*.csv"))
    if not files:
        print("no scans; run --stage scan first")
        return 1
    rows = pd.concat([pd.read_csv(f) for f in files])
    keep = gate_report(rows)
    rows = rows[rows["sub"].isin(keep)]

    print("\n" + "=" * 74)
    print("VALIDITY GATE (median Spearman with bipolar > 0.30)")
    from ieeg_validity_gate import gate_for
    for a in ("laplacian", "bipolar_skip"):
        vals = [r for _, d in rows.groupby("sub")
                if (r := gate_for(d, a)) is not None]
        if vals:
            m = float(np.median(vals))
            print(f"  {a:14s} median rho {m:+.3f}  n={len(vals)}  "
                  f"{'PASS' if m > 0.30 else 'FAIL -> uninformative'}")

    for armname in ARMS:
        print("\n" + "=" * 74)
        tag = "[PRIMARY]" if armname == "bipolar" else "[same-scale]"
        print(f"ARM: {armname}   {tag}")
        s1, s2 = [], []
        for sub, d in rows[rows.arm == armname].groupby("sub"):
            d = C.annotate(d, sub)
            if (r1 := C.per_subject_ps1(d)):
                r1["sub"] = sub
                s1.append(r1)
            if (r2 := C.per_subject_ps2(d)):
                r2["sub"] = sub
                s2.append(r2)
        t1 = pd.DataFrame(s1)
        if t1.empty:
            print("  P-S1: no subject met the group-size minimum")
            continue
        t1.to_csv(OUT / f"ps1_{armname}.csv", index=False)
        w = np.sqrt(t1.n_soz + t1.n_non).to_numpy()
        z, p = C.stouffer(t1.p.to_numpy(), w)
        pos = int((t1.rank_biserial > 0).sum())
        from scipy import stats
        sp = stats.binomtest(pos, len(t1), 0.5, alternative="greater").pvalue
        print(f"  P-S1  subjects {len(t1)}   Stouffer z = {z:+.3f}   "
              f"p = {p:.4g}")
        print(f"        direction {pos}/{len(t1)}  sign p = {sp:.4g} "
              f"(declared underpowered at n=10)")
        print(f"        median rank-biserial {t1.rank_biserial.median():+.3f}")
        print(f"        driven core: SOZ {t1.frac_soz_core.mean():.3f} vs "
              f"non-SOZ {t1.frac_non_core.mean():.3f}  "
              f"(gap {100*(t1.frac_non_core.mean()-t1.frac_soz_core.mean()):+.1f} pts)")
        t2 = pd.DataFrame(s2)
        if not t2.empty:
            z2, p2 = C.stouffer(t2.p.to_numpy(),
                                np.sqrt(t2.n_adj + t2.n_dist).to_numpy())
            print(f"  P-S2  (predicted NEGATIVE) z = {z2:+.3f}  p = {p2:.4g}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["scan", "test"], required=True)
    a = ap.parse_args()
    raise SystemExit(scan() if a.stage == "scan" else test())
