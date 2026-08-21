"""Confirmatory SOZ analysis: P-S1, P-S2, P-S3.

Runs the pre-registered tests in paper/ieeg_protocol.md, whose analysis
specification was committed before any soz, epz or rz column was read.

Stage 1 recomputes per-channel excess with channel labels for the 28
G3-clean bipolar subjects fixed by the gate (bipolar confirmatory, raw
sensitivity), because the gate stored summaries only. Stage 2 opens the
labels for the first time and applies the declared tests.

    python scripts/ieeg_soz_confirmatory.py --stage scan
    python scripts/ieeg_soz_confirmatory.py --stage test

Nothing in stage 2 may alter a prediction; the void conditions in the
protocol apply.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from ieeg_gate import (  # noqa: E402  pipeline reused verbatim
    DATA, DEV, DONOR_R2, MIN_DONORS, N_GHOSTS, arm, good_channels, norm_label,
    preprocess, read_edf,
)

OUT = Path("ExpOutput/ieeg_soz")
QUARANTINED = {"NIH1"}
MIN_GROUP = 3          # declared: fewer than 3 in either group -> excluded


def confirmatory_subjects() -> list[str]:
    """The cohort fixed by the gate: G3-clean bipolar, NIH1 quarantined."""
    g = pd.read_csv("ExpOutput/ieeg_gate_cohort.csv")
    bp = g[(g.tag == "bipolar") & (g.g_med <= 0.005)]
    return sorted(set(bp["sub"]) - QUARANTINED)


def scan() -> int:
    """Stage 1: per-channel excess with labels. No label column is read."""
    OUT.mkdir(parents=True, exist_ok=True)
    subs = confirmatory_subjects()
    print(f"cohort: {len(subs)} subjects\ndevice: {DEV}")
    for sub in subs:
        dest = OUT / f"excess_{sub}.csv"
        if dest.exists():
            print(f"[{sub}] done, skipping")
            continue
        t0 = time.time()
        try:
            scan_one(sub, dest, t0)
        except RuntimeError as exc:
            # The GPU is shared with the desktop; when VRAM is short cuBLAS
            # fails mid-run. Skip and continue: the loop is resumable, and a
            # second pass (CUDA_VISIBLE_DEVICES=-1) fills any gaps on CPU.
            print(f"[{sub}] FAILED: {str(exc)[:90]}", flush=True)
            if dest.exists():
                dest.unlink()
    return 0


def scan_one(sub: str, dest: Path, t0: float) -> None:
    if True:
        x, labels, fs = read_edf(DATA / f"sub-{sub}_run-01_ieeg.edf")
        idx = good_channels(sub, labels)
        x = x[:, idx]
        labels_n = [norm_label(labels[i]) for i in idx]
        rows = []
        for mode in ("bipolar", "raw"):
            xp, names, _ = preprocess_labelled(x, labels_n, fs, mode)
            r = arm(xp, mode, return_channels=True)
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for name, ex, sr in zip(names, r["excess"], r["self_r2"]):
                rows.append({"sub": sub, "arm": mode, "channel": name,
                             "excess": ex, "self_r2": sr,
                             "threshold": max(0.0, r["g_max"])})
        pd.DataFrame(rows).to_csv(dest, index=False)
        print(f"[{sub}] {len(rows)} channel-rows "
              f"({time.time()-t0:.0f}s)", flush=True)


def preprocess_labelled(x, labels, fs, mode):
    """preprocess(), but returning the surviving channel names."""
    from ieeg_gate import DECIM, SEG_SECONDS, montage
    T = (x.shape[0] // DECIM) * DECIM
    xx = x[:T].reshape(-1, DECIM, x.shape[1]).mean(1)
    fs2 = fs / DECIM
    seg = int(SEG_SECONDS * fs2)
    mid = xx.shape[0] // 2
    xx = xx[mid - seg // 2: mid + seg // 2]
    alive = xx.std(0) > 1e-6
    xx = xx[:, alive]
    names = [l for l, a in zip(labels, alive) if a]
    xx, names = montage(xx, names, mode)
    xx = np.diff(xx, axis=0)
    xx = np.nan_to_num(xx, nan=0.0, posinf=0.0, neginf=0.0)
    keep = xx.std(0) > 1e-9
    xx = xx[:, keep]
    names = [n for n, k in zip(names, keep) if k]
    return xx, names, fs2


# ---------------------------------------------------------------- stage 2

def contacts_of(channel: str) -> list[str]:
    """Bipolar 'RAI2-RAI1' -> both contacts; a raw name -> itself."""
    if "-" in channel:
        return [p.strip() for p in channel.split("-")]
    return [channel.strip()]


def shaft_of(contact: str) -> str:
    m = re.match(r"([A-Za-z']+)\d+$", contact)
    return m.group(1) if m else contact


def label_table(sub: str) -> pd.DataFrame:
    """Opens soz/epz/rz for the first time, per the declared specification."""
    t = pd.read_csv(DATA / f"sub-{sub}_channels.tsv", sep="\t")
    keep = [c for c in ("name", "soz", "epz", "rz") if c in t.columns]
    t = t[keep].copy()
    for c in ("soz", "epz", "rz"):
        if c in t.columns:
            t[c] = (t[c].astype(str).str.strip().str.lower()
                    .isin(["yes", "1", "true", "y"]))
    return t.set_index("name")


def annotate(df: pd.DataFrame, sub: str) -> pd.DataFrame:
    lab = label_table(sub)
    soz_set = set(lab.index[lab.soz]) if "soz" in lab.columns else set()
    rz_set = set(lab.index[lab.rz]) if "rz" in lab.columns else set()
    # declared: a derivation is SOZ if EITHER contact is SOZ (permissive)
    df = df.copy()
    df["is_soz"] = [any(c in soz_set for c in contacts_of(ch))
                    for ch in df.channel]
    df["is_rz"] = [any(c in rz_set for c in contacts_of(ch))
                   for ch in df.channel]
    soz_shafts = {shaft_of(c) for c in soz_set}
    df["adjacent"] = [any(shaft_of(c) in soz_shafts for c in contacts_of(ch))
                      for ch in df.channel]
    return df


def per_subject_ps1(d: pd.DataFrame) -> dict | None:
    a = d.excess[d.is_soz].to_numpy()
    b = d.excess[~d.is_soz].to_numpy()
    if len(a) < MIN_GROUP or len(b) < MIN_GROUP:
        return None
    u = stats.mannwhitneyu(a, b, alternative="less")
    rbc = 1 - 2 * u.statistic / (len(a) * len(b))       # +1 => SOZ below
    thr = float(d.threshold.iloc[0])
    return {"n_soz": len(a), "n_non": len(b), "p": float(u.pvalue),
            "rank_biserial": float(rbc),
            "median_soz": float(np.median(a)),
            "median_non": float(np.median(b)),
            "frac_soz_core": float((a > thr).mean()),
            "frac_non_core": float((b > thr).mean())}


def per_subject_ps2(d: pd.DataFrame) -> dict | None:
    n = d[~d.is_soz]
    a = n.excess[n.adjacent].to_numpy()
    b = n.excess[~n.adjacent].to_numpy()
    if len(a) < MIN_GROUP or len(b) < MIN_GROUP:
        return None
    u = stats.mannwhitneyu(a, b, alternative="greater")
    return {"n_adj": len(a), "n_dist": len(b), "p": float(u.pvalue),
            "rank_biserial": float(2 * u.statistic / (len(a) * len(b)) - 1)}


def stouffer(ps: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    z = stats.norm.isf(np.clip(ps, 1e-12, 1 - 1e-12))
    zc = float((w * z).sum() / np.sqrt((w ** 2).sum()))
    return zc, float(stats.norm.sf(zc))


def test() -> int:
    files = sorted(OUT.glob("excess_*.csv"))
    if not files:
        print("no scans found; run --stage scan first")
        return 1
    all_rows = pd.concat([pd.read_csv(f) for f in files])
    outcome = load_outcomes()

    for armname in ("bipolar", "raw"):
        print("\n" + "=" * 74)
        print(f"ARM: {armname}"
              f"{'   [CONFIRMATORY]' if armname == 'bipolar' else '   [sensitivity]'}")
        s1, s2 = [], []
        for sub, d in all_rows[all_rows.arm == armname].groupby("sub"):
            d = annotate(d, sub)
            r1 = per_subject_ps1(d)
            if r1:
                r1["sub"] = sub
                s1.append(r1)
            r2 = per_subject_ps2(d)
            if r2:
                r2["sub"] = sub
                s2.append(r2)

        t1 = pd.DataFrame(s1)
        if t1.empty:
            print("  P-S1: no subject met the declared group-size minimum")
            continue
        t1.to_csv(OUT / f"ps1_{armname}.csv", index=False)
        w = np.sqrt(t1.n_soz + t1.n_non)
        zc, pc = stouffer(t1.p.to_numpy(), w.to_numpy())
        pos = int((t1.rank_biserial > 0).sum())
        sign_p = stats.binomtest(pos, len(t1), 0.5, alternative="greater").pvalue
        print(f"  P-S1  subjects tested: {len(t1)} "
              f"(excluded {len(files) - len(t1)} for group size)")
        print(f"        Stouffer z = {zc:+.3f}   p = {pc:.4g}  (one-sided)")
        print(f"        direction:  {pos}/{len(t1)} subjects show depletion, "
              f"sign test p = {sign_p:.4g}")
        print(f"        median rank-biserial = {t1.rank_biserial.median():+.3f}"
              "   (+ = SOZ depleted)")
        print(f"        driven-core membership: SOZ "
              f"{t1.frac_soz_core.mean():.3f} vs non-SOZ "
              f"{t1.frac_non_core.mean():.3f}")

        t2 = pd.DataFrame(s2)
        if not t2.empty:
            t2.to_csv(OUT / f"ps2_{armname}.csv", index=False)
            z2, p2 = stouffer(t2.p.to_numpy(),
                              np.sqrt(t2.n_adj + t2.n_dist).to_numpy())
            print(f"  P-S2  (WEAK) subjects: {len(t2)}   z = {z2:+.3f}  "
                  f"p = {p2:.4g}   median rbc = "
                  f"{t2.rank_biserial.median():+.3f}")

        if outcome is not None and armname == "bipolar":
            m = t1.merge(outcome, on="sub", how="inner")
            if len(m) and m.success.nunique() > 1:
                g1 = m.rank_biserial[m.success]
                g0 = m.rank_biserial[~m.success]
                u = stats.mannwhitneyu(g1, g0, alternative="greater")
                print(f"  P-S3  (EXPLORATORY) success n={len(g1)} "
                      f"median rbc {g1.median():+.3f} | failure n={len(g0)} "
                      f"median rbc {g0.median():+.3f}   p = {u.pvalue:.4g}")
            else:
                print("  P-S3  (EXPLORATORY) not evaluable: "
                      "outcome subgroups absent or single-valued")
        elif armname == "bipolar":
            print("  P-S3  (EXPLORATORY) not evaluable: no outcome column")
    return 0


def load_outcomes():
    p = DATA / "participants.tsv"
    if not p.exists():
        return None
    t = pd.read_csv(p, sep="\t")
    idc = [c for c in t.columns if c.lower() in ("participant_id", "sub")]
    oc = [c for c in t.columns
          if any(k in c.lower() for k in ("outcome", "engel", "ilae"))]
    if not idc or not oc:
        return None
    col = oc[0]
    out = pd.DataFrame({
        "sub": t[idc[0]].astype(str).str.replace("sub-", "", regex=False),
        "raw_outcome": t[col].astype(str)})
    # Engel I / ILAE 1-2 / "success" count as success; anything else failure
    out["success"] = out.raw_outcome.str.strip().str.upper().str.match(
        r"^(I\b|1\b|2\b|ENGEL\s*I\b|SUCCESS|S\b)")
    print(f"\noutcome column used: {col!r}  "
          f"({int(out.success.sum())} success / "
          f"{int((~out.success).sum())} other)")
    return out[["sub", "success"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["scan", "test"], required=True)
    a = ap.parse_args()
    raise SystemExit(scan() if a.stage == "scan" else test())
