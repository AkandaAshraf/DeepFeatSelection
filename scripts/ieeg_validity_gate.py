"""Validity gate for the same-scale sensitivity arms, declared in
paper/ieeg_protocol.md before the arms were run.

An arm counts as a sensitivity check on P-S1 only if it measures the same
quantity as the confirmatory bipolar arm. The raw arm did not (median
Spearman rho 0.078 against bipolar), which is why its null was uninformative
rather than contradictory. The declared threshold is:

    median per-subject Spearman rho with bipolar > 0.30

An arm below that is reported as UNINFORMATIVE and its P-S1 result is not
interpreted as evidence either way. This script computes the gate only; it
reads no SOZ label.

    python scripts/ieeg_validity_gate.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

OUT = Path("ExpOutput/ieeg_soz")
GATE = 0.30          # declared threshold
ARMS = ("raw", "laplacian", "bipolar_skip")


def contacts(ch: str) -> set[str]:
    """Contacts a channel is built from, for any of the derivations."""
    ch = ch.strip()
    if ch.endswith("_lap"):
        stem = ch[:-4]
        m = re.match(r"([A-Za-z']+)(\d+)$", stem)
        if not m:
            return {stem}
        sh, n = m.group(1), int(m.group(2))
        return {f"{sh}{n-1}", f"{sh}{n}", f"{sh}{n+1}"}
    if "-" in ch:
        return {p.strip() for p in ch.split("-")}
    return {ch}


def gate_for(d: pd.DataFrame, armname: str) -> float | None:
    """Per-subject Spearman between an arm and bipolar, matched by contacts."""
    bp = d[d.arm == "bipolar"]
    other = d[d.arm == armname]
    if bp.empty or other.empty:
        return None
    bp_sets = [(contacts(c), e) for c, e in zip(bp.channel, bp.excess)]
    xs, ys = [], []
    for ch, ex in zip(other.channel, other.excess):
        cs = contacts(ch)
        # bipolar channels built from any of the same contacts
        vals = [e for s, e in bp_sets if s & cs]
        if vals:
            xs.append(ex)
            ys.append(float(np.mean(vals)))
    if len(xs) < 8:
        return None
    r = stats.spearmanr(xs, ys)[0]
    return float(r) if np.isfinite(r) else None


def main() -> int:
    files = sorted(OUT.glob("excess_*.csv"))
    if not files:
        print("no scans found")
        return 1
    rows = pd.concat([pd.read_csv(f) for f in files])
    present = sorted(set(rows.arm))
    print(f"subjects: {rows['sub'].nunique()}   arms present: {present}\n")

    print(f"VALIDITY GATE (declared: median Spearman rho with bipolar > "
          f"{GATE:.2f})\n")
    summary = []
    for armname in ARMS:
        if armname not in present:
            print(f"  {armname:14s} not yet scanned")
            continue
        vals = []
        for sub, d in rows.groupby("sub"):
            r = gate_for(d, armname)
            if r is not None:
                vals.append(r)
        if not vals:
            print(f"  {armname:14s} no comparable channels")
            continue
        v = np.array(vals)
        verdict = "PASS -> is a sensitivity arm" if np.median(v) > GATE \
            else "FAIL -> UNINFORMATIVE, result not interpreted"
        print(f"  {armname:14s} median rho {np.median(v):+.3f}   "
              f"IQR [{np.percentile(v,25):+.3f}, {np.percentile(v,75):+.3f}]   "
              f"n={len(v)}")
        print(f"  {'':14s} {verdict}\n")
        summary.append({"arm": armname, "median_rho": float(np.median(v)),
                        "n_subjects": len(v),
                        "passes_gate": bool(np.median(v) > GATE)})
    if summary:
        pd.DataFrame(summary).to_csv(OUT / "validity_gate.csv", index=False)
        print("written to", OUT / "validity_gate.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
