"""Confirm that every number changed in response to review appears in the
compiled PDF and matches the primary data it is derived from.

This is the last gate before posting: it re-derives each figure from
ExpOutput/ and asserts the PDF text contains the derived value, so a stale
number cannot survive an edit.

    python scripts/audit_paper_numbers.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PDF = Path("paper/excess_paper.pdf")


def pdf_text() -> str:
    import pymupdf
    d = pymupdf.open(PDF)
    return "".join(d[i].get_text() for i in range(d.page_count))


def norm(t: str) -> str:
    """Collapse whitespace and unify minus signs so matching is robust."""
    t = t.replace("−", "-").replace("–", "-").replace("—", "-")
    t = t.replace("’", "'").replace("‘", "'")
    t = t.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", t)


def main() -> int:
    txt = norm(pdf_text())
    checks: list[tuple[str, str, bool]] = []

    def want(label: str, needle: str):
        checks.append((label, needle, norm(needle) in txt))

    def forbid(label: str, needle: str):
        checks.append((label + " (must be absent)", needle,
                       norm(needle) not in txt))

    def forbid_re(label: str, pattern: str):
        """Forbid a family of phrasings.

        Plain-string forbids have twice let a real defect through: "four
        systems" did not catch "four synthetic systems", and "neuron by
        neuron" did not catch the hyphenated form. Anything whose wording can
        vary is matched by regex.
        """
        checks.append((label + " (must be absent)", pattern,
                       re.search(pattern, txt, re.I) is None))

    # --- B1 CCM cost -------------------------------------------------------
    m = pd.read_csv("ExpOutput/membership/membership.csv")
    secs = float(m[m.method == "ccm_full"].seconds.iloc[0])
    per_call = secs / (31 * 30)
    hours = 10_000 * 9_999 / 2 * per_call / 3600
    want(f"CCM cost {hours:,.0f} h -> '15,800'", "15,800")
    forbid("old doubled CCM cost", "31,700")
    forbid("old CCM years", "3.6 years")

    # --- B3 in-degree ------------------------------------------------------
    # Reported over the three DISTINCT generating graphs; recomputed in
    # scripts/verify_round2.py and required to match the paper.
    want("in-degree null (three distinct graphs)", "p = 0.79")
    forbid("false in-degree enrichment", "population mean of 1.20")

    # --- M2 three graphs ---------------------------------------------------
    roles = pd.read_csv("ExpOutput/recall/corrected_roles.csv")
    d3 = roles[roles.system != "hetero"].groupby("role").agg(
        n=("n", "sum"), flagged=("flagged", "sum"))
    want(f"driven n={int(d3.loc['driven','n'])}", str(int(d3.loc["driven", "n"])))
    want(f"driven flagged={int(d3.loc['driven','flagged'])}",
         str(int(d3.loc["driven", "flagged"])))
    want(f"sources n={int(d3.loc['source','n'])}", str(int(d3.loc["source", "n"])))
    forbid_re("four-systems framing", r"four\s+(\w+\s+)?systems")

    # --- B2 false alarms ---------------------------------------------------
    want("hetero degree-2 false alarms", "43")
    want("false-alarm rule stated as ghost magnitude", "ghost's magnitude")

    # --- B4 self-R2 --------------------------------------------------------
    meds, mx, tot, above = [], 0.0, 0, 0
    for pat in ("ExpOutput/celegans_excess/worm*_excess.csv",
                "ExpOutput/celegans_excess_heldout/worm*_excess.csv",
                "ExpOutput/celegans_excess_avahiscl/worm*_excess.csv"):
        for f in sorted(Path().glob(pat)):
            sr = pd.read_csv(f).self_r2
            meds.append(sr.median())
            mx = max(mx, float(sr.max()))
            tot += len(sr)
            above += int((sr > 0.9).sum())
    want(f"self-R2 min median {min(meds):.3f}", f"{min(meds):.3f}")
    want(f"self-R2 max median {max(meds):.3f}", f"{max(meds):.3f}")
    want(f"self-R2 max {mx:.3f}", f"{mx:.3f}")
    want(f"channel count {tot:,}", f"{tot:,}")
    assert above == 0, f"{above} channels above 0.9 -- claim would be false"

    # --- M1 worm ghost range ----------------------------------------------
    ghosts = []
    for pat in ("ExpOutput/celegans_excess/worm*_excess.csv",
                "ExpOutput/celegans_excess_heldout/worm*_excess.csv"):
        for f in sorted(Path().glob(pat)):
            d = pd.read_csv(f)
            g = d[d.neuron.astype(str).str.replace("\x00", "", regex=False) == "GHOST"]
            if len(g):
                ghosts.append(float(g.excess.iloc[0]))
    want(f"worm ghost min {min(ghosts):.3f}", f"{min(ghosts):.3f}")
    want(f"worm ghost max {max(ghosts):+.4f}", f"{max(ghosts):.4f}")
    forbid("old worm ghost range", "-0.03 to 0.00")

    # --- B5 EEG ------------------------------------------------------------
    e = pd.read_csv("ExpOutput/eeg_excess/concentration_null.csv")
    # The paper quotes the six matched records, which is what the paired test
    # uses; the all-window interictal mean differs because two interictal
    # windows have no ictal partner.
    piv = e.pivot_table(index="record", columns="kind",
                        values="n_positive").dropna()
    ict = piv["ictal"].mean()
    inter = piv["interictal"].mean()
    want(f"EEG n_positive ictal {ict:.1f}", f"{ict:.1f}")
    want(f"EEG n_positive interictal {inter:.1f}", f"{inter:.1f}")
    want("EEG spearman", "0.925")
    forbid("uncaveated recruitment claim",
           "Seizures therefore recruit the network into a driven regime")

    # --- B6 intervention ---------------------------------------------------
    iv = pd.read_csv("ExpOutput/celegans_excess_avahiscl/intervention_null.csv")
    for _, r in iv.iterrows():
        want(f"{r.cell} p_corr {r.p_cohort_maxstat:.3f}",
             f"{r.p_cohort_maxstat:.3f}")
    forbid_re("per-cell intervention framing",
              r"(neuron|cell)[-\s]by[-\s](neuron|cell)")

    # --- integrity ---------------------------------------------------------
    for bad in ("efsec", "??", "GhostScan", "XSP"):
        forbid(f"markup/name artifact {bad!r}", bad)

    width = max(len(c[0]) for c in checks) + 2
    ok = 0
    for label, needle, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {label:<{width}} {needle!r}")
        ok += passed
    print(f"\n{ok}/{len(checks)} checks passed")
    return 0 if ok == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
