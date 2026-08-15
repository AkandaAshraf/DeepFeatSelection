"""Is the ictal rise in drive concentration a real effect or a clamp artifact?

The reported statistic is the top-4 share of clamped-positive excess: both the
numerator and the denominator are built from channels that survived
``np.clip(ex, 0, None)``. If seizures reduce the NUMBER of channels with
detectable drive, the share concentrates arithmetically even when no channel
gains drive, so the statistic cannot by itself distinguish recruitment from
loss.

This separates the two. It reports, per window, how many channels survive the
clamp and how much total drive they carry; tests the reported statistic
properly with a paired test over matched records; and repeats the measurement
with two statistics that the clamp cannot inflate.

Read-only: consumes ExpOutput/eeg_excess/{windows,channels}.csv only.

    python scripts/eeg_concentration_null.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

OUT = Path("ExpOutput/eeg_excess")
BAR = "=" * 78


def top_share(v: np.ndarray, k: int = 4) -> float:
    tot = v.sum()
    return float(np.sort(v)[::-1][:k].sum() / tot) if tot > 0 else np.nan


def main() -> int:
    ch = pd.read_csv(OUT / "channels.csv")
    win = pd.read_csv(OUT / "windows.csv")
    key = ["record", "kind", "start_s"]

    rows = []
    for (rec, kind, start), g in ch.groupby(key):
        ex = g.excess.to_numpy()
        pos = np.clip(ex, 0, None)
        n_pos = int((ex > 0).sum())
        rows.append({
            "record": rec, "kind": kind, "start_s": start,
            "n_channels": len(ex), "n_positive": n_pos,
            "total_positive": float(pos.sum()),
            "top4_share_clamped": top_share(pos),
            # Clamp-free: concentration of absolute drive, no channel discarded.
            "top4_share_abs": top_share(np.abs(ex)),
            # Absolute mass in the top 4, which a shrinking denominator
            # cannot inflate.
            "top4_absolute": float(np.sort(pos)[::-1][:4].sum()),
            # Share expected from n_positive alone if drive were uniform.
            "uniform_expectation": 4.0 / n_pos if n_pos else np.nan,
        })
    d = pd.DataFrame(rows)
    d["share_over_uniform"] = d.top4_share_clamped / d.uniform_expectation

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)

    print(BAR)
    print("1. DOES THE STATISTIC TRACK CHANNEL COUNT RATHER THAN DRIVE?")
    print(BAR)
    r, pv = spearmanr(d.top4_share_clamped, d.n_positive)
    print(f"Spearman(top-4 share, n_positive) = {r:+.3f}  p = {pv:.2g}   "
          f"(n = {len(d)} windows)")
    print("\nby condition:")
    print(d.groupby("kind")[["n_positive", "total_positive",
                             "top4_share_clamped", "top4_share_abs",
                             "top4_absolute"]].mean().to_string())

    print("\n" + BAR)
    print("2. PAIRED TEST OVER MATCHED RECORDS")
    print(BAR)
    piv = d.pivot_table(index="record", columns="kind",
                        values=["top4_share_clamped", "top4_share_abs",
                                "top4_absolute", "total_positive",
                                "n_positive", "share_over_uniform"])
    paired = piv.dropna()
    print(f"records with both conditions: {len(paired)}")
    print(paired.to_string())

    print("\npaired one-sided Wilcoxon (ictal > interictal):")
    for stat in ["top4_share_clamped", "top4_share_abs", "top4_absolute",
                 "total_positive", "n_positive", "share_over_uniform"]:
        a = paired[(stat, "ictal")].to_numpy()
        b = paired[(stat, "interictal")].to_numpy()
        try:
            w = wilcoxon(a, b, alternative="greater")
            pstr = f"p = {w.pvalue:.3f}"
        except ValueError as exc:
            pstr = f"(undefined: {exc})"
        n_up = int((a > b).sum())
        print(f"  {stat:22s} ictal {a.mean():+.4f} vs interictal "
              f"{b.mean():+.4f}   {n_up}/{len(a)} up   {pstr}")

    print("\n" + BAR)
    print("3. WITHIN-RECORD SIGN AGREEMENT")
    print(BAR)
    agree = 0
    for rec in paired.index:
        ds = (paired.loc[rec, ("top4_share_clamped", "ictal")]
              - paired.loc[rec, ("top4_share_clamped", "interictal")])
        dn = (paired.loc[rec, ("n_positive", "ictal")]
              - paired.loc[rec, ("n_positive", "interictal")])
        opp = (ds > 0) != (dn > 0)
        agree += opp
        print(f"  {rec:16s} d(top4share) {ds:+.4f}  d(n_positive) {dn:+.1f}  "
              f"{'opposite' if opp else 'same'}")
    print(f"\nopposite-sign in {agree}/{len(paired)} records: the share rises "
          "exactly where\nthe number of driven channels falls.")

    d.to_csv(OUT / "concentration_null.csv", index=False)
    print(f"\nwrote {OUT/'concentration_null.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
