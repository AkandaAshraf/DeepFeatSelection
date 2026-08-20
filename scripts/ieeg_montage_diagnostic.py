"""Why do the bipolar and raw montages disagree on P-S1?

DIAGNOSTIC, NOT CONFIRMATORY. The pre-registered tests are already run and
recorded; this asks a methodological question about the disagreement and is
labelled exploratory wherever it appears.

The convenient hypothesis is that the shared recording reference destroys
the raw arm's ability to resolve the question at all, which would mean its
null is uninformative rather than contradictory. That hypothesis is exactly
the one favouring the arm that agreed with the prediction, so the checks
below are built so it can fail:

  D1 dispersion   if the reference merely adds a constant, ranks are
                  unaffected and saturation cannot explain a null rank test.
                  The story needs raw to carry EXTRA per-channel variance,
                  not just a higher mean. Measured, not assumed.
  D2 ghosts       the surrogate panel bounds the shared-signal artefact per
                  arm; raw should be worse if the reference is the culprit.
  D3 separation   effect size between SOZ and non-SOZ relative to
                  within-group spread, per arm.
  D4 agreement    map each bipolar derivation to its contacts and correlate
                  with the raw scores of those contacts. Low agreement means
                  the arms measure different quantities; high agreement with
                  divergent tests means something subtler.

    python scripts/ieeg_montage_diagnostic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from ieeg_soz_confirmatory import annotate, contacts_of  # noqa: E402

OUT = Path("ExpOutput/ieeg_soz")


def main() -> int:
    files = sorted(OUT.glob("excess_*.csv"))
    rows = pd.concat([pd.read_csv(f) for f in files])
    print(f"subjects: {rows['sub'].nunique()}   arms: {sorted(set(rows.arm))}")

    per = []
    for sub, d0 in rows.groupby("sub"):
        d0 = annotate(d0, sub)
        rec = {"sub": sub}
        for arm in ("bipolar", "raw"):
            d = d0[d0.arm == arm]
            if d.empty:
                continue
            ex = d.excess.to_numpy()
            thr = float(d.threshold.iloc[0])
            a = d.excess[d.is_soz].to_numpy()
            b = d.excess[~d.is_soz].to_numpy()
            sd = ex.std() + 1e-12
            rec[f"{arm}_sd"] = ex.std()
            rec[f"{arm}_iqr"] = float(np.subtract(*np.percentile(ex, [75, 25])))
            rec[f"{arm}_median"] = float(np.median(ex))
            rec[f"{arm}_thr"] = thr
            rec[f"{arm}_frac_core"] = float((ex > thr).mean())
            if len(a) >= 3 and len(b) >= 3:
                rec[f"{arm}_d"] = float((b.mean() - a.mean()) / sd)
        # D4: bipolar derivation vs the mean of its contacts in raw
        bp = d0[d0.arm == "bipolar"]
        rw = d0[d0.arm == "raw"].set_index("channel").excess
        if not bp.empty and not rw.empty:
            paired = []
            for ch, ex in zip(bp.channel, bp.excess):
                cs = [c for c in contacts_of(ch) if c in rw.index]
                if len(cs) == 2:
                    paired.append((ex, float(rw.loc[cs].mean())))
            if len(paired) >= 8:
                p = np.array(paired)
                rec["cross_rho"] = float(stats.spearmanr(p[:, 0], p[:, 1])[0])
                rec["n_paired"] = len(paired)
        per.append(rec)

    t = pd.DataFrame(per)
    t.to_csv(OUT / "montage_diagnostic.csv", index=False)

    def med(c):
        return float(t[c].median()) if c in t and t[c].notna().any() else float("nan")

    print("\n" + "=" * 70)
    print("D1  dispersion of excess (does raw carry EXTRA variance?)")
    print(f"     bipolar   SD {med('bipolar_sd'):.4f}   IQR {med('bipolar_iqr'):.4f}"
          f"   median {med('bipolar_median'):+.4f}")
    print(f"     raw       SD {med('raw_sd'):.4f}   IQR {med('raw_iqr'):.4f}"
          f"   median {med('raw_median'):+.4f}")
    ratio = med("raw_sd") / med("bipolar_sd")
    print(f"     raw/bipolar SD ratio = {ratio:.2f}")
    print("     -> ratio near 1 means the reference adds a LEVEL shift, which")
    print("        cannot change ranks; the saturation story then FAILS.")

    print("\nD2  driven-core membership and thresholds")
    print(f"     bipolar   frac above threshold {med('bipolar_frac_core'):.3f}"
          f"   threshold {med('bipolar_thr'):+.4f}")
    print(f"     raw       frac above threshold {med('raw_frac_core'):.3f}"
          f"   threshold {med('raw_thr'):+.4f}")

    print("\nD3  SOZ vs non-SOZ separation, in within-subject SD units")
    for arm in ("bipolar", "raw"):
        c = f"{arm}_d"
        if c in t:
            v = t[c].dropna()
            print(f"     {arm:8s} median d = {v.median():+.3f}   "
                  f"positive in {int((v>0).sum())}/{len(v)} subjects")

    print("\nD4  cross-montage agreement (bipolar vs mean of its contacts)")
    if "cross_rho" in t:
        v = t.cross_rho.dropna()
        print(f"     median Spearman rho = {v.median():+.3f}   "
              f"range [{v.min():+.3f}, {v.max():+.3f}]   n={len(v)} subjects")
        print("     -> high rho with divergent tests means the arms agree on")
        print("        ordering and the disagreement is NOT 'different quantity'.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
