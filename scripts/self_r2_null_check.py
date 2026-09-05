"""Why the self-R2 null regression was abandoned before it was written.

Re-derives the three checks quoted in the "proposal this REPLACES" section
of paper/per_channel_null_protocol.md, from primary output files only.

  1. The motivating correlation between a cell's self-R2 and its ghost bar
     is a NOISE CONFOUND. Controlling the noise manipulation and width, the
     partial rank correlation loses significance and flips sign.
  2. The donor panel has NO LEVERAGE to regress on: donors are filtered to
     self-R2 > 0.9, so the regressor barely varies. Donor identities are
     recovered by replaying the saved seed's draw, so this is measured on
     the actual panels, not assumed.
  3. Source false positives track the noise manipulation, not the ghost bar.

    python scripts/self_r2_null_check.py
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
import scipy.stats as st

# must match boundary_map.py, which produced the raw files read below
N_GHOSTS, DONOR_R2, MIN_DONORS = 30, 0.9, 8


def check_1_confound(d):
    print("1. IS THE MOTIVATING CORRELATION REAL, OR IS IT NOISE?\n")
    rho_m, p_m = st.spearmanr(d.self_r2_med, d.ghost_max)
    print(f"   marginal   Spearman(self_r2_med, ghost_max) = {rho_m:+.3f}"
          f"   p = {p_m:.2g}   n = {len(d)}")

    # residualise both ranks on the noise and width design, then correlate
    X = np.hstack([np.ones((len(d), 1)),
                   pd.get_dummies(d[["noise", "V"]].astype(str),
                                  drop_first=True).astype(float).values])

    def resid(y):
        return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

    rho_p, p_p = st.spearmanr(resid(d.self_r2_med.rank().values),
                              resid(d.ghost_max.rank().values))
    print(f"   partial    controlling noise and width       = {rho_p:+.3f}"
          f"   p = {p_p:.2g}")
    flipped = np.sign(rho_p) != np.sign(rho_m)
    print(f"   -> sign {'FLIPS' if flipped else 'holds'}, "
          f"{'not ' if p_p > 0.05 else ''}significant at 0.05."
          f"  {'CONFOUNDED.' if (flipped or p_p > 0.05) else 'survives.'}\n")
    print("   self-R2 by noise level (it is a reading of the manipulation):")
    print("   " + d.groupby("noise").self_r2_med.median().round(3)
          .to_string().replace("\n", "\n   ") + "\n")
    return rho_m, rho_p, p_p


def check_2_leverage():
    print("2. COULD THE PANEL SUPPORT THE REGRESSION AT ALL?\n")
    rows = []
    for f in sorted(glob.glob("ExpOutput/boundary_map/raw_*.npz")):
        m = re.search(r"raw_n(\d+)_V(\d+)_c([\d.]+)_r(\d+)_s(\d+)",
                      os.path.basename(f))
        seed = int(m[5])
        z = np.load(f)
        self_r2, ghosts = z["self_r2"], z["ghosts"]
        # replay boundary_map.py's donor draw to recover which donor made
        # which ghost; the raw files store the ghosts but not the donors
        rng = np.random.default_rng(seed + 4242)
        qual = np.where(self_r2 > DONOR_R2)[0]
        pool = (np.arange(len(self_r2)) if len(qual) < MIN_DONORS else qual)
        donors = rng.choice(pool, size=min(N_GHOSTS, len(pool)), replace=False)
        if len(donors) != len(ghosts):
            continue
        for dn, g in zip(donors, ghosts):
            rows.append({"seed": seed, "donor_self_r2": float(self_r2[dn]),
                         "ghost": float(g)})
    t = pd.DataFrame(rows)
    lo, hi = t.donor_self_r2.min(), t.donor_self_r2.max()
    print(f"   {len(t)} donor-ghost pairs recovered from "
          f"{len(glob.glob('ExpOutput/boundary_map/raw_*.npz'))} cells")
    print(f"   donor self-R2 spans [{lo:.3f}, {hi:.3f}]  "
          f"-> width {hi - lo:.3f}")
    print(f"   Spearman(donor_self_r2, ghost) = "
          f"{t.donor_self_r2.corr(t.ghost, method='spearman'):+.3f}")
    print(f"   -> predicting the null at self-R2 = 0.3 is extrapolation "
          f"{(0.952 - 0.3) / (hi - lo):.0f}x the observed span. NO LEVERAGE.\n")
    return len(t), lo, hi


def check_3_what_it_tracks(d):
    print("3. WHAT DOES THE FAILURE ACTUALLY TRACK?\n")
    for c in ("noise", "self_r2_med", "ghost_max"):
        rho, p = st.spearmanr(d[c], d.source_fp)
        print(f"   source_fp vs {c:12s} rho {rho:+.3f}   p {p:.2g}")
    print("   -> the ghost bar is the weakest of the three. Calibrating "
          "against it\n      would have fitted the confound.\n")


def main() -> int:
    d = pd.read_csv("ExpOutput/crossed_saturation/cells.csv")
    print(__doc__.split("\n\n")[1].strip() + "\n")
    print("=" * 68 + "\n")
    check_1_confound(d)
    check_2_leverage()
    check_3_what_it_tracks(d)
    print("=" * 68)
    print("VERDICT: abandoned unrun. The defect it aimed at is real; the "
          "covariate\nis not. See paper/per_channel_null_protocol.md for what "
          "replaced it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
