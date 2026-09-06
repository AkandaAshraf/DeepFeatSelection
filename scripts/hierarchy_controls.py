"""POST HOC, labelled. Is the module-share ordering real, or a size artefact?

The declared decisive prediction was refused: the random-module control
cleared the same localisation bar as clustering. The added secondary, module
share, ordered as predicted. Before believing that ordering, two checks.

  (a) SIZE. Module codes are sized 2*|module|. If clustered modules are
      systematically larger than random ones, a larger code explains more and
      the ordering is about width, not structure.
  (b) GRADED (Rule 125). Within an arm, a module holding a higher fraction of
      its members' parents should capture more of their inflow. That is a
      magnitude relationship no size difference produces.

    python scripts/hierarchy_controls.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ARMS = ("HIER-CLUST", "HIER-RAND", "HIER-TRUE")
CELLS = [(V, nz, s) for V in (30, 60) for nz in (0.0, 0.05)
         for s in (0, 1, 2)]


def main() -> int:
    print(__doc__.split("\n\n", 1)[1].rsplit("\n\n", 1)[0] + "\n")
    size_rows, mod_rows = [], []
    for V, nz, s in CELLS:
        z = np.load(f"ExpOutput/hierarchy/raw_V{V}_nz{nz}_s{s}.npz")
        isd, parent = z["is_driven"], z["parent"]
        n_src = max(3, V // 6)
        for arm in ARMS:
            lab, e2, e3 = z[arm + "_lab"], z[arm + "_e2"], z[arm + "_e3"]
            sizes = np.bincount(lab, minlength=lab.max() + 1)
            size_rows.append(dict(V=V, noise=nz, seed=s, arm=arm,
                                  mean_size=float(sizes.mean()),
                                  sd_size=float(sizes.std())))
            for mid in np.unique(lab):
                mem = np.where(lab == mid)[0]
                drv = mem[isd[mem]]
                if len(drv) < 2:
                    continue
                frac_in = float(np.mean(
                    [lab[parent[q - n_src]] == mid for q in drv]))
                tot2, tot3 = float(e2[drv].mean()), float(e3[drv].mean())
                den = tot2 + tot3
                mod_rows.append(dict(
                    V=V, noise=nz, seed=s, arm=arm, size=len(mem),
                    frac_parent_in=frac_in,
                    share=float(tot2 / den) if den > 0 else np.nan))
    sz = pd.DataFrame(size_rows)
    md = pd.DataFrame(mod_rows).dropna(subset=["share"])

    print("(a) MODULE SIZE by arm  (codes are sized 2 x |module|)")
    print("   " + sz.groupby("arm")[["mean_size", "sd_size"]].mean()
          .round(2).to_string().replace("\n", "\n   "))
    same = sz.groupby("arm").mean_size.mean()
    spread = same.max() - same.min()
    print(f"   spread in mean size across arms: {spread:.2f} variables")
    print("   -> " + ("sizes match, so the ordering is NOT a width artefact"
                      if spread < 0.5 else
                      "sizes DIFFER; the ordering may be a width artefact"))

    print("\n(b) GRADED, within arm: does a module that holds more of its")
    print("    members' parents capture more of their inflow?")
    for arm in ARMS:
        g = md[md.arm == arm]
        if g.frac_parent_in.nunique() < 3:
            print(f"   {arm:11s} n={len(g):3d}  frac_parent_in is constant "
                  f"at {g.frac_parent_in.iloc[0]:.2f}; test undefined")
            continue
        rho, p = spearmanr(g.frac_parent_in, g.share)
        print(f"   {arm:11s} n={len(g):3d}  rho {rho:+.3f}  p {p:.4f}")
    pooled = md[md.arm != "HIER-TRUE"]
    rho, p = spearmanr(pooled.frac_parent_in, pooled.share)
    print(f"   pooled (excl oracle, whose frac is always 1.0): "
          f"n={len(pooled)}  rho {rho:+.3f}  p {p:.2g}")
    print("\n   A positive graded relationship is the magnitude evidence the")
    print("   refused localisation metric could not supply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
