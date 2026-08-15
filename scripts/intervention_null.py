"""Do the per-cell intervention deltas survive a null?

Table 3 reports per-neuron changes in within-animal drivenness percentile
between wild-type and AVA-silenced cohorts. Percentiles are a within-animal
ranking, so if silencing depresses the identified neurons as a block, every
cell that was high in wild type must fall regardless of whether it is
individually affected. Nothing in the paper currently separates a per-cell
effect from that cohort-wide shift.

Three nulls are computed, on exactly the frames paper_figures._worm_profiles
builds, so they test the numbers the table reports.

1.  Within-worm permutation of percentiles among identified neurons. Cell
    identity is destroyed while the cohort shift is preserved, so this asks
    whether a named cell moved more than an arbitrary cell moved.

2.  Cohort-label permutation over all 5-versus-5 assignments of the ten
    animals, cell identity preserved. This asks whether the wild-type versus
    silenced split matters at all, and it is the test the cohort-level claim
    needs. A max-statistic variant corrects for testing many cells at once.

3.  A wild-type-only split-half contrast: how large a delta arises between two
    groups of animals that differ by nothing.

Read-only over the archived per-worm CSVs.

    python scripts/intervention_null.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

WT = ([(w, f"ExpOutput/celegans_excess/worm{w}_excess.csv") for w in (0, 1, 2)]
      + [(w, f"ExpOutput/celegans_excess_heldout/worm{w}_excess.csv")
         for w in (3, 4)])
KO = [(w, f"ExpOutput/celegans_excess_avahiscl/worm{w}_excess.csv")
      for w in range(5)]
CELLS = ["AVA", "AVE", "AVB", "VB01", "RMED", "VA01", "RIB", "RIM", "AIB", "ALA"]
N_PERM = 20000
BAR = "=" * 78


def load(paths) -> list[pd.DataFrame]:
    """Per-animal identified-neuron percentile frames, as the figure builds them."""
    out = []
    for _, path in paths:
        d = pd.read_csv(path)
        d["neuron"] = d.neuron.astype(str).str.replace("\x00", "", regex=False)
        d = d[d.neuron != "GHOST"].copy()
        d["identified"] = ~d.neuron.str.isdigit()
        d["pct"] = d.excess.rank(pct=True)
        d["cell"] = d.neuron.str.replace(r"[LR]$", "", regex=True)
        out.append(d[d.identified][["cell", "pct"]].reset_index(drop=True))
    return out


def cell_pct(f: pd.DataFrame, cell: str) -> float:
    """Per-animal mean percentile for a cell, or NaN if absent."""
    hit = f.cell == cell
    return float(f.loc[hit, "pct"].mean()) if hit.any() else np.nan


def cohort_pct(frames, cell) -> float:
    """Cohort statistic, pooled over rows exactly as the reported table is.

    paper_figures._worm_profiles concatenates every identified-neuron row from
    every animal and takes a single mean per cell, so an animal contributing
    two rows (a collapsed L/R pair) carries twice the weight of one
    contributing a single row. The nulls must test that statistic, not a
    mean of per-animal means, or the p-values would refer to a quantity the
    table does not report.
    """
    vals = np.concatenate([f.loc[f.cell == cell, "pct"].to_numpy()
                           for f in frames]) if frames else np.array([])
    return float(vals.mean()) if len(vals) else np.nan


def cohort_delta(wt_frames, ko_frames, cell) -> float:
    a, b = cohort_pct(wt_frames, cell), cohort_pct(ko_frames, cell)
    return np.nan if (np.isnan(a) or np.isnan(b)) else float(b - a)


def main() -> int:
    rng = np.random.default_rng(12345)
    wt, ko = load(WT), load(KO)
    print(f"loaded {len(wt)} wild-type and {len(ko)} silenced recordings; "
          f"identified neurons per animal: "
          f"{[len(f) for f in wt]} / {[len(f) for f in ko]}")

    obs = {c: cohort_delta(wt, ko, c) for c in CELLS}
    present = [c for c in CELLS if not np.isnan(obs[c])]
    print(f"cells resolvable in both cohorts: {present}")

    # Cohort-wide shift, the thing the per-cell claim must beat.
    shift = float(np.mean([f.pct.mean() for f in ko])
                  - np.mean([f.pct.mean() for f in wt]))
    print(f"\ncohort-wide mean percentile shift: {shift:+.4f} "
          "(zero by construction if percentiles are within-animal ranks)")

    # ---- null 1: within-worm identity permutation -------------------------
    # For each cell the statistic pools a known number of rows per cohort;
    # the null resamples that many rows from the same animals' percentile
    # pools, so the block shift is preserved and only identity is destroyed.
    pool_wt = np.concatenate([f.pct.to_numpy() for f in wt])
    pool_ko = np.concatenate([f.pct.to_numpy() for f in ko])

    # ---- null 2: cohort-label permutation ---------------------------------
    # Animals are relabelled, and the statistic is recomputed by pooling rows
    # within each relabelled cohort, matching the reported table exactly.
    allf = wt + ko
    n_all, n_wt = len(allf), len(wt)
    splits = list(itertools.combinations(range(n_all), n_wt))
    null2 = {}
    for c in present:
        st = []
        for sp in splits:
            g1 = [allf[i] for i in range(n_all) if i in sp]
            g2 = [allf[i] for i in range(n_all) if i not in sp]
            a, b = cohort_pct(g1, c), cohort_pct(g2, c)
            st.append(np.nan if (np.isnan(a) or np.isnan(b)) else b - a)
        null2[c] = np.array(st, dtype=float)
    colmax = np.nanmax(np.vstack([np.abs(null2[c]) for c in present]), axis=0)

    # ---- null 3: wild-type-only split-half --------------------------------
    floor = {}
    for c in present:
        have = [f for f in wt if (f.cell == c).any()]
        ds = []
        if len(have) >= 4:
            for comb in itertools.combinations(range(len(have)), len(have) // 2):
                g1 = [have[i] for i in comb]
                g2 = [have[i] for i in range(len(have)) if i not in comb]
                a, b = cohort_pct(g1, c), cohort_pct(g2, c)
                if not (np.isnan(a) or np.isnan(b)):
                    ds.append(b - a)
        floor[c] = (float(np.std(ds)) if ds else np.nan,
                    float(np.max(np.abs(ds))) if ds else np.nan)

    rows = []
    for c in present:
        o = obs[c]
        n_wt_rows = sum(int((f.cell == c).sum()) for f in wt)
        n_ko_rows = sum(int((f.cell == c).sum()) for f in ko)
        d_wt = rng.choice(pool_wt, (N_PERM, n_wt_rows)).mean(1)
        d_ko = rng.choice(pool_ko, (N_PERM, n_ko_rows)).mean(1)
        null1 = d_ko - d_wt
        p1 = float((null1 <= o).mean()) if o < 0 else float((null1 >= o).mean())
        good = null2[c][~np.isnan(null2[c])]
        p2 = float((np.abs(good) >= abs(o)).mean())
        pmax = float((colmax >= abs(o)).mean())
        sd, mx = floor[c]
        rows.append({"cell": c, "delta": o, "null1_mean": float(null1.mean()),
                     "p_identity": p1, "p_cohort": p2, "p_cohort_maxstat": pmax,
                     "wt_split_sd": sd, "wt_split_max": mx,
                     "inside_wt_noise": abs(o) <= mx if not np.isnan(mx) else None})
    res = pd.DataFrame(rows).sort_values("delta")

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.3f}".format)
    print("\n" + BAR)
    print("PER-CELL DELTAS AGAINST THREE NULLS")
    print(BAR)
    print(res.to_string(index=False))

    out = Path("ExpOutput/celegans_excess_avahiscl/intervention_null.csv")
    res.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("\ncells clearing p<0.05:")
    print(f"  identity null (is THIS cell special?)   : "
          f"{res[res.p_identity < 0.05].cell.tolist()}")
    print(f"  cohort null (does the split matter?)    : "
          f"{res[res.p_cohort < 0.05].cell.tolist()}")
    print(f"  cohort null, max-stat corrected         : "
          f"{res[res.p_cohort_maxstat < 0.05].cell.tolist()}")
    inside = res[res.inside_wt_noise == True].cell.tolist()  # noqa: E712
    print(f"  deltas inside the wild-type-only noise range: {inside}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
