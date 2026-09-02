"""Is CCM's membership collapse the method's fault, or our aggregation's?

Pre-registration: paper/aggregation_check_protocol.md, committed before any
alternative was scored. Reuses the edge matrices already saved by the V=60
and V=30 runs -- no new edge scoring, so this costs minutes rather than the
334 minutes those runs took.

The V=60 result attributed CCM's collapse to max-over-incoming amplifying a
single spurious edge per node. That was stated as a caveat rather than
tested. Four aggregations, declared in advance:

    MAX       max_p score[p -> q]                    (the incumbent)
    MEAN      mean_p score[p -> q]
    COUNT     #{p : score[p -> q] > 90th pct of the scan's own edges}
    TOP2MEAN  mean of the two largest incoming edges

    python scripts/aggregation_check.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

V60 = Path("ExpOutput/ccm_pcmci_v60")
V30 = Path("ExpOutput/ccm_pcmci_baseline")
OUT = Path("ExpOutput/aggregation_check")
MACE_V60_MEDIAN = 0.976


def auc(pos, neg):
    """Identical to error_metrics.auc; see scripts/test_auc_identical.py."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def _incoming(mat):
    """Columns of off-diagonal incoming edges, NaN on the diagonal."""
    m = mat.astype(float).copy()
    np.fill_diagonal(m, np.nan)
    return m


def agg_max(mat):
    return np.nanmax(_incoming(mat), axis=0)


def agg_mean(mat):
    return np.nanmean(_incoming(mat), axis=0)


def agg_count(mat):
    m = _incoming(mat)
    t = np.nanpercentile(m, 90)          # the scan's own edges set the bar
    return np.nansum(m > t, axis=0).astype(float)


def agg_top2mean(mat):
    m = _incoming(mat)
    s = np.sort(m, axis=0)               # NaNs sort to the end
    return np.nanmean(s[-3:-1, :], axis=0)


AGGS = {"MAX": agg_max, "MEAN": agg_mean, "COUNT": agg_count,
        "TOP2MEAN": agg_top2mean}


def score(mat, is_driven, is_source):
    return {k: auc(f(mat)[is_driven], f(mat)[is_source])
            for k, f in AGGS.items()}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    # ---- V=60, three seeds, CCM and PCMCI ------------------------------
    for seed in (0, 1, 2):
        sysd = np.load(V60 / f"system_s{seed}.npz")
        is_driven, is_source = sysd["is_driven"], sysd["is_source"]
        for meth, f in (("CCM", V60 / f"ccm_s{seed}.npz"),
                        ("PCMCI", V60 / f"pcmci_s{seed}.npz")):
            if not f.exists():
                continue
            mat = np.load(f)["ccm" if meth == "CCM" else "pcmci"]
            for a, v in score(mat, is_driven, is_source).items():
                rows.append({"V": 60, "seed": seed, "method": meth,
                             "aggregation": a, "auroc": v})

    # ---- V=30 cross-check (G4): a winner must not be width-specific ----
    f30 = V30 / "matrices.npz"
    if f30.exists():
        z = np.load(f30)
        is_driven, is_source = z["is_driven"], z["is_source"]
        for meth, key, transpose in (("CCM", "ccm", False),
                                     ("PCMCI", "pcmci", True)):
            mat = z[key]
            if transpose:
                mat = mat.T          # undo the V=30 run's transpose bug
            for a, v in score(mat, is_driven, is_source).items():
                rows.append({"V": 30, "seed": 0, "method": meth,
                             "aggregation": a, "auroc": v})

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "results.csv", index=False)

    for V in (60, 30):
        sub = d[d.V == V]
        if sub.empty:
            continue
        print(f"\nV={V}: membership AUROC by aggregation "
              f"({'median over 3 seeds' if V == 60 else 'single seed'})")
        piv = sub.pivot_table(index="method", columns="aggregation",
                              values="auroc", aggfunc="median")
        piv = piv[["MAX", "MEAN", "COUNT", "TOP2MEAN"]]
        print("   " + piv.round(3).to_string().replace("\n", "\n   "))

    ccm60 = d[(d.V == 60) & (d.method == "CCM")]
    med = ccm60.groupby("aggregation").auroc.median()
    best_alt = med.drop("MAX").idxmax()
    print(f"\nMACE median at V=60: {MACE_V60_MEDIAN:.3f}")
    print(f"incumbent MAX:       {med['MAX']:.3f}")
    print(f"best alternative:    {best_alt} {med[best_alt]:.3f}")

    print("\nVERDICT (rule fixed before running)")
    print(f"   G1 an alternative beats MAX ({med['MAX']:.3f}): "
          f"{'YES -> ' + best_alt if med[best_alt] > med['MAX'] else 'NO'}")
    if med[best_alt] >= MACE_V60_MEDIAN:
        print(f"   G2 CLAIM WEAKENED: {best_alt} reaches "
              f"{med[best_alt]:.3f} >= MACE {MACE_V60_MEDIAN:.3f}. CCM's "
              "collapse was our\n      adapter, not the method. The V=60 "
              "section must be rewritten.")
    else:
        print(f"   G2 CAVEAT DISCHARGED: best alternative {med[best_alt]:.3f} "
              f"< MACE {MACE_V60_MEDIAN:.3f}.\n      Three alternatives "
              "tried; the edges-to-membership conversion still fails.")

    if 30 in d.V.values:
        m30 = d[(d.V == 30) & (d.method == "CCM")].set_index(
            "aggregation").auroc
        print(f"   G4 at V=30, {best_alt} scores {m30[best_alt]:.3f} vs MAX "
              f"{m30['MAX']:.3f}: "
              f"{'consistent' if m30[best_alt] >= m30['MAX'] - 0.05 else 'WIDTH-SPECIFIC, reported as tuning'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
