"""One table: every dataset audited so far, same statistics, same thresholds.

The individual audits each answer "is this ranking identifiable", but the
interesting result only appears when they are put beside each other. Ordered by
how the columns came to exist rather than by subject matter, they separate
cleanly, and the separation is the point: deterministic redundancy tracks
*derivation*, not causal or biological relatedness.

Two statistics per dataset:

* worst-case reconstructability -- the largest R^2 of predicting one column from
  the others, i.e. how close the table comes to Proposition 1's condition;
* effective dimensions -- the participation ratio of the correlation spectrum,
  which counts directions actually occupied rather than columns present, and
  unlike equivalence classes registers many-to-one redundancy.

    python scripts/audit_comparison.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from deepfeatselect.data import prepare
from deepfeatselect.redundancy import redundancy_scores
from deepfeatselect.synthetic import redundancy_demo


def effective_dimensions(x: np.ndarray) -> tuple[float, int]:
    """Participation ratio of the correlation spectrum, and 95%-variance count."""
    corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    eig = np.clip(np.linalg.eigvalsh(corr), 0.0, None)
    if eig.sum() <= 0:
        return 0.0, 0
    participation = eig.sum() ** 2 / (eig**2).sum()
    ordered = np.sort(eig)[::-1]
    n_95 = int(np.searchsorted(np.cumsum(ordered) / ordered.sum(), 0.95) + 1)
    return float(participation), n_95


def summarise(name: str, origin: str, x: np.ndarray, names: list[str],
              seed: int = 0) -> dict:
    audit = redundancy_scores(x, names, seed=seed)
    pr, n_95 = effective_dimensions(x)
    counts = audit.verdict.value_counts()
    return {
        "dataset": name,
        "origin": origin,
        "n_rows": x.shape[0],
        "n_cols": x.shape[1],
        "max_r2": audit.r2_from_others.max(),
        "mean_r2": audit.r2_from_others.mean(),
        "not_identifiable": int(counts.get("not_identifiable", 0)),
        "partial": int(counts.get("partially_redundant", 0)),
        "identifiable": int(counts.get("identifiable", 0)),
        "eff_dims": pr,
        "n_pc_95": n_95,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--depmap", default="Data/depmap/CRISPRGeneEffect.csv")
    p.add_argument("--variants", default="Data/variants/myvariant_sample.csv")
    p.add_argument("--outdir", default="ExpOutput")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = []

    # Constructed to be exactly redundant: the ceiling case.
    demo = redundancy_demo(n=1500, seed=args.seed)
    rows.append(summarise("redundancy_demo", "simulated, exact",
                          np.asarray(demo["x"], dtype=np.float64),
                          list(demo["feature_names"]), args.seed))

    # Derived by formula: area and perimeter are functions of radius.
    cancer = load_breast_cancer()
    rows.append(summarise(
        "breast_cancer_wisconsin", "derived (geometry)",
        StandardScaler().fit_transform(cancer.data),
        [n.replace(" ", "_") for n in cancer.feature_names], args.seed))

    # Derived by training: ensembles are fitted functions of their neighbours.
    variants = Path(args.variants)
    if variants.exists():
        frame = pd.read_csv(variants)
        cols = [c for c in frame.columns if c != "label"]
        cover = frame[cols].notna().mean()
        keep = [c for c in cols if cover[c] >= 0.80]
        frame = frame.dropna(subset=keep)
        rows.append(summarise("clinvar_predictors", "derived (trained)",
                              frame[keep].to_numpy(dtype=np.float64), keep, args.seed))
    else:
        print(f"  skipping variants: {variants} not found")

    # Separately measured, shared biology: correlation with a noise floor.
    depmap = Path(args.depmap)
    if depmap.exists():
        from depmap_audit import COMPLEXES, CONTROLS, load_matrix
        panel = [g for genes in COMPLEXES.values() for g in genes] + CONTROLS
        data = load_matrix(depmap, panel)
        rows.append(summarise("depmap_crispr", "measured (shared cause)",
                              data.to_numpy(dtype=np.float64),
                              list(data.columns), args.seed))
    else:
        print(f"  skipping depmap: {depmap} not found")

    # Distinct clinical measurements: the negative control.
    clev = prepare("Data/processed.cleveland.data", task="binary", seed=args.seed)
    x_all = np.vstack([clev.x_train, clev.x_val, clev.x_test])
    numeric = [i for i in range(clev.n_features) if (clev.groups == i).sum() == 1]
    cols = [int(np.flatnonzero(clev.groups == i)[0]) for i in numeric]
    rows.append(summarise("cleveland_heart", "measured (distinct)",
                          x_all[:, cols], [clev.feature_names[i] for i in numeric],
                          args.seed))

    table = pd.DataFrame(rows).sort_values("max_r2", ascending=False)

    print("\n" + "=" * 100)
    print("REDUNDANCY AUDIT ACROSS EVERY DATASET, ORDERED BY HOW THE COLUMNS AROSE")
    print("=" * 100)
    show = table[["dataset", "origin", "n_rows", "n_cols", "max_r2", "mean_r2",
                  "not_identifiable", "partial", "identifiable", "eff_dims"]]
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(show.to_string(index=False))

    print("\ncolumns occupied vs columns present")
    for _, r in table.iterrows():
        bar = "#" * max(1, int(round(20 * r.eff_dims / r.n_cols)))
        print(f"  {r.dataset:<24} {r.eff_dims:5.1f} / {r.n_cols:<3.0f} "
              f"({r.eff_dims / r.n_cols:5.1%}) {bar}")

    print("\nreading")
    print("  Derived columns -- computed from their neighbours by formula or by a")
    print("  fitted model -- reach the non-identifiable band. Separately measured")
    print("  columns do not, however tightly the underlying biology couples them,")
    print("  because independent measurement error cannot be predicted away and")
    print("  puts a floor under the residual. Causal relatedness is not the test;")
    print("  derivation is.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(outdir / "audit_comparison.csv", index=False)
    print(f"\nwrote {outdir}/audit_comparison.csv")
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
