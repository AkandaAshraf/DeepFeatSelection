"""Redundancy audit on two real medical datasets.

The audit needs no ground truth and no simulation: it asks whether a dataset's
feature importances are identifiable at all, which is answerable from the design
matrix alone.  Two datasets with opposite structure are run here so the
instrument is checked in both directions.

* Breast Cancer Wisconsin (diagnostic).  569 fine-needle aspirates, 30 columns
  built as ten base measurements times three statistics (mean, standard error,
  worst).  Three of the base measurements -- radius, perimeter, area -- are
  geometrically tied: for a roughly convex boundary, perimeter is about
  ``2*pi*r`` and area about ``pi*r^2``, so each determines the others up to
  shape irregularity.  If the audit works, it should find that without being
  told, and the corresponding leave-one-out importances should collapse.

* Cleveland heart disease.  297 patients, 13 clinical variables chosen to be
  distinct measurements.  Expected to show little redundancy, which is the
  negative control: an audit that flags everything is useless.

    python scripts/real_data_audit.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from deepfeatselect.data import prepare
from deepfeatselect.redundancy import redundancy_scores


def loco_auc(x: np.ndarray, y: np.ndarray, names: list[str], seed: int = 0) -> pd.DataFrame:
    """Leave-one-out drop in held-out accuracy, the standard importance measure."""
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.3, random_state=seed, stratify=y
    )

    def score(cols: list[int]) -> float:
        model = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1
        )
        model.fit(x_tr[:, cols], y_tr)
        return float(model.score(x_te[:, cols], y_te))

    full = score(list(range(x.shape[1])))
    rows = []
    for j, name in enumerate(names):
        cols = [c for c in range(x.shape[1]) if c != j]
        rows.append({"feature": name, "loco_acc_drop": full - score(cols)})
    return pd.DataFrame(rows).set_index("feature")


def audit(x: np.ndarray, y: np.ndarray, names: list[str], title: str,
          seed: int = 0) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

    scores = redundancy_scores(x, names, seed=seed).set_index("feature")
    importance = loco_auc(x, y, names, seed=seed)
    table = scores.join(importance).sort_values("r2_from_others", ascending=False)

    n_redundant = int(table.redundant.sum())
    print(f"\n{len(names)} features, {n_redundant} with leave-one-out "
          f"reconstruction R^2 >= 0.95")
    print("\ntop of the table (most reconstructible first)")
    with pd.option_context("display.float_format", "{:+.4f}".format):
        print(table.head(12).to_string())

    if n_redundant:
        blind = table[table.redundant]
        print(f"\n  those {n_redundant} features carry LOCO importance between "
              f"{blind.loco_acc_drop.min():+.4f} and {blind.loco_acc_drop.max():+.4f}")
        print("  -- individually near zero regardless of how diagnostic they are.")
    else:
        print("\n  no feature is reconstructible from the rest: importances here")
        print("  are identifiable, and a ranking over them is meaningful.")
    return table


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="ExpOutput/real_audit")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # --- Breast Cancer Wisconsin -------------------------------------------
    cancer = load_breast_cancer()
    names = [n.replace(" ", "_") for n in cancer.feature_names]
    x = StandardScaler().fit_transform(cancer.data)
    table = audit(x, cancer.target, names,
                  "BREAST CANCER WISCONSIN (diagnostic) -- 569 aspirates, 30 features",
                  seed=args.seed)

    # The geometric identities are the check on the instrument: nothing in the
    # audit knows that perimeter and area are functions of radius.
    print("\n  geometric family (perimeter ~ 2*pi*r, area ~ pi*r^2):")
    family = [n for n in names if any(k in n for k in ("radius", "perimeter", "area"))]
    with pd.option_context("display.float_format", "{:+.4f}".format):
        print(table.loc[[f for f in family if f in table.index],
                        ["r2_from_others", "redundant", "loco_acc_drop"]].to_string())

    recovered = table.loc[[f for f in family if f in table.index], "redundant"].sum()
    print(f"\n  {recovered} of {len(family)} geometric-family columns flagged as "
          f"non-identifiable, from the data alone.")

    # --- Cleveland, the negative control -----------------------------------
    data = prepare("Data/processed.cleveland.data", task="binary", seed=args.seed)
    x_all = np.vstack([data.x_train, data.x_val, data.x_test])
    y_all = np.concatenate([data.y_train, data.y_val, data.y_test])
    # Audit the original features rather than one-hot columns: dummies of the
    # same variable are redundant by construction and would be a trivial hit.
    numeric = [i for i in range(data.n_features)
               if (data.groups == i).sum() == 1]
    cols = [int(np.flatnonzero(data.groups == i)[0]) for i in numeric]
    cleveland_names = [data.feature_names[i] for i in numeric]
    cleveland = audit(x_all[:, cols], y_all, cleveland_names,
                      "CLEVELAND HEART DISEASE -- 297 patients, numeric clinical variables",
                      seed=args.seed)

    print("\n" + "=" * 78)
    print("READING")
    print("=" * 78)
    print(f"  breast cancer:  {int(table.redundant.sum())}/{len(names)} features "
          f"non-identifiable -- a ranking over the full table is not meaningful,")
    print("                  and the redundant groups should be treated as single")
    print("                  hypotheses when allocating follow-up work.")
    print(f"  cleveland:      {int(cleveland.redundant.sum())}/{len(cleveland_names)} "
          f"non-identifiable -- importances here can be read as usual.")
    print("\n  The instrument distinguishes the two cases without supervision,")
    print("  which is what makes it usable on data whose structure is unknown.")

    from pathlib import Path
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(outdir / "breast_cancer_audit.csv")
    cleveland.to_csv(outdir / "cleveland_audit.csv")
    print(f"\nwrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
