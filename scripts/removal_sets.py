"""Set-level redundancy analysis across the benchmark and real datasets.

Single-feature ablation is provably uninformative under redundancy.  The
constructive question is what the *smallest set* is whose joint removal actually
destroys the signal, because that set is the coarsest grouping at which ablation
importance becomes meaningful again -- and because its size counts the distinct
routes the model has to the target.

Run on four datasets with different structure so the numbers are interpretable:
two synthetic ones where the answer is known in advance, one real dataset with
heavy redundancy, and one real dataset with almost none.

    python scripts/removal_sets.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from deepfeatselect.data import prepare
from deepfeatselect.redundancy import (
    equivalence_classes,
    group_loco,
    minimal_removal_set,
    redundancy_scores,
)
from deepfeatselect.synthetic import nonlinear_scm, redundancy_demo


def analyse(x, y, names, title, min_drop, max_size, expected=None, seed=0):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

    audit = redundancy_scores(x, names, seed=seed)
    n_redundant = int(audit.redundant.sum())
    print(f"  {len(names)} features, {n_redundant} individually redundant "
          f"(R^2 from others >= 0.95)")

    classes = equivalence_classes(x, names, seed=seed)
    if classes:
        for group in classes:
            print(f"    interchangeable: {{{', '.join(sorted(group))}}}")
    else:
        print("    no mutually-determining classes")

    removal, drop = minimal_removal_set(
        x, y, names, min_drop=min_drop, max_size=max_size, seed=seed
    )
    print(f"\n  minimal removal set (target drop >= {min_drop}):")
    if drop >= min_drop:
        print(f"    {{{', '.join(sorted(removal))}}}  -> R^2 drop {drop:+.4f}")
        print(f"    size {len(removal)}: the target is reachable "
              f"{len(removal)} independent way(s)")
    else:
        print(f"    none up to size {max_size}; best was "
              f"{{{', '.join(sorted(removal))}}} at {drop:+.4f}")
    if expected is not None:
        match = set(removal) == set(expected)
        print(f"    ground truth {{{', '.join(sorted(expected))}}} -- "
              f"{'MATCH' if match else 'MISS'}")

    # The repair, side by side: each member alone against the whole set.
    if drop >= min_drop and len(removal) > 1:
        singles = group_loco(x, y, names, [[n] for n in removal], seed=seed)
        grouped = group_loco(x, y, names, [removal], seed=seed)
        print(f"\n  single-feature LOCO within the set: "
              f"max {singles.r2_drop.abs().max():+.4f}")
        print(f"  group LOCO on the whole set:        {grouped.iloc[0].r2_drop:+.4f}")
        ratio = grouped.iloc[0].r2_drop / max(singles.r2_drop.abs().max(), 1e-9)
        print(f"  ratio: {ratio:.0f}x -- importance is defined for the group, "
              f"not its members")

    return {"dataset": title.split(" --")[0], "n_features": len(names),
            "n_redundant": n_redundant, "removal_set": " + ".join(sorted(removal)),
            "removal_size": len(removal), "drop": drop}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="ExpOutput/removal_sets")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = []

    # 1. The benchmark, where the answer is recorded by the generator.
    demo = redundancy_demo(n=1500, seed=args.seed)
    x = np.asarray(demo["x"], dtype=np.float64)
    y = (np.asarray(demo["y"]) > np.median(demo["y"])).astype(np.float64)
    rows.append(analyse(x, y, list(demo["feature_names"]),
                        "redundancy_demo -- chaotic driver with two proxies",
                        min_drop=0.10, max_size=3,
                        expected=["driver", "proxy_cos"], seed=args.seed))

    # 2. A DAG with a confounder, a child of the target, and pure noise.
    scm = nonlinear_scm(n=1500, seed=args.seed)
    rows.append(analyse(scm.x, scm.y.astype(np.float64), scm.feature_names,
                        "nonlinear_scm -- confounded DAG with an effect column",
                        min_drop=0.05, max_size=3, seed=args.seed))

    # 3. Real data, heavy redundancy.
    cancer = load_breast_cancer()
    names = [n.replace(" ", "_") for n in cancer.feature_names]
    rows.append(analyse(StandardScaler().fit_transform(cancer.data),
                        cancer.target.astype(np.float64), names,
                        "breast cancer wisconsin -- 569 aspirates, 30 features",
                        min_drop=0.05, max_size=3, seed=args.seed))

    # 4. Real data, almost none.
    data = prepare("Data/processed.cleveland.data", task="binary", seed=args.seed)
    x_all = np.vstack([data.x_train, data.x_val, data.x_test])
    y_all = np.concatenate([data.y_train, data.y_val, data.y_test]).astype(np.float64)
    numeric = [i for i in range(data.n_features) if (data.groups == i).sum() == 1]
    cols = [int(np.flatnonzero(data.groups == i)[0]) for i in numeric]
    rows.append(analyse(x_all[:, cols], y_all, [data.feature_names[i] for i in numeric],
                        "cleveland -- 297 patients, numeric clinical variables",
                        min_drop=0.05, max_size=3, seed=args.seed))

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    summary = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_colwidth", 44):
        print(summary.to_string(index=False))

    from pathlib import Path
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / "removal_sets.csv", index=False)
    print(f"\nwrote {outdir}/removal_sets.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
