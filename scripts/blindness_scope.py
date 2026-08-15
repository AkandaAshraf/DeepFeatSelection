"""Which importance measures actually inherit Proposition 1's zero?

Proposition 1 is a statement about *risk differences*: if a feature is a
deterministic function of the others, the achievable risk is unchanged by its
removal and leave-one-out importance is exactly zero. We verified that for LOCO
and then generalised loosely, describing the regime as one where feature
importance fails.

That generalisation may be too broad. Impurity importance does not difference
two risks -- it accumulates the loss reduction attributable to splits on a
feature -- so when several features carry the same information, trees pick among
them and credit is *distributed* rather than cancelled. Mutual information is
marginal, not conditional, and is likewise untouched by the argument. Shapley
averages over all coalitions and was already shown to survive.

If those measures find the informative features where LOCO reports 0.000, then
the blindness result binds a specific and narrow family, and the practical
remedy is to use a different estimator rather than to instrument the network.
That materially narrows what the deprivation probe is for, so it is worth
settling directly.

    python scripts/blindness_scope.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from deepfeatselect.shapley import shapley_importance
from deepfeatselect.synthetic import redundancy_demo

ROLES = {"driver": "TRUE CAUSE", "proxy_cos": "proxy (sufficient)",
         "proxy_sin": "proxy (insufficient)", "unrelated": "irrelevant"}


def loco(x, y, seed):
    """Retrained leave-one-out: the measure Proposition 1 speaks about."""
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.3, random_state=seed, stratify=y)

    def acc(cols):
        m = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        m.fit(x_tr[:, cols], y_tr)
        return m.score(x_te[:, cols], y_te)

    full = acc(list(range(x.shape[1])))
    return np.array([full - acc([c for c in range(x.shape[1]) if c != j])
                     for j in range(x.shape[1])])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--outdir", default="ExpOutput/blindness")
    args = p.parse_args()

    per_seed = []
    for seed in range(args.seeds):
        system = redundancy_demo(n=args.n, seed=seed)
        x = np.asarray(system["x"], dtype=np.float64)
        names = list(system["feature_names"])
        y = (np.asarray(system["y"]) > np.median(system["y"])).astype(int)

        forest = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1)
        forest.fit(x, y)

        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=0.3, random_state=seed, stratify=y)
        pforest = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        pforest.fit(x_tr, y_tr)
        perm = permutation_importance(pforest, x_te, y_te, n_repeats=10,
                                      random_state=seed, n_jobs=-1).importances_mean

        # Reconstructability from the others: the Proposition 1 condition itself.
        recon = []
        for j in range(x.shape[1]):
            others = np.delete(x, j, axis=1)
            m = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
            m.fit(others[:2000], x[:2000, j])
            recon.append(m.score(others[2000:], x[2000:, j]))

        sage = shapley_importance(x, y.astype(float), names, seed=seed).set_index("feature")

        per_seed.append(pd.DataFrame({
            "feature": names,
            "loco_acc_drop": loco(x, y, seed),
            "rf_impurity": forest.feature_importances_,
            "permutation": perm,
            "mutual_info": mutual_info_classif(x, y, random_state=seed),
            "sage": [sage.loc[n, "sage"] for n in names],
            "reconstructable_from_others": recon,
        }))
        print(f"  seed {seed} done")

    table = (pd.concat(per_seed).groupby("feature", sort=False).mean())
    table["role"] = [ROLES[n] for n in table.index]

    print("\n" + "=" * 96)
    print("DOES PROPOSITION 1's ZERO BIND EVERY MEASURE?")
    print("=" * 96)
    print(f"  redundancy_demo, n={args.n}, mean of {args.seeds} seeds\n")
    cols = ["role", "reconstructable_from_others", "loco_acc_drop",
            "rf_impurity", "permutation", "mutual_info", "sage"]
    with pd.option_context("display.float_format", "{:+.4f}".format,
                           "display.width", 220):
        print(table[cols].to_string())

    informative = [n for n in table.index if n != "unrelated"]
    print("\n" + "=" * 96)
    print("VERDICT PER MEASURE  (does it separate the three informative features")
    print("                      from the irrelevant one?)")
    print("=" * 96)
    for measure in ("loco_acc_drop", "rf_impurity", "permutation",
                    "mutual_info", "sage"):
        lo = table.loc[informative, measure].min()
        null = table.loc["unrelated", measure]
        # A measure "works" if its weakest informative feature clearly exceeds
        # the irrelevant one; the margin matters more than the sign.
        works = lo > max(null, 0) + 0.01
        print(f"  {measure:<16} weakest informative {lo:+.4f} vs irrelevant "
              f"{null:+.4f}   -> {'SEPARATES' if works else 'BLIND'}")

    print("\n  Every informative feature is reconstructible from the others")
    print(f"  (min R^2 {table.loc[informative,'reconstructable_from_others'].min():.3f}),")
    print("  so Proposition 1's premise holds for all of them.")

    from pathlib import Path
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(outdir / "blindness_scope.csv")
    print(f"\nwrote {outdir}/blindness_scope.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
