"""Does Shapley importance inherit the Proposition 1 zero? Measured, not assumed.

Leave-one-out is the ``S = N \\ {j}`` coalition alone, and Proposition 1 sends it
to zero under deterministic redundancy.  Shapley averages that coalition
together with every other one, so the prediction is that it survives.  Run on
the two systems where ground truth is known.

    python scripts/sage_comparison.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from deepfeatselect.shapley import loco_for_comparison, shapley_importance
from deepfeatselect.synthetic import nonlinear_scm, redundancy_demo


def compare(x, y, names, title, roles=None, seed=0):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

    sage = shapley_importance(x, y, names, seed=seed).set_index("feature")
    loco = loco_for_comparison(x, y, names, seed=seed).set_index("feature")
    table = sage.join(loco)
    if roles:
        table["role"] = [roles.get(n, "") for n in table.index]
    table["sage_rank"] = table.sage.rank(ascending=False).astype(int)

    with pd.option_context("display.float_format", "{:+.4f}".format):
        print(table.to_string())

    informative = table[table.get("role", pd.Series("", index=table.index)) != "irrelevant"]
    print(f"\n  method: {sage.method.iloc[0]}")
    print(f"  max |LOCO| over informative features: "
          f"{informative.loco.abs().max():+.5f}")
    print(f"  min  SAGE over informative features:  "
          f"{informative.sage.min():+.5f}")
    return table


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="ExpOutput/sage")
    args = p.parse_args()

    demo = redundancy_demo(n=args.n, seed=args.seed)
    x = np.asarray(demo["x"], dtype=np.float64)
    y = (np.asarray(demo["y"]) > np.median(demo["y"])).astype(np.float64)
    roles = {"driver": "TRUE CAUSE", "proxy_cos": "proxy (sufficient)",
             "proxy_sin": "proxy (not sufficient)", "unrelated": "irrelevant"}
    demo_table = compare(x, y, list(demo["feature_names"]),
                         "redundancy_demo -- every informative feature has LOCO = 0",
                         roles=roles, seed=args.seed)

    scm = nonlinear_scm(n=args.n, seed=args.seed)
    scm_roles = {}
    for i, name in enumerate(scm.feature_names):
        if scm.direct_causes[i]:
            scm_roles[name] = "direct cause"
        elif scm.effects[i]:
            scm_roles[name] = "EFFECT of target"
        elif scm.irrelevant[i]:
            scm_roles[name] = "irrelevant"
        elif scm.confounded[i]:
            scm_roles[name] = "confounded"
    scm_table = compare(scm.x, scm.y.astype(np.float64), scm.feature_names,
                        "nonlinear_scm -- does SAGE separate causes from effects?",
                        roles=scm_roles, seed=args.seed)

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    inf_demo = demo_table[demo_table.role != "irrelevant"]
    blind = inf_demo.loco.abs().max() < 0.01
    survives = inf_demo.sage.min() > 0.05
    print(f"  LOCO blind on redundancy_demo:      {blind} "
          f"(max |LOCO| = {inf_demo.loco.abs().max():.5f})")
    print(f"  SAGE survives the same redundancy:  {survives} "
          f"(min SAGE = {inf_demo.sage.min():.4f})")
    if survives:
        print("\n  => Shapley importance does NOT inherit the Proposition 1 zero.")
        print("     Prop 1 concerns one coalition; Shapley averages over all of them.")
        top = inf_demo.sage.idxmax()
        gap = inf_demo.sage.max() - inf_demo.sage.nlargest(2).iloc[-1]
        print(f"  => but its top feature is '{top}' with a margin of {gap:+.4f}")
        print("     over the runner-up: solving blindness is not solving orientation.")

    effect_rank = int(scm_table.loc[scm_table.role == "EFFECT of target", "sage_rank"].iloc[0])
    print(f"\n  SAGE rank of the target's CHILD on nonlinear_scm: "
          f"{effect_rank} of {len(scm_table)}")

    from pathlib import Path
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    demo_table.to_csv(outdir / "sage_redundancy_demo.csv")
    scm_table.to_csv(outdir / "sage_nonlinear_scm.csv")
    print(f"\nwrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
