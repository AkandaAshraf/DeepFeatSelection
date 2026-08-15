"""Per-statistic verdicts from the curve-discriminator run.

Each of the fourteen curve statistics is judged on its own against three
criteria, in increasing order of what it would take to matter:

1. separates cause from effect at all (AUROC above chance);
2. beats the incumbent, ``final_loss``, on that comparison -- otherwise the
   trajectory has told us nothing the endpoint did not;
3. drives P(effect ranks first) below its null.

Only the third corresponds to a practical improvement. Seven existing methods
score 1.0 there, and a statistic can pass the first two while still putting the
effect on top, which would leave the failure exactly where it was.

Replication across both systems is required. Fourteen statistics against three
comparisons is forty-two tests and something will clear any single threshold by
chance, so a hit on one system alone is reported as noise rather than as a
finding.

    python scripts/curve_verdicts.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Share of informative features that are effects, i.e. the rate at which an
# uninformative statistic puts an effect first. Not 0.5, and not the same for
# both systems, because the role counts differ.
NULLS = {"scm": 1 / 4, "demo": 2 / 3}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--indir", default="ExpOutput/curves")
    args = p.parse_args()

    indir = Path(args.indir)
    auroc = pd.read_csv(indir / "curve_auroc.csv", index_col=0)
    first = pd.read_csv(indir / "curve_effect_first.csv", index_col=0)

    baseline = auroc.loc["final_loss", "scm_cause_vs_effect"]

    table = pd.DataFrame({
        "auroc_scm": auroc["scm_cause_vs_effect"],
        "auroc_demo": auroc["demo_cause_vs_effect"],
        "auroc_noise": auroc["scm_cause_vs_noise"],
        "effect_first_scm": first["scm_effect_first"],
        "effect_first_demo": first["demo_effect_first"],
    })

    table["separates"] = table.auroc_scm > 0.5
    table["beats_final_loss"] = table.auroc_scm > baseline
    table["below_null_scm"] = table.effect_first_scm < NULLS["scm"]
    table["below_null_demo"] = table.effect_first_demo < NULLS["demo"]
    table["replicates"] = table.below_null_scm & table.below_null_demo

    def verdict(row) -> str:
        if row.replicates:
            return "CAUSAL SIGNAL"
        if row.below_null_scm or row.below_null_demo:
            return "one system only (noise)"
        if row.beats_final_loss:
            return "better ranking, effect still first"
        return "nothing"

    table["verdict"] = table.apply(verdict, axis=1)
    table = table.sort_values(["replicates", "beats_final_loss", "auroc_scm"],
                              ascending=False)

    print("=" * 96)
    print("PER-STATISTIC VERDICTS")
    print("=" * 96)
    print(f"  incumbent: final_loss AUROC {baseline:.3f} on cause-vs-effect")
    print(f"  nulls for P(effect first): scm {NULLS['scm']:.3f}, demo {NULLS['demo']:.3f}")
    print(f"  every existing method scores 1.000 on both\n")

    cols = ["auroc_scm", "auroc_demo", "auroc_noise",
            "effect_first_scm", "effect_first_demo", "verdict"]
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 220):
        print(table[cols].to_string())

    winners = table[table.replicates]
    print("\n" + "=" * 96)
    if len(winners):
        print(f"{len(winners)} statistic(s) push the effect off the top on BOTH systems:")
        for name, row in winners.iterrows():
            print(f"  {name}: effect first {row.effect_first_scm:.3f} (null "
                  f"{NULLS['scm']:.3f}) and {row.effect_first_demo:.3f} (null "
                  f"{NULLS['demo']:.3f})")
        print("\n  Before believing this: check it against the usable-information")
        print("  reading. The effect is also the cheapest feature to read here, so a")
        print("  statistic that demotes it may be tracking cost rather than causal")
        print("  role. The reversal manipulation separates the two.")
    else:
        print("No statistic pushes the effect off the top on both systems.")
        print("Prediction P4 holds: the loss trajectory carries no causal information")
        print("beyond its endpoint, and this route to orientation is closed.")
        better = table[table.beats_final_loss]
        if len(better):
            print(f"\n{len(better)} statistic(s) do rank better than final_loss while")
            print("still putting the effect first -- a better ordering of the same")
            print("mistake, which is not an improvement in the sense that matters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
