"""Biomarker triage on a simulated randomised trial with known ground truth.

The deliverable is a shortlist, not a verdict.  Wet-lab validation and follow-up
trials are the expensive step, so the useful thing an observational method can
do is cut the candidate set and say which candidates are interchangeable --
four assays reading one pathway are one hypothesis, not four, and testing them
separately spends four budgets to answer one question.

Sections, in the order a trial statistician would want them:

1. The average treatment effect, which randomisation identifies on its own.
   Shown to mark the boundary: nothing else in this package improves on it.
2. A redundancy audit, which says where importance rankings are not identifiable.
3. The blindness itself, measured.
4. Effect modification, and how unstable the ranking is across replications --
   the mechanism behind companion diagnostics that do not replicate.
5. The triage: what to spend an experiment on, and what not to.

    python scripts/biomarker_triage.py --n 4000 --replications 8
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from deepfeatselect.biomarker import (
    estimate_ate,
    simulate_trial,
    stratified_effect,
)
from deepfeatselect.redundancy import equivalence_classes, redundancy_scores


def loco_importance(x: np.ndarray, y: np.ndarray, names: list[str], seed: int = 0) -> pd.DataFrame:
    """Leave-one-out refit importance: the drop in held-out R-squared."""
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=seed)

    def score(cols: list[int]) -> float:
        model = RandomForestRegressor(n_estimators=150, min_samples_leaf=5,
                                      random_state=seed, n_jobs=-1)
        model.fit(x_tr[:, cols], y_tr)
        return float(model.score(x_te[:, cols], y_te))

    full = score(list(range(x.shape[1])))
    rows = []
    for j, name in enumerate(names):
        cols = [c for c in range(x.shape[1]) if c != j]
        rows.append({"feature": name, "loco_r2_drop": full - score(cols)})
    return pd.DataFrame(rows).sort_values("loco_r2_drop", ascending=False).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--replications", type=int, default=8,
                   help="independent trials used to measure ranking stability")
    p.add_argument("--measurement-noise", type=float, default=0.05)
    p.add_argument("--outdir", default="ExpOutput/biomarker")
    args = p.parse_args()

    trial = simulate_trial(n=args.n, seed=0, measurement_noise=args.measurement_noise)
    names = trial.feature_names

    print("=" * 78)
    print(f"SIMULATED RANDOMISED TRIAL  n={args.n}  treatment randomised 1:1")
    print("=" * 78)

    print("\n1. AVERAGE TREATMENT EFFECT -- what the trial design already answers")
    print("-" * 78)
    effect, se, half = estimate_ate(trial)
    print(f"  true ATE                 {trial.true_ate:+.4f}")
    print(f"  difference in means      {effect:+.4f}  95% CI [{effect - half:+.4f}, {effect + half:+.4f}]")
    print(f"  z = {effect / se:.1f}, so the drug beats placebo on average.")
    print("  Randomisation identifies this directly. No causal discovery is")
    print("  involved and none would improve on it. Everything below concerns")
    print("  the different question of WHICH patients benefit and WHAT to target.")

    print("\n2. REDUNDANCY AUDIT -- where rankings stop being identifiable")
    print("-" * 78)
    audit = redundancy_scores(trial.x, names, seed=0)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(audit.to_string(index=False))
    classes = equivalence_classes(trial.x, names, seed=0)
    print("\n  mutually-determining classes (interchangeable as predictors):")
    for group in classes:
        print(f"    {{{', '.join(sorted(group))}}}")
    if not classes:
        print("    none found at the pairwise threshold")
    redundant = audit[audit.redundant].feature.tolist()
    print(f"\n  features whose individual importance is NOT identifiable: {len(redundant)}")
    print(f"    {', '.join(redundant) if redundant else '(none)'}")

    print("\n3. THE BLINDNESS, MEASURED -- leave-one-out importance for the outcome")
    print("-" * 78)
    loco = loco_importance(trial.x, trial.outcome, names, seed=0)
    with pd.option_context("display.float_format", "{:+.5f}".format):
        print(loco.to_string(index=False))
    marker_rows = loco[loco.feature.isin(trial.redundant_markers)]
    print(f"\n  the four pathway readouts score between "
          f"{marker_rows.loco_r2_drop.min():+.5f} and {marker_rows.loco_r2_drop.max():+.5f}")
    print("  -- all near zero, because each is reconstructible from the others.")
    print(f"  '{loco.iloc[0].feature}' tops the table and is measured AFTER treatment:")
    print("  a descendant of the outcome, invalid as a biomarker, and the single")
    print("  most predictive column in the study.")

    print("\n4. EFFECT MODIFICATION -- who benefits, and does the answer replicate?")
    print("-" * 78)
    candidates = [n for n in trial.pre_treatment_names if n not in trial.confounders]
    print("  (pre-treatment columns only; conditioning on post-treatment")
    print("   variables biases subgroup effects however predictive they look)\n")

    per_rep = []
    for rep in range(args.replications):
        rep_trial = simulate_trial(n=args.n, seed=rep, measurement_noise=args.measurement_noise)
        row = {"replication": rep}
        for name in candidates:
            _, _, gap = stratified_effect(rep_trial, rep_trial.column(name))
            row[name] = gap
        per_rep.append(row)
    gaps = pd.DataFrame(per_rep).set_index("replication")

    summary = pd.DataFrame({
        "mean_gap": gaps.mean(),
        "std_gap": gaps.std(ddof=1),
        "times_ranked_first": (gaps.abs().rank(axis=1, ascending=False) == 1).sum(),
    }).sort_values("mean_gap", ascending=False)
    with pd.option_context("display.float_format", "{:+.4f}".format):
        print(summary.to_string())

    winners = gaps.abs().idxmax(axis=1)
    print(f"\n  top-ranked marker across {args.replications} independent trials:")
    for name, count in winners.value_counts().items():
        print(f"    {name:<18} won {count}/{args.replications}")
    n_distinct = winners.nunique()
    print(f"\n  {n_distinct} different markers took first place in {args.replications} runs.")
    if n_distinct > 1:
        print("  A companion diagnostic built on the winner of any single trial")
        print("  would name a different assay depending on which trial ran.")

    print("\n5. TRIAGE -- what the data does and does not license")
    print("-" * 78)
    cheapest = min(trial.assay_cost, key=trial.assay_cost.get)
    print(f"  DIAGNOSTIC USE: any member of {{{', '.join(sorted(trial.redundant_markers))}}}")
    print(f"    They are interchangeable as predictors, so the choice is a")
    print(f"    cost and reliability decision, not a statistical one.")
    print(f"    Cheapest assay in the class: {cheapest} "
          f"({trial.assay_cost[cheapest]:.0f} vs {max(trial.assay_cost.values()):.0f}).")
    print(f"\n  DRUG TARGET: not identifiable from this data.")
    print(f"    The class members are readouts of a latent pathway. Intervening")
    print(f"    on a readout does not move the pathway that generates it, so a")
    print(f"    knockout of any one of them is predicted to do nothing.")
    print(f"\n  EXCLUDED: {', '.join(trial.post_treatment)} (measured post-randomisation)")
    print(f"\n  SEARCH-SPACE REDUCTION")
    n_candidates = len(candidates)
    n_hypotheses = len([c for c in candidates if c not in trial.redundant_markers]) + len(classes)
    print(f"    candidate columns:                    {n_candidates}")
    print(f"    distinct testable hypotheses:         {n_hypotheses}")
    print(f"    (the {len(trial.redundant_markers)} pathway readouts are ONE hypothesis, not "
          f"{len(trial.redundant_markers)};")
    print(f"     testing them separately spends {len(trial.redundant_markers)} experimental budgets")
    print(f"     to answer one question)")

    from pathlib import Path
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(outdir / "redundancy_audit.csv", index=False)
    loco.to_csv(outdir / "loco_importance.csv", index=False)
    gaps.to_csv(outdir / "effect_modification_replications.csv")
    summary.to_csv(outdir / "effect_modification_summary.csv")
    print(f"\nwrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
