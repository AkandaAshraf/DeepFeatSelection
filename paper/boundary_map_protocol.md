# Pre-registration: a boundary map for MACE on known ground truth

Declared 2026-08-20, before the experiment was written or run.

## Why this, and why not the survey it replaces

The first proposal was a survey of public datasets for *premise satisfaction*
- length, self-baseline saturation, ghost cleanliness. An adversarial pass
killed it on this project's own evidence: the intracranial EEG cohort
satisfied every premise (29/29 subjects clean on the ghost gate, the best
saturation ever measured on real data, the donor rule engaged) and then
failed a pre-registered replication. **Premise satisfaction did not predict
validity.** A survey ranking datasets by premises would have put iEEG first.

Two further objections stood: the dataset pool would be chosen by us, so the
survey would measure our priors; and the gate is one of the surveyed
quantities, so excluding failures and reporting on survivors is circular.

The replacement measures what matters and can be measured: **precision and
recall against known ground truth, as a function of data properties.** Real
datasets can then be located within the map rather than ranked by proxy.

## The design

Synthetic systems where membership is known by construction. Four axes,
varied one at a time from a fixed centre so that each effect is attributable:

  n           samples per channel: 1000, 2000, 4000, 8000, 16000
  V           channels: 15, 30, 60, 120, 240
  coupling    drive strength: 0.05, 0.10, 0.20, 0.35, 0.50
  redundancy  duplicate channels carrying the same driver signal with added
              noise: 0, 1, 2, 4, 8 duplicates per driver

CENTRE: n = 4000, V = 60, coupling = 0.20, redundancy = 0. Three seeds per
cell; report the median and the spread.

## What is measured, per cell

  precision   of channels flagged above the ghost threshold, the fraction
              genuinely driven
  recall      of genuinely driven channels, the fraction flagged
  ghost       the ghost channel's excess, which must stay near zero
              everywhere; a cell where it does not is reported as VOID and
              its precision/recall are not interpreted
  saturation  the fraction of channels whose self-baseline exceeds 0.9, the
              premise Proposition 1 requires
  source_fp   the fraction of genuine SOURCES wrongly flagged as driven. The
              companion paper reports 0 of 63 across its systems; this
              tests whether that survives across the grid.

## Predictions, fixed before running

  B1  Recall rises with n and falls with redundancy. The redundancy effect
      is the sharper prediction: duplicated drivers should suppress recall
      because the code reaches the target through the duplicate, which is
      Takens redundancy acting exactly as the companion paper's Mechanism 1
      describes.
  B2  Precision stays high across the whole grid. The method is claimed to
      be precision-first; if precision degrades anywhere the claim is
      qualified by that region.
  B3  Saturation falls as V rises at fixed bottleneck, following the
      compression result already measured (ledger, Proposition 2 entry).
  B4  source_fp stays near zero everywhere. Sources are invisible by design;
      a region where they are flagged would be a serious defect and is the
      single most damaging thing this map could find.
  B5  NO PREDICTION is made for where recall becomes unusable. Locating that
      frontier is the point of the experiment.

## What would make this actionable

The output is a table a practitioner can consult before deploying: at your
n, V and expected redundancy, this is the precision and recall to expect,
and this is whether the theory's saturation premise is likely to hold. That
is the deliverable, whatever the numbers are.

## Honest limits, declared now

Synthetic systems are not real ones. Coupled logistic maps have clean
Takens structure, low noise and stationary dynamics; real recordings have
none of those reliably. The map bounds performance from ABOVE: where it
shows poor recall, real data will not be better. Where it shows good recall,
real data may still fail for reasons this design cannot produce - as the
iEEG replication demonstrated.

## Void conditions

Void if axes, centre, metrics or predictions are altered after any result is
seen, or if cells with a failing ghost are interpreted rather than reported
as void.

---

## Result (2026-08-20)

51 cells, 3 seeds each, 11.8 min. `ExpOutput/boundary_map/boundary_map.csv`.
No cell was ghost-void; the ghost panel stayed clean across the entire grid.

### Against the declared predictions

  B1 recall rises with n            FAILED - erratic, non-monotonic
  B1 recall falls with redundancy   FAILED - it ROSE (axis mis-implemented,
                                    see below)
  B2 precision high everywhere      HELD - 1.00 everywhere except 0.95 at
                                    redundancy 4-8
  B3 saturation falls as V rises    FAILED - our conceptual error, see below
  B4 source false positives ~0      HELD - 0.000 in EVERY cell of the grid
  B5 locate the recall frontier     FOUND - two frontiers, below

### B4 is the strongest result

The source false-positive rate is exactly 0.000 in all 51 cells, across
every n, V, coupling and redundancy tested. The companion paper reports 0 of
63 sources flagged across its systems; that now holds across a systematic
grid. The designed blindness to sources is the most robust property measured.

### Frontier 1: detections are capped, so recall is roughly constant/V

  V      flagged   driven   recall
   15      12        12      1.00
   30      22        25      0.88
   60       9        50      0.18
  120      23       100      0.23
  240      27       200      0.14

The ABSOLUTE number of channels surfaced stays in the 9-27 range while V
grows 16-fold. Recall does not decay gently; the detector surfaces a bounded
number of channels at a fixed bottleneck, so recall is approximately a
constant over V. This is the same compression limit found for Proposition
2's gap, reached from a different direction, and it says the bottleneck must
scale with V rather than being a fixed hyperparameter.

### Frontier 2: strong coupling destroys recall

Recall is flat at 0.18-0.22 for coupling 0.05-0.35 and collapses to 0.06 at
0.50. At strong forcing the driven channel synchronises with its driver, so
its own history predicts it well and the excess vanishes. This is the
synchrony failure documented for cross-map methods, appearing here in a
different estimator.

### The n axis is not about n: the max-based threshold is unstable

  n       ghost_max   recall
   1000     0.0013     0.28
   2000     0.0087     0.02
   4000     0.0016     0.18
   8000     0.0086     0.04
  16000     0.0014     0.22

Recall tracks the single worst ghost, not the sample size. Across all 51
cells the correlation between ghost_max and recall is -0.44. The declared
threshold is max(0, ghost panel max), so ONE unlucky surrogate out of thirty
sets the bar for the whole scan and can cut recall by an order of magnitude.
Within-n spread across seeds is tight, so this is systematic, not noise.

This is a defect in the method as specified, found by the map: the estimator
is sound but its DECISION RULE is hostage to an extreme order statistic. A
quantile of the ghost panel would be stable where the maximum is not. That
change is not made here - it would need its own pre-registration, and
changing a threshold after seeing which threshold looks better is exactly
the move this project refuses.

### Two errors of ours the map exposed

B3 was a CONCEPTUAL ERROR. Saturation is the self-baseline R2, which does
not involve the code at all, so neither V nor the bottleneck can affect it.
It was 1.00 everywhere because logistic maps are near-deterministic in their
own history. The prediction should never have been written.

The REDUNDANCY AXIS WAS MIS-IMPLEMENTED. Duplicates were added as extra
channels carrying the DRIVER's signal, which enriches the code with driver
information and makes driven channels MORE detectable - recall rose from
0.18 to 0.44. That is the opposite of the intended test, which was whether a
target reachable through a second route becomes harder to detect. The
redundancy row of this map measures driver enrichment and should be read as
such; the intended test remains unrun.

### Honest limits

As declared: coupled logistic maps have clean Takens structure, low noise
and stationary dynamics. This map bounds performance from ABOVE.
