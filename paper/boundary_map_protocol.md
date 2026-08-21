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
