# Pre-registration: do the TensorFlow and PyTorch pipelines agree?

Declared 2026-08-20, before the comparison was written or run.

## Why this matters

The companion paper's published results were produced by a TensorFlow
pipeline. Every result since - the worm corpus, the DepMap work's tooling,
the intracranial studies, the boundary map, and today's four experiments -
comes from a PyTorch reimplementation. The two have never been compared on
the same data.

If they disagree, the paper's numbers and everything built since are not
commensurable, and every cross-reference between them is unsafe. This is
housekeeping in the sense that no new science depends on it, and load-bearing
in the sense that a great deal of existing science does.

## Architectures, confirmed identical before running

Both implement the same masked autoencoder: input V*E, dense to
max(2b, 64) with tanh, dense to b; decoder mirrored. Same bottleneck, same
masking fraction, same optimiser and learning rate. The comparison is
therefore of implementations, not of designs.

## Design

The same synthetic system, the same seed, the same hyperparameters, scored by
both pipelines. Reported per channel:

  excess       the statistic itself
  ghost        the surrogate panel
  flagged      the set of channels above threshold

Metrics: Spearman correlation of per-channel excess between pipelines; mean
absolute difference; Jaccard overlap of the flagged sets.

Three systems are used, not one, so that agreement is not a property of a
single draw.

## Predictions, fixed before running

  F1  Spearman correlation of per-channel excess >= 0.90 between pipelines.
  F2  Jaccard overlap of flagged sets >= 0.70.
  F3  Both ghost panels sit near zero, with neither systematically higher.
  F4  NO PREDICTION on the mean absolute difference in excess. Different
      random initialisations and different floating-point paths guarantee
      some numerical difference; the question is whether it is small
      relative to the effect sizes the method reports, which are of order
      0.01 to 0.30.

## What follows

If F1 and F2 hold, the two pipelines are interchangeable for the purposes of
every claim made so far, and this is recorded once so it need not be
revisited.

If either fails, the paper's results and everything built on the PyTorch
pipeline must be treated as separate bodies of evidence until the
discrepancy is explained, and that must be stated wherever both are cited.

## Void conditions

Void if hyperparameters are tuned separately for the two pipelines, if the
systems are changed after seeing a result, or if a disagreement is explained
away without being reported.
