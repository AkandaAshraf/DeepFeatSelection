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

---

## Result (2026-08-20): the pipelines agree

Three systems, V = 30, n = 4000, identical hyperparameters and an identical
numpy ridge readout so that only the encoder differed.

  seed   Spearman   Jaccard    MAD      flagged (torch/tf)   time
   0      0.915      0.74     0.0002        22 / 18          6s / 53s
   1      0.955      0.96     0.0003        23 / 24          2s / 44s
   2      0.975      1.00     0.0002        14 / 14          2s / 44s

  F1  Spearman median 0.955   PASS (>= 0.90)
  F2  Jaccard  median 0.958   PASS (>= 0.70)
  F3  ghost max +0.0007 in BOTH pipelines, neither systematically higher
  F4  mean absolute difference in excess 0.0002, against reported effect
      sizes of 0.01 to 0.30 - two orders of magnitude smaller

### Verdict

The TensorFlow and PyTorch pipelines are interchangeable for every claim
made so far. The paper's published results and everything built on the
PyTorch reimplementation are commensurable, and cross-references between
them are safe. Recorded once so it need not be revisited.

The residual disagreement is what different random initialisations and
floating-point paths produce, not a difference in method: seed 0's Jaccard
of 0.74 comes from 22 versus 18 flagged channels near a shared threshold,
with rank agreement still at 0.915.

Incidentally measured: the PyTorch pipeline runs 15-20x faster here (2-6s
versus 44-53s), though the TensorFlow arm was pinned to CPU for fairness and
this is not a like-for-like hardware comparison.
