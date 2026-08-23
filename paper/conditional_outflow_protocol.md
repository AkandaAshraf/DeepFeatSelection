# Pre-registration: conditional outflow as the primary hypothesis

Declared 2026-08-23, before the experiment was written or run. Seeds 0-4 are
spent on the void error-metric sweep and are NOT reused here.

## Why, and what is being confirmed

The error-metric sweep (paper/error_metric_result.md) voided on an imported
bar. Its description showed that rescoring the same ridge predictions changes
nothing - R2, MAE, Spearman and Gaussian entropy agree within 0.04 AUC and
degrade together - while a change of CONDITIONING moved the sink proxy onto
the null at every coupling. That was one column of eight with no declared
bar, in a void run. It is a hypothesis, not a result.

This tests it as the primary hypothesis, on fresh seeds, with the risk it
carries built into the design rather than left as a caveat.

## The two statistics, both computed on every channel

  A1  MARGINAL, the current metric. Baseline is the code's own delay
      embedding; the question is what channel q's features add to predicting
      the code's future.
  C1  CONDITIONAL. Baseline is the code's delay embedding PLUS the features
      of every other channel; the question is what q adds that nothing else
      supplies. Proposition 1 says such a quantity is zero for a channel that
      is redundant given the others, which is what a sink is.

Both from the same autoencoder, same codes, same data, same ridge, same
squared-error scoring. The ONLY difference is the conditioning set, so any
difference between them is attributable to that and to nothing else.

## The risk this design exists to test

If a source is recoverable from its own sinks, the source is redundant too
and C1 should zero it along with them. The sweep did not show this, but its
system gave each source only two sinks. A redundancy axis is therefore part
of the design and not a follow-up: near-duplicate copies of each source are
added, which is the sharpest form of the risk, since a copy makes its source
almost exactly recoverable.

Copies are excluded from both truth sets, following the boundary map's
convention: a copy is neither a source nor driven.

## Design

Coupled logistic as throughout this line, n = 4000, b = 64, 25 epochs,
2 models, ghost appended exactly as in the deployed gate.

  couplings    0.30, 0.50, 0.70
  copies k     0, 1, 2 near-duplicates of EACH source (noise 0.02)
  seeds        5-14, ten fresh seeds
  90 cells

Outcome per cell per statistic: source-versus-sink AUC, plus the medians by
role and the ghost on each statistic's own scale.

## Predictions, fixed now

  P1  REPRODUCTION, DIRECTION ONLY, no numeric bar. At k = 0, A1's
      source-versus-sink AUC DEGRADES as coupling rises from 0.30 to 0.70.
      This is the qualitative claim that is actually established; stating it
      as a number is the mistake that voided the last three experiments
      (Rule 90). If the direction fails, this experiment is void.

  P2  PRIMARY AND DECISIVE. At k = 0, C1's source-versus-sink AUC EXCEEDS
      A1's at all three couplings, and the margin C1 minus A1 GROWS with
      coupling. Internally calibrated against A1 on the same cells, so no
      external constant is imported.

  P3  At k = 0, C1 places sinks on the null: the median C1 for sinks lies
      within one interquartile range of the median C1 for ISOLATED channels,
      at every coupling. Isolated channels are the reference because they are
      real dynamics coupled to nothing, which the ghost is not.

  P4  THE DECLARED RISK. As k rises, C1's source signal FALLS. Predicted: at
      k = 2, C1's source-versus-sink AUC is below its own k = 0 value. If it
      does not fall, the risk is not realised in this system. If it falls
      BELOW A1's AUC at the same k, then under redundancy the conditional
      statistic is worse than the current one, and that is an operating limit
      to report, not a failure to explain away.

  P5  The ghost stays clean for both statistics at every cell, judged on each
      statistic's own scale as the ghost median lying below the 5th
      percentile of that statistic's source distribution.

  P6  NO PREDICTION on A1's behaviour under redundancy.

## The rule, fixed now

  CONFIRMED           P2, P3 and P5 hold at k = 0. Conditioning removes the
                      sink-proxy confound, and the source-detection line
                      reopens on a measurement rather than a threshold.
  CONFIRMED WITH LIMIT
                      P2, P3 and P5 hold at k = 0 but P4 shows the collapse.
                      Reported with the operating envelope stated: the
                      statistic works below a redundancy level that is named.
  NOT CONFIRMED       P2 fails at k = 0. The sweep's column was noise or an
                      artefact of the spent seeds, and the line stays shut.

## Void conditions

Void if P1's direction fails, if seeds 0-4 are reused, if the redundancy axis
or couplings change after any result is seen, if C1 is reported without A1
beside it on the same cells, or if the AUC comparison in P2 is replaced by
any externally imported number.
