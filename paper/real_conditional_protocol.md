# Pre-registration: conditional outflow on a real physical system

Declared 2026-08-23, before any excess, outflow or AUC was computed on these
datasets. The fitness screen that selected them was run first and is reported
in full in paper/dataset_fitness_protocol.md.

## What is being tested

Conditional outflow was confirmed on coupled logistic maps this morning: at
coupling 0.70 the marginal statistic sits at chance, 0.517, while conditioning
holds at 0.872. Every previous version of this line also looked healthy on
synthetic data. The intracranial EEG result passed every synthetic and
within-cohort check and then failed a pre-registered replication, and the
first chamber attempt was voided. Synthetic confirmation is not evidence about
real systems.

The apparatus supplies structural ground truth: the experimenter SETS certain
variables, and no expert judgement is involved in knowing that a fan-load
setting is not caused by a pressure reading.

## Datasets, both of them

Two of four chamber datasets cleared the fitness gate. Both are tested and
both are reported; choosing between them was not a declared rule and is not
done (see the screen result).

  PRIMARY    wt_intake_impulse_v1 at decimation m = 10
             5 runs, 250,000 samples each, 25,000 after decimation
             lag_info +0.0224 against a pass mark of +0.0136
             SOURCES: hatch, load_in         SENSORS: the 11 measured
  SECOND     wt_walks_v1 at decimation m = 10
             2 runs (the 320,000-sample regime_jumps runs), 32,000 after
             decimation. lag_info +0.0158
             SOURCES: pot_1, pot_2, load_in, load_out   (hatch is constant
             in these runs and is excluded, Rule 80)
             DECLARED LIMITATION: two runs is thin, and both contain regime
             boundaries. It is reported for what it is.

Ground truth uses the CORRECTED assignment: load_in and load_out are settable
fan loads, not sensors. The first chamber protocol had them on the wrong side.

## The two statistics

Identical in every respect except the conditioning set, as in the synthetic
confirmation:

  A1  MARGINAL     baseline is the code's own delay embedding
  C1  CONDITIONAL  baseline also holds every other channel's features

Same autoencoder, codes, data, ridge and squared-error scoring. Any difference
is attributable to conditioning alone.

Capacity: b = 4V, following the rule that the first chamber attempt violated
by running at 2V. V includes the ghost channel.

## Predictions, fixed now

  R1  DECISIVE AND PRIMARY. C1's source-versus-sensor AUC EXCEEDS A1's, on
      BOTH datasets. Internally calibrated against A1 on the same runs; no
      external number is imported (Rule 90).

  R2  C1 places the sensors at or below the ghost: median C1 over sensors is
      at or below the median C1 of the ghost channel, on both datasets.

  R3  The ghost is clean for both statistics, judged on each statistic's own
      scale as the ghost median lying below the 5th percentile of that
      statistic's source distribution.

  R4  NO PREDICTION on absolute magnitudes, which are not comparable between
      a physical apparatus and a logistic map.

  R5  A permutation test is reported for C1: 10,000 shuffles of the role
      labels within each dataset, giving a p-value for the observed AUC. This
      is REPORTED, not a decision rule - the decision is R1. It is declared
      now so that a p-value cannot be introduced afterwards if R1 is
      unfavourable.

  R6  DECLARED LIVE OUTCOME. C1 fails to beat A1 on both datasets. Then the
      synthetic confirmation does not transfer, the statistic joins the iEEG
      result as something that passed every synthetic check and failed on real
      data, and the source-detection line closes. Recorded now so that
      outcome is a result and not an abandonment.

## A limitation that cannot be removed here

The synthetic system has ISOLATED channels - real dynamics coupled to nothing
- which are the correct reference for "influences nothing". A physical
apparatus has none: every sensor is downstream of something. The ghost is
therefore the only null available, and today's outflow result established that
the ghost is a valid null for absence of influence but NOT for the
proxy-carrying that defeats the marginal statistic. R2 is weaker than its
synthetic counterpart for exactly this reason, and is stated as such rather
than presented as equivalent.

## The rule, fixed now

  CONFIRMED ON REAL DATA   R1 holds on both datasets and R3 holds. The
                           statistic detects experimenter-set variables in a
                           physical system where ground truth is structural.
  PARTIAL                  R1 holds on one dataset and not the other.
                           Reported as such, with NO claim of transfer.
  FAILS                    R1 fails on both. The line closes and that closure
                           is the headline.

## Void conditions

Void if the ground-truth assignment is changed after seeing any result, if
either qualifying dataset is dropped from the report, if the decimation is
changed, if b is not 4V, or if any externally imported constant replaces the
A1 comparison in R1.
