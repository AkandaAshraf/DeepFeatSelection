# Pre-registration: what closes Proposition 2's lower-bound gap?

Declared 2026-08-20, before the experiment was written or run.

## The claim being tested

Proposition 2 states that MACE's population excess equals the fraction of a
variable's next-step variance its own history cannot explain. The implemented
estimator is a **lower bound** on that quantity: the readout is a ridge
regression *affine* in the code, so it can only use the code linearly. The
companion paper measures the shortfall against an oracle at 10-16% and says
so, but does not say what the shortfall is made of.

## The decomposition, which is the point of this experiment

The measured gap conflates two distinct losses, and they have opposite
implications:

  COMPRESSION LOSS  the b-dimensional code cannot carry everything the rest
                    of the system knows about q's future. No readout can
                    recover this; only a wider bottleneck can.
  READOUT LOSS      the affine readout cannot use what the code does carry.
                    A richer readout recovers this, at some cost in features.

If the gap is mostly compression, a richer readout is wasted work and the
paper's lower-bound language is as tight as it can be. If it is mostly
readout, the bound is loose for a fixable reason and the estimator should be
improved.

## Estimators compared, fixed now

On synthetic systems with known coupling:

  BASE         R2[ x_q(t+1) | phi(own lags) ]                    degree-3 poly
  AFFINE       R2[ x_q(t+1) | phi(own), z ]                      as implemented
  INTERACT     R2[ x_q(t+1) | phi(own), z, own_linear (x) z ]    bilinear
  ORACLE       R2[ x_q(t+1) | phi(own), Phi(all other channels) ]

ORACLE uses the full uncompressed state of every other channel, so it is the
ceiling the code approximates; the difference between ORACLE and INTERACT is
attributable to compression, and between INTERACT and AFFINE to the readout.

Excess for each estimator is that estimator's R2 minus BASE. Reported as the
mean over driven channels, and separately for the ghost, which must stay near
zero for every estimator.

## Predictions, fixed before running

  P-P1  A gap exists: ORACLE excess exceeds AFFINE excess on driven
        channels, reproducing the paper's finding on this system.
  P-P2  DECOMPOSITION. This is the quantity of interest and no direction is
        predicted, because the answer is genuinely unknown to us: report
        READOUT share = (INTERACT - AFFINE) / (ORACLE - AFFINE) and
        COMPRESSION share = (ORACLE - INTERACT) / (ORACLE - AFFINE).
  P-P3  The ghost stays near zero under INTERACT. A richer readout has more
        capacity to fit noise, so this is the check that the improvement is
        not overfitting. If the ghost rises materially, INTERACT is rejected
        regardless of what it does to the gap.
  P-P4  COST. Report the feature count of each readout. INTERACT multiplies
        the per-channel readout width; if it does so by a large factor it
        erodes the linear-cost claim that motivates the method, and that
        trade-off is reported whatever the accuracy result.

## What would make this actionable

If READOUT share is large AND the ghost stays clean AND the cost is
tolerable, the estimator should be changed and the paper's bound tightened.
If READOUT share is small, the lower-bound language stands as the honest
description and no change is made. Either outcome is reported.

## Void conditions

Void if the estimators or predictions are altered after any result is seen,
or if the ghost check is dropped.
