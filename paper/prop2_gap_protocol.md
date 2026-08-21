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

---

## Result (2026-08-20)

`scripts/prop2_gap.py` (single system) and `scripts/prop2_gap_sweep.py`
(scale sweep). Ghost passes throughout.

### Single system, V = 14

  affine +0.0060, interact +0.0082, oracle +0.0089
  total gap 33.1% of oracle
  READOUT share 76.8%, COMPRESSION share 23.2%
  ghost: affine +0.0016 -> interact +0.0026, PASS
  cost: 51 -> 147 features per channel (2.9x), independent of V

Taken alone this says the estimator should be changed: most of the gap is
recoverable, the ghost stays clean, and the extra features do not grow with
V so linear cost survives.

### But it does not survive the scale sweep

Bottleneck fixed at 32 throughout, as in the deployed method.

  V     gap       readout share   compression share
   14   +0.0029       76.8%            23.2%
   30   +0.0053       14.0%            86.0%
   60   +0.0039        7.9%            92.1%
  100   +0.0022       19.4%            80.6%

The readout share collapses from 77% at V = 14 to 8-19% at V = 60-100. This
was PREDICTED IN ADVANCE and stated in the script before it ran: the
bottleneck is fixed while the system grows, so the binding constraint
migrates from the readout to the compression.

V = 14 is the unrepresentative case. With 14 channels at E = 3 the embedding
is 42-dimensional and a 32-dimensional code barely compresses at all, so
almost nothing is lost to the bottleneck and the readout is the only thing
left to blame. That regime is not where MACE operates.

### Verdict

DO NOT CHANGE THE ESTIMATOR. At the scales the method is built for, roughly
80-92% of the gap is information the bottleneck never carried, which no
readout can recover. Paying 2.9x per-channel readout width to chase the
remaining 8-19% is a bad trade.

Proposition 2's lower-bound language stands, and this experiment says
something the paper did not: the bound is loose mainly because of
COMPRESSION, not because of the affine readout. The lever that would tighten
it is bottleneck width, not readout richness - and widening the bottleneck
costs encoder capacity and training time for every channel at once, rather
than 2.9x on a cheap per-channel ridge.

Had this been run only at V = 14 - the natural size for a quick synthetic
check - it would have produced a confident recommendation to change the
estimator in a paper currently under review. The scale sweep was the whole
experiment.
