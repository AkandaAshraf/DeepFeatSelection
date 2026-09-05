# Pre-registration: a residual shortcut, and taking the target out of the code

Declared 2026-09-05, before the experiment was written or run.
EXPLORATORY. Nothing can be adopted on this run whatever it shows.

## What this attacks

An external audit of the manuscript raised two defects that we verified
against the text and the implementation. Both are real.

  1. THE SELF-BASELINE IS A FIXED POLYNOMIAL. Proposition 1 needs the
     self-model class to be dense enough to saturate on an autonomous
     channel. Our self-model is degree-3 polynomial ridge on three own lags,
     which is not dense in anything. When it fails to saturate, nothing
     bounds excess above zero for an autonomous channel, and the
     no-false-positive guarantee lapses exactly where it is needed.
  2. THE CODE CONTAINS THE TARGET. The abstract says the code is of the
     "entire remaining system". The method forms the joint state from ALL
     variables and the code is computed from that, with no target masking at
     scoring. So the joint readout gets a learned nonlinear representation
     of the target's OWN lags that the polynomial baseline does not have.
     Improvement can then reflect better self-modelling rather than
     external drive.

These are one mechanism, not two: both let the joint side beat the self side
on a channel that receives nothing. The 2026-09-05 per-channel-null run
measured the consequence, source false positives of 0.128 and 0.244 at
observation noise 0.05 and 0.30, against a nominal ghost level of 0.032,
with the ghost panel clean throughout.

## The design

Two changes, crossed, so each is attributable.

RESIDUAL SHORTCUT. Replace the fixed polynomial self-baseline with a learned
residual trunk on the own lags, and give the full model an identity path from
that trunk's prediction:

    full(t) = own_pred(t) + delta(trunk_features, code)

The own trunk is trained FIRST, to convergence, and then FROZEN. This is not
a detail. Trained jointly, the optimiser is free to underfit the own path and
let the code branch carry prediction that own lags could have supplied, which
would manufacture excess on autonomous channels and defeat the purpose.
Two-stage training with a frozen trunk keeps the self-baseline as good as the
class allows, which is what the saturation argument requires.

TARGET EXCLUSION. The masked autoencoder is trained with random channel
subsets zeroed, so a code computed with the target's columns zeroed is in
distribution for the SAME encoder. Excluding the target therefore costs one
extra forward pass per target, not a retrained encoder, and the amortisation
that makes the method scale survives.

Five arms. All share one encoder per cell and one own-trunk per target, so
the self-baseline is identical across arms and only the code branch differs.

  RIDGE-INCL     incumbent: poly3(own lags) + code(full Z), two ridge solves
  RES-SC-INCL    frozen trunk, identity path, code(full Z)
  RES-SC-EXCL    frozen trunk, identity path, code(Z with target zeroed)
  RES-NOSC-INCL  no identity path, fresh net on [trunk features, code(full Z)]
  RES-NOSC-EXCL  no identity path, fresh net on [trunk features, code(excl)]

  excess(q) = R2(full model) - R2(own model)

with the same own model in every arm, so excess is comparable across arms.

  SYSTEM  boundary_map.make_system, V = 30, n = 4000, coupling 0.20,
          redundancy 0, b = 2V.
  NOISE   observation noise in {0.0, 0.30}. The high level is mandatory: at
          zero noise the incumbent's source false positives are already
          0.000 and the cell cannot show whether either change helps.
  SEEDS   0, 1, 2

30 cells. Sources are autonomous BY CONSTRUCTION, so their excess is the
guarantee under test.

## Reported quantities

Threshold-free where possible, because the ghost threshold is itself one of
the things under suspicion and Rule 120 forbids comparing rules that flag
different numbers without matching the count.

  mean and max excess on SOURCES   the guarantee says <= 0
  AUROC driven vs source           ranking quality, threshold-free
  source FP at matched top-k       k = the number of truly driven channels
  R2_own, learned trunk vs poly3   is the self-baseline actually stronger

## Predictions, fixed now

  X1  At noise 0.30, mean source excess is LOWER for EXCL arms than for the
      matching INCL arms. This is the target-exclusion prediction and it is
      the reason the experiment exists.
  X2  At noise 0.30, source FP at matched top-k is lower for EXCL than for
      the matching INCL arm.
  X3  The learned frozen trunk attains higher R2 on own lags than poly3
      ridge does, on the same channels and splits. If it does not, the
      shortcut is not supplying a stronger self-baseline and any change it
      produces has some other cause.
  X4  NO PREDICTION on AUROC. Ranking and thresholding are different
      properties and predicting either way would be storytelling.
  X5  NO PREDICTION on shortcut versus no shortcut. That contrast is the
      architectural question and we have no basis to call it.

## Declared risk

If the own trunk underfits, every ResNet arm inherits inflated excess and X1
becomes uninterpretable. X3 is the guard: a trunk that does not beat poly3
has not converged, and the run is reported as inconclusive rather than read.

## The rule, fixed now

NONE. This is exploratory and no arm can be adopted from it. A result in the
predicted direction licenses a properly powered pre-registration at more
widths, more seeds and a second generating family, and nothing else. A result
against the prediction is recorded as a negative and closes the line.

## Void conditions

Void if the arms do not share one encoder per cell and one own-trunk per
target; if the own trunk is not frozen before the code branch is trained; if
the grid, seeds or noise levels change after any result is seen; or if X1 is
judged anywhere other than noise 0.30.
