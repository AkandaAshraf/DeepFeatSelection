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

---

## Result (2026-09-05): both proposed repairs are no-ops; the identity path is not.

6 cells x 5 arms, 19.4 min. Two configurations failed the X3 guard before the
third passed, and X1 and X2 were not read from either.

  CONFIG 1  trunk fed poly3 features of data clipped at +-20, spanning
            several orders of magnitude into a raw linear layer.
            R2_own 0.014 against poly3's 0.117, winning on 0% of channels.
  CONFIG 2  raw own lags, scaled, but 1500 unregularised full-batch steps.
            R2_own -0.022. Worse.
  CONFIG 3  raw own lags, early stopping on the reserved validation fifth.
            Diagnosed on one channel: test R2 peaks at 0.165 by step 50 and
            decays to 0.02 by step 1350 while train loss keeps falling.
            Weight decay was tried at 1e-4 and 1e-2 and does far less.

X3 HOLDS on config 3. The learned trunk is a genuinely stronger self-baseline:

  noise      poly3 R2_own   trunk R2_own   trunk wins
  0.0           0.9930         0.9949          99%
  0.3           0.1314         0.1402          78%

### The declared decisive cell has no signal in it

AUROC, driven against source, mean of three seeds:

  arm              noise 0.0   noise 0.3
  RIDGE-INCL         0.984       0.440
  RES-SC-INCL        0.976       0.453
  RES-SC-EXCL        0.976       0.469
  RES-NOSC-INCL      0.797       0.301
  RES-NOSC-EXCL      0.704       0.272

**At noise 0.30 no arm reaches chance, the incumbent included.** X1 and X2
were declared to be judged there, so both were judged in a cell where nothing
works. They are recorded as FAILED by the letter and as UNINFORMATIVE in
substance. The cell was chosen because the incumbent's source false-positive
rate is highest there, which turns out to select a floor rather than
headroom: the highest failure rate marks the most noise, not the most room
to improve.

The source-FP readout was worse than uninformative. With 25 driven channels
of 30, top-k selects 83% of the system, so the metric's chance level is 0.833
and every arm scored 0.933 to 1.000. It cannot discriminate and should never
have been declared.

### What the informative cell says

At noise 0.0, paired across cells, on AUROC:

  contrast                     mean delta   wins   Wilcoxon p
  target exclusion, SC arms      +0.008      2/6      0.750
  target exclusion, NOSC arms    -0.061      1/6      0.062
  identity path vs none          +0.165      5/6      0.062
  ResNet shortcut vs incumbent   +0.003      2/6      1.000

  1. TAKING THE TARGET OUT OF THE CODE CHANGES NOTHING. The audit's concern
     is theoretically real, and it is empirically negligible here. Zeroing
     one channel of 30 from an encoder trained with 25% masking barely moves
     a 60-dimensional code, and the effect should shrink further as V grows.
     For the manuscript this means the "remaining system" wording is a
     specification error to correct, not a defect that moves any number.
  2. A BETTER SELF-BASELINE DOES NOT GIVE A BETTER STATISTIC. The trunk beats
     poly3 as a self-predictor at both noise levels and on 99% of channels at
     noise 0, and the resulting statistic is indistinguishable from the
     incumbent (+0.003, 2/6, p = 1.000). A stronger self-model absorbs more
     of the driven channels' signal too, and the two effects cancel.
  3. THE ARCHITECTURAL IDENTITY PATH CARRIES THE RESULT, and X5 declined to
     predict it. Removing the identity path costs 0.165 AUROC, the largest
     effect in the experiment. This is the third experiment in two days in
     which the contrast carrying no prediction dominates the one the
     decisive prediction was about; Rule 118 already names the pattern.

### Verdict

No adoption, as declared. Neither repair is worth carrying further in this
form. The identity-path effect is the only thing here worth a powered
pre-registration, and it is a statement about estimator architecture rather
than about either defect the audit raised.

### Not established

One system, one width, three seeds, redundancy 0. Half the grid was a floor.
The exclusion null is measured only at V = 30, where zeroing one channel is a
1-in-30 perturbation; it is not evidence about small V, where the same
perturbation is proportionally much larger.
