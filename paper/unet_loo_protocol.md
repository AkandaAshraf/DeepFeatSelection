# Pre-registration: a forecasting U-Net, and holding out one input at a time

Declared 2026-09-05, before the experiment was written or run.
EXPLORATORY. Nothing can be adopted on this run whatever it shows.

## Two closures stand against this, and they are named first

The proposal is: train a network that maps the state at t to the state at
t+1 for every variable at once, with a skip carrying each variable's own
history to its own output; then hold out each input variable in turn and
measure the damage to the prediction of all the others. Damage is the
source-detection score.

The readout is leave-one-out on a trained model, and this project has closed
that twice.

  1. MECHANISM 1 (manuscript, Section on maturity). Leave-one-out gain
     collapses as a model matures because it learns alternative routes. On
     real data the collapse endpoint replicates: AUROC 0.500 to three decimal
     places in all three worms tested, with no intermediate checkpoint
     informative. The collapse is redundancy-ordered, complete for a hub
     parent whose imprint is carried by six children.
  2. RULE 63 (ledger, 2026-08-20). A leave-one-out dual was rejected on paper
     as a difference-based importance score before it was built. The rule
     reads: when proposing a dual to an existing statistic, check first
     whether the dual reproduces a failure mode already catalogued for the
     original.

The reason is structural rather than a matter of tuning. Leave-one-out is a
difference of risks, and when a variable's information is recoverable from
the others the achievable risk does not change when it is removed. A system
with Takens redundancy is one where most variables are partly recoverable
from the others, which is the regime this method exists for.

There is a further tension specific to the proposal as made. Training with
random input removal is precisely what teaches a model to route around a
missing input, so it should DEEPEN the collapse rather than relieve it. That
makes masking an arm rather than an assumption.

## What is genuinely new, and why it is worth one run

The 2026-08-20 rejection failed on separation, not on maturity: measured
source +0.0060, isolated +0.0037, sink +0.0062. Sinks scored as high as
sources because a sink's history proxies its driver's history, so the
statistic reported shared-driver correlation as outflow.

A per-variable skip attacks exactly that. If variable k's own history reaches
k's own output directly, the bottleneck never has to carry it, so ablating j
can damage k only through a genuine cross-variable pathway. The skip is the
conditioning the rejected version lacked, supplied architecturally rather
than by a second regression. That is the one part of this that neither
closure covers.

## The design

Forecasting U-Net. Input the embedded state at t for all V variables, output
x_k(t+1) for every k at once:

    pred_k(t+1) = skip_k(own lags of k) + head_k(bottleneck(all lags))

Ablation readout, on the trained model, test segment only:

    damage(j -> k) = R2_k(full input) - R2_k(input with j's columns zeroed)
    loo_outflow(j) = mean over k != j of damage(j -> k)

Five arms. The first is a reference with a known answer, the other four cross
the two factors.

  ADD            REFERENCE, not a U-Net. The conditional additive outflow the
                 manuscript already reports: for each ordered pair, the gain
                 in predicting x_k(t+1) from adding j's lags to k's OWN lags,
                 averaged over k != j. Known to work on synthetics.
  MASK-SKIP      masked training, per-variable skips
  MASK-NOSKIP    masked training, no skips
  NOMASK-SKIP    no masking during training, per-variable skips
  NOMASK-NOSKIP  no masking, no skips

  SYSTEM  boundary_map.make_system, V = 30, n = 4000, coupling 0.20,
          redundancy 0. Sources are autonomous and drive; driven channels
          drive nothing. So true outflow is positive on sources and zero on
          driven channels, by construction.
  NOISE   observation noise in {0.0, 0.05}. NOT 0.30: yesterday's run put
          every arm at or below chance there, and Rule 122 forbids judging a
          contrast in a cell where the reference does not clear chance.
  SEEDS   0, 1, 2

6 cells x 5 arms.

## Reported quantity and its chance level

PRIMARY: AVERAGE PRECISION (area under the precision-recall curve), sources
as the positive class, on the outflow score.

  positives    5 sources of 30 channels
  BASE RATE    0.167, and average precision of a random ranker IS the base
               rate, so 0.167 is the chance level
  ceiling      lift is capped at 1 / 0.167 = 6.0x

Average precision rather than AUROC, and the reason is the class imbalance.
The positive class is 17% of channels. AUROC's false-positive rate takes the
25-channel negative class as its denominator, so a handful of sources ranked
above driven channels barely moves it and the number reads better than the
ranking deserves. Average precision uses precision, whose denominator is the
predicted-positive set, so it reports directly on what a practitioner would
act on: how many of the top picks are real.

It also matches this project's own framing. The enrichment table already
treats the method as triage and already fixes lift >= 2.0 as the bar below
which a deployment cannot demonstrate anything. Average precision divided by
the base rate IS that lift, so the two analyses are on one scale.

Rule 123 is satisfied: the chance level is computed above, before committing.

SECONDARY, descriptive only and reported for comparability with the
manuscript's existing outflow numbers: AUROC on the same scores; mean score
on sources against driven; per-arm wall clock; and the trained model's own
held-out R2, so an uninformative result can be separated from an untrained
one.

## Predictions, fixed now

  U1  DECISIVE. At noise 0.0, at least one U-Net arm reaches average
      precision above 0.33, pooled over seeds. That is twice the 0.167 base
      rate, and lift >= 2.0 is the bar the enrichment table already fixed as
      the minimum that demonstrates anything. If every arm sits near the base
      rate the line is CLOSED, confirming Mechanism 1 and Rule 63 from a
      new direction and at the architecture meant to escape them.
  U2  DECISIVE. NOMASK arms beat their MASK counterparts on average
      precision. The
      mechanism is stated above: masked training teaches route-around, which
      is the collapse. If masking turns out to HELP, the account of why
      leave-one-out collapses is wrong and that is worth more than the
      statistic.
  U3  DIRECTIONAL. SKIP arms beat their NOSKIP counterparts. With the skip,
      the bottleneck carries only cross-variable information, so ablation
      damage is not contaminated by the loss of the target's own history.
  U4  GUARD. The ADD reference clears average precision 0.33 at noise 0.0,
      the same twice-base-rate bar. If it does not, the cell cannot support
      any contrast and the run is UNINFORMATIVE rather than negative
      (Rule 122).
  U5  GUARD. Every U-Net reaches held-out R2 above 0.5 at noise 0.0 on its
      own forecasting task. A model that has not learned to forecast cannot
      be ablated informatively, and the ResNet run of the same day was
      derailed twice by exactly this before a guard caught it.

## The rule, fixed now

NONE, and no adoption is possible. The outcomes are:

  CLOSED        U1 fails with U4 and U5 intact. Leave-one-out is dead at this
                architecture too, the skip does not rescue it, and the line
                closes with a third independent measurement.
  OPEN          U1 holds with U4 and U5 intact. Licenses a powered
                pre-registration at more widths, a second generating family
                and a redundancy axis. Nothing is adopted, nothing enters the
                manuscript, on this run.
  UNINFORMATIVE U4 or U5 fails. The instrument, not the hypothesis, is what
                was measured.

## Void conditions

Void if the ablation is computed on anything but the held-out test segment;
if the arms do not share the same generated system and splits within a cell;
if the grid, seeds or noise levels change after any result is seen; if U1 is
judged anywhere other than noise 0.0; or if the ADD reference is dropped.

---

## Result (2026-09-05): UNINFORMATIVE by the declared rule, and the guard was right.

6 cells x 5 arms, 1.7 min.

  AVERAGE PRECISION (chance = base rate = 0.167; bar = 0.333)
  noise    ADD   MASK-SKIP  NOMASK-SKIP  MASK-NOSKIP  NOMASK-NOSKIP
  0.00    0.828    0.850       0.460        0.534         0.853
  0.05    0.242    0.714       0.355        0.590         0.817

  LIFT over base rate
  0.00    4.97     5.10        2.76         3.21          5.12
  0.05    1.45     4.29        2.13         3.54          4.90

  MODEL HELD-OUT R2 on its own forecasting task
  0.00      --     0.955       0.797        0.725         0.285
  0.05      --     0.837       0.570        0.601         0.265

  U4 HOLDS. ADD reference 0.828 against the 0.333 bar at noise 0.0.
  U5 FAILS. Minimum U-Net forecast R2 is 0.285.
  U1 HOLDS. Best U-Net 0.853, and 0.850 excluding the arm that failed U5.
  U2 FAILS, in the direction the protocol said would matter more than the
     statistic. Under SKIP, masking takes AP from 0.460 to 0.850.
  U3 FAILS as stated. SKIP beats NOSKIP only when masked; the two factors
     interact and the protocol predicted a main effect.

**The declared verdict is UNINFORMATIVE and it stands.** Nothing is adopted.

### The guard fired for exactly the right reason

U5 is scoped run-wide, so one arm's failure discards the run, which repeats
the scoping error Rule 119 already recorded for a precision floor. But the
guard was not merely pedantic here. The arm that failed it is the arm with
the HIGHEST average precision:

  arm             AP      forecast R2   grades with child count
  NOMASK-NOSKIP  0.835      0.275            rho -0.068
  MASK-SKIP      0.782      0.896            rho +0.498  (p = 0.005)

A model never trained with missing inputs treats a zeroed input as an
out-of-distribution shock. Its ablation damage separates sources from driven
channels while carrying NO information about how much a source actually
drives. U5 was written to catch a model that cannot forecast, and it caught
one whose detection score was an artefact.

### The graded positive control, post hoc and labelled

scripts/unet_loo_graded.py. The generator's parent assignment is recovered by
replaying each seed's draws, so every source has a known child count (2 to 9
across the cells). A statistic that measures outflow must track HOW MUCH a
variable drives, not merely which class it is in. Pooling 30 source
observations across 6 cells, ranked within cell:

  ADD            rho +0.342   p 0.064
  MASK-SKIP      rho +0.498   p 0.005
  NOMASK-SKIP    rho -0.020   p 0.917
  MASK-NOSKIP    rho +0.062   p 0.744
  NOMASK-NOSKIP  rho -0.068   p 0.722

Only MASK-SKIP grades. Three arms separate the classes to some degree while
carrying nothing about magnitude.

### Two properties the additive reference does not have

scripts/unet_loo_controls.py.

  1. NOISE ROBUSTNESS. At noise 0.05 the ADD reference collapses to AP 0.242,
     lift 1.45, and AUROC 0.397 which is BELOW chance. MASK-SKIP holds at
     0.714, lift 4.29. The manuscript's additive outflow is the arm that
     fails first here.
  2. INDEPENDENCE FROM SELF-PREDICTABILITY. Spearman between the score and
     the channel's own self-R2, across all channels: ADD +0.685, MASK-SKIP
     +0.037. The additive reference largely tracks how self-predictable a
     channel is; the skip architecture does not. That is what the skip was
     argued to do and it is the one prediction of the design that held.

### What this licenses

A powered pre-registration of MASK-SKIP alone, at more widths, a second
generating family and a redundancy axis, with U5 scoped per arm and the
graded child-count control declared in advance rather than added after. It
does not license any claim, any adoption, or any manuscript text.

### Not established

One width, one family, three seeds, redundancy 0, two noise levels. The
graded control pools five sources per cell and its 30 observations are not
independent. The ADD collapse at noise 0.05 rests on three seeds. Redundancy
is the axis most likely to break this and it was not run: duplicated sources
are exactly the condition under which leave-one-out is expected to die.
