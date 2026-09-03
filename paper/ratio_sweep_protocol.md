# Pre-registration: sinks per source as a continuous axis

Declared 2026-09-02, before any ratio other than 2 and 5.5 was run.

## Why

Today's shape experiment produced an explanation that currently rests on two
points and an argument:

> Marginal outflow improves as sinks per source rises --- the exact opposite
> of the conditional variant, which loses in that regime because the source
> becomes recoverable from what it drives. One structural parameter pushes
> the two variants in opposite directions.

Two points (2 sinks per source, 5.5 sinks per source) establish a direction.
They do not give the shape of either curve, and they do not locate the
crossover --- the ratio at which the conditional variant stops being the
better choice and the marginal one takes over. That crossover is the single
most useful number this line could produce for someone deciding which
variant to run on a real system, and it is currently unmeasured.

## Design, fixed now

Width and capacity are held fixed so the ratio is the only thing that moves:

  RATIO       sinks per source in {1, 2, 3, 5, 8, 11}, realised as
              (n_src, n_sink) = (12,12), (6,12), (4,12), (3,15), (2,16),
              (1,11). n_sink is held near 12 where the arithmetic allows,
              so the ratio changes mainly through n_src.
  WIDTH       n_iso padded so that V = 26 in every cell (25 + ghost), so
              b = 4V = 104 is constant and no cell is under-capacity.
  COUPLING    0.50, as in every prior outflow gate.
  STATISTICS  BOTH variants on the same runs: marginal outflow (A1) and
              conditional outflow (C1), so the crossover is measured on
              identical systems rather than across experiments.
  SEEDS       calibration 900-919, test 1000-1019 (20 each, unused before).

Reported per ratio: the sink-calibrated bar, thresholded sensitivity for
each variant, and per-run source-vs-sink AUC for each variant.

## Predictions, fixed now

  T1  Marginal sensitivity RISES with the ratio, monotonically within noise.
      This is today's mechanism stated as a curve.
  T2  Conditional AUC FALLS with the ratio. This is Rule 93's envelope
      stated as a curve: as a source acquires more sinks it becomes more
      recoverable from them, and conditioning removes its unique
      contribution.
  T3  DECISIVE. The two curves CROSS within the tested range. If they do,
      the crossover ratio is the deliverable and is reported with the
      spacing of the grid as its precision.
  T4  NO PREDICTION on where the crossover falls, or on whether either
      curve is monotone at every step rather than merely trending.
  T5  DISQUALIFYING. If the marginal curve does not rise, today's shape
      explanation is not a ratio effect and both the shape result and its
      published mechanism need re-examination. Declared now so that outcome
      is a result rather than a surprise.

## The rule, fixed now

  CROSSOVER FOUND    T1, T2, T3 hold. The paper gains a measured selection
                     rule: use the conditional variant below the crossover
                     ratio and the marginal one above it, with the caveat
                     that this is one synthetic family at one coupling.
  DIRECTION ONLY     T1 and T2 hold, T3 does not --- the curves trend
                     apart or together without crossing in range. Reported
                     as direction confirmed, crossover outside the tested
                     range, with the range stated.
  MECHANISM WRONG    T5 fires.

## Void conditions

Void if V is not held constant, if b is not 4V, if seeds 900-1019 have been
used before, if ratios are added or dropped after any result is seen, or if
a crossover is reported without the grid spacing as its precision.

## Result (2026-09-03, seeds 900-1019, 240 runs, 32.9 min, `ExpOutput/ratio_sweep/`)

   n_src n_sink ratio   A1_bar   C1_bar  A1_sens C1_sens  A1_auc C1_auc
     12    12    1.0   0.0013   0.0000    0.45    0.90    0.594  0.795
      6    12    2.0   0.0032   0.0000    0.85    1.00    0.715  0.924
      4    12    3.0   0.0034   0.0000    0.95    1.00    0.740  0.979
      3    15    5.0   0.0076   0.0000    0.85    1.00    0.844  1.000
      2    16    8.0   0.0110   0.0000    0.95    1.00    0.844  1.000
      1    11   11.0   0.0174   0.0000    0.75    0.90    1.000  1.000

  T1  marginal sensitivity vs ratio: Spearman +0.265. Rises, weakly, and not
      monotonically (0.45, 0.85, 0.95, 0.85, 0.95, 0.75). The rise is
      carried by the ratio-1 cell; from ratio 2 upward the curve is flat
      within noise (each rate has SE about 0.08 at n = 20).
  T2  conditional AUC vs ratio: Spearman +0.941. FAILS. The conditional
      variant does not lose as sinks per source rise; it improves
      monotonically and reaches the ceiling by ratio 5.
  T3  A1_auc - C1_auc: -0.201, -0.208, -0.240, -0.156, -0.156, +0.000. The
      conditional variant is ahead at every ratio below 11 and TIES it at
      11, where both are at the 1.000 ceiling. No crossing.
  T5  does not fire (T1's correlation is positive), but see T1: the rise
      it detects is not the smooth curve the mechanism described.

VERDICT: the declared rule has no branch for "T1 holds, T2 fails". No
crossover verdict is available. What the run establishes is the negative:
the mechanism written into the manuscript on 2026-09-02 --

> Marginal outflow improves as sinks per source rises --- the exact opposite
> of the conditional variant, which loses in that regime because the source
> becomes recoverable from what it drives. One structural parameter pushes
> the two variants in opposite directions.

-- is contradicted on the very family it was proposed for. The conditional
variant dominates the marginal one at every ratio tested, in AUC and in
thresholded sensitivity alike. It is withdrawn from the manuscript, and the
conditional variant's failure on the chamber (`paper/real_conditional_result.md`)
returns to unexplained.

### The verdict the script printed, and why it is wrong

`ExpOutput/ratio_sweep_run.log` ends with "CROSSOVER FOUND between ratio 8.0
and 11.0". Two bugs in the committed script produced that line, both fixed
after the run and disclosed in the script's docstring: (1) `np.sign` of the
tie at ratio 11 is 0, and the test `signs[i] != signs[i+1]` counted -1 -> 0
as a sign change, so a meeting at the ceiling was reported as a crossing;
(2) the verdict consulted T1 and T3 and never T2, so a rule that requires
all three was applied with one missing. Replaying the fixed verdict on the
recorded `results.csv` gives "T2 FAILS ... no crossover verdict is
available". The printed line is left on record as Rule 107 requires.

### What the generator audit of the same day does to this result

`paper/generator_audit.md`: the sweep's ratio-1 cell (12 sources, 12 sinks,
random parent assignment) has on average 4.1 orphan sources per run -- a
source with no sink, which by construction has zero outflow to detect --
and 2.05 phase-locked sources per run. Roughly half the "sources" in that
cell are undetectable by design, so its 0.45 is not a measurement of the
statistic at ratio 1 and the T1 correlation, which rests on that cell, is
not a measurement of a ratio effect. Effective ratios (sinks per non-orphan
source) are 1.5, 2.2, 3.1, 5.0, 8.0, 11.0. The per-run frames the script
did not save cannot be recovered without a re-run; the fixed script saves
them. No re-run is scheduled: the negative result on T2 does not depend on
the ratio-1 cell (C1 leads A1 at every ratio from 2 to 8 by 0.16-0.24 AUC
on cells with at most 0.65 orphans per run).
