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
