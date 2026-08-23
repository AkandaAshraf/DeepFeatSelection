# Pre-registration: replacing the 0.01 outflow bar with a calibrated one

Declared 2026-08-23, before the experiment was written or run.

## Why

The threshold audit found the outflow bar of 0.01 declared and never
justified (paper/threshold_audit.md, Rule 84). It is load-bearing: the
source-detection line was REOPENED on a margin of 0.0107 at coupling 0.50,
clearing the bar by seven percent. If 0.01 is too lenient, the reopening was
an artefact and the line should be closed. Nothing about the number as
chosen answers that question, because it was not chosen from anything.

A calibration argument is available after the fact - 0.01 is roughly ten
times the measured ghost of 0.001 - but it was invented during the audit, not
used to set the bar, and "ten times" is itself arbitrary. This replaces the
constant with an operating point that has a stated meaning.

## The replacement

The ghost is a circularly shifted copy of a real channel: same marginal
dynamics, no genuine influence on anything. Its outflow is therefore a draw
from exactly the null the bar is meant to exclude. The bar becomes a quantile
of that null instead of a constant:

  bar = q95( ghost outflow )   -> a 5% false-alarm rate, by construction

5% is not calibrated by this experiment either; it is a stated and
conventional operating point, and unlike 0.01 it means something specific
about how often a channel that influences nothing will be called a source.
The full null distribution is reported so any other rate can be read off it.

## Calibration and test are separate draws

The bar must not be fitted and tested on the same runs.

  CALIBRATION  seeds 100-129, coupling 0.50, 30 runs. Ghost outflow only.
               These 30 values define q95 and nothing else is taken from
               them.
  TEST         seeds 200-229, coupling 0.50, 30 runs. The regime where the
               statistic is claimed to work.
  CONTROL      seeds 200-229, coupling 0.30, 30 runs. The strongest regime
               where it is known NOT to work; it must largely fail the bar.

All at the deployed outflow configuration: b = 64, V = 15 plus ghost,
n = 4000, 25 epochs, 2 models, coupled logistic as in the coupling sweep.
Per run the statistic is the median outflow over the three true sources.

## Predictions, fixed now

  O1  The calibration ghost distribution is centred near zero, median within
      +/-0.002. If it is not, the null is not a null and the experiment is
      void.

  O2  q95(ghost) lands BELOW 0.01. Recorded as the expected case: the
      declared constant would then have been conservative, and the reopening
      stands a fortiori.

  O3  DECISIVE. At coupling 0.50, at least 80% of the 30 test runs have
      source outflow above the calibrated bar. That is the statistic's
      sensitivity at a 5% false-alarm rate, and it is the number that should
      have been reported when the line was reopened.

  O4  CONTROL. At coupling 0.30, no more than 20% of runs clear the bar. If
      the control clears at a rate close to the test, the bar separates
      nothing and the statistic is not usable at any threshold.

  O5  If q95(ghost) is at or ABOVE 0.01, the declared constant was too
      lenient. The reopening at 0.0107 was then inside the null and the line
      CLOSES. Declared now so that outcome cannot be softened later.

  O6  NO PREDICTION on how much of the original 0.0107 margin survives
      against the calibrated bar.

## What replaces what

If O3 and O4 hold, the reported quantity for outflow becomes sensitivity at a
5% false-alarm rate against the ghost null, and the constant 0.01 is retired
from this line rather than re-justified. The margin-over-ghost figure may
still be quoted, but never as a pass criterion.

## Void conditions

Void if the bar is computed from the test or control seeds, if the quantile
is changed after any source outflow is seen, if seeds are added or dropped
after a result, or if O5 is reached and reported as anything other than the
line closing.
