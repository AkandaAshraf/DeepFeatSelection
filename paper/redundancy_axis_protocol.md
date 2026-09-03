# Pre-registration: redundancy as a real axis

Declared 2026-09-02, before the experiment was written or run.

## Why: the crossed-saturation experiment named this and could not test it

The crossed-saturation run was inconclusive for a specific, diagnosed
reason. At constant b/V its two redundancy strata disagreed:

    Spearman(source_fp rate, V), noise > 0:   k = 0:  +0.672
                                              k = 2:  +0.002

With duplicates the failure saturates at every width; without them it grows
with width. Its result section stated the limitation plainly:

> That interaction was not anticipated by the protocol and is not tested by
> it -- k has two levels, which cannot characterise an interaction.

Two levels can detect that an interaction exists; they cannot describe its
shape, and cannot say whether k = 2 is already saturated or merely different
from k = 0. This runs the axis properly.

## A second defect this repairs

The boundary map's redundancy axis was mis-implemented and the paper
currently carries the caveat. Duplicates were added carrying the DRIVER's
signal, which enriches the code with driver information and makes driven
channels MORE detectable -- recall rose from 0.18 to 0.44, the opposite of
the intended test. `boundary_map.make_system` still does this: its
duplicates copy source channels.

That is the correct construction for THIS question (does duplicating a
source's signal break source blindness) and the wrong one for the question
the boundary map thought it was asking (does Takens redundancy among TARGETS
limit detection). Both are stated here so the axis is not mis-read a second
time. This experiment tests the former and does not claim the latter.

## Design, fixed now

  REDUNDANCY   k in {0, 1, 2, 4, 8} duplicates per source. Five levels, so
               the shape of the interaction is characterisable rather than
               merely detectable.
  WIDTH        V in {15, 30, 60}, b = 2V in every cell, so capacity ratio is
               constant at 2.0 and width is not confounded with capacity --
               the discipline the crossed-saturation run established.
  SATURATION   observation noise in {0.0, 0.05, 0.10}, the three levels
               where the crossed run showed the clearest signal. Reduced
               from five to keep the cell count affordable at five k levels.
  SEEDS        0, 1, 2.   5 x 3 x 3 x 3 = 135 cells.

Machinery is `boundary_map`'s, unchanged, as in the crossed run.

## Predictions, fixed now

  R1  REPRODUCTION. At k = 0 the width ordering seen in the crossed run
      reappears: Spearman(source_fp, V) > 0.4 at noise > 0. If the k = 0
      ordering does not reproduce, that ordering was noise and the crossed
      run's disagreement dissolves -- report it and stop.
  R2  DECISIVE. Source false positives rise monotonically with k at fixed
      width and noise. This is the claim that duplication of a source's
      signal is what breaks source blindness, and it is the mechanism Rule
      86 asserts on k = 2 evidence alone.
  R3  The width ordering WEAKENS as k rises: Spearman(source_fp, V)
      computed within each k level decreases from k = 0 to k = 8. This is
      the saturation hypothesis -- that duplicates raise the failure at
      every width until width stops mattering -- and it is what would
      explain the crossed run's disagreement.
  R4  NO PREDICTION on whether any k restores source blindness, or on the
      value of k at which the ordering vanishes.
  R5  Ghost median at or below 0.005 in every cell; dirty cells are
      excluded from R2 and R3 and reported separately.

## The rule, fixed now

  INTERACTION CHARACTERISED  R2 and R3 hold. The crossed run's disagreement
                             is explained: redundancy saturates the failure,
                             width matters only at low redundancy, and the
                             licensing premise depends on both jointly. The
                             saturation question closes with a described
                             interaction rather than a contradiction.
  NOT AS THEORISED           R2 or R3 fails. The saturating-interaction
                             story is wrong, the crossed disagreement stands
                             unexplained, and that is reported as such.
  DISSOLVED                  R1 fails: there was never a k = 0 ordering to
                             explain.

## Void conditions

Void if b is not 2V in every cell, if k levels or seeds change after any
result, if dirty-ghost cells enter R2 or R3, or if R4's declined
predictions are retrospectively claimed.
