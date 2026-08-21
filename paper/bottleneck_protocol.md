# Pre-registration: bottleneck width, and continuous versus binary codes

Declared 2026-08-22, before the experiment was written or run.

## Why width

Two findings converged on an untested conclusion:

  BOUNDARY MAP      the number of channels surfaced is CAPPED at 9-27
                    regardless of V, so recall is approximately a constant
                    over V rather than decaying gently
  PROPOSITION 2 GAP 80-92% of the shortfall against an oracle readout is
                    COMPRESSION - information the bottleneck never carried -
                    and only 8-19% is the readout

Both say the bottleneck, fixed at 32 throughout, is the binding constraint
at scale. Neither tested whether widening it helps. If recall is genuinely
capped by code capacity, scaling b with V should restore it; if recall is
capped by something else, widening will do nothing and both readings were
wrong about the mechanism.

## Why binary

A float bottleneck of width b is a continuous EMBEDDING: b real numbers,
carrying in principle unlimited information. A binary bottleneck of the same
width is a discrete ENCODING: b bits, carrying at most b bits. Comparing
them separates two explanations that the width axis alone cannot:

  CAPACITY  what matters is how much the code can carry, in which case
            binary at width b should behave like float at some much smaller
            width, and matching capacity should match performance
  GEOMETRY  what matters is the code's continuous structure - that the ridge
            readout can interpolate along it - in which case binary will
            underperform float at ANY width

The second would be a substantive fact about why the method works, and it is
not currently known either way.

Binary codes are produced by a hard sign threshold with a straight-through
gradient estimator, which is the standard construction; the architecture is
otherwise identical.

## Design

  V         30, 60, 120
  b         8, 16, 32, 64, 128
  code      float, binary
  seeds     0, 1, 2

n = 4000, coupling = 0.20, all other constants as deployed. Ground truth by
construction, as in the boundary map. Measured per cell: precision, recall,
number of channels flagged, ghost median and maximum, saturation, and source
false-positive rate.

## Predictions, fixed before running

  BS1  Recall rises with b at fixed V. If it does not, code capacity is NOT
       the binding constraint and both motivating findings were misread.
  BS2  At fixed b, recall falls with V, reproducing the boundary map. Scaling
       b in proportion to V should hold recall roughly constant; whether it
       does is the practical question.
  BS3  Binary underperforms float AT MATCHED WIDTH. Trivially expected on
       capacity grounds and recorded so it cannot later be presented as a
       finding.
  BS4  NO PREDICTION on whether binary at a larger width matches float at a
       smaller one. This is the capacity-versus-geometry question and the
       reason the binary arm is here. If matched capacity gives matched
       recall, the code is a channel and its geometry is incidental. If
       binary lags at every width, the continuous structure is doing work.
  BS5  Ghost stays clean for BOTH code types at every width. A wider code has
       more capacity to fit noise; if the ghost rises with b, that bounds
       usable width regardless of recall.
  BS6  Source false positives stay at zero. They held across all 51 cells of
       the boundary map at b = 32; whether that survives a wider code is
       unknown and would be the most consequential failure this could find.

## What would make this actionable

If BS1 and BS2 hold, the deployment guidance changes: b should be chosen
from V rather than fixed, and the published scans - including the
71,721-neuron zebrafish case run at b = 32 - were operating far below the
capacity their systems needed. That would qualify their recall, not their
precision.

## Void conditions

Void if the grid, the code constructions or the predictions are altered
after any result is seen, or if a wider code is recommended without the
ghost and source checks passing.
