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

---

## Result (2026-08-22)

90 cells: V in {30, 60, 120}, b in {8, 16, 32, 64, 128}, float and binary
codes, three seeds. The reference cell (V=60, b=32, float) reproduced the
boundary map exactly, so this is commensurable with what is already
published.

### BS1 HOLDS, and strongly: the compression limit is real and relievable

Median recall, float codes:

  b       8      16      32      64     128
  V=30   0.56   0.80    0.88    0.92    0.92
  V=60   0.18   0.18    0.18    0.34    0.78
  V=120  0.22   0.23    0.23    0.27    0.41

At V=60, widening from the deployed b=32 to b=128 takes recall from 0.18 to
0.78 - a 4.3-fold improvement on the same data with no other change. The
detection cap the boundary map found is NOT a property of the method. It is
a property of a 32-wide code.

Flagged-channel counts confirm the cap lifts rather than the accounting
changing: at V=60 the number surfaced goes 9, 9, 9, 17, 39 as b grows.

### BS2: scaling b in proportion to V does NOT suffice

  V=30  at b=32   recall 0.88
  V=60  at b=64   recall 0.34
  V=120 at b=128  recall 0.41

Doubling V while doubling b loses more than half the recall. Reading the
grid for what width restores recall to ~0.8: V=30 needs b=16-32, V=60 needs
b=128. That is a four- to eight-fold increase in b for a doubling of V -
SUPERLINEAR.

The practical rule this suggests is b of order 2V, which is a substantial
finding about the method's character: the delay embedding is V*E = 3V
dimensions, so b = 2V is a compression of only 1.5x. MACE does not work
through a tight bottleneck. It needs a code nearly as wide as its input, and
the "bottleneck" framing understates how much capacity the statistic
requires.

### BS3 HOLDS: binary loses at matched width, and the gap widens

Median recall by width:

  b        binary   float
   8        0.16     0.22
  16        0.21     0.18
  32        0.20     0.23
  64        0.23     0.35
 128        0.28     0.76

### BS4, which had no prediction: CAPACITY, not geometry

Binary is not blocked - it improves monotonically with width, and no
dimensions collapsed (dead dimensions were zero in every cell, so the
straight-through estimator trained properly). Binary at b=128 (recall 0.28)
sits between float at b=32 (0.23) and float at b=64 (0.35), so roughly FOUR
binary dimensions substitute for one float dimension in this task.

That is the capacity answer. The code's continuous geometry is not doing
special work; a discrete encoding reaches the same place at about four times
the width. What float buys is bits per dimension, not interpolation.

The caveat that keeps this honest: float does something binary does not at
the top of the range, accelerating from 0.35 to 0.76 between b=64 and b=128
while binary moves 0.23 to 0.28. Either binary needs widths beyond those
tested to show the same jump, or the constant factor grows once the code
approaches the input dimension. This grid cannot separate those, and wider
binary codes were not run.

### BS5 and BS6 HOLD: nothing breaks when the code widens

  ghost_max is FLAT across every width and both code types, 0.0013 to
  0.0015. A wider code does not inflate the null.
  source false positives are 0.000 in all 90 cells.
  precision is 1.00 in all 90 cells.

Widening the code costs nothing in precision, in source blindness, or in the
ghost. The only cost is training time.

### Consequence for the published scans

The zebrafish scan covered 71,721 channels at b=32. This grid shows b=32
already limiting at V=60. Its recall was therefore very likely far below
what the data supported.

That qualifies RECALL, not precision: precision held at 1.00 across every
cell here, and the paper's central claim is precision-first with recall
explicitly disclaimed. The published driven cores are not wrong; they are
smaller than they needed to be. No published claim requires correction, and
the paper's existing statement that absence is not evidence of autonomy is
strengthened - absence was partly a capacity artefact.

Extrapolation to V=71,721 is not attempted. The grid stops at V=120 and the
scaling is superlinear, so any extrapolation would be unsupported.

### Recommendation, and what it still needs

b should be chosen from V, not fixed. A defensible starting rule from this
grid is b of order 2V for full recall, or b of order V/2 for a fast scan
accepting reduced recall.

This has been measured on one generating process. Before it becomes
deployment guidance it needs confirmation on a different system class -
declared separately, not tested here.
