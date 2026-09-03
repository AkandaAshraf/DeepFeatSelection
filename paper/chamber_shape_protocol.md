# Pre-registration: does the chamber's SHAPE explain the tension?

Declared 2026-09-02, before any chamber-shaped synthetic system was run.

## The tension this exists to resolve

The sink-calibrated bar closed the synthetic source-detection line: at
coupling 0.50, sensitivity 0.53 at a 5% sink false-alarm rate, against a
declared 0.80. The chamber result stands at AUC 0.916. The status section
now states three candidate explanations and admits it cannot choose:

> Either the chamber's structure is more favourable than this synthetic
> family, or AUC is more forgiving than a thresholded rate, or the chamber
> result is fortunate. Our own evidence cannot distinguish these.

The first of the three is directly testable, and cheaply. The synthetic
family is 3 sources against 6 sinks and 6 isolated channels. The chamber is
**2 actuators against 11 sensors**. Those are different shapes, and
`coupled()` already exposes `n_src` and `n_sink`, so the chamber's shape can
be built in the synthetic family where ground truth is exact.

## Design, fixed now

Identical machinery, bar and protocol to `paper/sink_bar_protocol.md` --
same q95(sink) calibration, same alpha = 0.05, same disjoint calibration and
test seeds, same b = 4V, same coupling 0.50. The ONLY thing that changes is
the system's shape.

  SYNTHETIC SHAPE (incumbent)   n_src=3,  n_sink=6,  n_iso=6      V=15
  CHAMBER SHAPE                 n_src=2,  n_sink=11, n_iso=0      V=13
  CHAMBER SHAPE + ISOLATED      n_src=2,  n_sink=11, n_iso=2      V=15
      the third arm holds V fixed at 15 so that shape and width are not
      varied together -- the confound that made the crossed-saturation
      experiment inconclusive, and which is not repeated here.

  CALIBRATION seeds 500-529, TEST seeds 600-629, 30 runs each, per shape.

Both AUC and thresholded sensitivity are computed for every arm, because the
second candidate explanation (AUC is more forgiving than a rate) is
separable only if both are measured on the same runs.

## Predictions, fixed now

  C1  DECISIVE. At the chamber shape, sensitivity at a 5% sink false-alarm
      rate exceeds the incumbent shape's 0.53. If it does, the chamber's
      structure is a sufficient explanation for the tension and the paper
      says so.
  C2  NO PREDICTION on whether chamber-shape sensitivity reaches the 0.80
      bar. Exceeding 0.53 resolves the tension; reaching 0.80 would
      additionally reopen the line, and that is a separate and higher bar
      which is NOT claimed in advance.
  C3  AUC exceeds thresholded sensitivity in every arm. This is the second
      candidate explanation, and it is expected on general grounds -- AUC is
      threshold-free -- so it is recorded now rather than presented later as
      a discovery.
  C4  The third arm (chamber shape at V=15) matches the second arm within
      0.15 sensitivity. If it does not, shape and width are entangled after
      all and no arm is interpretable.

## The rule, fixed now

  SHAPE EXPLAINS    C1 holds. The tension resolves: the chamber's 2-against-11
                    structure is more favourable than the 3-against-6 family,
                    the closure stands as a statement about the synthetic
                    family rather than about the statistic everywhere, and
                    the paper's three-way admission collapses to one.
  SHAPE DOES NOT    C1 fails. Shape is eliminated, two candidates remain
                    (AUC forgiveness, or fortune), and the chamber result
                    becomes materially more suspect -- which is reported as
                    the headline, against our interest.

## Void conditions

Void if the bar, alpha, coupling or seeds differ from the sink-bar protocol;
if C4 fails and any arm is nonetheless interpreted; or if C2's higher bar is
retrospectively presented as having been predicted.
