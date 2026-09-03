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

---

## Result (2026-09-02): SHAPE EXPLAINS. And C3 eliminates a second candidate.

30 calibration + 30 test runs per arm, 32.5 min.

  shape                             bar      sens    AUC    src med  sink med
  synthetic (3 src / 6 sink / 6 iso)  0.0078  0.667  0.667   0.0116   0.0020
  chamber   (2 src / 11 sink / 0 iso) 0.0138  0.867  0.864   0.0220   0.0059
  chamber at V=15 (2 / 11 / 2)        0.0113  0.967  0.864   0.0226   0.0058

  C4  the two chamber arms differ by 0.10, inside the declared 0.15, so
      shape and width are NOT entangled and the arms are interpretable.
  C1  DECISIVE: 0.867 against the incumbent's 0.667. **SHAPE EXPLAINS.**

### The incumbent arm reads 0.667 here and 0.53 in the sink-bar run

Same configuration, different seed blocks (300-429 there, 500-629 here).
The gap is 0.137, which is 1.08 standard errors of the difference at n = 30
-- ordinary sampling noise, not a discrepancy. Two consequences, both
stated rather than buried:

  1. A sensitivity estimated on 30 runs carries SE ~0.09, so 0.53 has a 95%
     interval of [0.35, 0.71]. The closure should not have been reported as
     though 0.53 were precise, and is corrected here: the incumbent shape
     sits somewhere near 0.5-0.7.
  2. The closure verdict is unaffected. Both estimates fall below the
     declared 0.80, so the synthetic line closes on either seed block.

### C3 FAILED, and its failure is more useful than its success would have been

C3 predicted AUC would exceed thresholded sensitivity in every arm, on the
general ground that AUC is threshold-free. It does not: 0.667 vs 0.667,
0.864 vs 0.867, and 0.864 vs 0.967. AUC is equal or **lower**, never higher.

That eliminates the second of the three candidate explanations. The status
section proposed that the chamber's 0.916 might be inflated because AUC is
more forgiving than a thresholded rate. On this evidence it is not more
forgiving; if anything it is more conservative. The caveat that AUC here is
per-run source-vs-sink while the chamber's 0.916 is pooled over channels
still applies, so this is strong evidence rather than proof.

### The tension resolves

Of the three candidates the status section could not choose between:

  1. the chamber's structure is more favourable   -- CONFIRMED (C1)
  2. AUC is more forgiving than a rate            -- ELIMINATED (C3 failed)
  3. the chamber result is fortunate              -- no longer needed

The mechanism is legible. The chamber has 2 actuators driving 11 sensors,
roughly 5.5 sinks per source; the synthetic family has 3 sources driving 6
sinks, 2 per source. Outflow measures how much the rest of the system's code
depends on a channel, so a source driving more sinks has more influence to
detect. **Marginal outflow improves as sinks per source rises.**

That is the exact opposite of the conditional variant's envelope, where many
sinks per source make the source recoverable from what it drives and destroy
C1's advantage (Rule 93). The two variants are pushed in opposite directions
by the same structural parameter, which is a coherent account of both
results rather than two unrelated observations.

### What is NOT claimed

C2 explicitly declined to predict whether the chamber shape would clear 0.80.
It does -- 0.867 and 0.967 -- and that is reported, not claimed. Clearing
0.80 at the chamber's shape would, on the sink-bar protocol's own rule,
correspond to a working statistic at that shape. But that bar was declared
for a different experiment on a different shape, and importing it here after
the fact is precisely the move Rule 90 forbids. What this experiment
establishes is the comparison it declared: chamber shape beats synthetic
shape. Whether the line REOPENS at the chamber's shape needs its own
pre-registration, on fresh seeds, and is not asserted here.
