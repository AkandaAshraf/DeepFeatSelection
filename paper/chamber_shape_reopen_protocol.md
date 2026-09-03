# Pre-registration: does the source-detection line reopen at the chamber's shape?

Declared 2026-09-02, before any run on the seeds named below.

## Why this exists, stated against our own interest

Yesterday the sink-calibrated bar CLOSED the synthetic source-detection line:
sensitivity 0.53 (now corrected to roughly 0.5-0.7) against a declared 0.80,
on the 3-source/6-sink family. Today's shape experiment, run to explain a
different question, produced sensitivity **0.867 and 0.967** at the chamber's
2-source/11-sink shape -- comfortably past that same 0.80.

That was explicitly NOT claimed. C2 of the shape protocol declined in advance
to predict whether the chamber shape would clear 0.80, and importing a bar
declared for one shape onto another after seeing the numbers is what Rule 90
forbids. But the consequence is that the manuscript currently states the line
is closed while carrying a table that suggests otherwise, and that
inconsistency is ours to resolve rather than leave for a reviewer.

This is the confirmatory test, on seeds not yet used for anything.

## Design, fixed now

Identical in every respect to `paper/sink_bar_protocol.md` -- the protocol
that produced the closure -- except the system's shape and the seeds:

  SHAPE        n_src = 2, n_sink = 11, n_iso = 2  (V = 15 + ghost = 16)
               The V=15 arm, so width matches the incumbent family exactly
               and only the source:sink ratio differs. Today's C4 check
               showed the two chamber arms agree within 0.10, so this arm is
               representative of the shape rather than of its width.
  BAR          q95 of the SINK outflow distribution, alpha = 0.05.
  CAPACITY     b = 4V, coupling 0.50, n = 4000, 25 epochs, 2 models.
  CALIBRATION  seeds 700-729   (unused by any prior experiment)
  TEST         seeds 800-829   (unused by any prior experiment)

Both sensitivity and per-run AUC are recorded, as in the shape run.

## Predictions, fixed now

  P1  DECISIVE. Sensitivity at the chamber shape is at or above 0.80 -- the
      same bar, unchanged, that the incumbent shape failed. This is the
      claim today's numbers imply, and it is now being made in advance
      rather than read off afterwards.
  P2  NO PREDICTION on the exact value. Today's two arms gave 0.867 and
      0.967, a spread of 0.10 across shapes that differ only in isolated
      channels, so a point prediction would be false precision.
  P3  The calibrated bar lands in the same region as today's chamber arms
      (0.0113-0.0138). A materially different bar would mean the sink
      distribution is not stable across seed blocks and the comparison to
      the incumbent is unsafe.
  P4  DISQUALIFYING. If sensitivity falls below 0.80, today's 0.867/0.967
      were seed-block fortune, the closure stands unqualified, and the
      shape explanation for the synthetic/real tension weakens
      correspondingly -- because that explanation and this prediction rest
      on the same measurement.

## The rule, fixed now

  REOPENS AT THIS SHAPE   P1 holds. The line is not closed as a property of
                          the statistic; it is closed for source:sink ratios
                          near 1:2 and open near 1:5.5. Section 12's verdict
                          is rewritten to state the closure with its shape
                          condition attached, and the chamber result gains a
                          synthetic footing rather than standing alone.
  CLOSURE STANDS          P4 fires. The closure is unconditional on the
                          evidence available, today's high sensitivities are
                          reported as seed-block variation, and the shape
                          explanation is downgraded.

## Void conditions

Void if the bar, alpha, coupling, capacity or shape differ from those above;
if seeds 700-829 have been used by any earlier run; if the outcome is
reported without the incumbent 3/6/6 figure beside it; or if P1 fails and is
presented as anything other than the closure standing.

## Result (2026-09-03, seeds 700-829, 60 runs, `ExpOutput/chamber_shape_reopen/`)

  bar = q95(sink, calibration) = +0.01498
  P1  DECISIVE: 25/30 test runs clear -> sensitivity 0.83, Wilson [0.66, 0.93]
      vs bar 0.80. HOLDS, by two runs.
  AUC (per-run source vs sink, median) 0.864; source median +0.02312, sink
  median +0.00559. Incumbent 3/6/6: 0.53 (sink-bar seeds) / 0.667
  (chamber-shape seeds).
  Failing test seeds: 801, 816, 826, 828, 829. Two of them (801, 829) are
  dead runs (source median +0.0006 and +0.0000); the other three sit
  between +0.011 and +0.015, just under the bar.

VERDICT: REOPENS AT THIS SHAPE, by the rule fixed above. The margin is two
runs of thirty, and the interval reaches 0.66, so this is a pass at the
declared bar and not a demonstration of a comfortable one.

### P3 discrepancy, disclosed

The protocol names the region 0.0113-0.0138 (yesterday's two chamber arms).
The script committed before the run codes P3 as 0.008 <= bar <= 0.020. The
observed bar, +0.01498, is inside the coded window and outside the narrower
one. The coded window is the operative test because it was committed before
any run; but the reader should know that against the protocol's own words
P3 would have failed, and that the three sink-bar values now on record for
this shape (0.0113, 0.0138, 0.0150) span a factor of 1.3 across seed blocks.
The sink distribution is stable enough for a q95 to be meaningful, not
stable enough to quote to three figures. The bar is therefore reported as
"0.011-0.015 across seed blocks" wherever it appears.

### What the generator audit of the same day does to this result

`paper/generator_audit.md`: 24 of these 60 runs contain a phase-locked
source (see that note for the definition); all four dead runs do. On the 19
test runs with no locked source, 19/19 clear the same bar. So the result
above is, if anything, a lower bound on what the statistic does at this
shape; the pass is not an artefact of locking, but the two-run margin is.
The pre-registered re-test on a clean generator is
`paper/clean_generator_protocol.md`.
