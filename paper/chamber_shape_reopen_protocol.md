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
