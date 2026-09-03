# Pre-registration: calibrating the outflow bar against the sink distribution

Declared 2026-09-02, before any sink-calibrated bar was computed.

## Why the last attempt failed, and what that implies

Rule 84's second open item. The deployed outflow bar of 0.01 was declared and
never justified, and the line was reopened on a margin of 0.0107 -- clearing
by seven percent a number chosen from nothing.

The first repair attempt (`paper/outflow_bar_protocol.md`) calibrated the bar
as q95 of the GHOST null and failed decisively: at coupling 0.50 the bar came
out at $-0.00005$, and **sinks cleared it 28 times in 30**. The diagnosis was
that the ghost is the wrong null. Isolated channels -- real dynamics, coupled
to nothing -- sit exactly at the ghost level at every coupling, so the ghost
is a valid null for "influences nothing" and no null at all for
"carries a proxy of an influencer", which is what actually confuses outflow.

That result named the fix without performing it:

> Rule 84's second item stays OPEN, with a sharper specification than it had.
> The bar for outflow must be calibrated against the SINK distribution, not
> the ghost: the question is not "could a channel with no influence score
> this?" but "could a channel that merely carries a driver's signal score
> this?"

This is that calibration.

## The bar

For a stated false-alarm rate alpha against the confuser that actually
matters:

    bar = q(1 - alpha) of the SINK outflow distribution

Sinks are the right null because they are the failure mode: a sink
influences nothing, carries a strong proxy of its driver, and is what the
ghost-calibrated bar admitted. alpha = 0.05 is retained from the previous
protocol -- not calibrated by this experiment either, but a stated operating
point that means something specific: the rate at which a proxy-carrying
channel is called a source.

## Calibration and test are disjoint draws, as before

  CALIBRATION  seeds 300-329, coupling 0.50, 30 runs. SINK outflow only.
               These define q95 and nothing else is taken from them.
  TEST         seeds 400-429, coupling 0.50, 30 runs. Source outflow,
               scored against the calibrated bar.
  CONTROL      seeds 400-429, coupling 0.30. Declared as a WEAK-COUPLING
               control, not as a null: at 0.30 sources genuinely drive
               sinks, and the previous protocol's error was calling this
               regime "known not to work" when what was known was only that
               it failed the old 0.01 bar. Sensitivity there is reported for
               description, with no pass/fail attached.

Deployed configuration throughout: b = 64 (= 4V, the outflow capacity
requirement), V = 15 plus ghost, n = 4000, 25 epochs, 2 models.

## Predictions, fixed now

  S1  The sink-calibrated bar is HIGHER than the ghost-calibrated one
      ($-0.00005$). Trivially expected -- sinks score above the ghost by
      construction -- and recorded so it cannot later be presented as a
      finding.
  S2  DECISIVE. At coupling 0.50, sensitivity (fraction of the 30 test runs
      whose median source outflow clears the bar) is at least 0.80. This is
      the number that should have been reported when the line was reopened,
      and it is the first honest statement of what outflow can do at a
      stated false-alarm rate against the confuser that matters.
  S3  DISQUALIFYING. If the sink-calibrated bar exceeds the observed source
      outflow at coupling 0.50 for most runs -- that is, if sinks and
      sources are not separable at a 5% sink false-alarm rate -- then the
      0.01 bar was concealing an overlap rather than a margin, and the
      outflow line closes. Declared now so that outcome is a result.
  S4  NO PREDICTION on how the sink-calibrated bar compares to 0.01 itself.

## The rule, fixed now

  ADOPTED   S2 holds. The reported quantity for outflow becomes sensitivity
            at a 5% SINK false-alarm rate, 0.01 is retired from this line,
            and the paper quotes the calibrated bar with its alpha.
  CLOSED    S3 fires. The margin the line was reopened on does not survive
            calibration against the right null, and that is the headline.

## Void conditions

Void if the bar is computed from the test seeds, if alpha changes after any
source outflow is seen, if seeds are added or dropped after a result, or if
S3 fires and is reported as anything other than the line closing.

---

## Result (2026-09-02): CLOSED. S3 fires.

30 calibration + 30 test + 30 descriptive runs, b = 4V, 17.1 min.

  S1  sink-calibrated bar q95(sink) = +0.01051
      sink distribution: median +0.00201, range [-0.00025, +0.01471]
      ghost-calibrated bar was -0.00005; the declared bar was 0.01
      Higher than the ghost bar, as declared. Recorded, not a finding.

  S2  DECISIVE, coupling 0.50: 16 of 30 runs clear the bar.
      **Sensitivity 0.53**, against a declared bar of 0.80. FAIL.
      source median +0.01064   sink median +0.00174   margin +0.00013

  S3  0.47 of runs fail to clear. Sources and sinks OVERLAP at a 5% sink
      false-alarm rate.

  Descriptive, coupling 0.30: 0.00 of runs clear. No pass/fail attached.

**VERDICT: CLOSED, by the rule fixed in advance.** At a stated 5% false-alarm
rate against the confuser that actually matters, outflow separates sources
from sinks in barely half of runs. The 0.01 bar was concealing an overlap,
not describing a margin.

### The arithmetic that makes this unambiguous

The declared bar was 0.01. The sink-calibrated q95 is +0.01051 -- within 5%
of it. So the constant was, accidentally, an almost exactly correct sink
threshold. What was never reported is the sensitivity AT that threshold: 0.53.
The line was reopened on a "margin of +0.0107 over the ghost", and that
margin is now revealed as the distance between the source median and a null
that was never the relevant one. Against the relevant null the margin is
+0.00013 -- three orders of magnitude smaller, and indistinguishable from
zero given a sink distribution whose own range reaches +0.01471.

### What closes, and what does not

CLOSES: the claim that outflow reliably separates sources from sinks on
synthetic coupled-logistic systems at a stated false-alarm rate. Three prior
gates (capacity, coupling, maturity) tested outflow against the ghost or
against sinks without a calibrated rate, and all three passed. This is the
first test at a stated operating point against the right null, and it fails.

DOES NOT CLOSE, and must not be conflated: the causal-chamber result. That
is a different system with different structure (2 actuators against 11
sensors rather than 3 sources against 6 sinks), and it is measured by AUC
over channels, which is threshold-free, rather than by a per-run threshold.
AUC 0.916 with a run-clustered CI of [0.862, 0.962] stands as measured.

But the tension is real and is stated rather than resolved: **on synthetics
outflow separates sources from sinks only half the time at a 5% sink
false-alarm rate, while on the chamber it ranks actuators above sensors at
0.916.** One of three things is true -- the chamber's structure is more
favourable than the synthetic family, AUC is more forgiving than a
thresholded rate, or the chamber result is fortunate. This experiment cannot
distinguish them, and no further reading of these cells will.
