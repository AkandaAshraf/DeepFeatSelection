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
