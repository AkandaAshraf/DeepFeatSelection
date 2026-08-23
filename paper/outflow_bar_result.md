# Result: the ghost is the wrong null for outflow, and 0.01 is not retired

2026-08-23. Pre-registration: paper/outflow_bar_protocol.md, committed at
4a19b1a before the experiment was written or run. 90 runs, 24 minutes.

## Verdict as declared: DOES NOT SEPARATE

  O1  calibration ghost null, 30 runs, seeds 100-129
      median -0.0007, range [-0.0016, +0.0003]. A null, as required.

  O2  bar = q95(ghost) = -0.00005
      quantiles: q50 -0.0007  q75 -0.0004  q90 -0.0002  q95 -0.0001
                 q99 +0.0002
      Below 0.01, so O5 does not fire and the line does not close on that
      rule. The declared constant sat roughly two orders of magnitude above
      the null's 95th percentile.

  O3  sensitivity at coupling 0.50: 29 of 30 runs clear the bar, 0.97. PASS.

  O4  control at coupling 0.30: 25 of 30 clear it, 0.83. FAIL against the
      declared bar of 0.20.

By the rule fixed in advance, O4's failure gives DOES NOT SEPARATE.

## A design error in the pre-registration, which I own

O4 called coupling 0.30 "the strongest regime where it is known NOT to work".
That phrase was inherited from the coupling sweep, where 0.30 failed to clear
a margin of 0.01. It is a statement about the OLD bar, not about ground
truth: at coupling 0.30 sources genuinely drive sinks. A bar calibrated to
exclude no-influence channels should flag them. The control was defined in
terms of the threshold the experiment set out to replace, and it did not
survive the replacement.

So O4's failure does not, by itself, mean what its text says it means.

## The fact that does settle it, and points the same way

Every run recorded sink outflow. At coupling 0.50, against the calibrated
bar:

  sources clear   29 / 30
  SINKS clear     28 / 30

The bar admits driven channels almost exactly as often as it admits sources.
That is fatal to a null-calibrated bar and does not depend on the control
error at all.

The reason is visible in the existing coupling sweep, whose isolated channels
are a true within-run negative control - real dynamics, coupled to nothing:

  coupling   source    sink     ISOLATED   ghost
  0.05      -0.0004  -0.0007    -0.0007   -0.0007
  0.15      +0.0000  -0.0006    -0.0006   -0.0011
  0.30      +0.0009  -0.0003    -0.0006   -0.0006
  0.50      +0.0100  +0.0027    -0.0006   -0.0008
  0.70      +0.0099  +0.0062    -0.0005   -0.0012

Isolated channels sit exactly at the ghost level at every coupling, so the
ghost IS a valid null for "influences nothing". The problem is that sinks are
not at the null. A sink influences nothing either, but it CARRIES A PROXY of
its driver, so masking it removes information the rest of the code was using.

The quantity that confuses outflow is therefore not influence but
proxy-carrying, and the ghost has neither. Calibrating on it produces a bar
that every driven channel clears.

Note also that the sink proxy GROWS with coupling: +0.0027 at 0.50 to +0.0062
at 0.70, against a source signal that is flat at +0.0100 and +0.0099. That is
risk C3, declared before the coupling sweep and recorded there as survived,
returning at the top of the range. The source-sink gap narrows as driving
strengthens.

## What this means for the constant

0.01 is NOT retired. It is not replaced either, and it is now less
comfortable than before: what it was actually doing was suppressing the sink
proxy, not excluding a null. It has been serving as an implicit sink
threshold while being documented as a margin over the ghost, and as a sink
threshold it remains entirely uncalibrated.

Rule 84's second item stays OPEN, with a sharper specification than it had.
The bar for outflow must be calibrated against the SINK distribution, not the
ghost: the question is not "could a channel with no influence score this?"
but "could a channel that merely carries a driver's signal score this?"

## Read off existing data, claimed as nothing

At coupling 0.30 source outflow exceeds sink outflow in 25 of 30 runs, and at
0.50 in 27 of 30. The gap statistic may work below where the 0.01 margin
placed its floor. This comes from re-reading runs collected for another
purpose and is recorded here only so it is not lost. It is not a finding and
would need its own pre-registration.

## Rules

88. A null control must exclude what actually confuses the statistic. The
    ghost excludes influence; what confuses outflow is proxy-carrying. A bar
    calibrated against the wrong null admits every driven channel while
    looking rigorous.

89. When a threshold is replaced, re-derive every control that was defined in
    terms of the old one. O4 was inherited verbatim from the coupling sweep
    and described a regime as "known not to work" when what was known was
    only that it failed the threshold being discarded.
