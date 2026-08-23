# Audit: which declared constants were ever calibrated?

2026-08-23. Prompted by the chamber void, whose verdict rested on a self-R2
threshold of 0.95 that had never been checked against a system where the
method works - and which, when finally checked, fired there too (Rules 79,
83).

That defect has a shape: a constant fixed in advance, reported as a gate,
never given a positive control. This audit asks the same question of every
constant currently load-bearing in this project. It changes no result; it
records which numbers are evidence and which are assertions.

## Calibrated - a positive control exists

  GHOST BAR 0.005
    On systems where the method demonstrably works, ghost_max is 0.0013 to
    0.0015 and FLAT: across all 90 bottleneck cells and all 51 boundary-map
    cells, at every code width and both code types. The bar sits about four
    times above the observed null on working systems. Extensive positive
    control, accumulated rather than designed, but real.

  LENGTH FLOOR ~2,000
    The boundary map swept n over 1000, 2000, 4000, 8000, 16000 and measured
    recall at each. The floor is read off that sweep, not asserted.

  GATE-STATISTIC THRESHOLDS
    Not constants. Each candidate statistic has its threshold set to reject
    exactly 10% of good runs, so the comparison is at matched specificity by
    construction.

  FLAGGING THRESHOLD max(0, ghost panel max)
    Not a constant. Derived per scan from that scan's own surrogates.

  DEPMAP tau_prox = 0.8
    Anchored by controls in both directions: same-complex pairs score 0.90 to
    0.98, the unrelated pair PSMA1-KRT1 scores 0.04. The threshold sits below
    the known-positive band and far above the known-negative one. The control
    set is THIN - a handful of anchors, not a distribution - but it exists and
    points the right way, and the Perturb-seq replication independently
    re-derived a threshold by base-rate matching rather than inheriting this
    one. This is the constant under review at PLOS; it is not the chamber
    defect.

  DATASET FITNESS L50 = +0.0136
    Calibrated by construction, from the weakest coupling at which outflow is
    known to work.

## Uncalibrated - no positive control

  SATURATION, "self-R2 near 1"                      MOST CONSEQUENTIAL
    Proposition 1 licenses the guarantee only where the self-baseline
    saturates. "Near 1" has never been made a number. There is no saturation
    GATE anywhere in the codebase. What exists is a summary statistic,
    (self_r2 > 0.9).mean(), appearing in boundary_map.py, bottleneck_scaling.py
    and duplicate_channel.py, where 0.9 is an unexamined constant that is
    reported and never enforced.

    This matters more than the others because the failure mode is documented
    and severe: off the ceiling, up to 50% of sources are flagged, and the
    ghost panel passes every such scan (Rules 71-74). So the condition that
    licenses the central guarantee is qualitative, while the consequence of
    its absence is quantified. Whether 0.9 is the right operating point has
    never been tested against systems where blindness holds versus fails.

    This is not an error in the MACE manuscript, which states the premise as
    a premise and records the conditional blindness. It is a gap a reviewer
    can reasonably name, and there is currently no answer.

  OUTFLOW MARGIN BAR 0.01
    Declared as the bar used throughout this line and never justified. The
    line was REOPENED on a margin of 0.0107 at coupling 0.50 - clearing by
    seven percent. A calibration argument is available after the fact, that
    0.01 is roughly ten times the measured ghost of 0.001, but it was never
    made in the protocol and the bar was not chosen that way. Given how
    narrowly the reopening cleared it, this is the second item worth fixing.

  IEEG VALIDITY GATE rho > 0.30
    Declared, never calibrated. MOOT: the iEEG discovery result failed its
    pre-registered replication and was withdrawn, so this constant underpins
    no live claim.

  TF/TORCH FIDELITY F1 >= 0.90, F2 >= 0.70
    Declared. Low stakes - an agreement check between two implementations -
    and both passed comfortably at 0.955 and 0.958.

  SOURCE FALSE-POSITIVE RATE > 0.05 = "BAD"
    Declared. Never binds: the observed rate is 0.000 in all 90 bottleneck
    cells and all 51 boundary-map cells.

## What this changes

Nothing already published is retracted by this audit. The DepMap threshold
under review at PLOS is controlled in both directions, which was the outcome
that most needed checking and is the reason the audit was run before a
reviewer ran it.

Two items are open, in order:

  1. Make saturation a number. Pre-register a saturation gate with a positive
     control - systems where source blindness holds and systems where it
     fails - and report the operating point rather than inheriting 0.9 from a
     line of summary code.
  2. Justify or replace the 0.01 outflow bar, given that the line was
     reopened on 0.0107.

Rule added:

84. A licensing condition stated qualitatively is not a gate. If a guarantee
    holds only under a premise, the premise needs a number and that number
    needs a positive control, or the guarantee must be reported as
    unlicensed.
