# Pre-registration: making saturation a number

Declared 2026-08-23, before the experiment was written or run.

## Why

Proposition 1 licenses MACE's guarantee only where the self-baseline
saturates. "Near 1" has never been made a number (threshold audit,
paper/threshold_audit.md, Rule 84). The constant 0.9 exists in the codebase
in three places and was calibrated in none of them:

  boundary_map.py:195   saturation = (self_r2 > 0.9).mean()   REPORTED only
  bottleneck_scaling.py:143, duplicate_channel.py:165          same statistic
  boundary_map.py:32    DONOR_R2 = 0.9, which decides which channels may
                        donate a ghost surrogate                ENFORCED

The second is enforced and unexamined, which also means the ghost panel
itself degrades where saturation fails: too few qualifying donors triggers
the MIN_DONORS fallback and ghosts are drawn from the whole pool, including
channels whose self-baseline has headroom.

The cost of the missing gate is already quantified. Source blindness is a
property of the saturated regime, not of the method: source false positives
go from 0.000 to 0.20-0.23 with individual cells at 0.40 and 0.50 as the
self-baseline falls, and G3 passes every one of those scans because it
watches the ghost median (Rules 71, 72). The follow-up established that no
ghost statistic reaches usable sensitivity for this failure (Rules 73, 74).

So the failure is measured, the existing gate cannot catch it, and the
quantity that would catch it - self-R2 - is directly observable without any
surrogate. What is missing is the number and the evidence that it transfers.

The existing evidence is also too coarse to supply it. The observation-noise
extension measured self-R2 at 0.998, 0.644, 0.132 and 0.012. The threshold
must lie between 0.644 and 0.998, and nothing was measured there. The
inherited 0.9 sits inside that gap.

## The gate to be tested

PER-CHANNEL, not per-panel. The claim "channel q shows inflow" is licensed
only if channel q's OWN self-R2 is at or above s*. This is the deployable
form: a scan reports its licensed subset rather than passing or failing whole.

  retained(s)   channels with self_r2 >= s
  source FP(s)  flagged AND source AND retained, over source AND retained
  TP keep(s)    flagged AND driven AND retained, over flagged AND driven
                (that is, what fraction of true detections the gate costs)

## Design

System: boundary_map.make_system, the standard coupled logistic panel,
redundancy 0, n = 4000, all deployed constants (b = 32, E = 3, 20 epochs,
2 models, 30 ghosts). Observation noise is added on top of the system's own
0.005, which is how saturation is moved.

  DISCOVERY   V = 15, coupling 0.35, seeds 0, 1, 2
              obs noise 0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.20, 0.40
              27 cells, sampled densely below 0.10 because that is where the
              earlier data jumps from self-R2 0.998 to 0.644

  HELD-OUT    V = 30, coupling 0.20 and V = 30, coupling 0.50
              FRESH seeds 10, 11, 12
              obs noise 0, 0.02, 0.05, 0.10, 0.30
              30 cells. Different width, different coupling, different seeds.

s* is fitted on DISCOVERY only and applied unchanged to HELD-OUT.

## Predictions, fixed now

  S1  Source false positives rise as per-channel self-R2 falls, reproducing
      Rules 71-72 on this system. If this does not reproduce, the experiment
      is void and nothing after it means anything.

  S2  There exists an s* at which pooled discovery source FP is at or below
      0.05. Declared as the operating point to be fitted; the FULL curve of
      source FP against s is reported so any other bar can be read off it.

  S3  POSITIVE CONTROL, required by Rule 83. At s*, TP keep must be at least
      0.80. A gate that reaches zero source false positives by discarding the
      detections is not a gate, and if S3 fails at every s where S2 holds,
      the honest report is that no per-channel saturation gate exists.

  S4  DECISIVE. s*, carried over unchanged, holds on the held-out systems:
      source FP at or below 0.05 AND TP keep at or above 0.80 in both
      configurations. Failure here means the threshold is a property of the
      discovery system and not of the method, and it must be reported as
      not transferable rather than re-fitted.

  S5  NO PREDICTION on where the inherited 0.9 lands relative to s*. It is
      recorded now so that whatever the answer is, it cannot afterwards be
      presented as a finding either way.

  S6  NO PREDICTION on the ghost. G3 is expected to pass in cells where the
      gate is doing work, since that is the established failure (Rule 72);
      it is measured and reported so the two can be compared directly.

## Void conditions

Void if S1 does not reproduce, if s* is moved after any held-out number is
seen, if the noise grid or seeds are changed after a result, if held-out
cells are added or dropped after a result, or if s* is reported without the
TP-keep cost beside it.
