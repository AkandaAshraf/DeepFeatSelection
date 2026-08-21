# Pre-registration: can channel duplication manufacture drivenness?

Declared 2026-08-20, before the experiment was written or run.

## The hazard

MACE's excess for channel q is the gain in predicting q's next step when a
learned code of the system is added to a model of q's own history. If the
system contains a NEAR-COPY of q, the code carries that copy, and the copy
predicts q's next step almost perfectly. The excess should rise - even if q
is coupled to nothing at all.

That would be a false positive produced purely by sensor duplication, and it
is not exotic. Duplicated and near-duplicated channels are common in real
recordings: adjacent electrodes on a shaft, overlapping regions of interest,
a signal recorded twice through different processing paths. The published
zebrafish and intracranial scans both contain spatially adjacent channels.

## Why the ghost cannot catch it

The ghost channel is a CIRCULARLY SHIFTED copy of real data. The shift is
what makes it a valid null: it preserves the marginal dynamics and destroys
the simultaneous relationship. A duplicate is SIMULTANEOUS. The ghost
therefore probes a different failure mode entirely, and would stay clean
while duplicates inflated real channels.

If this hazard is real, the method's built-in falsification test is
structurally blind to it, and that must be stated wherever the method is
described.

## Design

Systems as in the boundary map: drivers, driven channels, and ISOLATED
channels coupled to nothing. To this we add, for a subset of the isolated
channels, k near-copies:

    copy = isolated_channel + noise_level * standard normal

  k            copies per duplicated channel: 0, 1, 2, 4
  noise_level  0.01, 0.05, 0.20   (relative to unit-variance channels)

Centre: V = 60, n = 4000, coupling = 0.20. Three seeds per cell.

The quantity of interest is the flag rate of the ORIGINAL isolated channels
that have copies, compared with isolated channels that do not. Both groups
are coupled to nothing, so any difference is caused by duplication alone.

## Predictions, fixed before running

  D1  Duplicated isolated channels are flagged more often than
      non-duplicated isolated channels. This is the hazard, and we expect it.
  D2  The effect weakens as noise_level rises, because a noisier copy
      carries less of the original.
  D3  THE GHOST STAYS CLEAN throughout. This is the point of the
      experiment: if the ghost rose alongside the false positives it would
      catch them, and there would be no hazard to report. We expect it to
      stay clean and therefore to MISS the failure.
  D4  NO PREDICTION for whether genuine sources are affected. Sources have
      held at zero false positives across 51 boundary-map cells; whether
      duplication breaks that is unknown and is the most consequential thing
      this could find.

## What follows from each outcome

If D1 holds: a practitioner warning is required, the method's description
must state that the ghost does not test for this, and a de-duplication
screening step should be recommended before any scan.

If D1 fails: duplication does not manufacture drivenness, which is a
robustness result and is reported as such.

## Void conditions

Void if the design, the noise levels or the predictions are altered after any
result is seen.
