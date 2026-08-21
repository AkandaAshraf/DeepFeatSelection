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

---

## Result (2026-08-20): D1 FAILED. No hazard - but only because the
## self-baseline saturates.

Flag rate for duplicated isolated channels: 0.00 in every cell, at every k
(1, 2, 4 copies) and every noise level (0.01, 0.05, 0.20). Plain isolated
channels: also 0.00. Sources: 0.00. Ghost flat throughout (max +0.0017 to
+0.0021, unchanged by duplication).

D1 predicted the hazard and it did not appear.

### Why not, and why that limits the test

Excess is the GAIN over the self-baseline. On coupled logistic maps an
isolated channel's own history predicts its next step almost perfectly - the
map is deterministic and a degree-3 polynomial of three lags represents it
exactly. The self-baseline is therefore at ceiling and there is NO HEADROOM
for the code to add anything, whatever the code contains. A perfect copy in
the code cannot inflate an excess that has nowhere to go.

So the mechanism that blocks the hazard is SATURATION - the very condition
Proposition 1 requires. Where the self-baseline saturates, duplication is
harmless.

That is a reassuring result and a narrow one. It says nothing about the
regime where the self-baseline does NOT saturate, and that regime is most of
real data: 0 of 1,276 worm calcium channels reached the saturation threshold
in the companion paper. The experiment as designed tested only the safe case.

### Declared extension, before running it

An OBSERVATION NOISE axis is added so that the self-baseline is moved off
ceiling deliberately:

    observed = channel + obs_noise * standard normal
    obs_noise: 0.0, 0.1, 0.3, 0.6

with k = 2 copies at copy-noise 0.05 held fixed. Self-baseline R2 is recorded
per cell so that the flag rate can be read against actual saturation rather
than assumed.

  D5  The hazard appears once the self-baseline is off ceiling: duplicated
      isolated channels are flagged more often than plain ones at high
      obs_noise, and the effect grows as saturation falls.
  D6  The ghost stays clean regardless, so it still fails to catch it.

If D5 also fails, duplication does not manufacture drivenness in either
regime and the method is robust to it, which is a stronger result than the
one this experiment set out to find.

## Extension result: source blindness fails when saturation fails, and the
## G3 gate does not catch it

k = 2 copies, copy-noise 0.05, observation noise added to move the
self-baseline off ceiling. Three seeds each.

  obs   self-R2  satur   dup-iso  plain  SOURCE  driven  ghost_max  ghost_med  G3
  0.0    0.998    0.89     0.00    0.00   0.00    0.24     0.0015    -0.0000  PASS
  0.1    0.644    0.03     0.20    0.03   0.20    0.06     0.0472    -0.0008  PASS
  0.3    0.132    0.00     0.07    0.10   0.23    0.04     0.1221    -0.0110  PASS
  0.6    0.012    0.00     0.13    0.03   0.03    0.04     0.0732    -0.0018  PASS

### D5: partially holds, and is the smaller finding

At obs 0.1 duplicated isolated channels are flagged at 0.20 against 0.03 for
plain ones. At higher noise the difference disappears into general
degradation. Duplication is a hazard, but a modest one and only in a band.

### The larger finding: SOURCE BLINDNESS IS CONDITIONAL ON SATURATION

Source false positives run 0.00 -> 0.20 -> 0.23 -> 0.03 as the self-baseline
falls from 0.998 to 0.012. Individual cells reach 0.40 and 0.50.

Source blindness held at EXACTLY 0.000 across all 51 cells of the boundary
map, where every channel saturated, and was recorded there as the method's
most robust property. It is not a property of the method. It is a property of
the SATURATED REGIME. Off ceiling, up to half the genuine sources are flagged
as driven.

This is the practical content of Proposition 1's premise, and it is sharper
than the paper states. The paper says the ghost's guarantee is licensed only
where the self-baseline saturates. This experiment says something stronger:
where it does not saturate, the central claim - that sources are invisible by
design - fails too.

### D6: the ghost rises but the GATE does not fire

ghost_max rises 30- to 80-fold, from 0.0015 to 0.047-0.122. So the panel does
respond. But the DECLARED GATE is on the ghost MEDIAN (<= 0.005), and the
median stays clean: -0.0008, -0.0110, -0.0018. Every one of the nine noisy
cells PASSES G3, including the cell where 50% of sources are flagged.

The gate is watching the wrong statistic for this failure. The panel's
median is unmoved because most surrogates are unaffected; the damage shows
in its spread and its maximum.

### Consequence for the published deployments

The worm calcium-imaging scan reports 0 of 1,276 channels reaching the
saturation threshold. That is the regime where source blindness fails here.
The companion paper already declines per-neuron claims on that data and
treats the ghost as an empirical meter rather than a theorem; this result
supports that caution and gives it a mechanism.

No change to the gate is made on the strength of this. A gate that watched
the ghost spread rather than its median would need its own pre-registration,
and choosing a statistic after seeing which one would have caught a failure
is the move this project refuses.
