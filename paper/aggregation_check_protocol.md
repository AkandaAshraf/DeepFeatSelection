# Addendum: is CCM's collapse the method's, or our aggregation's?

Declared 2026-09-02, before any alternative aggregation was scored. Uses the
CCM and PCMCI matrices already saved by the V=60 run; no new edge scoring.

## Why

The V=60 result section says, against our own interest:

> The aggregation is *ours*, not CCM's: maximum-over-incoming is the natural
> reduction of an edge matrix to a per-channel score, but a count above a
> per-scan threshold, or a calibrated per-source null, might degrade more
> gracefully. What is established is that pairwise edge recovery does not
> convert to membership at scale *under the obvious aggregation* -- not that
> no aggregation could.

That caveat is currently an assertion. A reviewer will ask whether we tried,
and the honest answer today is no. The expensive part -- 249 minutes of CCM
and 85 of PCMCI -- is already done and saved, so testing alternatives costs
minutes. This converts the caveat into a measurement either way.

## The alternatives, fixed now

All operate on the SAME saved matrices, scoring the same membership question
(driven vs source) by AUROC, at V = 60, three seeds. Four aggregations, of
which the first is the incumbent:

  A. MAX          score(q) = max_p score[p -> q]              (incumbent)
  B. MEAN         score(q) = mean_p score[p -> q]
                  Rationale: averages away a single spurious edge, which is
                  precisely the failure mechanism the V=60 run identified.
  C. COUNT        score(q) = #{p : score[p -> q] > t}, t = the 90th
                  percentile of that scan's own off-diagonal edge scores.
                  Rationale: a driven channel should have a real parent
                  above threshold; a source should have none. Threshold is
                  per-scan and data-derived, so no constant is imported.
  D. TOP2MEAN     score(q) = mean of the two largest incoming edges.
                  Rationale: a compromise -- robust to one spurious edge
                  but not diluted by 58 irrelevant ones as MEAN is.

These four are the complete declared set. No fifth is added after seeing
results, and the incumbent MAX is reported beside them in every table.

## Predictions, fixed now

  G1  At least one of B, C, D beats MAX's median CCM membership AUROC of
      0.674 at V = 60. Rationale: MAX is provably the most sensitive to the
      single-spurious-edge mechanism the V=60 run measured, so it should be
      the weakest of the four, and a better adapter should exist.
  G2  DECISIVE FOR THE PAPER'S CLAIM. No alternative reaches MACE's median
      of 0.976. If one does, the paper's positioning is wrong as written and
      the V=60 section must say that CCM with a better adapter matches MACE
      at this width.
  G3  NO PREDICTION on the ordering of B, C, D among themselves.
  G4  Whatever the best alternative achieves at V = 60, the same aggregation
      is also applied at V = 30 (matrices likewise saved). If an alternative
      wins at V = 60 it must not lose at V = 30, or it is width-specific
      tuning rather than a better adapter, and is reported as such.

## The rule, fixed now

  CAVEAT DISCHARGED   No alternative reaches MACE (G2 holds). The paper's
                      claim stands and the caveat is replaced by the
                      measurement: we tried three alternatives and the
                      conversion still fails.
  CLAIM WEAKENED      An alternative reaches or beats MACE. Reported as the
                      headline of that section; the V=60 conclusion is
                      rewritten to attribute CCM's collapse to our adapter
                      rather than to the method, and Table 12's row is
                      amended.

## Void conditions

Void if an aggregation outside {MAX, MEAN, COUNT, TOP2MEAN} is introduced
after any score is seen, if COUNT's percentile is changed after a result, or
if the V = 30 cross-check (G4) is omitted from the report.

---

## Result (2026-09-02): CAVEAT DISCHARGED. The adapter is not the problem.

Scored on the already-saved matrices; no new edge computation.

  V=60, median over three seeds
  method    MAX     MEAN   COUNT  TOP2MEAN
  CCM      0.674   0.680   0.590   0.658
  PCMCI    0.590   0.444   0.400   0.548

  V=30, single seed (G4 cross-check)
  CCM      1.000   1.000   0.720   1.000
  PCMCI    0.728   0.568   0.584   0.712

  MACE median at V=60: 0.976

**G2 holds: no alternative comes close to MACE.** The best of the three,
MEAN at 0.680, is 0.30 below MACE and only 0.006 above the incumbent MAX --
which is noise, not an improvement.

G1 technically holds (MEAN 0.680 > MAX 0.674) and is reported as
technically holding and substantively empty. A 0.006 difference over three
seeds is not evidence that MEAN is a better adapter, and it would be
dishonest to present it as one. The declared prediction was that MAX should
be the weakest of the four because it is the most sensitive to the
single-spurious-edge mechanism; what the data show is that MAX is roughly
tied with MEAN and better than both COUNT and TOP2MEAN, so the reasoning
behind G1 was wrong even though its letter was satisfied.

G4 is satisfied: MEAN scores 1.000 at V=30, tied with MAX, so nothing here
is width-specific tuning.

### What this settles

The V=60 section's caveat -- that a different aggregation might degrade more
gracefully -- is now a measurement rather than an assertion. Three
alternatives were declared in advance and tried; the edges-to-membership
conversion still fails, and the gap to MACE is 0.30 under every one of them.

The claim in the paper is unchanged in substance but can now be stated
without the hedge: pairwise edge recovery does not convert to membership at
V = 60 under any of four aggregations, including the two most obvious
robust alternatives to the maximum.

### What it does not settle

Four aggregations are not all aggregations. A per-source calibrated null --
scoring each channel against a distribution of what its own incoming edges
look like under a surrogate -- is a fifth family that was NOT declared and
is NOT run here, because adding it after seeing these numbers is precisely
the shopping this protocol forbids. It remains the honest open door, and
anyone with the released matrices can test it in minutes.
