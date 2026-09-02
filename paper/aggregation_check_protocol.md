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
