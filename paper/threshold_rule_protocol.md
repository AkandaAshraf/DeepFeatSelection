# Pre-registration: is the ghost-panel maximum the right decision rule?

Declared 2026-08-20, before the experiment was written or run.

## The defect this addresses

The boundary map found that recall tracks the single worst ghost rather than
the sample size: correlation -0.44 between ghost_max and recall across 51
cells, with recall swinging between 0.02 and 0.28 at fixed V, coupling and
redundancy. The declared decision rule is

    threshold = max(0, ghost panel max)

so one unlucky surrogate out of thirty sets the bar for an entire scan.

## Why this is a real question and not tuning

The choice of rule is a MULTIPLICITY DECISION, not an arbitrary knob. Taking
the panel maximum is a max-statistic family-wise error control: it asks that
no channel exceed the largest null value observed, which controls the chance
of ANY false positive across the scan. A panel quantile instead controls a
per-channel error rate. Both are defensible; they answer different questions.

The max rule is therefore principled but conservative, and - as the boundary
map showed - high variance, because the maximum of a finite panel is an
extreme order statistic with no stability guarantee at panel sizes of 30-50.

The question is whether the conservatism is buying anything, given that
measured precision is already 1.00 nearly everywhere and so has headroom.

## Rules compared, fixed now

  MAX        max(0, panel max)                     the deployed rule
  Q99        max(0, 99th percentile of the panel)
  Q95        max(0, 95th percentile of the panel)
  Q90        max(0, 90th percentile of the panel)
  MEAN3SD    max(0, mean + 3 sd of the panel)

All five are computed from the SAME ghost panels and the same per-channel
excess values, so no rule gets a different scan. The comparison is therefore
exact rather than a re-run with a different seed.

## Predictions, fixed before running

  T1  Recall rises as the rule is relaxed from MAX to Q90. Trivially
      expected; recorded so that it cannot later be presented as a finding.
  T2  NO PREDICTION for precision. How much precision each rule costs is the
      quantity of interest and is genuinely unknown.
  T3  THE DEFECT CLAIM. Recall under MAX is more variable - measured as the
      standard deviation of recall across cells that share (V, coupling,
      redundancy) and differ only in n and seed - than under the quantile
      rules. If MAX is NOT more variable, the defect identified by the
      boundary map was misdiagnosed and this is recorded as such.
  T4  Source false positives stay at zero under every rule. If relaxing the
      threshold starts flagging sources, that is disqualifying regardless of
      what it does to recall, because source blindness is the property the
      method is built on.

## The decision rule, fixed now

A rule replaces MAX only if, across the whole grid, it holds median
precision >= 0.95 AND zero source false positives AND lower recall variance
than MAX. If more than one rule qualifies, the most conservative qualifying
rule is chosen - not the one with the highest recall.

If no rule qualifies, MAX stands and the boundary map's finding is reported
as a limitation of the method rather than a defect to be fixed.

## Void conditions

Void if the rule set, the metrics or the decision rule are altered after any
result is seen, or if a rule is adopted on recall alone.

---

## Result (2026-08-20): NO RULE QUALIFIES. MAX stands, and the defect claim
## was wrong.

51 cells, five rules on identical scans and identical ghost panels.

  rule       median precision   median recall   worst-cell source FP
  MAX             1.000             0.220            0.000
  Q99             1.000             0.220            0.000
  Q95             1.000             0.280            0.200
  Q90             1.000             0.360            0.200
  MEAN3SD         0.000             0.000            0.000

### T3 REFUTED OUR OWN CLAIM

Recall variability across n and seed, at the centre (V, coupling,
redundancy):

  MAX      sd 0.105      Q99  sd 0.115      Q95  sd 0.118      Q90  sd 0.148
  MEAN3SD  sd 0.050  but precision 0.000 - it flags nothing, so its stability
                     is the stability of always returning the empty set

MAX is the LEAST variable of the non-degenerate rules, not the most. The
defect identified in the boundary map is therefore MISDIAGNOSED, and this is
recorded as a failed claim of ours rather than quietly dropped.

### What the ghost_max/recall correlation actually means

The correlation of -0.44 between ghost_max and recall is real. Our reading of
it - that one unlucky surrogate out of thirty sets the bar and costs recall -
does not survive this test. If the instability came from the EXTREMENESS of
the maximum, quantile rules would have been more stable. They are less
stable. That means the variability is not in the tail of the panel; the whole
panel shifts together between scans, and the maximum is tracking a real
property of each scan rather than sampling noise.

In other words the threshold is doing its job: when the null level is high,
the scan is genuinely noisier and the bar correctly rises. Low recall in
those cells is appropriate caution, not a defect.

### Why the relaxed rules are disqualified

Q95 and Q90 buy recall (0.22 -> 0.28 -> 0.36) and in the worst cell flag 20%
of genuine SOURCES. Source blindness is the property the method is built on
and the one thing that held perfectly across all 51 cells of the boundary
map; a rule that breaks it is disqualified regardless of its recall, exactly
as declared in T4 before any of this was run.

### Verdict

MAX stands, and is better justified than before. It is not merely a
conservative default: among the rules tested it is the only one that is
simultaneously non-degenerate, source-preserving, and least variable. The
boundary map's low-recall regions are a genuine limit of the method, not an
artefact of its decision rule.

Had the declared decision rule been "adopt whichever gives the most recall",
Q90 would have been adopted and the method's central property broken.
