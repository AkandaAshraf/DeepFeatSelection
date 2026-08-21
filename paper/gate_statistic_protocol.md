# Pre-registration: which ghost statistic should the gate watch?

Declared 2026-08-20, before the experiment was written or run.

## The defect, with direct evidence

The duplicate-channel experiment found that source blindness fails off
saturation: source false positives rise from 0.000 to 0.20-0.23, with cells
at 0.40 and 0.50. In those same cells the ghost panel's MAXIMUM rose 30- to
80-fold (0.0015 to 0.047-0.122) while its MEDIAN stayed clean (-0.0008 to
-0.0110).

The declared gate G3 tests the median. All nine failing cells passed it,
including one flagging half the genuine sources. The panel responded to the
failure; the gate was reading the wrong number.

Unlike the threshold-rule claim tested earlier today - which the evidence
refuted - this defect is directly observed, not inferred.

## The trap this design must avoid

The tempting move is to pick whichever statistic separates the nine observed
cells. That is fitting a gate to the failure that motivated it. Two
safeguards, fixed here:

  FRESH DATA   the comparison runs on new seeds (10-14) not used in the
               experiment that motivated it, and over a wider observation
               noise range.
  MATCHED      each candidate is calibrated to the SAME specificity on
  SPECIFICITY  known-good scans, then compared on sensitivity. A statistic
               cannot win by simply rejecting more.

## Definitions, fixed now

A scan is BAD if its source false-positive rate exceeds 0.05 - that is, if
more than one genuine source in twenty is flagged as driven. Source
blindness is the property the method rests on, so this is the failure that
matters.

A scan is GOOD otherwise.

Candidate gate statistics on the ghost panel:

  MEDIAN   the deployed G3 statistic
  MAX      the largest surrogate score
  P95      the 95th percentile
  IQR      the interquartile range (a pure spread measure)
  STD      the standard deviation

## Procedure, fixed now

1. Generate scans across observation noise 0.0 to 0.5 and seeds 10-14, so
   that both good and bad scans occur.
2. For each candidate statistic, set its threshold to the value that rejects
   exactly 10% of GOOD scans. This equalises specificity at 90% by
   construction.
3. Report each statistic's SENSITIVITY: the fraction of BAD scans rejected
   at that threshold.

## Predictions, fixed before running

  G1  MEDIAN has the lowest sensitivity. It is the incumbent and the
      evidence says it misses the failure; if it does not come last, the
      motivating observation was a coincidence and this is recorded as such.
  G2  A spread statistic (IQR or STD) outperforms MAX. Reasoning: MAX is a
      single order statistic and was shown this morning to be no less stable
      than quantiles, whereas the failure signature is the panel widening.
      This prediction may well be wrong and is recorded so that it can be.
  G3  NO PREDICTION for whether ANY statistic reaches usable sensitivity. If
      the best candidate rejects only a minority of bad scans at 90%
      specificity, then no ghost statistic protects against this failure and
      the honest conclusion is that saturation must be checked directly
      rather than inferred from the panel.

## The decision rule, fixed now

A statistic replaces MEDIAN only if its sensitivity exceeds MEDIAN's by at
least 0.20 at matched specificity. Otherwise the gate is unchanged and the
finding is reported as a limitation: the ghost panel does not protect
against loss of source blindness, and saturation must be reported alongside
every scan.

No statistic is adopted on this experiment alone. A change to the deployed
gate would require confirmation on a held-out generating process, declared
separately.

## Void conditions

Void if the candidate set, the bad-scan definition, the matched-specificity
procedure or the decision rule are altered after any result is seen.
