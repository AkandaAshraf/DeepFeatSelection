# Pre-registration: confirmatory replication of P-S1

Declared 2026-08-20, BEFORE any recording, channel table or label from the
held-out subjects was downloaded. Directory listings were read to identify
subjects and task names; no channel table was opened.

## Why this is confirmatory where the discovery study was not

The discovery cohort (29 subjects of ds003876, task-interictal) produced
P-S1 as a pre-registered result, but its two same-scale sensitivity arms
were declared AFTER the labels were open and are therefore post-hoc. This
study fixes everything in advance against subjects that have never been
scanned, so it can confirm rather than merely corroborate.

## Cohort, fixed now

The 10 ds003876 subjects absent from the discovery cohort, which were missed
there only because their files use a different task name:

  jh103, jh105, pt1, pt2, pt3, umf001, umf002, umf003, umf004, umf005

PRIMARY DATA: task-interictalawake, run-01, the direct match to the
discovery cohort's task-interictal. No subject may be added or removed after
this document is committed. If a subject's file is unavailable or unreadable
it is reported as such and excluded.

## Pipeline, unchanged from the discovery study

All constants as declared in the discovery pre-registration: decimate to
~256 Hz, middle 120 s segment, E=3, tau=1, first difference, 0.6/0.2/0.2
contiguous splits with embargo, degree-3 self-baseline, ridge alpha 1,
masked-AE M=4, bottleneck 32, mask 0.25, Adam 3e-3, batch 64, 25 epochs,
50-donor ghost panel filtered to self-R2 > 0.9 where 8 donors qualify.

Gate G1 (length), G2 (saturation, recorded) and G3 (ghost panel median
<= 0.005) as before. A subject failing G3 on the primary montage is excluded
and reported.

## Montages, fixed now

  PRIMARY:    bipolar
  SAME-SCALE: laplacian, bipolar_skip
  EXCLUDED:   raw and CAR

Raw is excluded on the criterion established in the discovery study, not on
its result: it correlates with bipolar at median Spearman 0.012 and so
measures a different quantity. The validity gate (median rho with bipolar
> 0.30) is re-applied here and reported; an arm failing it is uninformative,
not evidence.

## Label handling

Labels opened only after every gate verdict is recorded. Permissive
inheritance as declared: a derivation counts as SOZ if ANY constituent
contact is marked. For laplacian that is the centre contact and its two
neighbours. This works AGAINST the depletion being tested.

## The test, and the quantitative prediction

Per subject, one-sided Mann-Whitney on per-channel excess, SOZ ranked below
non-SOZ; pooled by Stouffer's z weighted by sqrt(channels); sign test
reported alongside; subjects with fewer than 3 channels in either group
excluded.

PREDICTED, from the discovery cohort:

  P-R1  bipolar pooled z > 0 with p < 0.05 one-sided. Given the discovery
        effect and 10 rather than 26 subjects, the expected z is about
        5.1 * sqrt(10/26) = 3.2, so this is adequately powered on the pooled
        test IF the effect is real at the discovered size.
  P-R2  median rank-biserial positive and in the range 0.10 to 0.30
        (discovery: 0.172 bipolar, 0.198 laplacian, 0.216 bipolar_skip).
  P-R3  driven-core membership lower for SOZ than non-SOZ, by roughly 5 to
        18 points (discovery: 9.5, 6.7 and 13.6 points across the three
        arms).
  P-R4  laplacian and bipolar_skip both positive in z, whatever their
        individual p-values.
  P-R5  P-S2 (spread-territory concentration) is predicted NEGATIVE, having
        been negative on all three same-scale arms in discovery. A positive
        P-S2 here would contradict the discovery study.

DECLARED UNDERPOWERED: the sign test. With 10 subjects it needs 9 of 10 to
reach p < 0.05 one-sided, so it is reported and not interpreted as a
failure if non-significant.

## What counts as what, fixed now

  REPLICATED           P-R1 holds AND P-R4 holds.
  PARTIALLY REPLICATED P-R1 holds but P-R4 does not, or the effect size
                       falls outside the P-R2/P-R3 ranges. Reported as
                       partial, with the discrepancy named.
  FAILED TO REPLICATE  P-R1 does not hold. This is reported in the abstract
                       of any write-up, and the discovery result is
                       downgraded accordingly in the ledger. No further
                       cohort will be sought to rescue it.

## Secondary, declared exploratory

task-interictalasleep exists for all 10 subjects. After the primary analysis
is complete and recorded, the same pipeline is run on the asleep recordings
as a WITHIN-SUBJECT state control: does the depletion persist across
vigilance state in the same electrodes? Exploratory because the discovery
cohort's vigilance state is not documented, so no directional prediction is
made. It cannot rescue a failed primary.

## Void conditions

Void if the labels are read before the gate verdicts are recorded, if the
cohort is changed after this commit, or if any prediction above is altered
after any label is seen.
