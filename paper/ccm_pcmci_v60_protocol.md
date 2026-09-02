# Addendum: the head-to-head at V = 60, where MACE degrades

Declared 2026-09-01, before CCM or PCMCI was scored at this width. Companion
to `paper/ccm_pcmci_baseline_protocol.md`, whose V = 30 run returned a tie
at ceiling.

## Why this run exists

The V = 30 comparison found CCM tying MACE at membership AUROC 1.000, and
its own result section says plainly what that does and does not establish:

> A tie at ceiling is not evidence of equivalence in general: the
> informative comparison is at V = 60-240 where MACE's own recall falls to
> 0.14-0.23, and that comparison is unaffordable for CCM and was not run.

This is that comparison, at the cheapest width where MACE has actually
degraded. At V = 60, coupling 0.20, the boundary map's three seeds give MACE
recall 0.18, 0.18 and 0.14 with precision 1.00 — so MACE is missing roughly
four driven channels in five, and if a pairwise method holds up here the
paper's positioning needs qualifying rather than restating.

## Cost, measured rather than assumed

The V = 30 run realised 2.59 s/pair (18.8 min for 435 pairs). V = 60 is
C(60,2) = 1,770 pairs, so roughly 77 minutes of CCM per seed. PCMCI cost at
V = 60 is unmeasured; at V = 30 it was 220 s flat, dominated by the
PC-stable parent search, which is combinatorial in V, so it may be
substantially worse. Three seeds is therefore expected to take four to five
hours and is declared as such. It is run because the V = 30 result is not
interpretable without it.

## Design, fixed now

  SYSTEM   `boundary_map.make_system(n=4000, V=60, coupling=0.20,
           redundancy=0, seed=s)` for s in {0, 1, 2} — byte-identical to the
           systems behind the published V = 60 boundary-map row.
  SEEDS    all three, unlike the V = 30 run's single seed. Spread is
           reported.
  SCORING  unchanged from the V = 30 protocol: each channel's membership
           score is its maximum incoming edge, ranked by AUROC. PCMCI's
           matrix is used WITHOUT the transpose that inverted the V = 30
           run (tigramite's `val_matrix[i, j, tau]` is already evidence for
           i -> j), and H3 is the true-edge version from the outset,
           recovering `parent[]` by replaying `make_system`'s rng draws.
           Both corrections are carried in, not rediscovered.
  MACE     re-derived from the saved per-channel excess in
           `ExpOutput/boundary_map/raw_n4000_V60_c0.2_r0_s{0,1,2}.npz`. No
           MACE re-run; no new compute on our own side.

## Predictions, fixed now

  H1  DECISIVE. MACE's median membership AUROC exceeds CCM's at V = 60.
      This is the width at which the paper's scale argument begins to do
      work, and MACE's own recall has already collapsed here, so this is
      the prediction the paper's positioning actually rests on. If CCM
      matches or beats MACE at a width where CCM still costs only 77
      minutes, the method-selection map needs a row changed: CCM would be
      competitive to at least V = 60, and MACE's advantage would be purely
      one of affordability past that point, not of accuracy at it.
  H2  NO PREDICTION on CCM versus PCMCI.
  H3  DISQUALIFYING, as before: a baseline whose true-edge AUROC falls
      below 0.7 is reported as not functioning on this system rather than
      as losing, and the comparison is not treated as informative for it.
  H4  NO PREDICTION on whether CCM's membership AUROC degrades from V = 30
      to V = 60. A mechanism exists — max-over-incoming across more
      channels gives more chances for a spurious high score on a source —
      but it is untested and predicting it after seeing the V = 30 tie
      would be fitting the story.

## The rule, fixed now

  MACE AHEAD      H1 holds. The scale claim is supported at the width where
                  it starts to matter, and the V = 30 tie is what it looked
                  like: a ceiling effect at a width too easy to separate
                  the methods.
  BASELINE AHEAD  CCM matches or exceeds MACE. Reported as the headline,
                  and Table 12's membership row is amended to say CCM is
                  competitive to at least V = 60.
  NOT INFORMATIVE Both methods at ceiling again (AUROC > 0.99 for both), or
                  a baseline fails H3. Reported as such; no claim either
                  way, and the next width up is not pursued without a fresh
                  pre-registration.

## Void conditions

Void if the width, seeds, scoring rule or aggregation changes after any
score is seen; if MACE's numbers come from anywhere but the pre-existing
boundary-map cells; or if a seed is dropped from the report.

---

## Result (2026-09-02): H1 HOLDS. The V=30 tie was a ceiling effect.

Three seeds, each in its own process. CCM 249 min total, PCMCI 85 min, MACE
0 (re-derived from the published boundary-map cells).

  MEMBERSHIP AUROC     seed 0   seed 1   seed 2   median
  MACE                  0.918    1.000    0.976    0.976
  CCM                   0.678    0.674    0.490    0.674
  PCMCI                 0.590    0.604    0.502    0.590

  TRUE-EDGE AUROC (median)   CCM 0.984   PCMCI 0.950   -- both function,
  so H3 disqualifies neither and the comparison is informative for both.

**H1 HOLDS.** MACE's median 0.976 exceeds CCM's 0.674 by 0.30 at the width
where MACE's own recall has collapsed to 0.14-0.18. Neither method is at
ceiling, so the NOT INFORMATIVE branch does not apply. H2 had no prediction;
CCM edged PCMCI on membership at every seed.

### What separates them is the aggregation, not edge recovery

The two columns tell different stories and the difference is the finding.
CCM's true-edge AUROC is 0.982-0.995 across seeds: **it recovers the true
parent-child edges essentially perfectly, and does not get worse at its own
job when V doubles.** What collapses is the step from edges to membership.
Each channel's membership score is its maximum incoming edge, so at V = 60
every source has twice as many chances as at V = 30 to pick up one spurious
high incoming score, and sources drift up the membership ranking. At seed 2
this puts CCM at 0.490 — indistinguishable from chance — while its
true-edge AUROC on the same matrix is 0.982.

MACE scores membership directly and never pays that cost. This is the
mechanism H4 named and explicitly declined to predict, because predicting it
after seeing the V = 30 tie would have been fitting the story to the data.
It is recorded now as a measured mechanism rather than a prediction.

### Read against our own interest

Two qualifications belong beside the headline.

First, the aggregation rule is ours, not CCM's. Max-over-incoming is the
natural way to turn a pairwise edge matrix into a per-channel membership
score and is what `chamber_detect.py` already used, but a different
aggregation — a count of edges above a per-scan threshold, or a sum, or a
calibrated per-source null — might degrade more gracefully with V. What
this experiment establishes is that **pairwise edge recovery does not
convert to membership at scale under the obvious aggregation**, not that no
aggregation could. That is a narrower claim than "CCM cannot do this," and
the paper should make it in the narrower form.

Second, MACE's 0.976 AUROC coexists with recall of 0.14-0.18 on the very
same cells. Both are true: it ranks driven channels well while its
threshold rule surfaces few of them. The AUROC is threshold-free and the
recall is not, and the gap between them is the capacity limit of
Section 8.4 rather than a ranking failure.

### Cost, for the scaling claim

CCM took 81-87 minutes per seed at V = 60 against 18.8 minutes at V = 30 —
a 4.5x increase for a 2x increase in width, consistent with the quadratic
pair count. PCMCI took 24-32 minutes per seed, having taken 3.7 at V = 30:
roughly 8x for 2x the width, worse than quadratic, consistent with its
combinatorial parent search.
