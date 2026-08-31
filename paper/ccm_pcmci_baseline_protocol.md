# Addendum: a head-to-head baseline on the boundary map's own ground truth

Declared 2026-08-31, before CCM or PCMCI was scored against MACE on this
data. Prompted by an adversarial review of the v2 manuscript: every baseline
comparison in the paper is baselines against each other (Table 12,
`sec:map`), and the one place an affordable head-to-head coincides with the
membership task — the boundary map's own synthetic systems, ground truth
fixed by construction — has none.

## Why this design, and what was cut from the first draft

The natural design is CCM and PCMCI on the boundary map's own centre cell
(n=4000, coupling=0.20, redundancy=0) at V=30 and V=60, three seeds, matching
`paper/boundary_map_protocol.md` exactly so MACE's already-published numbers
apply unchanged.

That design was costed empirically before committing to it, not assumed from
the paper's own introduction (which measured CCM at n=2000, not n=4000).
Measured on this machine: **CCM costs 2.22 s/pair at n=4000** (not the 1.1
s/pair quoted for n=2000 pairs) and **PCMCI costs 220s flat per V=30 cell**
(not per pair -- it is dominated by the PC-stable parent-selection step, not
by the pairwise MCI tests). At V=30 (435 pairs) that is roughly 16 minutes of
CCM plus 4 minutes of PCMCI per seed; at V=60 (1,770 pairs) roughly 65
minutes of CCM alone per seed, PCMCI's cost at V=60 unmeasured and likely
worse than linear in V given its combinatorial parent search. Three seeds at
both widths would be several hours, which the review's own "~35 min"
estimate did not anticipate and which is not run here.

**Cut to fit the budget, declared before any score is seen:**

  WIDTH   V = 30 only. V = 60 is deferred; its cost is now measured
          (≈65 min/seed for CCM alone) and can be run separately if wanted.
  SEEDS   seed 0 only, matching the boundary map's primary seed. This is a
          single-seed comparison and is reported as such — no claim of
          spread or significance is made from one draw.
  SYSTEM  `boundary_map.make_system(n=4000, V=30, coupling=0.20,
          redundancy=0, seed=0)` — byte-identical to the system MACE's
          published Table 4 row (V=30, recall 0.88) was scored on.

## What is measured

CCM and PCMCI both produce a directed pairwise score matrix. MACE's excess
statistic asks a different question per channel — is q driven by the rest of
the system — not a pairwise one, so the two are made commensurable the same
way `scripts/chamber_detect.py` already does it: each channel's membership
score is the MAXIMUM incoming edge score over all other channels,
`score(q) = max_p score[p -> q]`.

  CCM score[p -> q]     rho at max library size for (q xmap p), i.e.
                        evidence p causes q, via `deepfeatselect.ccm.ccm`
  PCMCI score[p -> q]   max |partial correlation| over lags 1..3 for p -> q,
                        via tigramite's ParCorr, `tau_max = 3` (matching
                        `chamber_detect.py`'s existing convention)

Per-channel membership scores are then ranked with AUROC separating driven
channels (true) from source channels (false) — a threshold-free summary,
chosen specifically because CCM and PCMCI have no analogue of MACE's ghost
panel to set one, and picking a threshold after seeing scores would be
exactly the move this project refuses.

## The comparison

MACE's own AUROC on the identical system is NOT re-run; MACE's published
precision/recall on this cell (`ExpOutput/boundary_map/boundary_map.csv`,
V=30 row: precision 1.00, recall 0.88) already exists at 3-seed resolution,
and re-deriving an AUROC from the SAME raw per-channel excess scores that
produced that row (`boundary_map.py` saves `excess` per cell) keeps the
comparison apples-to-apples without spending more compute.

## Predictions, fixed now

  H1  MACE's AUROC exceeds CCM's and PCMCI's on this system. This is the
      expected direction: MACE is a conditional statistic on a compressed
      code of the WHOLE system, while pairwise CCM and pairwise-conditioned
      PCMCI each see one other channel at a time, and the true generative
      structure here is "one channel driven by one specific parent," which
      does not obviously favour the multivariate statistic.
  H2  NO PREDICTION on the ORDER of CCM versus PCMCI relative to each other.
  H3  DISQUALIFYING: if CCM or PCMCI fails to distinguish driven from source
      on the DIRECT parent-child pairs it was built to detect (AUROC < 0.7
      restricted to true edges only, dropping indirect ones), that baseline
      is reported as not functioning on this system rather than as losing to
      MACE, and the comparison is not treated as informative for that
      method.

## Void conditions

Void if the width, seed, or aggregation rule (max over incoming edges) is
changed after any score is seen, or if MACE's number is taken from anywhere
other than the pre-existing, already-published boundary map cell.

---

## Result (2026-08-31): H1 FAILS. CCM ties MACE at this width.

V = 30, n = 4000, coupling 0.20, redundancy 0, seed 0. CCM 18.8 min
(435 pairs, 2.6 s/pair realised), PCMCI 3.7 min.

  method   membership AUROC      H3 true-edge AUROC
  CCM            1.000                 1.000
  PCMCI          0.728                 0.984
  MACE           1.000                 n/a (scores membership directly)

**H1 FAILED, as declared.** MACE does not exceed both baselines: CCM ties it
at 1.000. H2 had no prediction and CCM beat PCMCI on membership. H3 is
passed by both baselines once computed correctly (below), so neither is
disqualified and the comparison is informative for both.

### Two defects in our own scoring, found and fixed before reporting

PCMCI's first membership score was 0.008 -- near-perfect INVERSION, which is
the signature of a transposed matrix, not of a method failing. Both defects
were in `scripts/ccm_pcmci_baseline.py`, not in the baselines:

1. **Transpose bug.** tigramite's `val_matrix[i, j, tau]` is the dependence
   of j at lag 0 on i at lag -tau, so it is ALREADY evidence for i -> j.
   `scripts/chamber_detect.py` returns `val.max(axis=2)` with no transpose
   and is correct; this script added a `.T` and inverted every PCMCI edge.
   Corrected, PCMCI's membership AUROC is 0.728, not 0.008. CCM was
   unaffected: its direction convention was checked against
   `deepfeatselect.ccm`'s documented semantics before the run.

2. **H3 was diluted to uselessness.** As declared, H3 compared ALL
   source-to-driven pairs (5 x 25 = 125) against non-edges. But only 25 of
   those 125 are true edges -- each driven channel has exactly one parent --
   so the positive set was 80% non-edges and H3 was pinned near 0.5 by
   construction. It read 0.578 and 0.479 and could not have detected
   anything. The protocol called this "a weaker, conservative version of
   H3"; it was weaker to the point of being uninformative. Replaced with the
   true-edge version, recovering `parent[]` by replaying `make_system`'s rng
   draws in order. Both baselines then pass H3 comfortably.

Rescoring is in `scripts/ccm_pcmci_rescore.py` and needed no re-run: both
matrices were saved.

### What this means, stated against our own interest

The honest reading is the one the paper's own method-selection map already
gives: **at V = 30, where pairwise CCM is still affordable, CCM is not worse
than MACE at the membership task.** Both separate driven from source
channels perfectly on this cell. MACE's claim was never accuracy at widths a
pairwise method can reach; it is that those widths stop at some point and
the question does not.

The run also re-measures that scaling claim independently. 435 pairs took
18.8 minutes at n = 4000. Extrapolated to V = 10^4 (49,995,000 pairs) that
is roughly 30,800 compute-hours -- consistent with the paper's 15,800-hour
figure, which was measured at n = 2000, at twice the sample length.

### What this does NOT establish

One cell, one seed, one coupling, at the easiest width in the grid, where
MACE also scores a perfect 1.000 and so has no headroom to be beaten. A
tie at ceiling is not evidence of equivalence in general: the informative
comparison is at V = 60-240 where MACE's own recall falls to 0.14-0.23, and
that comparison is unaffordable for CCM (65+ min/seed at V = 60) and was not
run. Nothing here says CCM would keep pace where MACE degrades, or that it
would not.
