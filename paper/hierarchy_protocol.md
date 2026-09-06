# Pre-registration: nested codes, and at which scale a variable's drive lives

Declared 2026-09-06, before the experiment was written or run.
EXPLORATORY. Nothing can be adopted on this run whatever it shows.

## Why a hierarchy, and why not the flat merge that just failed

The 2026-09-05 unified model put both statistics on one network and was
REJECTED: inflow fell to average precision 0.237 against the incumbent's
0.895. The cause was diagnosed and is structural. One shared decoder
allocating finite capacity across every output reports where it spent that
capacity, not what each target gains. Rule 127: amortise the encoder, never
the readout.

A hierarchy is the construction that composes the three approaches WITHOUT
repeating that error. Each level keeps its own per-target ridge readout. Only
the codes are nested. Nothing is shared that Rule 127 forbids sharing.

It is also what a U-Net actually is. The defining idea is not the skip alone
but multi-resolution with a skip at every level, and that is the part of the
architecture no experiment in this sequence has used.

## The construction

Three levels of conditioning for a target q:

  level 1   phi_q(t), q's own lags. The existing self-baseline.
  level 2   a code over the MODULE containing q, with q's own columns zeroed
            before the module encoder.
  level 3   the global code over all V variables, as the incumbent computes it.

  excess_2(q) = R2[own + module] - R2[own]
  excess_3(q) = R2[own + module + system] - R2[own + module]
  total(q)    = excess_2(q) + excess_3(q) = R2[own + module + system] - R2[own]

Every readout is a per-target ridge fit, exactly as the incumbent does.

MODULE-LEVEL EXCLUSION, and why it is the one asymmetry against FLAT. A
module holds roughly six variables, so q is about a sixth of its own module
code and could re-represent its own history through it. That is the
false-positive channel the 2026-09-05 audit named. At the system level q is
one of thirty and exclusion was measured to change nothing, so the system
level matches the incumbent exactly and FLAT is left untouched. Module-level
exclusion is the only difference, and it is declared here rather than
discovered later.

## The claim under test

The LEVEL at which drive appears should localise it. A variable driven by
something inside its own module should score at level 2; a variable driven
from outside should score at level 3. Ground truth is available: the
generator's parent map is recovered by replaying each seed's draws, so
whether q's parent shares q's module is known.

## Arms

  FLAT        the incumbent. One system code, per-target ridge. Unchanged.
  HIER-CLUST  modules from agglomerative clustering on the correlation matrix
              of the first-differenced series. The deployable version.
  HIER-RAND   modules assigned at RANDOM, same sizes. THE CONTROL. If random
              modules localise as well as clustered ones, the hierarchy
              measures nothing about structure and the claim is empty.
  HIER-TRUE   modules from the ground-truth parent map, each source with its
              own children. AN ORACLE, not deployable, reported as an upper
              bound on what perfect clustering could buy.

The control and the oracle bracket the deployable arm from below and above.
If the oracle does not beat FLAT, clustering quality is irrelevant and the
idea is dead whatever the clustering does.

  SYSTEM      boundary_map.make_system, n = 4000, coupling 0.20, redundancy 0
  WIDTH       V in {30, 60}, bottleneck 2V at system level, 2*|module| at
              module level
  NOISE       {0.0, 0.05}. NOT 0.30, where the 2026-09-05 run put every arm
              at or below chance (Rule 122)
  SEEDS       0, 1, 2
  CLUSTERS    m = V // 6 modules, fixed. This MATCHES the generator's source
              count, which is an advantage handed to the method. Cluster-count
              selection is not solved here and any powered follow-up must
              remove this advantage.

12 cells x 4 arms.

## Metrics and their chance levels, computed before committing

Rule 123. Driven channels are 25 of 30 and 50 of 60, so driven-positive
average precision has a base rate above 0.8 and a lift ceiling below 1.25.
It cannot discriminate and is secondary only.

PRIMARY for detection: average precision for the MINORITY class, sources, at
base rate 0.167 at both widths, scored on the negated inflow since inflow
should rank sources last. Lift ceiling 6.0x.

PRIMARY for localisation: among DRIVEN channels only, the area under the
precision-recall curve for "q's parent is in q's module", scored by
excess_2 - excess_3. Its base rate is the fraction of driven channels whose
parent shares their module, which VARIES BY ARM and is reported per cell
rather than assumed. Lift over that per-cell base rate is the comparable
number.

## Predictions, fixed now

  H1  DISQUALIFYING. HIER-CLUST's total inflow average precision is within
      0.05 of FLAT's. Splitting the conditioning across levels must not cost
      detection. This is the clause that rejected the flat merge and it
      protects the same validated ground.
  H2  DECISIVE. In HIER-CLUST, among driven channels, excess_2 - excess_3
      ranks "parent in my module" above chance, at lift 1.5x or better over
      the per-cell base rate. This is the localisation claim and the reason
      the experiment exists.
  H3  CONTROL, DISQUALIFYING FOR THE INTERPRETATION. HIER-RAND does NOT
      clear the same bar. If random modules localise as well as clustered
      ones, H2 is an artefact of the two-level split rather than evidence
      about structure, and the result is reported as such.
  H4  ORACLE, NO PREDICTION on the size of the gap. HIER-TRUE is reported as
      the upper bound. If HIER-TRUE fails H2, the idea is dead independent
      of clustering.
  H5  GUARD, SCOPED PER ARM (Rule 124). The module level must be informative:
      mean excess_2 on driven channels must exceed zero. An arm whose module
      level is inert is disqualified and its H2 is vacuous; the run is not.
  H6  NO PREDICTION on whether level 3 adds anything once level 2 is
      present. That is the interesting descriptive number and predicting it
      would be storytelling.

## The rule, fixed now

NONE, and no adoption is possible on this run.

  REJECTED  H1 fails. The hierarchy costs detection and is abandoned.
  EMPTY     H2 holds but H3 also holds. The split produces a number that
            does not track structure, and the localisation claim is refused.
  CLOSED    H2 fails with H5 intact, and HIER-TRUE fails it too. Scale
            localisation does not work here and the line closes.
  OPEN      H1 holds, H2 holds, H3 does not. Licenses one powered
            pre-registration at more widths, a second generating family, and
            a cluster count chosen without knowing the answer. Nothing
            enters the manuscript.

## Void conditions

Void if any readout is shared across targets; if module assignment for
HIER-CLUST uses any ground truth; if the grid, seeds, noise levels or
cluster count change after any result is seen; if H2 is judged on any arm
other than HIER-CLUST; or if the random-module control is dropped.

---

## AMENDMENT, 2026-09-06, before the full run and after ONE smoke cell

A single smoke cell (V=30, noise 0, seed 0) was run to price the experiment
and it exposed a metric specification error. The amendment rests on
arithmetic that was available before any data, and it is recorded here with
what was actually observed so the change is checkable rather than trusted.

WHAT WAS OBSERVED IN THAT CELL, in full:

  arm          AP_src   loc_base   loc_lift    e2        e3
  FLAT         1.000       --         --        --        --
  HIER-CLUST   1.000     0.760      1.31     +0.00231  +0.00141
  HIER-RAND    1.000     0.080     12.50     +0.00034  +0.00285
  HIER-TRUE    1.000     1.000       nan     +0.00341  +0.00056

TWO DEFECTS, both arithmetic and both my error under Rule 123.

  1. THE ORACLE'S LOCALISATION IS UNDEFINED BY CONSTRUCTION. HIER-TRUE builds
     modules from the parent map, so every driven channel's parent is in its
     module, the base rate is 1.000 and there are no negatives. H4 as written
     cannot be computed for any dataset.
  2. THE BAR SAT ABOVE THE CEILING. Average precision is capped at 1.0, so
     lift is capped at 1/base_rate. With clustering putting 76% of parents
     in-module the ceiling is 1.32x, and H2 demanded 1.5x. HIER-CLUST could
     not have passed H2 whatever it measured. Base rates also differ across
     arms by an order of magnitude, 0.08 to 1.00, so lift is not comparable
     between arms in the first place.

AMENDED METRIC for localisation, among driven channels only:

  PRIMARY   AUROC of (excess_2 - excess_3) for "parent is in my module".
            Chance is 0.5 whatever the base rate, which is what makes the
            arms comparable when their base rates differ tenfold. Positive
            and negative counts are reported per cell so the imbalance the
            user rightly flagged stays visible.
  BAR       AUROC >= 0.65 for H2, and the same bar for the H3 control.
  SECONDARY average precision with its per-cell base rate, retained because
            AUROC flatters an imbalanced problem and both numbers together
            say more than either.
  H4        BECOMES DESCRIPTIVE. The oracle's localisation is undefined, so
            HIER-TRUE serves only as the upper bound on detection and on
            module-level informativeness.

ADDED SECONDARY, and it was suggested by the smoke cell rather than declared
before it. Labelled accordingly.

  MODULE SHARE = mean excess_2 / (mean excess_2 + mean excess_3) on driven
  channels: what fraction of the total inflow the module level captures.
  It is defined for every arm including the oracle, which is what the
  localisation metric is not.

  PREDICTION, fixed now and before the remaining 11 cells: the ordering is
  HIER-RAND < HIER-CLUST < HIER-TRUE. Better modules should capture more of
  the drive at the module level. The smoke cell gives 0.11, 0.62 and 0.86,
  and one cell is not a result; the prediction is that the ordering survives
  12 cells, two widths and two noise levels.

NOT AMENDED: the arms, the grid, the seeds, the cluster count, H1's
disqualifying non-inferiority clause, and H5's per-arm guard. Detection
saturated at 1.000 in the smoke cell, so H1 may prove uninformative at
V=30 and noise 0; that is a property of the cell, not a reason to change the
clause, and the harder cells are already in the declared grid.

---

## Result (2026-09-06): EMPTY by the declared rule, and the control was mis-designed.

12 cells x 4 arms, 9.8 min.

  DETECTION, average precision, sources positive (chance 0.167)
  V   noise    FLAT   HIER-CLUST   HIER-RAND   HIER-TRUE
  30  0.00    0.895     0.895        0.951       0.976
  30  0.05    0.781     0.819        0.681       0.757
  60  0.00    0.973     0.964        0.970       0.990
  60  0.05    0.387     0.391        0.484       0.527

  H1 HOLDS. HIER-CLUST 0.767 against FLAT 0.759. The hierarchy costs
     nothing in detection, which is what the merge failed to manage.
  H5 HOLDS for every arm. The module level is informative in all three.
  H2 HOLDS. Localisation AUROC 0.905.
  H3 ALSO CLEARS at 0.929, which fires the EMPTY verdict.

**The declared verdict is EMPTY and it stands.** Nothing is adopted.

### The control could not have failed, and that is our design error

H3 assumed that randomising the modules would break the localisation signal.
It cannot. The target is "is my parent in MY module", which is defined
relative to whichever partition the arm chose, so randomising the partition
changes the QUESTION rather than removing the signal. A sound mechanism
answers both versions correctly, and a higher AUROC for the random arm simply
reflects an easier discrimination: with random modules only 8 to 13 percent of
parents are in-module, so the positives are rare and distinctive, against 68
to 70 percent under clustering where the negatives are the rare class.

The control was built to test whether the mechanism depends on modules being
meaningful. It instead tested whether the mechanism works at all, and it does.

### What did discriminate, and it was declared before the remaining 11 cells

MODULE SHARE, the fraction of total inflow captured at the module level:

  HIER-RAND   0.382
  HIER-CLUST  0.923
  HIER-TRUE   1.127

The predicted ordering RAND < CLUST < TRUE holds. Two checks before believing
it (scripts/hierarchy_controls.py):

  SIZE. Mean module size is 6.0 in every arm, so the ordering is not a code
  width artefact. Only the size VARIANCE differs, 4.55 for clustering against
  0.00 for random and 2.03 for the oracle, and that unevenness is a real
  limitation noted below.

  GRADED (Rule 125). Within an arm, a module holding a higher fraction of its
  members' parents captures more of their inflow: rho +0.555 within
  clustering, +0.277 within random modules at p = 0.009, and +0.605 pooled
  across both at p = 5e-17 over 157 modules. The magnitude relationship the
  refused localisation metric could not supply.

The oracle's system-level excess is NEGATIVE at -0.00052. With a perfect
module the global code becomes a liability, its extra parameters costing
held-out accuracy. That is the signature of a sufficient conditioning set and
it is the cleanest single number in the run.

### The metric that matters was never measured

Under clustering, 70 percent of parents sit in the target's module and the
statistic says which at AUROC 0.905. A hit therefore narrows the search from
V candidates to the module's members. That is search-space reduction, this
project's own framing, and it is the quantity a deployment would use. It was
not scored, because the protocol posed localisation as a ranking problem
instead. Both arms answer the ranking question; only clustering delivers a
useful reduction, because the random arm's hits are rare.

That is a pre-registration for another day, not a claim here.

### Not established

Two widths, two noise levels, three seeds, one generating family, redundancy
0. The cluster count was fixed at V/6, which matches the generator's source
count and is an advantage handed to the method. Clustered modules are very
uneven, standard deviation 4.55 against a mean of 6.0, so "search within the
module" is not uniformly a six-candidate reduction. Detection saturates at
V=30 noise 0 and collapses at V=60 noise 0.05, so H1 was informative in only
part of the grid.

---

## CORRECTION, 2026-09-06, raised by Codex (task C01), verified independently

Two defects in the result above are confirmed by re-derivation from the
archived NPZ files, not by argument. Per Rule 116 this is appended rather than
edited into the original text, alongside what depended on it.

**The size control was vacuous.** The claim above, "mean module size is 6.0 in
every arm, so the ordering is not a width artefact," compared UNWEIGHTED mean
module size, which equals $V/m$ for every partition of $V$ items into $m$
groups and therefore reads 6.0 before any data exists. It could not have
discriminated. The quantity that matters is the width each TARGET sits in:

  arm          unweighted mean   target-weighted   driven-weighted
  CLUST             6.000            9.972             9.742
  RAND              6.000            6.000             6.000
  TRUE              6.000            6.706             6.847

Clustered modules are lopsided (`[16, 9, 2, 2, 1]` at V=30 seed 0 against
random's `[6, 6, 6, 6, 6]`), so a clustered target receives a module code about
1.66x wider on average. **The module-share ordering RAND < CLUST < TRUE is
confounded with module width and the "not a width artefact" claim is
WITHDRAWN.** Whether the ordering survives width-matching is now the open
question, addressed in paper/hierarchy_repair_protocol.md.

**The pooled graded-control p-value is invalid.** The reported
$\rho = +0.605$, $p = 5\times10^{-17}$ over 157 pooled modules treated modules
within a cell as independent observations. They are not: modules in a cell
share one generated system, one encoder and one train/validation/test split.
This is Rule 101's error in a new place. The correlation itself may still be
real; the stated significance is not defensible and is withdrawn. The repair
protocol tests within-cell instead.

**Clustering was also transductive**, fit on the full recording rather than
the training segment, which is a third defect in the same run and is repaired
in the same protocol.

**What is NOT withdrawn:** H1 (detection cost), H5 (module-level
informativeness), the oracle's negative system-level excess, and the
qualitative observation that a graded relationship exists — only its stated
significance.
