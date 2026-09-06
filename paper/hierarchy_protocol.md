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
