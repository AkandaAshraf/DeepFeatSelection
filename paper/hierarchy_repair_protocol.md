# Pre-registration: repairing the hierarchy evaluation

Declared 2026-09-06, before the script was written or run.
BOUNDED DIAGNOSTIC. This is a validity repair of an archived result on the
ORIGINAL grid, not a reopening of the rejected architecture and not a new
confirmatory study. No verdict word (adopt/reject/close) is licensed by this
run; see "Language" below for what is licensed instead.

## The three defects being repaired

All three are errors in the 2026-09-06 hierarchy run, confirmed by
re-derivation from its own artifacts and recorded in
`paper/hierarchy_protocol.md`'s correction and `paper/causal_detection_log.md`
Rule 131.

  1. TRANSDUCTIVE CLUSTERING. `modules_for` received the full recording and
     correlated over every row, so module assignment was fit on train,
     validation and test together. The encoder and every ridge readout are
     train-only, so clustering is the sole leak, but the archived result is
     transductive and is not evidence for a train-only deployment.

  2. THE SIZE CONTROL WAS VACUOUS. It compared unweighted mean module size,
     which is $V/m$ by construction and therefore reads 6.0 for every arm
     before any data exists. The quantity that matters is the width each
     TARGET receives. Measured across the archived 12 cells:

       arm          unweighted mean   target-weighted   driven-weighted
       CLUST             6.000            9.972             9.742
       RAND              6.000            6.000             6.000
       TRUE              6.000            6.706             6.847

     Clustered modules are lopsided, `[16, 9, 2, 2, 1]` at V=30 seed 0 against
     random's `[6, 6, 6, 6, 6]`, so a clustered target sits in a module about
     1.66x wider and receives a wider code. This establishes a CONFOUND. It
     does not establish that width explains the effect, and this protocol
     cannot establish that either, for reasons stated under "Language" below.

  3. STATISTICS. The graded relationship was reported as a Spearman over 157
     pooled modules at $p = 5\times10^{-17}$. Modules within a cell share one
     system, one encoder and one split, so that pools correlated units as
     independent. Twelve cells built from three seeds are not twelve
     independent draws either: cells sharing a seed share the same generated
     system. Nothing in this repair claims a significance level from three
     seeds. Within-cell correlations are reported descriptively, by seed, and
     no p-value is attached to the aggregate.

## Scope, fixed now

This run repeats the ORIGINAL grid on the ORIGINAL seeds: $V \in \{30, 60\}$,
noise $\in \{0.0, 0.05\}$, seeds $\{0, 1, 2\}$, 12 cells, exactly as archived.
No fresh seeds are introduced here. A powered confirmatory study on
independent systems, if the repair changes the picture enough to warrant one,
is separate future work with its own pre-registration and is not scoped by
this document.

## Arms

Five arms. Everything except module assignment is held fixed: generator,
ridge, widths, epochs, bottlenecks, metrics.

  FLAT              incumbent, one system code, per-target ridge. Unchanged
                     from the archived implementation.
  HIER-CLUST-TRAIN  clustering fit on TRAINING ROWS ONLY, repair 1.
  HIER-RAND-BAL     balanced random modules, equal sizes. The archived
                     control, retained as the original reference.
  HIER-RAND-SIZED   random modules whose size VECTOR matches
                     HIER-CLUST-TRAIN's in that cell, targets assigned to
                     sizes at random. Built from the SAME cell's train-only
                     clustering, not the archive, because repairs 1 and 2
                     interact: a different partition yields a different size
                     vector, and matching the archived one would match the
                     wrong widths.
  HIER-TRUE         oracle from the parent map, upper bound on module
                     informativeness, unchanged from the archived design.

**What HIER-RAND-SIZED does and does not control for.** It matches the
DISTRIBUTION of module widths a random assignment would produce against
clustering's. It does NOT match the width any SPECIFIC target receives: a
target that lands in a size-9 module under clustering may land in a size-2
module under HIER-RAND-SIZED, because sizes are drawn to targets at random
rather than by identity. The comparison COMPARES CLUSTERING AFTER MATCHING
THE AGGREGATE SIZE DISTRIBUTION -- nothing stronger. It supplies no
quantitative bound on how much of any effect is attributable to width, and
does not isolate a "pure width effect" free of which target got which
module, since module membership and width co-vary by construction in both
arms. This limitation is stated here rather than discovered after the run.

## Design

  SYSTEM   boundary_map.make_system, n = 4000, coupling 0.20, redundancy 0
  WIDTH    V in {30, 60}; system bottleneck 2V
  MODULE BOTTLENECK  $\max(4,\, 2|\text{module}|)$ for every module including
           singletons, exactly the archived formula. A singleton module gets
           bottleneck 4, not 2. The ACTUAL bottleneck width used for every
           module, in every arm and cell, is reported in a table -- not just
           asserted from the formula -- so a singleton floor taking effect is
           visible rather than inferred.
  RAW/TRAIN CUTOFF  the exclusive raw-row cutoff used to fit
           HIER-CLUST-TRAIN's clustering is an explicit integer, computed once
           per cell as `int(0.6 * m)` on the embedded manifold length `m`
           (matching the encoder's own train slice), and is SAVED alongside
           the module labels for that cell so it can be checked rather than
           re-derived from prose.
  NOISE    {0.0, 0.05}
  CLUSTERS m = V // 6, fixed, unchanged from the archived run
  SEEDS    0, 1, 2 — the archived seeds, and only these
  OUTPUT   ExpOutput/hierarchy_repair, a NEW directory; the archived
           artifacts in ExpOutput/hierarchy are preserved untouched

12 cells x 5 arms.

## Guards and artifacts declared before running

  FLAT FIDELITY GUARD, ON RAW SCORES, NOT ONLY THE SUMMARY. FLAT's detection
  average precision must match the archived FLAT's in the same cell within
  tolerance 0.03, AND the raw per-channel excess array must match the
  archived one within numerical tolerance ($10^{-6}$, allowing for
  nondeterministic GPU reduction order). Average precision alone can hide a
  changed score: two different rankings can produce the same summary number,
  so the array comparison is not a redundant check, and if only the AP
  matched the pipeline could have silently changed by something other than
  intended repairs. Either mismatch voids the comparison until found.

  TRAIN-ONLY INVARIANCE TEST, run once before the main script and its result
  reported alongside it: perturb every value in the validation and test rows
  with independent noise, re-run HIER-CLUST-TRAIN's clustering step on the
  perturbed array, and compare the resulting partition to the unperturbed
  one by ADJUSTED RAND INDEX, not by counting unequal numeric labels --
  cluster label numbers are arbitrary (permuting them describes the identical
  partition), so a naive label-equality count can report spurious
  "differences" that are really the same partition relabelled. The test
  passes only at ARI $= 1.0$ (partitions identical up to relabelling). If it
  is not, training-row-only fitting is not actually achieved and the repair
  has not repaired defect 1.

  SAVED PARTITIONS. The module-label array (`lab`) for every arm, in every
  cell, is written to the output `.npz` alongside the raw excess arrays. This
  is Rule 114's standing requirement, made explicit here because C01 exists
  because a summary statistic hid a defect that raw arrays would have shown.

  TARGET-WEIGHTED SIZE, reported as a first-class table for every arm in
  every cell, not as a post hoc control: unweighted mean, target-weighted
  mean $\sum(\text{size}_i^2)/V$, and driven-weighted mean (the same quantity
  restricted to driven channels).

## Metrics, their chance levels, and undefined cases predefined now

Rule 123. Driven channels are 25 of 30 and 50 of 60, so driven-positive
average precision has a base rate above 0.8 and a ceiling below 1.25x. It
cannot discriminate and is not used.

  DETECTION    average precision for the MINORITY class, sources, base rate
               0.167 at both widths, scored on the negated total inflow.
  MODULE SHARE mean excess_2 / (mean excess_2 + mean excess_3) on driven
               channels. **This is not a bounded fraction.** It is a ratio
               of two possibly-signed quantities and can exceed 1 (the
               archived HIER-TRUE run reported exactly this, 1.127) or be
               undefined when the denominator is near zero. Handling,
               declared now: if $|{\rm mean}(e_2)+{\rm mean}(e_3)| < 10^{-5}$
               the share is reported as undefined and the cell is EXCLUDED
               from share tables with the exclusion counted and reported,
               never dropped silently. Raw $e_2$ and $e_3$ means are reported
               alongside share in every table, so a reader can see the
               components a ratio can hide. No exclusion is chosen after
               seeing which cells it would remove.
  GRADED       within-cell Spearman between a module's fraction of in-module
               parents and its OWN share (that module's mean $e_2$ over that
               module's mean $e_2+e_3$, restricted to its driven members),
               reported per cell and summarised by seed (3 numbers: one
               median per seed across that seed's 4 cells). No pooled
               p-value; no significance claim from 3 seeds. PER-MODULE
               DEGENERATE HANDLING, separate from the cell-level rule above:
               any module whose own $|{\rm mean}(e_2)+{\rm mean}(e_3)| <
               10^{-5}$, or that has fewer than 2 driven members (Spearman is
               undefined on a single point), is excluded from that cell's
               correlation with the exclusion count reported alongside the
               correlation, never silently.

## What is predicted, and what is deliberately not claimed

  D1  FLAT FIDELITY. Stated above as a guard, not a finding.
  D2  Repair 1 (train-only clustering) is expected to change module
      assignments somewhat from the archived transductive ones, and the
      degree of change is reported descriptively as the ADJUSTED RAND INDEX
      between the archived partition and the train-only one, per cell --
      not a fraction of unequal numeric labels, since module numbering is
      arbitrary and a relabelled-but-identical partition would otherwise
      register as changed. No prediction on direction or size.
  D3  Under HIER-RAND-SIZED, module share is expected to sit between
      HIER-RAND-BAL's and HIER-CLUST-TRAIN's, since it inherits clustering's
      width distribution without clustering's target-to-module assignment.
      Reported descriptively per cell and by seed.
  D4  Whether HIER-CLUST-TRAIN's module share EXCEEDS HIER-RAND-SIZED's, per
      cell, is reported as a count (how many of 12 cells, and which), not as
      a pooled test statistic and not as grounds for an adopt/reject verdict.

**Language, fixed now.** This diagnostic can support only descriptive
statements of the form "the effect attenuates under the size-matched
control" or "the effect persists under the size-matched control" or "the
effect reverses," reported per cell and by seed. It cannot support the words
ADOPT, REJECT, CLOSE, WIDTH, or REPAIRED, because HIER-RAND-SIZED does not
isolate a pure width effect (see the box above) and three seeds do not
establish a rate. Any later document drawing a verdict from these results
must be a separate, freshly pre-registered study with independent systems.

## Void conditions

Void if clustering for HIER-CLUST-TRAIN touches any row outside the training
segment, or if the train-only invariance test fails; if HIER-RAND-SIZED's
size vector is taken from anywhere but the train-only clustering of the same
cell; if the archived output directory is modified; if module labels are not
saved for every arm and cell; if the module-bottleneck formula departs from
$\max(4, 2|\text{module}|)$; if the grid, cluster count, or seeds change from
the original 12 cells; if any verdict word from the excluded list above
appears in the writeup; or if a module-share exclusion is applied after
results are seen rather than by the predefined rule.

## Independent of this run

The correction to the archived protocol and ledger (size control vacuous,
pooled p-value invalid, clustering transductive) stands regardless of whether
this repair run proceeds, and is recorded in `paper/hierarchy_protocol.md`
and `paper/causal_detection_log.md`.

Separately recorded and NOT changed here: the shared ridge helper has no
intercept and uses sample variance. A fidelity issue for its own protocol.
