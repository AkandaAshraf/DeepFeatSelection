# Pre-registration: repairing the hierarchy evaluation (task C01)

Declared 2026-09-06, before the script was written or run.
EXPLORATORY. Nothing can be adopted on this run whatever it shows.

Raised by Codex as task C01 in the agents' shared log, verified independently
against the 12 archived NPZ files before acceptance. This is a **validity
repair of an archived result**, not a reopening of the rejected architecture
and not an attempt to adopt its secondary claim.

## The three defects being repaired

All three are errors in the 2026-09-06 hierarchy run, confirmed by
re-derivation from its own artifacts.

  1. TRANSDUCTIVE CLUSTERING. `modules_for` received the full recording and
     correlated over every row, so module assignment was fit on train,
     validation and test together. The encoder and every ridge readout are
     train-only, so clustering is the sole leak, but the archived result is
     transductive and is not evidence for a train-only deployment.

  2. THE SIZE CONTROL WAS VACUOUS. It compared unweighted mean module size,
     which is $V/m$ by construction and therefore reads 6.0 for every arm
     before any data exists. The quantity that matters is the width each
     TARGET receives. Measured across the same 12 cells:

       arm          unweighted mean   target-weighted   driven-weighted
       CLUST             6.000            9.972             9.742
       RAND              6.000            6.000             6.000
       TRUE              6.000            6.706             6.847

     Clustered modules are lopsided, `[16, 9, 2, 2, 1]` at V=30 seed 0 against
     random's `[6, 6, 6, 6, 6]`, so a clustered target sits in a module about
     1.66x wider and receives a wider code. This establishes a CONFOUND. It
     does not establish that width explains the effect, which is what this
     run exists to decide.

  3. STATISTICS. The graded relationship was reported as a Spearman over 157
     pooled modules at $p = 5\times10^{-17}$. Modules within a cell share one
     system, one encoder and one split, so that pools correlated units as
     independent. This is Rule 101 repeated. The test here is at the CELL
     level instead.

## Arms

Five arms. Everything except module assignment is held fixed: generator,
ridge, widths, epochs, bottlenecks, metrics.

  FLAT              incumbent, one system code, per-target ridge. Unchanged.
  HIER-CLUST-TRAIN  clustering fit on TRAINING ROWS ONLY, the repair.
  HIER-RAND-BAL     balanced random modules, equal sizes. The original
                    control, retained at Codex's request as the reference the
                    archived result used.
  HIER-RAND-SIZED   random modules whose size VECTOR exactly matches
                    HIER-CLUST-TRAIN's in that cell, assigned to targets at
                    random. THE WIDTH CONTROL.
  HIER-TRUE         oracle from the parent map, upper bound.

HIER-RAND-SIZED is built from the NEW train-only clustering, not from the
archive. Repairs 1 and 2 interact: train-only clustering yields a different
partition and therefore a different size vector, so matching the archived one
would match the wrong widths.

Size-matching equalises which widths EXIST, not which targets receive them.
Assigning the matched size vector to targets at random is the correct null for
"structure beyond width", and that is what is declared here.

## Design

  SYSTEM   boundary_map.make_system, n = 4000, coupling 0.20, redundancy 0
  WIDTH    V in {30, 60}; system bottleneck 2V, module bottleneck 2|module|
  NOISE    {0.0, 0.05}
  CLUSTERS m = V // 6, fixed, unchanged from the archived run
  OUTPUT   ExpOutput/hierarchy_repair, a NEW directory; archived artifacts
           are preserved untouched

  SEEDS, in two disjoint sets:
    DIAGNOSTIC    0, 1, 2   the archived seeds. Reusing them is a diagnostic
                            rerun against a known result, NOT a confirmatory
                            test, and is labelled so throughout.
    CONFIRMATORY  10, 11, 12  fresh systems never analysed. The decisive
                            prediction is judged HERE.

24 cells x 5 arms.

## Metrics and their chance levels, computed before committing

Rule 123. Driven channels are 25 of 30 and 50 of 60, so driven-positive
average precision has a base rate above 0.8 and a ceiling below 1.25x. It
cannot discriminate and is not used.

  DETECTION    average precision for the MINORITY class, sources, base rate
               0.167 at both widths, scored on the negated total inflow.
  MODULE SHARE mean excess_2 / (mean excess_2 + mean excess_3) on driven
               channels. Descriptive scale, compared BETWEEN arms within a
               cell, which is why no absolute chance level applies.
  GRADED       Spearman between a module's fraction of in-module parents and
               its share, computed WITHIN each cell, then a sign test over the
               12 cell-level values per seed set. The unit is the cell.

## Predictions, fixed now

  R1  DISQUALIFYING. On the confirmatory seeds, HIER-CLUST-TRAIN's detection
      average precision is within 0.05 of FLAT's. Repairing the leak must not
      cost detection. If it does, the repaired hierarchy is rejected whatever
      else it shows.
  R2  DECISIVE, judged on the CONFIRMATORY seeds only. HIER-CLUST-TRAIN's
      module share exceeds HIER-RAND-SIZED's. This is the entire question: if
      clustered modules capture more inflow only because they are wider, the
      size-matched random arm reproduces them and the archived ordering was a
      width artefact.
  R3  HIER-RAND-SIZED exceeds HIER-RAND-BAL. This isolates the pure width
      effect. Predicted because wider codes explain more; if the two match,
      width does not matter here and the archived control reached the right
      conclusion for the wrong reason.
  R4  GRADED, at the cell level. Within-cell Spearman between a module's
      in-module-parent fraction and its share is positive in at least 9 of the
      12 confirmatory cells (sign test $p < 0.05$ at 10 of 12; 9 is reported
      as directional).
  R5  GUARD, SCOPED PER ARM (Rule 124). Mean excess_2 on driven channels
      exceeds zero. An arm whose module level is inert is disqualified; the
      run is not.
  R6  NO PREDICTION. The transductive-versus-train-only gap on the diagnostic
      seeds, reported descriptively as how much the leak was worth.

## The rule, fixed now

NONE, and no adoption is possible.

  REJECTED   R1 fails. The repaired hierarchy costs detection.
  WIDTH      R2 fails with R5 intact. The archived module-share ordering was a
             width artefact, the secondary claim is WITHDRAWN in the ledger,
             and the localisation line closes.
  REPAIRED   R1, R2 and R5 hold. The archived finding survives repair on fresh
             systems. Still exploratory; licenses a powered localisation
             pre-registration scored on search-space reduction, and nothing
             else. Nothing enters the manuscript.

## Void conditions

Void if clustering for HIER-CLUST-TRAIN touches any row outside the training
segment; if HIER-RAND-SIZED's size vector is taken from anywhere but the
train-only clustering of the same cell; if the archived output directory is
modified; if R2 is judged on the diagnostic seeds; if the grid, cluster count,
seeds or metrics change after any result is seen; or if any pooled-module
p-value is reported as evidence.

## Independent of this run

The archived protocol result and ledger entry assert that mean module size is
6.0 in every arm and therefore the ordering is not a width artefact. That
sentence is unsupported. Per Rule 116 the correction is appended with the
corrected numbers, the defective claim and what depended on it, rather than
edited into the original text. That happens whether or not this run proceeds.

Separately recorded and NOT changed here: the shared ridge helper has no
intercept and uses sample variance. A fidelity issue for its own protocol.
