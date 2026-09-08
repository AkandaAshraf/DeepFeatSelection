# Pre-registration: large systems on one 8 GB GPU

Declared 2026-09-07, before the script was written or run.
EXPLORATORY. Nothing can be adopted from this run. No verdict word
(adopt/reject/close/width/repaired) is licensed; descriptive statements only.

## Two questions, kept apart

Statistical performance and compute scalability are different questions and
this document does not let one answer stand in for the other.

  STATISTICAL   at widths beyond the archived map, does the ranking still
                separate sources from driven channels, and does the deployed
                threshold rule still surface driven channels at all?
  COMPUTE       what does each arm cost in wall-clock, GPU memory and host
                RAM as V grows, and where on THIS hardware does it stop being
                runnable?

A width that cannot be run on this machine is reported as hardware-limited.
It is not reported as tested, and no smaller model is substituted for it.

## What the ledger already knows

The boundary map (n=4000, coupling 0.20, redundancy 0, seeds 0-2, b=32)
covers V up to 240: recall at the deployed rule falls 1.00 / 0.88 / 0.18 /
0.23 / 0.135 across V = 15 / 30 / 60 / 120 / 240, with precision 1.00 and
source FP 0.000 throughout. The bottleneck study established capacity b~2V.
The C01 repair established, at V=30 and 60, that clustered modules exceed a
size-matched random control on module share in 12 of 12 cells. This run
extends V and uses FRESH seeds; it does not re-test those cells and its
V=120 and 240 rows are context against the archive, not a replication of it.

## Design

  SYSTEM      boundary_map.make_system, coupling 0.20, redundancy 0
  DATA LENGTH n = 4000 at EVERY width. Fixed policy, declared: the same
              recording length the whole ledger uses. Consequence, stated
              rather than hidden: training rows (~2400) fall below the
              system-code width at V >= 500, so the largest widths are
              capacity-limited by construction. Reported as unequal capacity,
              not corrected.
  WIDTH       V in {120, 240, 500, 1000}, staged upward (below)
  NOISE       0.0 only. Noise is a separate axis; crossing it here would
              double a run that is already hardware-bound.
  SEEDS       100, 101, 102 -- fresh, disjoint from every earlier study
  GROUND TRUTH  the generator's own: n_src = V // 6 sources, the rest
              driven. Source prevalence is therefore ~0.167 at every V and
              is REPORTED beside every average precision, since AP's chance
              level is the prevalence.
  CAPACITY    system bottleneck 2V (capacity law), module bottleneck
              max(4, 2|module|), clusters m = V // 6 -- unchanged from C01.
              b=2V is used, NOT the archived map's b=32, so V=120 and 240
              rows here are not the archive's cells re-run.
  OUTPUT      ExpOutput/large_system, new directory

## Arms

  FLAT               incumbent: one system code, per-target ridge.
  HIER-CLUST-TRAIN   train-only clustering (C01 repair 1).
  HIER-RAND-SIZED    random modules matching HIER-CLUST-TRAIN's size vector
                     in the same cell (C01 repair 2). HIER-RAND-BAL is
                     dropped: it was the archived control, RAND-SIZED
                     supersedes it, and it costs a quarter of the hierarchy
                     budget.
  HIER-TRUE          oracle from the parent map, ceiling.
  SELFR2             cheap baseline 1: rank channels by ASCENDING self-R2
                     (a channel poorly predicted by its own lags is a
                     candidate for being driven). Zero extra cost; it is the
                     second term of the excess already computed.
  LAGCORR            cheap baseline 2: for each target, the largest
                     |corr(x_j(t), x_q(t+1))| over j != q, minus
                     |corr(x_q(t), x_q(t+1))|. Linear, O(V^2) on 4000 rows,
                     the classic Granger-lite. Fit on training rows only.

No CCM or PCMCI: both are infeasible at V >= 500 on this machine and were
already compared at V=60 in the ledger.

## Provenance, audited not assumed

A preflight asserts, on a small cell, each of: standardisation statistics
come from training rows only; the encoder is fit on the train slice only;
clustering is invariant to perturbation of validation and test rows (ARI =
1.0, the C01 test, re-run at V=120 here); ridge is fit on train indices and
scored on test indices; the target is the one-step lead. Any failure stops
the run before GPU work.

## Resource caps and stop rules, fixed now

  GPU     peak allocated + reserved <= 7.0 GB (8 GB card, 0.6 GB in use)
  HOST    process RSS <= 2.5 GB, AND system free RAM >= 1.0 GB at the start
          of every stage. The machine currently has 3.0 GB free of 27.9; the
          run does not attempt to free anything and treats the shortfall
          as an external constraint.
  TIME    <= 60 min per cell, <= 6 h total
  LOCK    exclusive .agent-lock (C01 machinery), validated resume

STAGING. All three seeds of a width run before the next width starts. A
breach of any cap at width V:
  - records the failure (which cap, at what value, in which arm and seed),
  - marks V and every larger width HARDWARE-LIMITED,
  - stops. It does not retry with a smaller bottleneck, fewer epochs,
    shorter data or a dropped arm. Shrinking silently is what this
    document forbids.
If a single ARM breaches at a width where the others complete, that arm is
marked infeasible at that width and the others' results stand, with the
missing arm shown as missing rather than the width shown as untested.

## Metrics, saved per seed, per arm

  ap_source        average precision, sources positive (chance = prevalence)
  prevalence       n_src / V
  recall_rule      driven recall at the DEPLOYED rule: threshold at
                   max(0, ghost panel max), 30 donor ghosts drawn from
                   channels with self-R2 > 0.9 (the incumbent's own rule,
                   unchanged; its miscalibration is already recorded)
  source_fp_rule   fraction of sources flagged at that rule
  n_flagged        count flagged at that rule
  mod_share        hierarchy arms only, with the C01 near-zero-denominator
                   exclusion rule and raw e2/e3 beside it
  secs             wall-clock per arm
  gpu_peak_mb      torch.cuda.max_memory_allocated + reserved, per arm
  host_rss_mb      process RSS after each arm
  partitions       module labels for every hierarchy arm (Rule 114)
  failures         every cap breach, with the value that breached

## Expectations, stated so they can be wrong

  E1  FLAT's recall at the deployed rule keeps falling with V, continuing
      the archived 0.23 -> 0.135 from 120 -> 240. Direction only.
  E2  FLAT's source AP stays well above prevalence at every width that
      runs. The established pattern is that ranking degrades far more
      slowly than thresholding, and it should hold here.
  E3  HIER-CLUST-TRAIN exceeds HIER-RAND-SIZED on module share at V >= 120,
      continuing C01's 12 of 12 at V = 30 and 60. Count per width, no
      magnitude prediction.
  E4  Hierarchy-arm cost grows roughly as V^2 (the ridge is O(n p^2) at
      p ~ 2V). A V=1000 cell is expected at 20-40 min. If it is far outside
      that, the cost model is wrong and is reported as such.
  E5  NO expectation on either baseline against FLAT. Stating one would be
      storytelling.
  E6  The width most likely to be hardware-limited is 1000, on host RAM
      rather than GPU memory.

Three seeds are not a rate. Nothing here supports a significance claim, and
none is made.

## Void conditions

Void if any cap is relaxed after a breach; if a model, bottleneck, epoch
count or data length is reduced to fit; if the grid, seeds or arms change
after any result is seen; if a hardware-limited width is described as
tested; if the provenance preflight is skipped; or if any verdict word
appears in the writeup.

---

## AMENDMENT, 2026-09-07, before any run

Two definitional corrections found on re-reading the script against this
document, before any cell was computed. Recorded here rather than silently
edited because the protocol is the thing the run is held to.

  GPU CAP METRIC. The cap was written as "peak allocated + reserved".
  PyTorch's reserved pool already contains allocated memory, so that sum
  double-counts and would trip the 7.0 GB cap at roughly 3.5 GB of real use.
  The metric is peak RESERVED, which is the true footprint on the card. The
  cap value is unchanged.

  DEPLOYED-RULE RECALL is reported for FLAT only. The rule is FLAT's own
  ghost panel threshold; applying that number to the hierarchy arms' total
  excess, a different statistic with its own null, is not the deployed rule
  and was never calibrated for it. Hierarchy arms report average precision
  and module share, as C01 did.

---

## Result (2026-09-08): descriptive only, as declared. No verdict word below.

All 12 cells completed (V in {120, 240, 500, 1000} x seeds 100/101/102), all
6 arms in every cell, 72 rows in cells.csv. failures.csv and
hardware_limits.csv are both empty: no cap breach at any width, including
V=1000. Verified from primary output files only
(ExpOutput/large_system/{cells,capacity,failures,hardware_limits}.csv and
the raw NPZ archives); no cell was rerun and no result file was touched to
produce this report.

### Caps observed vs. declared -- V=1000 was not hardware-limited

  metric                     worst observed    declared cap
  GPU peak reserved             964 MB           7000 MB
  host RSS                     1813 MB           2500 MB
  per-cell wall clock          32.4 min           60 min

Every degradation below happened at a fraction of every resource cap, at
every V tested. The statistical and compute questions this document keeps
apart give different answers: compute had headroom throughout; the
statistics did not.

### Capacity, per E4 and E6

  V                         120     240     500    1000
  capacity_ratio           9.25    4.80    2.35    1.19
  cell_secs, mean of 3    108.3   258.5   677.9  1900.9

capacity_ratio is training rows divided by (system code width + own-lag
feature count); it was declared to fall as V grows under the fixed n=4000
policy, and it does, reaching 1.19 at V=1000 -- rows and ridge parameters
nearly equal. E4 (a V=1000 cell in 20-40 min) holds: 1874-1942 s, i.e.
31.2-32.4 min. E6 (V=1000 most likely hardware-limited, on host RAM) does
not materialise: nothing was hardware-limited at any width; the caps table
above shows why.

### E1 -- FLAT recall_rule keeps falling: holds

  V                  120     240     500    1000
  mean (3 seeds)    0.883   0.653   0.095   0.005

Falls further at every step. Magnitudes are not compared against the
archived map (fresh seeds, b=2V rather than the archive's b=32, as
declared).

### E2 -- FLAT's source AP stays well above prevalence: holds through V=240,
### does not hold at V=500 or V=1000

  V           prevalence     FLAT ap_source (min-max, 3 seeds)
  120           0.167              0.881 - 1.000
  240           0.167              0.793 - 0.810
  500           0.166              0.110 - 0.115
  1000          0.166              0.095 - 0.099

At V=500 and V=1000, FLAT's ranking sits AT OR BELOW the chance level set by
prevalence, not above it. This is the sharpest divergence from a stated
expectation in this run, and it is stated plainly because the expectation
was stated plainly.

### E3 -- HIER-CLUST-TRAIN exceeds HIER-RAND-SIZED on module share, V>=120:
### holds at V=120 and V=240, reverses at V=500 and V=1000

  V       HIER-CLUST-TRAIN higher     HIER-RAND-SIZED higher
  120           3 of 3                          0 of 3
  240           3 of 3                          0 of 3
  500           0 of 3                          3 of 3
  1000          0 of 3                          3 of 3

mod_share turns negative for both hierarchy arms at V=500 and V=1000 (e.g.
V=500 seed=100: HIER-CLUST-TRAIN -1.958, HIER-RAND-SIZED -0.184), a range
the C01 test was never run in. Whether a sign flip in that regime is a
meaningful reversal of the V=120/240 pattern, or an artefact of the
underlying e2/e3 terms both turning negative once the whole system is
capacity-starved, is not established by this run. share_excluded is False
throughout, so the near-zero-denominator guard never fired even at these
extreme values -- worth a second look before this metric is read again at
V>=500.

### E5 -- no expectation was stated on either baseline against FLAT; what
### was observed

  V        SELFR2   LAGCORR    FLAT   HIER-CLUST-TRAIN  HIER-RAND-SIZED  HIER-TRUE
  120      0.908     0.786    0.960        0.994             0.953          0.991
  240      0.759     0.654    0.801        0.879             0.819          0.958
  500      0.789     0.626    0.113        0.148             0.116          0.736
  1000     0.796     0.658    0.096        0.098             0.097          0.105

SELFR2 (cost: zero, it is the self-R2 already computed) and LAGCORR (cost:
one O(V^2) pass on training rows) stay within a narrow band across all four
widths. FLAT and every hierarchy arm, including the oracle HIER-TRUE, fall
toward or below prevalence at V=1000. At the largest width tested, both
zero/near-zero-cost baselines rank sources well above every trained arm,
including the oracle.

### Source blindness, checked throughout though not a declared expectation

FLAT's source_fp_rule is 0.000 in all 12 cells, at every width. Recall falls
to zero without ever inventing a false positive.

### A data caveat, traced to source rather than assumed

torch.cuda.reset_peak_memory_stats() runs immediately before FLAT and before
each hierarchy arm (large_system.py:289, :325) but NOT before SELFR2 or
LAGCORR, which run first in every cell and touch no GPU. Their gpu_peak_mb
is therefore the watermark left by the PREVIOUS cell's last arm, not a
measurement of their own footprint -- confirmed by matching every value: 10
of 11 width/seed transitions show SELFR2's reading equal to the prior cell's
HIER-TRUE reading exactly (e.g. V=120 seed=101 SELFR2 reads 74.0, matching
V=120 seed=100's HIER-TRUE exactly). The one exception, V=500 seed=102 ->
V=1000 seed=100 (358.0 -> 22.0, resetting to the same low value seen at the
very first cell of the run), lines up exactly with the machine-crash and
resume already reported for this run between the V=500 and V=1000 stages:
a fresh process starts its CUDA peak counter fresh. Treat SELFR2/LAGCORR's
memory column as uninformative, not as zero, and treat the V=500/V=1000
boundary as a process boundary when reading any watermark-style column
across it.

### Not established

Three seeds, one generating family, one coupling, one data-length policy
(n=4000 fixed at every V, declared in advance to capacity-starve the
largest widths by construction rather than adjusted to compensate). No
noise axis was crossed. The mod_share sign flip at V>=500 is observed, not
explained. Nothing here licenses a claim about any V not tested, and per
this document's own scope nothing here is confirmatory of anything.

---

## Post hoc diagnosis (2026-09-08): the V=500/1000 collapse, adversarially checked

Requested separately from the run above: why does ranking collapse toward
or below chance at V=500 and V=1000, for every trained arm including the
oracle, while two near-zero-cost baselines do not degrade at all. No cell
was rerun; everything below is recomputed from the untouched saved arrays
in ExpOutput/large_system/raw_*.npz and from scripts/boundary_map.py and
scripts/hierarchy_repair.py as they stand. Disk growth from this diagnosis:
this section plus one new pre-registration, both plain text, well under
100 KB combined.

### The leading candidate, and precisely what is and is not established

boundary_map.ridge_r2 uses a FIXED penalty, ALPHA = 1.0, unconditional on
feature count, at every V. Three checks below converge on this as a
PLAUSIBLE mechanism. None of them, individually or together, proves fixed
alpha caused the real run's collapse -- (1) and (2) are POST HOC exploration
of the archived arrays, not pre-registered, and (3) reproduces a possible
mechanism on synthetic noise columns, not the real learned system code.
What they DO establish, without qualification, is that e2 (a small,
V-independent ridge) and e3 (a ridge conditioned on the full, V-scaled
system code) behave very differently as V grows -- that part is a direct
reading of the saved arrays, not an inference.

**1. The saved e2/e3 decomposition separates a small, stable ridge from a
large, V-scaled one, and only the large one fails.** hierarchy_repair.py's
module_readout computes two increments per target: e2 = r_om - base, a
ridge over own-lags (19 features) plus one small module code (~12-30
features depending on module size) -- feature count essentially
independent of V; and e3 = r_oms - r_om, which ADDS the full system code
(width 2V) to that same ridge -- feature count scaling directly with V.
Mean e2 on driven channels, 3-seed mean per width:

  V           120       240       500      1000
  e2       +0.0016   +0.0014   +0.0013   +0.0012    (HIER-CLUST-TRAIN)
  e2       +0.0035   +0.0036   +0.0034   +0.0036    (HIER-TRUE, oracle)

e2 is essentially FLAT across an 8x change in V, for both arms. e3 on the
same channels, driven against source, 3-seed mean:

  V              120       240       500      1000
  e3 driven   +0.0011   +0.0003   -0.0024   -0.0163    (HIER-CLUST-TRAIN)
  e3 source   -0.0002   -0.0005   -0.0013   -0.0084
  e3 driven   -0.0001   -0.0008   -0.0024   -0.0128    (HIER-TRUE, oracle)
  e3 source   -0.0002   -0.0005   -0.0012   -0.0068

By V=1000, e3 on driven channels (-0.0178 to -0.0128 across the three
hierarchy arms) is MORE negative than e3 on sources (-0.0084 to -0.0068):
the system-code-conditioned term does not merely lose signal, it inverts,
scoring the channels it should flag lowest, on average, below the channels
it should never flag.

**2. POST HOC exploration, not pre-registered: recomputing AP from the
saved arrays alone, dropping e3 entirely, recovers the oracle from failing
to strong.** No retraining; this is arithmetic on data already on disk,
3-seed mean. Reported because it is a striking pattern in already-collected
data, not as a confirmed result -- e2-alone was never a declared scoring
rule for any arm, and this comparison was constructed after seeing the
collapse it explains.

  V                 120     240     500    1000
  deployed (e2+e3) 0.991   0.958   0.736   0.105
  e2 alone         0.993   0.845   0.862   0.881

HIER-TRUE's deployed score falls from 0.958 to 0.105 between V=240 and
V=1000 -- to below the 0.166 prevalence floor. Scored on e2 alone, the same
arm holds at 0.845-0.993 across that entire range, INCLUDING at V=1000
where it reaches 0.881. Adding the system-code-conditioned term is net
NEGATIVE by V=500 for the oracle, not merely diminishing; the additive
combination e2+e3 is not robust to the regime where e3's own ridge is
poorly conditioned. HIER-CLUST-TRAIN (deployed 0.994/0.879/0.148/0.098
against e2-alone 0.284/0.283/0.337/0.405) and HIER-RAND-SIZED (deployed
0.953/0.819/0.116/0.097 against e2-alone 0.178/0.205/0.216/0.205) show the
same direction -- e2-alone AP exceeds deployed AP at V=500 and V=1000 for
both -- with lower absolute values throughout, consistent with their
weaker underlying e2 signal rather than a different mechanism.

**3. A synthetic check, using the deployed ridge_r2 helper unmodified,
establishes that fixed alpha CAN produce this failure mode in isolation --
not that it DID, on the real system code.** One genuinely informative
column (fixed signal strength, R2 ceiling 0.028 alone) buried among p-1
pure-noise columns, alpha=1.0, n_train=2397 matching large_system's actual
value:

  p          20     100     500    1000    2000    2400    3000
  R2      0.010   0.000   0.000   0.000   0.000   0.000   0.000

Held-out R2 is driven to the helper's own zero-clamp by p=100 -- p/n_train
= 0.042, far below where p approaches n_train (V=500's actual own+sys_code
width is ~1019, V=1000's is ~2019). This is a LOWER bound on how early the
effect can bite: the real system code is a learned, correlated
representation, not i.i.d. noise, so whether it fails earlier or later than
this synthetic check is not established here -- but the check shows the
mechanism is a generic property of this fixed-alpha estimator, not
something specific to large_system's pipeline, the generator, or the
embedding.

### Ruled out by this diagnosis, not merely unconsidered

  SCORING / CLASS ORIENTATION, checked as directed, against the code and
  against the archived arrays. Traced the sign convention through every
  arm: FLAT (average_precision_score(is_source, -flat), flat = joint minus
  own R2, high = driven, correctly negated for a SOURCE-positive AP);
  SELFR2 (-base fed to record(), double negation resolves to raw self-R2,
  sources ARE well-predicted by their own history, correctly oriented);
  LAGCORR (best cross-corr minus own autocorr, high = driven, correctly
  negated); every HIER arm scores e2+e3 with the same convention as FLAT,
  confirmed by recomputing cells.csv's own ap_source column from the
  archived is_source/is_driven arrays and the archived e2+e3 sums and
  matching it exactly, arm by arm, width by width, all 12 cells. The
  e2-alone recomputation above uses the identical -e2 convention. No
  orientation defect found anywhere.

  BASELINE FAIRNESS / HELD-OUT PROVENANCE. tr_i and te_i are computed once
  per cell (large_system.py:239) and passed BY REFERENCE into every arm's
  ridge calls and into module_readout; no arm recomputes or diverges from
  that split. SELFR2's score is itself a held-out ridge R2 (self_r2, the
  same quantity computed for every other arm's own-lag baseline).
  LAGCORR's score is a training-set correlation statistic, not a fitted
  model evaluated out-of-sample, so it is not "held out" in the same sense
  -- worth noting as a definitional difference, not a leakage or fairness
  defect, since it never touches test rows at all rather than touching them
  improperly.

  ALREADY-KNOWN RIDGE FIDELITY ISSUES (no intercept, population rather than
  sample variance). Confirmed present, unchanged, and explicitly held fixed
  across every arm in this comparison by prior instruction (recorded
  earlier for the C01 hierarchy work). Not implicated as a NEW finding here
  and not proposed for change as part of addressing the leading cause below
  -- a separate, smaller-magnitude issue from the fixed-alpha one.

### Ranked next tests

  1. LEADING CAUSE, CHEAPEST TEST. Pre-registered below
     (paper/ridge_alpha_scaling_protocol.md): a CPU-only synthetic
     extension of check 3 above, testing whether alpha scaled to feature
     count (or a cross-validated alpha) recovers a planted signal at the
     SAME (n_train, p) shapes large_system actually used, before spending
     any GPU time on a real retrain. No encoder training, no dataset
     touched, minutes of CPU time.
  2. CONTINGENT ON (1). If alpha-scaling recovers the synthetic signal, a
     real V=500/V=1000 retrain with a corrected alpha, to confirm the fix
     transfers from synthetic noise columns to the actual learned system
     code. This is the expensive, confirmatory step and is explicitly NOT
     licensed by this diagnosis alone.
  3. LOWER-RANKED, LARGER CHANGE. If (1) and (2) do not recover
     performance, an architectural alternative: score own, module and
     system-code contributions with SEPARATE small ridge fits and combine
     the predictions rather than concatenating all three into one design
     matrix, keeping every individual ridge's p small in the way e2's
     already is. Not designed or scoped here.

### Not established, stated plainly

Whether the real (correlated, learned) system code fails at the same p as
the i.i.d.-noise synthetic check, whether alpha-scaling is sufficient or
only necessary, and whether the CLUST-vs-RAND module-share reversal at
V>=500 has a fuller explanation than "both terms lose meaning once e3
crosses zero" are all open. No claim is made about any V not in the
original 12 cells.
