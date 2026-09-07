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
