# Parent screening: extended laptop replication

Registered before implementation and before any scientific seed is run.
This is a new resource envelope for the design in
`parent_screening_protocol.md` at commit `1da0192`, with the numerical
implementation at `8f81914`. The earlier resource-limited investigation
remains separate. No scientific pilot results have been inspected.

## Frozen design and prediction

Retain all six arms, V=240, n=4000, train/validation/test separation,
hyperparameters, unresolved gain threshold 0.01, all-candidate fallback,
metrics, chance reference RANDOM, and the original G1--G4 gate unchanged.
The hypothesis remains that learned clustered codes retain parents with
recall >=0.90, coverage >=0.80, and >=0.05 recall advantage over every
required baseline in both families while every target obeys k=ceil(.1*(V-1)).
Engineering abstention suggests this may fail; it is not scientific evidence.
Fresh pilot seeds: family1 24001--24006; family2 25001--25006.
Use all twelve systems without outcome-dependent early stopping.

## Resources and stopping

The user authorized extended execution on their laptop. Stage B has a
20-hour cumulative active runtime cap across restarts, including interrupted
work. Keep GPU 6 GiB, whole-process-tree RSS 3072 MiB, free RAM >=2048 MiB,
and output disk <=100 MiB. Any resource breach stops execution and is
resource-limited, not a scientific negative. A failed scientific gate ends
this direction. No Stage C or downstream run is authorized by this launcher;
a pass requires a separate review against the original confirmation protocol.
The registered LASSO solver/iteration limit remains unchanged; report its
nonconvergence counts.

## Power interruption and integrity

Use separate output `ExpOutput/parent_screening_laptop`. Persist completed
arms (including clustering) and completed systems with atomic replacement,
flushed writes, checksums and a code/configuration identity. On restart,
reuse only validated checkpoints of that identity. Replay the interrupted
arm with the same seed; never change a model or select a result on restart.
Ground truth is not passed to screening and is written after shortlists.
Persist active runtime in prepaid intervals of at most 60 seconds; a sudden
interruption may conservatively charge the unused interval. Powered-off time
does not count. Do not reset a cap breach automatically. Corrupt checkpoints
or code/config changes cause a clear refusal, not silent result mixing.
Use an exclusive OS-held run lock plus the repository training lock. A dead
process's own training lock can be reclaimed only after verifying ownership;
foreign/live locks must be refused. Resume is an explicit single command,
not an automatic login task. A power loss can lose the current arm, but
completed arms/systems are skipped. Storage hardware must honor flushed
writes; no software-only guarantee covers disk failure.

## Validation before launch

On engineering data only, compare uninterrupted and interrupted/resumed
outputs, verify completed work is skipped, check corrupt/config-mismatched
checkpoints are refused, and check runtime and concurrent-launch safeguards.
Commit the validated runner before launch. Preserve launch logs and results;
do not push commits.

## Result (2026-09-21): gate FAILED in both families; cheap lagged correlation is the finding

Twelve registered systems (family 1 seeds 24001-24006, family 2 seeds
25001-25006; V=240, n=4000), all six arms, one uninterrupted run of 10.6 h
active time on 2026-09-21 (log ExpOutput/parent_screening_laptop/
run_20260921_055204_909.log), no resource breach, no resume. Every metric
below was recomputed independently from the saved shortlists and truth
files after the run and matched the recorded pilot_metrics.json exactly;
the gate was re-evaluated independently with the same verdict. Audit gate
136 passed, 0 failed. Means over six seeds; chance recall k/(V-1) = 0.100;
budget = seeds in which every non-root target had |C_q| <= k = 24.

  family 1 (logistic maps)  recall  coverage  cand.frac  unresolved  budget
  RANDOM                     0.103    0.028     0.100      0.000       6/6
  LAGCORR                    0.510    0.344     0.100      0.000       6/6
  LASSO                      0.535    0.355     0.100      0.000       6/6
  PCA-GROUP                  0.999    0.998     0.967      0.963       0/6
  LEARNED-SIZEDRAND          0.999    0.998     0.978      0.976       0/6
  LEARNED-CLUSTERED          1.000    0.999     0.990      0.989       0/6

  family 2 (AR + tanh)      recall  coverage  cand.frac  unresolved  budget
  RANDOM                     0.098    0.036     0.100      0.000       6/6
  LAGCORR                    0.928    0.873     0.100      0.000       6/6
  LASSO                      0.891    0.838     0.100      0.000       6/6
  PCA-GROUP                  0.915    0.864     0.476      0.418       0/6
  LEARNED-SIZEDRAND          0.932    0.898     0.633      0.592       0/6
  LEARNED-CLUSTERED          0.929    0.888     0.598      0.553       0/6

Gate: G1 and G2 pass in both families on paper; G3 fails in both (arm 6 over
PCA-GROUP +0.000 / +0.014, over LEARNED-SIZEDRAND +0.001 / -0.003, over
LAGCORR +0.490 / +0.001); G4 fails in 6 of 6 seeds in both. GATE FAIL.
Stage C and the downstream test are not reached.

What the numbers mean, in order of importance:

1. The arm-6 recall is hollow. Under the registered unresolved rule the
   learned clustered screen abstained on 98.9% of non-root targets in
   family 1 and 55.3% in family 2, returning all 239 candidates for each,
   so its "recall" is the recall of not screening. This is the outcome
   predicted in amendment 2 of the original protocol before any pilot seed
   ran; it is now a pilot result, not an engineering observation. The
   registered 0.01 threshold decided the pilot; ranking quality was never
   tested at the fixed budget for the group arms.
2. Independently of abstention, the learned representation did not justify
   itself: on the identical partition, PCA codes at the same width match it
   in both families, and a random partition with the same size multiset
   matches it too. Neither the encoder nor the clustering earned its cost.
3. The cheap method is the finding. In family 2, maximum absolute lagged
   correlation at a strict 10% budget retains 92.8% of true parents and
   fully covers 87.3% of targets, against chance 0.100 / 0.036, with LASSO
   close behind (0.891 / 0.838). Under the protocol's own verdict list this
   is category 2, CHEAP METHOD IS SUFFICIENT, and the named baseline is
   LAGCORR (validation chose the raw variant in all six family-2 systems
   and the own-history-residualised variant in all six family-1 systems;
   both variants' shortlists are saved for every system).
4. Family 1 is hard for every arm: nothing that respects the budget exceeds
   0.535 recall or 0.355 coverage. For coupled logistic maps at this
   coupling, a 10% candidate budget is NOT USEFUL AT THIS BUDGET (category 3)
   for any of the six arms, learned or not.

Limitations that bound these statements: LASSO fits were non-converged at
the registered 2000-iteration limit in 791-851 of 1440 fits per family-1
system and 455-474 per family-2 system, so that baseline is under-stated
and the LASSO numbers are lower bounds on what the dictionary could do.
Two generator families, one coupling, one width, six seeds each: a pilot,
not a confirmation, and none of the four verdicts is a statement about
other widths, couplings or real recordings. The learned arms were not
tested at a scale-aware or zero abstention threshold; a study that does so
is a new design with fresh seeds, not a rerun of this one.

Verdict: the learned bounded-code screen does not earn a place in a
workflow at this budget. Family 2: CHEAP METHOD IS SUFFICIENT (LAGCORR).
Family 1: NOT USEFUL AT THIS BUDGET, any arm. No adoption, no Stage C.
