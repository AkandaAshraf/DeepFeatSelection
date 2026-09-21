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
