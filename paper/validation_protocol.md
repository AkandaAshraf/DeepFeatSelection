# External validation protocol: Causal Chambers + CauseMe

Pre-registered before any external result was computed (2026-08-09). The point
of this document is that every choice below was fixed while the only numbers we
had were from our own generators. Deviations get logged here with reasons, not
silently applied.

## Why these two

Every result in `causal_detection_log.md` shares one weakness: we generated the
data we validated on. The chambers are real physical devices (Gamella, Peters &
Bühlmann, Nat. Mach. Intell. 2025) whose ground-truth graph is known by
construction and intervention; CauseMe (Runge et al.) scores blind against
hidden ground truth on a public leaderboard, which also supplies the comparison
against PCMCI and the rest of the field that our internal work has avoided so
far. A method that only wins at home is not a tool.

## Methods under test — frozen as validated, no per-dataset tuning

The trio from `network_scale.py`, hyperparameters unchanged:

1. **MLP-LOCO recreation gain** (primary): recreate node U's delay embedding
   from the other nodes' embeddings; the gain from including V scores the link.
   MLP 32-32 tanh, Adam 3e-3, ≤25 epochs, patience 4, batch 64.
2. **kNN-LOCO** (k=5): identical structure, the cheap-baseline control.
3. **Pairwise CCM** (E=3 and optimal-E): the incumbent.

Directionality convention, fixed by Takens' imprint and applied identically
everywhere: the score for the directed link U→V is the gain of *V's* channels
in recreating *U* — the effect carries the cause's imprint, so the downstream
node is the one that helps recreate the upstream one.

Protocol constants carried over unchanged: contiguous train/early-stop/test
splits with an embedding-span embargo; standardisation from train statistics
only; recreation r² clamped at zero before differencing; embedding dimension
FIXED across all arms of a comparison (E=3, tau=1 for chamber sensor data —
declared here, chosen before inspecting any chamber series beyond variable
names). Statistics never flipped or re-oriented after seeing results.

## Causal Chambers

Dataset: `wt_walks_v1` (wind tunnel, actuator random walks — the dataset the
paper itself designates for time-series causal discovery). Ground truth: the
wind-tunnel graph as published by the authors (package `causalchamber`
ground-truth API or the paper's appendix), taken as-is.

Scoring: AUROC of directed link scores against the ground-truth adjacency,
plus null-calibrated detection. Real data has no zero-edge twin, so the null
is per-link circular-shift surrogates (3 independent shifts of the candidate
source's series; threshold = the maximum surrogate score). Circular shifts
preserve each series' autocorrelation while destroying temporal alignment,
matching `deepfeatselect.ccm.circular_shift_surrogate`.

Predictions on record before running:
- CCM will rank strongly wherever actuator→sensor links are near-deterministic
  but its absolute scores will again resist null calibration.
- The MLP-LOCO's calibrated detection should transfer if the network_scale
  result was real; if it collapses on real physics, the August arc's
  learned-estimator win was an artefact of our simulator family, and that
  conclusion goes in the ledger with the same prominence the win had.
- Exogenous actuator nodes (random-walk roots) have no parents: any method
  reporting confident incoming links to them is producing false positives we
  can count precisely.

## CauseMe — DROPPED (2026-08-09, before any run)

Registration failed (platform in beta), and the project policy is fully-open
sources only — no registration-gated platforms, no third-party submission
forms. The one function CauseMe uniquely offered, comparison against the field
standard, is preserved without it: **PCMCI runs locally** (tigramite, pip,
Runge's official implementation) on the same chamber data against the same
ground truth. PCMCI joins the method roster as the fourth arm, ParCorr test,
tau_max = 3 (matching E), directed score U→V = max |val| over lags.
CausalRivers noted as a possible future open benchmark (data on GitHub) but
shelved on the same trust grounds for now.

## Declared adaptation: ghost-source null (logged before any run)

The network experiments calibrated the noise floor on a matched zero-edge null
network; real data has no such twin, and per-link surrogate refits would cost
V² × shifts extra fits. The replacement, declared here: each target's LOCO
round includes one **ghost source** — a circularly shifted copy of a randomly
chosen real variable (shift and donor drawn per target). A ghost has the
marginal and autocorrelation structure of real data but no temporal alignment,
so any gain attributed to it is noise. The floor is the maximum ghost gain
across targets; calibrated detection counts true edges above it. One extra fit
per target. Constant columns (variables not varied in a given experiment) are
excluded from scoring for that experiment — a constant has nothing to detect
either way.

## Tier 2: the circadian clock circuit (declared 2026-08-10, before data inspection)

Dataset: **GSE11923** (Hughes et al. 2009), mouse liver transcriptome, 48
timepoints at 1-hour resolution — the highest-resolution standard circadian
time course, fully open on NCBI GEO. Platform GPL1261; probes mapped to gene
symbols via GEO's own annotation file, probes averaged per gene, series
z-scored. If a clock gene has no probe or a flat profile, it is reported as
excluded, not silently dropped.

Ground truth: the mammalian TTFL core, established by knockout genetics —
interventional truth, not inference. Directed edge list (transcript-level
reading of the circuit; activation and repression both count as edges since
the methods under test detect, they do not sign):

    Arntl -> Per1, Per2, Per3, Cry1, Cry2, Nr1d1, Nr1d2, Dbp
    Nr1d1 -> Arntl        Nr1d2 -> Arntl        Rora -> Arntl
    Per1 -> Nr1d1, Dbp    Per2 -> Nr1d1, Dbp    (CLOCK:BMAL1 repression
    Cry1 -> Nr1d1, Dbp    Cry2 -> Nr1d1, Dbp     routed through the complex)

Genes scored: Arntl, Per1, Per2, Per3, Cry1, Cry2, Nr1d1, Nr1d2, Rora, Dbp.
Arntl is the hub (out-degree 8): the maturity mechanism predicts its LOCO
edges are the fragile ones, so LOCO results are reported at BOTH an immature
checkpoint and the mature endpoint, per the maturity finding.

Declared analysis choices: embedding E=3, tau=3 hours (a quarter-period-scale
span; tau=1 on a 24-h oscillation sampled hourly makes embedding coordinates
nearly collinear). Series detrended by first-differencing before embedding —
the maturity experiment showed circular-shift ghosts are invalid on
nonstationary series. n=48 is TINY: after embedding and splits the test
segment is single-digit points, so the pre-registered expectation from our
own validity map is that kNN/CCM/PCMCI carry the detection and the MLP arms
starve; if the MLP does anything at all here it exceeds expectations. This is
the honest shape of real transcriptomic time courses and exactly why Tier 2
exists.

## Tier 2b: C. elegans whole-brain imaging (declared 2026-08-10, before any detection run)

THE APPLICATION TEST, governed by the project rule: the deliverable is what the
method finds in real neural data that the alternatives miss — named indirect
neuron pairs that CCM, kNN and PCMCI falsely report as connected and the
conditional MLP correctly rejects — or an honest "it adds nothing here".

Data: Kato et al. 2015 (OSF 2395t, fully open), WT_NoStim — five immobilised
worms, spontaneous dynamics, ~3,100 timepoints at ~2.9 Hz. Per-worm analysis
on worms 0-2 (declared now; 4 held out for replication if anything is found).
Traces: deltaFOverF_bc (bleach-corrected), z-scored, first-differenced per
the stationarity rule. Neurons: identified names intersected with the
connectome's, capped at 25 per worm ALPHABETICALLY (not by variance — that
would peek), E=3, tau=3 samples (~1 s, calcium timescale).

Ground truth: `herm_full_edgelist.csv` (OpenWorm/c302, Cook-derived).
Chemical synapses are directed edges; gap junctions count in both directions.
Truth is anatomical POSSIBILITY: a detected edge with no anatomical route is
a countable false positive, while an anatomical edge carrying no functional
signal in this state is expected and not counted against recall claims —
recall is reported but precision against anatomy is the deliverable metric.
The decisive stratum, fixed now: non-adjacent pairs with a 2-hop anatomical
route (mediated pairs). Strong coupling near synchrony is this preparation's
known regime, so the synchrony-chain scoreboard predicts classical methods
false-positive on this stratum and the conditional MLP rejects it.

Methods: the frozen four. MLP checkpointed at {2,5,10,25,50} epochs, all
reported. Ghost-source nulls (valid after differencing). PR-AUC beside AUROC
with baselines, per worm and pooled.

## Open-science deployments: climate and clinical EEG (declared 2026-08-15)

Publication stance now governing all of this: preprints (bioRxiv/arXiv) and
open code only; the deliverable is research people can use, not venue
acceptance. Two deployments of the intervention-validated excess detector,
predictions fixed before either dataset is opened.

**Climate (NCEP/NCAR Reanalysis-1 daily sea-level pressure, 1948-2024,
2.5-degree grid: V=10,512 cells, n~28,000 days; fully open, no registration
-- ERA5 was rejected because Copernicus requires an account).**
Preprocessing: remove day-of-year climatology (deseasonalise), then
first-difference; E=3, tau=1 day. The DAILY SUNSPOT NUMBER (SILSO) is
appended as a channel: a known exogenous forcing is a true root, and roots
are invisible-by-design to the drivenness statistic.
Pre-registered: (1) ghost ~ 0; (2) the sunspot channel's excess ~ 0 -- the
root-invisibility property tested on real data for the first time; (3) the
driven core is spatially CLUSTERED (fish-style nearest-neighbour test vs
random cells), not scattered. Exploratory, not pre-registered: WHERE the
driven core sits (storm tracks? tropics?) -- reported as a map with no
prior claim.

**Clinical EEG (PhysioNet CHB-MIT, subject chb01: 23 scalp channels, 256 Hz,
seizure annotations; fully open).** THE CAVEAT STATED UP FRONT, ALWAYS:
seizure-onset zones are SOURCES and sources are invisible to this statistic
by design -- the tool maps the spread network, never the origin, and any
output must say so.
Pre-registered: (1) ghost ~ 0 per record; (2) drivenness CONCENTRATION
(top-k share of total excess across channels) is higher during ictal
epochs than interictal ones -- seizures recruit the network into a driven
regime; (3) the ictal drivenness pattern is reproducible across chb01's
seizures (rank correlation of channel drivenness between seizure events).
Windows: per-epoch excess with the standing pipeline on 256 Hz data
downsampled 4x (calcium-free regime: tau=1 at 64 Hz), n per epoch >= 2,000
samples. Sensitivity to these declared constants is reported, not tuned.

## Metric addendum (2026-08-10)

Edge detection is class-imbalanced everywhere (11-27% positives), so
**PR-AUC (average precision) is reported beside AUROC in every evaluation,
with the positive-rate baseline stated**. AUROC alone already produced one
withdrawn conclusion (the circadian "regime reversal"). Rankings that
disagree between the metrics are reported as disagreeing, with PR-AUC
primary for tool advice since a user acts on the top of the list.

## What counts as success, decided now

The goal, binding on how results are framed: this is not a
contest and publication is not the point — the point is handing working
researchers a tool they can trust, so that people benefit. Comparisons against
CCM and PCMCI exist to make the tool's advice honest ("for your data shape,
use X"), not to declare winners.

- **Transfer**: MLP-LOCO's null-calibrated detection reproduces on
  `wt_walks_v1` (true edges above the ghost floor, few actuator false
  positives). If it does, the calibration recipe is real and travels.
- **The validity map is the deliverable**: whatever the outcome, the result is
  a measured method-selection guide — which estimator to trust at which data
  size, coupling strength, and network width, with a calibration procedure a
  user can run on their own data. PCMCI or CCM winning a regime goes into the
  map as a recommendation, with the same prominence as any win of ours.
- **Honest failure**: if the learned estimator transfers nowhere, the tool is
  the map plus the two small transferable findings (embedding-dimension
  detector; CCM k>E+1 upgrade), and the ledger says so plainly.
