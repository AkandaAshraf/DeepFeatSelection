# Mapping the driven core of large dynamical systems with a ghost-controlled excess-predictability statistic

**Akanda Ashraf** — preprint draft v1.0, 2026-08-15. Target: bioRxiv.

## Abstract

Identifying which variables of a large dynamical system are driven by the
system — as opposed to evolving autonomously — is a first-order question in
neuroscience, climate science and physiology, but existing causal-detection
methods cannot ask it at scale: pairwise convergent cross mapping (CCM)
requires a quadratic scan we measure at ~14,700 compute-hours for 10,000
variables, and conditional-independence frameworks either starve
(nearest-neighbour conditioning collapses beyond ~20 dimensions) or lose
validity near synchrony. We introduce a simple, scalable statistic:
**excess-over-self predictability**, the gain in one-step predictability of a
variable when a learned low-dimensional code of the entire remaining system is
added to a flexible model of the variable's own history. The code comes from
one unsupervised masked autoencoder ensemble; every per-variable readout is a
ridge regression, so a complete scan of 71,721 variables runs in minutes on a
laptop. The statistic is **audited by construction**: a *ghost channel* — a
circularly shifted copy of a real variable, carrying realistic marginals and
autocorrelation but no temporal alignment — is embedded in every analysis and
must score zero, giving each dataset a falsification test that requires no
ground truth. We validate on synthetic systems where truth is known (top-10
precision 30/30 across three system seeds; zero false alarms among 907
autonomous channels; both large-V confounds provably cancelled), then across
four open real-world domains: (i) in *C. elegans* whole-brain imaging the
statistic blindly ranks the canonical motor-command circuit (AVA, AVE, RIM,
AIB and motor neurons) as the brain's driven core in five of five wild-type
worms, with all pre-registered predictions passing on two held-out animals;
(ii) in worms with the command hub AVA genetically silenced, the statistic
tracks the intervention neuron-by-neuron — the silenced hub falls hardest,
its module collapses as pre-registered, and two replicated observations
emerge: RIM/AIB retain drive without AVA, and drive reorganises onto the
forward hub AVB; (iii) in a 71,721-neuron zebrafish recording the driven core
is spatially coherent (1.78× tighter than chance, >20σ) and
posterior-midline; (iv) in 77 years of daily sea-level pressure the driven
core is the deep tropics (99% of top cells within 30° of the equator — the
Walker/ENSO belt), and a daily solar-forcing channel scores exactly zero,
confirming the statistic's designed blindness to exogenous roots on real
data. We state the statistic's semantics and limits precisely: it detects
*being driven*, is blind to pure sources, is a high-precision top-k detector
rather than a full-population ranker, and requires roughly 1,500+ time
samples. All data are public, all analyses were pre-registered before data
inspection, and code is open.

## 1. Introduction

Causal structure discovery from time series has strong tools at small scale:
convergent cross mapping (CCM) exploits Takens' theorem on attractor
reconstructions [Sugihara et al. 2012; Takens 1981], and constraint-based
frameworks such as PCMCI condition away spurious associations [Runge et al.
2019]. But the question practitioners increasingly hold — *which of my ten
thousand variables does this system actually drive?* — sits outside every
tool's envelope. We measured the walls directly on common hardware: full
pairwise CCM costs 528 ms per pair, projecting to ~1.7 years of compute for a
single 10,000-variable scan; cross-mapping against principal-component
summaries fails outright in our tests (AUROC 0.26–0.57 with a maximal false
positive on its own control); k-nearest-neighbour conditioning starves by 18
conditioning dimensions; and PCMCI with partial correlation acquires false
positives on mediated links as coupling strengthens toward synchrony (33% →
100% across our coupling ladder).

This paper takes a deliberately narrower question — *membership*: is variable
X driven by the system at all? — and shows it can be answered at scale with
honest error control. Three ideas combine. First, **amortisation**: one
unsupervised masked autoencoder compresses the entire system into a
low-dimensional code, so conditioning on "everything else" is a single learned
representation rather than a combinatorial search. Second, an
**excess-over-self readout**: the statistic is the *gain* in predictability of
X's next step when the code is added to a flexible model of X's own history —
a quantity in the lineage of Granger causality and transfer entropy [Granger
1969; Schreiber 2000], estimated here at three orders of magnitude more
variables than published neural implementations demonstrate [cf. Tank et al.
2021]. Subtracting self-predictability is not optional: without it, we show
the raw readout degenerates at scale into a *typicality* meter that ranks
periodic and template-like channels highest (a companion paper analyses this
failure mode and two others). Third, a **ghost control**: every analysis
embeds a circularly shifted copy of a real channel. The ghost has the
marginal distribution and autocorrelation of real data and no temporal
alignment with anything; any excess it earns is, by construction, artifact.
The ghost is the tool's per-dataset falsification test — it caught every
failure mode we encountered during development, and it passes in every
analysis reported below.

## 2. The statistic

Let x_q(t) be variable q of a multivariate series, preprocessed by
per-variable standardisation and first-differencing (stationarity is a
prerequisite; we show below what breaks otherwise). Form delay embeddings
(E = 3 throughout) and the joint state Z(t) of all variables.

**Code.** Train M masked autoencoders (independent initialisations) on Z:
random subsets of channels are corrupted at the input each batch, the loss is
computed on masked positions only [after He et al. 2022], and a hard linear
bottleneck (width b ≪ dim Z) forces a compressed code c_m(t). Ensembling
over M minima separates data-determined structure from optimisation accident.

**Readout.** For each variable q and each model m:

    excess_m(q) = R²[ x_q(t+1) | poly(own lags of q) ⊕ c_m(t) ]
                − R²[ x_q(t+1) | poly(own lags of q) ]

with both terms fit by ridge regression on a training segment and evaluated
on a held-out contiguous tail (R² clamped at zero). The self-model's
function class matters: it must be flexible enough to absorb the variable's
own dynamics (we use polynomial features to degree 3; a linear self-model
provably under-absorbs and resurrects confounds). The reported score is the
consensus mean over models.

**Semantics, stated exactly.** A positive excess means the system's state
carries information about X's next step beyond X's own past — X is *driven*.
The converse direction is invisible: a pure source (a variable that drives
others but is driven by nothing) is perfectly predicted by its own history,
so the code can add nothing. This blindness is not a defect but a designed
asymmetry, and it is *testable*: known-exogenous variables must score zero,
which we verify on chamber actuators, silenced neurons, and solar forcing
below.

**Cost.** Training is minutes on a consumer GPU even at 71,721 variables
(~80 s per model with a standard `tf.data` input pipeline); each readout is
a ridge fit, ~7 ms per variable; saved encoder weights make any new readout
an evaluation rather than a retrain.

## 3. Validation where truth is known

**Synthetic membership.** Systems of V ∈ {30, 1000} logistic-map channels in
which a minority sub-web is coupled (sparse random DAG) and the rest evolve
autonomously; ground truth is membership by construction. At V=1001 (94
members, baseline precision 0.094), the consensus excess achieves
**precision@10 = 10/10 and precision@20 = 16–20/20 across three independent
system seeds (30/30 at top-10 overall)**, with all 907 autonomous channels —
including 61 periodic ones and the ghost — pinned at zero excess. A
heterogeneous pool (autonomous channels drawn from four dynamical families
including stochastic AR(1)) shows the same behaviour, and an imperfect
self-baseline does not manufacture false positives: the code must actually
carry the missing information, and for autonomous channels it does not.

Two boundaries measured honestly. The statistic is a **top-k detector**:
members whose drive share falls below the readout's noise floor receive
slightly negative excess and rank below the zero-pinned non-members, so
full-population AUROC is uninformative even when top-k precision is perfect —
average precision and top-k, against stated baselines, are the correct
metrics. And the **encoder is necessary**: an otherwise identical readout
conditioning on all raw dimensions (tuned ridge) fails (AP 0.10 vs 0.24),
while conditioning on the 128-dimensional learned code succeeds.

**Physical ground truth.** On a public wind-tunnel dataset built for causal
benchmarking [Gamella et al. 2025] our earlier edge-detection experiments
established the regime limits reported in the validity map (§8); the
present statistic's root-blindness prediction — exogenous actuators must
score zero incoming drive — anticipates the natural-forcing test of §7.

## 4. *C. elegans*: blind recovery of the motor-command circuit, and an interventional test

**Wild type.** Kato et al.'s whole-brain calcium imaging (five immobilised
worms, ~110–135 neurons, ~3,100 samples) [Kato et al. 2015]. Protocol fixed
before analysis: predictions were (1) ghost ≈ 0; (2) sensory neurons less
driven than command/motor neurons in the constant environment; (3) positive
correlation with anatomical in-degree from the connectome [White et al. 1986;
Cook et al. 2019]. All three passed on the three development worms, and — on
two animals held out untouched until the protocol froze — passed again with
the *strongest* anatomy correlations of the five (+0.30, +0.34). The top-10
most-driven neurons are 10/10 command/motor class in multiple worms: AVA,
AVE, RIM, AIB, RIB, RIV, RME, SMDV and ventral-cord motor neurons — the
canonical motor-command ensemble of the immobilised worm, produced by a
statistic with no access to biology.

**Intervention.** The same pipeline applied to five worms in which AVA — the
reverse-command hub, and among the top-ranked driven cells in wild type —
is silenced via a histamine-gated channel (AVA:HisCl) [Kato et al. 2015].
Pre-registered predictions passed: the silenced hub itself falls hardest
among the command core (percentile 0.92 → 0.38); its module and motor pool
collapse (AVE −0.28, RIB −0.29, VB01 −0.64, RMED −0.49, VA01 −0.45); ghosts
stay pinned in all five animals. Beyond the predictions, two observations
replicate in five of five silenced worms: **RIM and AIB retain network drive
almost unchanged** (−0.08 and −0.09 percentile) while every other module
member collapses — consistent with the strong AIB→RIM anatomical route
operating independently of AVA — and **drive reorganises onto the forward
hub AVB** (+0.34, its excess turning positive), the network's driven core
migrating to the forward-command module when the reversal hub is removed.
We report these as candidate findings pending full adjudication against the
reversal-circuit literature [e.g., Gray et al. 2005; Sordillo & Bargmann
2021]; whatever their novelty grade, an observational statistic tracked a
targeted genetic intervention neuron-by-neuron in the pre-registered
directions, which is the evidential class observational methods rarely reach.

## 5. Zebrafish: a whole-vertebrate-brain drivenness map

ZAPBench provides 71,721 segmented neurons × 7,879 timesteps of light-sheet
imaging in a larval zebrafish [Immer et al. 2025]. On a 2,241-step window
(open-loop, rotation and dark conditions), the full-brain excess map computes
in minutes on a laptop GPU. The ghost pins at −0.031 across the whole brain;
~2,800 neurons clear the excess threshold; and the driven core is a *place*:
the top-1,000 neurons are 1.78× more tightly packed than random draws (>20σ
against a spatial null), concentrated posteriorly and hard against the
midline — where the larval zebrafish's internally generated motor dynamics
(hindbrain oscillator, reticulospinal system) reside — in a test window
dominated by the dark condition, where internal dynamics prevail. Anatomical
region names await atlas registration and are not claimed; the spatial
coherence and its location are measured. Across two species, the same blind
statistic gives the same answer: in spontaneous conditions, the brain's
driven core is its motor-command system.

## 6. Clinical EEG: a confirmed effect and an instructive null

CHB-MIT scalp EEG, subject chb01: 23 channels, 256 Hz, seven annotated
seizures [Shoeb 2009; PhysioNet]. One caveat governs every output:
seizure-*onset* zones are sources, and **sources are invisible to this
statistic by design — the tool maps the spread network, never the origin.**
Pre-registered results: ghosts within tolerance in all 14 analysis windows;
drivenness **concentration** (top-4 channel share of excess) is higher
during seizures than between them (0.68 vs 0.47; five of six within-record
pairs), confirming that seizures recruit the network into a driven regime.
The third prediction failed, and the failure is the useful result: the
per-event *pattern* of driven channels does not replicate across the same
patient's seizures (mean pairwise Spearman 0.08). Whether spread genuinely
varies per event or single-window estimates are too noisy, the practical
warning is identical and, we believe, of clinical-methods relevance:
single-event connectivity maps at these window lengths should not be
trusted as patient signatures.

## 7. Climate: the tropics are the driven core, and solar forcing is invisible

Seventy-seven years of daily sea-level pressure (NCEP/NCAR Reanalysis-1,
10,512 grid cells, ~28,000 days) [Kalnay et al. 1996], deseasonalised and
differenced, with the SILSO daily sunspot number [SILSO] embedded as an
additional channel. Pre-registered gates: ghost −0.005; **sunspot channel
−0.003** — a real exogenous forcing, embedded among ten thousand
atmospheric variables, scores exactly nothing, the cleanest field test of
the statistic's designed root-blindness; and the driven core is spatially
clustered (~5σ). The exploratory map, on which we fixed no prior claim,
reproduces textbook atmospheric structure: **99% of the 500 most-driven
cells lie within 30° of the equator** (mean |lat| 6.1° vs 45.6° for the
grid) — the Walker/ENSO belt where tropical pressure is slaved to
planetary-scale circulation — while polar and mid-latitude storm-track
cells, dominated by locally generated baroclinic dynamics, rank most
autonomous. Every cell shows positive excess: one connected fluid has no
autonomous members, and the statistic behaves as the graded field physics
requires.

## 8. A measured validity map for method selection

Because the goal is a usable tool rather than a contest, we summarise where
each method earned trust in our measurements, including where ours should
not be used:

| regime | recommendation | basis |
|---|---|---|
| pairs / small systems, weak coupling | **CCM** | detects coupling 0.01 in seconds where learned methods need 0.04–0.08 |
| short real series (n ≤ 50) | CCM or PCMCI; conditional kNN for mediation checks | best average precision at 1.5–1.75× baseline on gene-circuit data; no trained estimator viable |
| noise-driven near-linear systems | **PCMCI** | best on wind-tunnel physics; CCM close behind |
| strong coupling near synchrony, mediated links | conditional learned models; *no* classical method | CCM, conditional kNN and PCMCI all acquire mediated false positives from coupling 0.3–0.5 |
| membership at V ≥ 10³ | **this statistic** | nothing classical runs; top-k precision with ghost control |
| edge-level maps on dense real circuits | *nothing we tested* | difference-based scores collapse under redundancy (companion paper) |

Absolute expectations matter as much as rankings: on real data the best
methods deliver roughly 1.5–2.4× baseline average precision — far from
clean recovery — and any tool that does not say so invites misuse.

## 9. Limitations

(1) Drivenness only; sources are invisible, and a source-detection
complement (does X's presence in the code improve *others'* excess?) is
future work. (2) Top-k semantics; weakly driven members are indistinguishable
from non-members. (3) n ≳ 1,500 samples required; this excludes most
transcriptomic time courses, which we verified rather than assumed (a
48-point circadian dataset defeats every learned arm). (4) Stationarity is
load-bearing: on undifferenced random walks the ghost control itself breaks
(shifted copies retain shared drift), which the ghost — correctly — reveals.
(5) Single recordings per condition in the biological analyses; the worm
intervention contrast is five animals per cohort but one lab's preparation.
(6) The self-baseline's function class governs transfer; degree-3
polynomials sufficed for every system here, but a misspecified self-model
resurrects confounds, and the ghost plus known-root channels are the
practical check. (7) The statistic estimates a Granger/transfer-entropy-type
quantity; our contribution is scale, controls and validation, not the
underlying causal concept.

## 10. Data and code availability

All datasets are public and unrestricted: Kato et al. worm imaging (OSF
2395t), ZAPBench (CC-BY 4.0), CHB-MIT (PhysioNet), NCEP/NCAR Reanalysis-1
(NOAA PSL), SILSO sunspot series, the wind-tunnel chamber datasets, and the
OpenWorm connectome edge list. Code, the full experiment ledger (including
negative results and voided attempts), and pre-registration documents are in
this repository.

## References

Cook, S.J. et al. (2019). Whole-animal connectomes of both C. elegans sexes. *Nature* 571.
Gamella, J.L., Peters, J., Bühlmann, P. (2025). Causal chambers as a real-world physical testbed for AI methodology. *Nat. Mach. Intell.*
Granger, C.W.J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica* 37.
Gray, J.M., Hill, J.J., Bargmann, C.I. (2005). A circuit for navigation in C. elegans. *PNAS* 102.
He, K. et al. (2022). Masked autoencoders are scalable vision learners. *CVPR*.
Immer, A. et al. (2025). ZAPBench: a benchmark for whole-brain activity prediction in zebrafish. *ICLR*.
Kalnay, E. et al. (1996). The NCEP/NCAR 40-year reanalysis project. *Bull. Am. Meteorol. Soc.* 77.
Kato, S. et al. (2015). Global brain dynamics embed the motor command sequence of C. elegans. *Cell* 163.
Runge, J. et al. (2019). Detecting and quantifying causal associations in large nonlinear time series datasets. *Sci. Adv.* 5.
Schreiber, T. (2000). Measuring information transfer. *Phys. Rev. Lett.* 85.
Shoeb, A. (2009). Application of machine learning to epileptic seizure onset detection. PhD thesis, MIT.
Sordillo, A., Bargmann, C.I. (2021). Behavioral control by depolarized and hyperpolarized states of an integrating neuron. *eLife* 10.
Sugihara, G. et al. (2012). Detecting causality in complex ecosystems. *Science* 338.
Takens, F. (1981). Detecting strange attractors in turbulence. *Lecture Notes in Mathematics* 898.
Tank, A. et al. (2021). Neural Granger causality. *IEEE TPAMI* 44.
White, J.G. et al. (1986). The structure of the nervous system of the nematode C. elegans. *Phil. Trans. R. Soc. B* 314.
