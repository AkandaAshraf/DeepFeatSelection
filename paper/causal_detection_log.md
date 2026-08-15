# Causal detection on dynamical systems: results log

Status ledger for the August 2026 arc that moved the project from i.i.d. tabular
data onto trajectories. Every entry is marked **VERIFIED** (survived an
adversarial re-run or direct re-derivation), **PROVISIONAL** (observed once,
no verification pass yet), or **VOID** (killed under verification, kept here so
the failure mode is not rediscovered). Scripts named per entry; numbers are in
`ExpOutput/`.

The framing question, sharpened over the arc: *compressibility
is the causal signature* (algorithmic Markov condition; on trajectories, Takens'
imprint / generalised synchrony). The open question was never whether the
principle holds — it is whether a **learned** estimator of it ever beats the
classical local-geometry estimators (kNN, simplex projection / CCM).

---

## The estimator ranking on pairs — VERIFIED

`scripts/masked_recreation.py`, `ExpOutput/masked_ae/`.

Masked-channel recreation (zero one channel of the joint delay embedding, ask a
small autoencoder's decoder to recreate it) confirms the imprint principle as a
graded signal: separation from control at coupling ≥ 0.04 on logistic pairs,
monotone 0.03 → 0.75 across coupling 0.04 → 0.32, +0.50 on Rössler–Lorenz.
Controls clean (white-noise pair −0.03..−0.01, full-reconstruction guard ≥ 0.993,
c=0 byte-identity asserted).

But on the same data and splits the ranking is **CCM > kNN > neural**, at every
coupling: CCM separates from c=0.01 (ρ 0.55–0.60), kNN from ~0.02, the AE only
from 0.04 and never catches kNN (0.75 vs 0.91 at c=0.32). Loss dilution ruled
out: a direct MLP trained only on the masked channel is no better (fails at
c=0.04 where kNN reads +0.46). At weak coupling the imprint is a local geometric
feature; neighbour methods read it off the manifold, small trained maps do not.

Same verdict for forecast-side readouts (`scripts/sequence_causal.py`,
`scripts/sequence_arch.py`): CCM detects coupling 0.01 in ~2 s where the
BiLSTM deprivation score needs ~0.08 and ~100 s under the verified protocol.
Attention weights sit at uniform (0.497–0.503) at every coupling — a softmax
under no pressure to leave uniform selects nothing. `mha_attn` fails outright
(control_max +5.2). Capacity does not rescue deprivation.

Caveat that stands: `sequence_arch.py` ran at `r_y=3.5` (generator default),
not the protocol's 3.7, so its numbers must not be compared against
`sequence_causal.py`'s. Within its own protocol, deprivation separated at
every coupling including 0.01 — which **refutes** the earlier "dynamic range"
argument that a near-perfect forecaster (base MSE 0.0003) kills difference-of-
losses scoring. That proposed proposition is dead; do not resurrect it.

## The chain at strong coupling — PROVISIONAL, the first learned-estimator win

`scripts/bottleneck_network.py --coupling {0.3,0.5,0.7,0.9}`,
`ExpOutput/synchrony_c*/`. Three seeds, chain A→B→C plus matched independent
triple; F1 over directed edge calls, null thresholds calibrated from the
independent triple.

The predicted pairwise-CCM false positive on the indirect A→C link finally
appears, and the methods break in theory-order:

| coupling | CCM E=3 | CCM opt-E | conditional kNN | conditional MLP |
|---|---|---|---|---|
| 0.3 | ok | ok | ok | ok |
| 0.5 | fooled | ok | fooled | ok |
| 0.7 | fooled | fooled | fooled | ok |
| 0.9 | fooled | fooled | fooled | ok |

ρ on the non-existent A→C: 0.22 → 0.39 → 0.56 → 0.67. Every method keeps
TP = 2/2 on the true edges at every coupling, so the MLP's clean sheet is not
conservatism: **conditional MLP retrieval is the only method with F1 = 1.000
across the entire non-divergent coupling range.** Its conditional gain reads
"A adds nothing beyond B" correctly even as B tends toward a function of A —
the dynamical analogue of the Prop-1 redundancy regime, which is where kNN's
local geometry around the mediator degenerates.

Not yet verified. The specific attack a verifier must run: is kNN's failure at
0.5+ a property of local geometry or of its null-threshold calibration? If the
latter, the MLP's edge over kNN shrinks (its edge over CCM would stand). Also
only one system family and 3 seeds.

## The 10-node network — PROVISIONAL: the ranking splits by metric

`scripts/network_scale.py --coupling 0.3`, `ExpOutput/network_scale/`. Ten
logistic maps, 12-edge random DAG, 2 seeds, matched zero-edge null network.

| method | AUROC(all) | AUROC(vs 2-hop) | null floor | edges above floor |
|---|---|---|---|---|
| kNN LOCO | 0.694 | 0.703 | +0.321 | 4% |
| MLP LOCO | 0.792 | 0.779 | **+0.039** | **62%** |
| pairwise CCM | **0.985** | **0.970** | +0.934 | 25% |

Three separate findings in one table:

1. **CCM remains the best ranker at coupling 0.3** — the 2-hop tax did not
   bite (0.970), consistent with the synchrony sweep where CCM is only fooled
   from 0.5. But its null floor is +0.934: on a network with NO edges, some
   independent pair cross-maps at rho 0.93. Likely the periodic-window
   degeneracy the `coupled_logistic` docstring warns about — r drawn from
   [3.6, 3.8] can land nodes in periodic windows, and quasi-periodic pairs
   cross-map spuriously. Consequence: CCM's scores cannot be thresholded
   against a matched null; a calibrated deployment detects only 25% of edges.
2. **The MLP conditional gain is the only calibratable detector**: null floor
   +0.039 (a difference statistic cancels shared degeneracy that absolute rho
   cannot), 62% of true edges above it.
3. **kNN starves in 18-dimensional conditioning exactly as predicted**: 4%
   above floor, full-recreation r2 0.047 vs the MLP's 0.146 on the harder
   seed. The learned estimator is now the only viable conditional method at
   this width.

Unverified; 2 seeds; the CCM-floor explanation (periodic windows) is a
hypothesis, checkable by inspecting the null network's series. Open cells:
coupling 0.5+ (where the synchrony sweep says CCM's ranking should also
degrade), and r draws restricted to verified-chaotic values so the null floor
comparison is fair to CCM.

## Voided results — do not rediscover these

- **Bottleneck dimension deficit** (`scripts/bottleneck_pair.py`): the
  "detection at c=0.01" was `optimal_embedding_dimension` assigning coupled
  arms E_y=2 vs the control's 1. The AE knee never moved in any arm; the width
  sweep measured the embedding selector. Salvage, real and cheap: **the
  response's optimal embedding dimension rises under coupling** — detected at
  c=0.01 by simplex self-prediction alone. Untested as a formal detector.
- **Dynamic-range "Proposition 3"** — refuted by `sequence_arch` (above).
- **Peakedness as understanding** — survives only gated: peakedness measures
  commitment, not correctness (gini 0.71 on a failed model vs 0.74 understood;
  on-target share 5% vs 83%). The regime gate (memorisation gap) must come
  first. Kurtosis (k comparable peaks vs one spike) is the surviving
  hypothesis, n=6.
- **Hopfield, all three angles**: dot-product retrieval diverges from the
  nearest state on an attractor as β grows (ρ → 0.11; the distance kernel
  converges to 1-NN as required, ρ → 0.985); the retrieval "win" over simplex
  equals plain 8-neighbour averaging (0.823 vs 0.826); energy does not predict
  cross-map error (Spearman ≤ 0.04, worse than nearest-neighbour distance at
  0.12). One real by-product: **CCM's k = E+1 truncation is suboptimal** —
  8-neighbour mean lifted convergence 0.500 → 1.000 on coupled logistic.
  Unreplicated beyond that system.

## First external test — the wind-tunnel chamber: the win did not transfer

`scripts/chamber_detect.py`, `ExpOutput/chamber/` (`evaluation_corrected.csv`
is authoritative — the first scoring read the LOCO matrices transposed
relative to the pre-registered orientation; correcting to the declared
convention moved MLP 0.545→0.572 pooled and slightly worsened its false
positives, so the bug was real but hid nothing). Real physics, 16 varying
variables, 26 ground-truth edges, three experiments (1k, 10k, 10k rows).

| method | AUROC (pooled) | edges above floor | actuator false positives |
|---|---|---|---|
| CCM | **0.675** | — | **0** |
| PCMCI | 0.636 | — | **0** |
| MLP-LOCO | 0.572 | 6% | 5.0 per experiment |
| kNN-LOCO | 0.510 | 12% | 6.3 per experiment |

The pre-registered failure clause applies: **the August learned-estimator win
was a property of our simulator family, not of the method.** The wind tunnel
is a stochastically driven system — actuators are random walks, sensors are
physics plus noise — and Takens' imprint, the rationale under both CCM and
recreation-LOCO, assumes deterministic dynamics. PCMCI was built for the
stochastic regime; CCM's local geometry still coped (near-deterministic
actuator→sensor responses); recreation-LOCO neither ranked well nor
calibrated: the ghost floor under-estimated the real noise floor and let
through edges pointing INTO exogenous actuators (up to 16 in one experiment),
which are false by construction. CCM and PCMCI produced zero such edges.

Validity-map rows this buys, stated as tool advice:
- Stochastically driven / noisy near-linear system → **PCMCI or CCM**; do not
  use recreation-LOCO there, and treat any method's edge into a known
  exogenous variable as a red flag anyone can check.
- The ghost-source calibration does NOT transfer as-is to stochastic real
  data; on such data it must be validated against known-exogenous nodes
  before its floor is trusted.
- The learned estimator's demonstrated niche (strong coupling, wide networks)
  remains simulator-only until tested on a real system with genuinely
  deterministic nonlinear dynamics — which is what Tier 2 (circadian
  oscillators) is for.

## The maturity hypothesis — mechanism CONFIRMED on synthetic, blocked on the chamber

`scripts/maturity_synthetic.py` + `scripts/maturity_sweep.py`,
`ExpOutput/maturity_synth/` + `ExpOutput/maturity/`. The hypothesis:
a mature model extracts a cause's imprint from ANY of its children (Takens),
so LOCO gains collapse exactly where the imprint is redundant — the method
worked in August because of limited training. The design: one
model per fit, checkpointed at epochs {2,5,10,25,50} of a single trajectory.

**Phase A (hub-shaped synthetic, truth free): the mechanism test passed
cleanly.** Hub-parent gains (6 children) rise, peak at 10 epochs (0.026),
then collapse to zero at 50 (−0.002), while single-child gains grow
monotonically the whole way (0.022 → 0.281) and never collapse. Uniform
overfitting cannot produce that crossing; redundancy-ordered collapse is
exactly what Takens implies. This is Propositions 1 and 2 operating in
training time. (2 seeds; aggregate AUROC non-monotone in the mean, peak 10
epochs at 0.685, but seed-noisy because the two strata pull opposite ways.)

**Phase B (chamber walk_2, blind): the pattern did not transfer, for two
identified reasons, both new validity-map rows.**
1. The control stratum is unmeasurable on this data: recreation of the
   single-child parents (pot_1, pot_2, hatch) has base r2 = 0 at every
   maturity, so all their LOCO gains are exactly 0. The mechanism test needs
   a measurable unique-imprint stratum and the chamber walks do not provide
   one.
2. The ghost floor RISES with maturity (0.048 → 0.352) instead of falling:
   actuators are nonstationary random walks, and a circular shift of a
   random walk still shares low-frequency drift with the original, so a
   mature model extracts real correlation from the "null" channel. The
   ghost calibration is only valid for stationary series — on drifting data
   it must be preceded by differencing/detrending (as the climate protocol
   already required) or replaced by a stationarity-respecting surrogate.
   Corollary observed: actuator false positives fall with maturity here
   (15 → 0) purely because the inflated floor suppresses everything.

No stopping rule validated on real data: the gradient-steps rule from Phase A
(~140 steps → chamber epoch ~1) predicted epoch 2, whose AUROC (0.588) was
second to epoch 25 (0.625) on a noisy single-seed curve. Status: maturity
mechanism PROVISIONAL-confirmed (synthetic, stratified, 2 seeds);
maturity as a deployable tool feature NOT yet demonstrated on real data.

## Tier 2 — the circadian clock: the validity map's prediction held exactly

`scripts/circadian_detect.py`, `ExpOutput/circadian/`. GSE11923, 10 core
clock genes, 48 hourly points, 19 TTFL edges from knockout genetics; all
choices pre-declared (E=3, tau=3h, first-differenced, probes averaged).

| method | AUROC | PR-AUC (baseline 0.211) |
|---|---|---|
| CCM | 0.644 | **0.321** |
| kNN-LOCO | 0.628 | **0.369** — best method |
| PCMCI | 0.544 | **0.315** |
| MLP-LOCO, best of 5 checkpoints | 0.492 | 0.220 |

METRIC CORRECTION (2026-08-10): with 19 positives in 90
candidates, AUROC flatters recall over precision. Under average precision
CCM and PCMCI TIE (0.321 vs 0.315) -- the "regime reversal" row previously
written here was an AUROC artifact and is withdrawn. Both classical methods
deliver ~1.5x the random baseline on real transcript data; the MLP's best
checkpoint sits at the baseline.

The pre-registered expectation transferred to real biology without
adjustment: at n=48 the classical arms carry the detection and the MLP is at
or below chance at EVERY maturity checkpoint — the window cannot open when
there is nothing to train on (Arntl hub gains: negative when immature,
exactly zero when mature; starvation, not mechanism). Two genuine findings:

1. **CCM and PCMCI are comparable on real data once the metric respects
   class imbalance** (chamber AP: PCMCI 0.265 vs CCM 0.256 on baseline
   0.108; circadian AP: 0.321 vs 0.315 on 0.211). No regime separation
   between them is demonstrated at these data sizes.
2. **kNN-LOCO is the best circadian method under average precision** (0.369
   vs CCM 0.321, PCMCI 0.315; 1.75x baseline) — the conditional structure,
   with a local estimator, wins the metric practitioners act on. The learned
   estimator does not travel to this data size, as the map predicted.
3. **Absolute performance on real data is modest for every method**:
   1.5-2.4x the random baseline. The tool must state this calibrated
   expectation up front.

Tool advice, as the map now reads: transcript-level time courses at n<=50 ->
conditional-kNN first (best precision AND the mediation check), CCM and
PCMCI beside it; no trained estimators; expect 1.5-1.75x baseline
precision, not clean recovery.

## The MLP's niche, tested from all sides (2026-08-10)

Three rapid cells sharpened where the learned estimator does and does not
belong; the project rule governs: a real use case, not "works better".

1. **Differenced chamber maturity sweep — dead substrate confirmed.** On
   differences the ghost floor finally behaves (falls monotonically, 0.002 →
   0.000) but recreation r2 collapses to 0.000 and AUROC to exactly 0.500:
   chamber walk increments carry nothing recreatable. Levels break the null,
   differences remove the signal. Recreation-LOCO has NO operating point on
   wt_walks, at any maturity. The maturity window remains synthetic-only.
2. **PCMCI on the wide synthetic network — my "linearity wall" prediction was
   WRONG.** PR-AUC 1.000, AUROC 1.000, both seeds (CCM 0.958, MLP 0.659).
   This map family's coupling is linear in the parent given the target's own
   lags, which ParCorr-with-conditioning reads perfectly. The "wide" pillar
   of the MLP's niche is gone on this family.
3. **PCMCI on the synchrony chain — the MLP's unique win SURVIVES its last
   challenger.** PCMCI's A→C false-positive rate: 33% at coupling 0.3, 67%
   at 0.5, 100% at 0.7 and 0.9 (p ~ 0.0000), true-edge recall 100%
   throughout. Near synchrony B is nearly collinear with A and linear
   conditioning cannot screen it. Combined chain scoreboard: CCM fooled from
   0.5, conditional kNN fooled from 0.5, PCMCI fooled from 0.3-0.7, the
   conditional MLP alone clean through 0.9 (still PROVISIONAL, one family,
   3 seeds).

The niche, final form: **mediation rejection in strongly coupled systems near
synchrony** — where collinearity defeats linear conditioning, locality
defeats kNN, and pairwise-ness defeats CCM. Real systems in that regime:
synchronised neural circuits, power grids, cardiac dynamics, coupled climate
oscillators. This is why the C. elegans whole-brain test (dense mediation,
strong coupling, anatomical ground truth, open data) is the decisive
application test: if the conditional MLP finds nothing there that the
classical trio misses, the tool ships classical and the MLP's contribution
is the mechanism understanding.

## Tier 2b — C. elegans: the application verdict, replicated 3/3

`scripts/celegans_detect.py`, `ExpOutput/celegans/`. Kato 2015 WT_NoStim,
worms 0-2, 25 identified neurons each against the anatomical connectome,
per the pre-registered protocol. Mean PR-AUC (baselines 0.19-0.26):

| method | worm 0 | worm 1 | worm 2 | ~lift |
|---|---|---|---|---|
| CCM | 0.435 | 0.386 | 0.486 | 2.0x |
| PCMCI | 0.343 | 0.358 | 0.400 | 1.7x |
| kNN-LOCO | 0.280 | 0.248 | 0.302 | 1.3x |
| MLP best checkpoint | 0.237 | 0.266 | 0.283 | 1.2x |
| MLP mature (e50) | =baseline | =baseline | =baseline | none |

Two findings, both clean:

1. **The niche did not survive real neural circuits.** The conditional MLP's
   mediation-rejection capability, real on synthetic chains, contributes
   nothing here: its best checkpoint trails every classical method on all
   three worms, so its "rejections" of mediated pairs are noise and the
   named-pairs deliverable is EMPTY by the project's own standard — no
   defensible list is presented.
2. **The maturity collapse replicated perfectly on real data**: at epoch 50
   every LOCO gain is exactly zero (AUROC 0.500 to the third decimal, PR-AUC
   pinned at baseline) on ALL THREE worms — dense connectome redundancy is
   total, the mature model routes around any single neuron, and the immature
   side never clears the noise floor first. The window closes from both ends
   on dense real circuits, exactly as the mechanism predicts. The theory
   transferred; the method did not.

FINAL FORM OF THE TOOL, per the project rule (benefit, not victory): CCM-led
detection (the workhorse on every real dataset: 0.256-0.486 AP at 1.5-2.2x
baseline), PCMCI beside it, conditional-kNN for mediation checks at small
scale, PR-AUC-with-baseline reporting, stationarity preprocessing, ghost
calibration with its validity limits stated, and the validity map itself.
The learned estimator's lasting contribution is the MATURITY MECHANISM — why
difference-based importance fails on redundant systems, confirmed synthetic
and now observed at full strength on real neural data — plus the negative
results that keep the next group from spending a month rediscovering them.

## The scale reopening — membership by masked recreation (2026-08-10/11)

`scripts/bottleneck_membership.py`, `ExpOutput/membership*/`. An
audit found the "bottleneck" had never compressed anything (hidden layers up
to 4x WIDER than input everywhere), so the compressibility hypothesis was
retested in its intended form: one masked autoencoder over the whole system,
a hard bottleneck, and an ABSOLUTE membership readout -- held-out r2 of
recreating a masked variable from the code (1 - MSE/Var, clamped) -- scored
by PR-AUC against constructed membership truth.

V=30 (10-member web among autonomous loners): works, weakly. AP 0.37 /
AUROC 0.65 on baseline 0.36, stable across compression 1.5x-46x (the width
axis is INERT), across zero vs uniform-noise fills (10-draw averaged), and
across 5-100 epochs -- the absolute readout does NOT suffer the maturity
collapse, as predicted. MAE-style masked-only loss widens the member-loner
mean gap (0.03 -> 0.06) but not the ranking. Full CCM still wins where
affordable (AP 0.53) with its own ghost pathology (0.31).

V=1000 (100 members among 900 loners): CHANCE under every configuration --
bottleneck 8/32/128/256, both fills, both losses (12 cells): AP 0.09-0.17 on
baseline 0.10, AUROC 0.46-0.52, and the ghost control at 0.64-0.75 where it
sat at 0.00 for every V=30 run. Mechanism, visible in the class means
(loners 0.30-0.33 recreate BETTER than members 0.20-0.27): with hundreds of
same-family autonomous series, the decoder's optimal move for any masked
channel is the FAMILY PRIOR -- a typical clean trajectory -- which loners fit
and coupling-perturbed members do not. At scale on homogeneous pools, masked
recreation measures TYPICALITY, not coupling. The inversion is noted as an
observation only (no post-hoc sign flipping; at |AUROC-0.5| < 0.05 there is
nothing to flip to anyway).

Classical methods at this scale, measured: full pairwise CCM 528 ms/pair ->
~14,700 h for one V=10,000 scan; CCM-vs-PCA broken at every V tested (AUROC
0.26-0.57, ghost 1.00). So NOTHING currently works at V=1000 on this
generator -- the scale niche is empty for everyone, not just for the AE.

What survives: the instrument. The absolute readout is maturity-stable, the
fill choice is innocuous (test-time-only evaluation validated), weights now
persist per checkpoint so readouts are evaluations not retrains, and the
ghost control caught the typicality artifact exactly as designed -- on real
data with no ground truth it would be the only warning available. One door
deliberately left unopened: heterogeneous real pools (e.g. the 45k-probe
circadian matrix) have no clean family for the prior to collapse onto, so
the artifact may not transfer -- untested, and recorded as such rather than
as hope.

## The ensemble at V=1000 — LLN works, and converges to the bias (2026-08-11)

`scripts/ensemble_membership.py`, `ExpOutput/ensemble/`. The original
design: 8 independently initialised masked AEs on ONE system, brute-force
analysis across minima. Every prediction got a clean answer.

1. "A single model reaches an unknown minimum" — CONFIRMED: single-model AP
   ranges 0.09-0.16 across the 8 minima on identical data.
2. "Law of large numbers" — CONFIRMED AS MECHANISM, REFUTED AS RESCUE: the
   consensus-AP prefix curve rises exactly as LLN predicts (0.111 → 0.148,
   AUROC 0.506 → 0.529) but decelerates to an asymptote of ~0.15 AP (~1.6x
   baseline). Averaging removes optimisation variance; what remains is the
   SHARED bias: the ghost scores 0.740 with cross-model std 0.0043 — near
   identical in every minimum. The artifact is a property of data+objective,
   not initialisation, and no ensemble size touches it.
3. The mechanism, nailed by the periodicity probe: at scale the masked
   readout is a PREDICTABILITY meter, not a coupling meter. Periodic loners
   score 0.758, the ghost 0.740, chaotic loners 0.288, members 0.250 —
   the ranking is complexity-inverted, and causal members (chaotic AND
   coupling-perturbed) sit at the bottom. The V=1000 failure is now one
   sentence: recreation r2 conflates "predictable from others" with
   "predictable at all".

The identified fix, for whoever continues: score membership as the EXCESS of
cross-recreation over self-predictability — r2(X | others) minus r2(X | own
history or an AR/complexity baseline) — which is what CCM's convergence test
implicitly does and our absolute readout omits. Untested here; recorded as
the concrete next design rather than run, since each of the last three
"one more fix" cycles at V=1000 failed and the pattern deserves respect.

## First entry into the V=1000 regime — excess-over-self (2026-08-14)

`scripts/excess_membership.py`, `ExpOutput/excess_poly/`. The diagnosis chain
paid off on its second implementation attempt. The statistic:

    excess(X) = r2(x_{t+1} | own lags, poly + encoder code)
              - r2(x_{t+1} | own lags, poly)

computed as ridge readouts on the SAVED ensemble encoders (no training). The
first attempt used a LINEAR self-baseline and failed exactly as a linear fit
of a quadratic map must (self r2 0.62; clock confound resurrected through the
weak baseline). With quadratic own-features the self-baseline matches theory
to three decimals -- loners 1.000, ghost 1.000, members 0.978 -- and:

- **precision@10 = 10/10, precision@20 = 16/20** at V=1001 (94 members,
  baseline 0.094). The strongly-driven core of a thousand-variable system,
  found in minutes where a CCM scan needs ~14,700 hours.
- **Zero false alarms among 907 non-members**: loners, periodic channels and
  the ghost all pinned at 0.0000. Both confounds (typicality, clock) are
  provably cancelled, not suppressed.
- AUROC is 0.28 and MISLEADING here: weakly-driven members (drive share
  below readout noise, ~60 of 75) get tiny negative excess and rank below
  the zero-pinned loners. The statistic is a high-precision top-k detector,
  not a full-population ranker. PR-thinking, adopted into the
  protocol, is what makes this legible at all.
- **Directional semantics discovered in the mechanism check**: root members
  (in-degree 0, pure causes) score exactly 0 -- the statistic detects "X is
  DRIVEN by the system", the Takens direction, and sources are invisible by
  construction, exactly as the chamber's exogenous actuators were. A
  source-detection complement is future work.
- **Deep finally earned its keep, measured**: the raw-ridge control (own
  poly + all 3,000 raw dims, tuned alpha) fails outright (AP 0.10). The
  128-dim encoder code is what makes the system legible to a small readout.
  Component necessity is now itemised: encoder necessary, quadratic
  self-baseline necessary, ensemble consensus helpful-not-critical (singles
  0.23-0.28, consensus 0.24), controls load-bearing.

REPLICATION AND TRANSFER CELLS (2026-08-14, `ExpOutput/excess_s1`,
`excess_s2`, `excess_het`, `excess_het_p3`): the result holds everywhere
tested. Seeds 1 and 2 (fresh graphs, fresh ensembles): precision@10 = 10/10
and precision@20 = 20/20 on BOTH, AP 0.32/0.37, self-baselines matching
theory again (loners 1.000, members 0.977). Heterogeneous pool (loners split
across logistic, sine, tent and AR(1) families -- no family prior to
collapse onto): precision@10 = 10/10, @20 = 18/20 at both poly degrees, all
loner families pinned at ~0 excess including the deliberately stochastic
AR(1) quarter. The transfer-critical robustness property demonstrated: at
poly-3 the loner self-baseline is imperfect (0.864, the AR quarter's
irreducible noise) yet loner excess STAYS at zero -- an imperfect self-model
does not manufacture false positives, because the code must actually carry
the missing information and for autonomous channels it does not. The old
masked-recreation consensus on the same systems: 5-8/10 at the top --
the readout, not the encoder, was always the difference.

Boundaries that remain: one generator FAMILY (all synthetic maps), and the
real-data self-baseline question (no exact function class exists there). Real data has no known exact self-function-class -- the quadratic trick
must become a flexible nonlinear self-baseline (GBM/kNN/higher poly), and
whether the excess survives that substitution is THE transfer question.
Replication across seeds and the heterogeneous-pool test are the immediate
next cells.

## The worm cell — real-data validation of the excess detector (2026-08-14)

`scripts/celegans_excess.py`, `ExpOutput/celegans_excess/`. Three worms of
Kato 2015 WT_NoStim, ALL recorded neurons (109-135 each), fresh 8-encoder
ensembles, poly-3 flexible self-baseline. All three pre-registered
predictions passed on all three worms:

1. **Ghost pinned at zero on real calcium dynamics** (+0.004, -0.011,
   -0.041): the refuse-false-positives asymmetry transferred off synthetic
   maps, even with self-baselines as weak as 0.18 (differenced calcium is
   mostly noise-like to a self-model) -- low self-predictability did NOT
   manufacture false positives, exactly as the hetero cell promised.
2. **Sensory < command/motor in the constant environment, 3/3 worms**
   (+0.006 vs +0.025; -0.018 vs +0.003; -0.002 vs +0.015). Sensory n is
   small (8/6/1) so per-worm evidence is weak, but the direction is uniform
   and mechanistically predicted: silent environment, silent drivers.
3. **Positive correlation with anatomical in-weight, 3/3** (+0.14, +0.23,
   +0.23), modest as pre-registered (most parents unrecorded).

THE NAMED DELIVERABLE, at last: the top-10 by excess is 10/10 command/motor
class on worms 1 and 2 -- AVAL/AVAR, AVEL/AVER, RIML/RIMR, AIBL/AIBR, RIB,
RIV, RME, SMDV, VB/DB -- which IS the motor-command circuit Kato et al.
identified as the backbone of the immobilised worm's global dynamics. A
statistic that knows nothing of biology, fed raw traces, independently
surfaced the canonical driven core of the worm brain, reversal-command
neurons first. This is the tool doing what it exists
for: telling a researcher something checkable, named, and true.

**HELD-OUT REPLICATION (worms 3-4, untouched by any design decision since
the Tier-2b protocol declared them reserved): ALL THREE PREDICTIONS PASSED
ON BOTH.** Ghost -0.030/-0.022; sensory < command/motor on both (0.004 vs
0.057; -0.007 vs 0.039); and the anatomical in-weight correlations are the
STRONGEST of all five worms (+0.30, +0.34 vs +0.14/+0.23/+0.23 on the
development set) -- the effect grew on data the method had never seen, the
opposite of what overfitting produces. Five of five worms; AVAL and the
command core top the held-out rankings as well. Status upgraded: the excess
detector is REPLICATED ON HELD-OUT REAL DATA.

Boundaries: drivenness direction only (sources invisible by design);
absolute excess is small (0.02-0.11); classes hand-curated.

## The fish — a whole-vertebrate-brain drivenness map (2026-08-15)

`scripts/zapbench_feasibility.py`, `ExpOutput/zapbench_full/`. ZAPBench
slice (conditions 7-9, n=2241), ALL 71,721 neurons, 4-encoder ensemble
trained on the RTX 3070 (57 s/model at 64 units after two OOM lessons:
allow-growth + slim model + chunked encoding + host-RAM diet), full-brain
excess readout in ~10 min. The V=100k regime targeted at the start of
the scale arc is now measured practice on a laptop, where a pairwise CCM
scan would cost ~2 years.

- **Ghost pinned at -0.031 across the whole brain** -- the run is valid by
  its own built-in control.
- 3.9% of neurons (~2,800) clear +0.01; top tail to +0.164 -- the same
  magnitude as the worm's command core.
- **The driven core is anatomically clustered, not scattered**: top-1000
  median nearest-neighbour distance 16.8 vs 29.8 +/- 0.6 for random draws
  (1.78x tighter, >20 sigma). It sits POSTERIOR (mean x 1204 vs brain 1059)
  and strongly MIDLINE-CONCENTRATED (y spread 133 vs 207), mid-depth --
  the hindbrain/midline signature where the larval zebrafish's
  internally-generated motor dynamics (hindbrain oscillator,
  reticulospinal system) live, and the test window is predominantly the
  dark condition, where internally-generated dynamics dominate. Region
  NAMES await atlas registration and are not claimed; the spatial
  coherence and its location are measured.

Two species, one statistic, the same answer: the brain's driven core is the
motor/command system, found blind both times. Remaining fish work: atlas
registration for named regions, and the dark-vs-stimulus condition contrast
from the saved encoders.

## The intervention test — AVA silencing, tracked neuron by neuron (2026-08-15)

`ExpOutput/celegans_excess_avahiscl/`, same pipeline, five AVA-HisCl worms
(Kato 2015's histamine-silenced line), compared to the banked five-WT
percentile profiles. Ghost pinned in all five (-0.008..-0.018).

**Pre-registered predictions: passed.** The silenced hub itself falls
hardest among the command core (AVA 0.92 -> 0.38 percentile) -- the internal
silencing control and the statistic agreeing. The reverse module and its
motor pool collapse as predicted: AVE -0.28, RIB -0.29, and the downstream
motor neurons VB01 -0.64, RMED -0.49, VA01 -0.45, RME/RMEV ~ -0.26. The
in-weight correlation collapses cohort-wide (+0.14..+0.34 -> ~0) and the
sensory-vs-command contrast scrambles -- both coherent consequences of
removing the hub the drive flowed through, observed rather than assumed.

**The two named findings from the discovery scan, replicated 5/5 in both
cohorts:**
1. **RIM's network-drivenness is largely AVA-independent**: 0.93 -> 0.84
   (-0.08) while its module partners drop -0.28..-0.64; RIM remains the most
   driven named cell in the silenced brains, and AIB holds beside it
   (-0.09). The connectome offers the candidate route: the strong AIB->RIM
   synapses. Claim: the AIB-RIM axis retains internal drive without AVA.
2. **Drive reorganises onto the forward hub**: AVB rises 0.49 -> 0.82
   (+0.34), its excess turning positive, with ALA rising beside it (+0.30).
   With the reversal hub silenced, the brain's driven core shifts to the
   forward-command module -- circuit-level reorganisation, quantified
   per neuron by an observational statistic under a genetic intervention.

Status: intervention-validated observational tracking is now demonstrated
(the statistic moved with the do()-operation, in the predicted directions,
with named cells). The two discovery claims are NOVEL-CANDIDATE pending a
literature pass (RIM/AIB reversal sub-circuit work, e.g. Sordillo &
Bargmann, must be checked before the word "novel" is printed anywhere).

## Clinical EEG deployment — concentration confirmed, reproducibility null (2026-08-15)

`scripts/eeg_excess.py`, `ExpOutput/eeg_excess/`. CHB-MIT chb01, six ictal
windows (annotated seizures) and eight interictal baselines, under the
standing caveat printed on every output: THIS MAPS SPREAD, NEVER ORIGIN.

1. Ghost: |excess| <= 0.034 across 14 windows, most < 0.01 -- acceptable at
   these short windows (n~2000), noted as noisier than the large-n runs.
2. **Concentration: CONFIRMED.** Ictal top-4 share 0.679 vs interictal
   0.470 (gini 0.705 vs 0.500); 5 of 6 within-record pairs in the predicted
   direction, the exception (record 03) reported, not excused. Seizures
   recruit the network into a concentrated driven regime, measurably.
3. **Cross-seizure reproducibility: NULL.** Mean pairwise Spearman of the
   ictal channel pattern is 0.081 over 15 pairs -- the per-event drivenness
   maps DO NOT replicate across chb01's seizures at this window length.
   Either spread genuinely varies per event or single-window estimates are
   too noisy for stable channel ranking; both readings carry the same
   practical warning, which is itself the clinically relevant finding:
   SINGLE-EVENT SPREAD MAPS FROM THIS FAMILY OF METHODS SHOULD NOT BE
   TRUSTED. The mean-ranked top channels lean left-temporal (F7-T7 first),
   adjacent to chb01's documented focus, but with event-level Spearman at
   0.08 this is suggestive only and is not claimed.

## Climate deployment — all gates passed; the tropics are the driven core (2026-08-15)

`scripts/climate_excess.py`, `ExpOutput/climate_excess/`. 77 years of daily
NCEP sea-level pressure, 10,512 cells, 4-encoder GPU ensemble (80 s/model
once the tf.data pipeline replaced the generator; hard 6 GB GPU cap held).

1. Ghost: -0.005 -- pinned.
2. **SUNSPOT: -0.003 -- root-invisibility passed on real exogenous
   forcing.** The daily solar series, embedded among 10,512 pressure cells,
   scored exactly nothing: the first natural-forcing test of the property
   the chamber actuators and silenced AVA only approximated.
3. Clustering: top-500 median NN 0.0435 vs 0.0676 +/- 0.0048 random (~5
   sigma) -- the driven core is a place.
And the place is the DEEP TROPICS: 99% of the top-500 cells lie within 30
degrees of the equator (mean |lat| 6.1 vs 45.6 for the grid), while the
least-driven cells are polar and mid-latitude storm-track regions. Blind,
this reproduces the textbook structure of atmospheric dynamics: tropical SLP
is slaved to planetary-scale circulation (the Walker/ENSO belt -- where the
Southern Oscillation Index itself is measured), while mid-latitude weather
is dominated by locally generated baroclinic chaos. Every cell shows
positive excess (frac>0.01 = 1.000): one connected fluid has no loners, and
the statistic behaves as a graded field exactly as it should.

Both cross-domain deployments (EEG, climate) completed same-day on open
data with pre-registered gates -- the open-science mode of working.

## Standing protocol (violations of each cost a result this arc)

1. Coupling sweeps run at `r_x=3.8, r_y=3.7`; c=0 must be byte-identical to
   `independent_logistic` and asserted, not trusted.
2. Contiguous three-segment splits with an embedding-span embargo at each seam;
   train-stats-only standardisation; PCA fit on train only.
3. Statistics, thresholds, and orientations fixed a priori in the docstring.
   `max(a, 1−a)` inflated a shuffled-label null to 0.804 once.
4. Every detection number sits beside its model's own learnability guard.
5. Detection = weakest coupled arm exceeds strongest control arm; overlapping
   ranges are a null.
6. Smoke-test the mechanics before the full run (caught: attention pooling that
   could not forecast, mean-pooling that destroyed mha, a gate threshold that
   could not reach zero, a null floor of +1.20 from differencing negative r2).
7. Clamp recreation r2 at zero before differencing.
8. Fixed embedding dimension across arms of a comparison — never let a selector
   vary D between coupled and control.
