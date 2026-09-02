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

## The intervention test — AVA silencing (2026-08-15)

> **PARTLY SUPERSEDED.** The per-cell resolution claimed in this entry did
> not survive the nulls added later the same day: only VB01 clears a
> max-statistic correction across the ten contrasted cells, AVA reaches
> p = 0.135, and the AVB rise sits inside the wild-type-only noise range.
> See "Second adversarial pass" and "AVB and the Kato passage" below.
> The cohort-level effect stands.

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

## Recall characterisation, and a flaw in the ghost donor (2026-08-15)

> **PARTLY SUPERSEDED.** This entry pools four systems. The heterogeneous
> pool reuses seed 0's coupled web byte-for-byte, so the distinct totals
> are 219 driven / 63 source / 18 isolated over three graphs, not
> 294 / 82 / 24. See "Second adversarial pass" below.

`scripts/recall_analysis.py`, `scripts/recall_probe.py`,
`scripts/ghost_calibration.py`, `scripts/ghost_tail.py`,
`scripts/ghost_corrected.py`, `ExpOutput/recall/`. All readouts on the
archived encoders; no retraining. The recomputed ghost matches the archived
value to delta 0.00e+00 on all four systems, which validates the shortcut.

**VERIFIED — the truth label was never a drivenness label.** `build_system`
sets `truth[:members] = deg > 0` with `deg` counting in- AND out-edges, so a
pure source is labelled positive while receiving no drive. 82 of 400
first-block channels across the four systems are such sources. This alone
explains a large part of the below-chance AUROC of 0.28: the label disagrees
with the estimand for roughly a fifth of its positives. Recall is now
reported against the driven subset and the roles are reported separately.

**VERIFIED — a single ghost is not a threshold, and its donor matters.** The
archived ghost sits at the 4th–22nd percentile of all scores, so "above the
ghost" admits ~900 of 907 non-members; the ghost is a validity check, not a
cutoff. Worse, `excess_membership.py` draws its donor from `[0, members)`,
i.e. from the coupled web. Proposition 1 (new theory section) guarantees a
ghost's null only when the donor's self-baseline saturates, which a driven
donor's does not. Measured on 200-ghost panels per system (800 total): every
ghost above +0.001 had a web donor with one-step self-predictability
0.84–0.95; ghosts from donors above 0.99 never exceeded +0.0004.

**VERIFIED — corrected rule.** Panel restricted to donors with self-
predictability > 0.99, threshold `max(0, panel max)`:

| rule | true pos | false pos | precision |
|---|---|---|---|
| single ghost, `> abs(ghost)` | 82 | 43 | 0.656 |
| corrected panel | 81 | 4 | **0.953** |

Role accounting at the corrected threshold, pooled over four systems:
driven 81/294 = **28% recall**; sources **0/82**; isolated **0/24**. The
zero on sources and isolated channels across four systems is the sharpest
confirmation to date that the estimand is drivenness and not participation.

**Headline results unchanged.** Top-10 precision remains 30/30; this
corrects the threshold rule and the recall claim, not the reported cores.

**The honest recall statement, now in the paper and abstract:** MACE is a
high-precision, low-recall detector. Absence from the driven core is NOT
evidence of autonomy.

## Adversarial review corrections (2026-08-15)

A multi-lens adversarial review of the paper raised 36 findings; 8 were
refuted under independent verification and 28 survived. Every surviving
finding below was re-derived from primary data before any text was changed
(`scripts/verify_review.py`), and the final state is gated by
`scripts/audit_paper_numbers.py`, which re-derives each number from
`ExpOutput/` and asserts it appears in the compiled PDF (38/38 passing).

**VOID - the pairwise-CCM cost was double-counted.** `ccm(i,j)` returns both
directions, but the benchmark loop calls it for every ORDERED pair and divides
by `v(v-1)`, so the projection assumed twice the necessary work. Corrected:
1.14 s per unordered pair, **15,800 h** (not 31,700) for V=1e4, worm 2.7 h (not
5 h). The climate (17,000 h) and fish (800,000 h) figures already used the
correct convention and are unchanged. The conclusion is unaffected: 1.8 years
is still infeasible.

**VOID - my own in-degree claim was directionally false.** The paper claimed
flagged channels have mean in-degree 1.63 against a population 1.20. The 1.20
is just `n_edges/MEMBERS`, a fixed generator constant carrying no detection
information, and it includes the 106 in-degree-zero channels that can never be
flagged. Against the admissible comparator: flagged 1.617 vs undetected driven
1.638, **Mann-Whitney p = 0.63**. There is no in-degree effect. The claim is
removed; the companion claim (no in-degree-zero channel is ever flagged, 0/106)
is verified and retained.

**VOID - "four systems" were three.** `build_system_hetero(seed=0)` and
`build_system(seed=0)` produce a byte-identical coupled web (max abs diff 0.0,
identical truth vectors); only the autonomous background differs. Distinct
totals are **219 driven / 63 source / 18 isolated** from three graphs, not
294/82/24. Precision on the three distinct graphs is **1.00** (63/63) and the
heterogeneous re-analysis is now reported separately (18 TP, 4 FP, 0.82).

**VOID - "zero false alarms" was false three ways.** The stated rule (above the
signed ghost) gives 742-900 false alarms since the ghost is negative; under
`|ghost|` the heterogeneous cell is **43, not 0**; and the "+0.0005 an order of
magnitude below the weakest top-20 member" sentence is inverted, the autonomous
channel (+4.03e-4) being LARGER than that system's weakest top-20 member
(+3.67e-4). All 43 are sine-family, the degree-2 function-class failure the
theory predicts; at degree 3 the count is 0 and the margin is 15x. Table now
labels degree and prints true counts.

**VERIFIED - Prop 1's premise is absent on every real deployment.** Across all
ten worm recordings self-R2 has median 0.179-0.222, max 0.352, and **0 of 1,276
channels reach 0.9**, against the 0.99 the donor rule requires. The paper's own
step 3 said "do not proceed"; the flagship deployment proceeded. Now disclosed:
on real data the ghost is an empirical artifact meter, not a corollary, and the
real results rest on their external checks alone. Step 3 is graded rather than
absolute, and the panel correction is scoped to synthetics.

**VOID - Table 2's worm ghost range.** Measured -0.041 to **+0.0035**; the table
said "-0.03 to 0.00". Wild-type worm 0's ghost is positive and by our own rule
that scan should have been discarded. Disclosed, with the note that worm 0 is
the weakest animal anyway (top-10 8/10 vs 10/10 for worms 1-2), so excluding it
strengthens rather than weakens the result.

**VOID - the EEG recruitment claim.** `concentration()` clamps at zero, so both
numerator and denominator are surviving-positive mass. Spearman(top-4 share,
n_positive) = **-0.925 (p=2e-6)**; seizures reduce driven channels 18.5 -> 11.7
and total drive 1.15 -> 0.61. Three clamp-free statistics show no ictal rise
(all p = 0.42-0.84). The reported statistic does rise (paired Wilcoxon
**p = 0.031**, previously reported with no test at all), but it reflects
redistribution onto fewer channels, not recruitment.

**VOID - "neuron by neuron" intervention resolution.** Percentiles are
within-animal ranks and the silenced cohort's identified neurons fall as a
block (-0.14), so any initially-high cell must fall. Three nulls added
(`scripts/intervention_null.py`), all computed on the pooled-row statistic the
table actually reports. Cohort-label null: VB01 p=0.032, AVA p=0.008, AVE
p=0.024. **After max-statistic correction over the ten cells, only VB01
survives** (p=0.032); AVA reaches p=0.135. RIB, AIB and AVB lie inside the
range wild-type-only split-halves already produce, so **the AVB rise is not
supported** and its reorganisation reading is now marked as an unsupported
conjecture. The cohort-level effect is real; cell-level resolution is withdrawn
from the abstract, highlights and conclusions.

**VERIFIED - Prop 2 needed a hypothesis Prop 1 states.** Eq. 1 applies the
polynomial map to the own lags only and appends the code RAW, so the readout is
affine in the code while the decoder is tanh-nonlinear. With an oracle code the
affine readout recovers 0.0394 of an identified 0.0443, and an
own-by-code-interaction readout recovers 0.0443 to 1e-12; the shortfall is
10-16% (`scripts/readout_class_gap.py`). MACE therefore computes a LOWER BOUND.
The error direction is favourable (false negatives, never false positives) and
Prop 1, which carries the no-false-positive guarantee, is untouched.

Also corrected: a mangled `\ref` shipping as "Section efsec:theory" in the PDF;
"Figure 1 shows the crossing" (the curves never cross in the plotted mean, and
the collapse appears in one of two seeds); Figure 2's body text quoting means
against a caption quoting medians; Takens cited for the box-counting result
that is Sauer, Yorke & Casdagli (added); Definition 1's missing invertibility
hypothesis; four bare tables promoted to numbered floats. A new **Data and
methods** section gives per-deployment E, tau, b, M, epochs, degree, optimiser,
splits and hardware, none of which appeared anywhere before, and corrects the
claim that tau came from the first minimum of the auto-mutual-information (no
such computation exists; tau was declared per dataset on physical grounds) and
that E=3 throughout (the synthetics use E=2).

## Literature audit of the worm claims (2026-08-15)

Two independent literature searches against the C. elegans primary literature.
Outcome: the METHOD appears novel (no prior work scores each neuron by residual
predictability from the rest of the brain), but several BIOLOGICAL claims were
either already published or anatomically wrong.

**RESOLVED, not a problem - no stimulus confound.** A flag was raised that
AVA_HisCl might have been recorded under stimulus while WT_NoStim was not,
which would confound silencing with stimulation. Checked directly
(`scripts/check_kato_arms.py`): the `dataset` field of every recording in both
arms ends `_1mMTet_basal_1080s`. Both arms are basal. The 8-state vs 4-state
annotation difference is annotation granularity (WT_NoStim carries
fwd/rev1/rev2/revsus/slow/dt/vt/nostate; AVA_HisCl carries a 4-level
reversal-only key), not a difference in recording condition.

**VOID - "AVA's module and motor pool" is anatomically wrong.** Of the five
cells listed, only AVE (gap-junction coupled to AVA) and VA01 (A-class) are
AVA's targets. VB01 is B-class, driven by AVB via gap junctions (Kawano et al.
2011). RIB is a forward-command neuron, anticorrelated with AVA. RMED is a head
motor neuron. Corrected to "drivenness declines across the command cycle",
which is also what Kato et al. describe (network-wide uncoupling) rather than
selective loss of AVA's outputs.

**VOID - the RIM/AIB retention was presented as an observation the method
surfaced.** It is published. Gordus et al. 2015 silenced AVA with the same
histamine-gated channel and found AIB and RIM not only retain responses but
respond FASTER and MORE RELIABLY without AVA. The AVA::HisCl recordings are
also not a new experiment; they are the perturbation published with this
dataset. Reframed as convergent validation: an unsupervised score, given no
labels and no knowledge of the manipulation, reproduces a known dissociation.
That is worth reporting and is now what we report.

**VOID - the AIB->RIM sign was wrong.** Piggott et al. 2011 show AIB INHIBITS
RIM, and it is RIM SUPPRESSION that triggers reversals "independently of the
AVA/AVD/AVE-mediated stimulatory circuit". The AVA-independent route is
disinhibitory, not a driving input. Corrected.

**VOID - "RIM's intrinsic dynamics".** No primary literature reports plateau
potentials, bistability or persistent currents in RIM; the only published RIM
whole-cell recordings support a graded, non-spiking, monotonic response, and
two modelling papers built on them classify RIM as the near-linear exemplar.
Sordillo & Bargmann attribute RIM's two modes to circuit-level mechanisms
(glutamate/tyramine when depolarised, gap junctions when hyperpolarised), and
Gordus et al. attribute the bimodality of AIB/RIM/AVA jointly to network
feedback. We now claim only that RIM and AIB carry drive that does not route
through AVA, which the interventional literature supports.

**VOID - the AVB rationale.** Meng et al. 2024 (Sci Adv) report that
inactivating AVA REDUCES AVB calcium, and that AVA and AVB are not strictly
reciprocally inhibitory: AVA tonically excites AVB extrasynaptically over
exactly the timescale a sustained histamine silencing occupies. Their model
predicts AVB should FALL. The rise was already downgraded to an unsupported
conjecture on statistical grounds; the contradicting prediction is now cited
alongside it.

**VERIFIED but reframed - the wild-type command ensemble.** Not a discovery.
Kato et al. identified it from unsupervised decomposition of these same
recordings, and Uzel et al. 2022 independently identified an overlapping hub
set and confirmed it causally by silencing. Recovering it validates the
statistic; it says nothing new about the worm. Cited accordingly.

**NEEDS AUTHOR VERIFICATION BEFORE POSTING.** Both searches were blocked by 403
on cell.com, ScienceDirect and bioRxiv; all full texts came via PMC/Europe PMC.
Specifically unverified against the primary PDF: (i) the exact wording of Kato
et al.'s report on preserved dynamics under AVA silencing, said to be their
Figure S6 - the text now attributes this generally rather than to a numbered
figure, but confirm before it stands; (ii) whether RIM is in Uzel et al.'s hub
set (their abstract does not name neurons); (iii) Ray & Gordus 2025 (Curr Biol)
reportedly find AIB activity is largely driven by AVA, which cuts against AIB
autonomy - not yet cited, and worth reading before a referee raises it.

## Second adversarial pass, on the corrections themselves (2026-08-15)

The first round's fixes were reviewed by four fresh lenses: 24 findings, 8
refuted, 16 survived (9 blocking). Nearly all of the blockers were introduced
by the corrections, which is the argument for reviewing a diff rather than
trusting it.

**VOID - "every deployment but one cleared its ghost gate".** The correction
adopted a POSITIVITY rule for the ghost, conceded wild-type worm 0 (+0.0035) as
the sole exception, and then failed to apply the same rule to the EEG. Measured
(ExpOutput/eeg_excess/windows.csv): **5 of 14 EEG windows have positive ghosts,
largest +0.0218, i.e. 6.2x worm 0.** Table 2 reported the EEG as
|.| <= 0.034 while every other row was signed, and 0.034 is the NEGATIVE
extreme, so the summary concealed the sign. Now reported as two exceptions with
the signed range. Consequence disclosed: only records 03, 15 and 26 have both
windows ghost-clean, and on those three the concentration result falls to
**p = 0.25** against p = 0.031 on all six. Both are reported.

**VOID - E=2 on the synthetics.** The new methods table said the V=1001 runs
used E=2, read off `network_scale.py` (the 10-node pairwise experiment). The
deployment imports `E = 3` from `bottleneck_membership.py`, and the archived
encoder settles it: `ExpOutput/ensemble/models/m0.weights.h5` has an input
layer of shape **(3003, 256)**, and 3003 = 1001 x 3. Self-undermining as well
as false, since Prop 1 needs E > 2*d_q and E=2 fails it by exactly one for a
one-dimensional map. Corrected in four places.

**VOID - the Introduction kept the withdrawn per-cell claim.** Abstract,
Highlights and Conclusions were corrected; the Introduction still read
"neuron-by-neuron". The audit gate forbade "neuron by neuron" as a literal
string and so certified 38/38 on a PDF containing the hyphenated form.

**VOID - "seizures concentrate a smaller total amount of detectable drive".**
Untested, and the median moves the other way: total positive drive falls in
3 of 6 records, median RISES 0.47 -> 0.61, paired p = 0.84. The mean fall
(1.15 -> 0.61) is carried entirely by one interictal window ~8x larger than any
other. The supported claim is the channel count: 18.5 -> 11.7, five of six,
one-sided p = 0.047.

**VOID - the seizure-onset zone counted as a blindness test.** Listed as one of
"three independent natural attempts" to break source-blindness. There is no
channel-level ground truth in the recording (channels.csv has no SOZ field, and
chb01-summary.txt carries only montage and seizure times), so nothing was
tested and the proposition was unfalsifiable as posed. Removed from the count;
restated as an untested expectation.

**VOID - the in-degree null pooled the heterogeneous re-analysis** (n = 81 and
213 = 294) immediately after the paper established that pooling double-counts
seed 0. Recomputed over the three distinct graphs: **1.60 vs 1.66, n = 63 and
156, p = 0.79 one-sided**. The null survives de-duplication slightly better.
"0 of 106" is now "0 of 81 over three distinct graphs (0 of 106 including the
re-analysis)".

**VOID - "encoder architecture, optimiser and splits are shared".** Climate uses
Adam(1e-3) with batch 128 and zebrafish Adam(1e-3) with batch 16, against
3e-3/64 elsewhere. Learning rate and batch are now table columns; the caption
claims only that architecture and splits are shared, which is true.

**VOID - "every series is first-differenced".** The synthetic pipeline is not
differenced (no np.diff in bottleneck_membership / ensemble_membership /
excess_membership). Corrected, with the right justification: stationarity, not
differencing, is what the circular-shift ghost requires, and the logistic web is
stationary on a bounded invariant set post-burn-in.

**Audit gate hardened.** Two plain-string forbids let real defects through
("four systems" missed "four synthetic systems"; "neuron by neuron" missed the
hyphenated form) while the gate reported 38/38. Both are now regexes. This is
the second time a green gate accompanied a live defect.

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
9. A ghost's donor must have a saturating self-baseline. Draw ghost donors
   from the top self-R2 band, never uniformly, and never from channels
   suspected of being driven. Use a panel, not one ghost; each ghost costs
   what one variable costs.
10. Report the statistic a p-value tests, not a near relative of it. The
    intervention nulls first used a mean-of-per-animal-means while the table
    reported a pooled-row mean; the deltas disagreed (AVB 0.301 vs 0.337) and
    the p-values referred to a quantity the paper never printed.
11. A number that appears in the paper must be re-derivable from ExpOutput by a
    script that runs in CI. `scripts/audit_paper_numbers.py` is that gate.
12. Check whether a biological "finding" is already published in the dataset's
    own source paper before calling it an observation the method surfaced.
    Three of the four worm claims were prior art, one of them in the same
    animals.
13. Verify cell identity against the connectome before grouping cells into a
    module. VB01, RIB and RMED were assigned to AVA's motor pool; none of them
    belongs to it.
14. Review the diff, not just the result. Nine of this round's blockers were
    created by the previous round's fixes.
15. A forbid-list gate must match phrasings, not literals. A gate that passes
    while the defect is present is worse than no gate, because it is trusted.
16. When a rule is adopted (here: a ghost is clean iff its sign is negative),
    apply it to every deployment in the same commit. The EEG was left unjudged
    by a rule the same sentence had just invoked.

## AVB and the Kato passage, verified against primary PDFs (2026-08-15)

A third literature pass obtained full text of Meng 2024, Kawano 2011 and Kato
2015 (the last from a course mirror; cell.com 403s throughout), which closes
one of the three open verification items and sharpens two claims.

**Kato 2015 AVA::HisCl passage, now verbatim.** AVA "substantial attenuation
... strong uncoupling of AVA from the global brain cycle"; AVE and RIM
"slightly attenuated"; A-class motor neurons "significant attenuation"; but
"their phase relationships with most other neurons appeared normal" and "the
cyclical dynamics and neuronal recruitment patterns were largely preserved",
concluding AVA "is not a privileged generator of motor commands". Note this is
NOT the earlier report that "RIM dynamics retained wild-type appearance" - Kato
says RIM was slightly attenuated in POWER while phase relationships stayed
normal. That distinction is favourable to us and the text now uses it: excess
measures coupling, not amplitude, so RIM can be attenuated in power and retain
drivenness. AVB is never mentioned in Kato's AVA-silencing results, and the
Fig. 5B bar labels are vector text that could not be extracted, so whether AVB
was plotted remains unknown.

**AVB: the literature is uniformly against a rise.** Meng et al. 2024 Fig. 4D,
verified: "Inactivation of AVA led to reduced AVB calcium dynamics." Silencing
AVA with the same channel slows forward locomotion ~50% (Pokala 2014); AVA
ablation shortens forward runs (Roberts 2016: 6.73 -> 3.14 s; Rakowski 2013:
8.98 -> 0.71 s). Kawano 2011: AVA receives synaptic input from AVB but makes NO
direct synapse onto it. Two coupling results also point away from us: Meng's
AVA-TeTx WEAKENS the AVA-AVB relationship, and Kato's silencing uncouples AVA
from the global cycle. The AVB paragraph now states all of this.

**Two honest counterweights, both retained.** Reciprocal inhibition is
contested rather than refuted: Roberts et al. 2016 model it explicitly, fit
behaviour well, and take as their motivating puzzle the same paradox at issue
here (silencing reverse-command neurons REDUCES forward dwell time), so the
canonical model already predicts non-monotonic responses to removing AVA. And
nobody has measured AVB's predictability under AVA silencing; Brennan & Proekt
2019 find removing AVA computationally leaves whole-brain predictive structure
intact and identify AVB as one of only two locomotion neurons with no
significant across-animal variability. We record observation and contrary
evidence together and claim neither. Three references added: roberts2016,
pokala2014, brennan2019.

**STILL UNVERIFIED, for the author.** (i) Whether AVB appears in Kato 2015
Fig. 5B and what its bar shows - needs a human eye on the figure, and it is the
single cleanest datum. (ii) The AVA->AVB entry in the Randi/Leifer 2023
functional atlas, which lives in their web tool rather than the paper text.
(iii) Whether RIM is in Uzel et al. 2022's hub set. (iv) Ray & Gordus 2025,
reportedly finding AIB is largely driven by AVA.

17. When a claim contradicts a field's direct measurements, cite those
    measurements yourself, up front, with their numbers. A referee reaches for
    them within a minute; pre-empting costs less than defending.

## WormWideWeb gate check: freely-moving worms can carry MACE (2026-08-16)

`scripts/wormwideweb_gate.py`, declared header, predictions fixed before any
activity values were seen. Two baseline + one GFP recording (Atanas & Kim
2023), 105-151 neurons x 1600 frames at ~0.6 s. Torch/CUDA reimplementation of
the pipeline (RTX 3070); a paper-grade result would re-run through the TF
pipeline for fidelity.

- G1 length: n=1596 < 2000 validated floor. MARGINAL as declared; stands.
- G2 saturation: self-R2 median 0.64-0.70, max 0.97-0.99, ~10-15% of channels
  above 0.9. STRIKINGLY higher than immobilised Kato data (median ~0.21, max
  0.35, 0% above 0.9). Consequence: this is the FIRST real dataset where the
  corrected ghost-donor rule (self-R2 > 0.9 donors) is satisfiable. Caveat:
  traces are the site's processed arrays; denoising may inflate
  self-predictability. GFP self-R2 is low (median 0.20), consistent with
  motion being poorly self-predictable.
- G3 stationarity: PREDICTION WRONG. Declared expectation was ghost inflation
  from behavioural-state nonstationarity; measured panels are clean (median
  -0.016 to -0.033, max +0.013, 4-6% positive). The discard rule does not
  fire on freely-moving data.
- G4 GFP artifact floor: REAL and material. Activity-free worm shows top
  excess +0.081 with 22/105 channels positive - shared motion masquerades as
  drivenness, exactly as declared. Platform floor = +0.081; any drivenness
  claim on this platform must clear it, not zero.
- Verdict per pre-fixed rule: both baselines clear (top excess +0.395 and
  +0.113 vs floor +0.081). Honest thinness: only the top ~4 channels
  (baseline-1) and ~1-2 (baseline-2) clear the floor - top-k semantics with
  small k.
- Wall-clock per animal on GPU: ~10 s total (8 s encoder training). Full
  90-dataset corpus projects to ~15 minutes.

18. An activity-independent fluorophore recording is the platform-level
    negative control: it measures the artifact floor the ghost cannot see
    (within-platform shared artifacts survive circular shifting of one
    channel). Where a GFP-like control exists, use it and report the floor.

## WormWideWeb corpus scan: pre-registered replication FAILS (2026-08-16)

`scripts/wormwideweb_corpus.py`, predictions in header before execution. All
91 recordings (35 baseline, 30 heat, 8 reFed, 8 sickness, 7 patchEncounter,
3 GFP), donor-filtered ghost panels (self-R2>0.9 donors; no fallback needed
on any baseline recording). Total wall-clock: 9.2 minutes on the laptop GPU
for the full corpus including all controls - the speed claim demonstrated at
corpus scale, and what made this honest null cheap.

**P2 PASS, emphatically. The GFP artifact band is large and the ghost cannot
see it.** The three activity-free recordings show top excess +0.081, +0.133,
+0.240 with 17-42 channels above their own ghost thresholds - while those
ghost panels are CLEAN (max +0.007 to +0.043). Shared motion is genuinely
shared information across channels; circular shifting destroys alignment, so
the ghost stays at zero while real GFP channels score large excess. The
pre-fixed floor rule (max over GFP animals) gives 0.240, which swallows most
of the baseline signal band (only 10/35 baseline recordings have even their
top channel above it).

**P1 FAIL, robustly.** Pooled over 35 baseline recordings, labelled channels
(conf>=2) clearing max(thr, floor): 6, of which command/motor 2 - exactly the
33% base rate, p=0.64. Sensitivity (post-hoc, labelled as such): at the gate
floor 0.081, 19 clear at 36.8% command (p=0.44); with NO GFP floor at all,
190 clear at 34.7% command (p=0.32). At every threshold the flagged channels
are command-enriched at BASE RATE. The failure is not the floor's severity:
in freely-moving worms the channels MACE flags are simply not the command
ensemble. Coherent reading: in moving animals, behaviour- and motion-locked
shared signal is brain-wide (the source papers' own finding), so "driven by
the rest of the system" stops being selective for the command core; the
immobilised prep is what made the original result clean. Not tested further;
recorded as the honest state.

**The gate's pass was premature.** Its floor came from ONE GFP animal
(0.081); three animals put it at 0.240. Verdict flips: freely-moving
WormWideWeb data cannot carry MACE mutant phenotyping as-is. Declared
fallback stands: immobilised datasets.

What survives with value: (a) the methodological finding - surrogate/ghost
controls are structurally blind to shared-artifact drivenness on
freely-moving imaging, and an activity-independent fluorophore control is
necessary, quantifying an artifact band comparable to the largest genuine
signals; without the GFP arm this scan would have reported ~14 driven
channels per animal as biology. (b) A pre-registered, controlled negative
result obtained for ~10 minutes of compute. (c) The corpus machinery,
reusable on immobilised data.

19. A control floor estimated from one animal is not a floor. The gate used
    one GFP recording (0.081); the corpus's three put it at 0.240, tripling
    the bar and flipping the verdict. Floors need the full control arm.

## DepMap calibration phase 1: the curve exists, the ceiling is low (2026-08-17)

`scripts/depmap_calibration.py` per paper/depmap_protocol.md with amendments
A7 (gene-label ghost; the declared row-permutation ghost was a no-op by
algebra and was caught when it returned bin-identical counts) and A8 (two-axis
equivalence measure; the positive control failed on lineage-corrected Pearson
exactly as the protocol's interpretation rule anticipated - canonical complex
pairs PSMA1-PSMB5 score Pearson 0.04 but proximity 0.90, because uniform
co-essentiality lives in the mean that centring erases).

Controls, all passing after A7+A8: positive 253 complex pairs P(e_prox>0.8) =
0.88 (median 0.93); negative random pairs at base rate; BOTH ghosts flat at
base rate across bins. 24Q4 release, 1,103 lines x 17,716 genes, 9,368 after
declared filters, 39.0M pairs, seconds per histogram on the GPU.

Pre-registered predictions:
1. Curve rises - PASS. P(e_prox>0.8 | r_obs bin) climbs monotonically from
   0.2% to ~18% at r_obs 0.60-0.70 (lift ~40x over the 0.42% base rate),
   ghost flat at base throughout.
2. Ceiling under 50% - PASS, emphatically. The measured ceiling is ~18% at
   the best-populated high bin; at the declared headline bin (r_obs>0.8, 13
   pairs) ZERO pairs are equivalent on either axis. Read for a practitioner:
   even at the strongest expression redundancy, the odds that two
   interchangeable-looking genes produce the same knockout phenotype are at
   best about 1 in 5, and the most extreme redundancy delivered 0 of 13.
3. Asymmetry - FAILS AS DECLARED, direction reversed at the pre-registered
   thresholds: P(prox>0.8 | r_obs>0.5) ~ 15% while P(r_obs>0.5 | prox>0.8)
   ~ 0.6%, because equivalent pairs (~164k) vastly outnumber redundant pairs
   (~7k). The prediction was written with soft literature thresholds in
   mind; at ours it inverts. Reported as a failed prediction.
4. Lineage correction matters - PASS. Correction removes ~40% of
   high-redundancy pairs (12,017 -> 6,911 at r_obs>0.5).

Pending: gene-bootstrap CIs on the prox axis, TNBC arm, r_obs(A|rest) arm,
write-up with the practitioner framing.

20. A control that cannot fail is not a control. The declared ghost was
    invariant under the very permutation it prescribed; it was caught only
    because bin-identical counts looked too good. Before trusting any
    control, state the mechanism by which it COULD fail.
21. When a positive control fails, suspect the measure before the biology.
    The protocol wrote this rule in advance and it paid for itself the same
    day.

## iEEG gate: the saturation premise is met on real data for the first time (2026-08-17)

`scripts/ieeg_gate.py` per paper/ieeg_protocol.md (pre-registered before any
download; one breach logged there - sub-NIH1 quarantined from the
confirmatory cohort after a shell command displayed one channel's SOZ
label). Two interictal SEEG subjects from OpenNeuro ds003876, 105-180
channels at ~1 kHz, decimated to 256 Hz, middle 120 s, CAR and raw-reference
arms, float64 ridge (float32's ridge term vanishes under the ~1e12 Gram
scale of reference-dominated lag features).

G1 length: PASS, n ~ 30,000 after decimation.

G2 saturation: PASS - historic for this project. On NIH2's raw-reference arm
self-R2 reaches median 0.714, max 1.000, with 32% of channels above 0.9: the
donor-filtered ghost rule engaged on real data for the FIRST time (every
prior real deployment fell back to uniform donors). NIH1 maxima 0.909/0.917.
Proposition 1's premise, absent on every dataset in the companion paper, is
met here. Caveat recorded: raw-reference saturation is partly the shared
reference being smooth and self-predictable; the premise is met, but partly
for platform reasons rather than neural ones.

G3 stationarity: segment-dependent, as the rule intends. NIH1's segment
fails under CAR (ghost median +0.026, 68% positive) and is discarded per
rule. NIH2 passes both arms (CAR median -0.070, 4% positive; raw-ref median
+0.0001 with a very tight panel, max +0.014).

G4 reference floor: characterised, with a surprise. CAR is NOT automatically
the clean arm - on NIH1 it made the ghosts worse (68% vs 46% positive),
consistent with CAR redistributing a nonstationary common component into
every channel. The donor-FILTERED panel on NIH2 raw-ref gives the tightest
null of any arm (max +0.014). Montage choice is therefore an open design
decision; a bipolar arm will be evaluated before any confirmatory run.

Near-universal drivenness (95-103 of 105 channels above ghost max on NIH2):
volume conduction plus genuinely shared field activity make the binary
reading uninformative, so the pre-registered P-S1 depletion test proceeds on
RANKS (SOZ depletion from the top-k), which the protocol wording already
specifies.

VERDICT: platform passes conditionally. Proceed with per-segment ghost
screening, a three-way montage comparison (CAR / raw / bipolar) on gate
subjects, then the confirmatory cohort with SOZ labels opened only after
each subject's gate is clean. NIH1 remains quarantined.

22. Reference/montage choices are controls, not conveniences: CAR moved a
    ghost panel from 46% to 68% positive on the same segment. Evaluate
    montages as arms with their own ghosts, never adopt one by convention.

## DepMap calibration: gene-bootstrap CIs (2026-08-17)

Phase C per A4, B=100, both axes, 58 s on GPU. Headline (prox axis):
base rate 0.42% [0.37, 0.47]; ceiling (r_obs 0.60-0.70 pooled) 17.4%
[7.8, 30.9]; lift 42.6x [18.2, 76.9]; r_obs>0.8 bin 0 in all 100 resamples,
reported with the rule-of-three bound (<=23%) rather than the bootstrap's
overconfident [0,0] at n=13. Files: bootstrap_prox.csv, bootstrap_sel.csv,
bootstrap_ci.csv; cover figure scripts/fig_depmap_calibration.py.

## iEEG montage comparison: CAR loses, bipolar wins on control grounds (2026-08-17)

`scripts/ieeg_gate.py` cohort run: 6 interictal subjects (ds003876) x 3
montage arms, ExpOutput/ieeg_gate_cohort.csv. Rule 22 vindicated by the
full table:

- CAR, the conventional default, is empirically the WORST arm: the only G3
  stationarity failure in the cohort (NIH1), the largest ghost maxima (mean
  0.099 vs ~0.03 for raw and bipolar), and near-zero donor qualification.
  Retired as a candidate.
- BIPOLAR passes G3 on all six subjects, produces the most selective maps
  (mean fraction of channels above threshold 0.53 vs 0.78 raw - volume
  conduction partially cancelled, which is its job), and engages the donor
  rule with no fallback on NIH4 and NIH5.
- RAW shows the highest saturation (NIH2 32% of channels above 0.9) but the
  least selective maps (frac-above ~0.78 and up to 1.00), consistent with a
  smooth shared reference inflating both self-predictability and apparent
  drivenness together.

Clean-gated arms (G3 pass AND donor rule engaged): NIH2-raw, NIH4-raw,
NIH4-bipolar, NIH5-bipolar. NIH4 is clean on both non-CAR montages.

DECLARED NOW, before any SOZ label is opened, as an amendment to
paper/ieeg_protocol.md: the confirmatory P-S1/P-S2/P-S3 analysis uses
BIPOLAR as the primary montage (structural cancellation of the identified
confound class, universal G3 pass, most selective maps), with RAW as the
labelled sensitivity arm and CAR dropped. Confirmatory inclusion = G3 pass
on the primary montage; the depletion test runs on ranks as declared.
NIH1 remains quarantined (gate/exploratory only).

## DepMap phase 2: the three declared arms (2026-08-17)

`scripts/depmap_arms.py`, measures imported from depmap_calibration so they
cannot drift from the published numbers.

### Arm A - TNBC: closed as declared-underpowered, with a definition failure

The receptor-low definition was gated by a positive control BEFORE the arm
ran, using documented cell-line receptor status only (no dependency data).
The first definition (lower tertile of ESR1/PGR/ERBB2 among breast lines)
recovered 4 of 19 documented TNBC lines: it FAILED its positive control, so
per rule 21 the definition was replaced rather than the biology
reinterpreted. Replacement: ESR1,PGR <= 1.0 log2(TPM+1) (the conventional
not-expressed line) and ERBB2 <= breast median (HER2+ lines carry the
amplified mode). Recovers 10/19 known TNBC, admits 0 of 8 documented
receptor-positive lines. n = 18 lines.

A separate bug was caught in the first definition: PGR's lower tertile IS
0.00, so a strict `<` against a threshold sitting on the distribution's floor
excluded exactly the receptor-negative lines it was meant to select (0 lines
selected). Boundary made inclusive.

Result, reported as the protocol requires rather than as a claim: at n = 18
the calibration DEGENERATES. 11.4% of pairs land above r_obs 0.6 (4.82M of
42M), against 0.33% in the 50-line breast panel (145k of 44M) - a ~35x
inflation of the high-redundancy bin, because r_obs is a noisy statistic at
n = 18. Lift collapses from ~11x (breast, 0.086 over 0.0078) to ~3.3x (TNBC,
0.042 over 0.0126). The TNBC arm supports NO subgroup claim, exactly as
declared in advance, and is not rescued by re-binning.

### Arm B - many-to-one: panel redundancy is not better than best-pair

Per gene: best-pair r_obs, panel R^2 from a ridge fit on its top-10
expression partners, and whether ANY partner is interventionally equivalent.
AUC for predicting has-an-equivalent-partner: best-pair 0.607, panel 0.614.
P(equiv | axis >= 0.6): best-pair 0.735 (n=623), panel 0.677 (n=2687).
The pre-registered question is answered in the negative: panel redundancy
carries essentially the same information as best-pair redundancy, and is not
the better selector. Base P(a gene has some equivalent partner) = 0.542.

### Arm C - real gene families (HGNC) replace the symbol-root proxy

Proxy agreement with HGNC gene groups: 0.703 (3,644 false positives, 1,626
missed). The published direction is unchanged: ceiling at r_obs >= 0.6 is
0.264 excluding real families vs 0.231 excluding proxy families. New finding:
genes INSIDE a real family have a LOWER ceiling (0.139) than genes outside
one (0.264) - consistent with the buffering blindness the study declared,
where paralogs that compensate for each other look non-equivalent under
single knockout precisely because they are redundant.

### Rules added

23. A definition is a measure. Gate it with a positive control before running
    the arm it defines - the TNBC definition failed at 4/19 and would have
    produced a confidently wrong subgroup analysis.
24. Calibration needs cohort size. Below a few dozen samples the observational
    axis inflates (here 35x in the top bin at n=18) and the curve flattens;
    a subgroup calibration is not a small version of the pan-cohort one.

## iEEG confirmatory cohort gated: 29 subjects, bipolar 29/29 (2026-08-17)

Full ds003876 interictal cohort fetched (29 subjects with both EDF and
channel table; the remaining site prefixes use different task names in their
filenames and were not retrieved). 87 arms, ExpOutput/ieeg_gate_cohort.csv.
The first attempt was lost to a power cut at subject 8; the gate is now
resumable, writing each (subject, montage) row as it completes.

The montage declared primary BEFORE any label was opened holds at cohort
scale:

  montage   G3 pass   donor rule engaged   mean ghost max   mean frac above
  bipolar     29/29          10                 0.031            0.58
  raw         28/29          15                 0.070            0.71
  car         25/29           6                 0.130            0.42

Bipolar is the only montage that passes the stationarity gate on every
subject, and it retains the smallest ghost maxima. CAR again fails most often
(4 subjects), confirming the six-subject finding at scale. Raw engages the
donor rule most often but produces the least selective maps.

Median bipolar channels retained: 113. Median self-R2 max: 0.972.

CONFIRMATORY COHORT (bipolar, G3 clean, NIH1 excluded by the standing
quarantine): 28 subjects. Of these, 10 are theory-licensed in the strong
sense (donor rule engaged, no fallback): NIH4, NIH5, PY18N002, PY18N007,
PY18N013, rns003, rns004, rns005, rns006, rns013. The pre-registered analysis
will report the full 28 as primary and the 10 as the licensed-subset
sensitivity arm, both declared here before any SOZ column is read.

No seizure-onset, resection or outcome column has been read at any point.

## Two published DepMap numbers corrected (2026-08-20)

Re-deriving every quoted number from its output file, before writing the
short paper, caught two errors in figures already published in FINDINGS.md.
Neither was fabricated; both are the same failure - a number whose stated
definition did not match how it was computed.

1. FLOAT BIN SELECTION. R_EDGES stores the 0.60 edge as 0.6000000000000001,
   so the bootstrap CI's `.isin([0.60, 0.65])` matched the 0.65 bin ONLY
   while the label claimed the pooled 0.60-0.70 range. The published
   "ceiling about 17% (95% CI 8-31), lift 43x (18-77)" is exactly
   reproducible as the 0.65 bin alone (222 pairs). Correctly pooled over
   0.60-0.70 (911 pairs): ceiling 17.0% [9.7, 27.3], lift 39.3x
   [23.0, 64.8]. The point estimate barely moves; the interval tightens, as
   it must with four times the pairs. Fixed with a tolerance-based mask.

2. THRESHOLD MISMATCH. "Lineage correction removes ~40% of apparent
   high-redundancy pairs" was computed at a lower redundancy threshold than
   the surrounding prose implied. Removal is threshold-dependent: 37% at
   r2>=0.4, 42% at >=0.5, 56% at >=0.6, ~70% at >=0.7. FINDINGS.md now
   quotes 56% and names the threshold.

`scripts/depmap_paper_numbers.py` re-derives every DepMap number from the
output files and is the audit gate for the short paper, matching the
companion paper's 37/37 gate.

Rule added:

25. Never select a histogram bin by floating-point equality. Bin edges built
    by linspace do not compare equal to their decimal labels, and the failure
    is silent: a selection that matches half of what it claims still returns
    a plausible number.

## iEEG confirmatory SOZ study: labels opened, mixed result (2026-08-20)

Stage 1 scanned all 28 confirmatory subjects with channel labels but no
label column read. Stage 2 opened soz/epz/rz and participants.tsv for the
first time and applied the tests fixed in commit 67571af, before any label
was seen. `scripts/ieeg_soz_confirmatory.py`, outputs in ExpOutput/ieeg_soz.

### P-S1 (confirmatory, bipolar): PASSES its declared test

Stouffer z = +5.107, p = 1.6e-07 one-sided, 26 subjects (2 excluded for the
declared group-size minimum). Driven-core membership: SOZ 51.7% vs non-SOZ
61.2%. Median rank-biserial +0.172.

Robustness, run because the pooled z and the sign test disagreed:
leave-one-out never breaks it (worst case, dropping NIH8, z = +4.39), and
dropping the top 5 contributors still leaves z = +2.05, p = 0.020. So the
result is NOT an artefact of one or two channel-rich subjects; it degrades
gracefully, as a distributed effect should.

But it is heterogeneous. 17 of 26 subjects show depletion (sign test
p = 0.084, not significant), and two subjects show clear REVERSAL: NIH4
(rbc -0.473) and NIH5 (rbc -0.242).

### The declared robustness checks do NOT corroborate it

  raw-reference sensitivity arm (P-S1): z = -0.640, p = 0.74. NULL. Dropping
    its top contributors drives it further negative. The arm was declared
    precisely to test P-S1's robustness to montage, and P-S1 does not
    survive it.
  P-S2 (WEAK): bipolar z = -1.995, p = 0.977 - failed and REVERSED. Raw
    z = +3.960, p = 3.8e-05 - significant in the PREDICTED direction.
  P-S3 (EXPLORATORY): prediction was weaker depletion in surgical failures.
    Observed the opposite: failure median rbc +0.256 (n=17), success +0.022
    (n=9). Failed, reversed.

### Verdict

The pre-registered primary test passed, and passed a robustness check it was
not required to pass. It is not corroborated: the sensitivity arm is null
and both secondary predictions failed, one of them reversed. Each montage
supports a DIFFERENT one of the two predictions and neither supports both -
which is what preprocessing-driven structure looks like, not what a
biological effect looks like.

Recorded as SUGGESTIVE, NOT ESTABLISHED. No claim that MACE's source
blindness is confirmed on clinical data may be made from this. The montage
dependence is now the primary open question, not the p-value.

Rule added:

26. When two preprocessing choices each support a different one of your
    predictions, and neither supports both, suspect the preprocessing before
    the biology. A pre-registered primary arm passing while its declared
    sensitivity arm returns null is a warning, not a technicality.

## Why bipolar and raw disagree: they are not the same measurement (2026-08-20)

DIAGNOSTIC, exploratory, run after the confirmatory tests were recorded.
`scripts/ieeg_montage_diagnostic.py`, 28 subjects, no rescanning.

The hypothesis under test was mine and it was the convenient one: that the
shared reference SATURATES the raw arm so its null is uninformative rather
than contradictory. It is only partly supported, and a stronger explanation
displaced it.

  D1 dispersion. Raw carries 26% MORE per-channel variance than bipolar
     (SD 0.079 vs 0.062), not merely a level shift, though its median excess
     is 4.5x higher (+0.071 vs +0.016). A pure level shift cannot change
     ranks, so this alone would not explain a null rank test.
  D2 saturation. Raw puts 89.9% of channels above threshold against 65.0%
     for bipolar. Partial support for the saturation story.
  D3 separation. SOZ vs non-SOZ, in within-subject SD units: bipolar median
     d = +0.453, positive in 22 of 26 subjects; raw median d = -0.079,
     positive in 9 of 25. The bipolar separation is more consistent than the
     rank test suggested (22/26 vs 17/26); the raw arm has nothing.
  D4 agreement. DECISIVE. Correlating each bipolar derivation against the
     mean of its own two constituent contacts in raw gives a median Spearman
     rho of +0.078 (range -0.551 to +0.628). The two montages rank the same
     tissue almost INDEPENDENTLY of one another.

Conclusion. The arms do not measure the same quantity. A bipolar derivation
is a difference of adjacent contacts and is dominated by the local gradient;
a referenced contact is dominated by the shared reference and far field.
Near-zero agreement is therefore expected BY CONSTRUCTION, which means the
raw arm was never a valid sensitivity check on P-S1. It was a different
measurement, not the same measurement under different preprocessing. Its
null neither corroborates nor contradicts the bipolar result.

This does NOT establish P-S1. It replaces one objection with another. The
raw null is no longer evidence against the finding, but drivenness on iEEG
is now shown to be montage-determined to the point where two standard
derivations rank channels independently, so any claim must name its spatial
scale. The bipolar finding reads as a LOCAL-FIELD phenomenon, invisible in
the referenced signal.

Status unchanged: SUGGESTIVE, NOT ESTABLISHED. What is needed is a genuine
sensitivity arm at the same spatial scale - a different local derivation
(e.g. Laplacian, or bipolar with alternate contact pairings) - not a
different spatial scale.

Rules added:

27. A sensitivity arm must vary the analysis choice while holding the
    measured quantity fixed. Two preprocessing pipelines that produce nearly
    uncorrelated orderings of the same channels are two measurements, and
    disagreement between them is uninformative rather than damning.
28. Check that a control and its target correlate at all before treating
    the control's null as evidence. We ran a 28-subject study before
    establishing that its two arms measured the same thing; they did not.

## B=1000 bootstrap precision check: pre-registered intervals hold (2026-08-20)

Declared before running (previous entry). Result, prox axis, float-safe
pooling over r2 0.60-0.70:

              B=100 (pre-registered)      B=1000 (precision check)
  base rate   0.42% [0.37, 0.47]          0.42% [0.37, 0.48]
  ceiling     17.0% [ 9.7, 27.3]          17.2% [10.6, 25.4]
  lift        39.3x [23.0, 64.8]          40.8x [25.2, 60.9]

Point estimates reproduce; intervals tighten modestly, as expected when the
percentile bounds move from ~2.5 to ~25 order statistics. No material
disagreement, so per the declared rule the pre-registered B=100 intervals
remain primary and the B=1000 run is reported alongside them in the
manuscript's Methods. Wall-clock 51 min on the laptop GPU shared with other
work.

## Two citation author lists were wrong in the submitted manuscript (2026-08-20)

Verification of the three citations flagged as sourced-from-search rather
than read found two errors. Both were REAL papers with FABRICATED author
lists - plausible names attached to genuine titles, which is the specific
failure mode that made verification necessary.

  WRONG: Pacini C, Duncan E, Goncalves E, Garnett MJ. The present and future
         of the Cancer Dependency Map. Nature Reviews Cancer. 2024.
  RIGHT: Arafeh R, Shibue T, Dempster JM, Hahn WC, Vazquez F. Nature Reviews
         Cancer. 2025;25(1):59-73. (Pacini and Garnett authored a different
         2024 paper, on clinically informed target prioritisation.)

  WRONG: Parvin S, Ramirez-Labrada A, Aumann S, Lu X. Targeting synthetic
         lethal paralogs in cancer. Trends in Cancer. 2023;9(5):397-409.
  RIGHT: Ryan CJ, Mehta I, Kebabci N, Adams DJ. Same title, journal, volume
         and pages.

Verified correct on checking: Boyle EA, Pritchard JK, Greenleaf WJ (Mol Syst
Biol 2018;14:e8594) and Ito T, Young MJ, Li R et al. (Nat Genet
2021;53:1664-1672).

The second error is the more embarrassing: the misattributed review is by
Colm Ryan, who was suggested to the journal as a reviewer for this
manuscript. Corrected locally; the submitted version carries the errors and
the editor is being notified.

Rule added:

62. A citation obtained from a search result is unverified until the title,
    journal, volume AND author list have been checked against the publisher
    record. Titles and journals are usually right in search summaries;
    author lists are where the fabrication happens, and they are the part a
    specialist reader notices first.

## Source-detection complement: derived, gated, REJECTED (2026-08-20)

paper/source_detection_note.md, scripts/source_outflow_gate.py.

MACE is blind to sources by design, which is why the iEEG study had to be
framed around SOZ depletion rather than detection. Proposed dual: outflow(q)
= what q's history adds to predicting the rest-of-system code, beyond that
code's own history, with q masked out of the code. Linear in V because the
masked autoencoder is already trained with channels missing, so masking at
inference is in-distribution and needs no retraining.

Predictions were written down before the gate ran. Outcome:

  S2 ghost      PASS after a fix (see below): +0.0040
  S3 maturity   PASS: +0.0054 / +0.0051 / +0.0064 at 5 / 15 / 40 epochs, no
                decay, so it is not a difference-based importance score
  S1 separation FAIL: source +0.0060, isolated +0.0037, sink +0.0062

A first implementation scored source, isolated and ghost identically at
+0.026. The baseline held one time point of the code while the added term
held three lags plus polynomials, so every channel gained by supplying
temporal depth. The GHOST is what exposed it: a channel coupled to nothing
cannot have real outflow, so an equal score was diagnostic. Fixed by giving
the baseline a delay embedding of the code.

The surviving failure is structural. Sinks score as high as sources because
a sink's history proxies its driver's history, so the statistic reports
shared-driver correlation as outflow - the mediated-false-positive failure
this project already documented for PCMCI, reached from the other side. Not
fixable by thresholds: the confound is in the estimand.

REJECTED as formulated, not carried to real data. The inflow x outflow
quadrant map remains the right shape; this outflow is the wrong second axis.

Rule added:

63. When proposing a dual to an existing statistic, check first whether the
    dual reproduces a failure mode already catalogued for the original.
    Leave-one-out on a trained model was rejected on paper as a
    difference-based importance score; the surviving candidate then failed
    on the confound its own companion paper attributes to a competitor
    method. The catalogue of known failures is a design constraint, not just
    a discussion section.

## Literature check on the rejected dual: the barrier is structural (2026-08-20)

Five-angle deep-research survey with adversarial verification per claim (100
agents completed, 3 verification agents lost to connection errors). Full
write-up in paper/source_detection_note.md. Three results bear on today's
rejection.

1. NOBODY HAS LINEAR COST IN V. ACD, neural Granger causality, PCMCI/PCMCI+,
   oCSE and Large Causal Models all enumerate on the order of V^2 candidate
   directed links; their efficiency comes from cheaper tests, not fewer.
   "Amortized" in this literature indexes over SAMPLES, not variables. The
   one runtime reported as independent of dimensionality (LCM) is constant
   because inputs are padded to a frozen Vmax = 12 with a head dominated by
   Theta(Vmax^2 * l_max). None of these methods produces a per-variable
   source score at all: source status is read off asymmetries of an edge
   tensor.

2. THE COMMON-DRIVER PROBLEM IS THE FIELD'S PROBLEM. ACD, neural GC, LCM and
   PCMCI/PCMCI+ all assume causal sufficiency; with an unobserved driver, a
   sink proxying it is reported as a directed source->sink edge - the exact
   error our outflow made. Only LPCMCI can represent an unobserved common
   driver, via the PAG's bidirected edge, and it screens by marking X<->Y
   rather than resolving a source.

3. WHY OURS FAILED WHERE PCMCI WOULD NOT HAVE. In our synthetic system the
   driver WAS observed, so PCMCI - conditioning on the actual candidate
   drivers - would have screened the sink correctly. We conditioned on the
   CODE, a 16-dimensional lossy compression, which is not a sufficient
   statistic for the state, so driver information leaked into the sink's
   increment.

The asymmetry this exposes is worth keeping: excess works at linear cost
because its conditioning set is the variable's own history, exact and
complete, with the code entering only as an additive predictor. Outflow needs
the code to BE the conditioning set, because screening the common driver is
the whole job, and a bottleneck cannot serve. LINEAR COST AND COMMON-DRIVER
SCREENING ARE IN TENSION, STRUCTURALLY. That reframes this morning's result
from a design error to a run-in with a real barrier, and it is a defensible
reason for MACE to remain a drivenness detector that says so plainly.

Rule added:

64. Amortising through a compressed shared representation buys linear cost by
    giving up completeness of the conditioning set. Any statistic whose job
    requires conditioning (screening off a confounder) rather than merely
    predicting cannot be amortised that way. Check which of the two a
    proposed statistic needs before designing for linear cost.

## Same-scale sensitivity arms CORROBORATE P-S1 (2026-08-20)

Declared before running (commit 8af14c0), including the arms, the validity
gate, the threshold and what would count as corroboration or as failure.
28 subjects, both new arms scanned with no subject losing an arm.

VALIDITY GATE FIRST, as declared. Median per-subject Spearman with the
confirmatory bipolar arm, channels matched by shared contacts:

  laplacian     +0.643   IQR [+0.546, +0.760]   PASS (threshold 0.30)
  bipolar_skip  +0.590   IQR [+0.477, +0.753]   PASS
  raw           +0.012   IQR [-0.193, +0.320]   FAIL - uninformative

Raw's failure reproduces the earlier diagnostic (+0.078 by a different
matching) and settles it: raw never was a sensitivity arm.

P-S1 ON THE ARMS THAT PASSED:

  arm            Stouffer z        p        direction   sign p    median rbc
  bipolar (conf)   +5.107   1.6e-07     17/26     0.084      +0.172
  bipolar_skip     +4.569   2.4e-06     19/26     0.014      +0.216
  laplacian        +4.185   1.4e-05     16/26     0.164      +0.198

Driven-core membership, SOZ vs non-SOZ: bipolar 0.517/0.612, bipolar_skip
0.499/0.635, laplacian 0.534/0.601.

BOTH ARMS MEET THE DECLARED CORROBORATION CRITERION (pass the gate, pooled z
positive at p < 0.05). The bipolar_skip arm is STRONGER than the
confirmatory arm on the conservative test: its sign test reaches p = 0.014
(19/26) where the confirmatory arm's did not (p = 0.084, 17/26), and its
effect size is larger (+0.216 vs +0.172).

P-S2 remains negative and reversed on all three same-scale arms
(-0.004 to -0.059), consistent across derivations. Its earlier significance
on the raw arm is now attributable to that arm measuring a different
quantity.

A BUG FOUND AND FIXED MID-ANALYSIS. The first run reported the laplacian arm
as having no testable subject. The cause was the contact parser: laplacian
channels are named SHAFTn_lap and the parser only understood the bipolar
SHAFTm-SHAFTn form, so no laplacian channel could ever match a SOZ label and
every subject fell below the group-size minimum. Fixed to return the centre
contact and its two neighbours, which is the declared permissive rule
applied to a three-contact derivation. This was an implementation defect,
not a choice made after seeing results: the rule it implements was declared
before the arms ran.

STATUS CHANGE. P-S1 moves from SUGGESTIVE to CORROBORATED ACROSS
DERIVATIONS. Three derivations that agree with one another on channel
ordering (rho 0.59-0.64) agree on the depletion; the one derivation that
disagrees measures something else and is excluded on a criterion fixed in
advance. What is NOT established: the effect remains modest (median rank
biserial ~0.2, roughly a 10-13 point gap in driven-core membership), the
labels were already open when these arms were declared, so this is a
post-hoc robustness check rather than confirmation, and P-S3's outcome
anchor still fails in the reverse direction.

Rule added:

65. A sensitivity arm needs its own implementation check. The laplacian arm
    silently produced no testable subject because a parser understood only
    one channel-naming convention. An arm that returns "nothing to test"
    should be treated as a bug report until proven otherwise, never as a
    null.

## P-S1 FAILED TO REPLICATE on the held-out cohort (2026-08-20)

Pre-registered at 93a95be before any held-out recording was downloaded.
10 subjects, task-interictalawake, ds003876. All 10 passed the G3 gate.
`scripts/ieeg_replication.py`, ExpOutput/ieeg_replication/.

RESULT AGAINST THE DECLARED PREDICTIONS:

  P-R1  bipolar pooled z > 0, p < 0.05     ACTUAL z = -1.455, p = 0.927
                                           FAILED, and in the wrong direction
  P-R2  median rank-biserial 0.10 to 0.30  ACTUAL +0.008          FAILED
  P-R3  core gap 5 to 18 points, SOZ lower ACTUAL -4.1 pts, SOZ HIGHER
                                                                  FAILED
  P-R4  both same-scale arms positive      laplacian +0.497 (p=0.31),
                                           bipolar_skip -2.100    FAILED
  P-R5  P-S2 negative                      +0.035 / +0.361 / +1.295
                                                                  FAILED

Also: bipolar_skip FAILED the validity gate on this cohort (median rho with
bipolar 0.218, below the declared 0.30) where it passed at 0.590 in
discovery, so its P-S1 is uninformative rather than contradictory. Laplacian
passed at 0.321, barely, against 0.643 in discovery.

VERDICT, by the criterion fixed in advance: FAILED TO REPLICATE.

Power is not an adequate explanation. The predicted z was +3.2 given the
discovery effect size and n=10; the observed z is -1.455. An effect of the
discovered magnitude would have shown positive even with 8-9 testable
subjects. The point estimate is not merely small, it is on the wrong side of
zero on two of three arms.

CONSEQUENCE, as declared: the discovery result is DOWNGRADED. P-S1 was
recorded as "corroborated across derivations" earlier today on the strength
of two post-hoc same-scale arms. That status is withdrawn. The honest
statement is now:

  P-S1 was significant in one cohort of 26 subjects, survived post-hoc
  same-scale sensitivity arms in that cohort, and DID NOT REPLICATE in 10
  held-out subjects of the same dataset and condition. It should not be
  described as an established finding, and no claim that MACE's source
  blindness is demonstrated on clinical data can rest on it.

NO FURTHER COHORT WILL BE SOUGHT, as declared. The two cohorts differ in
recording site (discovery: NIH, PY, rns; held-out: jh, pt, umf), which is a
plausible source of heterogeneity - and running a site-stratified reanalysis
of the discovery cohort now would be exactly the post-hoc subgroup hunt this
pre-registration exists to forbid. It is not run. If site heterogeneity is
to be tested it must be pre-registered as its own study on data not yet
examined.

WHAT THIS COST AND WHAT IT SAVED. Roughly a day of compute and analysis. It
saved publishing a false positive: the discovery result was one write-up
away from being claimed as a clinically anchored validation of MACE's
designed blindness.

Rules added:

66. A within-cohort sensitivity arm is not a replication. Three derivations
    of the same recordings agreed with each other and all three were wrong
    about held-out subjects. Agreement among analyses of one dataset
    measures analytic robustness, not generalisation.
67. Pre-register the replication before the discovery result is written up,
    not after. Had this been run a day later, the discovery finding would
    already have been in a manuscript.

## Proposition 2's gap is COMPRESSION, not readout (2026-08-20)

Pre-registered in paper/prop2_gap_protocol.md before the experiment existed.
Four estimators: base, affine (as implemented), interact (bilinear own-lags
x code), oracle (full uncompressed state of all other channels). The gap
decomposes as readout share (interact - affine) and compression share
(oracle - interact).

At V = 14 the answer looks actionable: 76.8% readout, 23.2% compression,
ghost clean, cost 2.9x per channel and independent of V. That would say the
estimator should be changed and the paper's bound tightened.

The scale sweep, with the bottleneck fixed at 32 as deployed:

  V=14: readout 76.8%   V=30: 14.0%   V=60: 7.9%   V=100: 19.4%

The readout share collapses. This was stated as the expectation IN THE
SCRIPT before it ran: a fixed bottleneck against a growing system moves the
binding constraint from readout to compression. At V = 14 the embedding is
42-dimensional and a 32-dimensional code barely compresses, so the readout
is the only thing left to blame - an artefact of the toy size, not a
property of the method.

VERDICT: DO NOT CHANGE THE ESTIMATOR. At deployment scale 80-92% of the gap
is information the bottleneck never carried. The finding worth keeping is
that Proposition 2's bound is loose mainly because of compression, so the
lever is bottleneck width, not readout richness - and that lever is
expensive in a different place, hitting encoder capacity for all channels
rather than a cheap per-channel ridge.

Rule added:

68. Test a proposed improvement at deployment scale before recommending it.
    A synthetic system small enough to iterate on quickly is small enough to
    invert the answer: here the toy size made the readout look like the
    binding constraint when at real scale it contributes under a fifth.

## Threshold rule tested; our own defect claim refuted (2026-08-20)

Pre-registered in paper/threshold_rule_protocol.md. Five decision rules
evaluated on identical scans and identical ghost panels from the boundary
map's 51 cells.

The boundary map found recall correlating -0.44 with ghost_max and we called
the max-based threshold a defect: one unlucky surrogate setting the bar. The
test refutes that. Recall variability at the centre: MAX 0.105, Q99 0.115,
Q95 0.118, Q90 0.148. MAX is the LEAST variable non-degenerate rule.
MEAN3SD is more stable still at 0.050 but has precision 0.000 - it flags
nothing.

The logic that settles it: if the instability came from the extremeness of
the maximum, quantiles would be more stable. They are less stable. So the
whole panel shifts between scans and the maximum tracks a real property of
each scan. The threshold is working: a high null level means a noisy scan
and the bar correctly rises.

Q95 and Q90 buy recall (0.22 -> 0.28 -> 0.36) and flag up to 20% of genuine
SOURCES in the worst cell. Disqualified by T4, declared before the run.

VERDICT: MAX stands, better justified than before - the only rule tested
that is non-degenerate, source-preserving and least variable at once. The
boundary map's low-recall regions are a real limit of the method, not an
artefact of its decision rule.

Rules added:

69. Fix the decision criterion before comparing alternatives, and make it
    multi-criteria. Had the criterion been "most recall", Q90 would have won
    and broken source blindness - the property the method exists on.
70. When a diagnosis of instability is offered, test it by the mechanism it
    implies. If the fault were an extreme order statistic, less extreme
    statistics would be steadier; they were not, which refuted the diagnosis
    without needing any new data.

## Source blindness is conditional on saturation, and G3 misses the failure
## (2026-08-20)

paper/duplicate_channel_protocol.md, scripts/duplicate_channel.py.

The experiment was designed to test whether duplicating a channel
manufactures drivenness. D1 FAILED on saturated systems: flag rate 0.00
everywhere, because an isolated logistic channel's own history predicts it
almost exactly and the excess has no headroom. A declared extension added
observation noise to move the self-baseline off ceiling.

  obs   self-R2   dup-iso  plain  SOURCE flagged  ghost_max  ghost_med  G3
  0.0    0.998     0.00    0.00       0.00         0.0015    -0.0000   PASS
  0.1    0.644     0.20    0.03       0.20         0.0472    -0.0008   PASS
  0.3    0.132     0.07    0.10       0.23         0.1221    -0.0110   PASS
  0.6    0.012     0.13    0.03       0.03         0.0732    -0.0018   PASS

TWO FINDINGS, the second much larger than the one sought.

1. Duplication is a modest hazard in a band: 0.20 vs 0.03 at obs 0.1 only.

2. SOURCE BLINDNESS IS NOT A PROPERTY OF THE METHOD. It is a property of the
   saturated regime. Source false positives go from 0.000 to 0.20-0.23, with
   individual cells at 0.40 and 0.50, as the self-baseline falls. The
   boundary map recorded 0.000 across 51 cells and called it the most robust
   property measured; every one of those cells was saturated.

AND THE GATE DOES NOT CATCH IT. ghost_max rises 30-80x, so the panel
responds - but the declared G3 gate is on the ghost MEDIAN, which stays
clean. All nine noisy cells PASS G3, including one flagging 50% of sources.
The gate watches the wrong statistic for this failure: the median is unmoved
because most surrogates are unaffected, and the damage appears in the spread.

This sharpens Proposition 1's premise. The paper says the ghost's guarantee
is licensed only under saturation; this says that without saturation the
central claim - sources invisible by design - fails as well. The worm scan
reports 0 of 1,276 channels saturating, which is exactly this regime; the
paper's existing refusal to make per-neuron claims there now has a mechanism.

No gate change is made. Choosing a statistic after seeing which one would
have caught a failure is the move this project refuses; it needs its own
pre-registration.

Rules added:

71. A property measured only inside a regime is a property OF that regime
    until tested outside it. Source blindness read as 0.000 across 51 cells
    and was a fact about saturation, not about the method.
72. A gate is only as good as the statistic it watches. G3 watches the ghost
    median and passed every scan in which half the sources were being
    flagged, because the median is insensitive to the failure while the
    spread is not.

## The ghost panel cannot protect against loss of source blindness
## (2026-08-20)

paper/gate_statistic_protocol.md, scripts/gate_statistic.py. 30 fresh-seed
scans, 18 good and 12 bad, each candidate calibrated to 89% specificity so
none could win by rejecting more.

  MEDIAN sens 0.00   MAX 0.00   P95 0.00   IQR 0.08   STD 0.33

G1 HELD: the incumbent median rejects none of the twelve bad scans.

G2 HELD, and corrected us. The duplicate-channel run showed ghost_max rising
30-80 fold in failing cells and we read that as evidence a max-based gate
would catch the failure. On fresh seeds MAX has sensitivity 0.00. Per scan
it is often the wrong way round: at obs 0.3 the BAD scan has ghost_max
+0.0033 and the GOOD scan +0.1171. An aggregate difference in means is not a
classifier and we treated it as one.

By the declared rule STD qualifies (+0.33 over the median, above the 0.20
bar) and is eligible for confirmation on a held-out process. But 0.33 is a
minority: two-thirds of scans flagging up to half the genuine sources would
still pass. That is not protection.

CONCLUSION: no statistic of the ghost panel reaches usable sensitivity for
this failure. The panel tests stationarity, which is its purpose; it does
not test the saturation premise and cannot be made to. Saturation must be
reported DIRECTLY - the self-baseline R2 is already computed for every
channel at no extra cost, and a low value should disqualify source-blindness
claims. No gate change is adopted here; that needs its own pre-registration
on held-out data, since testing it on the motivating data would repeat the
error just corrected.

Rules added:

73. An aggregate difference in means is not a classifier. Ghost_max differed
    30-80 fold between failing and clean cells in aggregate and had exactly
    zero per-scan discriminating power on fresh seeds.
74. When a diagnostic cannot be made to detect a failure, report the
    underlying quantity instead of hunting for a better proxy of it.
    Saturation was measured all along and never reported alongside results.

## TF and PyTorch pipelines agree (2026-08-20)

paper/tf_torch_fidelity_protocol.md, scripts/tf_torch_fidelity.py. Three
systems, identical hyperparameters, identical numpy ridge readout so only
the encoder differed.

  Spearman median 0.955 (PASS, >= 0.90); Jaccard median 0.958 (PASS,
  >= 0.70); ghost max +0.0007 in both; mean absolute difference in excess
  0.0002 against reported effect sizes of 0.01-0.30.

The paper's TensorFlow results and everything built on the PyTorch
reimplementation - the worm corpus, the intracranial studies, the boundary
map and today's experiments - are commensurable. Cross-references between
them are safe. Recorded once so it need not be revisited.

## Perturb-seq: the richer phenotype gives a LOWER ceiling (2026-08-20)

Pre-registered in paper/perturbseq_protocol.md before the data was read.
K562 genome-scale Perturb-seq pseudo-bulk, 11,258 perturbations x 8,248
genes; universe 6,950 genes, 24.1M pairs. Observational axis held fixed at
the DepMap definition so only the phenotype changed. On-target transcripts
removed from both vectors of every pair, as declared.

CONTROLS, which gate the study:
  positive  253 CORUM same-complex pairs, median cosine +0.271, 49.0% above
            threshold
  negative  921 random pairs, median +0.005, 0.43% above threshold
  ghost     permuted labels collapse the positive control to +0.006, 1.6%
  separation +0.486; POSITIVE CONTROL PASSES, study is not void

RESULT at the declared calibration (tau chosen so the base rate matches
DepMap's 0.42%; tau=0.2 gives 0.424%):

  PERTURB-SEQ  ceiling 4.0%   lift 9.4x
  DEPMAP       ceiling 17.0%  lift 39.3x

PS2 declared no direction and all three outcomes were informative. The
answer is LOWER. The DepMap ceiling was not depressed by its growth-only
phenotype; measured on the whole transcriptomic response, redundant-looking
genes are LESS often equivalent - about one in twenty-five rather than one
in six.

Interpretation: growth is low-dimensional, so two genes can hit the same
proliferation number by unrelated routes. The transcriptome does not
collapse that way, and pairs that matched on growth separate once the full
response is visible. The DepMap figure is an UPPER bound.

CONSEQUENCE FOR THE SUBMISSION: no correction required. The paper's caution
is reinforced, and its declared limitation - that growth-only phenotyping
might understate equivalence - is now measured and runs the other way.

Rule added:

75. A limitation should be measured, not just declared. The submitted paper
    named its growth-only phenotype as the thing that might be distorting
    the headline; measuring it took one evening and showed the distortion
    ran opposite to the direction assumed.

## Source-detection retry at capacity: confound fixed, statistic still weak
## (2026-08-22)

Declared in paper/source_detection_note.md before running.
scripts/source_outflow_capacity.py, b in {16, 32, 64, 128}, three seeds.

  b      source    sink     isolated   src-sink   ghost
   16   +0.0064  +0.0062    +0.0037    +0.0011   +0.0047
   32   +0.0021  +0.0006    -0.0001    +0.0016   +0.0004
   64   +0.0015  -0.0003    -0.0006    +0.0021   -0.0005
  128   +0.0012  -0.0003    -0.0006    +0.0015   -0.0006

THE CONFOUND WAS CAPACITY, and our original diagnosis was right. The failure
was sinks scoring as high as sources (+0.0062 vs +0.0060) at b=16. By b=64
sinks fall to -0.0003 and isolated to -0.0006 while sources stay positive,
with the gap positive in 3/3 seeds at every b from 32 up.

This CORRECTS the deep-research conclusion. That pass argued linear cost and
confounder screening are structurally in tension because amortising through
a bottleneck surrenders completeness. Too strong: the tension relaxes once
the code is given the width the method needs anyway (b of order 2V), and the
sink confound disappears.

THE REJECTION STILL STANDS, for a different reason. The declared bar was a
source-sink gap above 0.01; the best is +0.0021 at b=64. Sources sit at
+0.0015 against a ghost of -0.0005 - a margin of 0.002, real and consistent
but an order of magnitude below the 0.01-0.30 range of effects MACE reports
on real data. The statistic is now unconfounded and too weak, which is a
different failure from the one recorded on Thursday.

Not settled: whether the weakness is fundamental or an artefact of this
system's coupling of 0.3. A harder-driving source should produce more
outflow. Not run; the rejection stands until it is.

Rule added:

76. Re-test a rejection when its stated mechanism stops applying. This one
    was rejected because the code was lossy; when a separate experiment
    showed the code must be nearly as wide as its input anyway, the stated
    mechanism no longer held and the retry changed the diagnosis even though
    the verdict survived.

## Source detection: LINE REOPENED, and the two statistics are complementary
## (2026-08-22)

Declared before running. scripts/source_outflow_coupling.py, b=64,
coupling in {0.05, 0.15, 0.30, 0.50, 0.70}, three seeds.

  coupling  source    sink     ghost    margin   src-sink gap
    0.05   -0.0004  -0.0007  -0.0007   +0.0003     +0.0003
    0.15   +0.0000  -0.0006  -0.0011   +0.0010     +0.0006
    0.30   +0.0009  -0.0003  -0.0006   +0.0019     +0.0016
    0.50   +0.0100  +0.0027  -0.0008   +0.0107     +0.0098
    0.70   +0.0099  +0.0062  -0.0012   +0.0103     +0.0070

All four declared conditions hold. Source outflow grows ~25-fold with
coupling, so the weak signal recorded on Thursday was not intrinsic; the
margin clears the 0.01 bar at coupling 0.50 and 0.70; sinks stay separated
in 3/3 seeds at every coupling with the gap GROWING rather than closing; the
ghost stays within 0.0013 everywhere.

The C3 risk - that harder driving strengthens the sink's proxy and recloses
the confound - did not materialise, though its first sign appears at 0.70
where sink outflow reaches +0.0062 and the gap narrows.

UNANTICIPATED AND MORE INTERESTING: the two statistics are COMPLEMENTARY.
The boundary map found MACE's own recall flat at 0.18-0.22 through coupling
0.35 and collapsing to 0.06 at 0.50, where synchrony makes a driven
channel's own history predict it and the inflow excess vanishes. Outflow
clears its bar at exactly 0.50 and above. Where inflow works outflow has no
signal; where inflow collapses outflow separates sources. The crossover sits
near coupling 0.5 and neither statistic covers the whole range alone.

NOT ESTABLISHED: synthetic logistic maps only; the margin clears but not
comfortably; the S3 maturity check was run at coupling 0.35 and has NOT been
repeated where the statistic works, which it must be before any further
claim; no real recording touched. Reopened line, not a working method.

Rule added:

77. A statistic rejected as too weak may have been measured only where it is
    weak. This one was killed at coupling 0.3 and works at 0.5; the regime
    was a parameter of our synthetic system, not of the world.

## Source detection survives the maturity check (2026-08-22)

Declared before running; third and final test.
scripts/source_outflow_maturity.py, b=64, epochs {5,15,25,40,80} at
coupling 0.5 and 0.7, three seeds.

  coupling 0.5   src 0.0046 -> 0.0108 -> 0.0087 across 5 to 80 epochs
                 sink 0.0027 -> 0.0015 (FALLS)
                 gap 0.0044 -> 0.0072 (WIDENS)
  coupling 0.7   src 0.0052 -> 0.0099 -> 0.0066
                 gap -0.0013 at 5 epochs -> +0.0064 at 80

M1, M2, M3 all PASS at both couplings. The shape is rise-then-plateau, not
rise-then-collapse: outflow peaks near 15-25 epochs and settles slightly
below, which is an estimator converging rather than Mechanism 1.

The sink is the informative part. At coupling 0.5 sink outflow FALLS with
training while the source rises, so the gap widens - the opposite of a
difference-based importance score. At 0.7 the untrained encoder starts with
the gap NEGATIVE (sinks above sources) and training fixes it: an immature
code cannot separate them, a trained one can.

STATUS: unconfounded, usable above coupling ~0.5, not Mechanism 1 in that
regime, and complementary to inflow which collapses to synchrony exactly
where outflow begins to work.

ALL SYNTHETIC. No real recording touched. The next step is a real dataset
with known sources, pre-registered - and on this project's own record the
iEEG result passed every synthetic and within-cohort check before failing
replication, so that step should be expected to be the hard one.

## Chamber source test: UNINFORMATIVE, the system is sampled too slowly
## (2026-08-22)

Pre-registered in paper/chamber_source_protocol.md before any statistic
touched the data. Causal chamber wind tunnel, 28 runs, actuators hatch/pot_1
/pot_2 known as sources by construction.

PROTOCOL ERROR DISCLOSED: the document says runs are 1,016 samples, written
after inspecting only the first file. They are 1,016 to 320,000, median
10,000; 27 of 28 are above the validated floor. No prediction was affected.

RESULT: null. Source outflow -0.0005 vs sensor -0.0004 per run, gap positive
in 13 of 28 runs which is chance; concatenated arm the same. Ghost clean.

DECLARED DIAGNOSTIC FIRES: sensor self-baseline R2 median 0.993 to 0.998,
against a declared near-synchrony threshold of 0.95. Ten of thirteen sensors
are essentially deterministic from their own history - rpm 0.998, pressures
0.998-0.999, load 0.997. The chamber settles faster than it is sampled, so
the system state already encodes the actuator and the actuator's history
adds nothing beyond it.

VERDICT: UNINFORMATIVE. No claim either way. Outflow has not been shown to
work on real data and has not been shown to fail. The test did not happen.

This is the least satisfying of the three outcomes declared in advance and
the correct one. Without the diagnostic declared beforehand, the same null
would have been readable either as a failure of the statistic or as an
excuse for one.

NOT RUN: three sensors are not synchronous (mic 0.003, signal_1 0.200,
signal_2 0.226). Restricting to them after seeing which failed the
diagnostic is the post-hoc subgroup hunt this project refuses. It needs its
own pre-registration specifying the subset from the diagnostic before any
outflow is computed.

Rule added:

78. Check the sampling rate against the system's settling time before
    choosing a dataset to test a lagged-influence statistic on. A system
    that reaches equilibrium within one sample carries no lagged influence
    to detect, however well its ground truth is known.

CORRECTION, same day: the chamber result above is VOID, not uninformative.
An adversarial audit found the verdict rests on a diagnostic that does not
discriminate, applied to a dataset that does not match the pre-registration.
Full amendment in paper/chamber_source_protocol.md.

  - the 0.95 synchrony threshold fires on the synthetic system at coupling
    0.50, where outflow clears the 0.01 bar that reopened this line (sink
    self-R2 0.9586). A threshold that fires where the method works cannot
    license "near-synchronous, therefore no test happened".
  - pot_1 and pot_2 are exactly constant in 10 of 28 runs, so source outflow
    is identically zero there by construction; those runs supply 8 of the 13
    reported positives. On the 18 valid runs the gap is positive in 5/18 -
    worse than reported, not better.
  - the data is three experiment families, not 28 random-walk runs; two
    regime-jump runs are 71.8% of the concatenated arm.
  - actuator self-R2 (0.999, never reported) is as high as the sensors', so
    high self-R2 here is oversampling, not settling. The "sample faster"
    prescription was backwards.
  - CH3 registered at 0.005, coded at 0.02; chamber_source.py imports a
    poly3 that does not exist, so the committed script cannot reproduce the
    reported diagnostic; CH1 was never evaluated.
  - ran at b = 2V, half the relative width of every test that reopened the
    line, so the null is confounded with under-capacity.

Rules added:

79. A pre-registered escape hatch must be calibrated against a system where
    the method is known to WORK before it is trusted to excuse a null. If the
    diagnostic fires there too, it does not discriminate, and invoking it is a
    claim rather than a check.

80. Verify that every declared ground-truth variable actually varies in every
    run before treating runs as replicates. A constant column is not a source;
    it is a zero that the statistic cannot help but reproduce.

81. Describe a dataset from all of its files, not from the first one and not
    from the directory name. Composition - how many experiment families, how
    the samples are distributed across them - is part of the pre-registration,
    and getting it wrong invalidates arms that depend on pooling.

82. Run a real-data test at the configuration the method was shown to work at.
    A null obtained below that capacity is confounded and carries no
    information about the data.

2026-08-23  DATASET FITNESS CHECK - the chamber is disqualified on evidence
Pre-registration: paper/dataset_fitness_protocol.md, committed at 608ddab
before the quantity was computed. Rule 79 applied to the check Rule 78 asked
for: the pass mark is calibrated against systems where outflow is known to
work, not declared as a constant.

Quantity: median over targets of R2[x_q(t+1) | own lags (+) driver lags] -
R2[x_q(t+1) | own lags]. Linear, poly2, no autoencoder trained.

  REFERENCE  coupling 0.05 +0.0007 | 0.15 +0.0044 | 0.30 +0.0114 (L30, floor)
             0.50 +0.0136 (L50, pass mark, outflow WORKS) | 0.70 +0.0189
             ghost at zero throughout; monotone in coupling
  CHAMBER    16 actuators_random_walk runs, lag_info median -0.0000,
             max over runs +0.0000, ghost -0.0000

VERDICT: DISQUALIFIED. The chamber sits below coupling 0.05 - the weakest
system tested - by an order of magnitude. The actuators' history adds nothing
to any sensor beyond that sensor's own history. There was no lagged influence
to detect, so the dataset was never a test of the method.

This vindicates the earlier conclusion and confirms the audit's charge
against its reasoning. "The test did not happen" was correct; the self-R2
diagnostic was not what showed it and could not have, since that diagnostic
fires at coupling 0.50 where outflow works.

LIMIT: linear in poly2, so a purely nonlinear lagged dependence would be
missed. What licenses it is that the same expansion returns +0.0114 to
+0.0189 where the method works.

No re-run at b = 4V. The chamber line is closed on evidence.

Rule added:

83. A check used to DISQUALIFY must be shown to return a positive, in the
    same units, on a system where the method works. Without that positive
    control its zero is uninformative and the disqualification is an
    assumption wearing a number.

2026-08-23  THRESHOLD AUDIT - which declared constants were ever calibrated?
Full record in paper/threshold_audit.md. Prompted by the chamber void; asks
the Rule 79 question of every load-bearing constant in the project. Changes
no result.

  CALIBRATED  ghost bar 0.005 (ghost_max 0.0013-0.0015 flat across all 90
              bottleneck and 51 boundary cells, so the bar is 4x the null on
              systems where the method works); length floor ~2,000 (read off
              the n sweep); gate-statistic thresholds (matched 10% specificity
              by construction); flagging threshold (per-scan surrogates, not a
              constant); DepMap tau_prox = 0.8 (same-complex pairs 0.90-0.98,
              PSMA1-KRT1 0.04 - thin anchor set but controlled both ways, and
              cross-checked independently by the Perturb-seq base-rate
              matching); dataset fitness L50.

  UNCALIBRATED
              SATURATION - most consequential. Self-R2 "near 1" licenses
              Proposition 1 and has never been made a number. No saturation
              gate exists; only a reported statistic, (self_r2 > 0.9).mean(),
              in three scripts, where 0.9 is unexamined. The failure mode is
              quantified - off ceiling, up to 50% of sources flagged, ghost
              passes (Rules 71-74) - while the licensing condition is not.
              OUTFLOW BAR 0.01 - declared, never justified; the line was
              reopened on a margin of 0.0107, clearing by 7%.
              iEEG validity gate 0.30 - moot, that result was withdrawn.
              tf/torch F1 0.90 and F2 0.70 - low stakes, passed comfortably.
              source FP rate 0.05 - never binds, observed 0.000 throughout.

The constant under review at PLOS is controlled in both directions. That was
the outcome most in need of checking, and the reason for auditing before a
reviewer does.

Rule added:

84. A licensing condition stated qualitatively is not a gate. If a guarantee
    holds only under a premise, the premise needs a number and that number
    needs a positive control, or the guarantee must be reported as
    unlicensed.

2026-08-23  SATURATION GATE - VOID, and Rules 71-72 narrowed
Pre-registration: paper/saturation_gate_protocol.md at 49078fa. Full result:
paper/saturation_gate_result.md. 57 cells, 10.5 min.

S1, the declared reproduction check, FAILED on the discovery grid. Source
false positives are 0.000 at every observation-noise level down to a
self-baseline of 0.060 (V=15, coupling 0.35, 27 cells). What falls instead is
RECALL, 1.00 to 0.25: the method stops seeing drivenness before it starts
inventing it. The script stopped at S1 as instructed; no s* was fitted.

WHY IT DID NOT REPRODUCE. The observation-noise cells that produced Rules
71-72 were run ONLY at k=2, with duplicated channels present. The crossed
cell - low saturation, no duplicates - does not exist in
ExpOutput/duplicate_channel/duplicate_channel.csv. These 27 cells are that
missing cell. The claim was generalised past its evidence.

HELD-OUT CELLS, computed before S1 was evaluated and reported as description
only, nothing fitted to them: at V=30 the failure DOES appear without any
duplicates, source FP reaching 0.80.

TOGETHER, at matched self-baseline:
  self-R2 ~0.87   V=15 srcFP 0.00   V=30 srcFP 0.20
  self-R2 ~0.65   V=15 srcFP 0.00   V=30 srcFP 0.40
The same self-R2 carries different risk at different width, so a
one-dimensional per-channel saturation bar cannot exist in the proposed form.
Rule 84's first item is answered partially and not as expected: the licensing
premise is at least two dimensional.

CONFOUND NAMED, NOT RESOLVED: b was held at 32 for both widths, so b/V is 2.1
at V=15 and 1.07 at V=30. The V=30 grid was under-capacity by inherited
default - recall 0.48-0.52 even at zero noise. Width and capacity are not
separated and no further reading of these cells will separate them.

NEEDED: saturation x b/V held at 2 x k in {0,2}, crossed. Until then the
honest statement is that sources can be flagged at reduced saturation, seen
with duplicates at V=15 and without them at V=30 under-capacity, and the
conditions are not established.

Rules added:

85. Show that the discovery system exhibits the failure before fitting a gate
    against it. This gate was designed against a phenomenon the discovery
    grid did not contain; the pre-registered reproduction check caught it and
    nothing else would have.

86. Rules 71 and 72 are NARROWED. Their observation-noise evidence is k=2
    only. The no-duplicate cell at low saturation was never run and, when
    run, gives source FP 0.000 at V=15 down to self-R2 0.06. The failure is
    real; its conditions were never established.

87. If the same value of a licensing quantity carries different risk under
    different capacity, it cannot be the gate on its own. A premise that is
    one number in the theory may need two in practice.

2026-08-23  CALIBRATED OUTFLOW BAR - the ghost is the wrong null; 0.01 stands
Pre-registration: paper/outflow_bar_protocol.md at 4a19b1a. Full result:
paper/outflow_bar_result.md. 90 runs, 24 min.

  O1  ghost null, 30 calibration seeds: median -0.0007, range [-0.0016,
      +0.0003]. A null, as required.
  O2  bar = q95(ghost) = -0.00005, two orders below the declared 0.01. O5
      does not fire; the line does not close on that rule.
  O3  sensitivity at coupling 0.50: 29/30 clear the bar, 0.97. PASS.
  O4  control at coupling 0.30: 25/30 clear it, 0.83. FAIL (bar 0.20).
  VERDICT as declared: DOES NOT SEPARATE.

DESIGN ERROR OWNED: O4 called coupling 0.30 "known NOT to work", inherited
from the coupling sweep where it failed a margin of 0.01. That is a statement
about the OLD bar, not ground truth - at 0.30 sources genuinely drive sinks,
and a no-influence bar should flag them. The control was defined in terms of
the threshold being replaced and did not survive the replacement.

WHAT SETTLES IT REGARDLESS: at coupling 0.50, sources clear the bar 29/30 and
SINKS clear it 28/30. A null-calibrated bar admits driven channels almost as
often as sources.

WHY: the isolated channels in the existing coupling sweep - real dynamics,
coupled to nothing - sit exactly at the ghost level at every coupling
(-0.0005 to -0.0007). So the ghost IS a valid null for "influences nothing".
Sinks are not at the null because a sink CARRIES A PROXY of its driver, so
masking it removes information the code was using. What confuses outflow is
proxy-carrying, not influence, and the ghost has neither.

Also: the sink proxy GROWS with coupling, +0.0027 at 0.50 to +0.0062 at 0.70,
against a flat source signal of +0.0100 and +0.0099. Risk C3, recorded as
survived when the coupling sweep ran, returns at the top of the range.

CONSEQUENCE: 0.01 is not retired and not replaced. What it was actually doing
is suppressing the sink proxy while being documented as a margin over the
ghost, and as a sink threshold it is entirely uncalibrated. Rule 84's second
item stays OPEN with a sharper specification: calibrate against the SINK
distribution, not the ghost.

NOTED, CLAIMED AS NOTHING: source outflow exceeds sink outflow in 25/30 runs
at coupling 0.30 and 27/30 at 0.50. From re-reading runs collected for
another purpose; needs its own pre-registration to be a finding.

Rules added:

88. A null control must exclude what actually confuses the statistic. The
    ghost excludes influence; what confuses outflow is proxy-carrying. A bar
    calibrated against the wrong null admits every driven channel while
    looking rigorous.

89. When a threshold is replaced, re-derive every control defined in terms of
    the old one.

2026-08-23  ERROR-METRIC SWEEP - VOID on an imported bar, over a table worth
re-running. Pre-registration: paper/error_metric_protocol.md at b5dfa70.
Full result: paper/error_metric_result.md. 15 cells, 8 metrics, one
autoencoder per cell so all metrics see identical codes and identical data.

M1 required A1 (the current metric) to reproduce: source-vs-sink AUC >= 0.80
at coupling 0.50 and degrading at 0.70. Got 0.767 / 0.704 / 0.542. The
degradation reproduced exactly; the 0.80 bar did not. VOID; the script
stopped at M1 without evaluating M2, M3 or M5.

THE BAR WAS IMPORTED FROM A DIFFERENT STATISTIC. 0.80 came from the
outflow-bar run, where source exceeded sink in 27 of 30 RUNS - a run-level
comparison of medians. AUC here is a CHANNEL-level ranking. At coupling 0.50
the medians separate cleanly (+0.0126 vs +0.0028) while individual channels
overlap enough to give 0.704. Third instance today of setting a bar or
control by importing a number from a different quantity (O4, S1, M1).

DESCRIPTION ONLY, nothing fitted or claimed:

  SOURCE vs SINK AUC     c=0.30  c=0.50  c=0.70
  A1_r2  (current)        0.767   0.704   0.542
  A2_nmae                 0.749   0.676   0.542
  A3_spearman             0.762   0.713   0.547
  A4_gauss_nats           0.762   0.671   0.531
  A5_rff_r2               0.618   0.664   0.553  ghost-excluded
  B1_recon_mse            0.444   0.400   0.367  ghost-excluded
  B2_recon_mae            0.329   0.327   0.253  ghost-excluded
  C1_conditional          0.916   0.929   0.924

  - Family A is one metric in four costumes: rescoring the same ridge
    predictions changes nothing, all agree within 0.04 and degrade together.
    A4 was declared expected-degenerate and its largest gap is 0.033.
  - Family B is worse than useless, which answers one reading of the
    question: the network's own reconstruction error ranks SINKS ABOVE
    SOURCES (AUC below 0.5) with a dirty ghost. A sink is reconstructed
    largely from its driver, so masking it damages everything correlated with
    it and the measure reads correlation, not direction.
  - C1 conditional pins sinks at the null (-0.00007) exactly where isolated
    channels and the ghost sit, at every coupling, while the source signal
    RISES with coupling (+0.00053, +0.00212, +0.00290) instead of being
    overtaken. Under R2 the sink proxy grows +0.00283 to +0.00740 against a
    source of +0.01140. Cost is magnitude: about 4x smaller than R2, still
    ~40x its own null. The declared risk - that sources recoverable from
    their sinks collapse too - did not appear here and is untested under
    denser redundancy.

NOT A FINDING. C1 was one column of eight, on one system family, with no
declared bar, from a void run. A proper test needs C1 as the primary
hypothesis, FRESH seeds, a qualitative reproduction check with no imported
constant, the decisive comparison expressed as C1 against A1 on the same
cells, and a redundancy axis.

Rule added:

90. Never import a numeric bar from one statistic to another. A run-level
    comparison of medians and a channel-level ranking are different
    quantities; a threshold meaning something for one means nothing for the
    other. Three of today's experiments voided or misfired on this mistake.

2026-08-23  CONDITIONAL OUTFLOW - CONFIRMED at k=0. The line reopens.
Pre-registration: paper/conditional_outflow_protocol.md at 2e1c431. Full
result: paper/conditional_outflow_result.md. 90 cells, fresh seeds 5-14,
17 min. A1 and C1 differ ONLY in the conditioning set.

  SOURCE vs SINK AUC, k=0      c=0.30  c=0.50  c=0.70
  A1 marginal (current)         0.768   0.655   0.517
  C1 conditional                0.856   0.877   0.872
  margin                       +0.087  +0.222  +0.356

  P1 direction reproduces (0.768 -> 0.517). P2 PRIMARY passes: C1 above A1 at
  all three, margin grows. P3 passes: C1 puts sinks on the isolated reference
  at every coupling. P5 passes: both ghosts clean.

At coupling 0.70 the current metric is at CHANCE - it cannot tell a source
from its own sink where driving is strongest. Conditioning holds at 0.872,
and the gap widens with coupling because the sink proxy that defeats the
marginal statistic is exactly what conditioning removes.

WHY, as a reason rather than a fit: a sink at t is a function of its driver
at t-1 and itself at t-1, both already in the conditioning set, so it adds
nothing unique. A source at t carries its own value at t, which the lagged
sinks do not yet contain. C1 is conditional Granger causality in code space,
not a difference-based importance score.

CONTROL, not pre-declared: sinks use r in [3.5,3.7], sources AND isolated
share [3.7,3.9]. Source-vs-ISOLATED AUC is ~0.90 for BOTH statistics at every
coupling, so neither tracks the parameter range, and conditioning fixes the
source-vs-sink comparison without breaking what already worked.

EFFECT SIZES: source median +0.00172 vs null -0.00010 at c=0.70, ~17x. Source
q25 +0.00005 against sink q75 -0.00002, so ~13% of pairs mis-order. AUC 0.87
is not a clean threshold.

P4 THE DECLARED RISK IS UNTESTED, NOT CLEARED. Copies did not zero the
sources - C1 AUC rose to 0.98 at k=1,2 - but adding copies grows the panel
while b stayed at 64, so b/V moved 4.00 -> 3.37 -> 2.91. The redundancy axis
is confounded with capacity: the FOURTH confounded axis today, repeating the
saturation experiment's mistake. Tempering it: the confound moves in the
direction that should have made the risk MORE likely, and P2/P3/P5 were all
declared and evaluated at k=0 where capacity does not move. The k>0
improvement is UNEXPLAINED and is not claimed as a benefit of redundancy.

STATUS: the line reopens on a measurement, not a threshold - with a declared
prediction, fresh seeds, and the statistic it replaces beside it. But this is
one synthetic family, coupled logistic, which is where every previous version
of this line also looked healthy. Next step is a real dataset clearing the
dataset-fitness gate at L50, which exists so the data is qualified before the
statistic runs on it.

Rules added:

91. Vary one thing. Adding channels to test redundancy also changed the panel
    size and hence the capacity ratio, so the axis measured two things at
    once. When an axis changes the channel count, hold b/V fixed by
    construction.

92. When a statistic confuses two roles, examine what it CONDITIONS on before
    what it SCORES. Rescoring the same predictions cannot separate what the
    conditioning set has already merged. A day was spent on thresholds and
    error functions over a difference that lived in neither.

2026-08-23  REAL-DATA CONDITIONAL OUTFLOW - does not transfer; the MARGINAL
statistic does. Pre-registration: paper/real_conditional_protocol.md at
79f00cc. Full result: paper/real_conditional_result.md. 11.2 min.

Datasets selected by the fitness gate, both qualifying ones tested and
reported, corrected ground truth (load_in/load_out settable), b = 4V.

  PRIMARY wt_intake_impulse_v1 m=10, 5 runs, 10 source / 55 sensor channels
    A1 marginal    AUC 0.916  source +0.00568  ghost CLEAN  perm p 0.0001
    C1 conditional AUC 0.751  source +0.00209  ghost DIRTY  perm p 0.0052
    R1 FAILS
  SECOND  wt_walks_v1 m=10, 2 runs
    A1 AUC 0.605 ghost DIRTY p 0.19 | C1 AUC 0.632 ghost DIRTY p 0.13
    R1 passes, but both near chance, both dirty, neither significant

Declared verdict PARTIAL. It should be read as R6 for C1: the dataset where
R1 passed carries no weight, and on the only clean, significant dataset
conditioning LOSES decisively to the statistic it was meant to replace.

WHY - the risk declared before the synthetic run, arriving on real data. The
synthetic protocol named it: "a source recoverable from its own sinks is
redundant too", and the redundancy axis meant to test it was confounded with
capacity. The impulse dataset has 2 actuators and 11 downstream sensors, so
the actuators ARE recoverable from the sensors and conditioning removes their
unique contribution along with the redundancy. C1's source signal is 2.7x
smaller than A1's and its source 5th percentile is NEGATIVE (-0.0035). The
synthetic system gave each source two sinks; this gives each actuator five or
six. Conditioning wins where the driver is not reconstructable from what it
drives - the opposite of most real systems.

UNEXPECTED, not the hypothesis, A1's permutation test NOT pre-declared and
computed only because withholding it would misinform: the EXISTING marginal
statistic reaches AUC 0.916 on a real physical system with structural ground
truth, clean ghost, p = 0.0001. First real-data positive in this line. One
dataset, one protocol, 5 runs, 10 vs 55 channels. To replicate, not announce.

It does NOT contradict the voided chamber run, which used m=1 (no lagged
influence to find), b=2V (half the needed capacity) and a truth set with two
of five actuators labelled as sensors. Correct all three and the same
apparatus yields a signal. The fitness gate and the b=4V rule earned their
place.

PROTOCOL ERRORS DISCLOSED: (1) the protocol says hatch is constant in the
regime_jumps runs - wrong, sd is 17.6 over 320k and 0.000 over the first 20k,
which is what was inspected; the analysis used real data and scored five
sources, so no number is affected. Second truncated-read error today. (2) R3
ghost-cleanliness FAILED on both datasets but was declared as a prediction
rather than as disqualifying, unlike the error-metric protocol the same day.
Had it been disqualifying the second dataset would have been excluded and the
verdict would have been sharper.

Rules added:

93. A conditional statistic measures unique contribution, so it fails exactly
    where its target is redundantly represented. Before choosing between a
    marginal and conditional form, count driven channels per source: real
    systems usually have many measured consequences of few controls, which is
    where conditioning loses.

94. Rule 81 extended: describe a dataset from all of its ROWS as well as all
    of its files. Twice today a truncated read reached a committed protocol.

95. Declare validity conditions as DISQUALIFYING, not as predictions. A ghost
    check that fails but counts only as a missed prediction lets an
    uninterpretable dataset into the verdict.

2026-08-24  LIGHT-TUNNEL SCREEN - DISQUALIFIED at every decimation; the
replication target falls. Pre-registered addendum + result in
paper/dataset_fitness_protocol.md; scripts/lt_screen.py; every cell of all
five candidate datasets reported. lag_info +0.0000 everywhere (best single
run +0.0025, a quarter of the L30 floor); ghost at zero; three of five
datasets never clear the 2,000-sample floor at any decimation.

Physical reading: decimation rescued the wind tunnel because rotors and air
have inertia, so a sampling rate exists at which actuator history carries
information. The light tunnel's actuator-to-sensor path is light -
within-sample at any achievable logging rate. Decimation changes how
dynamics are sampled; it cannot create dynamics the physics does not have.
The one lagged pathway (polariser servo, pol -> angle) is two of fifteen
targets; a servo-only test would need its own pre-registration naming those
targets in advance, and is a far narrower question.

CONSEQUENCE: the AUC 0.916 wind-tunnel result cannot be replicated on the
light tunnel. A second apparatus must have experimenter-set drivers AND
internal dynamics slower than its sampling - mechanical, thermal or
chemical, not optical.

Rule added:

96. Decimation can only recover lag that the physics provides. An apparatus
    whose source-to-sensor propagation is within-sample at every achievable
    logging rate cannot test a lagged-influence statistic at any decimation,
    and the fitness gate detects this for the cost of a linear regression.

2026-08-24  NIST UR5 SCREEN - DISQUALIFIED; a third disqualification mode.
Pre-registered addendum + result in paper/dataset_fitness_protocol.md;
scripts/nist_screen.py; 18 runs, all cells reported. Best cell +0.0003
against L50 +0.0136; ghost at zero.

The declared risk fired with a second jaw: the position servo closes within
one 8 ms sample (control-bandwidth synchrony), and the slower dynamics that
might carry lag at deeper decimation - thermal mass, trajectory-level
inertia - are unreachable because runs are 0.9-1.4 min, so m > 5 drops
every run below the 2,000-sample floor. Not absent physics; recordings too
short to decimate to where the physics lives.

Three screens, three modes: wind tunnel QUALIFIED (dynamics reached by
decimation); light tunnel DISQUALIFIED (no lag at any rate - optics); UR5
DISQUALIFIED (lag plausible, recordings too short to reach it).

Rule added:

97. Run length bounds the reachable decimation. A dataset can fail the gate
    not because its physics lacks lag but because n < floor x (dynamics
    timescale / sampling interval), and that inequality is checkable from
    the file listing before anything is downloaded. Check it first.

2026-08-24  BATTERY SCREEN - DISQUALIFIED at the declared grid; design error
disclosed. Pre-registered addendum + result in
paper/dataset_fitness_protocol.md; scripts/battery_screen.py; per-cell
report in ExpOutput/battery_screen/per_cell.csv. NASA PCoE randomized-usage
mirror, 28 cells, ~4.7M grid points, randomized current as the exogenous
source, voltage and temperature driven.

  m=1 +0.0010   m=2 +0.0018   m=5 +0.0100 (max cell +0.0192)   m>=10
  unreachable. L50 +0.0136. DISQUALIFIED by the declared rule.

MY ERROR: Rule 97 applied to the span, not the segment. Segments are ~12,000
points, so m > 6 was dead on arrival - knowable from the composition sweep
before the grid was declared. The upper half of the pre-registered grid
could never run.

DESCRIPTION ONLY: lag_info rises monotonically toward the truncation edge in
every family; skewed-high-40C reaches exactly L50 at m=5 with 2/4 cells
clearing. The NIST mode, softened: dynamics plausibly live at tens of
minutes to hours, and the declared grid cannot reach them.

ANOMALY: uniform charge+discharge family (RW9-12), all four cells, lag_info
-0.13 to -0.17 at m=1/2 with clean ghosts. Unexplained; no claim.

FOLLOW-UP LICENSED ONLY AS SPLIT-SAMPLE: coarser grid fixed from physics,
discovery on 4 named cells, verdict on the 24 held-out - because the regrid
is post-hoc motivated and anything less is grid-shopping.

Rule added:

98. Rule 97's inequality binds at the longest contiguous SEGMENT, not the
    recording span. Protocol outages partition a months-long record into
    pieces, and the reachable decimation range must be computed from the
    piece, before the grid is declared.

2026-08-25  BATTERY SPLIT-SAMPLE RE-SCREEN - DISQUALIFIED. Final for this
dataset. Pre-registration + result in paper/dataset_fitness_protocol.md;
scripts/battery_screen2.py; ExpOutput/battery_screen2/cells.csv.

Grid 300 s from thermal physics and protocol step duration, m=1 only by
Rule 98 arithmetic done in advance, discovery cells named before running.

  HELD-OUT VERDICT  median +0.0102 (17 cells) vs L50 +0.0136  ghost -0.0004
  DISCOVERY         +0.0035 (3 cells, no weight)
  best family: skewed high 40C +0.0133, 1/3 cells clearing

SECOND ARITHMETIC ERROR DISCLOSED: promised 24 held-out / 4 discovery, got
17 / 3. Eight cells - the whole uniform charge+discharge family (RW9-12) and
the whole uniform variable-charge family (RW1,2,7,8) - have no segment
reaching 2,000 points at 300 s (longest 1,102-1,550), because their protocol
interleaves reference cycles more often. Rule 98 one level finer: the
inequality binds per FAMILY, and I checked pooled rather than shortest. Same
error class as the rule I had just written. The criterion was fixed in
advance and applied identically to all 28, so nothing was chosen - but the
declared counts were wrong and RW9 (anomaly family) vanished, leaving the
-0.13/-0.17 anomaly unexplained and untestable at this grid.

TWO SCREENS TOGETHER: 60 s gave +0.0100, 300 s with independent held-out
cells gives +0.0102. Five-fold coarsening moved it by 0.0002, so the first
screen's ascent was approach to a PLATEAU BELOW L50, not a truncated climb
toward it. The dataset carries lagged influence (far above zero, ghost
clean) but not enough for outflow to work.

FOUR SCREENS, FOUR VERDICTS: wind tunnel QUALIFIED; light tunnel
disqualified by physics (96); UR5 by run length (97); battery on the
quantity itself at two independent grids - the first to fail because the
influence is genuinely too weak rather than because the recording could not
express it.

Rule added:

99. A sample-floor inequality binds at the SHORTEST stratum, not the pooled
    median. Checking it against pooled segment lengths silently drops whole
    families and changes the declared design after the fact.

2026-08-25  PRONTO SCREEN - DISQUALIFIED, negative at every reachable grid.
Pre-registration + result in paper/dataset_fitness_protocol.md;
scripts/pronto_screen.py; ExpOutput/pronto_screen/screen.csv. Cranfield
multiphase flow rig, 4 test-days at 1 Hz, sources = 4 commanded flow set
points, driven = 22 measured process variables, PID outputs excluded as
feedback-computed.

  m=1 -0.0001 | m=2 -0.0009 | m=5 -0.0014 | m=10 -0.0199 (1 day). Ghost at
  zero. L50 +0.0136. DISQUALIFIED.

A FORECAST OF MINE, RECORDED WRONG BEFORE RUNNING: I predicted from the
technical report that the 5-7 min flow-regime settling time would force a
grid coarser than the floor allows. That conflated two timescales - the set
points move on seconds and the loops answer there - and the composition
inspection corrected it. Kept in the protocol rather than deleted.

A COMPOSITION ERROR CORRECTED: the addendum said 0626Testday5 had flat set
points (Rule 80). Wrong - that file has 69 different columns and no SP.CV
channels at all. Exclusion stands, justification corrected.

NEGATIVE VALUES, second occurrence. Likely mechanism, offered as a reading:
three varying sources add 54 poly2 features to a 9-feature own-lag baseline,
and a ridge at fixed regularisation pays a held-out penalty for uninformative
features - a penalty that grows as decimation shrinks n, which is the
observed monotone pattern (-0.0001 at 25k samples to -0.0199 at 2.5k). On
that reading negative lag_info means "no information, plus the cost of
asking". A feature-count control would test it; not run.

WHY IT FAILS: loops close on their set points within a sample, while the
slow flow-regime dynamics live at 5-7 min where the floor puts a 60 s grid
out of reach (420 samples vs 2,000). The UR5 mode on a chemical plant.

FIVE SCREENS: wind tunnel QUALIFIED; light tunnel (physics, R96); UR5 (run
length, R97); battery (influence too weak, R98/R99, two independent grids);
PRONTO (both sides, negative throughout).

2026-08-25  ADVERSARIAL REVIEW OF THE v2 MANUSCRIPT - eight defects fixed
Six independent review lenses over paper/mace_v2.tex, each finding then
attacked by a separate refute-by-default verifier; 15 findings verified, 8
survived. Every survivor was then checked by hand against primary sources
before any edit. One lens (statistics) died on an API error and has NOT been
re-run - that pass is still owed.

THE SERIOUS ONE. The 2026-08-16 WormWideWeb corpus scan - 91 freely-moving
C. elegans recordings, pre-registered, primary prediction FAILED at base
rate (6 labelled channels clear, 2 command/motor, p=0.64) - was ABSENT from
the manuscript. Verified: grep for wormwideweb/GFP/freely-moving on the .tex
returned zero. Meanwhile Limitations (5) read "the one study in this paper
that reached a pre-registered held-out replication failed it", which was
true only because this one was left out. The paper's whole defence is that
it reports its failures, and it was inviting reviewers into a ledger that
contained an omitted one.

Also omitted: that corpus's GFP finding. Activity-free recordings score top
excess +0.081/+0.133/+0.240 with 17-42 channels above threshold while their
ghost panels stay CLEAN (max +0.007 to +0.043). The ghost is structurally
blind to within-platform shared artifacts - a SECOND blindness beside the
saturation one - and the paper's claim that "each scan reports its own
artifact floor" was unqualified.

FIXED: a new paragraph in the worm subsection reporting the failed corpus
replication and scoping the worm validation to IMMOBILISED preparations; the
GFP artifact floor and the ghost's second blindness stated; the artifact-
floor claim qualified in place; Limitations (5) rewritten to cover BOTH
failed generalisation tests.

SEVEN MORE, all verified before fixing:
  - "at the deployed b=32 the statistic was confounded" - WRONG, it was
    b=16 (source_outflow_gate.py:33; ledger L1978). Fixed.
  - "b=64, the b~2V rule applied to V=16" - WRONG, 2x16=32, so b=64 is 4V.
    The sweep shows outflow needs TWICE the relative width inflow does.
    Fixed, and the carve-out added to the recipe and Limitations (2): b~2V
    for inflow, b~4V for outflow. This also resolves an apparent
    self-contradiction, where the paper condemned the first chamber run for
    b=2V while the protocol recorded it following the published 2V rule.
  - "the apparatus settles within a sample" - re-committed the exact
    mechanism the 2026-08-22 chamber audit voided as argued backwards.
    Settling and oversampling are opposite conditions, and the paper's own
    screen.csv rises +0.0002 -> +0.0158 -> +0.0405 with decimation, which is
    the oversampling signature. Fixed to the corrected reading.
  - "laginfo -0.0000" was the pre-correction number; the corrected re-screen
    gives +0.0002. Fixed. Disqualification unaffected.
  - linear-cost-in-V stated unconditionally while the new capacity rule
    makes fixed b under-capacity: at b ~ 2V the scan is quadratic in V
    again. Scoped to the fixed-b operating point, with the wall-clock
    figures labelled as such.
  - the capacity rule was given without its measured validity range
    (b <= 128, V <= 240, one generating process). Added.
  - abstract said "95-rule ledger"; it is at 99. Fixed.

Rule added:

100. Have someone else attack the manuscript, not just the experiments. Six
     lenses over a paper whose numbers all passed an audit gate found an
     omitted pre-registered replication failure, two wrong capacity
     constants, and a mechanism the ledger had already corrected. An audit
     gate checks that quoted numbers match their files; it cannot see what
     the paper does not mention.

2026-08-31  STATISTICS RE-REVIEW: the chamber's channel-level permutation p
was invalid; the qualitative result survives correct inference. Prompted by
the adversarial review's statistics lens, which crashed on an API error
2026-08-25 and was re-run standalone.

The 0.916 AUC result (Table 8, real_conditional.py) reports permutation
p=1e-4 by shuffling all 65 source-or-sensor channels (2 source + 11 sensor
per each of 5 runs) as independent draws. They are not: channels within a
run share one trained encoder and one physical realisation of that run's
noise. Checked directly (scripts/chamber_cluster_check.py): the ratio of
between-run to within-run variance in sensor A1 scores is 3.45 - substantial
clustering. The effective sample size for inference is 5 runs, not 65
channels, and the channel-level p-value is invalid as a significance test.

Two checks temper this rather than resolve it by assumption. First, source
exceeds sensor within EVERY ONE of the 5 runs individually (not just in
aggregate), which rules out the confound that the whole effect is a
between-run artefact. Second, a correctly specified run-level cluster
bootstrap - resample the 5 runs with replacement, recompute AUC each time,
2,000 resamples - gives AUC 0.920, 95% CI [0.862, 0.962], excluding chance
(0.5) in every resample.

VERDICT: the qualitative claim (source detection separates actuators from
sensors, well above chance) SURVIVES properly specified inference. The
channel-level p=1e-4 does not survive and is downgraded to descriptive.
Fixed in paper/mace_v2.tex: abstract now cites the run-clustered CI instead
of the invalid p-value; Table 8's p-value column is relabelled to state its
actual scope; the prose after the table adds the clustering diagnostic, the
within-run check, and the correct bootstrap CI. Added to the audit gate
(scripts/mace_v2_numbers.py), 38/38 passing.

Rule added:

101. When channels are scored within runs (or subjects, or any grouping
     variable), the run is the unit of independence for inference, not the
     channel, however many channels there are. A permutation test that
     shuffles channel labels without respecting that grouping is invalid
     regardless of how small the resulting p-value is - check the
     between-group/within-group variance ratio before trusting it, and use
     a cluster bootstrap or permute at the group level instead.

2026-08-31  CCM/PCMCI HEAD-TO-HEAD: H1 FAILS, CCM ties MACE at V=30.
Pre-registration: paper/ccm_pcmci_baseline_protocol.md at d16876a, committed
before any score. Closes the adversarial review's finding 6 (no baseline is
ever compared to MACE directly, only to other baselines).

V=30, n=4000, coupling 0.20, seed 0 - the boundary map's own centre cell, so
ground truth is byte-identical to MACE's published Table 4 row. CCM 18.8 min
(435 pairs), PCMCI 3.7 min.

  method   membership AUROC   true-edge AUROC
  CCM           1.000              1.000
  PCMCI         0.728              0.984
  MACE          1.000              n/a

H1 FAILED AS DECLARED: MACE does not exceed both baselines. CCM ties it at
1.000. Reported as it fell.

TWO DEFECTS IN OUR OWN SCORING, found because PCMCI first read 0.008 -
near-perfect INVERSION, which is a transposed matrix, not a failing method:
  1. TRANSPOSE BUG. tigramite's val_matrix[i,j,tau] is dependence of j on i
     at lag -tau, i.e. ALREADY i->j. chamber_detect.py returns
     val.max(axis=2) with no transpose and is correct; the new script added
     a .T and inverted every PCMCI edge. Corrected: 0.728, not 0.008. CCM
     unaffected (direction checked against deepfeatselect.ccm's documented
     semantics before the run).
  2. H3 DILUTED TO USELESSNESS. As declared it compared ALL 125
     source-to-driven pairs against non-edges, but only 25 are true edges
     (one parent per driven channel), so the positive set was 80% non-edges
     and H3 was pinned near 0.5 by construction - it read 0.578/0.479 and
     could not have detected anything. Replaced with the true-edge version,
     recovering parent[] by replaying make_system's rng draws in order.
     Both baselines then pass comfortably.
Rescored in scripts/ccm_pcmci_rescore.py; no re-run needed, matrices saved.

WHAT IT MEANS, against our own interest: at V=30, where pairwise CCM is
still affordable, CCM is not worse than MACE at the membership task. This is
what the paper's own method-selection map already says, now measured rather
than asserted. MACE's claim was never accuracy at widths a pairwise method
can reach.

WHAT IT DOES NOT ESTABLISH: one cell, one seed, at the easiest width in the
grid, where MACE also scores 1.000 and has no headroom to be beaten. A tie
at ceiling is not equivalence. The informative comparison is V=60-240 where
MACE's recall falls to 0.14-0.23, and that is exactly where CCM becomes
unaffordable (65+ min/seed at V=60, quadratic). Not run.

SCALING CLAIM RE-MEASURED INDEPENDENTLY: 435 pairs in 18.8 min at n=4000
extrapolates to ~30,800 compute-hours at V=10^4, consistent with the paper's
15,800-hour figure measured at half the sample length.

Added to paper/mace_v2.tex as sec:map:headtohead, and to the audit gate:
43/43 passing.

Rule added:

102. A near-zero AUROC is a transpose, not a result. An inverted score matrix
     produces near-perfect anti-separation, which looks like catastrophic
     method failure and is actually an indexing bug. Check the orientation
     against a known-correct implementation before reporting any baseline as
     having failed.

2026-09-02  V=60 HEAD-TO-HEAD: H1 HOLDS. The V=30 tie was a ceiling effect.
Pre-registration: paper/ccm_pcmci_v60_protocol.md at ea29539, before any
score at this width. Three seeds, one process each. CCM 249 min, PCMCI 85
min, MACE 0 (re-derived from published boundary-map cells).

  MEMBERSHIP AUROC   s0     s1     s2     median
  MACE              0.918  1.000  0.976  0.976
  CCM               0.678  0.674  0.490  0.674
  PCMCI             0.590  0.604  0.502  0.590
  true-edge median: CCM 0.984, PCMCI 0.950 - both function, H3 excludes
  neither.

H1 HOLDS: MACE leads CCM by 0.30 at the width where MACE's own recall has
collapsed to 0.14-0.18. Neither at ceiling, so the comparison discriminates,
unlike V=30 where both scored 1.000.

THE MECHANISM IS AGGREGATION, NOT EDGE RECOVERY. CCM's true-edge AUROC is
0.982-0.995 - it recovers parent-child edges essentially perfectly and does
NOT degrade at its own job when V doubles. What collapses is edges ->
membership: a channel's score is its max incoming edge, so at V=60 each
source has twice the chances of V=30 to collect one spurious high incoming
score. At seed 2 CCM sits at 0.490 (chance) on membership while its
true-edge AUROC on the same matrix is 0.982. MACE scores membership directly
and never pays this. This is exactly the mechanism H4 named and DECLINED to
predict, because predicting it after seeing the V=30 tie would have been
fitting the story - recorded now as measured, not predicted.

READ AGAINST OUR OWN INTEREST, both stated in the paper:
  - The aggregation is OURS, not CCM's. Max-over-incoming is the natural
    reduction and is what chamber_detect.py already used, but a count above
    a per-scan threshold or a calibrated per-source null might degrade more
    gracefully. Established: pairwise edge recovery does not convert to
    membership at scale UNDER THE OBVIOUS AGGREGATION. Not: CCM cannot do
    this.
  - MACE's 0.976 AUROC coexists with recall 0.14-0.18 on the identical
    cells. It ranks driven channels well while its threshold rule surfaces
    few. That gap is the capacity limit of the bottleneck section, not a
    ranking failure.

SCALING RE-MEASURED: CCM 81-87 min/seed at V=60 vs 18.8 at V=30 - 4.5x for
2x width, as the quadratic pair count implies. PCMCI 3.7 -> 24-32 min,
roughly 8x, worse than quadratic, consistent with its combinatorial parent
search.

MEMORY: the first V=60 attempt OOM'd on a 1 MB allocation and discarded 90
min of completed CCM. Fixed (dc3c2fd): worker baseline cut 690 -> 89 MB by
pre-generating systems in a throwaway process, loading ccm.py by path to
bypass the deepfeatselect __init__ that pulls keras, and inlining auc()
rather than importing error_metrics (which pulls torch). Peak per seed then
508-635 MB and all three completed.

Rules added:

103. A method can be excellent at its own task and useless at yours. CCM
     recovered true edges at AUROC 0.98+ while its membership score sat at
     chance, because the reduction from edges to a per-node score amplifies
     one spurious edge per node as the panel grows. When adapting a method
     across tasks, the adapter is a design choice you own and must report as
     yours, not a property of the method you are comparing against.

104. Measure the run's own memory before blaming the machine. A crash on a
     1 MB allocation looked like external pressure; 564 MB of the process's
     690 MB baseline turned out to be torch and tensorflow that the
     computation never used. Import cost is a real experimental parameter on
     a loaded machine.

2026-09-02  AGGREGATION CHECK: the caveat is discharged, the adapter is not
the problem. Pre-registration: paper/aggregation_check_protocol.md, before
any alternative was scored. Reused the saved V=60/V=30 matrices; minutes,
not the 334 min the edge scoring took.

  V=60 median over 3 seeds    MAX    MEAN   COUNT  TOP2MEAN
  CCM                        0.674  0.680  0.590   0.658
  PCMCI                      0.590  0.444  0.400   0.548
  MACE 0.976

G2 HOLDS: no alternative approaches MACE. Best is MEAN at 0.680, still 0.30
below, and only 0.006 above the incumbent.

G1 technically holds and is substantively empty - 0.006 over three seeds is
noise. Recorded as such rather than as a win, and the REASONING behind G1
was wrong even though its letter was met: the prediction was that MAX would
be weakest because it is most exposed to the single-spurious-edge mechanism,
but MAX ties MEAN and beats both COUNT and TOP2MEAN.

G4 satisfied: MEAN ties MAX at 1.000 at V=30, so no width-specific tuning.

CONSEQUENCE: the V=60 section's hedge ("a different aggregation might
degrade more gracefully") becomes a measurement. Three declared alternatives
tried, conversion still fails, gap to MACE 0.30 under all four.

NOT SETTLED, and stated: a per-source calibrated null is a fifth family not
declared and not run, because adding it after seeing these numbers is the
shopping the protocol forbids. Open door; the matrices are released.
