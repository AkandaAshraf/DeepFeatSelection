# Why masked models and difference-based importance fail at causal discovery on redundant systems

**Akanda Ashraf** — preprint draft v1.0, 2026-08-15. Target: arXiv (cs.LG / stat.ML).

## Abstract

We analyse three failure mechanisms that defeat learned estimators of causal
structure on dynamical systems, each diagnosed on synthetic systems with
known truth and then confirmed on real data, and we derive the correction
that follows from the diagnosis. **(1) Maturity collapse.** Difference-based
importance (leave-one-covariate-out and relatives) measures the loss of a
trained model when one input is removed. On systems where a cause's imprint
is redundant across several effects — which Takens' theorem makes generic
for dynamical systems, since any single effect's delay embedding
reconstructs its driver — the difference collapses to zero *precisely when
training succeeds*: a mature model re-routes around any single removed
input. We demonstrate the signature that separates this mechanism from
overfitting (redundant-parent scores collapse with training epochs while
unique-parent scores grow monotonically) and observe the collapse at full
strength on real neural data, where mature leave-one-out gains are exactly
zero across dense connectome redundancy. Difference-based importance
therefore works *because of* restricted capacity or truncated training, not
despite them — a training-time restatement of the known dependence of
feature-importance gaps on function class. **(2) Typicality substitution.**
Masked-autoencoder recreation of a held-out channel, an attractive
foundation-model-style readout for "is this variable coupled to the
system?", degenerates at scale into a *predictability* meter: on a
1,000-channel benchmark the recreation score ranks periodic autonomous
channels highest (0.76) and genuinely coupled chaotic channels lowest
(0.25), inverting the causal ordering, because the decoder's optimal
response to a masked slot is the family-typical trajectory. **(3) Shared
bias defeats ensembling.** Averaging over independently initialised models
removes optimisation variance exactly as the law of large numbers predicts
(a rising consensus curve) — and converges to signal *plus shared bias*: the
typicality artifact is near-identical across eight independent minima
(cross-model σ = 0.004 on the control channel), so no ensemble size
rescues the readout; bias, not variance, is the binding constraint. The
correction implied by all three mechanisms is to score **excess over
self-predictability** — the gain from the system's compressed state beyond a
flexible model of the variable's own history, a Granger/transfer-entropy
quantity amortised by one unsupervised encoder — which cancels the
typicality term by construction, is immune to maturity collapse (it is not
a difference across redundant alternatives), and turns chance-level
detection into 30/30 top-10 precision on the same benchmark, with the
function-class of the self-baseline as the single load-bearing choice. We
package the operational protocol that caught every failure here: embedded
ghost channels as per-dataset falsification tests, training-trajectory
checkpointing, separation-based verdicts, and precision-recall reporting
against stated baselines.

## 1. Introduction

Learned models are increasingly used to extract causal structure from
multivariate time series: ablate an input and measure the loss; mask a
channel and measure its recreation; train a forecaster and read its
attributions. These readouts inherit failure modes that are invisible in
aggregate benchmarks and fatal in deployment. This paper isolates three of
them with controlled experiments, traces each to a mechanism, and shows the
correction. Throughout, systems are coupled logistic-map networks (with
heterogeneous and stochastic variants), plus real neural recordings for
confirmation; all thresholds and orientations were fixed before results
were inspected, and every analysis embeds a *ghost channel* — a circularly
shifted copy of a real variable — whose score must be zero, as a built-in
falsification test.

## 2. Maturity collapse of difference-based importance

**Mechanism.** Let a target variable have parent P whose influence also
reaches k−1 sibling effects. Takens' theorem implies each sibling's delay
embedding generically suffices to reconstruct P's state, so the information
carried by any single input channel is redundant. A model of limited
capacity or training extracts the most accessible route; removing that
input then costs measurable loss. A *mature* model has learned the
alternative routes, and removing any single input costs nothing: the
leave-one-out difference is zero exactly when learning has succeeded. This
is the function-class dependence of importance gaps [cf. V-information, Xu
et al. 2020] expressed along the training trajectory.

**Signature.** The prediction that separates this mechanism from
overfitting: scores for inputs with redundant imprints collapse as training
proceeds, while scores for inputs with *unique* imprints grow. On a
hub-structured network (one parent with six children, two single-child
parents), checkpointing every leave-one-out model across its own training
trajectory shows exactly the crossing: hub-parent gains rise, peak at
partial maturity (epoch ~10), and fall to zero by epoch 50, while
single-child gains grow monotonically throughout (0.02 → 0.28). Uniform
degradation — the overfitting signature — is absent.

**Real-data confirmation.** On C. elegans whole-brain recordings scored
against the anatomical connectome, mature leave-one-out gains are *exactly*
zero (AUROC 0.500 to three decimals, average precision pinned at baseline)
across all animals tested: dense biological redundancy realises the
mechanism at full strength, and no intermediate checkpoint rescues it — the
window closes from the noise side before it opens from the redundancy side.

**Consequences.** (i) Reported importance from a fully trained model is
uninterpretable on redundant systems, and most interesting systems are
redundant. (ii) If difference-based scores must be used, training-trajectory
checkpointing is mandatory, and the operative regime is *partial* maturity —
with the caveat that on sufficiently dense systems no usable window exists.

## 3. Typicality substitution in masked-recreation readouts

**Mechanism.** Score channel X by masking it at the input of a trained
masked autoencoder and measuring recreation R² of X at the output — a
natural "foundation-model" probe of coupling. At small V this weakly works.
At V = 1,000 (100 coupled members among 900 autonomous channels) it inverts:
periodic autonomous channels score 0.76, a shifted-copy ghost 0.74, chaotic
autonomous channels 0.29, and true members 0.25. The decoder's
loss-optimal output for a masked slot is the *family-typical* trajectory;
recreation therefore measures how typical a channel is, not how coupled —
and coupled channels, perturbed off the family manifold by their drive, are
the *least* typical. Width of the bottleneck (8→256), fill values
(zero vs noise), and masked-only loss all fail to repair the inversion,
because none changes what the optimal blind completion is.

**Warning.** Any pipeline that ranks variables by masked-recreation quality
— including repurposed pretrained masked models — inherits this confound on
pools with family structure, and large homogeneous panels (grids, sensor
fleets, transcript families) are exactly such pools.

## 4. Ensembles average variance, not bias

The natural rescue — train many models from independent initialisations and
trust the consensus — was tested directly: eight masked autoencoders on one
fixed system, identical data, only the seed varying. The law of large
numbers behaves exactly as advertised: the consensus average-precision
curve rises with ensemble size (0.111 → 0.148) with the 1/√M deceleration
profile. It converges to the wrong place. The ghost's recreation score is
0.740 ± 0.004 *across the eight independent minima*: the typicality
artifact is a property of the data-plus-objective, reproduced in every
basin, and averaging over minima cannot touch it. Ensemble consensus is a
variance instrument; these failures are bias.

## 5. The correction: excess over self-predictability

All three mechanisms point at the same repair. Score membership as

    excess(q) = R²[ x_q(t+1) | own-history features ⊕ code ]
              − R²[ x_q(t+1) | own-history features ]

where the code is the ensemble encoder's compressed state of the whole
system. The subtraction cancels typicality (a channel predictable from its
own structure gains nothing from the code); the statistic is not a
difference across redundant alternatives, so maturity collapse does not
apply (empirically: stable from 5 to 100 training epochs); and its lineage
is explicit — a Granger/transfer-entropy-type quantity [Granger 1969;
Schreiber 2000] amortised to thousands of variables by unsupervised
compression.

One choice is load-bearing: the **self-model's function class**. With a
linear self-baseline the correction *fails* (a linear fit of a quadratic
map leaves residual self-structure that the code then "explains", scoring
periodic channels +0.26); with degree-3 polynomial features the
self-baselines match theory to three decimals and the confound is cancelled
rather than suppressed — autonomous channels of four different dynamical
families, including stochastic AR(1), pin at zero excess while true members
surface (top-10 precision 10/10 on three independent systems; 16–20/20 at
top-20). The residual semantics are stated in the companion applications
paper: drivenness only, top-k precision, ≥~1,500 samples.

Two further measured facts complete the picture. The learned encoder is
*necessary*: the same excess readout conditioning on all raw dimensions
(regularised) fails, AP 0.10 vs 0.24 — compression is what makes a
thousand-variable system legible to an honest low-variance readout. And the
readout transfers to real systems across four domains (neural imaging with
interventional confirmation, whole-brain zebrafish, clinical EEG, 77 years
of atmospheric reanalysis), with the ghost and known-exogenous channels
(silenced neurons, solar forcing) scoring zero in every deployment.

## 6. Operational protocol

The practices that caught every failure in this paper, offered as a
checklist: (1) embed a ghost channel in every analysis and treat any
nonzero ghost as invalidating; (2) include known-exogenous variables where
they exist — they must score zero; (3) checkpoint readouts across the
training trajectory rather than reporting one maturity; (4) declare
statistics, orientations and thresholds before inspecting results, and
report separation (worst positive vs best control) rather than mean
differences; (5) report precision-recall against stated baselines — with
class imbalance, AUROC conceals both top-k success and bulk failure, and it
reversed a conclusion once in this work; (6) difference or deseasonalise
nonstationary series first — circular-shift surrogates are invalid on
drifting data, and the ghost will (correctly) expose this; (7) publish the
negative results: every mechanism here was found by an experiment that
failed.

## 7. Related work

Granger causality [1969] and transfer entropy [Schreiber 2000] define the
conditional-predictability quantity we estimate; neural estimators [Tank et
al. 2021 and successors] demonstrate tens of variables where the amortised
form here runs at 10³–10⁵. Convergent cross mapping [Sugihara et al. 2012]
implicitly performs a self-baseline via convergence-with-library-size;
our excess statistic makes that control explicit and scalable. The
function-class dependence of importance echoes V-information [Xu et al.
2020]; the maturity collapse is its training-time expression. Masked
autoencoding [He et al. 2022] supplies the representation machinery whose
naive causal readout §3 cautions against. Benchmarks with physical ground
truth [Gamella et al. 2025] and constraint-based discovery [Runge et al.
2019] frame the method-selection map in the companion paper.

## 8. Limitations

The mechanisms are demonstrated on map-based synthetic families plus real
neural confirmation; flow-based and noise-dominated regimes were probed
more narrowly. The excess correction's dependence on the self-model's
function class is characterised but not theoretically bounded; on real data
the practical guards are the ghost and known-root channels. The typicality
analysis concerns recreation *readouts*; masked pretraining as
representation learning is untouched by it.

## References

Gamella, J.L., Peters, J., Bühlmann, P. (2025). Causal chambers as a real-world physical testbed for AI methodology. *Nat. Mach. Intell.*
Granger, C.W.J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica* 37.
He, K. et al. (2022). Masked autoencoders are scalable vision learners. *CVPR*.
Runge, J. et al. (2019). Detecting and quantifying causal associations in large nonlinear time series datasets. *Sci. Adv.* 5.
Schreiber, T. (2000). Measuring information transfer. *Phys. Rev. Lett.* 85.
Sugihara, G. et al. (2012). Detecting causality in complex ecosystems. *Science* 338.
Takens, F. (1981). Detecting strange attractors in turbulence. *Lecture Notes in Mathematics* 898.
Tank, A. et al. (2021). Neural Granger causality. *IEEE TPAMI* 44.
Xu, Y. et al. (2020). A theory of usable information under computational constraints. *ICLR*.
