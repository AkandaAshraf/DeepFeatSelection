# Pre-registration: bounded-code parent screening at a fixed candidate budget

Declared 2026-09-20, before scripts/parent_screening.py or any generator
file for this experiment exists. Executes a local planning brief
(planning-only, itself not a
registered protocol) on the user's explicit instruction to proceed. That
brief is not evidence of anything; every choice it left open is resolved
below, and this document, not the brief, is what governs the run.

## What this is and is not

The intended output is a per-target SHORTLIST of candidate parent
variables at a declared budget, handed to a downstream causal-discovery
method. It is not a causal graph, not an intervention recommendation, not
a calibrated p-value, and not a source detector. Success is measured as
retained-parent recall and complete-target coverage at a fixed candidate
fraction, against beatable cheap baselines, not against a causal-discovery
guarantee.

## Literature positioning, checked before writing this

Fetched primary sources directly (not taken from the brief's summary
alone) for the two most load-bearing comparisons:

- Allione, Del Tatto and Laio, arXiv:2501.10886v3. CONFIRMED by fetching
  the abstract: "dynamical community detection," optimises Information
  Imbalance, demonstrates up to 80 variables, evaluates against COMMUNITY
  causal-graph recovery, not individual-parent recovery. The abstract does
  not address causal sufficiency or noise robustness; not claimed here.
- Wengel Mogensen, UAI 2020. CONFIRMED: inexpensive causal screening under
  ancestral faithfulness, combines theory with an empirical linear-Hawkes
  application. Screening before a fuller method is its explicit framing,
  same target as this protocol, at the graphical/point-process level
  rather than the group-code level.
- Gao et al., AISTATS 2026 (v300). CONFIRMED: a FULL discovery algorithm
  that exploits stationarity-constrained minimal separating sets, not a
  pre-screen. Related but not competing with this protocol's task.
- Cheng et al. (CUTS+), AAAI 2024. PARTIALLY CONFIRMED from the abstract
  page: targets high-dimensional, irregularly-sampled time series via
  coarse-to-fine discovery and a message-passing GNN. Variable counts and
  whether its output is pairwise or coarser were NOT resolved from the
  abstract; NOT claimed here, and Section 3's comparison requirement is
  scoped accordingly (see below).
- Runge et al., Science Advances 2019 (PCMCI). Not re-fetched; this
  project already runs the official Tigramite implementation elsewhere
  (scripts/chamber_detect.py, scripts/ccm_pcmci_v60.py) and PCMCI is used
  downstream in Stage C's utility test, not re-litigated as a citation.

OVERLAP TABLE

| Method | Reduces search space before discovery | Individual-parent output | Demonstrated width | Verified how |
|---|---|---|---|---|
| This protocol | Yes, by design | Yes, primary metric | Target 500-1000 | -- |
| Community detection (2501.10886) | Yes | No (community-level) | <=80 | fetched abstract |
| Causal screening (Wengel Mogensen) | Yes | Graph-level, not group-code | Not stated in abstract | fetched abstract |
| Minimal separating sets (Gao 2026) | No, full algorithm | Yes | Not stated in abstract | fetched abstract |
| CUTS+ | Partially (coarse-to-fine) | Unresolved | Unresolved | abstract page only |
| PCMCI (Runge 2019) | No, full algorithm | Yes | Used downstream here | prior use in repo |

CONCLUSION: the closest prior work by task (individual-parent shortlist
before a discovery method) is community detection and Wengel Mogensen's
screening, and both differ from this protocol's evaluation regime -- group
output rather than individual, or graph-level rather than a bounded-width
learned code, respectively -- at a width this protocol tests (500-1000)
that the confirmed community-detection ceiling (80) does not reach. This
does NOT establish novelty; it establishes that the exact mechanism and
evaluation here were not found already covered by the sources checked.
Per Section 3, at least one modern method (community detection, since its
code uses DADApy and is independently installable) gets a fair comparison
before any state-of-the-art claim; CUTS+ is comparison-optional and its
absence, if it occurs, is reported as a missing baseline, not hidden.

## Two generator families, full equations

Both families share ONE graph-construction procedure so that only the
DYNAMICS differ between them; this is a deliberate design choice, stated
because the brief did not require it but it removes a confound between
"which family" and "which topology."

### Shared topology construction

Given V and seed:

1. Draw a uniform random permutation of {0,...,V-1}; this permutation IS
   the topological order. n_root = max(3, round(0.12*V)) variables at the
   START of that order are roots (no parents, by construction -- there are
   no earlier-order variables for them to draw from in the star/chain
   sense below, and root status is also asserted directly, not merely
   implied by position).
2. For each of the V - n_root non-root variables, in topological order:
   draw indegree d ~ Uniform{1,2,3}; draw d DISTINCT parents uniformly at
   random from all variables strictly EARLIER in topological order (this
   allows a root or a non-root as a parent, which is what produces chains
   and common drivers rather than only source-to-sink stars); draw one
   delay per parent edge, independently, from Uniform{1,2,3}.
3. ORPHAN REPAIR, deterministic, no randomness: for every root with zero
   children after step 2, take the variable immediately following it in
   topological order and add the orphan root as one of its parents,
   replacing that variable's highest-index existing parent if it is
   already at indegree 3. Re-check; this repair is a single pass and is
   sufficient because at most n_root roots can be orphaned and each repair
   consumes one later variable's parent slot without creating a cycle
   (topological order is preserved: the repair only ever adds an EARLIER
   variable as a parent of a LATER one).
4. Record the realised parent map Pa(q) for every non-root q, the delay
   for every edge, and n_root, before any dynamics are simulated.

At V=240: n_root=29. At V=500: n_root=60. At V=1000: n_root=120. Mean
non-root indegree is 2 by construction (Uniform{1,2,3}).

### Family 1: corrected sparse coupled logistic-map family

Extends scripts/clean_generator.py's audited construction (single driver,
star topology) to the shared multi-parent DAG above, and extends its
phase-lock rejection from SOURCES only to EVERY channel: a phase-locked
non-root channel is exactly as undetectable as a phase-locked source, and
restricting the gate to sources was never justified by the mechanism, only
by the star topology that made every non-root a pure sink. This is a
deliberate strengthening of Rule 111 ("gate the generator with the paper's
own gate"), stated because it goes beyond what the historical generator
did, not because the brief asked for it verbatim.

For every channel i, draw r_i ~ U(3.6, 3.9), REJECTED and redrawn via
generator_audit.is_locked_r(r_i) (imported, not reimplemented) until an
unlocked value is found. For every edge (j -> i, delay d_ij), draw a
jitter multiplier eta_ij ~ U(0.85, 1.15). Fix coupling c = 0.20 (the
centre value used throughout this project's boundary_map and large_system
work, kept for comparability, not re-tuned here).

  root i:      x_i(t+1) = clip( r_i * x_i(t) * (1 - x_i(t)),  0, 1 )
  non-root i:  k = x_i(t)
               drive = mean_{j in Pa(i)} [ eta_ij * x_j(t - d_ij) ]
               x_i(t+1) = clip( r_i*k*(1-k)*(1-c) + c*drive*(1-k),  0, 1 )

This is scripts/clean_generator.py's update rule generalised from a single
parent at fixed delay to a mean over 1-3 parents at per-edge delays; the
functional form (r*k*(1-k), the (1-c)/(1-k) coupling shape) is unchanged
from the already-audited mechanism.

Burn-in: max delay is 3, so simulate 500 + n steps from
x(0) ~ U(0.2, 0.8)^V and discard the first 500. n = 4000 kept observations.

VALIDITY TEST, checked after simulation, before any encoder sees the data:
  (a) no NaN/Inf anywhere in the kept trajectory;
  (b) per-channel boundary-clip fraction (samples exactly at 0 or 1) does
      not exceed 0.30. This is the exact defect Rule 116's disclosure
      named at coupling 0.50 ("83% of sinks spend >30% of steps at the
      clip boundary"); here it is a REJECT-AND-REDRAW gate on the whole
      seed, not a disclosed-but-kept artefact, capped at 20 redraws before
      declaring that seed infeasible;
  (c) every root has >=1 child (assert on the repaired graph, should
      always hold; a failure here is a bug, not a data property);
  (d) near-synchrony recorded, not gated: for every edge, |corr(x_j(t-d),
      x_i(t))| over the train block; edges above 0.98 are logged by count
      and by which targets they touch, since Proposition 1's synchrony
      collapse is a real, disclosed limit of the underlying theory and
      silently omitting its incidence here would misrepresent how often
      the primary metric is being tested against synchronised pairs.

Observation noise, added AFTER the validity test, at the declared primary
level: for channel i, sigma_i = 0.05 * std(clean_train_block[:, i]), and
x_obs = x_clean + N(0, sigma_i^2) i.i.d. over time.

### Family 2: stable sparse nonlinear autoregressive family

Same graph construction as Family 1. Self-dynamics are heterogeneous
stable AR(2) with mixed signs; parent influence is an additive tanh
nonlinearity by default, with a designated subset using a multiplicative
(synergy) mechanism instead (Section on stress panels).

Self-dynamics: draw (a_i, b_i) ~ U(-0.6,0.6) x U(-0.3,0.3); accept only if
|a_i|+|b_i| < 0.9 AND both roots of z^2 - a_i*z - b_i have modulus < 1
(checked directly via numpy.roots); redraw otherwise. This is a standard
sufficient stability condition for a linear AR(2) recursion, checked by
direct root computation rather than asserted from the coefficient bounds
alone.

Parent term (additive targets): gamma_i ~ U(0.4, 0.8); for each edge,
weight w_ij ~ U(0.3, 0.7); drive_i(t) = mean_{j in Pa(i)} [ w_ij *
tanh(x_j(t - d_ij)) ].

  root i:      x_i(t+1) = a_i*x_i(t) + b_i*x_i(t-1) + eps_i(t)
  non-root i:  x_i(t+1) = a_i*x_i(t) + b_i*x_i(t-1)
                        + gamma_i * drive_i(t) + eps_i(t)

Innovation noise eps_i(t) ~ N(0, sigma_i^2) i.i.d. over time, independent
across channels; sigma_i = 0.7 for roots (their only variance source),
0.5 for non-roots (declared, not tuned to any target self-predictability
level).

Burn-in and kept length identical to Family 1: 500 discarded, n=4000 kept,
x(0) ~ N(0, 1)^V.

VALIDITY TEST: (a) no NaN/Inf and |x_i(t)| < 50 at every step (reject and
redraw the seed otherwise, cap 20 redraws); (b) post-burn-in per-channel
variance finite and within [1e-4, 100]; (c) orphan check as Family 1; (d)
the empirical self-predictability distribution (own-lag-only ridge R2 per
channel, using the SAME ridge procedure Section "The screening method"
defines) is RECORDED, not asserted matched to Family 1 merely because
parameter ranges overlap -- per the brief's explicit instruction that
matching parameter ranges is not the same claim as matching self-
predictability, and both families' distributions are reported side by
side in the pilot output.

Observation noise: identical 0.05 * train-std convention as Family 1.

## Screening method: bounded-code parent screening

### Partition, deterministic, size-capped, train-only

Adapts scripts/hierarchy_repair.py's cluster_train_only (correlation
distance of first differences on TRAINING RAW ROWS ONLY) from a fixed
cluster COUNT to a deterministic MAXIMUM GROUP SIZE of 8, since the
brief's budget arithmetic (8-dim group code, 64-column regression cap)
assumes a size cap, not a count.

Procedure, frozen: compute d = diff(x_train, axis=0); c = corrcoef(d.T),
NaN to 0; distance = 1 - |c|. Run scipy.cluster.hierarchy.linkage with
average linkage on the condensed distance form. Process the linkage
matrix's merges IN THE ORDER GIVEN (ascending distance, linkage's own
tie-break for equal distances, which is deterministic given deterministic
input) and accept a merge only if the two current clusters' combined size
is <= 8; skip a merge that would exceed 8 and continue to the next one in
the list (a skipped merge's two clusters remain separate and may still
each merge with something else later). This terminates with every final
group of size 1-8, deterministic given x_train, with no cluster COUNT
declared in advance.

The same procedure with the SAME frozen distance definition and tie
ordering is applied for every arm that needs a partition (clustered and
size-matched random controls); nothing about clustering changes between
arms except which arm supplies the size multiset to match.

### Group encoder: one small masked autoencoder per group

Per group G (E=3 lags per member, as elsewhere in this repo): bottleneck
width b_G = min(8, 2 * |G|) (an 8-dimensional cap per the brief, floored
by twice the group's own member count so a 1-2 member group is not forced
to an oversized code). Architecture and masking identical to
wormwideweb_gate.MaskedAE and boundary_map's training loop (EPOCHS, BATCH,
MASK unchanged from boundary_map.py's constants), EXCEPT masking is
applied per-MEMBER rather than per-group: each training batch independently
zeroes each member's E columns with probability MASK, so that later
zeroing exactly member q's columns (to exclude q from its own group's code
when scoring target q) is in-distribution for the trained network, the
same argument scripts/resnet_shortcut.py already established and measured
(target exclusion moved AP by +0.008, negligible but the argument for WHY
it is safe to do carries over unchanged).

No global code, no attention, no shared decoder across groups: each
group's encoder and decoder are trained and used only for that group,
matching Rule 127 (amortise the encoder, never the readout) at the group
granularity instead of the whole-system granularity that rule was written
against.

### Readout: one ridge fit per (target, candidate group), own history plus that group's code

Own-history features: poly3(own 3 lags) as boundary_map.poly3 computes it
(19 features: 3 linear + 6 quadratic + 10 cubic). Regression input for one
(q, G) query is 19 (own) + up to 8 (G's code, or 0 if q's own group G_q is
being scored and |G_q|=1) <= 27 columns, well inside the declared 64-column
cap; the 64 figure is documented as a ceiling that this design does not
approach, not a binding constraint.

RIDGE WITH INTERCEPT AND TRAIN-INTERNAL VALIDATION-SELECTED PENALTY. The
archived boundary_map.ridge_r2 (alpha=1.0 fixed, no intercept) is NOT
reused for this experiment's readout; paper/ridge_alpha_scaling_protocol.md
already found that a fixed, unscaled penalty is a real and previously
undisclosed limitation, and Rule 127's amortise-the-encoder-not-the-readout
finding makes THIS readout's fidelity load-bearing in a way it was not for
the diagnostic that used the archived helper. New procedure, frozen:
add a constant column of ones to X (proper intercept, not centering);
carve the LAST 15% of the training block (temporal, not random) as an
internal selection slice; select alpha from the fixed grid {0.1, 1, 10,
100, 1000} by lowest squared error on that internal slice; refit at the
selected alpha on the FULL training block; report R2 on the SEPARATE
validation block (never the internal selection slice, never test) for
group scoring, and on the SEPARATE test block only for the final
confirmation-stage numbers. This grid is fixed now and is not widened
after seeing any result, per Rule 132's lesson that a correction is a
claim needing its own check: selected-alpha-at-grid-boundary rates are
recorded per arm per stage as a diagnostic, exactly as
ridge_alpha_scaling_protocol.md's result required for its own grid.

SCORING, UNCLIPPED: gain(q, G) = R2_val(own + code_G) - R2_val(own alone),
both R2 values computed WITHOUT clipping negative values before the
subtraction (R2 itself can be negative when the fit underperforms
predicting the validation mean; the SUBTRACTION uses the raw signed
values). NO NONNEGATIVE TRANSFORM IS APPLIED ANYWHERE in ranking or
selection; gain is used as a signed real number throughout, and stating
that no transform exists is the "specify separately" the brief asked for.

### Candidate set construction, k=ceil(0.10*(V-1))

For target q: rank ALL groups G (the whole system, not only q's own
group) by gain(q,G) descending; ties broken by ascending group id, frozen,
never touching truth. Walk the ranked list, adding EVERY member of each
group to C_q (deduplicated if q's own group is included, since q itself
is excluded from its own candidate set by definition) until the next
whole group would push |C_q| past k. For that boundary group only, add
its members in ascending ORIGINAL VARIABLE INDEX order until |C_q| = k
exactly, or until the group is exhausted (in which case |C_q| < k and no
further group is added, since the brief's budget is a ceiling, not a
target to force-fill from truth-blind padding beyond the ranked list).
Report k, k/(V-1) exactly (not asserted as 0.10), and the realised mean
|C_q| including any boundary truncation.

UNRESOLVED RULE, frozen before any result: if max_G gain(q,G) <= 0.01 (a
value fixed now, roughly twice the ~0.005 clean-ghost floor this project
has measured elsewhere as a noise-scale reference, not fitted to this
pilot's own distribution), q is UNRESOLVED for that arm. An unresolved
target's C_q is ALL V-1 other variables -- not excluded from any
denominator, not scored as a success, the maximally conservative fallback
a screen can honestly report when it found nothing.

### Cost accounting

This evaluates roughly V times the number of groups (V * (V/4 to V/8)
depending on realised group sizes), NOT linear in V. Wall-clock and peak
memory for clustering, every group's encoder training, every (target,
group) readout, and selection are all measured and reported; Stage A
measures this empirically before Stage B's time budget is trusted.

## Required baseline arms

Same k, same data, same allowed own-lags (3), same target set, same
partition-construction code where a partition is needed.

1. RANDOM: k candidates drawn uniformly without replacement from the
   other V-1 variables. Analytic chance level: E[recall] = k/(V-1) as
   stated in the brief's arithmetic below; no training.
2. MAX ABSOLUTE LAGGED CORRELATION, two variants -- raw
   max_d |corr(x_j(t-d), x_q(t))| over d in {1,2,3}, and a train-only
   RESIDUALISED version that first fits q's own-lag ridge (same procedure
   as the screening method's own-history term) and computes the same
   correlation against q's residual. The DEPLOYABLE variant is chosen by
   VALIDATION-block performance (whichever gives higher validation
   candidate-fraction-matched recall proxy: mean |correlation| rank
   overlap with the top-gain groups is not available pre-truth, so the
   proxy is each variant's own validation R2 improvement when its top-k
   candidates are added as a group to q's own-history fit; whichever is
   higher is the deployed baseline), frozen before test truth is opened,
   both variants reported.
3. SPARSE SPLINE/LAG DICTIONARY: for target q, a lasso-type sparse fit
   (scikit-learn Lasso, temporal validation for the penalty path, same
   frozen grid width as the ridge grid above, five points log-spaced) on
   a fixed univariate dictionary per candidate variable j: {x_j(t-1),
   x_j(t-2), x_j(t-3), spline-basis(x_j(t-1)) at 4 knots}, one such block
   per candidate j, all V-1 blocks concatenated; the k variables with the
   largest summed absolute selected coefficient magnitude across their
   block form the shortlist. This can fail on genuine interactions
   (reported separately via the pure-synergy stress panel, not hidden by
   omission here).
4. PCA CONTROL: identical partition to arm 6 below, but each group's code
   is PCA (fit on training rows only) at the SAME dimension b_G the
   learned encoder uses for that group, not a separately tuned dimension.
   A tie between this and arm 6 means the learned representation has not
   justified itself, stated as the brief requires.
5. SIZE-MATCHED RANDOM PARTITION: hierarchy_repair.sized_random's
   procedure (exact size multiset preserved, membership randomised)
   applied to the size-8-capped clustering's realised size list, with
   learned per-group encoders exactly as arm 6, so the only difference
   from arm 6 is whether group membership is informative.
6. LEARNED GROUP SCREEN, TRAIN-ONLY CLUSTERED (the candidate of interest):
   the full bounded-code method above.

At least one modern scalable method gets a fair comparison before any
state-of-the-art claim (Section "Literature positioning"). Community
detection (arXiv:2501.10886) is the primary candidate since its DADApy
dependency is independently installable without a large download; if
installation or a fair matched-setting comparison is infeasible within
Stage B/C's own time caps, the result is labelled an internal feasibility
result and the missing comparison is named, not silently dropped.

## Ground truth, split, and behavioral protection

Truth is Pa(q) as constructed above, held by the evaluator only; the
screen's API receives only the observed (noised) series and V. Own lags
are always available to every arm and are excluded from both the
candidate budget and the parent-recall metric. Roots are recorded
separately and cannot inflate recall through an empty parent set (a root
contributes 0 to both the recall numerator and denominator, and 1 to
complete-target coverage's numerator trivially -- reported, with roots'
contribution to coverage stated explicitly rather than silently pooled).

Splits: contiguous 60/20/20 train/validation/test on the KEPT n=4000
samples, with raw-observation support disjointness verified by direct
enumeration of the actual lag/forecast/preprocessing indices touched at
each seam, following Rule 132's lesson (verify the claim by enumeration,
not by trusting a formula matched to a different pipeline's convention)
-- scripts/parent_screening_split_test.py, modelled on
scripts/test_embargo_boundary.py's pattern.

CAUGHT WHILE IMPLEMENTING, corrected here rather than left standing: this
protocol's first draft stated the required embargo as
(E-1)*max_delay = 6, reasoning by analogy from a different pipeline's
differencing convention. That reasoning does not apply here -- a candidate
group's own multi-lag delay structure is a property of the GENERATOR, not
of the SCREENING METHOD's feature construction, which only ever reads a
fixed E-length window aligned the same way for every variable, own or
candidate. A row's true raw support is that E-length window plus the
single forward touch to its t+1 target: E+1 consecutive raw indices, and
enumeration (the split test's own oracle-verified search, not a second
formula) confirms the minimal sufficient embargo is exactly E=3, not 6.
The oracle in that test failed on its own first version too, for an
unrelated reason (the synthetic example it checked against had a
pre-existing gap that masked the very overlap it was meant to detect) --
fixed before trusting the oracle's verdict on anything else. The
implemented embargo is E, verified equal to the enumerated minimum, not
merely sufficient for it.

Hyperparameter selection (ridge alpha, lasso penalty) uses the training
block's OWN internal temporal holdout, never validation or test. The
validation block builds every shortlist. The test block is not touched by
partition, encoder, readout, or selection until after all of a stage's
shortlists are frozen; pilot truth is opened only after all pilot
predictions are saved, and confirmation truth only after all confirmation
shortlists are frozen. Perturbing test observations must leave
partitions, trained weights, selected alphas and every C_q unchanged --
checked directly as a Stage A correctness test (perturb test rows,
re-run scoring, assert byte-identical C_q for every target and arm).

## Success arithmetic, fixed before any run

For non-root target set Q, |Q| = V - n_root:

    retained-parent recall   = sum_q |C_q ^ Pa(q)| / sum_q |Pa(q)|
    complete-target coverage = mean_q  1[Pa(q) subset of C_q]
    candidate fraction       = sum_q |C_q| / (|Q| * (V-1))
    macro target recall      = mean_q ( |C_q ^ Pa(q)| / |Pa(q)| )

Random-shortlist analytic references: E[retained-parent recall] = k/(V-1);
for a target with exactly d distinct parents, P[complete coverage] =
C(V-1-d, k-d) / C(V-1, k), the standard hypergeometric form. CHECKED, not
assumed: the brief stated this as binom(k,d)/binom(V-1,d), which reads as
a different quantity but is numerically IDENTICAL to the form above at
every (V-1,k,d) this protocol uses (verified directly, six cases spanning
V=240 to V=1000, agreement to 1e-12) -- an instance of the general
hypergeometric identity C(n,k)C(k,d) = C(n,d)C(n-d,k-d). Recorded as a
cross-check that held, not a correction: the two forms are the same
number, and Stage A verifies this numerically once more, in code, before
either formula is trusted for the pilot's own (V,k) values.

Repeated lags from one parent: screening is by parent VARIABLE only; every
allowed lag (1-3) for a variable in C_q is handed to the downstream method
whole. Lag-specific edge recovery is a separate, secondary endpoint,
reported only if Stage C is reached.

## Practical bars, design requirements, not results

- >= 0.90 retained-parent recall at k/(V-1) ~ 0.10, separately at V=500 and
  V=1000, in EACH primary family.
- >= 0.80 complete-target coverage.
- >= +0.05 recall over EVERY required deployable baseline at the same
  budget, paired at the graph-seed level.
- Confirmation additionally requires: lower 95% graph-level bootstrap
  bound on recall > 0.90, and lower bound on every paired recall
  difference > 0.05's own bound > 0 -- resampled over whole graph-seed
  blocks, paired across arms and widths, not over channels (Rule 101).
  These are CONJUNCTIVE; failing any one fails the confirmation gate.
  A bootstrap interval is an empirical statement about this generator and
  these seeds, not a distribution-free coverage guarantee, and is reported
  as such.

## Resource caps, measured before finalising, stop rather than shrink on breach

Per Rule 124 (guards scoped per arm, not per run) and the ridge_alpha_
scaling precedent (measure the baseline before declaring a cap): Stage A
measures import and one-workload memory footprint before Stage B's caps
are treated as final. Provisional caps, to be confirmed or corrected as a
labelled amendment (never silently) once measured:

  GPU allocation          <= 6 GiB
  process-tree host RSS   <= 3 GiB (measured across child processes)
  free host RAM           >= 2 GiB
  new experiment disk     <= 100 MiB (ExpOutput/parent_screening/ only;
                           compact per-target scores, candidate lists,
                           config/seed hashes, selected hyperparameters,
                           runtime/memory measurements, atomic completion
                           manifests -- no raw arrays, no checkpoints)

`.agent-lock` claimed before any training, released after; the current
GPU has 953 MiB in use by something else at protocol time (verified via
nvidia-smi), leaving comfortable headroom under 6 GiB but recorded here so
a future reader knows the machine was not idle when this cap was set.
Breach: record which cap, at what value, at which cell; stop; report.
Does not retry with a smaller grid or fewer seeds.

## Staged plan

STAGE A -- protocol and feasibility only, no large training, engineering
seeds only (excluded from every scientific result):
  - random-shortlist chance-level formulas verified numerically against
    the corrected hypergeometric derivation above;
  - label orientation verified on a hand-built A->B->C chain (3 variables,
    known Pa) run through the full scoring pipeline;
  - group-size-cap clustering verified deterministic (same input, two
    runs, identical output) and every group <= 8;
  - size-matched random control verified to reproduce the exact size
    multiset;
  - split disjointness verified by enumeration (parent_screening_split_
    test.py);
  - test-block perturbation invariance verified (partitions/weights/alphas/
    C_q unchanged);
  - both generator families' validity tests run at V=30 (cheap) for 3
    engineering seeds each, confirming no infinite redraw loops and
    recording realised clip/near-synchrony/variance diagnostics;
  - import and one small (V=30) end-to-end workload's memory and wall-
    clock measured, informing Stage B's resource caps before they are
    trusted.

STAGE B -- preregistered pilot: V=240, n=4000, 6 independent graph seeds
per family (seeds drawn fresh and checked unused by any prior experiment
in this repository before being frozen), primary noise level only, all
required arms including community detection if feasible within this
stage's own cap. Maximum 1 hour total training/analysis. If Stage A's
measurements show that cap is infeasible for the full arm set, this
document is amended with a cost estimate before Stage B is launched, not
silently shrunk.

GATE (pilot bars decide investment; a pilot result is never itself
confirmation): proceed to Stage C only if, in BOTH primary families, arm 6
reaches pilot mean recall >= 0.90, complete-target coverage >= 0.80, and
beats every required baseline (arms 1-5, plus any feasible modern method)
by >= 0.05 recall at the same budget. If a baseline already meets the
task as well, that baseline is preferred and the learned line stops here.

STAGE C -- gated. Untouched V=500 and V=1000, 12 graph seeds per family
(48 primary system-cells), same architecture/score/split/budget/baselines
frozen from Stage B, no new sweeps. Stress panels (below) with their own
fixed seeds, not silently multiplying the primary run. Six hours maximum
for confirmation plus stress panels together; an infeasible estimate is
reported before launch, and a cap hit mid-run preserves all artefacts and
is reported resource-limited with no complete-study claim.

DOWNSTREAM UTILITY, only if Stage C's confirmation bars are met: PCMCI
with and without the screen, on a separately preregistered feasible
subset (V=60 and V=120, explicitly not deployment-scale evidence),
identical Tigramite conditional-independence test both arms, separate time
blocks for screening versus downstream testing, all targets, excluded
candidates treated as unknown/removed tests rather than known-absent
edges. One hour total cap, engineering-seed cost estimate required before
launch. Target: measured >=5x end-to-end speedup with <=0.02 absolute
edge-recall loss and <=0.02 absolute empirical false-discovery-proportion
increase versus unscreened PCMCI. An unfinished baseline run is censored,
not scored as a measured runtime.

## Stress panels, reported separately, never pooled with primary success

- DUPLICATED MEASUREMENTS: every root i gets one noisy duplicate x_i' =
  x_i + N(0, (0.02)^2) appended as an extra observed channel (not a
  parent-set member, not a source, excluded from both truth sets, matching
  boundary_map's existing redundancy convention). Applied to EVERY
  eligible root, not a subset. Reports individual-parent coverage (must
  contain the true original) and equivalence-set coverage (original OR
  duplicate) with separate denominators and separate candidate-cost
  accounting, since a duplicate satisfying equivalence coverage is not the
  same finding as identifying the interchangeable pair, and neither is a
  physical intervention target.
- HIDDEN COMMON DRIVER: the first 20% of roots (by topological order) are
  REMOVED from the observed dataset entirely after generation (their
  children keep whatever parent structure they had, now partly
  unobserved). Reports resulting shortlists and any spurious co-implication
  among a hidden driver's remaining observed children, as a named
  assumption violation, not as evidence about latent-confounder
  robustness.
- PURE SYNERGY: exactly 3 non-root Family-2 targets per seed (fixed count,
  not a fraction that could round to zero at small V) use the
  multiplicative parent mechanism (2 parents, w_i * tanh(x_j1)*tanh(x_j2),
  replacing the additive term) instead of the additive default, so neither
  parent has marginal signal by construction. Reported as the declared
  boundary of every arm that relies on marginal or additive signal
  (arms 2, 3, and the screening method's own per-group additive gain);
  a failure here is kept, not deleted, per the brief's explicit
  instruction.

No claim in this document extends observational prediction to
intervention validity in any of these three panels; masking a model input
is not a physical intervention, stated once here and not repeated as a
disclaimer at every mention below.

## Deliverables

Protocol (this file); scripts/parent_screening.py and
scripts/parent_screening_split_test.py, committed only after this
protocol; compact metrics under ExpOutput/parent_screening/; one
comparison plot; a short decision report appended to this file after each
stage, in the same append-only style as every other protocol in this
repository. Missing baselines and untested widths stated plainly, not
implied by silence.

Final verdict, one of exactly four, decided by the arithmetic above, not
by which reads best:

  1. USEFUL SHORTLIST WITHIN TESTED ASSUMPTIONS -- confirmation bars
     passed; downstream speed evidence stated separately from the
     screening result.
  2. CHEAP METHOD IS SUFFICIENT -- no arm-6 advantage survives; the
     winning cheap baseline is named and recommended instead.
  3. NOT USEFUL AT THIS BUDGET -- a valid run, practical bars failed;
     reported and preserved as a negative result.
  4. INCONCLUSIVE / RESOURCE-LIMITED -- invalid controls, insufficient
     precision, a missing required arm, or incomplete execution; no
     success claim of any kind.

This protocol does not end in a recommendation to widen the architecture
search; its purpose is to decide whether this representation earns a
place in a real workflow, not to motivate the next version of itself.

## Void conditions

Void if the graph-construction procedure, either family's equations, the
size-8 group cap, the ridge grid, the ridge intercept, k's formula, the
unresolved threshold (0.01), the resource caps, or any stage's seed list
change after any result -- pilot or confirmation -- from that stage is
seen; if test-block observations influence partition, encoder, readout,
selection, or the deployable-baseline choice in arm 2 before that stage's
truth is declared open; if a stress panel's seeds or count are adjusted
after seeing its own result; if Stage C is launched without Stage B's gate
having passed in BOTH families; or if any downstream PCMCI run is scored
on anything but the identical conditional-independence test and identical
data for both the screened and unscreened arm.
