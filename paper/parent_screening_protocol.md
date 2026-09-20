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
3. ORPHAN REPAIR, as implemented after two review-driven rewrites (the
   first draft of this step, preserved in the amendment section below,
   described a repair that was wrong): for every root with zero children
   after step 2, choose as heir the FIRST NON-ROOT variable, searched in
   topological order from a rotating offset, that has indegree < 3, and
   append the orphan root to that heir's parent list with a freshly drawn
   delay from Uniform{1,2,3}. Roots are NEVER eligible heirs (both
   generators' root branches ignore parent[root] entirely, so an edge into
   a root would exist in the recorded map and not in the simulated data).
   No existing edge is ever displaced, so no child-count decrement is ever
   needed, and no variable's indegree ever exceeds 3, preserving the
   registered 1-3 indegree range without an exception. Topological order
   is preserved: the repair only ever adds an EARLIER-ordered root as a
   parent of a LATER-ordered non-root. If no non-root variable has spare
   indegree capacity anywhere, the seed is redrawn (the orphan_free
   validity gate fails), not patched.
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
               drive = mean_{j in Pa(i)} [ eta_ij * x_j((t+1) - d_ij) ]
               x_i(t+1) = clip( r_i*k*(1-k)*(1-c) + c*drive*(1-k),  0, 1 )

CORRECTED, caught by review: the first draft wrote drive as
x_j(t - d_ij), delay measured from the OWN-lag time t. The implemented
and now EMPIRICALLY VERIFIED convention (scripts/parent_screening.py's
[3c] lag-impulse test, an impulse on the parent at a known raw index,
checked against which raw index the effect on x_i(t+1) traces back to,
for d=1,2,3) is delay measured from the TARGET time t+1: x_j((t+1)-d_ij),
the standard lag convention in this literature (a lag-d edge predicts the
TARGET using a d-step-back value, not a (d+1)-step-back one). d=1 under
this convention reads x_j(t), the same raw time as i's own most recent
lag -- contemporaneous with i's own last observation, not one step
further back. Text corrected to match the verified implementation, not
the other way around.

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
tanh(x_j((t+1) - d_ij)) ], the SAME target-relative lag convention as
Family 1, verified for both families by the [3c] production-path
lag-impulse test (this line was missed in the first correction and still
read x_j(t - d_ij), caught by review).

  root i:      x_i(t+1) = a_i*x_i(t) + b_i*x_i(t-1) + eps_i(t)
  non-root i:  x_i(t+1) = a_i*x_i(t) + b_i*x_i(t-1)
                        + gamma_i * drive_i(t) + eps_i(t)

Innovation noise eps_i(t) ~ N(0, sigma_i^2) i.i.d. over time, independent
across channels; sigma_i ~ Uniform(0.4, 0.6), drawn independently for
EVERY channel from the same distribution, whether root or non-root.

CORRECTED, disclosed before any pilot ran: this paragraph first set
sigma_i = 0.7 for roots and 0.5 for non-roots, a value that depended
directly on root/non-root status. The brief this protocol executes says
noise distributions must not directly encode that status, and a fixed
0.7-versus-0.5 split does exactly that: a screening method could partly
succeed by reading channel noise SCALE as a root/non-root signature
rather than by reading dependence structure at all. Caught by review of
the Stage A implementation, fixed in code and here before any pilot seed
was run, so no scientific result was produced under the role-encoding
version and none needs to be set aside.

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
average linkage on the condensed distance form, producing the FULL
dendrogram: every merge node, unconditionally, in scipy's own numbering.
Then CUT the tree top-down: starting from the root, descend into a node's
two children whenever that node's subtree exceeds 8 leaves, and emit a
node as one group the moment its subtree has <= 8 leaves. This terminates
with every final group of size 1-8, deterministic given x_train (scipy's
own tie-break for equal distances is deterministic given deterministic
input), with no cluster COUNT declared in advance.

CORRECTED, caught by review: this paragraph first described a DIFFERENT
algorithm -- process merges in ascending-distance order and SKIP any merge
that would exceed the cap, leaving those two clusters separate. That
algorithm is WRONG and was never what the shipped code does: scipy's
linkage output numbers every internal node by its merge order, and later
rows reference earlier internal nodes by that number whether or not this
code chose to use them, so skipping a merge leaves a dangling reference
for every later row that names it (discovered as an IndexError the first
time the skip-based version ran). The shipped implementation is the
top-down cut described above, which never skips a merge and never needs
to. The protocol text was not updated when the code was fixed; it is now,
and scripts/parent_screening.py's [3b] behavior check confirms a forced
correlated block lands in one group under the cut-based rule, alongside
the existing determinism and size-cap checks.

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
separately and EXCLUDED ENTIRELY from every primary metric's numerator and
denominator, recall and complete-target coverage alike (the root count per
system is reported alongside, never folded in). CORRECTED, caught by
review: an earlier draft of this paragraph said a root contributes "1 to
complete-target coverage's numerator trivially", counting an empty
required-parent set as a vacuous success. That would have inflated
coverage in exactly proportion to how many roots a system happens to have,
and it never matched the implementation, whose coverage function already
filtered to non-empty parent sets only. Text corrected to match the
code, which was right.

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
- Confirmation additionally requires, as SEPARATE conditions (the first
  draft of this bullet ran them together into a malformed sentence, caught
  by review): (a) the lower 95% graph-level bootstrap bound on retained-
  parent recall exceeds 0.90; and (b) for EVERY required deployable
  baseline, the POINT ESTIMATE of the paired recall difference is >= 0.05
  AND the lower 95% bootstrap bound of that same paired difference is
  strictly > 0 (i.e. the interval excludes zero improvement; the bound
  itself is NOT required to reach 0.05, only to clear zero). Resampling is
  over whole graph-seed blocks, paired across arms and widths, never over
  channels (Rule 101). These are CONJUNCTIVE; failing any one fails the
  confirmation gate. A bootstrap interval is an empirical statement about
  this generator and these seeds, not a distribution-free coverage
  guarantee, and is reported as such.

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

FROZEN SEED LISTS, exact integers, declared before any scientific
execution. Checked against every seed in scripts/*.py and paper/*.md at
freeze time: no five-digit seed appears anywhere else in this repository,
and none of the four blocks below overlaps any existing range (the
highest prior named range is 1400-1430; engineering seeds used so far in
this experiment are 9001-9003, 900/910/920/930/940, 9500, 9600, all
excluded from every scientific result and none inside a block below).

  Stage B pilot, family 1:            20001 20002 20003 20004 20005 20006
  Stage B pilot, family 2:            21001 21002 21003 21004 21005 21006
  Stage C confirmation, family 1:     22001 22002 22003 22004 22005 22006
                                      22007 22008 22009 22010 22011 22012
  Stage C confirmation, family 2:     23001 23002 23003 23004 23005 23006
                                      23007 23008 23009 23010 23011 23012

Stress-panel seeds are frozen separately, only if Stage C is reached, in
their own amendment committed before that panel runs. A seed used in any
stage is retired: it cannot be reused for a later stage or a rerun under
a changed protocol, and a changed protocol after any result gets a fresh
block, not these seeds again.

STAGE B -- preregistered pilot: V=240, n=4000, 6 independent graph seeds
per family (the exact seeds are the frozen lists above), primary noise
level only, all
required arms including community detection if feasible within this
stage's own cap. Maximum 1 hour total training/analysis. If Stage A's
measurements show that cap is infeasible for the full arm set, this
document is amended with a cost estimate before Stage B is launched, not
silently shrunk.

GATE (pilot bars decide investment; a pilot result is never itself
confirmation), MACHINE-CHECKABLE: proceed to Stage C only if EVERY one of
the following holds, computed by the pilot script and printed as
PASS/FAIL per family per condition, in BOTH primary families at once:

  G1  arm 6's pilot mean retained-parent recall >= 0.90;
  G2  arm 6's pilot mean complete-target coverage >= 0.80, non-root
      targets only (roots excluded, count reported alongside);
  G3  arm 6's mean recall exceeds EVERY required baseline's (arms 1-5,
      plus any feasible modern method) by >= 0.05 at the same budget;
  G4  FIXED-BUDGET INTEGRITY, no relaxation: EVERY non-root target's
      candidate set satisfies |C_q| <= k (the same k = ceil(0.10*(V-1))
      for every arm), with an UNRESOLVED target counted at its full V-1
      fallback size, so any unresolved target fails G4 by construction.
      Only floating-point tolerance is allowed on the reported candidate
      fraction sum|C_q| / (|Q|*(V-1)), which must not exceed k/(V-1). This
      is the guard against a recall number produced by abstention rather
      than screening: an arm that returns V-1 candidates for a target
      cannot pass a fixed-budget screen by doing so, and abstention is
      reported (below), never absorbed as a success.

The unresolved rate is REPORTED per family and per arm, descriptively, and
carries NO separate pass/fail threshold: an earlier draft of this gate
added a G5 (unresolved fraction <= 0.10) and a 1.5x candidate-budget
allowance, both flagged by review as new scientific thresholds introduced
after the registered design, one of them a relaxation of the registered
10% budget to roughly 15%. Both are withdrawn. Validity is determined by
the original fixed-budget condition G4 alone; nothing is added to it here.

If a baseline already meets the task as well (G3 fails against it), that
baseline is preferred and the learned line stops here.

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

---

## Pre-pilot amendment (2026-09-20): eight defects found by independent review of the Stage A implementation, all fixed before any pilot

Registered before any scientific pilot ran. No pilot seed has been
executed, so no result exists under any defective version and none needs to
be set aside or re-registered; the frozen seed lists above were chosen after
these corrections and are untouched. Each defect below was found by an
independent reviewer of the Stage A code, reproduced or confirmed against
the actual implementation before being fixed rather than taken on trust,
and is now guarded by a test in scripts/parent_screening.py's stage_a() or
scripts/parent_screening_split_test.py.

1. ORPHAN REPAIR. Reproduced: 14 of 500 engineering seeds at V=30 had a
   pre-repair orphan root, and in 7 of those 14 the repair's heir was
   itself a root, so the repair edge was recorded in the parent map but
   dynamically inert (both generators' root branches never read
   parent[root]). The displacement branch also never decremented the
   displaced parent's child count. A first fix allowed indegree 4 for
   repaired heirs; review correctly flagged that as broadening the
   registered 1-3 range. Final design: non-root heirs only, indegree
   strictly <= 3, nothing ever displaced. Guarded by: 1000-seed regression
   (zero orphans remaining, max observed indegree 3, no root ever a target
   of a parent edge); independently re-run by the reviewer over seeds
   0..499 covering every-root-has-child, roots-have-no-parents, indegree
   1..3, unique edges, DAG ordering, delays 1..3.
2. LAG CONVENTION, PROTOCOL TEXT. The protocol wrote x_j(t - d) where the
   code implements x_j((t+1) - d). Resolved in favour of the code, which
   uses the standard target-relative lag convention. Guarded by [3c],
   which now drives the PRODUCTION step functions (family1_step_value,
   family2_drive, extracted into named functions the real generators
   themselves call) with a controlled impulse and asserts the observed
   response onset at impulse_index + d for d = 1, 2, 3 in both families. A
   first version of this test checked a hand-written copy of the
   recurrence and could not have caught drift in either real generator;
   caught by the reviewer and rewritten.
3. CLUSTERING DESCRIPTION. The protocol still described the skip-oversized-
   merges algorithm the code had already abandoned for a build-full-tree-
   then-cut-top-down design after the skip version was found to corrupt
   scipy's linkage node numbering. Text corrected; [3b] adds a behavioral
   check that a forced-correlated block lands in one group.
4. FAMILY 2 NOISE ENCODED ROLE. sigma_i was 0.7 for roots and 0.5 for
   non-roots, directly encoding the status the brief said noise must not
   encode. Replaced with sigma_i ~ Uniform(0.4, 0.6) for every channel.
5. INTERNAL RIDGE SEAM UNEMBARGOED. ridge_r2_val split its own training
   block at [:cut]/[cut:] with no embargo, sharing raw lag/target support
   at that inner seam, the same leakage the outer splits were built to
   prevent, missed at the second seam. Fixed with the same E-row embargo.
   The split arithmetic now lives in ONE function, internal_val_split,
   which both ridge_r2_val and the regression test call; a first version of
   the test hand-duplicated the arithmetic while a comment claimed it did
   not, caught by the reviewer. A too-small input now RAISES
   TooSmallForEmbargo instead of falling back to a same-slice arrangement
   (which would let validation selection see its own training rows).
   Guarded by the split test's outer-seam check, internal-seam check, an
   embargo=0 ORACLE for each proving the check would fail if the embargo
   were removed, and a reject-not-fallback check. The split test also no
   longer re-derives raw support from a formula: it feeds index-valued data
   through the production own_lag_window and reads the touched raw indices
   from its output.
6. GATE MACHINE-CHECKABILITY. Unresolved targets keep all V-1 candidates,
   so a run resolving nothing could show high recall via fallback alone.
   The gate is now four explicit, printed conditions (G1-G4 above), with
   G4 a STRICT fixed-budget check: every target's |C_q| <= k, an
   unresolved target counted at V-1 so it fails by construction, floating-
   point tolerance only. My first draft of this fix itself needed
   correction, caught by review before it was committed: it introduced a
   1.5x candidate-budget allowance (about 15% against the registered 10%)
   and a new unresolved-fraction threshold of 0.10, both new scientific
   criteria the registered design never had. Both withdrawn; the unresolved
   rate is reported descriptively and validity rests on G4 alone. Metrics
   candidate_fraction, unresolved_fraction and budget_ok added to the
   script.
7. ROOTS IN COVERAGE. Protocol text said roots contribute a trivial 1 to
   complete-target coverage; the code already excluded them entirely.
   Text corrected to match the code; root count reported alongside.
8. MALFORMED CONFIRMATION SENTENCE. Rewritten as two separate conditions:
   point estimate of the paired recall difference >= 0.05, and the lower
   bootstrap bound of that difference > 0.

Also frozen here: the exact pilot and confirmation seed lists (above), and
perturbation invariance extended from a two-target, hand-fixed-group toy
to the production pipeline (real clustering, real per-group encoders, real
candidate-set construction, every target). SCOPE OF THAT LAST CHECK, stated
plainly: it covers the one arm that currently exists end to end (arm 6, the
learned clustered screen). Arms 2-5 are not yet implemented; each gets its
own perturbation-invariance check when it is built, and Stage B is not
launched until all are in place.

Audit gate re-run before this commit: 136 passed, 0 failed. Stage A,
re-run in full after every fix above: passes, 36.0 s, no GPU-heavy work.

---

## Pre-pilot amendment 2 (2026-09-20): Stage B definitions frozen, abstention rule retained, resource status and cost estimate

Registered before any scientific pilot ran. No pilot seed has been executed.
Engineering seeds used so far, all excluded from every scientific result and
none inside a frozen block: 900, 910, 920, 930, 940, 9001-9003, 9500, 9600,
9700, 9702, 9800, 9801, 9900 (9701 is reserved for a timing run that has not
happened). This amendment freezes definitions the protocol left implicit; it
changes no registered parameter, arm, seed list, threshold or cap.

1. THE UNRESOLVED THRESHOLD (0.01) IS RETAINED, AND WHAT IT WILL PROBABLY DO
   IS STATED IN ADVANCE. While building Stage B, a smoke run on engineering
   seed 9800 (family 1, V=24, n=1200) showed the learned clustered screen
   declaring about 95% of targets unresolved. The scale of the per-target
   maximum group gain was then measured on engineering seeds 9800 (family 1)
   and 9801 (family 2), V=24, n=1200, learned clustered arm, encoder seed =
   group id (scripts/parent_screening_arms.py --engineering-diag; artifact
   ExpOutput/parent_screening/engineering_gain_scale.json, sha256
   801687bf90325752755e9baa1f5af354898b3aebdf682ecffd270930dfd9dfd0):
   family 1, seed 9800, 21 non-root targets: max group gain median +0.0024,
   90th percentile +0.0049, maximum +0.0061, none above 0.01 (all 21
   unresolved); the 3 roots, median +0.0001. Family 2, seed 9801, 21
   non-root targets: median +0.0050, 90th percentile +0.0195, maximum
   +0.0553, 29% above 0.01 (71% unresolved); the 3 roots, median +0.0013.
   The threshold was fixed at registration from a noise-scale reference (the
   ~0.005 clean-ghost floor measured elsewhere in this project) without first
   measuring what gain a driven target actually achieves under these
   generators at the registered coupling; that is the Rule 122/123 error of
   fixing a bar without measuring the metric's scale, made here in the
   protocol. A first internal draft of this amendment WITHDREW abstention.
   Independent review rejected that, correctly: the threshold is a specific
   registered rule, changing it after observing that the method would fail
   it redefines the evaluated method, and a fixed-budget failure caused by
   abstention is a valid negative outcome. The threshold is therefore
   unchanged, no second no-abstention arm is added, and any scale-aware
   abstention rule would be a new design needing its own protocol and fresh
   seeds. PREDICTION, stated before any pilot seed runs: at V=240 arms 4-6
   will abstain on a large share of non-root targets, so G4 fails for arm 6
   because of the registered abstention rule; if so the pilot is reported
   as exactly that, a valid negative caused by the registered rule, and is
   NOT reported as evidence about ranking quality in either direction. The
   share of unresolved targets, the recall over resolved targets only, and
   every arm's per-target maximum gain are reported descriptively and gated
   on nothing.

2. PREPROCESSING, ALL ARMS. Each variable is standardised with the mean and
   standard deviation of the raw rows [0, R), R = MAX_DELAY + t_last + 2,
   where t_last is the last TRAIN row: exactly the raw support of the
   training block. No validation or test observation enters any mean, scale,
   dictionary knot, clustering distance, encoder, readout or penalty choice.
   The first differences used by the clustering are taken inside that span.

3. BASELINE DEFINITIONS, exact.
   RANDOM: k indices drawn without replacement from the other V-1 variables,
   RNG seeded by (system seed, target).
   LAGCORR: for every candidate j and target q, the absolute Pearson
   correlation over TRAIN rows between x_j((t+1)-d) and the target
   x_q(t+1), maximised over d in {1,2,3}; the residualised variant replaces
   the target by q's own-history ridge residual (same readout and alpha
   selection as the screening method, TRAIN rows only). The k largest form
   the shortlist. The deployed variant is the one with the larger mean
   VALIDATION R2 gain when each target's shortlist (each candidate
   contributing its single best-lag column) is appended to that target's own-
   history ridge; both variants are reported.
   LASSO: per candidate variable a 7-column block [its lag-1, lag-2 and
   lag-3 values, and hinge(lag 1) = max(0, lag1 - knot) at the 20/40/60/80%
   TRAIN quantiles of lag 1], every column standardised with TRAIN
   statistics; the target is q's own-history
   ridge residual; scikit-learn Lasso, no intercept, tol 1e-4, at most 2000
   iterations, warm-started along a five-point penalty path of 10^-1,
   10^-1.5, 10^-2, 10^-2.5, 10^-3 times alpha_max. The penalty is chosen on
   the embargoed internal split of TRAIN with EVERY fitted quantity (the own-
   history residualiser, the residual's centring and scale, alpha_max, the
   coefficients) taken from the inner-fit rows only and the inner-validation
   rows transformed with those fixed quantities; the final model is refit on
   all TRAIN rows at the chosen RELATIVE penalty. The k candidates with the
   largest summed absolute coefficient over their block are the shortlist.
   Non-converged fits are counted and reported. The registered phrase "same
   frozen grid width as the ridge grid, five points log-spaced" leaves the
   span open; the two-decade relative span above (narrower than the ridge
   grid's four decades, chosen for convergence and sparsity at V=240) is a
   disclosed interpretation fixed here before any pilot seed, not a claim of
   equivalence. A chosen penalty sitting at a grid end would handicap this
   baseline and so flatter arm 6 in G3; that caveat applies to any G3 pass.
   PCA-GROUP: the arm-6 partition; each group's code is PCA fitted on TRAIN
   rows of its members' concatenated own-lag windows, b_G components, a
   target inside a group scored with its own window zeroed (the same
   convention the learned arm's per-member masking makes in-distribution).
   LEARNED-SIZEDRAND: hierarchy_repair.sized_random on the arm-6 size
   multiset, seed = system seed + 777, learned encoders as arm 6.
   Ties everywhere break by ascending variable index; nothing consults truth.

4. GATE MECHANICS. evaluate_gate returns FAIL, printing every reason, for any
   input that is not exactly two families x six arms x the six registered
   seeds each, with every metric a finite number in [0,1]; the two cases an
   independent reviewer reproduced (empty input, and one family holding one
   perfect arm-6 cell) are regression cases. G4 is evaluated on arm 6;
   budget_ok, the unresolved share and recall over resolved targets only are
   reported for every arm. A baseline arm that abstains (arms 4 and 5 apply
   the same rule) is counted at its V-1 fallback size in its own recall, and
   that inflation of a comparator makes G3 harder for arm 6, not easier.

5. PILOT MECHANICS. Each system's shortlists are written to disk before that
   system's truth file is written; all metrics are computed only after the
   twelfth system; every cap is enforced inside the target, group and
   encoder-epoch loops with the process TREE measured (a child process is
   counted); completions resume only under an identical code and
   configuration hash and their elapsed time counts against the 1 h cap; the
   runner refuses to start unless a reviewer's clearance file names the
   committed HEAD and every script and this protocol are committed.

6. MISSING REQUIRED COMPARISON. The community-detection method
   (arXiv:2501.10886, DADApy) is not installed and installation is not
   permitted here. Per the clause above, any Stage B result is an internal
   feasibility result and the missing comparison is named; G3 is evaluated
   against arms 1-5 only, and no state-of-the-art claim is available.

7. RESOURCE STATUS AND COST ESTIMATE. The caps under "Resource caps" were
   provisional pending measurement; they are confirmed unchanged. Measured
   2026-09-20 on this machine: 28.5 GB installed; available RAM 2.1-2.4 GB
   with no experiment running and 0.9-1.8 GB while one torch process ran
   (own working set 506 MB after imports, 613 MB after the CUDA context,
   about 720 MB after preparing one V=240 system with GPU allocation 147
   MiB; the V<=16 preflight process was observed at 1.28 GB), the remainder
   held by a virtual machine, an editor and browsers that are not this
   project's to close. Under the
   retained 2 GiB floor a guarded Stage B run would breach at its first
   poll on this machine as it stands. That is classified RESOURCE-BLOCKED,
   not a scientific negative, and no V=240 timing was run for the same
   reason. COST ESTIMATE, an extrapolation and NOT a measurement (engineering
   seed 9800, V=24, n=1200, twelve groups: LAGCORR 0.9 s, LASSO 2.8 s,
   clustering 0.1 s, PCA-GROUP 2.3 s, LEARNED-SIZEDRAND 16.7 s, LEARNED-
   CLUSTERED 11.9 s; the LASSO figure is from the draft before its leakage
   fix, which changes what is fitted on, not how much; scoring grows with
   targets x groups, encoders with groups, LASSO with targets x candidates x
   rows): roughly 0.6-1.5 h per
   system at V=240, so about 7-18 h for the twelve systems against the 1 h
   cap. By the Stage B clause above this is the cost estimate that must be
   recorded before launch; the cap, the arm set, the seeds and the cell
   count are unchanged, nothing has been shrunk, and how the estimate is
   resolved against the cap is a separate labelled amendment. Stage B has not
   been launched.

Checks behind this amendment: the arms preflight (gate refusals, tree RSS
with a spawned child, per-loop breach granularity, LASSO inner-validation
leakage test with a leaky reference that must change, arms scoring equal to
the reviewed reference scoring to 0.0, all-arm invariance with a same-input
determinism control and a train-span sensitivity control) passed; Stage A
re-run in full after the refactor passes all ten checks (109.0 s), the
label-orientation gains (B +0.0259, C +0.0006) unchanged to four decimals;
audit gate 136 passed, 0 failed.
