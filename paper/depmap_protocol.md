# Pre-registration: does observational redundancy predict interventional equivalence?

Declared 2026-08-16, before the expression matrix was downloaded and before any
calibration was computed. The Chronos knockout matrix was already on disk from
an earlier audit (`scripts/depmap_audit.py`, CORUM positive control only); no
expression data, no joined matrix and no calibration figure existed when this
was written. Deviations get logged here with reasons, not silently applied.

## The question

For an ordered gene pair (A, B):

> **P( intervening on A is equivalent to intervening on B | A and B are
> observationally redundant at level r )**

Every gene-importance ranking read as a guide to what to perturb relies on this
quantity. We can find no published estimate of it. The deliverable is a
**calibration curve** over r, not a yes/no.

## Why DepMap

The question needs paired observational and interventional measurements on the
same system at scale. DepMap is one of the few places that exists: ~1,100 cancer
cell lines with expression profiles and genome-wide CRISPR knockout screens,
public, quarterly, no application.

## Measures, fixed now

* **`r_obs(A|B)`** — squared Pearson correlation of expression across cell
  lines. For simple linear regression this equals the cross-validated R^2 in
  closed form, which is what makes the full 17,916^2 screen affordable.
* **`r_obs(A|rest)`** — R^2 predicting A's expression from the remaining panel
  (ridge, cross-validated). Computed on the shortlist only. Declared because
  the companion work showed pairwise and many-to-one redundancy diverge, and
  each catches cases the other misses.
* **`e_int(A,B)`** — Pearson correlation of Chronos profiles across cell lines.
  Two genes are functionally equivalent when removing either harms the same
  lines.

Threshold for "interventionally equivalent", fixed before looking:
**tau = 0.5**. The headline number is `P(e_int > 0.5 | r_obs > 0.8)`. The full
curve is reported regardless.

## Predictions, fixed before any calibration is computed

1. **The curve rises.** Redundancy carries *some* information about
   interventional equivalence. A flat curve refutes this.
2. **The ceiling is low.** Even at `r_obs > 0.8`, `P(e_int > 0.5)` is **under
   50%**. This is the prediction we most expect to matter.
3. **The relation is asymmetric.** `P(r_obs high | e_int high)` >
   `P(e_int high | r_obs high)`. Functionally equivalent genes are often
   co-expressed; co-expressed genes are often not functionally equivalent.
4. **Lineage correction matters materially.** The uncorrected curve sits
   visibly above the corrected one. If removing lineage changes nothing, our
   confound model is wrong.

Any of these failing is a result and gets reported as one. A flat curve would be
the strongest finding available here: it would mean expression-based redundancy
tells you nothing about what happens when you intervene.

## Confounds and the handling declared for each

1. **Lineage.** Lung lines resemble lung lines in both matrices, so an
   uncorrected calibration partly measures tissue identity. One-hot lineage is
   regressed out of both matrices before any correlation. Both corrected and
   uncorrected curves are reported.
2. **Pan-essential genes.** Ribosome and proteasome subunits are essential
   everywhere, so their Chronos profiles are near-constant and correlate for
   trivial reasons. Filter on Chronos variance; report with and without
   pan-essentials.
3. **Expression floor.** Genes near zero TPM give noise-dominated correlations.
   Filter on expression level and variance, thresholds fixed from the marginal
   distributions before any pairing.
4. **Chromosomal proximity.** CRISPR screens carry residual copy-number and
   neighbour effects. Same-cytoband pairs are flagged and reported separately,
   never silently pooled.
5. **Multiplicity.** ~1.6e8 pairs. This is a calibration over binned
   populations, not per-pair testing, so no per-pair correction applies. Bin
   counts and confidence intervals are reported with every curve.

## Controls

* **Positive:** CORUM protein-complex members must score high on both measures.
  If they do not, a measure is wrong rather than a hypothesis refuted.
* **Negative:** random gene pairs matched on expression level and essentiality
  magnitude must score low on both.
* **Ghost:** one matrix is permuted across cell lines and carried through the
  identical pipeline. Its calibration curve must be flat. This is the ghost
  channel of the companion paper, transplanted to a cross-sectional setting,
  and it is the check that the pipeline itself is not manufacturing the curve.

## TNBC case study, and its declared weakness

Triple-negative breast cancer is defined here by low ESR1, PGR and ERBB2
expression rather than by subtype label, because that is what the clinical
definition maps to. Thresholds are set from the marginal expression
distributions of the breast lines before any dependency data is examined.

**Declared before running: this arm is underpowered.** DepMap holds roughly
50-60 breast lines, of which perhaps 20-30 are triple-negative. A calibration
curve on that n will be visibly noisier than the pan-cancer one, and we say so
here so that a noisy subgroup result cannot later be presented as the headline.
The pan-cancer calibration is the primary result. The TNBC questions are
secondary: which dependencies are selective to these lines, and whether the
calibration is worse there than pan-cancer.

If the TNBC arm is too noisy to support a claim, that is reported as such and
not rescued by re-binning.

## What counts as success, decided now

Success is a calibrated number with its confidence interval and its controls
passing, whatever the number turns out to be. It is not a high number.

The result is void if: the ghost curve is not flat; CORUM pairs do not score
high on both; or the corrected and uncorrected curves are indistinguishable
while lineage is known to structure both matrices.

## What this cannot establish

Cell lines are not tumours. Chronos measures growth in a dish, not clinical
response. `e_int` measures similarity of knockout phenotype profile, which is a
proxy for functional equivalence and not proof of it. Nothing here nominates a
drug target, and no claim of clinical relevance follows from it.

## Addendum (2026-08-16, later the same day; expression data still not on disk)

A second adversarial pass on this design, before any download. Six amendments,
all declared while the only data on disk is the Chronos matrix.

**A1. Pair taxonomy: `e_int` is conditional on a phenotype existing.**
Single-knockout profiles cannot see buffered redundancy: if A and B compensate
for each other, knocking out either alone does nothing, both Chronos profiles
are flat, and their correlation is noise — `e_int` reads LOW for exactly the
most functionally equivalent pairs (De Kegel & Ryan 2019; the Ito 2021, Parrish
2021 and Dede 2020 double-knockout screens exist because of this). Declared
handling: pairs are partitioned BEFORE calibration into (i) informative — at
least one gene shows a dependency (Chronos below a fixed effect threshold in at
least a fixed number of lines, thresholds set from marginals before pairing)
and (ii) phenotype-free — neither gene has a phenotype anywhere, where
single-KO equivalence is trivially satisfied but unmeasurable. Class (ii) is
reported as its own stratum and never pooled into the curve. The headline
calibration therefore answers "does redundancy predict SINGLE-intervention
equivalence", stated as such, and the phenotype-free stratum is the declared
signature of buffering rather than a failure of the hypothesis.

**A2. Confound 6 — shared-artifact inflation in high-identity paralogs.**
RNA-seq multi-mapping inflates `r_obs` and CRISPR guide cross-targeting
inflates `e_int`, on the SAME pairs. Correlated measurement error on both axes,
concentrated in the top redundancy bin, could manufacture a rising curve; the
permuted-line ghost cannot catch it because the artifact lives within genes,
not across lines. Handling: pairs flagged as paralogs (published paralog list;
gene-family symbol root as fallback), curves reported with the flagged pairs
included and excluded. If the curve's rise survives only with paralogs
included, that is reported as artifact-compatible, not as calibration.

**A3. Positive control vs pan-essential filter collision.** CORUM complexes
are enriched for pan-essential genes, which the declared variance filter
removes; as first written, the control and the filter could not both survive.
Resolution: the CORUM control validates the instrument and runs on the
UNFILTERED matrices; a selective-complex subset whose members pass the
variance filter is reported alongside. The calibration itself remains
filtered as declared.

**A4. Confidence intervals must respect pair dependence.** Pairs share genes,
so pair-level resampling is invalid. Declared: curve CIs by bootstrap over
GENES (resample genes, rebuild pairs each replicate); measurement noise by
bootstrap over cell lines. No pair-level bootstrap anywhere.

**A5. Phase-0 literature gate (standing protocol rule 12, applied in
advance).** Before any calibration is computed, a documented search for prior
art on expression-redundancy-to-co-essentiality calibration: at minimum
Wainberg 2021 (co-essentiality almanac), Pan 2018, Boyle 2018, the DepMap
dependency-predictability papers (Dempster et al.), De Kegel & Ryan 2019, and
CausalBench. Comparing co-expression and co-essentiality NETWORKS is adjacent
but not this calibration; if the calibration itself turns out to be published,
this study becomes a declared replication and extension and this document is
amended to say so before proceeding.

**A6. Data release pinning.** The Chronos matrix already on disk is recorded
with its DepMap release before anything is joined; expression and Model.csv
are downloaded from the SAME release, or the Chronos matrix is re-downloaded
to match. Release identifiers go in the results log.

## A5 gate outcome (2026-08-17, before the expression matrix was opened)

Documented search performed before any calibration. Nearest neighbours found,
none of which computes the forward calibration:

- Wainberg et al. 2021 (Nat Genet) benchmarks co-essentiality NETWORKS against
  co-expression NETWORKS (COXPRESdb) at recovering known functional
  interactions - a network-recovery comparison across different panels, not a
  pair-level P(e_int | r_obs) calibration on matched lines. Cite.
- arXiv 2603.20955 (2026) reports the REVERSE conditional on DepMap: ~98% of
  top co-essential pairs have expression cosine > 0.5. This is our prediction
  3's direction and must be cited as the nearest published number; it does not
  measure the forward direction practitioners rely on.
- Dempster et al. predictability work maps a gene's OWN omics features to its
  OWN dependency (biomarker direction), not pair redundancy to pair
  equivalence.
- De Kegel & Ryan 2019 (paralog buffering), CausalBench (perturb-seq algorithm
  benchmark): adjacent as declared in A1/A5.

VERDICT: no published forward calibration found. The study proceeds as a
first measurement, with the above cited at the point of claim and the claim
phrased as "we can find no published estimate", not "nobody has done this".

Release pinned (A6): DepMap 24Q4 Public, figshare article 27993248. Newer
releases (25Qx, 26Q1) exist but sit behind the portal's bot verification;
24Q4 is the newest release with direct, scriptable, checksummable URLs, and
all three files (Model.csv, CRISPRGeneEffect.csv,
OmicsExpressionProteinCodingGenesTPMLogp1.csv) are drawn from that single
article, Chronos re-downloaded to guarantee the match. The earlier on-disk
CRISPRGeneEffect.csv (release unrecorded) is retired from this analysis.

A6 note (2026-08-17): md5 of the retired Data/depmap/CRISPRGeneEffect.csv
equals the freshly downloaded 24Q4 file (6edf7ade09b9b34199210b559d4745d3), so
the earlier audit's matrix is retroactively identified as 24Q4. The matched
trio (Model, Chronos, Expression) is now verified from article 27993248.

## A7: the declared ghost was vacuous, and is replaced (2026-08-17, logged
## the moment it was discovered, before any result is read as final)

The declared ghost ("one matrix is permuted across cell lines") is a no-op BY
ALGEBRA for this statistic: e_int correlates gene columns WITHIN the Chronos
matrix, and a row permutation applied to the whole matrix permutes both
columns identically, leaving every within-matrix correlation unchanged. The
run exposed this by returning bin-identical hit counts (1815 = 1815 in every
r_obs bin). A control that cannot fail is not a control.

Replacement, declared now: GENE-LABEL permutation. Chronos gene columns are
relabelled by a fixed permutation (seed 0) over the kept gene set, so pair
(A,B)'s r_obs from expression is paired with e_int of (sigma(A), sigma(B))
from knockouts. Under this null the calibration curve must be flat at the
base rate; the void condition transfers to THIS ghost.

Also implemented in the same pass, closing gaps the first run left open:
- CORUM-style positive control using the curated complex lists already
  declared in scripts/depmap_audit.py (proteasome core/lid etc.), evaluated
  on UNFILTERED matrices per A3.
- Negative control: random gene pairs matched on expression mean and Chronos
  std (quantile-matched), must sit at base rate.
- SENSITIVITY arm, labelled as such and not replacing the declared measure:
  e_int recomputed as Spearman correlation, because z-scored Chronos
  profiles are heavy-tailed and near-one-hot profiles can make a Pearson
  e_int > 0.5 out of a single shared outlier line. The declared Pearson
  headline stands; the Spearman arm bounds how much of the curve is
  outlier-driven.

## A8: the declared e_int is one axis of a two-axis quantity (2026-08-17,
## triggered by the positive control failing exactly as the protocol's own
## interpretation rule anticipated)

The positive control failed: 253 curated complex pairs scored P(e_int>0.5) =
0.00, median 0.22, on the declared lineage-corrected Pearson. Direct
verification of the mechanism on raw Chronos: PSMA1-PSMA2 Pearson 0.30,
PSMA1-PSMB5 0.04, EXOSC4-EXOSC8 0.27 - canonical same-complex pairs, near
zero. Cause: Pearson across lines measures covariance of VARIATION, and
pan-essential subunits have flat profiles whose equivalence lives in the
shared mean, which centring erases. The protocol's rule applied: "a measure
is wrong rather than a hypothesis refuted."

Amendment, declared before any curve is recomputed. Interventional
equivalence has two axes and both are reported:

  e_prox = 2<kA,kB> / (<kA,kA> + <kB,kB>) on raw (uncentred, lineage-
           uncorrected) Chronos profiles: "the two knockouts produce the
           same overall growth phenotype." Verified on the controls:
           same-complex pairs 0.90-0.98, PSMA1-KRT1 0.04. tau_prox = 0.8,
           set between those margins before any curve is computed.
  e_sel  = the original declared measure (lineage-corrected Pearson):
           "the two knockouts share a selective vulnerability pattern."
           tau_sel stays 0.5.

Known conflation, stated in advance: ANY two pan-essential genes score high
on e_prox (verified: COPA-MED14 0.93 across complexes) - and under a growth
screen that is semantically true, since both produce the same phenotype in
every line. Interpretation therefore REQUIRES the A1/A2 strata: e_prox
curves are reported separately for the pan-flat stratum, the has-phenotype
stratum, and overall. The positive-control gate now applies to e_prox (must
be high for complex pairs) AND the negative control to both axes.

The headline question splits accordingly:
  P(e_prox > 0.8 | r_obs > 0.8)  - same phenotype given redundancy
  P(e_sel  > 0.5 | r_obs > 0.8)  - same selective pattern given redundancy

## Post-hoc precision check on the bootstrap (declared 2026-08-20)

The pre-registration fixed B = 100 gene-level bootstrap replicates, and the
published intervals use that value. A 95% percentile interval from 100
replicates places its bounds on roughly the 2.5th and 97.5th order
statistics, which is coarse; an adversarial read of the manuscript flagged
it. B = 1000 is therefore run as a POST-HOC PRECISION CHECK, declared here
before it is run and reported alongside the pre-registered B = 100 rather
than replacing it. Outputs are written to separate files (suffix _b1000) so
both remain available. If the two disagree materially, both are reported and
the disagreement is the finding; the pre-registered value is not discarded
in favour of whichever looks better.
