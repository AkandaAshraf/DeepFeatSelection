# Does observational interchangeability predict interventional equivalence?

A proposal for calibrating redundancy-based feature importance against CRISPR
knockout ground truth.

*Draft v0.1. Citations are from a targeted rather than systematic search and
should be verified before submission; see "Threats to the contribution" below.*

---

## 1. Goal

Measure, on real human cell biology, how well **observational redundancy**
predicts **interventional equivalence**.

Concretely: if two genes are informationally interchangeable in expression data —
each reconstructible from the other, so that no importance measure can separate
them — how often does knocking out one produce the same phenotype as knocking out
the other?

The output is a calibration curve, not a yes/no. Given observational redundancy
at level *r*, what is the probability that the two genes are functionally
equivalent under intervention? That quantity is currently assumed rather than
measured, and it is assumed in both directions by different communities.

## 2. Research gap

Three literatures meet here and none of them answers the question.

**Feature importance under dependence** establishes that correlated features break
importance estimates (Strobl et al. 2008; Hooker, Mentch & Zhou 2021), and that
under exact redundancy every risk-difference measure returns identically zero
(Proposition 1 of the companion draft). Shapley approaches restore a non-zero
number (Covert et al. 2020) but cannot order within a redundant set, because the
symmetry axiom requires ties — the explicit motivation for Asymmetric Shapley
Values (Frye, Rowat & Feige 2020).

**Causal feature selection** establishes that predictive optimality recovers the
Markov blanket — parents, children and spouses — not the causes (Koller & Sahami
1996; Tsamardinos & Aliferis 2003; Aliferis et al. 2010). So a purely predictive
ranking cannot orient, and everyone agrees intervention is what settles it.

**Genomics** has both halves of the measurement independently. Co-expression
networks group transcriptionally co-regulated genes (Langfelder & Horvath 2008);
co-essentiality networks group genes whose knockout phenotypes agree across cell
lines, and are known to recover protein complexes more sharply than co-expression
does (Pan et al. 2018; Wainberg et al. 2021).

**The gap.** The first two literatures assert that observational interchangeability
does not imply causal equivalence, and demonstrate it theoretically or on
simulations. The third has the data to check it and asks a different question —
which network better recovers known pathways — rather than treating the
observational structure as a *predictor* of the interventional structure and
scoring its calibration.

Nobody, as far as we can determine, has estimated: **P(intervening on A ≡
intervening on B | A and B are observationally redundant at level r)**.

That quantity is what a practitioner implicitly relies on whenever they read a
feature-importance ranking as a guide to what to perturb. It is being used
without ever having been measured.

**Why the gap exists.** It needs paired observational and interventional data on
the same system at scale. DepMap is one of the few places that exists: ~1,100
cancer cell lines with both expression profiles and genome-wide CRISPR knockout
screens, publicly released.

## 3. Related work, and what this is not

**CausalBench** (Chevalley et al. 2023) is the nearest neighbour: it uses
Perturb-seq to benchmark causal-discovery algorithms on gene expression, scoring
recovered networks against perturbation-validated edges. Our question is narrower
and different in kind. CausalBench asks *how accurate is this algorithm*; we ask
*how far does a specific failure mode reach* — given that predictive methods
cannot separate redundant features, how often does that redundancy actually
correspond to functional equivalence? A method could score poorly on CausalBench
while our calibration is high, or vice versa.

**Perturb-seq** (Dixit et al. 2016; Replogle et al. 2022) supplies richer
interventional readouts than a growth phenotype and would be the natural
follow-up, at the cost of far fewer perturbations per experiment.

**Co-essentiality module discovery** is established practice. We are not proposing
to rediscover modules. We are proposing to use them as the interventional answer
key against which an observational statistic is scored.

## 4. Proposal

### 4.1 Data

DepMap public release: expression matrix **E** (cell lines x genes, log TPM) and
CRISPR gene-effect matrix **K** (cell lines x genes, Chronos scores), joined on
cell line identifier. Both quarterly, open, no application.

### 4.2 Measures

For an ordered gene pair (A, B):

* **Observational redundancy** `r_obs(A|B) = ` cross-validated R^2 of predicting
  E_A from E_B. Also computed against the full panel, `r_obs(A|rest)`, since the
  companion work showed pairwise and many-to-one redundancy diverge and each
  misses cases the other catches.
* **Interventional equivalence** `e_int(A,B) = ` correlation between K_A and K_B,
  the knockout profiles across cell lines. Two genes are functionally equivalent
  when removing either harms the same cell lines.

### 4.3 Analysis

1. **Calibration.** Bin pairs by `r_obs` and report the distribution of `e_int`
   within each bin. The headline is the curve, plus P(e_int > tau | r_obs > 0.8)
   for a pre-set tau.
2. **Asymmetry.** Compare P(e_int high | r_obs high) against P(r_obs high | e_int
   high). These need not be equal and the difference is interpretable.
3. **Positive control.** Protein complex members (CORUM) should score high on
   both. If they do not, a measure is wrong rather than a hypothesis refuted.
4. **Negative control.** Random gene pairs matched on expression level and
   essentiality magnitude should score low on both.
5. **Confounding.** Both matrices are structured by cell lineage: lung lines
   resemble lung lines in expression and in dependency. Correlations must be
   computed after removing lineage, or the calibration measures tissue identity.
6. **Orientation, secondary.** Where a pair is observationally redundant, test
   whether RESIT orientation on expression predicts which knockout has the larger
   effect. This is the first test of that machinery against real interventions.

### 4.4 Scope and cost

A few hundred MB of downloads, no application, compute dominated by the pairwise
regressions and tractable if restricted to a stratified sample of pairs rather
than all ~1.6x10^8. Feasible on a laptop.

## 5. Hypotheses

**H1 (primary).** Observational redundancy is a poor predictor of interventional
equivalence. Among gene pairs with `r_obs > 0.8`, the majority will *not* show
interventional equivalence.
*Reasoning:* transcriptional co-regulation is not functional requirement. Two
genes driven by one transcription factor are mutually predictable in expression
while only one may be required for growth.
*Refuted if:* high observational redundancy reliably implies high interventional
equivalence — which would be the more useful result, making the cheap
observational audit a proxy for expensive perturbation.

**H2 (asymmetry).** The implication runs more strongly from intervention to
observation than the reverse: P(r_obs high | e_int high) > P(e_int high | r_obs
high).
*Reasoning:* functional partners are usually co-regulated, but co-regulation is
far more common than shared requirement.
*Refuted if:* the two conditionals are comparable, indicating a symmetric
relationship and undermining the direction of the argument.

**H3 (mechanism).** Pairs that are observationally redundant but interventionally
distinct will be enriched for shared upstream regulation rather than shared
complex membership.
*Reasoning:* this is the specific structure that produces the dissociation, and
it makes the failure predictable rather than merely observed.
*Refuted if:* such pairs show no regulatory enrichment relative to matched
controls.

**H4 (positive control, must hold).** Members of the same protein complex will
show both high observational redundancy and high interventional equivalence.
*If this fails*, the measures are miscalibrated and H1-H3 are uninterpretable.

## 6. What a result would establish

The companion draft's central caution — predictive interchangeability says
nothing about causal role — currently rests on synthetic systems of our own
construction. The fair objection is that the benchmark was built to produce the
conclusion, and we have no answer to it.

This replaces that with a measured quantity on real human cell biology with real
interventions: not a claim about a simulation, but a calibration between two
things that are routinely conflated. It converts an argument into a number, and
the number is directly useful — it tells a target-discovery pipeline how much
weight an observational importance ranking can bear before a perturbation
experiment is needed.

## 7. Threats to the contribution

* The literature search behind Section 2 was targeted, not systematic. One claim
  in the companion draft has already been demoted this way (the Shapley symmetry
  argument is the stated motivation for Asymmetric Shapley Values). Assume
  further prior art until a proper search says otherwise, and check the
  co-essentiality literature specifically, which is closest to this question.
* Growth phenotype is a coarse interventional readout: two genes can be equally
  required for proliferation while doing entirely different things. A negative
  result may reflect the readout rather than the biology, and Perturb-seq would
  discriminate.
* Cancer cell lines are not normal tissue, and dependencies are partly artefacts
  of the lines' own genetic backgrounds.
* CRISPR knockout is not a clean intervention: incomplete knockouts, copy-number
  effects on cutting toxicity, and compensation by paralogues all blur the
  contrast. Chronos corrects some of this; none of it disappears.
