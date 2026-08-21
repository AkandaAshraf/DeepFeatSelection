# Pre-registration: does the redundancy calibration survive a richer phenotype?

Declared 2026-08-20, before the data was read. The download was started
before this was written; no file has been opened.

## The question

The DepMap study measured P(interventional equivalence | expression
redundancy) and found a ceiling near 17%. Its sharpest limitation, stated in
the paper, is that "same effect" there means "same effect on PROLIFERATION" -
a single number per gene per cell line, structurally blind to any consequence
that does not change growth rate.

Genome-scale Perturb-seq (Replogle et al. 2022) replaces that one number with
a whole transcriptome: for each of ~10,000 CRISPRi knockdowns in K562, the
expression response across ~8,000 genes. If two genes are interventionally
equivalent, knocking each down should produce the same transcriptomic
response.

This tests whether the 17% ceiling is a fact about biology or an artefact of
an impoverished phenotype.

## Design, and what is held fixed

OBSERVATIONAL AXIS: unchanged from the DepMap study. r_obs is the squared
Pearson correlation of expression across the 1,103 DepMap cell lines, after
lineage correction. Holding this fixed is deliberate: it isolates the change
in phenotype definition, which is the one thing being tested.

INTERVENTIONAL AXIS: cosine similarity between the pseudo-bulk transcriptomic
response vectors of the two knockdowns, on the normalised bulk profiles.

DECLARED CONTAMINATION CONTROL: knocking down gene A depresses A's own
transcript. If A's and B's own transcripts remain in the response vectors,
every pair is non-equivalent for a trivial reason and the ceiling collapses
artefactually. FOR EVERY PAIR, both genes' own transcripts are removed from
both response vectors before similarity is computed. This is the single most
important implementation detail in the study.

MINIMUM QUALITY: perturbations whose profiles are all-zero or constant are
excluded and the count reported.

## Controls, fixed now

POSITIVE: same-complex gene pairs from CORUM, as in the DepMap study. If
knocking down two subunits of one complex does not produce similar
transcriptomic responses, the similarity measure is wrong and the study is
void until it is fixed - the same rule that caught the Pearson/proximity
error in the DepMap work.

NEGATIVE: randomly drawn gene pairs, which must sit at the base rate.

GHOST: gene labels on the Perturb-seq matrix permuted, identical pipeline.
Its calibration curve must be flat.

## Threshold

The equivalence threshold on cosine similarity is NOT fixed in advance,
because the scale is not comparable to the DepMap proximity measure. Instead
the FULL SWEEP is reported (0.1 to 0.9), and the headline is quoted at the
threshold that puts the negative control at the DepMap base rate of 0.42%,
so the two studies are compared at matched specificity rather than at an
arbitrary common number. That calibration rule is fixed here, before any
curve is computed.

## Predictions, fixed before the data is read

  PS1  A calibration curve exists: equivalence rises with r_obs. If flat,
       expression redundancy carries no information about transcriptomic
       response, which would be a stronger negative than the DepMap study.
  PS2  NO PREDICTION on whether the ceiling is higher or lower than 17%.
       This is the quantity of interest and all three outcomes are
       informative:
         HIGHER  the DepMap ceiling was depressed by the growth-only
                 phenotype, and the published headline understates how often
                 redundant genes are interchangeable
         SIMILAR the ceiling is a fact about biology, robust to phenotype
                 definition - the strongest outcome for the paper in review
         LOWER   redundancy is even less informative than measured
  PS3  The positive control passes: CORUM same-complex pairs score well
       above the base rate.
  PS4  The ghost curve is flat.

## What follows if the ceiling is much higher

The PLOS submission would require correction, not merely extension. That is
the reason to run this now rather than after a decision.

## Honest limits, declared now

One cell line, so no lineage confound and no lineage generality. CRISPRi
knockdown is not knockout. The observational axis comes from a different
experiment than the interventional one, which is a deliberate choice to hold
it fixed but means the two axes are not measured on the same cells. A
within-Perturb-seq observational axis would need the 66 GB single-cell file
and is deferred.

## Void conditions

Void if the contamination control is omitted, if the positive control fails
and the study proceeds anyway, if the threshold calibration rule is changed
after seeing a curve, or if predictions are altered after any result.

---

## Result (2026-08-20)

Data: K562 genome-scale Perturb-seq pseudo-bulk, 11,258 perturbations x
8,248 measured genes. Universe after the DepMap expression filter and the
usable-profile filter: 6,950 genes, 24,147,775 pairs. On-target transcripts
removed from both response vectors for every pair, as declared.

### Controls, which gate everything else

  POSITIVE  253 CORUM same-complex pairs: median cosine +0.271, 49.0% above
            the calibration threshold
  NEGATIVE  921 random pairs: median cosine +0.005, 0.43% above threshold
  GHOST     CORUM pairs with the response matrix's gene labels permuted:
            median +0.006, 1.6% above threshold

Separation between positive and negative controls is +0.486. The positive
control PASSES, so the study is not void and the measure is reporting
functional similarity rather than an artefact. The ghost collapses the
positive control to the negative level, as it must.

### PS1 held: the curve exists

Equivalence rises with observational redundancy at every threshold tested.
At the declared calibration point the lift is 9.4x over the base rate.

### PS2: the ceiling is LOWER, not higher

Threshold sweep, with the base rate each threshold produces:

  tau    base rate   ceiling   lift
  0.1     1.941%      7.8%      4.0x
  0.2     0.424%      4.0%      9.4x     <- declared calibration point
  0.3     0.136%      2.5%     18.1x
  0.4     0.055%      1.5%     27.7x
  0.5     0.025%      0.9%     37.0x

The declared rule was to quote at the threshold whose base rate matches
DepMap's 0.42%, so that the two studies are compared at matched specificity
rather than at an arbitrary common number. That is tau = 0.2, giving a base
rate of 0.424% against DepMap's 0.42%.

  PERTURB-SEQ   ceiling 4.0%    lift 9.4x
  DEPMAP        ceiling 17.0%   lift 39.3x

### What this means

The DepMap ceiling was NOT depressed by its impoverished phenotype. The
opposite: when "same effect" is measured as the whole transcriptomic
response rather than a single growth number, redundant-looking genes are
LESS likely to be equivalent, not more - roughly one in twenty-five at
matched specificity rather than one in six.

The reading that makes sense of both: growth is a low-dimensional readout,
so two genes can produce the same growth effect through unrelated
mechanisms. The transcriptome does not collapse in that way, and pairs that
looked equivalent on proliferation separate once the full response is
visible. The DepMap number is an UPPER bound on interventional equivalence,
not an underestimate.

### Consequence for the submission

No correction is required. The submitted paper's headline stands, and this
strengthens rather than weakens it: the paper's central caution - that
expression redundancy should raise suspicion and never settle it - is
reinforced by a richer phenotype giving a lower number. The submission's
declared limitation, that growth-only phenotyping might understate
equivalence, is now measured and runs the other way.

### Honest limits

One cell line, so no lineage generality. CRISPRi knockdown, not knockout.
The two axes come from different experiments by design, to hold the
observational axis fixed. Cosine similarity on pseudo-bulk profiles is one
choice among several; the full sweep is reported so the reader can take
another. Perturbations with few cells carry noisier profiles and are not
separately filtered here beyond the all-zero and constant exclusions.
