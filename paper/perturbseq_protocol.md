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
