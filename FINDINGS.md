# Findings

Everything this repository has established, in one place, in plain language.
Every number below is produced by a script in [`scripts/`](scripts/), governed
by a pre-registered protocol in [`paper/`](paper/), and recorded — including
every negative result and every correction — in the
[experiment ledger](paper/causal_detection_log.md). Nothing here asks to be
trusted; everything can be re-derived.

The repository holds two lines of work. **MACE** detects which variables of a
large dynamical system are *driven* by the system, from time series. The
**redundancy line** asks what feature-importance and gene-importance rankings
can and cannot tell you when variables are interchangeable. They meet in the
DepMap study below.

---

## Part 1 — The MACE method ([preprint](paper/excess_paper.pdf), not peer reviewed)

### 1. Drivenness can be scanned at previously impossible scale

A complete scan of **71,721 zebrafish neurons runs in ~14 minutes on a
consumer laptop**. Our own wall-clock measurement of the pairwise alternative
(convergent cross mapping) projects **~15,800 compute-hours** for a scan of
10,000 variables, and roughly 800,000 GPU-hours at the zebrafish width. The
gap is structural: MACE trains one shared code per system and asks each
variable one cheap question, so cost grows linearly with variables instead of
quadratically.

*What it means for you:* questions of the form "which of my 10³–10⁵
simultaneously recorded variables are driven by the rest?" moved from
infeasible to routine.

### 2. Why learned causal readouts fail, in three mechanisms

Before proposing anything, the paper isolates three separable ways learned
models break on redundant dynamical systems:

- **Maturity collapse** — difference-based importance scores rise, peak, and
  collapse to zero as a model learns alternative routes to redundant
  information. A mature model is exactly the wrong one to read importances
  from.
- **Typicality substitution** — masked-reconstruction scores rank channels by
  how *typical* they are, not how coupled. A surrogate channel with no causal
  connection to anything outscored genuinely coupled variables.
- **Ensembles average variance, not bias** — eight independently trained
  models agreed with each other while all being wrong the same way.
  Consensus is not validity.

*What it means for you:* pointing a foundation-model-style probe at scientific
time series and reading structure off it is not a neutral act. Each mechanism
has a concrete experimental signature you can check for.

### 3. The statistic has a derivation, not just a benchmark

Under stated assumptions, MACE's population value equals **the fraction of a
variable's next-step variance its own history cannot explain** — zero exactly
for autonomous variables. The embedded **ghost channel** (a time-shifted copy
of real data that must score zero) follows as a corollary, giving every scan a
built-in falsification test that needs no ground truth.

Two honest boundaries, stated in the paper: the guarantee holds only when a
variable's own history nearly saturates prediction (**on calcium-imaging data
it does not** — 0 of 1,276 worm channels reach the threshold, so there the
ghost is an empirical meter, not a theorem), and the implemented estimator is
a *lower bound* on the identified quantity.

### 4. High precision, low recall — and what a silence means

On synthetic systems with known ground truth: **precision 0.95 at 28%
recall** (1.00 at 29% on the three fully independent systems). Zero of 63
genuine sources were ever flagged — the designed blindness to sources held
everywhere it was tested.

*What it means for you:* what MACE names is very likely real. What it stays
silent about means nothing. **Absence from the driven core is not evidence a
variable is autonomous.**

### 5. Real-world deployments, each with its control

- ***C. elegans* (immobilised):** the top-driven neurons are the canonical
  motor-command ensemble (10/10 command/motor class in two of three
  development animals; held-out animals confirmed), recovered with no
  anatomical labels. Under genetic silencing of the command hub AVA, the
  statistic registered the perturbation **at the level of the affected
  module** in pre-registered directions — after adding permutation nulls,
  only one cell (VB01) survives multiplicity correction, so per-neuron
  resolution is explicitly *not* claimed.
- **Zebrafish (71,721 neurons):** the driven core is spatially coherent and
  posterior-midline — structure, not scatter — with the ghost at −0.031.
- **Clinical EEG (scalp):** seizures concentrate drive onto **fewer**
  channels (18.5 → 11.7, p = 0.047) *without* increasing total drive
  (p = 0.84) — redistribution, not recruitment. Per-event maps failed to
  replicate across seizures (Spearman 0.08): single-event connectivity maps
  should not be treated as patient signatures.
- **Climate (77 years of sea-level pressure):** the driven core is the deep
  tropics, matching known dynamics; an embedded solar-forcing channel scored
  −0.003 — the designed blindness to exogenous sources, confirmed by physics.

---

## Part 2 — New studies (August 2026, after the paper)

### 6. The replication that failed, and the control that caught it

Scanning all **91 freely-moving worm recordings** on WormWideWeb (9.2 minutes
total): the command-core result does **not** replicate in moving animals —
flagged channels match the command-class base rate at every threshold tested.

The important finding is *why we know to distrust the scan*: three
**GFP recordings** — an activity-independent fluorophore, so any "drivenness"
is motion artifact by construction — scored top excess up to **+0.24 with
perfectly clean ghost panels**. Shared motion is genuinely shared information;
a surrogate control is structurally blind to it.

*What it means for you:* on freely-moving imaging data, surrogate/ghost
controls are **not sufficient**. An activity-independent control channel is
necessary, and it defines the artifact floor every claim must clear. Without
the GFP arm, this scan would have reported ~14 driven neurons per animal —
plausible, publishable, and wrong.

### 7. Expression redundancy barely predicts interventional equivalence (DepMap)

The question, which we could find no published measurement of: *if two genes
look interchangeable in expression data, how often does knocking them out do
the same thing?* Using DepMap 24Q4 (1,103 cancer cell lines × 17,716 genes,
39 million gene pairs):

> **The calibration curve rises — redundancy multiplies the odds of
> interventional equivalence about 39-fold (95% CI 23–65) — but the ceiling
> is about 17% (95% CI 10–27), and at the most extreme redundancy observed,
> 0 of 13 pairs were equivalent (a count that only rules out rates above
> ~1 in 4).** The ceiling pools the r² 0.60–0.70 bins; intervals are
> gene-level bootstrap, B=100, per the protocol.

Supporting findings:

- **Tissue identity fakes redundancy:** lineage correction removes 56% of
  apparent high-redundancy pairs (r² >= 0.6); the fraction removed grows with
  redundancy, from 37% at r² >= 0.4 to about 70% at r² >= 0.7. Uncorrected co-expression partly measures
  "these are both lung lines".
- **Correlation of knockout profiles is the wrong equivalence measure for
  essential genes:** canonical same-complex pairs (e.g. PSMA1–PSMB5) score
  Pearson 0.04 but proximity 0.90 — uniform co-essentiality lives in the
  mean, which centring erases. The study reports both axes.
- A pre-registered prediction (the direction of the asymmetry) **failed** and
  is reported as failed.

*What it means for you:* choosing a perturbation target by expression
similarity is evidence-informed guessing — a ~39x lift on a sub-1% base rate.
Interchangeable-looking genes are, four times out of five or worse, *not*
interchangeable under intervention.

### 8. The first real data where MACE's theory fully applies (intracranial EEG)

On interictal SEEG (OpenNeuro, ~1 kHz, 105–180 channels), the self-baseline
**saturates for the first time on real data** — self-R² up to 1.000, with 32%
of channels above the 0.9 donor threshold on one arm. The theory's ghost
guarantee, which no previous real deployment could license, is licensed here,
and the donor-filtered ghost panel produced the tightest null yet measured
(max +0.014).

Two design findings with general value:

- **The reference montage is a control, not a convention.** Common-average
  referencing *worsened* one subject's ghost panel (46% → 68% positive) by
  redistributing a nonstationary common component into every channel.
  Montages must carry their own ghosts.
- Stationarity is segment-by-segment: one subject's segment failed the ghost
  gate and was discarded by rule.

### 8b. That pre-registered study ran, and it did not replicate

The study on file ([protocol](paper/ieeg_protocol.md)) tested whether the
clinically annotated seizure-onset zone — a *source*, which MACE is blind to
by design — is depleted from the driven core. Predictions were frozen before
any label was opened. It is reported here in full because the outcome is
negative.

**Discovery cohort (28 subjects).** The primary test passed: SOZ channels
were depleted from the driven core, pooled z = +5.107, p = 1.6×10⁻⁷. It
survived leave-one-out and the removal of the five most influential
subjects. Two same-scale sensitivity arms, declared after the labels were
open and so explicitly post-hoc, both agreed (laplacian z = +4.185,
bipolar-skip z = +4.569).

**Held-out cohort (10 subjects, same dataset, same condition).** A
[replication protocol](paper/ieeg_replication_protocol.md) with five
quantitative predictions was committed before any held-out recording was
downloaded. **All five failed.** The primary arm gave z = −1.455 (p = 0.93)
against a predicted +3.2 — in the wrong direction — with a driven-core gap
of −4.1 points where +5 to +18 was predicted. Power does not explain it: the
estimate is on the wrong side of zero.

**Status: not an established finding.** No claim that MACE's source blindness
is demonstrated on clinical data rests on this. As declared in advance, no
further cohort was sought to rescue it, and a site-stratified reanalysis was
explicitly declined as the post-hoc subgroup hunt the pre-registration exists
to forbid.

*What it means for you:* three derivations of one dataset agreed with each
other and all three were wrong about held-out subjects. **Agreement among
analyses of one cohort measures analytic robustness, not generalisation.**
The replication cost about a day and prevented a false positive from being
published as a clinically anchored validation.

---

## Part 3 — Rules this work paid for

Distilled from the [ledger](paper/causal_detection_log.md)'s 22 standing
rules; each was purchased with a concrete mistake, ours included.

1. **A control that cannot fail is not a control.** One of our pre-registered
   null controls was invariant under the very permutation it prescribed —
   caught only because its output was *too* identical.
2. **When a positive control fails, suspect the measure before the biology.**
3. **Surrogate nulls miss shared artifacts.** Use an activity-independent
   channel where one exists (GFP; a known-exogenous variable) and report the
   artifact floor it defines.
4. **A floor estimated from one control is not a floor** — ours tripled when
   the control arm went from one recording to three.
5. **A ghost's guarantee depends on its donor:** draw surrogate donors from
   channels whose own history nearly saturates prediction, never uniformly.
6. **Absence is not autonomy** — precision-first detectors say nothing by
   silence.
7. **Check whether a "finding" is already published** — three of four
   biological observations we initially framed as novel were prior art, one
   of them in the same dataset's own source paper.
8. **Report the statistic the p-value actually tests**, and re-derive every
   published number from primary data by script
   ([`audit_paper_numbers.py`](scripts/audit_paper_numbers.py)).
9. **Reference/montage/preprocessing choices are experimental arms** with
   their own controls, never conventions to adopt silently.
10. **Negative results are findings.** The failed replication above is as
    load-bearing as any success in this repository.

---

## Provenance

- Preprint: [`paper/excess_paper.pdf`](paper/excess_paper.pdf) — not peer
  reviewed. CC BY 4.0.
- Pre-registrations: [`paper/validation_protocol.md`](paper/validation_protocol.md),
  [`paper/depmap_protocol.md`](paper/depmap_protocol.md),
  [`paper/ieeg_protocol.md`](paper/ieeg_protocol.md).
- Full record including voided findings:
  [`paper/causal_detection_log.md`](paper/causal_detection_log.md).
- All datasets are public and fetched by the scripts; none are redistributed
  here.
- Archived releases: [doi:10.5281/zenodo.21988145](https://doi.org/10.5281/zenodo.21988145).

This work is personal, unfunded, and unconnected to any employment. Its aim
is that the measurements above be *useful* — which is why the failures are
documented with the same care as the successes.
