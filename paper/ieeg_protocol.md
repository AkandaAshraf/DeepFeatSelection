# Pre-registration: MACE drivenness maps on intracranial EEG

Declared 2026-08-17, before any recording, channel table or label from either
dataset was downloaded. Datasets: OpenNeuro ds003876 (interictal, 39
subjects) and ds003029 (ictal runs, ~35 subjects on the public mirror, with
channel-level seizure-onset-zone annotations and surgical outcomes). Both
fully open.

## Why this data

The data shape is the best match MACE has yet met on real recordings:
50-150 simultaneous channels at kHz sampling, so n is two orders of
magnitude above the validated floor. If the self-baseline can saturate on
any real data, it is here, and Proposition 1's premise becomes testable on a
real deployment for the first time.

## The scientific frame, fixed now

MACE is blind to sources by design. The seizure-onset zone is a source by
clinical definition. In the companion paper this had to be withdrawn as an
untested assertion because scalp data had no channel-level truth; ds003029
has that truth, so the blindness becomes a pre-registered PREDICTION rather
than a claim:

  P-S1. SOZ channels are DEPLETED from the interictal driven core relative
        to non-SOZ channels of the same subject (one-sided, per-subject,
        pooled by meta-analysis over subjects).
  P-S2. The driven core concentrates in non-SOZ channels anatomically
        adjacent to the SOZ (spread territory) more than in distant
        channels, where adjacency is defined from electrode naming groups,
        not coordinates, and this is declared a WEAK prediction.
  P-S3. Outcome anchor: in surgical-failure patients the SOZ-depletion
        effect (P-S1) is weaker than in success patients - if the labelled
        SOZ missed the true source, the labels are wrong precisely there.
        Declared exploratory: outcome subgroup sizes are unknown at
        declaration time.

No prediction requires MACE to find sources. Silence remains uninformative
by the companion paper's own recall result.

## Gate first (this document precedes the gate run)

The standing gate, applied to 2-3 interictal subjects before anything else:

  G1 length: interictal segments give n >= 60,000 samples at native rate;
     after the declared decimation, n >= 15,000. PASS expected by
     construction; recorded, not assumed.
  G2 saturation: self-R2 distribution per channel. PREDICTION: substantially
     higher than calcium imaging (field potentials are smoother and faster
     sampled); whether any channel exceeds 0.9 is the open question that
     decides if the corrected ghost-donor rule is usable on real data.
  G3 ghost panel: 50 donors, circular shifts from the middle half. If the
     panel median sits clearly above zero the segment is non-stationary and
     the run is discarded per the companion paper's rule.
  G4 shared-reference floor: iEEG channels share a recording reference, and
     a common reference is shared signal, which MACE will read as
     drivenness, exactly as shared motion was read in the freely-moving
     worm (ledger 2026-08-16). There is no GFP-equivalent here. Declared
     mitigation: common-average re-reference before analysis, plus a
     REFERENCE-FLOOR control: the same pipeline run on the same segment
     WITHOUT re-referencing; the difference bounds the reference artifact.
     If CAR-referenced ghosts stay clean and the no-CAR run shows gross
     inflation, the artifact is understood and controlled; if CAR does not
     clean it, the platform fails the gate and we say so.

Pipeline constants, declared: decimate by mean-pooling to ~256 Hz; E=3,
tau=1 at the decimated rate; per-channel standardisation on train
statistics; first-differencing; contiguous 0.6/0.2/0.2 splits with
embedding-span embargo; degree-3 polynomial self-baseline; ridge alpha 1;
masked-AE ensemble M=4, bottleneck 32, mask 0.25, Adam 3e-3, batch 64, 25
epochs; ghost donors filtered to self-R2 > 0.9 where at least 8 channels
qualify, uniform otherwise with the fallback flagged.

## What is NOT opened before the gate

No channels.tsv column describing SOZ, resection, or outcome is read during
the gate. Channel tables are parsed for names, types and sampling rate
only. The SOZ columns are opened only after the gate verdict and only for
the pre-registered tests above.

## Void conditions

The study arm is void if: the gate fails on all tried subjects (G3 or G4);
or the SOZ analysis is run on any subject whose gate was not clean; or
predictions are altered after any SOZ label has been seen.

## Breach log (2026-08-17, same session as the gate)

While verifying a re-download, a shell `head -2` of sub-NIH1's channels.tsv
displayed the header row and the FIRST CHANNEL's label columns (RAI1: soz
yes, rz yes) before the gate had run. The predictions above were declared
and timestamped before any file from either dataset was downloaded, so they
cannot have been shaped by the exposure; the void clause on altering
predictions is not triggered. Handling, effective immediately: sub-NIH1 is
QUARANTINED from the confirmatory P-S1/P-S2/P-S3 cohort and may serve the
gate and exploratory analyses only, with this exposure noted wherever it
appears. Confirmatory subjects are drawn from the remaining cohort with
channel tables read by scripts that select name and type columns only;
channel tables are never opened with shell tools again (rule added to the
ledger).

## Montage amendment (2026-08-17, declared before any SOZ label was opened)

The three-way montage comparison is complete (ledger, same date). Primary
montage for the confirmatory analysis: BIPOLAR (G3 pass on all six gate
subjects; structurally cancels the shared reference and far-field volume
conduction; most selective maps). RAW is the sensitivity arm; CAR is dropped
after producing the cohort's only stationarity failure and its largest ghost
maxima. Confirmatory inclusion requires a G3-clean bipolar gate for the
subject's segment. All prior declarations, including the P-S1..P-S3
predictions and the NIH1 quarantine, stand unchanged.

## Analysis specification (2026-08-20, declared before any label was opened)

The predictions were frozen before download; the montage was chosen on gate
evidence alone. Neither anticipated how a BIPOLAR derivation inherits a
contact-level label, so those rules are fixed here, before the soz, epz and
rz columns are read for the first time.

Label inheritance. A bipolar channel SHAFTm-SHAFTn spans two contacts. It is
counted SOZ if EITHER constituent contact is marked soz, because a
derivation touching the onset zone is not a clean non-SOZ observation. The
same rule applies to rz (resection) and epz (early propagation). Declared
now because the permissive rule works AGAINST P-S1: it moves borderline
channels into the SOZ group, diluting any depletion effect.

P-S1 test. Per subject, a one-sided Mann-Whitney U on per-channel excess,
SOZ ranked BELOW non-SOZ. Ranks, not thresholds, so the result does not
depend on where the driven-core cut is placed. Effect size is the rank
biserial correlation. Subjects with fewer than 3 SOZ or 3 non-SOZ bipolar
channels are excluded and the exclusion reported. Pooling across subjects is
Stouffer's z weighted by sqrt(number of channels). A per-subject sign test
on the direction of the effect is reported alongside, as it assumes less.
Secondary, threshold-based: the proportion of SOZ vs non-SOZ channels above
the subject's own ghost-panel threshold.

P-S2 (WEAK). Among non-SOZ channels only: those on a shaft that carries at
least one SOZ contact (adjacent, spread territory) versus those on shafts
with none (distant). Same one-sided rank test, adjacent ranked ABOVE
distant. Reported as weak whatever it shows.

P-S3 (EXPLORATORY). Per-subject P-S1 effect sizes split by surgical outcome
from participants.tsv, success versus failure under the file's own coding,
compared with a rank test. Subgroup sizes are unknown at declaration.

Arms. Bipolar is confirmatory. Raw reference is the sensitivity arm and is
reported whatever it shows. CAR is not run: it failed the gate on four
subjects.

Multiplicity. Three predictions, each with one primary test. P-S1 is the
confirmatory result; P-S2 and P-S3 carry their declared weak/exploratory
status and are not corrected against P-S1.

Cohort. The 28 G3-clean bipolar subjects fixed by the gate. NIH1 remains
quarantined. No subject may be added or removed after this point.
