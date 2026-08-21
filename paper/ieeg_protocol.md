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


## Same-scale sensitivity arms for P-S1 (declared 2026-08-20, POST-HOC)

STATUS OF THIS DECLARATION. The SOZ labels are already open. Nothing below
can be confirmatory, and it is not offered as such. This is a post-hoc
robustness check, declared in full before it is run so that the definitions
cannot be tuned against a known answer. The pre-registered result stands as
recorded whatever this produces.

WHY. The declared raw-reference sensitivity arm turned out not to be a
sensitivity arm at all: it correlates with bipolar at a median Spearman rho
of 0.078 across the 28 subjects, so it measures a different quantity (a
referenced far-field signal rather than a local gradient) and its null is
uninformative about P-S1 (ledger, rules 27-28). A real sensitivity arm must
vary the derivation while holding the spatial scale fixed.

THE ARMS, fixed now.

  LAPLACIAN. For contact n on a shaft, x_n - (x_{n-1} + x_{n+1}) / 2, using
  the two immediately adjacent contacts on the same shaft. Contacts at
  either end of a shaft, and contacts whose neighbours are absent or were
  dropped by the variance guard, are excluded. Channel label: SHAFTn_lap.

  BIPOLAR-SKIP. x_{n+2} - x_n on the same shaft: the same difference
  operator as the confirmatory arm at twice the contact spacing. Label:
  SHAFTn+2-SHAFTn. This varies the spacing while keeping the derivation.

VALIDITY GATE, applied BEFORE any P-S1 test on these arms. Each arm is
mapped to the bipolar channels sharing its contacts and the per-subject
Spearman correlation of excess is computed. An arm counts as same-scale,
and therefore as a legitimate sensitivity check, only if its median
correlation with bipolar across subjects exceeds 0.30. An arm failing this
gate is reported as failing it and its P-S1 result is NOT interpreted as
evidence either way - the same treatment the raw arm now receives.

THE TEST, unchanged from the pre-registration. One-sided Mann-Whitney on
per-channel excess, SOZ ranked below non-SOZ, per subject; Stouffer's z
weighted by sqrt(channels); sign test reported alongside; the same
minimum-group-size exclusion of 3.

WHAT COUNTS AS WHAT, fixed now.

  CORROBORATION: an arm passes the validity gate AND its pooled z is
  positive with p < 0.05 one-sided.
  FAILURE TO CORROBORATE: an arm passes the validity gate and its pooled z
  is not positive at p < 0.05. This would be substantive evidence against
  the bipolar result and will be reported as such, in the abstract of any
  write-up, not buried.
  UNINFORMATIVE: an arm fails the validity gate.

No further arms will be added after these two. If both fail to corroborate,
the P-S1 finding is downgraded from suggestive to unsupported in the ledger.
