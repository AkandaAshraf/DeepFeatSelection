# Forecast-horizon adequacy check: does each model's skill decay with lead time?

Registered 2026-09-23, before the script exists. Small diagnostic, no GPU-heavy
training, no manuscript claim. Question: which of the model classes this
project has been using are adequate one-step dynamical models of a single
channel, judged by how their out-of-sample skill behaves as the forecast
horizon grows. This is the same diagnostic empirical dynamic modelling uses
to choose the embedding dimension for convergent cross-mapping (simplex
projection, skill against prediction horizon; Sugihara and May 1990).

## Why the criterion is principled, not taste

For a stationary process and a fixed conditioning set (here: the channel's own
last E values), the mean squared error of the optimal h-step predictor
E[x(t+h) | past] is non-decreasing in h. So a model's test skill that RISES
with horizon by more than estimation noise means that model is not the right
predictor at the shorter horizon. A model whose skill stays FLAT with horizon
on these generators is also suspect: every system here has finite memory
(logistic maps are chaotic; the AR(2) parts have all roots inside the unit
circle), so predictability must be lost as h grows, and flat skill indicates
information from the future. Smooth decay itself is the chaos signature of
Sugihara and May and is REPORTED, not required: the AR family can decay
non-smoothly and still be correctly modelled.

## Systems

The two generator families of paper/parent_screening_protocol.md, unchanged
code (scripts/parent_screening.py family1_generate, family2_generate), V=24,
n=4000, observation noise as registered there. Fresh seeds, disjoint from
every earlier block: family 1 26001-26004, family 2 27001-27004. All 24
channels of each system are targets (roots included: they have their own
dynamics). One additional system per family is NOT a dynamical system:
i.i.d. Gaussian noise of the same shape (seeds 26091, 27091), the negative
control.

## Rows, split, horizons

Own-lag window of E values ending at t, target x(t+h), horizons h = 1..10.
Rows t are restricted so every horizon uses the same rows. Contiguous
60/20/20 train/validation/test with an embargo of E+10 rows at each seam
(covers the widest window plus the furthest target). Standardisation with
train-row statistics only. Validation selects anything that is selected;
every reported skill is on TEST rows. Skill = unclipped R2 against the test
mean (0 = climatology); persistence x(t+h)=x(t) is reported alongside.

## Models (one fit per target per horizon, direct h-step)

- M1 SIMPLEX: simplex projection, E+1 nearest neighbours in the train
  library, exponential distance weights. E chosen per target from 1..6 by
  validation skill at h=1 (the standard CCM rule), then fixed for all h.
- M2 LINEAR: ridge on the last 3 values, intercept, alpha by the embargoed
  internal split already used in the project (PS.ridge_r2_val).
- M3 POLY3: the project's standard readout, ridge on poly3 of the last 3
  values, same alpha rule.
- M4 MASKED-AE: the project's masked autoencoder (PS.train_group_encoder,
  group of one channel, bottleneck 2) trained once per target on train rows,
  its code concatenated with poly3 features, same ridge. The learned model
  class this project uses.
- C-LEAK (deliberately broken, must fail): M3's features taken from the window
  ending at t+h-1 instead of t, i.e. always a one-step forecast. Proves the
  flat-skill criterion catches leakage.

## Adequacy criterion, frozen

A (model, target) is ADEQUATE iff all three hold on test skill s(h):
  (a) LEARNS: s(1) >= 0.10 and s(1) > persistence skill at h=1;
  (b) NON-INCREASING: s(h+1) <= s(h) + 0.02 for every h in 1..9 (0.02 is the
      allowance for estimation noise between separately fitted horizons);
  (c) DECAYS: s(1) - s(10) >= 0.05.
A model class is ADEQUATE ON A FAMILY if at least 80% of that family's
dynamical targets (96 = 4 seeds x 24) are adequate. Reported per model per
family with counts, plus which criterion fails.

Controls that must behave, or the check itself is void:
  - every model on the i.i.d. noise systems fails (a) on at least 95% of
    channels (chance level: s(1) ~ 0);
  - C-LEAK fails (c) on at least 80% of dynamical targets where M3 passes (a).
If either control misbehaves, no model verdict is reported.

## Reporting and limits

Per model and family: adequacy rate, failure breakdown by criterion, median
s(h) curve, and for M1 the chosen-E distribution. This is a diagnostic of
one-channel forecasting adequacy on two synthetic families. It does not
re-score any earlier result, and does not by itself say why an earlier result
held or failed; a model class that fails here is a candidate explanation to
test, not a proven cause. No threshold or model changes after results.

## Resources

CPU plus light GPU (192 tiny encoders); `.agent-lock` held; expected well
under 30 minutes; free RAM checked before launch and >= 2 GiB throughout;
outputs under ExpOutput/forecast_horizon_check/ (compact CSV/JSON, < 5 MB).

## Pre-run amendment (2026-09-23, before the script exists or any system is generated)

Added at the user's direction, because the autoencoder is the component meant
to raise screening recall and the encoder that actually screened in the
parent-screening pilot is the GROUP encoder, not a one-channel one:

- M5 GROUP-AE: the size-capped clustering of the standardised TRAIN span
  (PS.cluster_size_capped), one masked autoencoder per group
  (PS.train_group_encoder, per-member masking, bottleneck min(8, 2|G|),
  encoder seed = group id), trained once per system on train rows; for
  target q, its own group's code concatenated with poly3 of q's own last 3
  values, same ridge. Same adequacy criterion and 80% bar as M1-M4. The
  code includes q's own window, so this is a forecasting-adequacy test of
  the representation, not a parent-screening score.

Nothing else changes.
