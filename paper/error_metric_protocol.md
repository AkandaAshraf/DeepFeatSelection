# Pre-registration: does the error metric decide what MACE can see?

Declared 2026-08-23, before the experiment was written or run.

## Why

Every statistic in this project is a difference of squared-error R2 from a
linear ridge probe. That choice has never been varied. It was inherited from
the first implementation and has been held fixed through the boundary map,
the bottleneck study, the source-detection line and every real-data scan.

The immediate motivation is the outflow failure of the same day. A bar
calibrated on the ghost null admits sinks 28/30 because a sink CARRIES A
PROXY of its driver, so masking it removes information the code was using.
Three attempts to fix that with thresholds have now failed. If the sink
proxy and the source signal differ in KIND rather than in size, no threshold
on a single metric will separate them and only a different measurement can.

If no metric separates them, that is worth knowing too, and it closes the
line for a principled reason rather than a threshold one.

## Two readings, both covered

"The error of the produced output from the autoencoder" can mean the error of
the probe fitted on the code, or the error of the reconstruction the network
actually emits. Both are tested. One training run per cell serves both, so
covering both costs almost nothing over covering one.

## The metrics

FAMILY A - error of the READOUT on the code. One ridge fit per case; A1-A4
are all computed from the same predictions, so they differ only in how error
is scored.

  A1  R2            squared error. THE CURRENT METRIC, the baseline.
  A2  normalised MAE   1 - MAE/MAE(mean predictor). Robust: scores typical
                       error rather than extreme error.
  A3  Spearman      rank agreement between prediction and target. Scale-free
                    and insensitive to monotone distortion.
  A4  Gaussian entropy reduction   0.5*log(var_y/var_resid), in nats.
                    EXPECTED DEGENERATE with A1 - it is a monotone transform
                    of R2 for Gaussian residuals - and declared as such now
                    so that a null result cannot later be presented as a
                    finding, and a NON-null result would mean the residuals
                    are far from Gaussian.
  A5  nonlinear readout   R2 with random Fourier features appended to the
                    channel features. Tests whether the code's information is
                    linearly accessible, which is what Proposition 2 says the
                    implemented readout only lower-bounds.

FAMILY B - error of the RECONSTRUCTION the network emits. No probe at all.

  B1  outflow_recon(q) = mean over channels r != q of the increase in the
      network's own reconstruction error for r when q is masked at the input,
      relative to no masking. Squared error.
  B2  as B1 with absolute error.

  This is a different statistic, not a different scoring of the same one: it
  uses the quantity MACE currently computes and discards.

FAMILY C - a change of CONDITIONING, not of error function, stated plainly as
such because it is the only candidate here that targets the confound directly.

  C1  conditional outflow. The baseline includes the features of ALL other
      channels, so q's contribution is measured as UNIQUE information rather
      than as marginal gain. Proposition 1 says such a quantity is zero for a
      channel that is redundant given the others - which is exactly what a
      sink is. The risk, and the reason no prediction is made about it, is
      that a source recoverable from its own sinks is redundant too.

## Design

System: coupled logistic, the one used throughout this line, n = 4000, b = 64,
25 epochs, 2 models. Couplings 0.30, 0.50 and 0.70; seeds 0-4. 15 cells. The
autoencoder is trained ONCE per cell and every metric is computed from it, so
the metrics are compared on identical codes and identical data.

Primary outcome, per metric per coupling: the AUC of ranking SOURCES above
SINKS by that metric's outflow. AUC is used rather than a margin because the
metrics are on different scales and no common threshold exists between them.
Source-versus-isolated AUC is reported alongside it.

Every metric carries its own ghost, computed on that metric's own scale.

## Predictions, fixed now

  M1  A1 reproduces what is already known: source-versus-sink AUC high at
      coupling 0.50 and DEGRADING at 0.70, where the sink proxy is strongest
      (+0.0027 to +0.0062 against a flat source signal). If A1 does not
      reproduce, the experiment is void.

  M2  DECISIVE. At coupling 0.70, at least one metric reaches source-versus-
      sink AUC >= 0.90. NO PREDICTION as to which.

  M3  A4 lands within 0.02 AUC of A1 at every coupling. Declared as the
      expected-degenerate case.

  M4  NO PREDICTION on A2, A3, A5, B1 or B2.

  M5  C1 drives SINK outflow to its ghost level. NO PREDICTION on whether
      sources survive it. If sources collapse with the sinks, the redundancy
      is irreducible and no conditioning can separate them.

  M6  DISQUALIFYING. A metric whose ghost is not clean on its own scale is
      excluded regardless of its AUC. Ghost cleanliness is judged per metric
      as the ghost sitting below the 5th percentile of that metric's source
      distribution, since 0.005 is an R2-scale number and does not transfer.

  M7  LIVE OUTCOME, declared now. No metric reaches M2. Then the sink proxy
      is informational rather than an artefact of squared error, no error
      measure distinguishes "removed information that mattered" from "removed
      a copy of information that mattered", and the outflow line CLOSES on
      that ground. This is recorded in advance so that it is a result rather
      than an abandonment.

## Void conditions

Void if M1 does not reproduce, if metrics are added or dropped after any AUC
is seen, if a metric's ghost is judged by a bar other than the one above, if
the couplings or seeds are changed after a result, or if the winner under M2
is selected and then reported without the other metrics beside it.
