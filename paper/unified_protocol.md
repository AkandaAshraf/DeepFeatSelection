# Pre-registration: one model, both statistics, and the axis that should kill it

Declared 2026-09-05, before the experiment was written or run.
EXPLORATORY. Nothing can be adopted on this run whatever it shows.

## The unification, and why it is one thing rather than three

Three approaches were run over two days and they are not three competitors.

  AUTOENCODER   masked autoencoder code, ridge readout, INFLOW. The
                incumbent, with months of validation behind it.
  RESNET        learned residual trunk as self-baseline with an identity
                path, INFLOW. Measured a no-op against the incumbent:
                +0.003 in ranking, 2 cells of 6, p = 1.000.
  U-NET         forecasting network with per-variable skips, ablation
                readout, OUTFLOW. Verdict uninformative; its masked-skip arm
                was the only one that graded with drive magnitude.

The ResNet shortcut and the U-Net skip are the SAME construction at different
scales, per-target and all-targets-at-once, and the masked autoencoder is the
encoder inside both. So one architecture yields both statistics:

  stage 1   per-variable skip paths ONLY, trained as an own-history
            forecaster for every variable, then FROZEN.
  stage 2   bottleneck and decoder, masked training, added to the frozen
            skip outputs.

  inflow(q)   = R2_q(stage 2 full) - R2_q(stage 1 skip alone)
  outflow(j)  = mean over k != j of [ R2_k(full) - R2_k(input j zeroed) ]

Freezing stage 1 is not a detail. Trained jointly, the optimiser may underfit
the skips and let the bottleneck carry prediction that own history could have
supplied, manufacturing inflow on autonomous channels. The ResNet run
established that a frozen two-stage fit is what keeps the self-baseline
honest.

This is the inflow-by-outflow quadrant map the 2026-08-20 entry called "the
right shape" while rejecting the outflow that was then available. Sinks are
high inflow and low outflow, sources the reverse, isolated channels low on
both, mediators high on both.

## What combining puts at risk

The incumbent inflow statistic carries a physical ground truth, a head-to-head
against convergent cross mapping, a 51-cell boundary map and an audit gate.
Computing inflow from the unified model instead REPLACES that validated
component with an unvalidated equivalent, and the ResNet run already measured
the replacement as buying nothing.

So non-inferiority on inflow is DISQUALIFYING here. A combination that
detects outflow beautifully while degrading inflow trades validated ground
for a pilot result, and this protocol refuses that trade in advance.

A second risk is shared failure. Two statistics from one model fail together
where two machineries fail independently, and Mechanism 3 records that
consensus is not evidence of validity. The ghost channel remains external to
the model and is retained as the artifact meter for exactly this reason.

## The design

  ARMS      UNIFIED     the two-stage model above, both statistics
            RIDGE       incumbent inflow, poly3 + code, ridge readout
            ADD         incumbent-style additive conditional outflow
  SYSTEM    boundary_map.make_system, V = 30, n = 4000, coupling 0.20,
            bottleneck 2V
  NOISE     {0.0, 0.05}. NOT 0.30, where the same day's run put every arm at
            or below chance (Rule 122).
  REDUNDANCY {0, 2}. THE DECISIVE AXIS. Duplicates carry a source's signal,
            so ablating a source leaves its duplicate and the model routes
            around it. This is the condition under which leave-one-out is
            expected to die, and it is the reason this run exists rather
            than a repeat of yesterday's.
  SEEDS     0, 1, 2

12 cells.

## Metrics and their chance levels, computed before committing

Rule 123. The driven class is 25 of 30 channels, so average precision with
driven as positive has a base rate of 0.833 and a lift ceiling of 1.2x. It
CANNOT demonstrate discrimination and is reported as secondary only.

PRIMARY for both statistics: average precision for identifying the MINORITY
class, sources, at base rate 0.167 and a lift ceiling of 6.0x. The inflow
statistic is scored on its negation, since a source is what inflow should
rank last. Both statistics then sit on one scale and are directly comparable.

## Predictions, fixed now

  N1  DISQUALIFYING. At redundancy 0 and noise 0.0, the unified model's
      inflow average precision is within 0.05 of the RIDGE incumbent's. If
      the combination costs validated inflow performance it is REJECTED
      whatever it does for outflow.
  N2  DECISIVE. At redundancy 0, unified outflow clears twice base rate,
      0.333, replicating the masked-skip result on cells that include a
      redundancy manipulation it has not seen.
  N3  DECISIVE, AND PREDICTED TO GO AGAINST US. At redundancy 2, outflow
      average precision FALLS relative to redundancy 0. The mechanism is
      stated above and is the documented reason leave-one-out dies. If it
      falls to within 0.05 of the 0.167 base rate, leave-one-out outflow is
      CLOSED for redundant systems, which is every real system.
  N4  GRADED CONTROLS, declared in advance this time rather than added after
      (Rule 125). Inflow must track coupling on driven channels. Outflow
      must track child count among sources, recovered by replaying each
      seed's generator draws. A statistic that separates classes without
      tracking magnitude is not measuring what it claims.
  N5  GUARD, SCOPED PER ARM (Rule 124). Each model's own held-out forecast
      R2 must exceed 0.5. An arm that fails is disqualified; the run is not.
  N6  NO PREDICTION on the quadrant map's four-way separation. It is
      reported descriptively.

## The rule, fixed now

NONE, and no adoption is possible on this run. Outcomes:

  REJECTED      N1 fails. The combination costs inflow and is abandoned;
                the two statistics stay on separate machinery.
  CLOSED        N3 fires its kill condition. Ablation outflow dies under
                redundancy and the unification keeps only its inflow half,
                which the ResNet run already showed buys nothing. In that
                case the whole line closes and the incumbent stands alone.
  OPEN          N1 holds, N2 holds, and N3 falls without reaching the kill
                condition. Licenses one powered pre-registration at more
                widths and a second generating family. Nothing enters the
                manuscript.

## Void conditions

Void if stage 1 is not frozen before stage 2 is trained; if ablation is
computed anywhere but the held-out test segment; if the arms do not share the
generated system and splits within a cell; if the grid, seeds, noise or
redundancy levels change after any result is seen; if N1 is judged anywhere
other than redundancy 0 and noise 0.0; or if either graded control is dropped.
