# Pre-registration: is the ensemble leaving readout capacity unused?

Declared 2026-09-05, before the experiment was written or run.

## The observation this tests

Capacity is the binding constraint on recall. The bottleneck study measured
it directly: at V = 60, widening the code from 32 to 128 takes recall from
0.18 to 0.78 with precision unchanged at 1.00. Recall as low as 0.09 has
been measured at the deployed width.

The deployed estimator trains `MODELS = 2` encoders and then **averages the
two gains**, so each ridge readout sees only one width-32 code:

```
excess(q) = mean_m [ R2(own lags + code_m) - R2(own lags) ]
```

Two encoders are already trained and paid for. Concatenating their codes
instead would give the readout 64 columns for **exactly the same encoder
training cost**, since the encoders are identical in both cases:

```
excess(q) = R2(own lags + code_1 + code_2) - R2(own lags)
```

If capacity is the binding constraint, this is free recall. If it is not,
averaging is doing something the concatenation loses and that is worth
knowing.

The theory says it could go either way, which is why this is worth running.
Section 5 argues the raw alternative fails because estimation variance grows
with $VE/n$ while the code carries $b/n$; doubling the readout width doubles
that term. Capacity and estimation variance pull in opposite directions here.

## Design

Three arms. A and B share the same two trained encoders, so their comparison
is exactly paired -- the only difference is how the readout uses them:

  A  AVERAGE   M = 2 encoders at b = 32; mean of the two gains.
               Readout sees 32 columns. THE INCUMBENT.
  B  CONCAT    the SAME two encoders; codes concatenated in one readout.
               Readout sees 64 columns. Zero extra encoder cost.
  C  WIDE      M = 1 encoder at b = 64; single gain.
               Readout sees 64 columns, from one encoder rather than two.

C is the control that separates the two explanations. If B and C are
comparable, the gain is readout width and one could simply widen b. If B
exceeds C, ensemble diversity contributes something width alone does not.

  WIDTH   V in {30, 60, 120}
  SEEDS   0, 1, 2
  SYSTEM  boundary_map.make_system, n = 4000, coupling 0.20, redundancy 0 --
          the boundary map's centre, so the incumbent arm should reproduce
          the published recall of 0.88 / 0.18 / 0.23.

Each arm computes its own ghost panel with its own aggregation, so the
threshold is internally consistent within an arm. 27 cells.

## Predictions, fixed now

  C1  DECISIVE. CONCAT beats AVERAGE on recall at V = 60, the
      capacity-limited cell where the incumbent sits at 0.18 and has the
      most headroom. This is the prediction the capacity account makes.
  C2  NO PREDICTION on CONCAT versus WIDE. That is the diversity question
      and predicting it either way would be storytelling.
  C3  DISQUALIFYING. Precision stays at 1.00 and the source false-positive
      rate stays at 0.000 in every arm and every cell. If concatenation buys
      recall by flagging sources, it is REJECTED whatever it does for
      recall: source blindness is the property the method exists on, and
      Rule 69 records that choosing "most recall" as the criterion would
      have broken it once already.
  C4  The ghost panel stays clean (median <= 0.005) in every arm.
  C5  NO PREDICTION at V = 30, where the incumbent already reaches 0.88 and
      there is little headroom to detect a difference.
  C6  Wall-clock is recorded per arm. B is expected to cost the same as A up
      to one extra ridge solve per channel; if it does not, the "free"
      framing is wrong and must be corrected.

## The rule, fixed now

  ADOPT   C1 holds and C3 and C4 hold. Concatenation becomes the default
          aggregation, the deployed recall figures are re-derived under it,
          and the manuscript's capacity section reports that part of the
          published recall shortfall was an aggregation choice rather than a
          property of the statistic.
  REJECT  C3 fails. Recorded as an attempted improvement that broke source
          blindness, and averaging stands.
  NULL    C1 fails with C3 intact. Averaging is not leaving capacity on the
          table; the ensemble is doing something other than supplying width,
          and the recall limit is not an aggregation artefact. Reported as a
          negative result and the incumbent stands.

## Void conditions

Void if arms A and B are computed from different encoders (they must share
the identical two), if the grid or seeds change after any result is seen, if
the ghost panel for an arm is computed with a different aggregation than
that arm's excess, or if C1 is judged anywhere other than V = 60.
