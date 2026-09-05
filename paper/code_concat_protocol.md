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

---

## Result (2026-09-05): C1 holds by the letter; C2 makes it moot.

27 cells, 1.5 min. Recall, median over three seeds:

  V      AVERAGE   CONCAT   WIDE
  30       0.88     0.92    0.92
  60       0.18     0.24    0.30
  120      0.23     0.24    0.24

  C3 HOLDS: precision 1.000 and source false positives 0.000 in every arm
     and every cell. Nothing here was bought by flagging sources.
  C4 HOLDS: ghost clean in all 27 cells.
  C1 HOLDS at V=60 by the declared median criterion, 0.24 against 0.18.

### C1 holds weakly, and the per-seed detail matters

  seed      AVERAGE   CONCAT   WIDE
    0         0.18     0.24    0.30
    1         0.18     0.46    0.46
    2         0.14     0.12    0.22

CONCAT beats AVERAGE in two seeds of three and is WORSE in the third. On
n = 3 with a reversal, the median is thin evidence and is reported as such.

### C2 had no prediction and resolves decisively: width, not diversity

**WIDE is at least as good as CONCAT in three seeds of three, and better in
two, while costing HALF the encoder time** (2.5 s against 5.1 s). Against
AVERAGE it wins in three of three. One encoder at width 64 dominates two
encoders at width 32 on recall and on cost simultaneously.

The mechanism is visible in the numbers. Concatenating two independently
trained 32-wide codes gives the readout 64 columns, but the two codes are
partly redundant with one another, so it buys less than a genuine 64-wide
code. CONCAT at 0.24 sits between b = 32 (0.18) and b = 64, which is exactly
what partial redundancy predicts.

### An independent reproduction of the bottleneck study

The bottleneck grid measured V = 60 at b = 32 -> 0.18 and b = 64 -> 0.34.
Today, on a different code path, AVERAGE (readout width 32) gives **0.18
exactly** and WIDE (b = 64) gives 0.30. The capacity curve reproduces.

### What we actually do, and why it is not what the rule says

The declared rule fires ADOPT: C1 held with C3 and C4 intact. **We are not
adopting concatenation**, and the reason is that C2 -- the contrast the
protocol explicitly declined to predict -- dominates the contrast C1 was
about. Making concatenation the default when a cheaper arm beats it in every
seed would be following the letter of a criterion the experiment has
outgrown.

What the experiment supports:

  - Part of the published recall shortfall IS an aggregation artefact.
    AVERAGE leaves readout capacity unused, and both alternatives recover
    some of it at no cost to precision or source blindness.
  - The remedy is WIDENING, which Section 12.4 already recommends as
    b ~ 2V. Concatenation is an inferior partial substitute for it.
  - A genuinely new and unregistered observation: MODELS = 2 at width b
    appears dominated by MODELS = 1 at width 2b, on recall AND on encoder
    cost. That would change a deployed default, so it is NOT changed here.
    It needs its own pre-registration, more seeds, and an account of what
    the ensemble was for -- variance reduction and the cross-model spread
    diagnostic, neither of which recall measures.

### Not established

Three seeds, one coupling, one generating family, three widths. The
dominance of WIDE over CONCAT is consistent across all three seeds but three
is not many, and at V = 120 all three arms are within 0.01, so the effect is
concentrated at V = 60 where b/V leaves the most headroom.
