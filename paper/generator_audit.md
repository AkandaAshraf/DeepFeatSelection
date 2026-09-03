# Audit of the coupled-logistic generator (2026-09-03)

Post-hoc. Nothing here was pre-registered; it was prompted by the failing
seeds of the reopen run and by the ratio sweep's uninterpretable ratio-1
cell. Script `scripts/generator_audit.py`, output
`ExpOutput/generator_audit/` and `ExpOutput/generator_audit_run.log`. The
pre-registered consequence is `paper/clean_generator_protocol.md`.

The generator under audit is `coupled()` in
`scripts/source_outflow_coupling.py`, used unchanged by every outflow
experiment since 2026-08-22: the capacity, coupling and maturity gates, the
outflow bar, the sink-calibrated bar (the closure), the crossed saturation
run, the chamber-shape run, the reopen, the conditional-outflow run and the
ratio sweep. Sources and isolated channels are logistic maps with
r ~ U(3.7, 3.9); sinks are logistic maps with r ~ U(3.5, 3.7) plus
`coupling * x[parent] * (1 - k)`, clipped to [0, 1]; each sink's parent is a
uniform random source; 0.01 observation noise.

## 1. A fifth of the source parameter range is phase-locked

Scanning r over [3.7, 3.9] at 4001 points, with a source called locked when
its Lyapunov exponent is <= 0.05 or its max |autocorrelation| over lags 1-12
is >= 0.90:

  locked fraction 0.192
  windows: [3.7016, 3.7028] (period 7), [3.7382, 3.7448] (period 5 and its
  band), [3.8284, 3.8569] (period 3 and its band)

  P(at least one locked source per run):
    n_src   1     2     3     6     12
            0.19  0.35  0.47  0.72  0.92

The period-3 window at 3.8284 is the textbook one; the generator's range was
chosen as "the chaotic regime above 3.57" without excluding it. An earlier
check with a lag-3-only detector found 14.2%; it missed the period-5 window,
which is where two of the reopen's dead runs (seeds 801, 829: r = 3.7435 and
3.7398) sit.

## 2. Every dead run has a locked source

Replaying the generator on the recorded seeds and testing the observed
source series (max |ac| >= 0.90):

                     runs with a    dead runs      of which with
                     locked source  (source<0.002) a locked source
  sink-bar 3/6/6     26/60          8              8
  reopen   2/11/2    24/60          4              4

  median source outflow by number of locked sources per run
                     0        1        2
  sink-bar 3/6/6     0.01247  0.00785  -0.00006   (n = 34, 22, 4)
  reopen   2/11/2    0.02655  0.01737  -0.00006   (n = 36, 20, 4)

Restricting to runs with no locked source, post hoc:

  sink-bar 3/6/6   declared            bar +0.01051   16/30 = 0.53 [0.36, 0.70]
                   clean test, same bar               12/17 = 0.71 [0.47, 0.87]
                   clean cal and test  bar +0.00557   17/17 = 1.00 [0.82, 1.00]
  reopen 2/11/2    declared            bar +0.01498   25/30 = 0.83 [0.66, 0.93]
                   clean test, same bar               19/19 = 1.00 [0.83, 1.00]
                   clean cal and test  bar +0.01585   19/19 = 1.00 [0.83, 1.00]

These are subsets selected after the fact, on 17-19 runs; they say the
closure is in doubt, not that it is overturned. Note in particular that at
the SAME bar the clean 3/6/6 subset reaches only 0.71: the closure's fate on
a clean generator depends on the bar moving when the calibration block is
clean too. That is what the pre-registered re-run measures.

## 3. The paper's own fitness gate disqualifies locked sources

For every sink in the reopen test seeds, the gate quantity the paper uses
to decide whether a dataset can be tested at all -- the R2 gain from adding
the parent's lags to the sink's own lags, poly2 features, E = 3:

  parent           n     own-lag R2   dR2 median   IQR
  chaotic          262   0.9567       +0.0211      [0.0178, 0.0268]
  locked           68    0.9981       +0.0005      [0.0004, 0.0011]
  calibrated gate                     L50 +0.0136, L30 +0.0114

A sink of a periodic parent predicts itself from its own lags to R2 0.998
and gains nothing from the parent. The manuscript's saturation premise
(Section 2) is that a source is detectable because its targets cannot be
predicted from their own past; a locked source violates that premise
exactly, and the gate that screens real datasets for it would reject these
sinks. The generator has been feeding the statistic systems the paper's own
gate would refuse.

## 4. Two further defects, not fixed in the re-run

CLIPPING. At coupling 0.50 the sink update `r k(1-k) + 0.5 x_parent (1-k)`
exceeds 1 often enough that 83% of sinks spend more than 30% of steps at the
[0, 1] boundary (0% at coupling 0.30, 100% at 0.70; 9 of 30 reopen runs
have every sink clipped). A clipped sink carries less of its parent's signal
than the coupling term suggests. Every outflow gate since the coupling sweep
has run at 0.50.

ORPHANS. The parent assignment is a uniform draw with replacement, so a
source may receive no sink. Such a source influences nothing and its
outflow is correctly zero -- but it is counted as a source in sensitivity
and AUC. In the ratio sweep:

  cell (src/sink/iso)  orphans per run  effective ratio  locked per run
  12/12/1              4.10 (max 6)     1.0 -> 1.5       2.05
   6/12/7              0.65 (max 2)     2.0 -> 2.2       1.15
   4/12/9              0.15 (max 1)     3.0 -> 3.1       0.70
   3/15/7              0.00             5.0              0.55
   2/16/7              0.00             8.0              0.45
   1/11/13             0.00             11.0             0.20

The ratio-1 cell has about half its sources null by construction and is
not a measurement of anything. In the 3/6/6 family the expected orphan
count is 3 * (2/3)^6 = 0.26 per run, and 34 of the 130 sink-bar and
chamber-shape seeds at that shape (26%) contain a source with no sink; at
2/11/2 it is 0 of 130. An orphan caps that run's source-vs-sink AUC near 2/3
by itself and pulls the per-run source median down, so this defect also
falls harder on the 3/6/6 shape than on 2/11/2 -- a second way, besides
locking, in which the shape comparison of 2026-09-02 was not between equal
generators. It is a defect of the truth labels, not of the statistic, and
it has been present in every synthetic outflow number reported.

Both are left as they are in the clean-generator re-run so that it changes
one thing (Rule 91). Each needs its own pre-registration if it is to be
fixed.

## 5. What this does and does not say

It does not say the statistic works. It says three of the numbers that
decided the synthetic line -- the closure at 0.53, the reopen at 0.83, and
the ratio sweep's rising marginal curve -- were measured on systems in
which a substantial fraction of the "sources" could not be detected by any
method the paper endorses, and on truth labels that call orphan channels
sources. The chamber result (real data, real actuators, gate-screened) is
untouched by any of this. The manuscript's mechanism for the shape effect
was already falsified by the ratio sweep before this audit and is withdrawn
independently of it.
