# Pre-registration: the outflow closure re-tested on a generator with no phase-locked sources

Declared 2026-09-03, before any run on the seeds named below. Written after,
and because of, the post-hoc generator audit of the same date
(`scripts/generator_audit.py`, `paper/generator_audit.md`).

## What the audit found, stated so the bet below is legible

Every outflow experiment since 2026-08-22 has used `coupled()` in
`scripts/source_outflow_coupling.py`, which draws each source's and each
isolated channel's logistic parameter from r ~ U(3.7, 3.9). That range is
not uniformly chaotic. Three windows -- [3.7016, 3.7028], [3.7382, 3.7448]
and [3.8284, 3.8569] -- give a periodic or band-periodic orbit (Lyapunov
exponent <= 0.05, or max |autocorrelation| over lags 1-12 >= 0.90). Together
they cover 19.2% of the draw range, so a run with three sources has a locked
source with probability 0.47, and one with two sources 0.35.

A locked source is undetectable by construction, by the paper's own fitness
gate: a sink driven by a periodic parent predicts itself from its own lags
(own-lag R2 0.998), and the parent's lags add +0.0005 against the gate's
+0.0136. Every "dead" run in the sink-bar and reopen experiments (source
median outflow below 0.002; 8/60 and 4/60) has a locked source. Restricting
the recorded runs to those with no locked source, the sink-bar's 16/30 = 0.53
becomes 12/17 = 0.71 at the same bar and 17/17 = 1.00 at a bar calibrated on
clean sinks; the reopen's 25/30 = 0.83 becomes 19/19 = 1.00 either way.

Those restricted numbers are post-hoc, on small clean subsets, with a bar
recalibrated after the fact. They motivate a prediction; they do not test
one. This does.

## What changes, and only this

  GENERATOR    `coupled_clean()` in `scripts/clean_generator.py`. Identical to
               `coupled()` except that every draw from U(3.7, 3.9) -- source
               and isolated channels alike -- is rejection-sampled against
               `generator_audit.is_locked_r`. Sink parameters (U(3.5, 3.7)),
               the random parent assignment, the coupling term, the clip to
               [0, 1] and the 0.01 observation noise are unchanged. The
               orphan-source and clipping defects the audit also noted are
               deliberately NOT fixed here: one change per experiment
               (Rule 91).

Everything else is the sink-bar / reopen protocol verbatim:

  BAR          q95 of the sink outflow distribution on the calibration
               block, alpha = 0.05, calibrated separately per shape.
  CAPACITY     b = 4V, coupling 0.50, n = 4000, 25 epochs, 2 models
               averaged (`G.analyse`), per-run median over sources and over
               sinks, exactly `sink_bar.one`.
  SHAPES       3/6/6 (the closed family, V = 15 + ghost) and 2/11/2 (the
               reopen shape, V = 15 + ghost). Same width, same b = 64.
  SEEDS        3/6/6:  calibration 1100-1129, test 1200-1229
               2/11/2: calibration 1300-1329, test 1400-1429
               None used by any prior experiment. Different blocks per
               shape, as in the incumbent runs.
  RECORDED     per run: source/sink/isolated/ghost median outflow, per-run
               source-vs-sink AUC; per channel: role, r, parent, number of
               sinks assigned, outflow, max |autocorrelation| of the
               observed series. The last is a check that the rejection
               worked, not a result.
  SIZE         30 + 30 per shape, as in the incumbent protocols. At a true
               sensitivity of 0.85, 30 test runs fall below 0.80 about one
               time in five; at 0.95, about one in fifty. Wilson intervals
               are reported with every rate (Rule 110).

## Predictions, fixed now

  Q1  DECISIVE. Sensitivity at 3/6/6 on the clean generator is at or above
      0.80 -- the bar the same shape failed on the dirty generator (0.53,
      corrected to 0.5-0.7). If it holds, the closure of 2026-09-02 was an
      artefact of the generator, not a property of the statistic.
  Q2  Sensitivity at 2/11/2 on the clean generator is at or above 0.80. The
      reopen already passed on the dirty generator (0.83); a clean generator
      should not make it worse.
  Q3  The shape effect persists on the clean generator in the threshold-free
      measures: median source outflow and per-run AUC are higher at 2/11/2
      than at 3/6/6. NO PREDICTION on whether thresholded sensitivity still
      separates the shapes -- both may sit at the ceiling.
  Q4  DISQUALIFYING. More than 3 of 60 runs at either shape are dead
      (source median outflow below 0.002). The audit attributes every dead
      run to a locked source; if dead runs persist without one, that
      attribution is wrong and no verdict on Q1 is drawn from this run.
  Q5  The sensitivity gap between the shapes shrinks from the dirty
      generator's 0.30 (0.53 vs 0.83), because locking hurt the shape with
      more sources harder (0.47 vs 0.35 chance of a locked source per run).
  Q6  VOID CONDITION. The ghost clears the calibrated bar in more than 5% of
      runs at either shape. That would mean the pipeline leaks and nothing
      here is interpretable.
  Q7  NO PREDICTION on the bar values. The audit's clean-subset bars
      (+0.0056 at 3/6/6, +0.0159 at 2/11/2) are reported beside them.

## The rule, fixed now -- one branch per outcome

  Q6 fails or Q4 fires ........ NO VERDICT. Reported as such; the locking
                                 explanation is incomplete and the next step
                                 is diagnosis, not another bar.
  Q1 holds, Q2 holds .......... CLOSURE WAS AN ARTEFACT. Section 12's
                                 closure is withdrawn and replaced by: the
                                 statistic detects sources at both shapes on
                                 a generator with chaotic sources; the earlier
                                 closure measured the generator, not the
                                 statistic. The shape result of 2026-09-02 is
                                 re-described as obtained on the defective
                                 generator, with Q3's clean-generator reading
                                 beside it.
  Q1 fails, Q2 holds .......... CLOSURE STANDS WITH A SHAPE CONDITION.
                                 Locking contributed to the 3/6/6 failure but
                                 did not cause it; the reopen result is
                                 confirmed on a clean generator; the shape
                                 effect is real.
  Q1 holds, Q2 fails .......... ANOMALOUS. The audit's clean-subset numbers
                                 (19/19 at 2/11/2) make this outcome nearly
                                 impossible; if it happens, the clean subsets
                                 were not representative and both shape
                                 results are re-examined before any closure
                                 verdict is stated.
  Q1 fails, Q2 fails .......... CLOSED ON THE CLEAN GENERATOR. The locking
                                 explanation is wrong; the closure stands and
                                 the reopen's 0.83 is downgraded to seed-block
                                 fortune.

Q3 and Q5 are reported under every branch and do not change the verdict.

## Void conditions

Void if any parameter above differs from the sink-bar protocol other than
the generator's rejection step; if seeds 1100-1429 have been used before; if
the bar is calibrated on anything but the same-shape calibration block; if
Q6 fails; or if the outcome is reported without the dirty-generator figures
(0.53 / 0.667 and 0.83) beside it.

## Result (2026-09-03, seeds 1100-1429, 120 runs, 20.5 min, `ExpOutput/clean_generator/`)

  shape    bar        sensitivity              AUC    source med  sink med  dead
  3/6/6    +0.00876   26/30 = 0.87 [0.70,0.95] 0.889  +0.01205    +0.00239  2/60
  2/11/2   +0.01360   30/30 = 1.00 [0.89,1.00] 0.977  +0.02546    +0.00761  0/60

  dirty generator, same shapes and protocol: 0.53 (read 0.5-0.7) and 0.83.

  Q1  DECISIVE. 3/6/6 sensitivity 0.87 >= 0.80. HOLDS.
  Q2  2/11/2 sensitivity 1.00 >= 0.80. HOLDS.
  Q3  Shape effect, threshold-free: source median +0.01205 -> +0.02546,
      AUC 0.889 -> 0.977. PERSISTS.
  Q4  Dead runs 2/60 and 0/60, at most 3 allowed. Does not fire.
  Q5  Sensitivity gap +0.13 against the dirty generator's +0.30. SHRINKS.
  Q6  The ghost clears the bar in 0.00 of runs at both shapes. Clean.
  Q7  Bars came out at +0.00876 and +0.01360 against the audit's clean-subset
      +0.00557 and +0.01585, so those post-hoc bars were the right order of
      magnitude and no better than that.

  Rejection check: the largest max |autocorrelation| over any source channel
  in 120 runs is 0.856, below the 0.90 lock criterion. No locked source
  survived rejection.

VERDICT: **CLOSURE WAS AN ARTEFACT**, by the rule fixed above. The synthetic
source-detection line does not close. It was closed on 2026-09-02 at
sensitivity 0.53 by a generator that drew about one source in five from a
phase-locked window of the logistic family; on the same protocol, the same
bar construction, the same capacity and the same shape, with only those
windows removed from the draw, the same statistic reaches 0.87. The earlier
number measured the generator.

### The two dead runs, and what they say

Q4 allowed up to three dead runs per shape as a check on the locking
explanation. Two occurred, both at 3/6/6, and neither has a locked source
(seeds 1205 and 1224, maximum source |ac| 0.75). Both have an **orphan**
source -- a source the random parent draw assigned no sink:

  seed 1205  sources drive 2, 4 and 0 sinks; outflows -0.0005, +0.0496, -0.0009
  seed 1224  sources drive 1, 5 and 0 sinks; outflows -0.0008, +0.0416, -0.0005

With three sources the run statistic is the median of three numbers, so one
orphan and one weakly-connected source put the median at zero however well
the statistic works on the source that actually drives the system. The dead
runs on the clean generator are a defect of the truth labels and of the
run-level median, not of the statistic and not of locking.

### Post-hoc: the orphan defect accounts for the rest

Everything in this subsection was computed after seeing the result and is
reported as a diagnosis, not as a test. Restricting both the calibration and
the test block to runs in which every source drives at least one sink:

  shape    orphan-free bar   sensitivity              AUC     source med
  3/6/6    +0.00728          20/20 = 1.00 [0.84,1.00] 1.000   +0.01227
  2/11/2   +0.01360          29/29 = 1.00 [0.88,1.00] 1.000   +0.02558

Orphans occur in 22 of 60 runs at 3/6/6 and 1 of 60 at 2/11/2, because
`n_sink = 6` random draws over 3 sources leave one empty far more often than
11 draws over 2. At 3/6/6 the per-run AUC median is 0.944 on orphan-free
runs against 0.556 on runs with an orphan, and orphan sources have median
outflow -0.00077 against +0.01448 for connected ones -- correctly zero for a
channel that drives nothing, and wrongly counted as a miss.

So on a generator with chaotic sources and no orphan labels, **both shapes
detect sources in every test run**. What remains of the shape effect is a
difference in the size of the signal, not in whether it is found: median
source outflow is still twice as large at 2/11/2 (+0.0256) as at 3/6/6
(+0.0123).

### Post-hoc: what the size difference tracks

Pooling every non-orphan source in all 120 runs and grouping by the number
of sinks that source actually drives:

  sinks driven   1       2       3       4       5       6       7       8
  n              39      70      43      34      31      26      15      13
  outflow med    0.0081  0.0135  0.0189  0.0227  0.0248  0.0272  0.0319  0.0318

Monotone across the whole range (0.0341 at 9 sinks, n = 5; 0.0411 at 11,
n = 1). This is the *marginal* half of the mechanism withdrawn on the
evidence of the ratio sweep -- that outflow grows with how much of the
system depends on a channel -- and at the level of the individual source it
is clean and monotone. It is not a re-instatement of the withdrawn claim:
what the ratio sweep refuted was the *conditional* half, that the two
variants are pushed in opposite directions, and that refutation stands
because the conditional variant improved with the ratio as well. Nor is this
a pre-registered result. It is a post-hoc regularity on a defective-label
generator, and it is the obvious thing to pre-register next.

### What this changes and what it does not

CHANGES. The closure of 2026-09-02 is withdrawn. The manuscript's Section 12
must state that the synthetic basis for the source-detection complement
holds at both tested shapes once the generator is fixed, and must disclose
that every synthetic outflow number reported before 2026-09-03 -- the
capacity, coupling and maturity gates, the outflow bar, the closure, the
crossed-saturation run, the chamber-shape comparison, the reopen and the
ratio sweep -- was measured with phase-locked sources present.

DOES NOT CHANGE. The chamber result (AUC 0.916, real actuators, gate-
screened) never depended on this generator. The ratio sweep's refutation of
the "opposite directions" mechanism does not depend on it either: the
conditional variant led at every ratio, and locking and orphans, which hurt
both variants on the same runs, cannot produce that ordering. The reopen
result (0.83 dirty, 1.00 clean here) is strengthened, not overturned.

NOT ESTABLISHED. Two shapes, one coupling, one generating family, 30 test
runs each. The clipping defect (83% of sinks at the [0,1] boundary at
coupling 0.50) was deliberately left in place and is untested. Whether the
statistic separates sources from sinks at other couplings on a clean
generator has not been re-measured, and the earlier coupling sweep was run
on the defective one.
