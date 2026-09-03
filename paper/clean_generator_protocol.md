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
