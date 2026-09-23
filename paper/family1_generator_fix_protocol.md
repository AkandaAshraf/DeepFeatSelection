# Family 1 generator defect and corrected generator (family 1b)

Registered 2026-09-23, before the corrected generator's code exists and before
any non-engineering seed of it is drawn.

## The defect, measured

The family-1 generator of paper/parent_screening_protocol.md updates a driven
channel as

    x_q(t+1) = r_q x_q(1 - x_q)(1 - c) + c * D_q (1 - x_q),   c = 0.20,

with D_q the mean weighted lagged parent value. The first term is a logistic
map with effective parameter r_q(1 - c) = 2.88 - 3.12, below the onset of chaos
(3.57) and at the period-doubling point. Measured (forecast-horizon check run,
script 455e757, seeds 26001-26004; engineering scans on seeds 9950-9952, excluded from
every result):

- driven channels hold simplex forecast skill 0.98-1.00 from horizon 1 to 10,
  while root channels (no coupling) decay 0.99 -> 0.79 or lower; median
  lag-1 / lag-2 autocorrelation -0.99 / +0.99; 0.6% of variance remains after
  removing the period-2 alternation;
- the same locking occurs at every r range tried (3.6-3.9 to 3.9-4.0) and
  vanishes with coupling set to zero, so the coupling form is the cause;
- at c = 0.20 the median best-lag |correlation| between a target and its TRUE
  parents (0.965) is LOWER than with non-parents (0.987).

Consequence stated now, not later: family 1 driven targets in the parent-
screening study (both the stopped original and the completed laptop pilot)
were period-2 oscillators with almost no parent-attributable variance. The
family-1 halves of those results are recorded as measurements on a defective
generator; they are not evidence about screening chaotic systems. The family-2
halves are not affected by this defect.

## Corrected generator (family 1b), frozen

Diffusive coupled-map form (Kaneko), roots unchanged:

    f_i(x) = r_i x (1 - x)
    root:    x_q(t+1) = f_q(x_q(t))
    driven:  x_q(t+1) = (1 - c) f_q(x_q(t)) + c * mean_j eta_qj f_j(x_j(t+1-d_j))

with c = 0.30, r_i ~ Uniform(3.8, 4.0) rejecting periodic-window values by the
existing _is_locked_r rule, eta ~ Uniform(0.85, 1.15), clip to [0, 1], the same
shared DAG (build_dag), lag convention, burn-in 500, observation noise 0.05 x
train std, and validity checks as family 1. Chosen from the engineering scan:
driven simplex skill 0.95 -> 0.14 over h = 1..6 and parent vs non-parent
|corr| 0.343 vs 0.062 (seeds 9950-9952). Implemented as a NEW function in a
new module; the family-1 code of the closed studies is not modified.

## Validation on fresh seeds, frozen

Seeds 28001-28004 (V=24, n=4000) plus an i.i.d. noise system 28091. Run the
forecast-horizon adequacy check exactly as registered in
paper/forecast_horizon_check_protocol.md (same models M1-M5 and C-LEAK, same
criterion, controls and 80% bar) on these systems. Family 1b is ACCEPTED as a
chaotic benchmark only if, on its driven channels: the simplex reference M1 is
adequate on >= 80%; median lag-2 autocorrelation is below 0.5 in absolute
value; and median parent |corr| exceeds median non-parent |corr| in every
seed. If any fails, family 1b is rejected and no model verdict is drawn on it.

If accepted, the per-model adequacy on family 1b is the answer to the original
question for chaotic dynamics: which of M1-M5 (in particular the masked and
group autoencoders) behave as adequate dynamical models. Family 2's low own-
history predictability (median s(1) ~ 0.10) is recorded as a property of that
generator, not a defect, and is not changed here.

Resources: CPU plus light GPU, `.agent-lock`, expected ~10 minutes, outputs in
ExpOutput/forecast_horizon_check_f1b/.

## Pre-run amendment (2026-09-23, engineering seeds only; no fresh seed drawn)

The acceptance bar "median lag-2 autocorrelation below 0.5 in absolute value"
was set without measuring that statistic's scale on the corrected generator,
the error Rule 133 names. Measured on engineering seeds 9960-9961 (V=24,
n=2000): the corrected generator's driven channels have median lag-2
autocorrelation 0.40-0.66 at every coupling and r range tried, while their
forecast skill decays genuinely (simplex 0.94 -> about 0 by h=10) and parents
separate cleanly from non-parents (0.35 vs 0.07). Linear lag-2 correlation is
therefore a poor proxy for period-2 locking in a nonlinear map. It is REPLACED,
before any fresh seed, by the statistic that actually exposed the defect: the
median share of a driven channel's variance left after removing the period-2
alternation, var(z(t) - z(t-2)) / (2 var z). Measured scale: defective
generator 0.006, corrected generator 0.371 (c=0.3), white noise 1.0, a pure
period-2 cycle 0. Bar: >= 0.20. Lag-2 autocorrelation is still reported.
Nothing else changes: generator parameters (c = 0.30, r ~ U(3.8, 4.0)), seeds,
models, the adequacy criterion and the other two acceptance bars stand as
registered.

## Result (2026-09-23): family 1b REJECTED under the frozen rule

Run a41eb8e, seeds 28001-28004 plus noise 28091, 457 s, no resource breach.
Controls passed (noise fails (a) on 100% for every model; C-LEAK fails (c) on
96/96). Acceptance on driven channels:

  simplex M1 adequate          0.75   (bar 0.80)            FAIL
  period-2 variance left       0.419  (bar 0.20; defective 0.006)   pass
  parent vs non-parent |corr|  0.356-0.377 vs 0.047-0.078 in all 4 seeds   pass

All 24 of M1's failures are criterion (b): simplex skill turns negative after
h ~ 6 (median -0.04 to -0.18) and fluctuates upward there by more than the
0.02 allowance. The rule is applied as frozen: family 1b is REJECTED and no
model verdict is drawn on it. Neither the tolerance nor the acceptance rule is
changed after this result.

Descriptive only, explicitly not a verdict: on these chaotic driven data the
ridge models decay smoothly and are adequate on 94-99% of targets (M2 0.94,
M3 0.99, M4 0.99, M5 0.99); median skill h=1 -> 5 -> 10: M3 0.95 -> 0.24 ->
0.03, M4 0.95 -> 0.24 -> 0.03, M5 0.96 -> 0.26 -> 0.04. Neither autoencoder
exceeds the poly3 readout by more than 0.04 at any horizon. The generator's
physical defect is fixed (period-2 locking gone, parents recoverable); what
failed is a bar on the reference method's noisy negative-skill tail. Any
further use of family 1b needs a new registration.
