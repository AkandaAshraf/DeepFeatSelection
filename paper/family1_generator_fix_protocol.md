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
