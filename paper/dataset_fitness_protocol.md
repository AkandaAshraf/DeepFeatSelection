# Pre-registration: is a dataset fit to test a lagged-influence statistic?

Declared 2026-08-23, before the quantity below was computed on any data.

## Why this exists

The chamber source-detection run was voided because its escape hatch - a
self-R2 threshold of 0.95 declared in advance - fired on the synthetic system
at coupling 0.50, where outflow demonstrably works. The threshold was
declared but never calibrated, so invoking it was a claim rather than a
check (Rule 79).

This document is that rule applied to the thing the chamber run should have
tested first: whether a candidate dataset carries any lagged driver-to-target
information at its own sampling rate. If it does not, no statistic of this
family can succeed on it, and the dataset is disqualified on evidence rather
than on assumption. Rule 78 said to check; this specifies what the check is.

## The quantity

Linear, no autoencoder, no code, no outflow. For each target channel q:

  base     R2[ x_q(t+1) | poly2(own delay embedding of x_q) ]
  with_d   R2[ x_q(t+1) | poly2(own) (+) delay embedding of the DRIVERS ]
  lag_info(q) = with_d - base

reported as the median over targets. Same embedding (E = 3, TAU = 1), same
ridge (ALPHA = 1.0), same chronological 60/20/20 split, same code path as the
deployed gate. This is a property of the DATA. It says nothing about MACE.

GHOST CONTROL: the identical computation with the driver block circularly
shifted by n/3. It must sit at or below 0.002. A ghost above that means the
split or the embedding is leaking and the check itself is void.

## The threshold, calibrated rather than declared

The reference is the coupled logistic system used throughout this line, where
it is already established which couplings the statistic works at:

  coupling  0.05  0.15  0.30  outflow margin below the 0.01 bar - FAILS
  coupling  0.50  0.70              margin 0.0107, 0.0103 - CLEARS

lag_info is computed at all five couplings, three seeds each, giving the
value the method needs. Write L50 for the median at coupling 0.50, the
weakest system on which outflow is known to work, and L30 for coupling 0.30,
the strongest on which it is known to fail.

## The rule, fixed now

  QUALIFIES     chamber lag_info >= L50. The data carries at least as much
                lagged driver information as the weakest system where the
                statistic works. A null there would be about the method, so
                the clean re-run is worth its cost.
  DISQUALIFIED  chamber lag_info <= L30. The data carries no more lagged
                information than systems where the statistic is known to
                fail. No re-run. The chamber is excluded on evidence, and
                this check becomes the pre-flight gate for every future
                candidate dataset.
  AMBIGUOUS     strictly between L30 and L50. Reported as such, and treated
                as DISQUALIFIED for the purpose of spending an hour of
                compute, because a test that starts ambiguous cannot end
                decisive.

## Data

The 16 actuators_random_walk runs only: one experiment family, all three
declared sources vary in every run, no regime boundary inside any run
(Rules 80, 81). The loads_hatch_mix runs are excluded because pot_1 and
pot_2 are constant there, and the regime_jumps runs because a chronological
split crosses a regime boundary. Both exclusions are made now, on grounds
already documented, and not after seeing any result.

Drivers: hatch, pot_1, pot_2. Targets: the 13 sensors. Per run, then the
median across the 16 runs.

## Void conditions

Void if the threshold is moved after the chamber number is seen, if the
ghost control exceeds 0.002, if runs are added or dropped after a result, or
if the outcome is reported for only one of the two systems.

---

## Result (2026-08-23): the chamber is DISQUALIFIED on evidence

### Reference, computed first

  coupling   lag_info    ghost     outflow status (already established)
  0.05       +0.0007    -0.0000    fails
  0.15       +0.0044    -0.0001    fails
  0.30       +0.0114    -0.0001    fails          <- L30, the floor
  0.50       +0.0136    -0.0002    WORKS          <- L50, the pass mark
  0.70       +0.0189    -0.0001    WORKS

lag_info rises monotonically with coupling and the ghost sits at zero
throughout, so the quantity measures what it was built to measure. The two
couplings where outflow clears the 0.01 bar are the two highest values.

### Candidate

  16 actuators_random_walk runs, one family, all three sources varying

  lag_info median  -0.0000   range [-0.0020, +0.0000]
  ghost    median  -0.0000   range [-0.0007, +0.0000]

### Verdict

DISQUALIFIED. The chamber sits at zero - not merely below the L30 floor but
below coupling 0.05, the weakest system tested, by an order of magnitude.
The maximum over all 16 runs is +0.0000. At this sampling rate the actuators'
history adds nothing to any sensor beyond that sensor's own history.

There is no lagged influence in this dataset for any statistic of this family
to find. The chamber was never a test of the method, and it could not have
been. That was true before any autoencoder was trained.

### What this settles, and what it does not

It settles the chamber: no re-run at b = 4V, no re-registration, no further
compute. The exclusion now rests on a measured absence rather than on the
uncalibrated self-R2 threshold that voided the earlier attempt.

It also vindicates the earlier conclusion while confirming the audit's charge
against its reasoning. "The test did not happen" was right. The self-R2
diagnostic was not what showed it and could not have shown it, since that
same diagnostic fires at coupling 0.50 where the method works. Right answer,
wrong instrument, now with the right one.

LIMIT, stated rather than left implicit: lag_info is linear in a poly2
expansion of the delay embedding. A purely nonlinear lagged dependence
invisible to that expansion would not be detected. The check is not a proof
of absence. What licenses its use here is that the same expansion registers
+0.0114 to +0.0189 on the systems where the method works, so it is sensitive
at exactly the scale that matters, and the chamber returns zero against it.

### Standing use

This becomes the pre-flight gate for every future candidate dataset in this
line. A dataset is not selected until it has been run and has cleared L50,
and the number is reported whatever it says.

---

## Addendum (2026-08-23): decimation, and a corrected ground truth

Declared before any lag_info was computed on the new candidate datasets.

### A ground-truth error in the voided chamber run, disclosed

paper/chamber_source_protocol.md listed load_in and load_out as DRIVEN
sensors. They are settable fan loads, and they VARY in every family of
wt_walks_v1:

  actuators_rw     hatch 9.16  pot_1 28.92  pot_2 66.23  load_in 0.28  load_out 0.18
  loads_hatch_mix  hatch 22.50 pot_1  0.00  pot_2  0.00  load_in 0.35  load_out 0.50
  regime_jumps     hatch  0.00 pot_1 67.52  pot_2 78.47  load_in 0.29  load_out 0.29

So two of the five actuators were on the wrong side of the truth set, and
hatch is additionally constant in the regime_jumps runs - which is why its
self-R2 read 0.000 there. This is the fifth defect in that run and does not
change its VOID status. It is recorded because any new test on this apparatus
must not inherit it.

CORRECTED, and fixed now for every chamber dataset:

  SETTABLE (sources)  hatch, pot_1, pot_2, load_in, load_out
  MEASURED (driven)   current_in, current_out, rpm_in, rpm_out,
                      pressure_upwind, pressure_downwind, pressure_ambient,
                      pressure_intake, mic, signal_1, signal_2
  EXCLUDED            timestamp, config, counter, flag, intervention,
                      osr_*, v_*, res_*

Per dataset, a settable variable counts as a source only if it actually
varies there (Rule 80). Constants are excluded from the truth set, not
scored as zeros.

### Decimation, because sampling rate is a choice and not a fact

The wind tunnel is sampled far faster than its actuators move: actuator
lag-1 autocorrelation is 0.99 in every dataset inspected. That is the
oversampling Rule 78 warns about, and it is not a property of the apparatus
but of the rate at which its log is read. Taking every m-th sample makes the
actuator's motion fast relative to the sampling interval.

Choosing m to maximise a RESULT would be fitting. Choosing m so that the DATA
carries lagged influence at all is dataset qualification, and it is decided
by the gate, before any statistic is run, and reported in full.

  GRID          m in {1, 2, 5, 10, 20, 50}
  MINIMUM n     after decimation a run must retain at least 2,000 samples,
                the validated floor; shorter cells are dropped and reported
                as dropped.
  RULE          a dataset QUALIFIES at the SMALLEST m whose lag_info reaches
                L50 = +0.0136. If no m reaches it, the dataset is
                DISQUALIFIED.
  REPORTING     every candidate at every m, whatever the numbers say, with
                the ghost control beside each.

### Candidates, fixed now

  wt_walks_v1            already disqualified at m = 1; re-screened across the
                         grid with the corrected truth set
  wt_changepoints_v1     abrupt actuator steps
  wt_bernoulli_v1        randomised fan loads
  wt_intake_impulse_v1   impulse responses

Void if the grid or the candidate list changes after any number is seen, if a
dataset is selected at other than the smallest qualifying m, or if any
candidate screened is left unreported.

---

## Screen result (2026-08-23): two of four chamber datasets qualify

Reference recomputed with the identical code path: L30 = +0.0114,
L50 = +0.0136. Unchanged, as required.

  wt_walks_v1            m=1  +0.0002 | m=2 +0.0007 | m=5 +0.0018
                         m=10 +0.0158 QUALIFIES | m=20 +0.0162 | m=50 +0.0405
                         at m>=10 only 2 runs retain 2,000 samples: the two
                         320k-sample regime_jumps runs. 4 live sources there
                         (hatch is constant in them).
  wt_changepoints_v1     m=1 +0.0000, and no run retains 2,000 samples at any
                         higher m. DISQUALIFIED.
  wt_bernoulli_v1        +0.0001, +0.0000, +0.0000, +0.0001, +0.0004, +0.0015
                         across the grid. DISQUALIFIED. Despite the name, the
                         randomised loads are held for many samples: their
                         lag-1 autocorrelation is 0.99.
  wt_intake_impulse_v1   m=1 +0.0008 | m=2 +0.0015 | m=5 +0.0083
                         m=10 +0.0224 QUALIFIES | m=20 +0.0254 | m=50 +0.0000
                         5 runs of 250,000 samples, 25,000 after decimation.
                         2 live sources: hatch and load_in.

Ghost at or below +0.0011 in every cell of every dataset.

The earlier verdict on wt_walks_v1 stands as given: at the sampling rate its
logs are distributed in, it carries no lagged influence. Decimation is not a
rescue of that result but a different question - whether the APPARATUS can
carry lagged influence at some rate - and the answer for two of these
datasets is yes.

The m=50 collapse for the impulse dataset, from +0.0254 to +0.0000, is
consistent with the transient being aliased away entirely at that spacing.
No claim is made about it.

### A selection rule that was NOT declared

The script broke the tie between the two qualifying datasets by taking the
larger lag_info. The protocol declared how a dataset qualifies but not how to
choose among several that do, so that tie-break is an undeclared rule. It is
not used. BOTH qualifying datasets go forward, and both are reported.
