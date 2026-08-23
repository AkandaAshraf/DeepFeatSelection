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
