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
