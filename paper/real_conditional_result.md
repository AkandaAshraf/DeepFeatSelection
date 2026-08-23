# Result: conditioning does NOT transfer to real data. The marginal statistic
# does.

2026-08-23. Pre-registration: paper/real_conditional_protocol.md, committed at
79f00cc before any statistic was computed on these datasets. 11.2 minutes.

## The declared verdict is PARTIAL. It should be read as a failure for C1.

  PRIMARY  wt_intake_impulse_v1, m=10, 5 runs, 10 source and 55 sensor channels
    A1 marginal     AUC 0.916   source +0.00568  sensor -0.00002  ghost CLEAN
    C1 conditional  AUC 0.751   source +0.00209  sensor +0.00001  ghost DIRTY
    R1 FAILS: 0.751 < 0.916.   permutation p: C1 0.0052, A1 0.0001

  SECOND   wt_walks_v1, m=10, 2 runs, 10 source and 22 sensor channels
    A1 marginal     AUC 0.605   ghost DIRTY   permutation p 0.19
    C1 conditional  AUC 0.632   ghost DIRTY   permutation p 0.13
    R1 passes: 0.632 > 0.605

The rule as written returns PARTIAL, because R1 held on one dataset. That
label flatters the result. The dataset where R1 passed has BOTH statistics
near chance, BOTH ghosts dirty, and NEITHER distinguishable from a random
relabelling. Its pass carries no weight. On the only dataset where the ghost
is clean and the effect is significant, R1 fails decisively.

In substance this is R6: the synthetic confirmation does not transfer.
Conditioning was confirmed this morning on coupled logistic maps at AUC 0.87
and does not survive contact with a physical system.

## Why, and it is the risk that was declared in advance

The synthetic pre-registration named it: "a source recoverable from its own
sinks is redundant too". The redundancy axis meant to test it was confounded
with capacity and the risk went untested. Real data supplied the test.

The impulse dataset has TWO actuators and ELEVEN sensors, all downstream. The
actuators are therefore recoverable from the sensors, so conditioning on every
other channel removes the actuator's unique contribution along with the
redundancy it was meant to strip. The numbers agree: C1's source signal is
+0.00209 against A1's +0.00568, about 2.7 times smaller, and C1's fifth
percentile over sources is NEGATIVE at -0.0035 - some actuators score below
zero once their own sensors are conditioned on.

The synthetic system gave each source two sinks. This one gives each actuator
five or six. Conditioning works where the driver is not reconstructable from
what it drives, and fails where it is. That is an operating envelope, and it
is the opposite of what a scan of a real system usually offers, since real
systems typically have many measured consequences of few controls.

This explanation is consistent with a risk declared before the synthetic run
and with the effect sizes here. It is not itself tested: that would need the
ratio of driven channels to sources varied deliberately, with capacity held
fixed.

## The unexpected part, stated with its caveats first

It was not the hypothesis, A1 is the baseline rather than a candidate, and
A1's permutation test was NOT pre-declared - it is computed here because
reporting C1's and withholding A1's would misinform. With that said:

  the EXISTING marginal outflow statistic reaches AUC 0.916 on a real
  physical system with structural ground truth, with a clean ghost and a
  permutation p of 0.0001

That is the first real-data positive this line has produced. It rests on one
dataset, one experimental protocol, 5 runs, and 10 source channels against 55.
It is a result to replicate, not to announce.

It does not contradict the voided chamber run. That run used m = 1, where the
fitness gate now shows there is no lagged influence to find; b = 2V, half the
capacity the method needs; and a ground truth with two of five actuators
labelled as sensors. Correct all three and the marginal statistic works on the
same apparatus. The fitness gate and the b = 4V rule earned their place here:
qualify the data first, give the method its capacity, and the statistic that
found nothing finds something.

## Protocol errors to disclose

1. This protocol states that hatch is constant in the regime_jumps runs and
   excludes it. WRONG. Its standard deviation over the full 320,000 samples is
   17.6; it reads 0.000 over the first 20,000, which is what I inspected. The
   analysis used the real data and scored five sources, so no number is
   affected, but the description was wrong. This is the second time today a
   claim about this data came from a truncated read.

2. R3, the ghost-cleanliness check, FAILED on both datasets - for C1 on the
   primary and for both statistics on the second. This protocol declared R3 as
   a prediction but did not make it DISQUALIFYING, while the error-metric
   protocol earlier the same day did exactly that. Had it been disqualifying,
   the second dataset would have been excluded outright and the primary would
   have reported A1 only. That gap is mine and it is the reason the verdict
   printed PARTIAL rather than something sharper.

## Status of the line

Conditional outflow: CONFIRMED on synthetic data, DOES NOT TRANSFER to a real
system, for a reason that was written down before either run. It is not
discarded - it has an envelope, sparse driven-to-source ratios - but it is not
the general fix it appeared to be this morning.

Marginal outflow: one clean, significant real-data result, on a qualified
dataset at correct capacity. It needs replication on a second apparatus before
it is worth anything, and the fitness gate is the instrument for finding one.

## Rules

93. A conditional statistic measures unique contribution, so it fails exactly
    where its target is redundantly represented. Before choosing between a
    marginal and a conditional form, look at how many driven channels each
    source has: real systems usually have many measured consequences of few
    controls, which is the regime where conditioning loses.

94. Rule 81 extended: describe a dataset from all of its ROWS as well as all
    of its files. Twice today a claim about the data came from a truncated
    read, and both times it reached a committed protocol.

95. Declare validity conditions as DISQUALIFYING, not as predictions. A ghost
    check that fails but only counts as a missed prediction lets an
    uninterpretable dataset into the verdict.
