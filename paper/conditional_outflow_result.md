# Result: CONFIRMED at k = 0. Conditioning separates what scoring cannot.

2026-08-23. Pre-registration: paper/conditional_outflow_protocol.md,
committed at 2e1c431 before the experiment was written or run. 90 cells,
fresh seeds 5-14, 17 minutes.

## The declared outcomes

  SOURCE vs SINK AUC, k = 0      c=0.30   c=0.50   c=0.70
  A1 marginal (current metric)    0.768    0.655    0.517
  C1 conditional                  0.856    0.877    0.872
  margin C1 - A1                 +0.087   +0.222   +0.356

  P1  A1 degrades 0.768 -> 0.517. REPRODUCES, direction only as declared.
  P2  PRIMARY. C1 above A1 at all three couplings, margin grows +0.087 to
      +0.356. PASS.
  P3  C1 places sinks on the isolated reference at every coupling: |diff|
      0.00000, 0.00002, 0.00005 against IQRs of 0.00005, 0.00007, 0.00008.
      PASS.
  P5  Ghost clean on both statistics' own scales. PASS.

VERDICT: CONFIRMED, by the rule fixed in advance.

At coupling 0.70 the current metric is at 0.517 - chance. It cannot tell a
source from its own sink where the driving is strongest. Conditioning holds
at 0.872 there, and the gap widens with coupling precisely because the sink
proxy that defeats the marginal statistic is what conditioning removes.

## Medians by role, k = 0

           c=0.30     c=0.50     c=0.70
  source  +0.00027   +0.00153   +0.00172
  sink    -0.00009   -0.00007   -0.00005
  isolated -0.00008  -0.00009   -0.00010
  ghost   -0.00008   -0.00009   -0.00012

Sinks, isolated channels and the ghost are indistinguishable under C1 at
every coupling. Under A1 the sink proxy grows to +0.00740 and swallows the
source signal.

## Why it works, which is a reason and not a fit

A sink at time t is a function of its driver at t-1 and of itself at t-1.
Both are already available to the conditioning set from the other channels,
so the sink adds nothing unique. A source at time t carries its own value at
t, which the lagged sinks do not yet contain, so it retains unique
information about the next step. C1 is therefore conditional Granger
causality computed in the code space, not a feature-importance score - and
that is a different object from the difference-based measures this project
has repeatedly had to discard.

## Control, NOT pre-declared, reported as a check

Sinks are generated with r in [3.5, 3.7] while sources AND isolated channels
share r in [3.7, 3.9]. If either statistic were tracking the parameter range
rather than the causal role, sources and isolated would score alike. They do
not: isolated channels sit at the null while sources do not.

  SOURCE vs ISOLATED AUC          c=0.30   c=0.50   c=0.70
  A1                               0.896    0.904    0.896
  C1                               0.889    0.913    0.918

Both statistics separate a source from an isolated channel, stably. The
difference is entirely in the source-versus-SINK comparison. Conditioning
fixes what was broken without breaking what already worked.

## Effect sizes, stated plainly

The numbers are small in absolute terms. At coupling 0.70 the source median
is +0.00172 against a null of -0.00010, roughly seventeen times the null. The
source lower quartile is +0.00005 and the sink upper quartile is -0.00002, so
better than three quarters of sources sit above better than three quarters of
sinks, with about thirteen percent of pairs mis-ordered. That is what an AUC
of 0.87 is, and it is not a clean threshold.

## P4, the declared risk: NOT CLEARED, because I confounded the axis

  C1 AUC        c=0.30   c=0.50   c=0.70      A1 for comparison
  k = 0          0.856    0.877    0.872      0.768  0.655  0.517
  k = 1          0.987    0.982    0.981      0.968  0.862  0.827
  k = 2          0.986    0.972    0.972      0.964  0.894  0.862

The risk was that near-duplicate copies would make a source recoverable from
its own sinks and zero it along with them. It did not happen. But adding
copies also grows the panel, and b was held at 64, so b/V moved from 4.00 at
k = 0 to 3.37 at k = 1 to 2.91 at k = 2. The redundancy axis is confounded
with capacity - the fourth confounded axis today, after the saturation
experiment's identical mistake.

So the honest status of P4 is UNTESTED, not cleared. Two things temper that.
The confound moves in the direction that should have made the risk MORE
likely to appear, since the bottleneck study established that lower capacity
costs detection, and the risk still did not appear. And P2, P3 and P5 - the
predictions the verdict rests on - were all declared and evaluated at k = 0,
where no capacity change occurs.

The improvement at k > 0, from 0.87 to 0.98, is UNEXPLAINED and is NOT
claimed as a benefit of redundancy. It is an anomaly in a confounded cell.
The clean version holds b/V fixed while k varies and has not been run.

## Status of the line

The source-detection line reopens, on a measurement rather than a threshold,
with a declared prediction, on fresh seeds, against the statistic it
replaces. That is what four earlier attempts today failed to achieve.

What it is not: this is one synthetic system family, coupled logistic maps,
which is where every previous version of this line also looked healthy. The
chamber test is the standing reminder that real data is where these die. The
next step is a real dataset that clears the dataset-fitness gate at L50, and
the gate exists precisely so that the dataset is qualified before the
statistic is run on it.

## Rules

91. Vary one thing. Adding channels to test redundancy also changed the
    panel size and therefore the capacity ratio; the axis measured two
    things at once. When an axis changes the number of channels, hold b/V
    fixed by construction.

92. When a statistic confuses two roles, examine what it CONDITIONS on
    before what it SCORES. Rescoring the same predictions - squared error,
    absolute error, rank, entropy - cannot separate what the conditioning
    set has already merged. A day was spent on thresholds and error
    functions over a difference that lived in neither.
