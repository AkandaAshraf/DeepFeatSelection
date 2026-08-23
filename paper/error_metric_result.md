# Result: VOID on a bar I imported wrongly, over a table worth re-running

2026-08-23. Pre-registration: paper/error_metric_protocol.md, committed at
b5dfa70 before the experiment was written or run. 15 cells, 8 metrics, one
autoencoder per cell.

## Verdict: VOID by the declared rule

M1 required that the current metric reproduce what is already known: source-
versus-sink AUC high at coupling 0.50 and degrading at 0.70. Coded as
AUC >= 0.80 at 0.50 AND degradation at 0.70.

  A1_r2   0.767 at c=0.30   0.704 at c=0.50   0.542 at c=0.70

The degradation reproduced exactly. The 0.80 bar did not, so the experiment
is void and the script stopped at M1 without evaluating M2, M3 or M5.

## The bar was imported from a different statistic - my error

0.80 came from the outflow-bar run, where source outflow exceeded sink
outflow in 27 of 30 RUNS. That is a run-level comparison of two medians. AUC
here is a CHANNEL-level ranking, 15 sources against 30 sinks per coupling.
The two are not the same quantity and one does not set a bar for the other:
at coupling 0.50 the medians separate cleanly, +0.0126 against +0.0028, while
individual channels overlap enough to give 0.704.

This is the third time in one day that a bar or control was set by importing
a number from a different quantity - O4's control inherited from the bar it
was replacing, S1's discovery grid chosen without checking it contained the
failure, and now M1. That pattern is the finding of the day and is recorded
as Rule 90.

The void stands as declared. Nothing below is fitted, selected or claimed.

## The table, reported as description

SOURCE vs SINK AUC

  metric              c=0.30   c=0.50   c=0.70
  A1_r2  (current)     0.767    0.704    0.542
  A2_nmae              0.749    0.676    0.542
  A3_spearman          0.762    0.713    0.547
  A4_gauss_nats        0.762    0.671    0.531
  A5_rff_r2            0.618    0.664    0.553   ghost-excluded
  B1_recon_mse         0.444    0.400    0.367   ghost-excluded
  B2_recon_mae         0.329    0.327    0.253   ghost-excluded
  C1_conditional       0.916    0.929    0.924

Three observations, none of them findings.

FAMILY A IS ONE METRIC IN FOUR COSTUMES. R2, normalised MAE, Spearman and
Gaussian entropy agree to within 0.04 AUC everywhere and degrade together.
Rescoring the same ridge predictions changes nothing. A4 was declared
expected-degenerate with A1 and its largest gap is 0.033, marginally outside
the declared 0.02 - close enough to leave the declaration standing and not
close enough to call it exact.

FAMILY B IS WORSE THAN USELESS, AND THIS ANSWERS ONE OF THE TWO READINGS OF
THE QUESTION. Using the network's own reconstruction error gives AUC of 0.25
to 0.44 - BELOW 0.5, meaning sinks outrank sources - and its ghost is dirty.
The quantity MACE computes and discards deserves to be discarded. A sink is
reconstructed largely from its driver's signal, so masking a sink damages the
reconstruction of everything correlated with it, and the measure reads
correlation rather than direction.

C1 DOES WHAT PROPOSITION 1 SAYS IT SHOULD. Medians by role:

           c=0.30     c=0.50     c=0.70
  source  +0.00053   +0.00212   +0.00290     rising with coupling
  sink    -0.00006   -0.00008   -0.00007     pinned at the null
  isolated -0.00006  -0.00008   -0.00009
  ghost   -0.00009   -0.00011   -0.00015

against the current metric, where the confound is plainly visible:

  source  +0.00057   +0.01256   +0.01140
  sink    -0.00048   +0.00283   +0.00740     the proxy, growing

Conditioning on the other channels puts sinks exactly where isolated channels
and the ghost are, at every coupling, while the source signal RISES with
coupling instead of being overtaken. That is the sink-proxy confound removed
rather than thresholded, and it is what Proposition 1 predicts for a quantity
that measures unique rather than marginal contribution.

The cost is magnitude: the source signal is about four times smaller than
under R2, 0.0029 against 0.0114 at coupling 0.70, which is expected since
unique information is a subset of marginal information. Against its own null
of -0.00007 it is still roughly 40x.

The risk named in the pre-registration - that a source recoverable from its
own sinks would be redundant too and collapse with them - did not appear in
this system. Whether it appears in systems with denser redundancy is untested
and is exactly what a proper test must include.

## What this is not

A void experiment. C1 was one of eight metrics on one system family at three
couplings with five seeds, and its bar was never declared because the script
stopped at M1. Reporting it as a result would be selecting the winner from a
table after the fact, which is the move this project refuses and the reason
the ghost, the isolated channels and every other metric are printed beside it
above.

## What a proper test needs, specified before it runs

  - C1 as the PRIMARY hypothesis with its own declared prediction, not one
    column of eight.
  - FRESH SEEDS. Seeds 0-4 are spent; the confirmation must not reuse them.
  - A reproduction check stated as the QUALITATIVE claim that is actually
    established - that R2's source-sink separation degrades as coupling rises
    - with no imported numeric bar.
  - The decisive comparison expressed as C1's AUC against A1's AUC on the
    same cells, since that difference is internally calibrated and needs no
    external constant.
  - A redundancy axis, to test the declared risk that C1 kills sources which
    are recoverable from their sinks.

## Rule

90. Never import a numeric bar from one statistic to another. A run-level
    comparison of medians and a channel-level ranking are different
    quantities, and a threshold that means something for one means nothing
    for the other. Three of today's experiments voided or misfired on this
    single mistake.
