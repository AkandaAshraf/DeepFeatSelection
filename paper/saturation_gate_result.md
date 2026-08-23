# Result: the saturation gate is VOID, and the finding it was built on is
# narrower than the record says

2026-08-23. Pre-registration: paper/saturation_gate_protocol.md, committed at
49078fa before the experiment was written or run. 57 cells, 10.5 minutes.

## Verdict: VOID by the declared rule

S1 was the reproduction check: source false positives must rise as saturation
falls, and "if this does not reproduce, the experiment is void and nothing
after it means anything". On the discovery grid it did not reproduce.

  DISCOVERY  V = 15, coupling 0.35, 27 cells

  obs noise   0.00  0.01  0.02  0.03  0.05  0.07  0.10  0.20  0.40
  self-R2    0.981 0.972 0.955 0.935 0.868 0.785 0.651 0.317 0.060
  source FP   0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  recall      1.00  1.00  1.00  1.00  0.92  0.67  0.33  0.25  0.25
  G3          pass  pass  pass  pass  pass  pass  pass  pass  pass

Source false positives are ZERO at every level, down to a self-baseline of
0.06. The script stopped at S1 as instructed. No s* was fitted. No held-out
number was used to fit anything.

What happens instead, without duplicates, is that RECALL collapses - 1.00 to
0.25 - while precision on sources holds. The method loses the ability to see
drivenness before it starts inventing it.

## Why it did not reproduce: the original cell was never run

Rules 71-72 read "source blindness is not a property of the method, it is a
property of the saturated regime". Checking the source data,
ExpOutput/duplicate_channel/duplicate_channel.csv:

  source flag rate, by observation noise and k (k = number of duplicates)

  obs\k      0      1      2      4
  0.0      0.00   0.00   0.00   0.00
  0.1        -      -    0.20     -
  0.3        -      -    0.23     -
  0.6        -      -    0.03     -

The observation-noise extension was run ONLY at k = 2. Every cell that
produced the finding contained duplicated channels. The crossed cell - low
saturation, NO duplicates - does not exist in that dataset. The 27 discovery
cells here are that missing cell, and they show 0.000 throughout.

So the claim was generalised past its evidence: what was measured is that
sources are flagged at low saturation WHEN DUPLICATES ARE PRESENT. Whether
low saturation alone suffices was never tested until now, and in this system
it does not.

## The held-out cells, which ran before the void triggered

They were computed before S1 was evaluated, so the numbers exist and
suppressing them would be worse than reporting them. They are reported as
description. Nothing is fitted to them.

  V = 30, coupling 0.20        noise  0.00  0.02  0.05  0.10  0.30
                             self-R2 0.994 0.969 0.865 0.623 0.120
                           source FP  0.00  0.00  0.20  0.40  0.40

  V = 30, coupling 0.50        noise  0.00  0.02  0.05  0.10  0.30
                             self-R2 0.984 0.965 0.896 0.756 0.325
                           source FP  0.00  0.00  0.20  0.80  0.40

At V = 30 the failure DOES appear, without any duplicates, reaching 0.80.

## What the two grids say together

Saturation alone does not determine the risk. At matched self-baseline:

  self-R2 ~ 0.87    V = 15  source FP 0.00     V = 30  source FP 0.20
  self-R2 ~ 0.65    V = 15  source FP 0.00     V = 30  source FP 0.40

The same self-R2 carries different risk at different panel width. A
ONE-DIMENSIONAL per-channel saturation bar therefore cannot exist in the form
proposed: no threshold on self-R2 can separate a safe cell from an unsafe one
when both sit at 0.87. That is visible by reading the table and required no
fitting, which is why it survives the void.

Rule 84's first item gets a partial answer, and it is not the one expected.
The licensing premise cannot be made into a single number. It is at least two
dimensional.

## The confound, named rather than resolved

b was held at 32 for both widths, so the code-to-system ratio was NOT matched:
b/V is 2.1 at V = 15 and 1.07 at V = 30. The bottleneck study established that
detection is capacity-limited and that b of order 2V is needed, so the V = 30
grid was run under-capacity by design inheritance rather than by choice - its
recall is 0.48-0.52 even at zero noise, against 1.00 at V = 15.

Width and capacity are therefore not separated here. The failure may be a
capacity effect appearing under noise, a width effect, or an interaction. This
experiment cannot say which, and no further reading of these cells will make
it say.

## What is needed, specified before it is run

A properly crossed design, which nothing in this project has yet done:

  saturation (observation noise)  x  b/V held constant at 2  x  k in {0, 2}

That separates capacity from width, and duplication from saturation. Until it
exists, the honest statement of the hazard is: sources can be flagged at
reduced saturation, this has been observed with duplicates present at V = 15
and without duplicates at V = 30 under-capacity, and the conditions under
which it happens are not established.

## Rules

85. Show that the discovery system exhibits the failure before fitting a gate
    against it. The gate here was designed against a phenomenon that the
    discovery grid did not contain, which the pre-registered reproduction
    check caught and nothing else would have.

86. Rules 71 and 72 are NARROWED. The observation-noise cells that produced
    them were run only at k = 2; the no-duplicate cell at low saturation was
    never run and, when run, shows source FP 0.000 at V = 15 down to
    self-R2 0.06. The failure is real but its conditions were never
    established.

87. If the same value of a licensing quantity carries different risk under
    different capacity, that quantity cannot be the gate on its own. A
    premise that is one number in the theory may need two in practice, and
    which it is has to be measured.
