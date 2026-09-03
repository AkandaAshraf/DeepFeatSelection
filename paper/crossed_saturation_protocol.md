# Pre-registration: separating width from capacity in the saturation premise

Declared 2026-09-02, before the experiment was written or run.

## The confound this exists to remove

Rule 84's first item, and the specific defect that made the earlier
saturation-gate experiment uninterpretable.

That run (`paper/saturation_gate_protocol.md`) voided at its own reproduction
check: on the V = 15 discovery grid, source false positives were 0.000 at
every observation-noise level down to a self-baseline of 0.06. The held-out
V = 30 grids, computed before the void triggered, DID show the failure,
reaching 0.80. At matched self-baseline the risk differed by width:

    self-R2 ~0.87    V=15: 0.00 source FP     V=30: 0.20
    self-R2 ~0.65    V=15: 0.00               V=30: 0.40

But b was held at 32 for both widths, so b/V was 2.1 at V = 15 and 1.07 at
V = 30. **Width and capacity were varied together.** The result section said
so and declined to resolve it:

> Width and capacity are not separated here. The failure may be a capacity
> effect appearing under noise, a width effect, or an interaction. This
> experiment cannot say which, and no further reading of these cells will
> make it say.

It also specified the fix, which is what this runs:

> saturation (observation noise) x b/V held constant at 2 x k in {0, 2}

## Design, fixed now

Three crossed axes. b is set FROM V so the capacity ratio is constant, which
is the whole point:

  WIDTH        V in {15, 30, 60}, with b = 2V in every cell (30, 60, 120).
               b/V = 2.0 everywhere, so any width effect that survives is
               not a capacity effect.
  SATURATION   observation noise in {0.0, 0.02, 0.05, 0.10, 0.30}, added on
               top of the system's own 0.005, the same ladder the earlier
               run used so the cells are comparable.
  REDUNDANCY   k in {0, 2} duplicated channels, because the original
               observation-noise finding came ONLY from k = 2 cells (Rule
               86) and the no-duplicate case was never crossed with width.

  SEEDS        0, 1, 2. 3 x 5 x 2 x 3 = 90 cells.

System and scan machinery are `boundary_map`'s, unchanged. Measured per
cell: source false-positive rate, per-channel self-R2, recall, precision,
ghost median and max.

## Predictions, fixed now

  X1  REPRODUCTION. At k = 2, source false positives rise as the
      self-baseline falls, in at least one width. If nothing reproduces at
      any width or redundancy, the experiment is void and nothing else in it
      is interpretable.
  X2  DECISIVE. With b/V held at 2, source false positives at matched
      self-R2 are NOT ordered by width. That is the hypothesis that the
      earlier width effect was capacity in disguise. If the ordering
      survives at constant b/V, width is a real second dimension and the
      licensing premise genuinely needs two numbers.
  X3  At k = 0 the failure is absent or much weaker than at k = 2, at every
      width. This is Rule 86 restated as a prediction: the original finding
      came only from duplicated cells.
  X4  NO PREDICTION on whether b = 2V removes the failure altogether. If
      source false positives are 0.000 in all 90 cells, that is a finding
      about the capacity rule rather than about saturation, and is reported
      as such.
  X5  Ghost median stays at or below 0.005 in every cell. A cell whose ghost
      is dirty is excluded from the X2 comparison and reported separately.

## The rule, fixed now

  CAPACITY      X2 holds: at constant b/V the width ordering disappears. The
                earlier width effect was under-capacity, the saturation
                premise is one-dimensional after all, and a per-channel
                self-R2 gate becomes possible again -- which the earlier run
                declared impossible on confounded evidence.
  TWO DIMENSIONS X2 fails: the ordering survives at constant b/V. Width is
                real, the premise needs two numbers, and the earlier
                conclusion stands on unconfounded evidence.
  VOID          X1 does not reproduce anywhere.

## Void conditions

Void if b is not 2V in every cell, if the noise ladder or seeds change after
any result is seen, if k = 0 cells are dropped, or if X2 is judged on cells
whose ghost failed X5.

---

## Result (2026-09-02): INCONCLUSIVE. The two redundancy strata disagree.

90 cells, b = 2V throughout, 0 ghost-dirty (X5 clean everywhere).

  X1 REPRODUCTION: HOLDS. At k = 2 source false positives reach 0.67, and
     rise as the self-baseline falls at every width.

  X3: HOLDS directionally. k = 0 mean rate 0.122, 15 of 36 noisy cells
     non-zero; k = 2 mean rate 0.217, 25 of 36. Duplicates make the failure
     worse, as Rule 86 implies -- but k = 0 is NOT clean, reaching 0.50.

  X4 (no prediction was made): b = 2V does not remove the failure. It does
     substantially restore RECALL: at zero noise, 1.00 / 0.92 / 0.61 for
     V = 15 / 30 / 60, against 0.18 at V = 60 with b = 32. The capacity rule
     works for detection and does nothing for source blindness.

### X2: the script's verdict and the declared criterion disagree

`scripts/crossed_saturation.py` printed **TWO DIMENSIONS**. That verdict
should not be relied on: its rule was "spread of median source_fp across
widths > 0.05", and the medians it compared (0.000, 0.000, 0.200) are
medians over ALL noise levels including the all-zero nz = 0 cells, of a rate
quantised in steps of 1/3, 1/5 and 1/10 because n_src is 3, 5 and 10. That
is a poor summary and it is not what X2 declared.

X2 as written asks whether source false positives **at matched self-R2** are
ordered by width. Computed that way, on the declared k = 2 basis:

    Spearman(source_fp rate, V), k = 2, noise > 0:   +0.002
    mean rate by width:   V=15  0.250   V=30  0.167   V=60  0.233

There is no width ordering at k = 2. At matched noise the largest width is
worst only at nz = 0.02; at nz = 0.05 and 0.10 the SMALLEST width is worst,
and at nz = 0.30 the middle one is. In absolute counts the flagged-source
tally is 1, 1, 2 out of 3, 5, 10 at nz = 0.05 -- similar counts, different
denominators, which is most of what the earlier "width effect" was.

**But the k = 0 stratum says the opposite:**

    Spearman(source_fp rate, V), k = 0, noise > 0:   +0.672
    mean rate by width:   V=15  0.000   V=30  0.117   V=60  0.250

At k = 0, width orders the failure cleanly and monotonically.

### Verdict: INCONCLUSIVE, and the protocol's own choice is why

The declared basis (k = 2) supports CAPACITY: no width ordering survives at
constant b/V. The undeclared-but-run k = 0 stratum supports TWO DIMENSIONS
with a strong monotone ordering. **The two halves of the same experiment
disagree, and the protocol's decision to judge X2 on k = 2 alone was made
before anyone knew that choice would decide the answer.**

Adopting the k = 2 result because it is the declared one, while a
same-experiment stratum with a Spearman of +0.672 says otherwise, would be
letting an arbitrary prior choice settle a question the data does not
settle. The honest verdict is that this experiment does not resolve width
versus capacity, and the reason is now specific: **the answer depends on
redundancy.** With duplicates present the failure saturates at every width;
without them it grows with width.

That interaction was not anticipated by the protocol and is not tested by
it -- k has two levels, which cannot characterise an interaction. Rule 84's
first item therefore stays open, with a sharper statement than before: the
licensing premise depends on at least self-baseline, width and redundancy
jointly, and the earlier one-dimensional gate proposal remains dead.

### Disclosed

The implemented rule did not match the declared criterion, and this was
noticed only after seeing results -- exactly the situation in which
reinterpretation is suspect. Both readings are therefore reported, the
script's printed verdict is left in the log, and the conclusion drawn is the
conservative one (inconclusive) rather than either of the two the strata
would separately support.
