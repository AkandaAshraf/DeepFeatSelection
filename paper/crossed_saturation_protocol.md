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
