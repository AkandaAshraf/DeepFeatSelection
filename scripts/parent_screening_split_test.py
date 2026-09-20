"""Enumerated raw-support disjointness for parent_screening's actual row
construction, not a formula re-derived from reading the code.

REWRITTEN after review: the first version's raw_support() was itself a
formula, the exact risk Rule 132 warns about (a plausible-looking formula
can be wrong even when derived carefully). This version feeds INDEX-VALUED
data (x[t] = t) through the PRODUCTION own_lag_window function directly,
so the returned own-lag and target VALUES literally name the raw indices
touched -- there is no separate formula to get wrong. The internal
ridge_r2_val seam (found unembargoed by review, now fixed) is checked the
same way, using its actual cut arithmetic, not a re-derived one.

A known-bad embargo (0, adjacent seam) is included as an ORACLE: if this
test cannot detect that overlap, the test itself is broken, not lenient.
The oracle failed once already while writing this file (a pre-existing gap
in the first harness masked the overlap it was meant to detect) -- fixed
before trusting its verdict on anything else.

    python scripts/parent_screening_split_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import parent_screening as PS  # noqa: E402


def touched_indices(n: int, max_delay: int = PS.MAX_DELAY):
    """Runs own_lag_window on x[t]=t (E channels irrelevant, uses column
    0). Returns (own_support[i], target_value[i]) read directly from the
    function's own output values -- the actual raw indices it touched."""
    x = np.arange(n, dtype=float).reshape(-1, 1)
    own, target, t_index = PS.own_lag_window(x, 0, max_delay)
    row_support = [set(own[i].tolist()) | {float(target[i])}
                  for i in range(len(target))]
    return row_support


def support_union(row_support, idx):
    out: set[float] = set()
    for i in idx:
        out |= row_support[i]
    return out


def main() -> int:
    ok = True
    n_obs = 4000
    row_support = touched_indices(n_obs)

    print("ORACLE: an ADJACENT seam with embargo=0 must show an overlap, "
          "or this test cannot detect anything")
    a = 100
    tr0, te0 = list(range(0, a)), list(range(a, 200))
    oracle_overlap = bool(support_union(row_support, tr0)
                          & support_union(row_support, te0))
    print(f"   overlap at embargo=0, adjacent seam: {oracle_overlap}   "
          f"-> {'PASS (oracle detects it)' if oracle_overlap else 'FAIL -- test is broken'}")
    ok &= oracle_overlap

    print("\n[outer seams] splits_for's OWN returned tr/va/te, checked via "
          "own_lag_window's actual output values, not a re-derived formula")
    tr, va, te, m, embargo = PS.splits_for(n_obs)
    row_support_m = touched_indices(n_obs)  # same n_obs -> same m, reuse
    s_tr = support_union(row_support_m, tr.tolist())
    s_va = support_union(row_support_m, va.tolist())
    s_te = support_union(row_support_m, te.tolist())
    tv, vt_, tt = bool(s_tr & s_va), bool(s_va & s_te), bool(s_tr & s_te)
    print(f"   train/val overlap: {tv}   val/test overlap: {vt_}   "
          f"train/test overlap: {tt}")
    outer_ok = not (tv or vt_ or tt)
    print(f"   declared embargo: {embargo}   -> {'PASS' if outer_ok else 'FAIL'}")
    ok &= outer_ok

    print("\n[internal ridge seam] calls PS.internal_val_split directly -- "
          "the exact function ridge_r2_val uses, not a re-derived copy "
          "(review caught the first version hand-duplicating this "
          "arithmetic despite a comment claiming otherwise)")
    n_tr = len(tr)
    itr_pos, iva_pos = PS.internal_val_split(n_tr)   # THE production function
    inner_tr_positions = tr[itr_pos]
    inner_va_positions = tr[iva_pos]
    s_itr = support_union(row_support_m, inner_tr_positions.tolist())
    s_iva = support_union(row_support_m, inner_va_positions.tolist())
    inner_overlap = bool(s_itr & s_iva)
    print(f"   internal train/val overlap: {inner_overlap}   "
          f"-> {'PASS' if not inner_overlap else 'FAIL'}")
    ok &= not inner_overlap

    print("\n[internal seam ORACLE] proves the check above has teeth: an "
          "unembargoed version of the SAME split must show an overlap")
    cut = max(int(n_tr * (1 - PS.VAL_INTERNAL_FRAC)), 1)
    bad_tr_positions = tr[np.arange(0, cut)]      # embargo=0, the bug as filed
    bad_va_positions = tr[np.arange(cut, n_tr)]
    s_bad_tr = support_union(row_support_m, bad_tr_positions.tolist())
    s_bad_va = support_union(row_support_m, bad_va_positions.tolist())
    bad_overlap = bool(s_bad_tr & s_bad_va)
    print(f"   overlap with embargo=0 (the original bug): {bad_overlap}   "
          f"-> {'PASS (oracle detects it)' if bad_overlap else 'FAIL -- test is broken'}")
    ok &= bad_overlap

    print("\n[reject-not-fallback] internal_val_split must RAISE on inputs "
          "too small to embargo, not silently reuse rows across the seam")
    try:
        PS.internal_val_split(2 * PS.E)  # deliberately too small
        raised = False
    except PS.TooSmallForEmbargo:
        raised = True
    print(f"   raised TooSmallForEmbargo on a tiny input: {raised}   "
          f"-> {'PASS' if raised else 'FAIL'}")
    ok &= raised

    print(f"\n{'ALL CHECKS PASS' if ok else 'AT LEAST ONE CHECK FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
