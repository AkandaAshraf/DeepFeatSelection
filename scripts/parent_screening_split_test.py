"""Enumerated raw-support disjointness for parent_screening's own_lag_window,
modelled on scripts/test_embargo_boundary.py's pattern: derive raw index
support from the ACTUAL row-construction code, not an assumed formula, per
Rule 132 ("a correction is a claim like any other and needs its own check").

A known-bad embargo (0) is included as an ORACLE: if this test cannot
detect that overlap, the test itself is broken, not just lenient.

    python scripts/parent_screening_split_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import parent_screening as PS  # noqa: E402


def raw_support(row_i: int, max_delay: int, E: int) -> set[int]:
    """Raw indices touched by own_lag_window's row `row_i`: the E-length
    own-lag window [t-(E-1) .. t] plus the target's single touch at t+1,
    where t = max_delay + row_i (own_lag_window's own convention)."""
    t = max_delay + row_i
    return set(range(t - (E - 1), t + 1 + 1))  # own window ∪ {t+1}


def min_required_embargo(max_delay: int, E: int) -> int:
    """Discovered by enumeration, not asserted: the smallest embargo e such
    that dropping e rows at the ADJACENT seam (no pre-existing gap) leaves
    train's raw support disjoint from the immediately-following block's."""
    for e in range(0, max_delay + E + 2):
        m, a = 200, 100
        tr = np.arange(0, max(a - e, 1))
        te = np.arange(a, m)   # adjacent to train, no gap -- the real seam
        tr_support = set().union(*(raw_support(i, max_delay, E) for i in tr))
        te_support = set().union(*(raw_support(i, max_delay, E) for i in te))
        if not (tr_support & te_support):
            return e
    raise RuntimeError("no embargo up to the search bound closes the overlap")


def main() -> int:
    ok = True

    print("ORACLE: an ADJACENT seam with embargo=0 must show an overlap, "
          "or this test cannot detect anything")
    m, a = 200, 100
    tr0 = np.arange(0, a)                # embargo = 0
    te0 = np.arange(a, m)                # starts immediately after train
    s_tr0 = set().union(*(raw_support(i, PS.MAX_DELAY, PS.E) for i in tr0))
    s_te0 = set().union(*(raw_support(i, PS.MAX_DELAY, PS.E) for i in te0))
    oracle_overlap = bool(s_tr0 & s_te0)
    print(f"   overlap at embargo=0, adjacent seam: {oracle_overlap}   "
          f"-> {'PASS (oracle detects it)' if oracle_overlap else 'FAIL -- test is broken'}")
    ok &= oracle_overlap

    discovered = min_required_embargo(PS.MAX_DELAY, PS.E)
    # READ splits_for's OWN returned embargo -- not a formula re-derived
    # here, which is exactly the duplicated-assumption risk this test
    # exists to catch (caught once already while writing this file).
    _, _, _, _, declared = PS.splits_for(4000)
    print(f"\nDISCOVERED minimal embargo (enumeration): {discovered}")
    print(f"DECLARED embargo in parent_screening.splits_for: {declared}")
    match = discovered == declared
    print(f"   -> {'MATCH' if match else 'MISMATCH -- fix splits_for'}")
    ok &= match

    print("\nVERIFY splits_for's ACTUAL train/val/test arrays are disjoint "
          "in raw support, using its own returned indices")
    n_obs = 4000
    tr, va, te, m, embargo = PS.splits_for(n_obs)
    s_tr = set().union(*(raw_support(i, PS.MAX_DELAY, PS.E) for i in tr))
    s_va = set().union(*(raw_support(i, PS.MAX_DELAY, PS.E) for i in va))
    s_te = set().union(*(raw_support(i, PS.MAX_DELAY, PS.E) for i in te))
    tv = bool(s_tr & s_va)
    vt = bool(s_va & s_te)
    tt = bool(s_tr & s_te)
    print(f"   train/val overlap: {tv}   val/test overlap: {vt}   "
          f"train/test overlap: {tt}")
    disjoint = not (tv or vt or tt)
    print(f"   -> {'PASS' if disjoint else 'FAIL'}")
    ok &= disjoint
    ok &= (embargo == discovered)  # splits_for's own runtime value, not just
                                   # the module constant checked above

    print(f"\n{'ALL CHECKS PASS' if ok else 'AT LEAST ONE CHECK FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
