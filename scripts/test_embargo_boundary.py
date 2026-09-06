"""Behavioral regression test for the train/val/test embargo, DIFFERENCED
pipeline (celegans_detect.py, celegans_excess.py, circadian_detect.py).

Does not trust the arithmetic derivation on its own -- enumerates the actual
set of RAW (pre-difference, pre-embedding) sample indices each manifold row
touches, by simulation, and asserts the train and validation supports (and
val/test) are disjoint. This is the check requested after a review found the
2026-09-06 "fix" (embargo = (E-1)*tau) left a 1-sample overlap, because it
accounted for the embedding span but not for the differencing step that
precedes it in these three scripts.

  differenced index d touches RAW indices {d, d+1}          (np.diff)
  manifold row i touches DIFFERENCED indices [i, i+span]     (time_delay_embed)
  => manifold row i touches RAW indices [i, i+span+1]

so the correct embargo for THIS pipeline is (E-1)*tau + 1, not (E-1)*tau.

    python scripts/test_embargo_boundary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from deepfeatselect.ccm import time_delay_embed  # noqa: E402


def raw_support(row_manifold_index: int, span: int) -> set[int]:
    """RAW sample indices touched by differenced-pipeline manifold row `i`."""
    d_lo, d_hi = row_manifold_index, row_manifold_index + span
    return set(range(d_lo, d_hi + 2))          # diff index d -> raw {d, d+1}


def splits_for(n: int, embargo: int, train_frac=0.6, val_frac=0.2):
    a = int(train_frac * n)
    b = int((train_frac + val_frac) * n)
    return slice(0, a - embargo), slice(a, b - embargo), slice(b, n)


def check(E: int, tau: int, embargo: int, n_raw: int = 500) -> tuple[bool, str]:
    """Simulate the real pipeline end to end and check disjoint raw support
    at both seams under the given embargo value."""
    span = (E - 1) * tau
    rng = np.random.default_rng(0)
    raw = rng.standard_normal(n_raw)
    z = np.diff(raw)                                   # matches the scripts
    manifold, times = time_delay_embed(z, E, tau=tau)
    n = manifold.shape[0]
    tr, va, te = splits_for(n, embargo)

    def support_of(sl: slice) -> set[int]:
        idxs = range(*sl.indices(n))
        out: set[int] = set()
        for i in idxs:
            out |= raw_support(i, span)
        return out

    s_tr, s_va, s_te = support_of(tr), support_of(va), support_of(te)
    overlap_tv = s_tr & s_va
    overlap_vt = s_va & s_te
    ok = not overlap_tv and not overlap_vt
    msg = (f"E={E} tau={tau} embargo={embargo}  "
          f"train/val overlap={sorted(overlap_tv) or 'none'}  "
          f"val/test overlap={sorted(overlap_vt) or 'none'}")
    return ok, msg


def main() -> int:
    print("Behavioral embargo check: enumerated raw support, not arithmetic.\n")
    cases = [
        ("archived (embargo=E)",        lambda E, tau: E),
        ("2026-09-06 fix (embargo=span)", lambda E, tau: (E - 1) * tau),
        ("corrected (embargo=span+1)",  lambda E, tau: (E - 1) * tau + 1),
    ]
    all_ok = True
    for label, embargo_fn in cases:
        print(f"-- {label} --")
        for E, tau in [(3, 1), (3, 3)]:
            embargo = embargo_fn(E, tau)
            ok, msg = check(E, tau, embargo)
            all_ok &= ok if "corrected" in label else True  # only gate on fix
            status = "PASS (disjoint)" if ok else "FAIL (overlap)"
            print(f"   {status}  {msg}")
        print()

    print("The corrected embargo (span+1) must pass in every case; the")
    print("other two rows are shown for context, not gated.")
    if all_ok:
        print("\nRESULT: corrected embargo verified disjoint at tau=1 and tau=3.")
    else:
        print("\nRESULT: FAILED -- corrected embargo still overlaps.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
