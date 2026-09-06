"""Behavioral regression: PRODUCTION splits_for, not a copied helper.

A prior version of this test defined its own splits_for and its own raw-
support formula, so reverting the production fix would still pass -- caught
in review. This version imports splits_for directly from each of the three
affected modules, and derives raw index support from time_delay_embed's own
returned `times` array rather than assuming (E-1)*tau; a change to the
embedding function itself would be caught here too, not just a change to
the embargo constant.

Two usage patterns exist across the three modules and both are checked:

  SAME-ROW (celegans_detect.py, circadian_detect.py, and the autoencoder
  fit in celegans_excess.py): manifold row i's own raw support is compared
  directly across the tr/va and va/te seams.

  FORECAST-SHIFTED (celegans_excess.py's ridge readout): a training PAIR is
  (row i, row i+1's own value), since `target[idx + 1]` is used. That pair's
  raw support is row i's support UNION row (i+1)'s single-sample touch, one
  more than the plain row support -- included explicitly, not assumed to be
  covered by the same-row check.

A known-bad case (embargo = span, not span+1) is included as an ORACLE: if
this test cannot detect that overlap, the test itself is broken, not just
lenient.

    python scripts/test_embargo_boundary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from deepfeatselect.ccm import time_delay_embed  # noqa: E402
import celegans_detect as _cd   # noqa: E402
import celegans_excess as _ce   # noqa: E402
import circadian_detect as _ci  # noqa: E402

MODULES = {"celegans_detect": _cd, "celegans_excess": _ce,
          "circadian_detect": _ci}


def raw_support_from_times(times: np.ndarray, row: int, tau: int, E: int,
                            forecast_shift: bool = False) -> set[int]:
    """Raw (pre-difference) sample indices touched by manifold row `row`,
    derived from time_delay_embed's OWN `times` output, not an assumed
    formula: differenced index d touches raw {d, d+1} (np.diff), and row
    `row`'s differenced indices are {times[row] - k*tau for k in range(E)}.

    If `forecast_shift`, also include row `row+1`'s own (offset-0) raw
    touch, matching a training pair (input=row, target=row+1's value) as
    celegans_excess.py's ridge readout actually uses.
    """
    diffs = {int(times[row] - k * tau) for k in range(E)}
    out: set[int] = set()
    for d in diffs:
        out |= {d, d + 1}
    if forecast_shift:
        d1 = int(times[row + 1])            # column 0 = offset 0 = times[i]
        out |= {d1, d1 + 1}
    return out


def build_manifold(n_raw: int, E: int, tau: int):
    rng = np.random.default_rng(0)
    raw = rng.standard_normal(n_raw)
    z = np.diff(raw)
    manifold, times = time_delay_embed(z, E, tau=tau)
    return manifold, times


def check_same_row(mod_name: str, splits_for, E: int, tau: int,
                    n_raw: int = 800) -> tuple[bool, str]:
    manifold, times = build_manifold(n_raw, E, tau)
    n = manifold.shape[0]
    tr, va, te = splits_for(n)
    tr_i = range(*tr.indices(n))
    va_i = range(*va.indices(n))
    te_i = range(*te.indices(n))

    def support(idxs):
        out: set[int] = set()
        for i in idxs:
            out |= raw_support_from_times(times, i, tau, E)
        return out

    s_tr, s_va, s_te = support(tr_i), support(va_i), support(te_i)
    ov1, ov2 = s_tr & s_va, s_va & s_te
    ok = not ov1 and not ov2
    return ok, (f"{mod_name} tau={tau} SAME-ROW  tr/va overlap="
               f"{sorted(ov1) or 'none'}  va/te overlap={sorted(ov2) or 'none'}")


def check_forecast_shift(n_raw: int = 800) -> tuple[bool, str]:
    """celegans_excess.py's actual pattern: tr_idx = arange(tr.start,
    tr.stop-1), te_idx = arange(te.start, n-1), each i predicting row i+1.
    Not an adjacent seam (va sits between tr and te), included for
    completeness per review request to include forecast-target indexing."""
    E, tau = _ce.E, _ce.TAU
    manifold, times = build_manifold(n_raw, E, tau)
    n = manifold.shape[0]
    tr, va, te = _ce.splits_for(n)
    tr_idx = range(tr.start, tr.stop - 1)
    te_idx = range(te.start, n - 1)

    def support(idxs):
        out: set[int] = set()
        for i in idxs:
            out |= raw_support_from_times(times, i, tau, E,
                                          forecast_shift=True)
        return out

    s_tr, s_te = support(tr_idx), support(te_idx)
    ok = not (s_tr & s_te)
    return ok, (f"celegans_excess tau={tau} FORECAST-SHIFTED tr/te "
               f"overlap={sorted(s_tr & s_te) or 'none'} (not an adjacent "
               f"seam; checked for completeness)")


def oracle_known_bad(mod_name: str, mod, n_raw: int = 800) -> tuple[bool, str]:
    """embargo=span (the FIRST, incomplete correction) must still show an
    overlap under this same enumeration. If it does not, this test cannot
    be trusted to have caught the real fix either."""
    E, tau = mod.E, mod.TAU
    manifold, times = build_manifold(n_raw, E, tau)
    n = manifold.shape[0]
    span = (E - 1) * tau
    a = int(mod.TRAIN_FRACTION * n)
    b = int((mod.TRAIN_FRACTION + mod.VAL_FRACTION) * n)
    tr = range(0, a - span)
    va = range(a, b - span)

    def support(idxs):
        out: set[int] = set()
        for i in idxs:
            out |= raw_support_from_times(times, i, tau, E)
        return out

    overlap = support(tr) & support(va)
    detected = bool(overlap)
    return detected, (f"{mod_name} ORACLE (embargo=span, known-bad) "
                      f"overlap detected={detected} {sorted(overlap)}")


def main() -> int:
    print("Behavioral embargo check against PRODUCTION splits_for.\n")
    all_ok = True

    for name, mod in MODULES.items():
        e0, t0 = mod.E, mod.TAU
        for tau in (1, 3):
            mod.E, mod.TAU = 3, tau
            try:
                ok, msg = check_same_row(name, mod.splits_for, mod.E, mod.TAU)
            finally:
                mod.E, mod.TAU = e0, t0
            print(f"   {'PASS' if ok else 'FAIL'}  {msg}")
            all_ok &= ok

    ok, msg = check_forecast_shift()
    print(f"   {'PASS' if ok else 'FAIL'}  {msg}")
    all_ok &= ok

    print("\nOracle: does this enumeration detect the KNOWN-BAD case?")
    for name, mod in MODULES.items():
        det, msg = oracle_known_bad(name, mod)
        print(f"   {'PASS' if det else 'FAIL (test cannot detect defects)'}"
              f"  {msg}")
        all_ok &= det

    print(f"\nRESULT: {'ALL PASS' if all_ok else 'FAILURES ABOVE'}")
    return 0 if all_ok else 1


def check(E: int, tau: int, embargo: int) -> tuple[bool, str]:
    """Standalone entry point kept for the audit gate's import, testing an
    arbitrary embargo value against a real time_delay_embed manifold using
    the times-derived support function above (not the production
    splits_for, since this checks a hypothetical embargo value directly)."""
    manifold, times = build_manifold(800, E, tau)
    n = manifold.shape[0]
    a = int(0.6 * n)
    b = int(0.8 * n)
    tr, va = range(0, a - embargo), range(a, b - embargo)

    def support(idxs):
        out: set[int] = set()
        for i in idxs:
            out |= raw_support_from_times(times, i, tau, E)
        return out

    ov = support(tr) & support(va)
    return not ov, f"E={E} tau={tau} embargo={embargo} overlap={sorted(ov)}"


if __name__ == "__main__":
    raise SystemExit(main())
