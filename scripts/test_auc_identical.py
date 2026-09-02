"""Assert the inlined auc() in ccm_pcmci_v60.py matches error_metrics.auc.

ccm_pcmci_v60.py inlines auc() rather than importing it, because
error_metrics imports torch and source_outflow_gate (~330 MB) for six lines
of numpy, and the V=60 run died of an out-of-memory. Inlining a metric is
exactly the kind of shortcut that lets two copies drift apart silently, so
this test exists to make that impossible.

    python scripts/test_auc_identical.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def load_inlined():
    spec = importlib.util.spec_from_file_location(
        "_v60", str(HERE / "ccm_pcmci_v60.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["_v60"] = m
    spec.loader.exec_module(m)
    return m.auc


def main() -> int:
    inlined = load_inlined()
    from error_metrics import auc as original   # pulls torch; fine in a test

    rng = np.random.default_rng(0)
    cases = [
        (rng.normal(1, 1, 40), rng.normal(0, 1, 60)),      # typical
        (rng.normal(0, 1, 10), rng.normal(0, 1, 55)),      # the chamber shape
        (np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0])),  # all ties
        (np.array([2.0, 3.0]), np.array([0.0, 1.0])),      # perfect
        (np.array([0.0, 1.0]), np.array([2.0, 3.0])),      # inverted
        (np.array([]), np.array([1.0, 2.0])),              # empty positives
        (np.array([1.0, 2.0]), np.array([])),              # empty negatives
    ]
    bad = 0
    for i, (p, n) in enumerate(cases):
        a, b = inlined(p, n), original(p, n)
        same = (np.isnan(a) and np.isnan(b)) or a == b
        print(f"  case {i}: inlined {a!r:>22}  original {b!r:>22}  "
              f"{'OK' if same else 'DIFFERS'}")
        bad += not same

    # random fuzz
    for _ in range(2000):
        p = rng.normal(size=rng.integers(1, 30))
        n = rng.normal(size=rng.integers(1, 30))
        if inlined(p, n) != original(p, n):
            bad += 1
    print(f"\n2000 random comparisons + {len(cases)} edge cases: "
          f"{'ALL IDENTICAL' if not bad else f'{bad} MISMATCHES'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
