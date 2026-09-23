"""Corrected family-1 generator (family 1b): diffusive coupled logistic maps.

Pre-registration: paper/family1_generator_fix_protocol.md (7d20cce). The
original family1_generate in parent_screening.py is left untouched; its
driven channels lock into a period-2 cycle (see the protocol).

    root:   x_q(t+1) = f_q(x_q(t))
    driven: x_q(t+1) = (1 - c) f_q(x_q(t)) + c * mean_j eta_qj f_j(x_j(t+1-d_j))
    f_i(x) = r_i x (1 - x),  c = 0.30,  r_i ~ U(3.8, 4.0) minus periodic windows
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import parent_screening as PS  # noqa: E402


def family1b_step_value(x, t, q, r, eta, parent, is_root, coupling):
    k = x[t, q]
    f_q = r[q] * k * (1 - k)
    if is_root[q]:
        return f_q
    drive = np.mean([eta[q][j] * r[j] * x[max(t - d + 1, 0), j]
                     * (1 - x[max(t - d + 1, 0), j]) for j, d in parent[q]])
    return (1 - coupling) * f_q + coupling * drive


def family1b_generate(V: int, n: int, seed: int, coupling: float = 0.30,
                      r_range=(3.8, 4.0)):
    for attempt in range(PS.MAX_REDRAWS):
        rng = np.random.default_rng(seed * 101 + attempt)
        order, is_root, parent, n_root = PS.build_dag(V, seed * 101 + attempt)
        r = np.empty(V)
        for i in range(V):
            while True:
                cand = float(rng.uniform(*r_range))
                if not PS._is_locked_r(cand):
                    r[i] = cand
                    break
        eta = {q: {j: float(rng.uniform(0.85, 1.15)) for j, _ in plist}
               for q, plist in parent.items()}
        total = 500 + PS.MAX_DELAY + n
        x = np.empty((total, V))
        x[0] = rng.uniform(0.2, 0.8, V)
        for t in range(total - 1):
            nxt = np.array([family1b_step_value(x, t, q, r, eta, parent,
                                                is_root, coupling)
                            for q in range(V)])
            x[t + 1] = np.clip(nxt, 0.0, 1.0)
        x_clean = x[500:]
        if not np.all(np.isfinite(x_clean)):
            continue
        clip_frac = np.mean((x_clean <= 1e-9) | (x_clean >= 1 - 1e-9), axis=0)
        if clip_frac.max() > 0.30 or not PS.orphan_free(is_root, parent):
            continue
        x_kept = x_clean[PS.MAX_DELAY:PS.MAX_DELAY + n]
        a = int(0.6 * n)
        noise = 0.05 * (x_kept[:a].std(axis=0) + 1e-12) * np.random.default_rng(
            seed * 101 + attempt + 5000).standard_normal(x_kept.shape)
        return dict(x_obs=x_kept + noise, x_clean=x_kept, parent=parent,
                    is_root=is_root, order=order, n_root=n_root,
                    clip_frac_max=float(clip_frac.max()),
                    attempts=attempt + 1, family="family1b")
    raise RuntimeError(f"family1b_generate: no valid draw in {PS.MAX_REDRAWS} "
                       f"attempts at V={V} seed={seed}")
