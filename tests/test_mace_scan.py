"""The packaged scan must find driven channels and clear its own controls."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mace import MaceConfig, scan


def coupled_system(n=900, v=12, n_driven=4, seed=0):
    """Logistic drivers push a minority of channels; the rest are autonomous."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(3.6, 3.8, v)
    x = rng.uniform(0.2, 0.8, (n, v))
    for t in range(n - 1):
        drive = np.zeros(v)
        for j in range(n_driven):
            drive[j] = 0.3 * x[t, v - 1 - j]      # last channels drive first
        x[t + 1] = np.clip(r * x[t] * (1 - x[t]) - drive * x[t], 0, 1)
    driven = np.zeros(v, bool)
    driven[:n_driven] = True
    return x, driven


FAST = MaceConfig(degree=2, models=2, epochs=6, ghosts=12, bottleneck=8,
                  difference=False, device="cpu")


def test_scan_separates_driven_from_autonomous():
    x, driven = coupled_system()
    res = scan(x, FAST)
    assert res.excess[driven].mean() > res.excess[~driven].mean()
    top4 = np.argsort(-res.excess)[:4]
    assert driven[top4].sum() >= 3


def test_ghost_panel_bounds_autonomous_channels():
    x, driven = coupled_system()
    res = scan(x, FAST)
    assert res.threshold < res.excess[driven].max()
    assert float(np.median(res.ghosts)) < 0.05


def test_report_carries_the_controls():
    x, _ = coupled_system(n=700)
    res = scan(x, FAST)
    gate = res.gate()
    assert set(gate) == {"G1_length", "G2_saturation", "G3_stationarity"}
    assert "not evidence" in res.summary().lower() or \
           "NOT evidence" in res.summary()
    frame = res.to_frame()
    assert {"channel", "excess", "self_r2", "driven"} <= set(frame.columns)
