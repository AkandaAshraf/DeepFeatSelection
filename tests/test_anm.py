import numpy as np
import pytest

from deepfeatselect.anm import anm_orient

N = 600


def test_nonlinear_anm_orients_forward():
    # A non-invertible mechanism: the reverse regression averages the two
    # branches and leaves a large structured residual, so the asymmetry is
    # robust at unit-test sample sizes rather than borderline.
    rng = np.random.default_rng(0)
    x = rng.normal(size=N)
    y = x**2 + 0.3 * rng.normal(size=N)
    assert anm_orient(x, y, seed=0).verdict == "x->y"


def test_orientation_flips_with_argument_order():
    """The same pair fed reversed must yield the mirrored verdict, not a new one."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=N)
    y = x**2 + 0.3 * rng.normal(size=N)
    assert anm_orient(y, x, seed=0).verdict == "y->x"


@pytest.mark.slow
def test_invertible_nonlinear_pair_orients_at_scale():
    """The hardest identifiable case: smooth invertible f, so no branch
    structure helps.  The backward-dependence signal is real but weak, and at
    n=600 a single draw can sit on the alpha boundary; at n=2000 it is clean."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    y = np.tanh(2.0 * x) + x**3 / 4.0 + 0.3 * rng.normal(size=2000)
    assert anm_orient(x, y, seed=0).verdict == "x->y"


def test_linear_gaussian_is_undecided():
    """The textbook unidentifiable case must refuse, not guess."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=N)
    y = 0.8 * x + 0.6 * rng.normal(size=N)
    assert anm_orient(x, y, seed=0).verdict == "undecided: both admissible"


def test_independent_pair():
    rng = np.random.default_rng(3)
    result = anm_orient(rng.normal(size=N), rng.normal(size=N), seed=0)
    assert result.verdict == "independent"


def test_common_cause_refuses_both_directions():
    rng = np.random.default_rng(4)
    s = rng.normal(size=N)
    x = np.tanh(2.0 * s) + 0.2 * rng.normal(size=N)
    y = s**2 - 1.0 + 0.2 * rng.normal(size=N)
    assert anm_orient(x, y, seed=0).verdict == "no ANM in either direction"


def test_deterministic_non_invertible_orients_by_functional_asymmetry():
    """No noise anywhere, yet the direction is recoverable: the effect is a
    function of the cause, the cause is not a function of the effect."""
    rng = np.random.default_rng(5)
    u = rng.uniform(0.0, 1.0, size=N)
    proxy = np.cos(2.0 * np.pi * u)
    assert anm_orient(u, proxy, seed=0).verdict == "x->y (deterministic)"
    assert anm_orient(proxy, u, seed=0).verdict == "y->x (deterministic)"


def test_deterministic_bijection_is_refused():
    rng = np.random.default_rng(6)
    x = rng.uniform(-1.0, 1.0, size=N)
    y = 2.0 * x + 1.0
    assert anm_orient(x, y, seed=0).verdict == "deterministic bijection: unidentifiable"


def test_result_carries_both_pvalues():
    rng = np.random.default_rng(7)
    x = rng.normal(size=N)
    y = np.tanh(2.0 * x) + 0.3 * rng.normal(size=N)
    result = anm_orient(x, y, seed=0)
    assert 0.0 < result.p_forward <= 1.0
    assert 0.0 < result.p_backward <= 1.0
    assert result.r2_forward > 0.5
