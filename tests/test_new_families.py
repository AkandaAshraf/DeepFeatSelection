import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from deepfeatselect.redundancy import redundancy_scores
from deepfeatselect.shapley import loco_for_comparison
from deepfeatselect.synthetic import manifold_redundancy, parity_redundancy


def _cv_acc(x, y, seed=0):
    return cross_val_score(
        RandomForestClassifier(n_estimators=120, random_state=seed), x, y, cv=3
    ).mean()


# --- parity_redundancy -----------------------------------------------------

def test_parity_summary_determines_the_target_exactly():
    """summary == (-1)^y, so one threshold on it recovers the label."""
    system = parity_redundancy(n=1200, seed=0)
    x, y = system["x"], system["y"]
    names = system["feature_names"]
    summary = x[:, names.index("summary")]
    assert np.array_equal((summary < 0).astype(np.float64), y)


def test_parity_bits_jointly_determine_the_target():
    system = parity_redundancy(n=1200, n_bits=8, k=4, seed=0)
    x, y = system["x"], system["y"]
    names = system["feature_names"]
    cols = [names.index(n) for n in system["causes"]]
    assert np.array_equal(x[:, cols].sum(axis=1) % 2, y)


def test_parity_irrelevant_bits_carry_nothing():
    system = parity_redundancy(n=1500, seed=0)
    x, y = system["x"], system["y"]
    names = system["feature_names"]
    for name in system["irrelevant"]:
        acc = _cv_acc(x[:, [names.index(name)]], y)
        assert acc < 0.58, name


def test_parity_single_causal_bit_is_uninformative_alone():
    """No subset of parity bits is partially informative; that is the point."""
    system = parity_redundancy(n=1500, seed=0)
    x, y = system["x"], system["y"]
    names = system["feature_names"]
    acc = _cv_acc(x[:, [names.index("bit_0")]], y)
    assert acc < 0.58


def test_parity_redundancy_is_relative_to_the_function_class():
    """Redundancy is a property of (data, class), not of the data alone.

    ``summary`` and the parity bits are informationally interchangeable, so
    Proposition 1 -- an infimum over *all measurable* functions -- sends both
    leave-one-out importances to zero. A forest cannot compute XOR over four
    inputs, so for that class the substitution is unavailable: deleting
    ``summary`` costs real accuracy, while deleting a single bit costs nothing
    because ``summary`` still carries the label. The asymmetry is Proposition 2
    appearing inside the value function itself, and it is why this family sits
    alongside ``redundancy_demo`` rather than replacing it.
    """
    system = parity_redundancy(n=1500, seed=0)
    loco = loco_for_comparison(
        system["x"], system["y"], system["feature_names"], seed=0
    ).set_index("feature")

    assert loco.loc["summary", "loco"] > 0.10
    for name in system["causes"]:
        assert abs(loco.loc[name, "loco"]) < 0.05, name
    for name in system["irrelevant"]:
        assert abs(loco.loc[name, "loco"]) < 0.05, name


# --- manifold_redundancy ---------------------------------------------------

def test_manifold_sensors_are_individually_redundant():
    """Any two sensors invert the latent, so each is reconstructible."""
    system = manifold_redundancy(n=1500, n_sensors=6, seed=0)
    audit = redundancy_scores(
        system["x"], system["feature_names"], seed=0
    ).set_index("feature")
    for name in system["feature_names"]:
        if name.startswith("sensor_"):
            assert audit.loc[name, "redundant"], name
    assert not audit.loc["unrelated", "redundant"]


def test_manifold_aligned_sensor_is_individually_sufficient():
    system = manifold_redundancy(n=1500, seed=0)
    names = system["feature_names"]
    aligned = _cv_acc(system["x"][:, [names.index("sensor_0")]], system["y"])
    assert aligned > 0.95


def test_manifold_off_axis_sensor_needs_a_partner():
    """A single oblique projection cannot resolve the sign of s1 on its own."""
    system = manifold_redundancy(n=1500, n_sensors=6, seed=0)
    names = system["feature_names"]
    alone = _cv_acc(system["x"][:, [names.index("sensor_3")]], system["y"])
    paired = _cv_acc(
        system["x"][:, [names.index("sensor_3"), names.index("sensor_4")]], system["y"]
    )
    assert alone < 0.90
    assert paired > alone + 0.05


def test_manifold_is_reproducible_and_bounded():
    a = manifold_redundancy(n=400, seed=1)
    b = manifold_redundancy(n=400, seed=1)
    assert np.array_equal(a["x"], b["x"])
    sensors = a["x"][:, :-1]
    assert np.abs(sensors).max() <= 1.0  # tanh range
