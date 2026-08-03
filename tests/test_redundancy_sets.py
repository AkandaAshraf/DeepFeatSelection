import numpy as np
import pytest

from deepfeatselect.redundancy import group_loco, minimal_removal_set, redundancy_scores
from deepfeatselect.synthetic import redundancy_demo


@pytest.fixture(scope="module")
def demo():
    system = redundancy_demo(n=1500, seed=0)
    x = np.asarray(system["x"], dtype=np.float64)
    y = (np.asarray(system["y"]) > np.median(system["y"])).astype(np.float64)
    return x, y, list(system["feature_names"]), system


def test_every_informative_feature_is_individually_redundant(demo):
    """The premise of Proposition 1, verified on the benchmark."""
    x, _, names, _ = demo
    audit = redundancy_scores(x, names, seed=0).set_index("feature")
    for name in ("driver", "proxy_cos", "proxy_sin"):
        assert audit.loc[name, "redundant"], name
    assert not audit.loc["unrelated", "redundant"]


def test_minimal_removal_set_matches_the_generator(demo):
    """y is reachable via driver or via proxy_cos, so both must go before it dies."""
    x, y, names, system = demo
    removal, drop = minimal_removal_set(x, y, names, min_drop=0.10, max_size=3, seed=0)
    assert set(removal) == {"driver", "proxy_cos"}
    assert drop >= 0.10
    # Ground truth recorded independently by the generator.
    sufficient = {frozenset(s) for s in system["minimal_sufficient_sets"]}
    assert sufficient == {frozenset({"driver"}), frozenset({"proxy_cos"})}
    # A removal set must intersect every sufficient set, or a route survives.
    for group in sufficient:
        assert group & set(removal)


def test_group_loco_is_large_where_single_feature_loco_is_zero(demo):
    """The repair: importance is well defined for the group, not its members."""
    x, y, names, _ = demo
    singles = group_loco(x, y, names, [["driver"], ["proxy_cos"]], seed=0)
    assert singles.r2_drop.abs().max() < 0.02

    grouped = group_loco(x, y, names, [["driver", "proxy_cos"]], seed=0)
    assert grouped.iloc[0].r2_drop > 0.10


def test_removing_an_irrelevant_feature_never_reaches_the_threshold(demo):
    x, y, names, _ = demo
    dropped = group_loco(x, y, names, [["unrelated"]], seed=0)
    assert abs(dropped.iloc[0].r2_drop) < 0.02
