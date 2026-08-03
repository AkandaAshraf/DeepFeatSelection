import numpy as np
import pandas as pd
import pytest

from deepfeatselect.data import Dataset
from deepfeatselect.experiment import summarise
from deepfeatselect.train import TrainConfig, train_one

FEATURES = ["a", "b", "c"]


def _runs(gate_rows):
    rows = []
    for i, gates in enumerate(gate_rows):
        row = {"seed": i, "epochs_run": 10, "test_f1": 0.8}
        row.update(dict(zip(FEATURES, gates)))
        rows.append(row)
    return pd.DataFrame(rows)


def test_summarise_ranks_by_mean_share():
    runs = _runs([[0.6, 0.3, 0.1], [0.5, 0.4, 0.1]])
    out = summarise(runs, FEATURES)
    assert list(out["feature"]) == ["a", "b", "c"]
    assert out["importance"].sum() == pytest.approx(1.0)


def test_summarise_normalises_each_run_before_averaging():
    """A run with uniformly larger gates must not dominate the average."""
    runs = _runs([[1.0, 1.0, 8.0], [100.0, 100.0, 800.0]])
    out = summarise(runs, FEATURES).set_index("feature")
    assert out.loc["c", "importance"] == pytest.approx(0.8)


def test_summarise_drops_collapsed_runs():
    runs = _runs([[0.6, 0.3, 0.1], [0.0, 0.0, 0.0]])
    out = summarise(runs, FEATURES).set_index("feature")
    assert out.loc["a", "importance"] == pytest.approx(0.6)


def test_summarise_raises_when_everything_collapsed():
    with pytest.raises(ValueError, match="l1_gate is too strong"):
        summarise(_runs([[0.0, 0.0, 0.0]]), FEATURES)


def test_bootstrap_ci_brackets_the_mean():
    runs = _runs([[0.6, 0.3, 0.1], [0.4, 0.4, 0.2], [0.5, 0.35, 0.15]])
    out = summarise(runs, FEATURES)
    assert (out["ci_low"] <= out["importance"]).all()
    assert (out["importance"] <= out["ci_high"]).all()


def _synthetic(n=600, n_informative=2, n_noise=6, seed=0):
    """Only the first `n_informative` features carry any signal about y."""
    rng = np.random.default_rng(seed)
    n_features = n_informative + n_noise
    x = rng.normal(size=(n, n_features)).astype("float32")
    logit = x[:, :n_informative].sum(axis=1) * 3.0
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(np.int64)

    cut_a, cut_b = int(n * 0.6), int(n * 0.8)
    return Dataset(
        x_train=x[:cut_a], y_train=y[:cut_a],
        x_val=x[cut_a:cut_b], y_val=y[cut_a:cut_b],
        x_test=x[cut_b:], y_test=y[cut_b:],
        feature_names=[f"f{i}" for i in range(n_features)],
        groups=np.arange(n_features, dtype=np.int32),
        n_classes=2,
    )


@pytest.mark.slow
def test_l1_penalty_actually_sparsifies_the_gates():
    """The core methodological claim: L1 on the gates must shrink useless ones.

    An L1 term on the gate alone would be toothless, because the network could
    shrink a gate and inflate the matching first-layer column at no cost. This
    checks that the L1 + weight-decay combination really does drive noise gates
    down relative to informative ones.
    """
    data = _synthetic()
    base = dict(task="binary", epochs=120, patience=20, batch_size=64, learning_rate=3e-3)

    unpenalised = train_one(data, TrainConfig(l1_gate=0.0, **base), seed=0)
    penalised = train_one(data, TrainConfig(l1_gate=0.05, **base), seed=0)

    def signal_share(gates):
        return gates[:2].sum() / gates.sum()

    assert signal_share(penalised.gates) > signal_share(unpenalised.gates)
    # The informative features must come out on top under the penalty.
    assert set(np.argsort(-penalised.gates)[:2]) == {0, 1}
    # And the noise gates must be genuinely suppressed, not merely reordered.
    assert penalised.gates[2:].mean() < unpenalised.gates[2:].mean()


@pytest.mark.slow
def test_train_one_reports_usable_heldout_metrics():
    result = train_one(
        _synthetic(),
        TrainConfig(task="binary", epochs=120, patience=20, learning_rate=3e-3),
        seed=0,
    )
    assert result.metrics["test_auc"] > 0.85
    assert 0.0 <= result.metrics["test_f1"] <= 1.0
    assert result.epochs_run > 0
