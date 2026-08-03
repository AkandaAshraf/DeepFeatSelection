import numpy as np
import pytest

from deepfeatselect.data import Dataset
from deepfeatselect.probe import (
    _js_divergence,
    _predictive_distribution,
    ablate_feature,
    ablation_probe,
    cka,
    free_energy,
    hopfield_energy,
    loco_importance,
)
from deepfeatselect.train import TrainConfig

LOG2 = 0.6931471805599453
LOG4 = 1.3862943611198906


def test_free_energy_matches_hand_computed_values():
    # -logsumexp of k equal logits is -(logit + log k).
    logits = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert np.allclose(free_energy(logits), [-LOG2, -(1.0 + LOG2)])
    assert free_energy(np.zeros((1, 4)))[0] == pytest.approx(-LOG4)


def test_free_energy_single_logit_is_negative_softplus():
    """The binary head is the two-class case with the first logit pinned at zero."""
    z = np.array([-2.0, 0.0, 3.0])
    two_class = np.stack([np.zeros_like(z), z], axis=1)
    assert np.allclose(free_energy(z), free_energy(two_class))
    assert free_energy(np.array([0.0]))[0] == pytest.approx(-LOG2)
    # softplus saturates at zero for very negative logits, without overflowing.
    assert free_energy(np.array([-800.0]))[0] == pytest.approx(0.0)
    assert free_energy(np.array([800.0]))[0] == pytest.approx(-800.0)


def test_free_energy_accepts_a_column_vector_like_a_binary_head_emits():
    z = np.array([[-2.0], [0.5]])
    assert np.allclose(free_energy(z), free_energy(z.reshape(-1)))


def test_free_energy_binary_branch_agrees_with_a_matching_two_logit_model():
    """Stronger than pinning the first logit at zero by hand: check the two
    parameterisations really are the same classifier before comparing energies,
    so the equivalence is not asserted about two different models."""
    rng = np.random.default_rng(4)
    z = rng.normal(size=200) * 6.0
    two_class = np.stack([np.zeros_like(z), z], axis=1)

    sigmoid = 1.0 / (1.0 + np.exp(-z))
    softmax = np.exp(two_class) / np.exp(two_class).sum(axis=1, keepdims=True)
    assert np.allclose(sigmoid, softmax[:, 1])

    assert np.allclose(free_energy(z), free_energy(two_class))


def test_free_energy_shifts_with_the_logits():
    """E(f + c) = E(f) - c: exactly the offset the softmax discards."""
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(6, 4))
    assert np.allclose(free_energy(logits + 1.7), free_energy(logits) - 1.7)


def test_free_energy_rejects_higher_rank_input():
    with pytest.raises(ValueError, match="1-D or 2-D"):
        free_energy(np.zeros((2, 3, 4)))


def _memories(n=32, d=64, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d))


def test_hopfield_energy_is_lower_near_a_stored_memory():
    rng = np.random.default_rng(1)
    memories = _memories()
    stored = memories[7]

    near = stored + 1e-3 * rng.normal(size=stored.shape)
    # The far query is projected off the stored pattern and rescaled to the same
    # norm, so the comparison is about retrieval and not about the 0.5*q@q term,
    # which would make any large query look high-energy on its own.
    far = rng.normal(size=stored.shape)
    far = far - (far @ stored) / (stored @ stored) * stored
    far *= np.linalg.norm(stored) / np.linalg.norm(far)

    assert hopfield_energy(near, memories)[0] < hopfield_energy(far, memories)[0]


def test_hopfield_energy_is_lowest_at_the_memory_itself():
    memories = _memories()
    stored = memories[3]
    interpolated = [hopfield_energy(t * stored, memories)[0] for t in (0.0, 0.5, 1.0)]
    assert interpolated[2] < interpolated[1] < interpolated[0]


def test_hopfield_energy_handles_a_batch_of_queries():
    memories = _memories()
    energies = hopfield_energy(memories[:5], memories)
    assert energies.shape == (5,)
    assert np.allclose(energies, [hopfield_energy(m, memories)[0] for m in memories[:5]])


def test_hopfield_energy_is_bounded_below_by_construction():
    """The log N and max-norm constants exist to keep E non-negative."""
    memories = _memories()
    assert (hopfield_energy(memories, memories) >= 0).all()


def test_hopfield_energy_rejects_bad_arguments():
    memories = _memories(n=4, d=3)
    with pytest.raises(ValueError, match="beta must be positive"):
        hopfield_energy(np.zeros(3), memories, beta=0.0)
    with pytest.raises(ValueError, match="does not match memory dimension"):
        hopfield_energy(np.zeros(5), memories)
    with pytest.raises(ValueError, match="non-empty 2-D array"):
        hopfield_energy(np.zeros(3), np.zeros((0, 3)))


def _softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def test_js_divergence_is_symmetric_and_zero_against_itself():
    rng = np.random.default_rng(0)
    p = _softmax(rng.normal(size=(20, 4)))
    q = _softmax(rng.normal(size=(20, 4)))
    assert _js_divergence(p, q) == pytest.approx(_js_divergence(q, p))
    assert _js_divergence(p, p) == pytest.approx(0.0)


def test_js_divergence_is_bounded_by_one_bit():
    """The bound is what makes the number readable without calibrating a scale;
    it only holds in base 2, so a missing /log(2) would show up here."""
    disjoint_p = np.array([[1.0, 0.0], [0.0, 1.0]])
    disjoint_q = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert _js_divergence(disjoint_p, disjoint_q) == pytest.approx(1.0)

    rng = np.random.default_rng(1)
    p = _softmax(rng.normal(size=(50, 5)) * 6)
    q = _softmax(rng.normal(size=(50, 5)) * 6)
    assert 0.0 <= _js_divergence(p, q) <= 1.0


def test_js_divergence_survives_exact_zero_probabilities():
    """A saturated sigmoid emits exact 0.0 and 1.0; a bare log would give NaN."""
    p = np.array([[1.0, 0.0]])
    q = np.array([[0.5, 0.5]])
    assert np.isfinite(_js_divergence(p, q))
    assert _js_divergence(p, q) > 0.0


def test_js_divergence_is_computed_on_distributions_not_logits():
    """Rows must be normalised first: the same predictions expressed as logits
    give a different, unbounded answer."""
    logits = np.array([[2.0, -1.0], [0.5, 3.0]])
    probs = _softmax(logits)
    assert _js_divergence(probs, probs[:, ::-1]) <= 1.0
    assert _js_divergence(logits, logits[:, ::-1]) > 1.0


def test_predictive_distribution_expands_a_binary_head_to_two_columns():
    """A sigmoid head emits one column; JS needs both classes to sum to one."""
    probs = np.array([[0.9], [0.1], [0.5]])
    dist = _predictive_distribution(probs, "binary")
    assert dist.shape == (3, 2)
    assert dist.sum(axis=1) == pytest.approx(1.0)
    assert dist[:, 1] == pytest.approx(probs.reshape(-1))
    multi = _softmax(np.random.default_rng(0).normal(size=(3, 4)))
    assert _predictive_distribution(multi, "multiclass") is multi


def _reps(n=40, d=5, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d))


def test_cka_of_a_representation_with_itself_is_one():
    a = _reps()
    assert cka(a, a) == pytest.approx(1.0)


def test_cka_is_invariant_to_orthogonal_rotation():
    a = _reps()
    b = _reps(seed=1)
    q, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(a.shape[1], a.shape[1])))
    assert cka(a, a @ q) == pytest.approx(1.0)
    assert cka(a, b @ q) == pytest.approx(cka(a, b))


def test_cka_is_invariant_to_isotropic_rescaling():
    a = _reps()
    b = _reps(seed=1)
    assert cka(a, 7.5 * a) == pytest.approx(1.0)
    assert cka(a, 0.01 * b) == pytest.approx(cka(a, b))


def test_cka_is_invariant_to_translation():
    """Column centring is part of the definition, not a preprocessing step."""
    a = _reps()
    b = _reps(seed=1)
    assert cka(a, b + 3.0) == pytest.approx(cka(a, b))


def test_cka_is_symmetric_and_below_one_for_unrelated_representations():
    a = _reps()
    b = _reps(seed=1)
    assert cka(a, b) == pytest.approx(cka(b, a))
    assert 0.0 <= cka(a, b) < 0.9


def test_cka_compares_representations_of_different_widths():
    a = _reps(d=5)
    b = _reps(d=9, seed=1)
    assert 0.0 <= cka(a, b) <= 1.0


def test_cka_of_a_constant_representation_is_nan():
    a = _reps()
    assert np.isnan(cka(a, np.ones((a.shape[0], 3))))


def test_cka_requires_matching_example_counts():
    with pytest.raises(ValueError, match="same examples"):
        cka(_reps(n=10), _reps(n=11))


def _synthetic(n=600, n_informative=2, n_noise=6, seed=0, groups=None):
    """Only the first `n_informative` features carry any signal about y."""
    rng = np.random.default_rng(seed)
    n_columns = n_informative + n_noise
    x = rng.normal(size=(n, n_columns)).astype("float32")
    logit = x[:, :n_informative].sum(axis=1) * 3.0
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(np.int64)

    groups = np.arange(n_columns, dtype=np.int32) if groups is None else np.asarray(groups)
    cut_a, cut_b = int(n * 0.6), int(n * 0.8)
    return Dataset(
        x_train=x[:cut_a], y_train=y[:cut_a],
        x_val=x[cut_a:cut_b], y_val=y[cut_a:cut_b],
        x_test=x[cut_b:], y_test=y[cut_b:],
        feature_names=[f"f{i}" for i in range(int(groups.max()) + 1)],
        groups=groups,
        n_classes=2,
    )


def test_ablate_feature_zeroes_every_column_of_a_one_hot_group():
    """A nominal feature spans several columns and all of them have to go."""
    groups = np.array([0, 1, 1, 1, 2], dtype=np.int32)
    data = _synthetic(n=50, n_informative=1, n_noise=4, groups=groups)
    ablated = ablate_feature(data, 1)

    for before, after in (
        (data.x_train, ablated.x_train),
        (data.x_val, ablated.x_val),
        (data.x_test, ablated.x_test),
    ):
        assert after.shape == before.shape
        assert (after[:, 1:4] == 0).all()
        assert (after[:, [0, 4]] == before[:, [0, 4]]).all()


def test_ablate_feature_keeps_the_architecture_comparable():
    """Column count and group vector must survive, or the two models differ."""
    data = _synthetic(n=50, n_informative=1, n_noise=4)
    ablated = ablate_feature(data, 2)
    assert ablated.n_columns == data.n_columns
    assert (ablated.groups == data.groups).all()
    assert ablated.feature_names == data.feature_names
    # The source dataset must not be touched: probes reuse it for every feature.
    assert (data.x_train[:, 2] != 0).any()


def test_ablate_feature_rejects_an_unknown_feature():
    data = _synthetic(n=20, n_informative=1, n_noise=2)
    with pytest.raises(ValueError, match="feature_index must be in"):
        ablate_feature(data, 99)


def _fast_config(**kwargs):
    base = dict(
        task="binary",
        l1_gate=0.05,
        epochs=120,
        patience=20,
        batch_size=64,
        learning_rate=3e-3,
    )
    base.update(kwargs)
    return TrainConfig(**base)


@pytest.mark.slow
def test_ablating_a_noise_feature_barely_moves_auc():
    """The sanity check the whole probe rests on: removing nothing costs nothing."""
    data = _synthetic()
    result = ablation_probe(data, _fast_config(), feature_index=5, seed=0)

    assert result.feature == "f5"
    assert result.n_columns == 1
    assert abs(result.delta_auc) < 0.05
    # The remaining probes must still be well defined, not NaN or out of range.
    assert 0.0 <= result.js_divergence <= 1.0
    assert result.energy_shift >= 0.0
    assert result.hopfield_shift >= 0.0
    assert 0.0 <= result.representation_cka <= 1.0


@pytest.mark.slow
def test_loco_with_an_empty_feature_selection_returns_an_empty_table():
    """`features` is public, and the sort key does not exist on an empty frame."""
    data = _synthetic(n=120, n_informative=1, n_noise=2)
    config = _fast_config(epochs=3, patience=2, hidden_units=4, n_hidden_layers=1)
    table = loco_importance(data, config, features=[], progress=False)
    assert len(table) == 0


@pytest.mark.slow
def test_loco_reports_one_row_per_feature_and_ranks_signal_above_noise():
    data = _synthetic(n=400, n_informative=1, n_noise=3)
    config = _fast_config(epochs=60, patience=15, hidden_units=16, n_hidden_layers=1)
    table = loco_importance(data, config, seed=0, progress=False)

    assert len(table) == data.n_features
    assert set(table["feature"]) == set(data.feature_names)
    assert not table[["delta_auc", "js_divergence", "representation_cka"]].isna().any().any()
    # Sorted by delta_auc, so the one informative feature must come out on top.
    assert table.iloc[0]["feature"] == "f0"
