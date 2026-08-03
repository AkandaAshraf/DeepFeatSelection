import numpy as np
import pytest

from deepfeatselect.biomarker import estimate_ate, simulate_trial, stratified_effect
from deepfeatselect.redundancy import (
    equivalence_classes,
    pairwise_predictability,
    redundancy_scores,
)


def test_trial_is_reproducible_and_well_formed():
    a, b = simulate_trial(n=500, seed=0), simulate_trial(n=500, seed=0)
    assert np.array_equal(a.x, b.x)
    assert a.x.shape == (500, len(a.feature_names))
    assert set(np.unique(a.treatment)) == {0.0, 1.0}


def test_ground_truth_roles_are_disjoint():
    trial = simulate_trial(n=500, seed=0)
    roles = [set(trial.redundant_markers), set(trial.post_treatment),
             set(trial.confounders), set(trial.irrelevant)]
    for i, first in enumerate(roles):
        for second in roles[i + 1:]:
            assert not (first & second)
    for group in roles:
        assert group <= set(trial.feature_names)


def test_post_treatment_columns_excluded_from_candidates():
    trial = simulate_trial(n=500, seed=0)
    assert "symptom_score" not in trial.pre_treatment_names
    assert "biopsy_marker" in trial.pre_treatment_names


def test_randomisation_recovers_the_true_ate():
    """The estimator the trial design licenses, checked against ground truth."""
    covered = 0
    for seed in range(8):
        trial = simulate_trial(n=4000, seed=seed)
        effect, _, half = estimate_ate(trial)
        if abs(effect - trial.true_ate) <= half:
            covered += 1
    # A 95% interval over 8 replications: allow one miss before complaining.
    assert covered >= 7


def test_treatment_is_independent_of_the_latent_driver():
    """Randomisation must break the confounding, or the trial is not a trial."""
    trial = simulate_trial(n=4000, seed=0)
    corr = np.corrcoef(trial.treatment, trial.latent_driver)[0, 1]
    assert abs(corr) < 0.05


def test_redundancy_audit_separates_readouts_from_noise():
    trial = simulate_trial(n=1200, seed=0)
    audit = redundancy_scores(trial.x, trial.feature_names, seed=0).set_index("feature")
    for marker in trial.redundant_markers:
        assert audit.loc[marker, "redundant"], marker
    for name in ("noise_1", "noise_2", "sex"):
        assert not audit.loc[name, "redundant"], name


def test_pairwise_predictability_is_directional():
    """A squared readout is predictable from the pathway, not the reverse."""
    trial = simulate_trial(n=1500, seed=0)
    matrix = pairwise_predictability(trial.x, trial.feature_names, seed=0)
    forward = matrix.loc["biopsy_marker", "blood_marker_b"]
    backward = matrix.loc["blood_marker_b", "biopsy_marker"]
    assert forward > 0.9
    assert backward < forward - 0.2


def test_equivalence_classes_are_symmetric_and_nontrivial():
    trial = simulate_trial(n=1500, seed=0)
    classes = equivalence_classes(trial.x, trial.feature_names, seed=0)
    assert classes, "expected at least one interchangeable class"
    joined = {name for group in classes for name in group}
    assert "noise_1" not in joined
    assert joined <= set(trial.redundant_markers)


def test_effect_modification_detected_on_a_monotone_readout():
    trial = simulate_trial(n=4000, seed=0)
    _, _, gap = stratified_effect(trial, trial.column("biopsy_marker"))
    _, _, null_gap = stratified_effect(trial, trial.column("noise_1"))
    assert gap > 0.5
    assert abs(null_gap) < 0.2


def test_squared_readout_is_redundant_yet_useless_for_stratification():
    """Informational redundancy does not imply clinical interchangeability.

    blood_marker_b tracks the pathway's magnitude, not its sign, so a median
    split on it does not separate high benefit from low even though the column
    is reconstructible from the others.
    """
    trial = simulate_trial(n=4000, seed=0)
    _, _, gap = stratified_effect(trial, trial.column("blood_marker_b"))
    assert abs(gap) < 0.2
