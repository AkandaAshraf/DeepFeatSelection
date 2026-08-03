"""A simulated randomised trial in which biomarker ranking is not identifiable.

The motivating problem is real and costly: a trial reports that patients high on
marker X benefit most from a drug, that finding becomes a companion diagnostic,
and it fails to replicate elsewhere.  Proposition 1 of the accompanying draft
gives one concrete mechanism.  When several assays read out the same latent
pathway, each is a near-deterministic function of the others, so every
risk-difference importance measure scores each of them at approximately zero and
whichever one comes out on top is decided by sampling accident rather than by
biology.

What this module is *not* for
-----------------------------
Deciding whether the drug beats placebo.  Treatment is randomised here, which
identifies the average treatment effect directly; a difference in means answers
it and no causal-discovery machinery is involved or needed.  ``simulate_trial``
records the true ATE so that the analysis script can show the randomised
estimate recovering it, precisely to mark the boundary between the question
randomisation already answers and the question it does not.

What it *is* for
----------------
The question randomisation leaves open: *which measurable quantity should the
companion diagnostic be built on*, and *which one should a drug programme try to
intervene on*.  Those are different questions with different answers, and the
data cannot separate the members of a redundant set for either.

Ground truth recorded by the simulator
--------------------------------------
* ``latent_driver`` -- pathway activity ``P``, the quantity that actually
  modifies the treatment effect.  Unobserved, as it would be in practice.
* ``redundant_markers`` -- the assays that are functions of ``P``.  All are
  legitimate diagnostics; none is a valid drug target, because intervening on a
  readout does not move the pathway that generates it.
* ``post_treatment`` -- a symptom score measured *after* treatment.  It is a
  descendant of the outcome, so it is the most predictive column in the table
  and is invalid as a biomarker.  Including it is a standard trap and is
  included here to be caught.
* ``true_ate`` and ``effect_modifier`` -- for scoring the analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TrialData:
    """One simulated trial, with the causal roles of every column recorded."""

    x: np.ndarray
    feature_names: list[str]
    treatment: np.ndarray
    outcome: np.ndarray
    latent_driver: np.ndarray

    true_ate: float
    redundant_markers: tuple[str, ...]
    post_treatment: tuple[str, ...]
    confounders: tuple[str, ...]
    irrelevant: tuple[str, ...]
    # Cost is not causal information, but it is the tie-breaker a clinician
    # actually has once the data has been shown to be indifferent.
    assay_cost: dict[str, float] = field(default_factory=dict)

    @property
    def pre_treatment_names(self) -> list[str]:
        """Columns legitimate for effect-modification analysis.

        Anything measured after randomisation is excluded: conditioning on a
        post-treatment variable opens a path through the outcome and biases the
        subgroup effect, however predictive the column looks.
        """
        return [n for n in self.feature_names if n not in self.post_treatment]

    def column(self, name: str) -> np.ndarray:
        return self.x[:, self.feature_names.index(name)]


def simulate_trial(
    n: int = 4000,
    seed: int = 0,
    measurement_noise: float = 0.05,
    tau0: float = 0.30,
    tau1: float = 0.80,
) -> TrialData:
    """Simulate a 1:1 randomised trial with a redundantly-measured pathway.

    Args:
        measurement_noise: assay noise on each readout.  At zero the markers are
            exactly redundant and Proposition 1 applies exactly; the default is
            small but non-zero, which is the realistic case and makes the
            resulting importances small rather than identically zero.
        tau0: treatment effect at average pathway activity, so the true ATE is
            ``tau0`` up to the centring of ``P``.
        tau1: effect modification -- how much the benefit grows with ``P``.  This
            is the quantity a companion diagnostic is trying to exploit.
    """
    rng = np.random.default_rng(seed)

    age = rng.uniform(40.0, 80.0, size=n)
    age_z = (age - 60.0) / 10.0
    sex = rng.integers(0, 2, size=n).astype(float)

    # Latent pathway activity, driven partly by demographics and partly by
    # unexplained biology. Centred so that tau0 is interpretable as the ATE.
    pathway = np.tanh(0.8 * age_z) + 0.3 * sex + 0.7 * rng.standard_normal(n)
    pathway = pathway - pathway.mean()

    # A technical artefact shared by the two blood assays, as a plate or batch
    # effect would be. It is predictive of nothing causal.
    batch = rng.standard_normal(n)

    def noisy(values: np.ndarray) -> np.ndarray:
        return values + measurement_noise * rng.standard_normal(n)

    biopsy = noisy(pathway)
    blood_a = noisy(np.tanh(1.5 * pathway)) + 0.10 * batch
    # Non-invertible on purpose: a squared readout cannot recover the sign of the
    # pathway on its own, so it is redundant *given the others* without being
    # pairwise equivalent to any of them.
    blood_b = noisy(pathway**2 - 1.0) + 0.10 * batch
    imaging = noisy(2.0 * pathway)

    treatment = rng.integers(0, 2, size=n).astype(float)

    # Outcome: higher is better. The benefit of treatment grows with pathway
    # activity, which is what makes P the true effect modifier.
    benefit = tau0 + tau1 * pathway
    outcome = (
        0.5 * np.tanh(pathway) - 0.2 * age_z
        + treatment * benefit
        + 0.5 * rng.standard_normal(n)
    )

    # Measured after treatment, so a descendant of the outcome. Highly
    # predictive and completely invalid as a biomarker.
    symptom_score = 1.2 * outcome + 0.3 * rng.standard_normal(n)

    names = [
        "age", "sex", "biopsy_marker", "blood_marker_a", "blood_marker_b",
        "imaging_score", "batch_artifact", "noise_1", "noise_2", "symptom_score",
    ]
    x = np.column_stack([
        age_z, sex, biopsy, blood_a, blood_b, imaging, batch,
        rng.standard_normal(n), rng.standard_normal(n), symptom_score,
    ])

    return TrialData(
        x=x,
        feature_names=names,
        treatment=treatment,
        outcome=outcome,
        latent_driver=pathway,
        # E[P] = 0 by construction, so the ATE is tau0 exactly.
        true_ate=float(tau0),
        redundant_markers=("biopsy_marker", "blood_marker_a", "blood_marker_b", "imaging_score"),
        post_treatment=("symptom_score",),
        confounders=("age", "sex"),
        irrelevant=("batch_artifact", "noise_1", "noise_2"),
        assay_cost={
            "biopsy_marker": 1200.0,   # invasive
            "imaging_score": 400.0,
            "blood_marker_a": 40.0,
            "blood_marker_b": 40.0,
        },
    )


def estimate_ate(trial: TrialData) -> tuple[float, float, float]:
    """Difference in means, with a standard error and a normal-approximation CI.

    Deliberately the simplest possible estimator: under randomisation this is
    unbiased for the ATE, and the point of showing it is that the hard machinery
    elsewhere in this package is not needed for, and does not improve on, the
    question a trial was designed to answer.
    """
    treated = trial.outcome[trial.treatment == 1]
    control = trial.outcome[trial.treatment == 0]
    effect = float(treated.mean() - control.mean())
    se = float(np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control)))
    return effect, se, 1.96 * se


def stratified_effect(trial: TrialData, values: np.ndarray) -> tuple[float, float, float]:
    """Treatment effect above and below the median of ``values``, and the gap.

    Valid because treatment is randomised and the stratifying variable is
    measured before randomisation, so within each stratum the comparison is
    still a randomised one.  The gap is the effect-modification signal a
    companion diagnostic would be built on.
    """
    high = values >= np.median(values)
    out = []
    for mask in (high, ~high):
        treated = trial.outcome[mask & (trial.treatment == 1)]
        control = trial.outcome[mask & (trial.treatment == 0)]
        out.append(float(treated.mean() - control.mean()))
    return out[0], out[1], float(out[0] - out[1])
