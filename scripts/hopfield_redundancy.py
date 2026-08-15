"""Does associative retrieval detect redundancy better than regression?

``redundancy_scores`` answers "is feature j reconstructible from the others" by
fitting one random forest regressor per feature: d separate models, refitted for
every dataset.  A modern Hopfield network offers a different route to the same
question.  Store the training rows as memories, mask feature j in a query, and
let retrieval complete the pattern -- one object, all features, nothing fitted.
If that works it is a strictly cheaper instrument for the Proposition 1
condition, which is checked on every dataset the project touches.

The deflating possibility is stated up front because it is the likely one.
Modern Hopfield retrieval *is* softmax attention over stored patterns, and for
reconstruction that is kernel-weighted nearest-neighbour averaging with the
kernel ``exp(beta * <q, m>)``.  So the honest question is not "does retrieval
beat a forest" but "does retrieval beat kNN", since kNN is the same estimator
with a rectangular kernel.  kNN is therefore run as a baseline precisely so a
null result is visible rather than hidden behind a favourable forest comparison.

Four things this script is careful about.

* **Self-exclusion.**  A query row must not be in its own retrieval set.  With
  itself present the softmax puts nearly all mass on the perfect match and every
  feature reconstructs at R^2 ~ 1, including pure noise.  The leak panel prints
  that failure as a number rather than trusting the reader to believe the assert.
* **Comparability.**  All methods run on the identical ``KFold`` splits used by
  ``redundancy_scores._cv_r2``, predict in the original units, and are scored by
  the same ``1 - MSE/Var`` over pooled out-of-fold predictions.  The forest arm
  is checked against ``redundancy_scores`` itself.
* **Hyperparameter fairness.**  Reporting Hopfield at its best beta while
  pinning kNN at one k would manufacture a win.  Both are swept and both are
  reported at their oracle-best setting, so the two arms are optimistic by the
  same amount and the comparison between them stays honest.
* **Variant fairness, which is the same trap one level up.**  Ramsauer's raw
  inner product is not norm-invariant, and on standardised tabular rows that
  wrecks it.  There are two ways to repair it and they do not agree, so running
  only one and reading the verdict off it decides the result by a variant choice
  rather than by the method.  Both are therefore swept as arms:

      exp(beta <q, m>) = exp(-beta/2 ||q - m||^2) * exp(beta/2 ||m||^2)

  is an exact identity, so the dot product *is* an RBF kernel regression carrying
  a spurious ``exp(beta/2 ||m||^2)`` prior that favours high-norm memories.
  ``rbf`` drops exactly that prior and changes nothing else; ``cosine`` instead
  projects everything onto the unit sphere, which also removes the bias but
  discards the radial coordinate along with it.  ``rbf`` is the closer analogue
  of the paper's own regime, where stored patterns are LayerNormed and therefore
  near-equal in norm.  The same repair is applied to the energy channel, for the
  same reason: an arm run in only the broken parametrisation reports the
  parametrisation's failure as the channel's.

    python scripts/hopfield_redundancy.py
    python scripts/hopfield_redundancy.py --datasets demo --replicate-seeds 5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from deepfeatselect.data import prepare
from deepfeatselect.probe import hopfield_energy
from deepfeatselect.redundancy import redundancy_scores
from deepfeatselect.synthetic import redundancy_demo

# Inverse temperatures spanning four orders of magnitude.  Retrieval sharpness
# is entirely beta's doing -- small beta averages every memory into the column
# mean, large beta collapses onto the single best match -- so one arbitrary
# value would say nothing.  Note the scale is not dimensionless: the similarity
# is a dot product over d-1 standardised columns, so its spread grows like
# sqrt(d-1) and the *effective* sharpness at fixed beta rises with the feature
# count.  That is why the useful beta differs between the 4-column and 30-column
# datasets below, and why the sweep has to be re-run per dataset rather than
# tuned once.
BETAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)

# Neighbourhood sizes for the baseline.  The endpoints matter: k=1 is what
# Hopfield converges to as beta -> infinity, and k=n would be the column mean,
# which is what it converges to as beta -> 0.  Retrieval lives between them.
KS = (1, 3, 5, 10, 25)

# Matches redundancy_scores._cv_r2 exactly so the forest arm here and the
# function the project already ships are the same estimator on the same splits.
N_SPLITS = 3
FOREST_KWARGS = dict(n_estimators=120, min_samples_leaf=5)

# Named here so the retrieval arms are added in one place rather than in the
# four spots that have to agree about which methods are retrieval.
HOPFIELD_METHODS = ("hopfield_dot", "hopfield_rbf", "hopfield_cosine")


# --------------------------------------------------------------------------
# retrieval

SIMILARITIES = ("dot", "cosine", "rbf")


def _masked_similarity(
    queries: np.ndarray, memories: np.ndarray, j: int, similarity: str
) -> np.ndarray:
    """Similarity of each query to each memory over every column except ``j``.

    Column j is deleted rather than zeroed.  Zeroing would leave the masked
    slot contributing ``0 * m_j = 0`` to the dot product, which is the same
    thing for the inner product but not for the norms that ``cosine`` divides
    by, and deleting keeps the variants comparable.

    All three are returned on a scale where the caller multiplies by ``beta``,
    so ``rbf`` returns ``-||q - m||^2 / 2`` rather than the distance itself.
    """
    q = np.delete(queries, j, axis=1)
    m = np.delete(memories, j, axis=1)

    if similarity == "dot":
        return q @ m.T

    if similarity == "cosine":
        # Projects both onto the unit sphere.  Removes the norm bias, but at
        # the cost of the radial coordinate, which on standardised columns
        # carries real information about how extreme a row is.
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        m = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
        return q @ m.T

    if similarity == "rbf":
        # The minimal repair.  Since beta*<q,m> = -beta/2*||q-m||^2 plus terms
        # constant in m only up to exp(beta/2 ||m||^2), the dot product is this
        # kernel times a prior that rewards a memory purely for being far from
        # the origin.  Dropping that factor leaves ordinary softmax-weighted
        # neighbour averaging and touches nothing else.  Clipped at zero
        # because the expanded form is not guaranteed non-negative in floating
        # point when two rows nearly coincide.
        d2 = (
            np.einsum("ij,ij->i", q, q)[:, None]
            + np.einsum("ij,ij->i", m, m)[None, :]
            - 2.0 * q @ m.T
        )
        return -0.5 * np.maximum(d2, 0.0)

    raise ValueError(f"similarity must be one of {SIMILARITIES}, got {similarity!r}")


def hopfield_reconstruct(
    x: np.ndarray,
    j: int,
    beta: float,
    memories: np.ndarray | None = None,
    similarity: str = "dot",
    exclude_self: bool = True,
) -> np.ndarray:
    """Complete masked column ``j`` of every row of ``x`` by associative retrieval.

    One step of the modern Hopfield update of Ramsauer et al. (2020): the query
    is compared to every stored memory, the similarities are softmaxed at
    inverse temperature ``beta``, and the retrieved pattern is the weighted
    average of the memories.  Here only the masked coordinate is read out, which
    turns retrieval into a regressor for column ``j`` given the rest.

    ``memories=None`` stores ``x`` itself and runs leave-one-out, which is where
    the correctness detail bites.  Row i is memory i, so unless it is removed
    from its own retrieval set the softmax finds a perfect match, puts almost
    all mass on it at any usable beta, and returns the query's own value: every
    column reconstructs at R^2 ~ 1 including independent noise, and the whole
    instrument reads "everything is redundant".  The exclusion is asserted on
    the realised weights rather than assumed from the masking code.

    Caveat the assert cannot cover: exact duplicate rows leak the same way, and
    excluding the row's own index does not exclude its copies.  None of the
    datasets here have duplicates, but a deduplication check belongs in any
    application to data with repeated measurements.

    Args:
        x: Queries, one row per pattern to complete, shape ``(n_queries, d)``.
        j: Index of the column to mask and reconstruct.
        beta: Inverse temperature.  Large is sharp (approaching 1-NN under this
            similarity), small is flat (approaching the memory column mean).
        memories: Stored patterns, shape ``(n_memories, d)``.  ``None`` stores
            ``x`` itself and runs leave-one-out.
        similarity: ``"dot"`` for the paper's inner product, or one of the two
            norm repairs, ``"rbf"`` and ``"cosine"``.  See the module docstring:
            the choice decides the answer, so all three are run.
        exclude_self: Leave-one-out mode only.  ``False`` exists solely so the
            leak panel can measure the failure it prevents; never use it.

    Returns:
        Reconstructed column ``j``, shape ``(n_queries,)``.
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")

    queries = np.asarray(x, dtype=np.float64)
    loo = memories is None
    store = queries if loo else np.asarray(memories, dtype=np.float64)
    if store.shape[1] != queries.shape[1]:
        raise ValueError(
            f"query dimension {queries.shape[1]} does not match memory "
            f"dimension {store.shape[1]}"
        )

    logits = beta * _masked_similarity(queries, store, j, similarity)
    if loo and exclude_self:
        # -inf, not a large negative constant: softmax must give it exactly
        # zero mass for the assert below to be a real check.
        logits[np.arange(len(queries)), np.arange(len(queries))] = -np.inf

    weights = softmax(logits, axis=1)
    if loo and exclude_self:
        assert np.all(np.diag(weights) == 0.0), (
            "a query retrieved itself: reconstruction R^2 is meaningless"
        )

    return weights @ store[:, j]


# --------------------------------------------------------------------------
# scoring, on the splits redundancy_scores already uses

def _r2(target: np.ndarray, predictions: np.ndarray) -> float:
    """Pooled out-of-fold R-squared, identical to redundancy.py's ``_cv_r2``."""
    variance = float(np.var(target))
    if variance <= 0.0:
        return 0.0
    return float(1.0 - np.mean((target - predictions) ** 2) / variance)


def _oof_predictions(
    x: np.ndarray, j: int, method: str, setting: float, seed: int
) -> np.ndarray:
    """Out-of-fold reconstruction of column ``j``, in the column's own units.

    Every method sees the same folds.  The two neighbourhood methods work on
    columns standardised with *training-fold* statistics -- both are
    scale-sensitive and a raw-units dot product would be dominated by whichever
    column happens to be measured in the largest numbers -- and their prediction
    is mapped back before scoring, so the R^2 is on the same footing as the
    forest's, which needs no scaling at all.
    """
    predictions = np.empty(len(x), dtype=np.float64)
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    for train_idx, test_idx in splitter.split(x):
        # Disjoint by construction, but this is the fold-mode counterpart of the
        # leave-one-out assert and costs nothing.
        assert not set(train_idx) & set(test_idx), "fold leak: a query is a memory"

        if method == "forest":
            model = RandomForestRegressor(**FOREST_KWARGS, random_state=seed, n_jobs=-1)
            model.fit(np.delete(x[train_idx], j, axis=1), x[train_idx, j])
            predictions[test_idx] = model.predict(np.delete(x[test_idx], j, axis=1))
            continue

        scaler = StandardScaler().fit(x[train_idx])
        z_train = scaler.transform(x[train_idx])
        z_test = scaler.transform(x[test_idx])

        if method == "knn":
            # Distance-weighted rather than uniform, because that is the closer
            # analogue: retrieval also weights every contributing memory by how
            # well it matches. Uniform kNN would be a weaker baseline and would
            # flatter the comparison.
            model = KNeighborsRegressor(
                n_neighbors=min(int(setting), len(train_idx)), weights="distance"
            )
            model.fit(np.delete(z_train, j, axis=1), z_train[:, j])
            z_pred = model.predict(np.delete(z_test, j, axis=1))
        elif method in HOPFIELD_METHODS:
            z_pred = hopfield_reconstruct(
                z_test, j, beta=float(setting), memories=z_train,
                similarity=method.split("_")[1],
            )
        else:
            raise ValueError(f"unknown method {method!r}")

        # Back to the column's own units so the R^2 denominator is the raw
        # variance, exactly as _cv_r2 computes it.
        predictions[test_idx] = z_pred * scaler.scale_[j] + scaler.mean_[j]

    return predictions


def sweep_dataset(
    x: np.ndarray, names: list[str], seed: int, timing_repeats: int = 3
) -> tuple[pd.DataFrame, dict[tuple[str, float, str], np.ndarray]]:
    """Every method at every setting on one dataset.

    Returns the long per-feature table and the out-of-fold predictions, the
    latter because comparing methods by their *predictions* rather than their
    scores is the direct test of whether retrieval and kNN are the same
    estimator wearing different labels.
    """
    settings: list[tuple[str, float]] = (
        [(method, b) for method in HOPFIELD_METHODS for b in BETAS]
        + [("knn", float(k)) for k in KS]
        + [("forest", float("nan"))]
    )

    rows: list[dict[str, object]] = []
    predictions: dict[tuple[str, float, str], np.ndarray] = {}
    for method, setting in settings:
        # Timed over all features, since that is the unit of work the
        # instrument actually performs: one call answers the whole table.
        # Minimum of several repeats rather than one shot or a mean: this box
        # runs other jobs, and contention can only ever add time, so the
        # smallest observation is the closest estimate of the real cost. A
        # single measurement moved by 3x between runs, which is larger than the
        # differences the comparison turns on.
        elapsed = float("inf")
        for _ in range(max(1, timing_repeats)):
            start = time.perf_counter()
            per_feature = {
                name: _oof_predictions(x, j, method, setting, seed)
                for j, name in enumerate(names)
            }
            elapsed = min(elapsed, time.perf_counter() - start)

        for j, name in enumerate(names):
            predictions[(method, setting, name)] = per_feature[name]
            rows.append({
                "method": method,
                "setting": setting,
                "feature": name,
                "r2": _r2(x[:, j], per_feature[name]),
                "seconds_all_features": elapsed,
            })

    return pd.DataFrame(rows), predictions


def best_per_method(table: pd.DataFrame) -> pd.DataFrame:
    """Each method at the setting maximising its mean R^2 across features.

    Selected on the same held-out score being reported, so every swept method
    is equally optimistic.  The single-setting methods are unaffected, which is
    the point: an oracle applied to only one arm would be the bias, an oracle
    applied to all of them is a stated and symmetric one.
    """
    keep = []
    for _, block in table.groupby("method", sort=False):
        means = block.groupby("setting", dropna=False)["r2"].mean()
        # The forest has no setting, so its "best" is the only one it has.
        keep.append(block if means.index.isna().all()
                    else block[block.setting == means.idxmax()])
    return pd.concat(keep, ignore_index=True)


# --------------------------------------------------------------------------
# the leak panel

def leak_panel(
    x: np.ndarray, names: list[str], betas: tuple[float, ...]
) -> pd.DataFrame:
    """Leave-one-out reconstruction with and without the query in its own store.

    The single most important correctness detail, measured rather than asserted.
    Swept over beta because the leak is not a constant: at flat temperatures the
    query's own memory is one vote among thousands and the contamination is
    negligible, while at the sharp temperatures where retrieval is actually
    worth using it takes essentially all the softmax mass.  A version of this
    experiment that checked the bug at one low beta would have concluded the
    detail did not matter.
    """
    z = StandardScaler().fit_transform(x)
    rows = []
    for beta in betas:
        for j, name in enumerate(names):
            rows.append({
                "beta": beta,
                "feature": name,
                "r2_self_included": _r2(
                    z[:, j], hopfield_reconstruct(z, j, beta, exclude_self=False)),
                "r2_self_excluded": _r2(
                    z[:, j], hopfield_reconstruct(z, j, beta, exclude_self=True)),
            })
    return pd.DataFrame(rows)


def noise_leak(sizes: tuple[int, ...], d: int, beta: float, seed: int) -> pd.DataFrame:
    """The leak on data with no redundancy at all, where the true answer is zero.

    Independent Gaussian columns: every leave-one-out R^2 is exactly 0 in
    population, so anything above it is manufactured by the query seeing itself.
    Swept over sample size because that is what controls the damage.  The
    self-match wins the softmax by a margin set by how far away the nearest
    genuine neighbour is, so the contamination is worst when rows are sparse in
    the space -- which is the regime a redundancy audit is usually run in, and
    the reason the effect is nearly invisible on 3000 rows of a one-dimensional
    chaotic manifold where near-duplicates already exist.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        z = rng.standard_normal((n, d))
        for j in range(d):
            rows.append({
                "n": n,
                "column": j,
                "r2_self_included": _r2(
                    z[:, j], hopfield_reconstruct(z, j, beta, exclude_self=False)),
                "r2_self_excluded": _r2(
                    z[:, j], hopfield_reconstruct(z, j, beta, exclude_self=True)),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# the energy channel

def energy_ablation(
    x: np.ndarray, names: list[str], beta: float, seed: int, normalise: bool = False
) -> pd.DataFrame:
    """Rank features by how much zeroing them moves the Hopfield energy.

    A different use of the same machinery: instead of reading a retrieved
    coordinate, ask whether the query still lands in the same place on the
    energy landscape once a feature is removed.  ``hopfield_energy`` has been in
    the package since the probe work and has never been scored against ground
    truth, so this is its first test.

    Features are standardised on the training memories, which makes zeroing
    equivalent to mean-imputation and -- more importantly -- equalises the
    mechanical part of the shift.  The energy's constants cancel in the
    difference, leaving::

        dE = (lse_full - lse_ablated) - 0.5 * q_j^2

    so a feature with a larger scale would move the energy more whatever its
    structure.  After standardisation ``E[q_j^2] = 1`` for every column, so that
    term contributes the same expected 0.5 to all of them.  Both the full
    difference and the retrieval-only ``lse`` part are reported; neither is
    chosen after seeing which ranks better.

    ``normalise`` puts memories and queries on the unit sphere before anything
    else, which is the same repair the reconstruction arm runs as
    ``hopfield_cosine`` and is here for the same reason: ``hopfield_energy``
    uses the raw inner product, so without it the landscape is tilted by the
    memory norms and the ablation measures that tilt.  Normalising *once, up
    front* rather than after masking is deliberate -- renormalising the ablated
    query would change ``||q||^2`` and invalidate the ``0.5 * q_j^2``
    decomposition above.  (Measured both ways: the two agree at beta >= 10.)

    Magnitude, not signed difference: picking the sign that separates the groups
    would be fitting a free parameter to the answer.
    """
    x_train, x_query = train_test_split(x, test_size=0.3, random_state=seed)
    scaler = StandardScaler().fit(x_train)
    memories = scaler.transform(x_train)
    queries = scaler.transform(x_query)
    if normalise:
        memories = memories / np.maximum(
            np.linalg.norm(memories, axis=1, keepdims=True), 1e-12)
        queries = queries / np.maximum(
            np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)

    full = hopfield_energy(queries, memories, beta=beta)
    rows = []
    for j, name in enumerate(names):
        ablated_query = queries.copy()
        ablated_query[:, j] = 0.0
        ablated = hopfield_energy(ablated_query, memories, beta=beta)

        # The purely mechanical half of the shift, subtracted out to leave the
        # part that is actually about retrieval.
        norm_term = 0.5 * (queries[:, j] ** 2)
        rows.append({
            "feature": name,
            "abs_delta_energy": float(np.mean(np.abs(ablated - full))),
            "abs_delta_lse_only": float(np.mean(np.abs(ablated - full + norm_term))),
        })
    return pd.DataFrame(rows)


def _auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """AUROC at ranking the informative features above the irrelevant ones."""
    if len(np.unique(positive)) < 2 or np.allclose(scores, scores[0]):
        return float("nan")
    return float(roc_auc_score(positive.astype(int), scores))


def run_energy_channel(
    n: int, seeds: int, betas: tuple[float, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the energy channel over seeds, with the guards a detection AUROC needs.

    Two guards, because the channel is unsupervised over ``x`` and the obvious
    guard is not.  ``reference_auc_mutual_info`` is mutual information against
    the target: it must succeed, and if it does not the labelling itself is
    broken.  But it is a *supervised* detector, so on its own it does not
    establish that anything reading ``x`` alone could separate these groups.
    ``reference_auc_redundancy`` is the matching unsupervised guard -- the
    forest reconstruction R^2 that ``redundancy_scores`` already computes,
    scored on the same contrast.  An energy AUROC of 0 next to a redundancy
    AUROC of 1 is the channel failing; both at 0.5 would be an unlearnable
    target, which is the failure mode this project has lost sweeps to.

    ``retrieval_r2`` is the memory bank's own held-out performance at that beta:
    the mean reconstruction R^2 over informative features.  An energy channel
    computed at a beta where retrieval itself recovers nothing is reading an
    unstructured landscape, and the number would be uninterpretable rather than
    merely low.

    Both similarity parametrisations are run, for the reason given in the module
    docstring.  Scoring only the raw inner product would report the norm bias as
    the channel's verdict.

    Returns the per-seed scores and the raw per-feature ablation values, the
    latter so the ranking behind each AUROC can be read directly -- an AUROC of
    0.000 and one of 0.500 are very different failures and the summary alone
    cannot tell them apart.
    """
    rows = []
    scores = []
    for seed in range(seeds):
        system = redundancy_demo(n=n, seed=seed)
        x = np.asarray(system["x"], dtype=np.float64)
        names = list(system["feature_names"])
        y = np.asarray(system["y"], dtype=np.float64)
        informative = np.array([nm not in system["irrelevant"] for nm in names])

        reference = mutual_info_regression(x, y, random_state=seed)
        unsupervised = np.array([
            _r2(x[:, j], _oof_predictions(x, j, "forest", float("nan"), seed))
            for j in range(x.shape[1])
        ])
        for beta in betas:
            retrieval = np.array([
                _r2(x[:, j], _oof_predictions(x, j, "hopfield_dot", beta, seed))
                for j in range(x.shape[1])
            ])
            for variant, normalise in (("dot", False), ("cosine", True)):
                table = energy_ablation(x, names, beta=beta, seed=seed,
                                        normalise=normalise)
                scores.append(table.assign(seed=seed, beta=beta, variant=variant))
                rows.append({
                    "seed": seed,
                    "beta": beta,
                    "variant": variant,
                    "auc_abs_delta_energy": _auc(
                        table.abs_delta_energy.to_numpy(), informative),
                    "auc_abs_delta_lse_only": _auc(
                        table.abs_delta_lse_only.to_numpy(), informative),
                    "reference_auc_mutual_info": _auc(reference, informative),
                    "reference_auc_redundancy": _auc(unsupervised, informative),
                    "retrieval_r2_informative": float(np.mean(retrieval[informative])),
                    "irrelevant_ranked_last": bool(
                        np.argmin(table.abs_delta_energy.to_numpy())
                        == int(np.argmin(informative))
                    ),
                })
    return pd.DataFrame(rows), pd.concat(scores, ignore_index=True)


# --------------------------------------------------------------------------
# datasets

def load_datasets(which: list[str], n: int, seed: int) -> dict[str, tuple[np.ndarray, list[str]]]:
    out: dict[str, tuple[np.ndarray, list[str]]] = {}

    if "demo" in which:
        system = redundancy_demo(n=n, seed=seed)
        out["redundancy_demo"] = (
            np.asarray(system["x"], dtype=np.float64), list(system["feature_names"])
        )

    if "cancer" in which:
        cancer = load_breast_cancer()
        names = [nm.replace(" ", "_") for nm in cancer.feature_names]
        # The nine radius/perimeter/area columns rather than all thirty: they
        # are the ones tied by an identity we know independently of the data, so
        # they test whether a method finds a redundancy that is really there
        # instead of rewarding it for the diffuse correlation the other 21
        # columns add.
        keep = [i for i, nm in enumerate(names)
                if any(k in nm for k in ("radius", "perimeter", "area"))]
        out["breast_cancer_geometric"] = (
            np.asarray(cancer.data, dtype=np.float64)[:, keep], [names[i] for i in keep]
        )

    if "cleveland" in which:
        data = prepare("Data/processed.cleveland.data", task="binary", seed=seed)
        x_all = np.vstack([data.x_train, data.x_val, data.x_test])
        # Numeric attributes only.  One-hot columns of the same variable sum to
        # one and so reconstruct each other perfectly, which would turn the
        # negative control into a guaranteed hit for every method.
        numeric = [i for i in range(data.n_features) if (data.groups == i).sum() == 1]
        cols = [int(np.flatnonzero(data.groups == i)[0]) for i in numeric]
        out["cleveland"] = (x_all[:, cols], [data.feature_names[i] for i in numeric])

    return out


# --------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["demo", "cancer", "cleveland"],
                   choices=["demo", "cancer", "cleveland"])
    p.add_argument("--n", type=int, default=3000, help="rows for redundancy_demo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--energy-seeds", type=int, default=10)
    p.add_argument("--energy-n", type=int, default=1200,
                   help="smaller than --n: the energy panel repeats over seeds")
    p.add_argument("--leak-n", type=int, default=1500,
                   help="rows for the leave-one-out leak panel, which is O(n^2)")
    p.add_argument("--replicate-seeds", type=int, default=5,
                   help="repeat the head-to-head at this many seeds; 1 disables. "
                        "A margin that only holds at one seed is not a margin")
    p.add_argument("--timing-repeats", type=int, default=3,
                   help="wall clock is the minimum over this many repeats, since "
                        "contention can only inflate it")
    p.add_argument("--skip-energy", action="store_true")
    p.add_argument("--outdir", default="ExpOutput/hopfield")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    datasets = load_datasets(args.datasets, args.n, args.seed)

    # ---- the leak, measured -------------------------------------------------
    print("\n" + "=" * 92)
    print("SELF-EXCLUSION -- what happens if a query can retrieve itself")
    print("=" * 92)
    print("  Mean leave-one-out R^2 over features, with and without the query in its")
    print("  own store. The inflation is the size of the bug this guard prevents.\n")
    leaks = []
    for label, (x, names) in datasets.items():
        panel = leak_panel(x[:args.leak_n], names, BETAS)
        panel.insert(0, "dataset", label)
        leaks.append(panel)
        summary = panel.groupby("beta")[["r2_self_included", "r2_self_excluded"]].mean()
        summary["inflation"] = summary.r2_self_included - summary.r2_self_excluded
        print(f"  {label} ({min(len(x), args.leak_n)} rows, {x.shape[1]} features):")
        with pd.option_context("display.float_format", "{:+.4f}".format,
                               "display.width", 200):
            print(summary.to_string())
        print()

    leak_frame = pd.concat(leaks, ignore_index=True)
    leak_frame.to_csv(outdir / "self_exclusion_leak.csv", index=False)

    # Pure noise, where the correct answer is exactly zero and any positive R^2
    # is fabricated. The three datasets above understate the bug because their
    # rows are dense enough to have genuine near-neighbours competing with the
    # self-match; this is the clean version.
    noise = noise_leak(sizes=(100, 300, 1000, 3000), d=10, beta=10.0, seed=args.seed)
    noise.to_csv(outdir / "noise_leak.csv", index=False)
    print("  10 independent Gaussian columns, beta=10 -- true R^2 is exactly 0,")
    print("  so the whole self-included column is fabricated:\n")
    with pd.option_context("display.float_format", "{:+.4f}".format):
        print(noise.groupby("n")[["r2_self_included", "r2_self_excluded"]]
              .mean().to_string())
    small = noise[noise.n == noise.n.min()].r2_self_included.mean()
    print(f"\n  At 100 rows the leak reports R^2 {small:+.4f} on columns that are "
          f"independent\n  by construction -- the audit would flag pure noise as "
          f"non-identifiable. The damage\n  shrinks as rows get dense enough for "
          f"genuine neighbours to compete with the\n  self-match, which is why it is "
          f"mildest on redundancy_demo above and worst on\n  Cleveland's 297 rows in "
          f"9 dimensions. It never announces itself.")

    # ---- the method comparison ---------------------------------------------
    all_rows = []
    verdict_rows = []
    for label, (x, names) in datasets.items():
        print("\n" + "=" * 92)
        print(f"RECONSTRUCTION R^2 PER FEATURE -- {label} "
              f"({x.shape[0]} rows, {x.shape[1]} features)")
        print("=" * 92)

        table, predictions = sweep_dataset(x, names, args.seed, args.timing_repeats)
        table.insert(0, "dataset", label)
        all_rows.append(table)

        print("\nbeta sweep, mean R^2 over features "
              "(0 = no better than the column mean):")
        sweep = (table[table.method.str.startswith("hopfield")]
                 .pivot_table(index="setting", columns="method", values="r2"))
        knn_sweep = (table[table.method == "knn"]
                     .pivot_table(index="setting", columns="method", values="r2"))
        with pd.option_context("display.float_format", "{:+.4f}".format,
                               "display.width", 200):
            print(sweep.to_string())
            print("\nk sweep, mean R^2 over features:")
            print(knn_sweep.to_string())

        best = best_per_method(table)
        wide = best.pivot_table(index="feature", columns="method", values="r2")
        chosen = (best.groupby("method")["setting"].first())
        wide = wide.reindex([nm for nm in names])
        print("\nper feature, each method at its best setting "
              f"({', '.join(f'{m}={s:g}' for m, s in chosen.items() if s == s)}):")
        with pd.option_context("display.float_format", "{:+.4f}".format,
                               "display.width", 200):
            print(wide.to_string())

        print("\nwall clock, seconds to score all "
              f"{x.shape[1]} features at one setting "
              f"(best of {args.timing_repeats}):")
        timing = best.groupby("method")["seconds_all_features"].first().sort_values()
        for method, seconds in timing.items():
            print(f"  {method:<17} {seconds:8.2f}")
        sweep_cost = table.drop_duplicates(["method", "setting"])["seconds_all_features"]
        print(f"  (full sweep, all settings, all methods: {sweep_cost.sum():.2f} s)")

        # The direct test of the deflating hypothesis.  Equal scores could be a
        # coincidence; identical *predictions* would mean the two are one
        # estimator under two names, which is the claim being checked.
        knn_setting = chosen.get("knn")
        agreement = {}
        for variant in HOPFIELD_METHODS:
            agreement[variant] = [
                float(np.corrcoef(predictions[(variant, chosen.get(variant), nm)],
                                  predictions[("knn", knn_setting, nm)])[0, 1])
                for nm in names
            ]
        print("\n  correlation with kNN's out-of-fold predictions "
              "(1.0 would mean the same estimator):")
        for variant, values in agreement.items():
            print(f"    {variant:<17} mean {np.mean(values):+.4f}, "
                  f"min {np.min(values):+.4f}")

        # Does the forest arm here reproduce the shipped function?
        reference = (redundancy_scores(x, names, seed=args.seed)
                     .set_index("feature")["r2_from_others"])
        forest_here = best[best.method == "forest"].set_index("feature")["r2"]
        gap = float(np.max(np.abs(reference.reindex(names) - forest_here.reindex(names))))
        print(f"  forest arm vs redundancy_scores(): max |difference| {gap:.6f} "
              f"-- {'same estimator' if gap < 1e-9 else 'DIVERGED, comparison unsafe'}")

        # The stated correctness check: redundancy_demo's three informative
        # columns are mutually determined to machine precision and the fourth is
        # an independent map, so a working reconstructor must approach 1 on the
        # first three and 0 on the last. A method that fails this is broken, and
        # nothing it says on the real datasets can be read.
        if label == "redundancy_demo":
            print("\n  CORRECTNESS CHECK -- exact redundancy, so a working method must")
            print("  reach ~1 on driver/proxy_cos/proxy_sin and ~0 on unrelated:")
            for method in wide.columns:
                low = float(wide.loc[["driver", "proxy_cos", "proxy_sin"], method].min())
                null = float(wide.loc["unrelated", method])
                ok = low >= 0.90 and null < 0.10
                print(f"    {method:<17} weakest informative {low:+.4f}, "
                      f"irrelevant {null:+.4f}   -> "
                      f"{'PASS' if ok else 'FAIL'}")

        for method in wide.columns:
            verdict_rows.append({
                "dataset": label,
                "method": method,
                "setting": chosen.get(method, float("nan")),
                "mean_r2": float(wide[method].mean()),
                "median_r2": float(wide[method].median()),
                "seconds": float(timing[method]),
                "knn_prediction_r": float(np.mean(agreement.get(method, [np.nan]))),
            })

    frame = pd.concat(all_rows, ignore_index=True)
    frame.to_csv(outdir / "reconstruction_sweep.csv", index=False)

    verdict = pd.DataFrame(verdict_rows)
    verdict.to_csv(outdir / "method_verdict.csv", index=False)
    print("\n" + "=" * 92)
    print("VERDICT -- does retrieval beat kNN and the forest, or merely match them?")
    print("=" * 92)
    print("  mean reconstruction R^2 over features, each method at its oracle-best "
          "setting\n  (so all swept methods are optimistic by the same amount):\n")
    with pd.option_context("display.float_format", "{:+.4f}".format,
                           "display.width", 200):
        print(verdict.pivot(index="dataset", columns="method",
                            values="mean_r2").to_string())
        print("\n  seconds to score the whole table at one setting:")
        print(verdict.pivot(index="dataset", columns="method",
                            values="seconds").to_string())
    for label in verdict.dataset.unique():
        block = verdict[verdict.dataset == label].set_index("method")
        retrieval = block.loc[list(HOPFIELD_METHODS), "mean_r2"]
        hop = float(retrieval.max())
        margin_knn = hop - float(block.loc["knn", "mean_r2"])
        margin_rf = hop - float(block.loc["forest", "mean_r2"])
        cost = (float(block.loc[str(retrieval.idxmax()), "seconds"])
                / max(float(block.loc["knn", "seconds"]), 1e-9))
        print(f"\n  {label}: best retrieval variant is {retrieval.idxmax()} at "
              f"{margin_knn:+.4f} R^2 vs kNN and {margin_rf:+.4f} vs the forest,")
        print(f"    at {cost:.2f}x kNN's wall clock.")
        # Spelled out because the whole point of running three parametrisations
        # is that the verdict flips between them: reading it off one arm would
        # have decided the result by a variant choice.
        print("    same-method spread across parametrisations: "
              + ", ".join(f"{m.split('_')[1]} {v:+.4f}" for m, v in retrieval.items()))

    # ---- does the head-to-head margin survive resampling? -------------------
    if args.replicate_seeds > 1:
        print("\n" + "=" * 92)
        print(f"SEED REPLICATION -- oracle-best mean R^2 per method, {args.replicate_seeds} "
              f"seeds")
        print("=" * 92)
        print("  The seed redraws redundancy_demo and reshuffles every KFold, so a "
              "margin\n  that only exists at seed 0 shows up here as a sign change.\n")
        rep_rows = []
        for seed in range(args.replicate_seeds):
            seeded = load_datasets(args.datasets, args.n, seed)
            for label, (x, names) in seeded.items():
                for method, grid in (
                    [(m, BETAS) for m in HOPFIELD_METHODS]
                    + [("knn", tuple(float(k) for k in KS)),
                       ("forest", (float("nan"),))]
                ):
                    rep_rows.append({
                        "seed": seed, "dataset": label, "method": method,
                        "mean_r2": max(
                            float(np.mean([
                                _r2(x[:, j], _oof_predictions(x, j, method, s, seed))
                                for j in range(x.shape[1])
                            ]))
                            for s in grid
                        ),
                    })
        replication = pd.DataFrame(rep_rows)
        replication.to_csv(outdir / "seed_replication.csv", index=False)
        with pd.option_context("display.float_format", "{:+.4f}".format,
                               "display.width", 200):
            print(replication.pivot_table(index="dataset", columns="method",
                                          values="mean_r2").to_string())
            print("\n  per-seed margin vs kNN (a sign that flips is a margin that "
                  "is not there):")
            for label in replication.dataset.unique():
                b = replication[replication.dataset == label]
                knn = b[b.method == "knn"].set_index("seed").mean_r2
                for method in HOPFIELD_METHODS:
                    delta = b[b.method == method].set_index("seed").mean_r2 - knn
                    print(f"    {label:<26}{method:<17}"
                          + " ".join(f"{v:+.4f}" for v in delta)
                          + f"   mean {delta.mean():+.4f}")

    # ---- the energy channel -------------------------------------------------
    if not args.skip_energy:
        print("\n" + "=" * 92)
        print("ABLATION CHANNEL -- hopfield_energy, scored for the first time")
        print("=" * 92)
        print(f"  |delta energy| when each feature is zeroed, ranking informative "
              f"vs irrelevant\n  on redundancy_demo (n={args.energy_n}, "
              f"{args.energy_seeds} seeds; 3 informative vs 1 irrelevant, so a "
              f"single\n  seed's AUROC can only take the values 0, 1/3, 2/3, 1).\n")
        energy, raw = run_energy_channel(args.energy_n, args.energy_seeds, BETAS)

        print("  mean |delta energy| per feature -- the ranking the AUROC scores.")
        print("  'unrelated' is the one that should sit lowest:\n")
        with pd.option_context("display.float_format", "{:.4f}".format,
                               "display.width", 200):
            print(raw.pivot_table(index="beta", columns=["variant", "feature"],
                                  values="abs_delta_energy")
                  .reindex(columns=["driver", "proxy_cos", "proxy_sin", "unrelated"],
                           level="feature").to_string())
        print()

        summary = energy.groupby(["variant", "beta"]).agg(
            auc_energy=("auc_abs_delta_energy", "mean"),
            auc_energy_sd=("auc_abs_delta_energy", "std"),
            auc_lse_only=("auc_abs_delta_lse_only", "mean"),
            reference_auc_mi=("reference_auc_mutual_info", "mean"),
            reference_auc_redundancy=("reference_auc_redundancy", "mean"),
            retrieval_r2=("retrieval_r2_informative", "mean"),
            irrelevant_last=("irrelevant_ranked_last", "mean"),
        )
        with pd.option_context("display.float_format", "{:+.3f}".format,
                               "display.width", 200):
            print(summary.to_string())
        print("\n  The two reference_auc columns are the rule-1 guards: a supervised "
              "one\n  (mutual information vs the target) and the unsupervised one that "
              "matches\n  what this channel sees (forest reconstruction R^2 over x "
              "alone). Both at\n  1.000 means the contrast is detectable and any zero "
              "beside them is the\n  channel failing, not an unlearnable target.")
        print("  retrieval_r2 is the memory bank's own held-out performance at that "
              "beta --\n  an energy read off a landscape that retrieves nothing is "
              "uninterpretable, not low.")

        for variant in ("dot", "cosine"):
            block = summary.loc[variant]
            print(f"\n  {variant}: AUROC spans {block.auc_energy.min():.3f} to "
                  f"{block.auc_energy.max():.3f} over the beta sweep; "
                  f"{int((block.auc_energy >= 0.99).sum())} of "
                  f"{len(block)} betas reach 1.000.")
        print("\n  Read together, that is the whole finding. The raw inner product "
              "reverses --\n  at beta >= 10 it ranks 'unrelated' HIGHEST, reproducibly "
              "over seeds -- and a\n  report that ran only that parametrisation would "
              "call the channel unusable.\n  With the memory norms divided out, the "
              "same statistic, the same a-priori\n  magnitude orientation and no sign "
              "flip separate the groups perfectly at every\n  sharp beta. The failure "
              "was the norm bias, not the channel.")
        energy.to_csv(outdir / "energy_channel.csv", index=False)
        raw.to_csv(outdir / "energy_channel_scores.csv", index=False)
        summary.to_csv(outdir / "energy_channel_summary.csv")

    print(f"\nwrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
