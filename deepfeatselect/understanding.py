"""Did the model understand the structure, or memorise the rows?

Validation loss cannot tell those apart. A network that fits 4200 rows perfectly
and generalises nothing, and one that never fit anything, both report a
validation loss at chance -- and every scaling experiment in this project
recorded only that number.

Two instruments here, neither needing ground truth.

**The memorisation gap.** Training loss against validation loss. A large gap is
memorisation; both at chance is failure to fit. Cheap, standard, and absent from
everything measured so far.

One trap in obtaining it, which we fell into: the training loss reported by
Keras' ``fit`` is *not* comparable to validation loss. It is computed with
dropout ACTIVE and averaged over batches while the weights are still changing,
whereas validation loss is computed with dropout OFF on the epoch's final
weights. Differencing them measures the dropout asymmetry as much as
generalisation, and it *understates* the gap -- the wrong direction when the
question is whether the model memorised. Re-evaluate the training set in
inference mode instead, on the restored weights.

**The shape of the ablation profile.** This is the more interesting one. An
irreducible k-way interaction has a signature memorisation cannot imitate: a
model that genuinely represents it needs every one of the k members, so removing
any one destroys the function, while removing a noise feature does nothing. The
profile of ablation deltas across features is therefore *peaked* -- k large
values, the rest flat. A model doing lookup has no such structure: every column
is part of the key, so removing any one blurs it slightly and the profile is
uniform.

Peakedness is measurable without knowing which features matter, which is what
makes it useful. It answers "did this model understand?" rather than "which
features are important?", and the first question gates whether the second has an
answer worth reading.

**What measurement showed, and the scope it applies in.** On oblique_interaction
at d=20, k=4, six real runs:

    freq   regime       held-out AUC   gini   top4_share   on_target_share
      2    understood      0.848       0.737     0.825          0.825
      4    failed          0.618       0.626     0.667          0.076
      8    failed          0.625       0.710     0.777          0.053

The claim is conditional -- it describes a model that *represents* the
interaction -- so only the first row tests it, and there it holds exactly: the
profile is peaked and 0.825 of the mass sits on true members, detection 1.000.
The other two rows are models that never fitted, which the gap instrument
correctly labelled ``failed``; they say nothing about the claim either way.

What they do establish is a usage constraint. Read *without* the regime gate,
peakedness has a false-positive mode: a model that barely fitted is as
concentrated as one that understood (0.710 against 0.737) while its mass lies
off-target, below the 0.20 that 4 features in 20 would get at random. So the two
instruments are not independent readings to be compared -- the gap gates the
profile, and a profile from a ``failed`` model is not interpretable.

``kurtosis`` may separate the two peaked cases directly, which would make the
gate cheaper. It is not monotone in learning quality: shuffled arms sit at -1.5
to 0.5, the understood arm at 1.07-1.13, the failed arms at 6.9-9.9. An
irreducible k-way interaction should leave *k comparable* peaks, whereas an
unfitted model leaves one runaway spike, and kurtosis is the statistic that
distinguishes those shapes. On n=6 this is a hypothesis, not a result.

**A scope limit on this whole module.** Everything above is measured on i.i.d.
draws. Claims about a cause leaving an imprint on its effect (Takens, 1981;
Sugihara et al., 2012) are theorems about trajectories on an attractor and are
not in evidence here either way -- there is no manifold in this data for an
imprint to live on. For those, see :mod:`deepfeatselect.ccm` and the coupled
dynamical systems in :mod:`deepfeatselect.synthetic`.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def gini(values: np.ndarray) -> float:
    """Concentration of a non-negative vector: 0 uniform, approaching 1 if one
    entry carries everything.

    Applied to ablation deltas, this is how far the model's dependence is
    concentrated on a few features rather than spread across all of them.
    """
    v = np.sort(np.abs(np.asarray(values, dtype=np.float64)))
    n = len(v)
    total = v.sum()
    if n == 0 or total <= 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * (index * v).sum()) / (n * total) - (n + 1.0) / n)


def profile_shape(deltas: np.ndarray) -> dict[str, float]:
    """Peakedness of an ablation profile, by several measures at once.

    They are not independent and that is deliberate: a single summary can be
    fooled by one outlier, and agreement between them is the evidence.

    ``top_k_share`` needs no distributional assumption at all -- it is simply
    the fraction of total dependence carried by the largest few features -- and
    is the one to trust when the others disagree.
    """
    v = np.abs(np.asarray(deltas, dtype=np.float64))
    total = v.sum()
    if total <= 0 or len(v) < 4:
        return {"gini": 0.0, "kurtosis": 0.0, "top4_share": 0.0,
                "participation": float(len(v))}

    ordered = np.sort(v)[::-1]
    # Effective number of features actually carrying dependence: 1 if one
    # feature carries everything, d if all carry equally.
    share = v / total
    participation = float(1.0 / (share**2).sum())

    return {
        "gini": gini(v),
        "kurtosis": float(stats.kurtosis(v, fisher=True, bias=False)),
        "top4_share": float(ordered[:4].sum() / total),
        "participation": participation,
    }


def memorisation_gap(train_loss: float, val_loss: float,
                     chance: float = float(np.log(2))) -> dict[str, float]:
    """Train against validation, and the regime that combination implies.

    ``regime`` is one of:

    * ``understood`` -- both losses beat chance and the gap is modest;
    * ``memorised`` -- training loss well below chance, validation at or above
      it: the rows were fitted and nothing transferred;
    * ``failed`` -- both at chance, so nothing was fitted at all.

    The distinction matters because ablation deltas mean different things in
    each: informative under the first, a property of the stored rows under the
    second, and pure noise under the third.
    """
    gap = val_loss - train_loss
    if train_loss >= 0.9 * chance and val_loss >= 0.9 * chance:
        regime = "failed"
    elif val_loss >= 0.9 * chance:
        regime = "memorised"
    else:
        regime = "understood"
    return {"train_loss": train_loss, "val_loss": val_loss,
            "memorisation_gap": gap, "regime": regime}
