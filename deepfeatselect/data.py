"""Loading, encoding and splitting for the Cleveland heart-disease data.

The original implementation L2-normalised every column over the *entire*
dataset and fed the raw integer codes of the nominal attributes straight into
the network.  Both are fixed here:

* scaling statistics are fitted on the training split only, so no validation or
  test information leaks into the transform;
* nominal attributes (``cp``, ``restecg``, ``slope``, ``thal``) are one-hot
  encoded instead of being treated as ordinal magnitudes.

One-hot encoding means a single *feature* can span several *columns*, so the
loader also returns a ``groups`` vector mapping each column back to the feature
it came from.  :class:`~deepfeatselect.model.FeatureGate` uses it to keep one
gate per feature rather than one per column.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Column order of processed.cleveland.data, and how each attribute should be
# treated.  "nominal" attributes get one-hot encoded; everything else is used
# as a single numeric column.  ``ca`` is a count of major vessels and ``slope``
# has a natural ordering, but slope has only three levels so one-hot costs
# little and avoids asserting an equal spacing we cannot justify.
CLEVELAND_COLUMNS = (
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
)

NOMINAL_FEATURES = frozenset({"cp", "restecg", "slope", "thal"})

TARGET_COLUMN = "num"


@dataclass(frozen=True)
class Dataset:
    """A fully prepared train/validation/test split.

    ``x_*`` are encoded and scaled design matrices with ``n_columns`` columns.
    ``feature_names`` has ``n_features`` entries, one per *original* attribute,
    and ``groups[j]`` gives the index into ``feature_names`` of the feature that
    produced column ``j``.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    groups: np.ndarray
    n_classes: int

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_columns(self) -> int:
        return self.x_train.shape[1]


def load_raw(csv_path: str | Path, feature_names: list[str] | None = None) -> pd.DataFrame:
    """Read the raw CSV and drop rows with missing values.

    The UCI file marks missing entries with ``?`` and only six rows are
    affected, so dropping is cheaper than imputing and matches the original
    behaviour -- but here it is done vectorised rather than by iterating rows,
    and the count is reported so the loss is visible.
    """
    names = list(feature_names or CLEVELAND_COLUMNS) + [TARGET_COLUMN]
    df = pd.read_csv(csv_path, header=None, names=names, na_values="?")

    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"dropped {n_dropped} of {n_before} rows containing missing values")

    return df.astype(float)


def load_feature_names(attribute_file: str | Path) -> list[str]:
    """Read attribute names from a single-line CSV, as used by ``Data/attributes``."""
    return [str(name).strip() for name in pd.read_csv(attribute_file, header=None).values[0]]


def _encode(
    df: pd.DataFrame, feature_names: list[str]
) -> tuple[np.ndarray, np.ndarray, OneHotEncoder | None, list[str]]:
    """One-hot the nominal attributes and record the column-to-feature mapping."""
    nominal = [f for f in feature_names if f in NOMINAL_FEATURES]
    numeric = [f for f in feature_names if f not in NOMINAL_FEATURES]

    # Columns come out numeric-block-first, so group indices must address this
    # order rather than the input order -- otherwise gates get reported against
    # the wrong feature names.
    ordered = numeric + nominal

    blocks: list[np.ndarray] = []
    groups: list[int] = []

    if numeric:
        blocks.append(df[numeric].to_numpy(dtype=np.float64))
        groups.extend(ordered.index(f) for f in numeric)

    encoder: OneHotEncoder | None = None
    if nominal:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        blocks.append(encoder.fit_transform(df[nominal].to_numpy()))
        for feature, categories in zip(nominal, encoder.categories_):
            groups.extend([ordered.index(feature)] * len(categories))

    return np.hstack(blocks), np.asarray(groups, dtype=np.int32), encoder, ordered


def _group_scale(groups: np.ndarray) -> np.ndarray:
    """Per-column factor equalising the total variance contributed by each feature.

    After standardisation every column has unit variance, so a feature spread
    over ``k`` one-hot columns would contribute ``k`` times the variance of a
    single continuous feature and pick up a correspondingly inflated gate.
    Dividing each of its columns by ``sqrt(k)`` puts every *feature* on equal
    footing -- the same normalisation group-lasso applies to its blocks.
    """
    sizes = np.bincount(groups)
    return 1.0 / np.sqrt(sizes[groups])


def prepare(
    csv_path: str | Path,
    feature_names: list[str] | None = None,
    task: str = "binary",
    val_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 0,
) -> Dataset:
    """Load, encode, split and scale the dataset.

    Args:
        task: ``"binary"`` collapses the 0-4 severity target to absent/present.
            ``"multiclass"`` keeps all five levels, reproducing the original
            setup -- but note the rarest class has only 13 examples, so those
            importances carry very little signal.
    """
    if task not in {"binary", "multiclass"}:
        raise ValueError(f"task must be 'binary' or 'multiclass', got {task!r}")

    names = list(feature_names or CLEVELAND_COLUMNS)
    df = load_raw(csv_path, names)

    y = df[TARGET_COLUMN].to_numpy(dtype=np.int64)
    if task == "binary":
        y = (y > 0).astype(np.int64)
    n_classes = int(y.max()) + 1

    x, groups, _, ordered_names = _encode(df, names)

    # Two stratified splits: hold out the test set, then carve validation out of
    # what remains. Stratifying keeps the rare severity levels represented in
    # every split, which a contiguous `validation_split=0.5` could not do.
    x_fit, x_test, y_fit, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=seed
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_fit,
        y_fit,
        test_size=val_size / (1.0 - test_size),
        stratify=y_fit,
        random_state=seed,
    )

    # Fit the scaler on the training split alone, then apply it everywhere.
    scaler = StandardScaler().fit(x_train)
    column_scale = _group_scale(groups)
    transform = lambda a: scaler.transform(a) * column_scale  # noqa: E731

    return Dataset(
        x_train=transform(x_train),
        y_train=y_train,
        x_val=transform(x_val),
        y_val=y_val,
        x_test=transform(x_test),
        y_test=y_test,
        feature_names=ordered_names,
        groups=groups,
        n_classes=n_classes,
    )
