"""Training a single gated model and scoring it on held-out data.

The original pipeline reported feature weights with no accompanying measure of
whether the model had learned anything -- a network that failed to fit still
emitted a full set of importances.  :func:`train_one` always returns test-set
metrics alongside the gates so a useless run is visible rather than silently
averaged into the ranking.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import keras
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from .data import Dataset
from .model import FeatureGate, HierarchyProjection, build_model


@dataclass
class TrainConfig:
    """Hyperparameters for one training run."""

    task: str = "binary"
    # Chosen from scripts/l1_sweep.py: the best held-out AUC on this dataset and
    # the only setting that selected against any feature. See the README on why
    # the sparsity path here is not monotone.
    l1_gate: float = 1.0
    l2_dense: float = 1e-3
    hidden_units: int = 128
    n_hidden_layers: int = 3
    dropout: float = 0.5
    noise: float = 0.005
    learning_rate: float = 3e-3
    loss: str = "ce"
    epochs: int = 2000
    # A smaller batch buys more update steps per epoch, which matters because
    # the gates shrink per *step*: at batch 64 there are only three an epoch.
    batch_size: int = 32
    patience: int = 40
    class_weight: bool = True
    proximal: bool = True
    hierarchy: bool = True
    # Only binds when comparable to the first layer's weight scale; Glorot init
    # here gives max|W| around 0.2, so M=10 would never engage.
    hierarchy_m: float = 1.0


@dataclass
class RunResult:
    """Gates and held-out scores from a single model."""

    gates: np.ndarray
    metrics: dict[str, float] = field(default_factory=dict)
    epochs_run: int = 0
    seed: int = 0

    def as_row(self, feature_names: list[str]) -> dict[str, float]:
        row: dict[str, float] = {"seed": self.seed, "epochs_run": self.epochs_run}
        row.update(self.metrics)
        row.update(dict(zip(feature_names, self.gates)))
        return row


def configure_devices(memory_growth: bool = True) -> list[str]:
    """Enable per-process GPU memory growth and report the visible devices.

    Without this a single TensorFlow process claims the whole card up front,
    which stops several models from training concurrently on one GPU.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if memory_growth:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                # Already initialised; the setting is fixed for this process.
                pass
    return [gpu.name for gpu in gpus] or ["CPU"]


def _targets(y: np.ndarray, n_classes: int, task: str) -> np.ndarray:
    if task == "binary":
        return y.astype("float32").reshape(-1, 1)
    return keras.utils.to_categorical(y, num_classes=n_classes)


def _score(y_true: np.ndarray, probs: np.ndarray, task: str) -> dict[str, float]:
    if task == "binary":
        p = probs.reshape(-1)
        y_pred = (p >= 0.5).astype(int)
        scores = {
            "test_auc": float(roc_auc_score(y_true, p)),
            "test_f1": float(f1_score(y_true, y_pred)),
        }
    else:
        y_pred = probs.argmax(axis=1)
        scores = {"test_f1": float(f1_score(y_true, y_pred, average="macro"))}

    scores["test_balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred))
    scores["test_acc"] = float((y_pred == y_true).mean())
    return scores


def train_one(data: Dataset, config: TrainConfig, seed: int = 0, verbose: int = 0) -> RunResult:
    """Train one model and return its gates and test-set scores."""
    keras.utils.set_random_seed(seed)

    model = build_model(
        n_columns=data.n_columns,
        groups=data.groups,
        n_classes=data.n_classes,
        task=config.task,
        l1_gate=config.l1_gate,
        l2_dense=config.l2_dense,
        hidden_units=config.hidden_units,
        n_hidden_layers=config.n_hidden_layers,
        dropout=config.dropout,
        noise=config.noise,
        learning_rate=config.learning_rate,
        loss=config.loss,
        proximal=config.proximal,
    )

    weights = None
    if config.class_weight:
        classes = np.unique(data.y_train)
        balanced = compute_class_weight("balanced", classes=classes, y=data.y_train)
        weights = dict(zip(classes.tolist(), balanced.tolist()))

    # Stock EarlyStopping replaces the vendored copy of TensorFlow's callback
    # that the original carried purely to bolt on model saving; restoring the
    # best weights here gives the same result without the duplicated code.
    stopper = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.patience,
        mode="min",
        restore_best_weights=True,
        verbose=verbose,
    )

    gate_layer: FeatureGate = model.get_layer("feature_gate")
    callbacks = [stopper]
    if config.hierarchy:
        callbacks.append(HierarchyProjection(gate_layer, m=config.hierarchy_m))

    history = model.fit(
        data.x_train,
        _targets(data.y_train, data.n_classes, config.task),
        validation_data=(data.x_val, _targets(data.y_val, data.n_classes, config.task)),
        epochs=config.epochs,
        batch_size=config.batch_size,
        shuffle=True,
        class_weight=weights,
        callbacks=callbacks,
        verbose=verbose,
    )

    probs = model.predict(data.x_test, verbose=0)

    return RunResult(
        gates=gate_layer.gate_values(),
        metrics=_score(data.y_test, probs, config.task),
        epochs_run=len(history.history["loss"]),
        seed=seed,
    )


def config_from_namespace(args) -> TrainConfig:
    """Build a :class:`TrainConfig` from parsed CLI arguments."""
    fields = set(asdict(TrainConfig()))
    return TrainConfig(**{k: v for k, v in vars(args).items() if k in fields})
