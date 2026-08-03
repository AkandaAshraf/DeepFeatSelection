"""Neural-network feature selection via learnable per-feature gates."""

from .data import Dataset, prepare
from .experiment import report, run_experiment, summarise
from .model import FeatureGate, build_model, soft_f1_loss
from .train import RunResult, TrainConfig, train_one

__version__ = "0.2.0"

__all__ = [
    "Dataset",
    "FeatureGate",
    "RunResult",
    "TrainConfig",
    "build_model",
    "prepare",
    "report",
    "run_experiment",
    "soft_f1_loss",
    "summarise",
    "train_one",
]
