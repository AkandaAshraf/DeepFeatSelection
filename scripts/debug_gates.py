"""Print raw gate values after a single run, to see what the penalty actually did."""

import sys

import numpy as np

from deepfeatselect.data import prepare
from deepfeatselect.train import TrainConfig, train_one

l1 = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
lr = float(sys.argv[2]) if len(sys.argv) > 2 else 3e-3
bs = int(sys.argv[3]) if len(sys.argv) > 3 else 32
m = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0

data = prepare("Data/processed.cleveland.data", task="binary", seed=0)
cfg = TrainConfig(
    task="binary", l1_gate=l1, learning_rate=lr, batch_size=bs, patience=40, hierarchy_m=m
)
res = train_one(data, cfg, seed=0)

print(f"\nl1={l1}  lr={lr}  batch={bs}  M={m}  prox_threshold/step={lr * l1:.4g}")
print(f"epochs_run={res.epochs_run}  steps~={res.epochs_run * int(np.ceil(len(data.y_train) / bs))}")
print(f"test_auc={res.metrics.get('test_auc'):.4f}")
print("\nraw gates:")
for name, g in sorted(zip(data.feature_names, res.gates), key=lambda t: -t[1]):
    print(f"  {name:<10} {g:.6f}")
print(f"\nexact zeros: {(res.gates == 0).sum()} / {len(res.gates)}")
print(f"gate sum={res.gates.sum():.4f}  min={res.gates.min():.6f}  max={res.gates.max():.6f}")
