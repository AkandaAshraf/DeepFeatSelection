"""How much does an affine-in-code readout lose against the identified quantity?

Proposition 2 identifies excess with the share of a variable's next-step
variance its own history cannot explain, but the identification needs the
joint readout to be able to represent the target as a function of the code.
Eq. 1's readout applies the polynomial map to the OWN LAGS only and appends
the code raw, so it is affine in the code while the coupling enters the
generator multiplicatively.

This measures the resulting gap under the most favourable possible conditions:
an oracle code (the full raw state, so hypothesis (i) holds exactly), abundant
samples, and negligible ridge penalty. Whatever shortfall survives here is
attributable to the readout's function class alone.

    python scripts/readout_class_gap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent))
from network_scale import random_dag, simulate  # noqa: E402

N, NODES, ALPHA = 6000, 12, 1e-6
COUPLINGS = [0.15, 0.30]
TARGETS = [2, 3, 5]
E = 2


def embed(v: np.ndarray, e: int) -> np.ndarray:
    return np.column_stack([v[i:len(v) - e + 1 + i] for i in range(e)])


def r2(pred, y):
    return max(0.0, 1.0 - np.mean((pred - y) ** 2) / (np.var(y) + 1e-12))


def fit(cols, y, tr, te):
    m = Ridge(alpha=ALPHA)
    m.fit(cols[tr], y[tr])
    return r2(m.predict(cols[te]), y[te])


rows = []
for coupling in COUPLINGS:
    rng = np.random.default_rng(0)
    edges = random_dag(NODES, int(1.5 * NODES), rng)
    x = simulate(N, edges, NODES, coupling, 0)
    parents = {j: [i for i, jj in edges if jj == j] for j in range(NODES)}

    n = N - E + 1
    tr = slice(0, int(0.7 * n))
    te = slice(int(0.75 * n), n - 1)
    tr_i = np.arange(tr.start, tr.stop - 1)
    te_i = np.arange(te.start, te.stop)

    # Oracle code: the full raw state at time t. Hypothesis (i) holds exactly.
    code = x[E - 1:E - 1 + n, :]

    for q in TARGETS:
        if not parents.get(q):
            continue
        own = embed(x[:, q], E)[:n]
        own_p = np.column_stack([own, own ** 2, own[:, 0] * own[:, 1]])
        y = x[E:E + n, q]

        self_r2 = fit(own_p, y, tr_i, te_i)
        affine = fit(np.column_stack([own_p, code]), y, tr_i, te_i)
        inter = np.column_stack([own_p, code,
                                 own[:, [0]] * code, own[:, [1]] * code])
        rich = fit(inter, y, tr_i, te_i)

        identified = 1.0 - self_r2          # Prop 2's value when joint R2 = 1
        rows.append({"coupling": coupling, "q": q, "parents": len(parents[q]),
                     "self_r2": self_r2, "joint_affine": affine,
                     "joint_interact": rich,
                     "identified": identified,
                     "excess_affine": affine - self_r2,
                     "excess_interact": rich - self_r2,
                     "shortfall_%": 100 * (1 - (affine - self_r2) / identified)})

f = pd.DataFrame(rows)
pd.set_option("display.width", 170)
pd.set_option("display.float_format", "{:.6f}".format)
print("=" * 92)
print("AFFINE-IN-CODE READOUT vs THE IDENTIFIED QUANTITY (oracle code)")
print("=" * 92)
print(f.to_string(index=False))
print(f"\nshortfall of the affine readout: "
      f"{f['shortfall_%'].min():.1f}% to {f['shortfall_%'].max():.1f}%")
print(f"interaction readout recovers the identified value to within "
      f"{np.abs(f.excess_interact - f.identified).max():.2e}")
Path("ExpOutput/recall").mkdir(parents=True, exist_ok=True)
f.to_csv("ExpOutput/recall/readout_class_gap.csv", index=False)
print("wrote ExpOutput/recall/readout_class_gap.csv")
