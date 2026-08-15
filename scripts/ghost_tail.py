"""Which donors produce a ghost that violates its own null?

A circular shift destroys alignment between a channel and the rest of the
system, which is why a ghost is expected to score zero. But the systems here
are deterministic, so the state at time t formally determines the state at
t+\\Delta; what actually destroys the information is chaotic divergence over
\\Delta steps. The null therefore holds because of mixing, not by
construction, and it should fail exactly for donors whose dynamics stay
predictable across the shift: periodic or weakly chaotic channels.

This script tests that prediction. For each ghost it records the donor's
family, the donor's Lyapunov-like divergence rate, and the ghost's excess,
then asks whether the positive tail is concentrated in the predictable
donors.

    python scripts/ghost_tail.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from bottleneck_membership import build_system, build_system_hetero  # noqa: E402

V, MEMBERS, N, COUPLING, N_GHOSTS = 1000, 100, 2000, 0.3, 200
FAMILY = {0: "logistic", 1: "sine", 2: "tent", 3: "AR(1)"}
CASES = [("seed0", 0, False), ("seed1", 1, False),
         ("seed2", 2, False), ("hetero", 0, True)]


def predictability(series: np.ndarray, lag: int = 1) -> float:
    """How well a channel's next value is fixed by its current one.

    A one-nearest-neighbour analogue prediction on the scalar series: for
    each point find its closest neighbour in value and predict the
    neighbour's successor. High skill means the channel is close to
    periodic or to a low-noise deterministic map, so a shifted copy stays
    predictable and the ghost null is fragile.
    """
    x = series[:-lag]
    y = series[lag:]
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]
    # Nearest neighbour in value, excluding self, via adjacency in sort order.
    pred = np.empty_like(ys)
    pred[0], pred[-1] = ys[1], ys[-2]
    left, right = np.abs(xs[1:-1] - xs[:-2]), np.abs(xs[1:-1] - xs[2:])
    pred[1:-1] = np.where(left < right, ys[:-2], ys[2:])
    return float(max(0.0, 1.0 - np.mean((pred - ys) ** 2) / (np.var(ys) + 1e-12)))


rows = []
for label, seed, hetero in CASES:
    gpath = Path(f"ExpOutput/recall/ghosts_{label}.npy")
    if not gpath.exists():
        print(f"missing {gpath}, skipping")
        continue
    g = np.load(gpath)
    builder = build_system_hetero if hetero else build_system
    x, truth = builder(V, MEMBERS, N, COUPLING, seed)

    grng = np.random.default_rng(seed + 4242)
    donors = grng.choice(V, size=min(N_GHOSTS, V), replace=False)

    for gi, d in zip(g, donors):
        if d < MEMBERS:
            fam = "web member"
        elif hetero:
            fam = FAMILY[(d - MEMBERS) % 4]
        else:
            fam = "logistic"
        rows.append({"system": label, "donor": int(d), "family": fam,
                     "excess": float(gi),
                     "self_predictability": predictability(x[:, d])})

frame = pd.DataFrame(rows)
frame.to_csv("ExpOutput/recall/ghost_tail.csv", index=False)
pd.set_option("display.width", 170)
pd.set_option("display.float_format", "{:.5f}".format)

print("=" * 84)
print("GHOST EXCESS vs DONOR PREDICTABILITY")
print("=" * 84)
frame["positive"] = frame.excess > 0
for sysname, sub in frame.groupby("system", sort=False):
    pos, neg = sub[sub.positive], sub[~sub.positive]
    print(f"\n{sysname}: {len(pos)}/{len(sub)} ghosts above zero")
    print(f"   mean donor predictability -- positive ghosts "
          f"{pos.self_predictability.mean() if len(pos) else float('nan'):.4f}"
          f"  |  non-positive {neg.self_predictability.mean():.4f}")
    if len(pos):
        print("   positive-ghost donors:")
        print(pos[["donor", "family", "excess", "self_predictability"]]
              .sort_values("excess", ascending=False).to_string(index=False))

print("\n" + "=" * 84)
print("BY DONOR FAMILY (heterogeneous system)")
print("=" * 84)
het = frame[frame.system == "hetero"]
if len(het):
    print(het.groupby("family").agg(
        n=("excess", "size"), mean_excess=("excess", "mean"),
        min_excess=("excess", "min"), max_excess=("excess", "max"),
        frac_positive=("positive", "mean"),
        mean_predictability=("self_predictability", "mean")).to_string())

print("\n" + "=" * 84)
print("POOLED: does predictability separate the violating ghosts?")
print("=" * 84)
hi = frame[frame.self_predictability > 0.99]
lo = frame[frame.self_predictability <= 0.99]
for nm, s in (("donor predictability > 0.99", hi),
              ("donor predictability <= 0.99", lo)):
    if len(s):
        print(f"{nm}: n={len(s):3d}  frac ghosts above zero "
              f"{s.positive.mean():.3f}  max excess {s.excess.max():+.5f}")
print("\nwrote ExpOutput/recall/ghost_tail.csv")
