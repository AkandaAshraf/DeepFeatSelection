"""Can a network's internals see a feature that accuracy provably cannot?

On redundancy_demo, leave-one-out accuracy for the true driver is exactly zero
(the proxies reconstruct the target perfectly), so any detector defined as a
difference of achievable risks is blind by theorem.  This experiment asks
whether permutation-invariant functionals of the trained network -- activation
geometry, weight spectra, compressibility, learning speed, residual dependence
-- shift when the driver is removed, beyond a null calibrated on features that
are irrelevant by construction.

Design notes that matter:
- Small network on purpose.  With enough capacity the reduced model synthesises
  the missing feature for free and every contrast vanishes along with the
  accuracy contrast; the probes only have something to see when re-deriving the
  composite costs representation.
- Contiguous time split, so residual autocorrelation is meaningful.
- Arms are paired by seed, and significance is a z-score of the candidate's
  paired deltas against the deltas of injected pure-noise features under the
  same seeds.

    python scripts/probe_internals.py --n 1500 --seeds 3
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from deepfeatselect.data import Dataset
from deepfeatselect.netstats import train_and_measure
from deepfeatselect.probe import ablate_feature
from deepfeatselect.synthetic import redundancy_demo
from deepfeatselect.train import TrainConfig


def make_labels(x: np.ndarray, names: list[str], y_raw: np.ndarray, label: str) -> np.ndarray:
    """Two encodings of the same system, differing only in which readout is cheap.

    ``successor_median`` -- 1[u_{t+1} > median].  The logistic map is symmetric
    about u = 1/2 and so is cos(2*pi*u), so the two symmetries coincide and the
    label becomes a *single threshold* in proxy_cos while remaining a two-sided
    interval in driver space.  proxy_cos is the cheap coordinate.

    ``driver_threshold`` -- 1[u_t > 1/2].  Now cos(2*pi*u) = cos(2*pi*(1-u))
    maps both classes onto identical values, so proxy_cos is exactly
    uninformative on its own, while sin(2*pi*u) changes sign at u = 1/2 and the
    driver is a single threshold.  The cheap coordinates swap.

    Deterministic redundancy survives in both: under the second label, dropping
    the driver leaves the sign of proxy_sin, and dropping proxy_sin leaves the
    driver, so conditional Shannon information stays degenerate and the
    dissociation stays testable.
    """
    if label == "successor_median":
        return (y_raw > np.median(y_raw)).astype(np.int64)
    if label == "driver_threshold":
        return (x[:, names.index("driver")] > 0.5).astype(np.int64)
    raise ValueError(f"unknown label encoding {label!r}")


def build_dataset(
    n: int, n_nulls: int, seed: int, label: str = "successor_median"
) -> tuple[Dataset, list[str]]:
    system = redundancy_demo(n=n, seed=seed)
    x = np.asarray(system["x"], dtype=np.float64)
    names = list(system["feature_names"])

    # Injected standard-normal columns, irrelevant by construction: they are the
    # null pool the candidate contrasts are scored against.
    rng = np.random.default_rng(seed + 1)
    x = np.hstack([x, rng.standard_normal((len(x), n_nulls))])
    names += [f"null_{i + 1}" for i in range(n_nulls)]

    y = make_labels(x, names, np.asarray(system["y"]), label)

    # Contiguous split: residual autocorrelation on the test block is only
    # interpretable if the block preserves time order.
    a, b = int(0.6 * len(x)), int(0.8 * len(x))
    scaler = StandardScaler().fit(x[:a])
    return Dataset(
        x_train=scaler.transform(x[:a]), y_train=y[:a],
        x_val=scaler.transform(x[a:b]), y_val=y[a:b],
        x_test=scaler.transform(x[b:]), y_test=y[b:],
        feature_names=names,
        groups=np.arange(len(names), dtype=np.int32),
        n_classes=2,
    ), names


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--n-nulls", type=int, default=2)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--hidden-units", type=int, default=16)
    p.add_argument("--n-hidden-layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--outdir", default="ExpOutput/internals")
    p.add_argument(
        "--label",
        choices=("successor_median", "driver_threshold"),
        default="successor_median",
        help="target encoding; the reversal experiment compares the two",
    )
    args = p.parse_args()

    data, names = build_dataset(args.n, args.n_nulls, seed=0, label=args.label)
    print(f"label encoding: {args.label}")
    config = TrainConfig(
        task="binary",
        # No gate penalty and no stochastic layers: the measurement target is
        # the representation itself, and dropout/noise would smear every
        # functional with their own randomness.
        l1_gate=0.0, dropout=0.0, noise=0.0,
        hidden_units=args.hidden_units, n_hidden_layers=args.n_hidden_layers,
        learning_rate=3e-3, epochs=args.epochs, batch_size=128,
        hierarchy=False, class_weight=False,
    )

    arms = ["full"] + names
    rows = []
    for seed in range(args.seeds):
        for arm in arms:
            # Each seed redraws the system, so the split and the trajectory vary
            # together with the initialisation.
            seed_data, _ = build_dataset(args.n, args.n_nulls, seed=seed, label=args.label)
            if arm == "full":
                arm_data = seed_data
            else:
                arm_data = ablate_feature(seed_data, names.index(arm))
            result = train_and_measure(arm_data, config, seed=seed)
            rows.append({"arm": arm, "seed": seed, **result.metrics})
            print(f"  seed {seed}  drop={arm:<10} "
                  f"val_loss={result.metrics['val_loss_final']:.4f} "
                  f"area={result.metrics['val_loss_area']:.4f} "
                  f"hsic_p={result.metrics['resid_hsic_p']:.3f}")

    df = pd.DataFrame(rows)
    metrics = [c for c in df.columns if c not in ("arm", "seed")]

    # Paired deltas: each arm minus the full model under the same seed.
    full = df[df.arm == "full"].set_index("seed")
    deltas = []
    for arm in names:
        sub = df[df.arm == arm].set_index("seed")
        for seed in sub.index:
            row = {"arm": arm, "seed": seed}
            for m in metrics:
                row[m] = sub.loc[seed, m] - full.loc[seed, m]
            deltas.append(row)
    dd = pd.DataFrame(deltas)

    null_arms = [n for n in names if n.startswith("null_") or n == "unrelated"]
    cand_arms = [n for n in names if n not in null_arms]
    null_d = dd[dd.arm.isin(null_arms)]

    print("\nmean paired delta per arm (arm minus full, averaged over seeds)")
    print("=" * 78)
    summary = dd.groupby("arm")[metrics].mean().reindex(names)
    with pd.option_context("display.float_format", "{:+.4f}".format, "display.width", 200):
        print(summary.T.to_string())

    print("\nz-scores of candidate deltas against the injected-null distribution")
    print("=" * 78)
    z_rows = {}
    for m in metrics:
        mu, sd = null_d[m].mean(), null_d[m].std(ddof=1)
        if sd == 0 or not np.isfinite(sd):
            continue
        z_rows[m] = {
            arm: (dd[dd.arm == arm][m].mean() - mu) / sd for arm in cand_arms
        }
    zt = pd.DataFrame(z_rows).T
    with pd.option_context("display.float_format", "{:+.2f}".format):
        print(zt.to_string())

    detected = zt[zt.abs().max(axis=1) > 2.0]
    print(f"\nfunctionals with |z| > 2 for at least one causal-set feature: "
          f"{len(detected)} of {len(zt)}")
    for m in detected.index:
        best = zt.loc[m].abs().idxmax()
        print(f"  {m:<28} strongest on {best:<10} z={zt.loc[m, best]:+.2f}")

    from pathlib import Path
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "internals_raw.csv", index=False)
    dd.to_csv(outdir / "internals_deltas.csv", index=False)
    zt.to_csv(outdir / "internals_zscores.csv")
    print(f"\nwrote {outdir}\\internals_raw.csv, internals_deltas.csv, internals_zscores.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
