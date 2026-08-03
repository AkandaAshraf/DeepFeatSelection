"""Does the deprivation signature vanish when capacity stops being scarce?

Proposition 2 is the mechanism behind every internal measurement in this
project: a network denied a redundant feature must re-derive it from the
others, and the probe sees the *cost* of that re-derivation.  The proposition
carries a falsifiable consequence -- widen the network until synthesising the
missing feature is free, and every signature must decay to the null floor.

If the signatures persist at large width, the proposed mechanism is wrong, and
whatever the probe detects is something other than a capacity cost.  This is the
experiment that can kill the explanation while leaving the measurements intact.

    python scripts/capacity_sweep.py --widths 4 8 16 32 64 128 --seeds 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_internals import build_dataset  # noqa: E402

from deepfeatselect.netstats import train_and_measure  # noqa: E402
from deepfeatselect.probe import ablate_feature  # noqa: E402
from deepfeatselect.train import TrainConfig  # noqa: E402


def run_width(width: int, args) -> pd.DataFrame:
    data, names = build_dataset(args.n, args.n_nulls, seed=0, label=args.label)
    config = TrainConfig(
        task="binary", l1_gate=0.0, dropout=0.0, noise=0.0,
        hidden_units=width, n_hidden_layers=args.n_hidden_layers,
        learning_rate=3e-3, epochs=args.epochs, batch_size=128,
        hierarchy=False, class_weight=False,
    )

    rows = []
    for seed in range(args.seeds):
        seed_data, _ = build_dataset(args.n, args.n_nulls, seed=seed, label=args.label)
        for arm in ["full"] + names:
            arm_data = (seed_data if arm == "full"
                        else ablate_feature(seed_data, names.index(arm)))
            result = train_and_measure(arm_data, config, seed=seed)
            rows.append({"arm": arm, "seed": seed, **result.metrics})
    return pd.DataFrame(rows), names


def z_scores(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    metrics = [c for c in df.columns if c not in ("arm", "seed")]
    full = df[df.arm == "full"].set_index("seed")

    deltas = []
    for arm in names:
        sub = df[df.arm == arm].set_index("seed")
        for seed in sub.index:
            deltas.append({"arm": arm, **{m: sub.loc[seed, m] - full.loc[seed, m]
                                          for m in metrics}})
    dd = pd.DataFrame(deltas)

    nulls = [n for n in names if n.startswith("null_") or n == "unrelated"]
    candidates = [n for n in names if n not in nulls]
    null_d = dd[dd.arm.isin(nulls)]

    out = {}
    raw = {}
    spread = {}
    for m in metrics:
        sd = null_d[m].std(ddof=1)
        if sd == 0 or not np.isfinite(sd):
            continue
        mu = null_d[m].mean()
        out[m] = {arm: (dd[dd.arm == arm][m].mean() - mu) / sd for arm in candidates}
        # Kept separately so the numerator and the denominator of every z can be
        # inspected: a z that grows because the null spread shrank is an artefact
        # of the normalisation, not a stronger effect.
        raw[m] = {arm: dd[dd.arm == arm][m].mean() - mu for arm in candidates}
        spread[m] = sd
    return pd.DataFrame(out).T, pd.DataFrame(raw).T, pd.Series(spread)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--widths", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--n-nulls", type=int, default=2)
    p.add_argument("--n-hidden-layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--label", default="successor_median")
    p.add_argument("--outdir", default="ExpOutput/capacity")
    args = p.parse_args()

    summary_rows = []
    for width in args.widths:
        df, names = run_width(width, args)
        z, raw, spread = z_scores(df, names)
        totals = z.abs().sum()
        n_sig = (z.abs() > 2).sum()
        print(f"\nwidth {width:>4}  "
              + "  ".join(f"{c}: sum|z|={totals[c]:6.1f} nsig={n_sig[c]:2d}"
                          for c in z.columns))
        for arm in z.columns:
            summary_rows.append({
                "width": width, "feature": arm,
                "sum_abs_z": totals[arm], "max_abs_z": z[arm].abs().max(),
                "n_significant": int(n_sig[arm]),
                # Loss-level blindness should hold at every width (Prop 1 is a
                # population statement); only the internal channels should decay.
                "val_loss_z": z.loc["val_loss_final", arm] if "val_loss_final" in z.index else np.nan,
                # The two halves of every z, so a growing ratio can be attributed
                # to a real effect or to a shrinking null spread.
                "raw_pr_norm": raw.loc["act_pr_norm_data", arm] if "act_pr_norm_data" in raw.index else np.nan,
                "raw_val_loss_area": raw.loc["val_loss_area", arm] if "val_loss_area" in raw.index else np.nan,
                "null_sd_pr_norm": spread.get("act_pr_norm_data", np.nan),
                "null_sd_val_loss_area": spread.get("val_loss_area", np.nan),
            })

    summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 78)
    print("CAPACITY SWEEP -- Proposition 2 predicts decay toward the null floor")
    print("=" * 78)
    pivot = summary.pivot(index="width", columns="feature", values="sum_abs_z")
    with pd.option_context("display.float_format", "{:8.1f}".format):
        print("\nsum |z| across all functionals")
        print(pivot.to_string())
    print("\nchannels with |z| > 2")
    print(summary.pivot(index="width", columns="feature", values="n_significant").to_string())

    print("\nverdict per feature (first width -> last width):")
    for feature in pivot.columns:
        first, last = pivot[feature].iloc[0], pivot[feature].iloc[-1]
        direction = "DECAYS" if last < 0.5 * first else "PERSISTS"
        print(f"  {feature:<12} {first:7.1f} -> {last:7.1f}   {direction}")

    # The artefact check. If the null spread falls faster than the raw effect,
    # the growth in z is normalisation rather than signal.
    print("\n" + "=" * 78)
    print("RAW EFFECT vs NULL SPREAD -- is the z growth real?")
    print("=" * 78)
    for channel, raw_col, sd_col in [
        ("participation ratio", "raw_pr_norm", "null_sd_pr_norm"),
        ("val-loss area", "raw_val_loss_area", "null_sd_val_loss_area"),
    ]:
        print(f"\n{channel}")
        sd_by_width = summary.groupby("width")[sd_col].first()
        raw_by_width = summary.pivot(index="width", columns="feature", values=raw_col)
        table = raw_by_width.copy()
        table["null_sd"] = sd_by_width
        with pd.option_context("display.float_format", "{:+.5f}".format):
            print(table.to_string())
        first_sd, last_sd = sd_by_width.iloc[0], sd_by_width.iloc[-1]
        print(f"  null spread {first_sd:.5f} -> {last_sd:.5f} "
              f"({'SHRINKS' if last_sd < 0.5 * first_sd else 'stable'})")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / "capacity_sweep.csv", index=False)
    print(f"\nwrote {outdir}/capacity_sweep.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
