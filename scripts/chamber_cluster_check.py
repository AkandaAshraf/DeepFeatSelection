"""Is the chamber's channel-level permutation test valid? No -- and here is
the correct inference in its place.

Found by the statistics lens of the 2026-08-31 adversarial review: Table 8's
permutation p (real_conditional.py's perm_p) shuffles all 65 channels
(2 source + 11 sensor, per each of 5 runs) as if they were independent
draws. They are not -- channels within a run share one trained encoder and
one physical realisation of that run's noise.

This script (a) measures the clustering directly (between-run / within-run
variance ratio in sensor A1 scores), (b) checks whether the source-vs-sensor
separation holds WITHIN every run individually, which rules out the
confound that would make the whole effect a between-run artefact, and
(c) replaces the invalid channel-level permutation test with a run-level
cluster bootstrap: resample the 5 runs with replacement, recompute AUC each
time.

    python scripts/chamber_cluster_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from error_metrics import auc  # noqa: E402

IN = Path("ExpOutput/real_conditional/channels_wt_intake_impulse_v1.csv")
OUT = Path("ExpOutput/chamber_cluster_check")
N_BOOT = 2000
SEED = 0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(IN)
    runs = sorted(d.run.unique())
    print(f"{len(d)} channel-level rows across {len(runs)} runs "
          f"({d[d.role=='source'].shape[0]} source, "
          f"{d[d.role=='sensor'].shape[0]} sensor)\n")

    # ---- clustering diagnostic: is the channel-level i.i.d. assumption ---
    # actually false? One-way ANOVA-style variance decomposition of sensor
    # A1 scores by run.
    sen = d[d.role == "sensor"]
    grand = sen.A1.mean()
    by_run = sen.groupby("run").A1
    ss_between = float(((by_run.mean() - grand) ** 2 * by_run.size()).sum())
    ss_within = float(sum(((sen[sen.run == r].A1 - m) ** 2).sum()
                          for r, m in by_run.mean().items()))
    df_b, df_w = len(runs) - 1, len(sen) - len(runs)
    ratio = (ss_between / df_b) / (ss_within / df_w)
    print(f"CLUSTERING  between-run / within-run variance ratio in sensor "
          f"scores: {ratio:.2f}")
    print("  (>> 1 means channels cluster strongly by run; the 65 "
          "channel-level rows are\n   not exchangeable, and the effective "
          "sample size for inference is the run count.)\n")

    # ---- does the effect hold WITHIN every run, or is it a between-run --
    # artefact (e.g. two noisy runs driving the whole pooled AUC)?
    print("WITHIN-RUN CHECK  source median vs sensor median, per run")
    all_within = True
    for r, g in d.groupby("run"):
        s_med = g[g.role == "source"].A1.median()
        n_med = g[g.role == "sensor"].A1.median()
        ok = s_med > n_med
        all_within &= ok
        print(f"  run {r}: source {s_med:+.4f}  sensor {n_med:+.4f}"
              f"  {'source > sensor' if ok else 'FAILS'}")
    print(f"  -> {'holds in every run: not a between-run artefact' if all_within else 'FAILS in at least one run'}\n")

    # ---- the valid test: run-level cluster bootstrap ----------------------
    rng = np.random.default_rng(SEED)
    obs = auc(d[d.role == "source"].A1.values, d[d.role == "sensor"].A1.values)
    boot = []
    for _ in range(N_BOOT):
        sample = rng.choice(runs, size=len(runs), replace=True)
        sub = pd.concat([d[d.run == r] for r in sample])
        s = sub[sub.role == "source"].A1.values
        n = sub[sub.role == "sensor"].A1.values
        if len(s) and len(n):
            boot.append(auc(s, n))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    below_chance = float((boot <= 0.5).mean())
    print(f"RUN-LEVEL CLUSTER BOOTSTRAP  ({N_BOOT} resamples of "
          f"{len(runs)} runs)")
    print(f"  observed AUC (channel-pooled): {obs:.3f}")
    print(f"  bootstrap mean {boot.mean():.3f}, std {boot.std():.3f}")
    print(f"  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  fraction of resamples at or below chance (0.5): "
          f"{below_chance:.3f}")

    pd.DataFrame({"boot_auc": boot}).to_csv(
        OUT / "run_cluster_bootstrap.csv", index=False)
    pd.DataFrame([{
        "clustering_ratio": ratio, "holds_in_every_run": all_within,
        "observed_auc": obs, "boot_mean": boot.mean(),
        "boot_std": boot.std(), "ci_lo": lo, "ci_hi": hi,
        "frac_at_or_below_chance": below_chance,
    }]).to_csv(OUT / "summary.csv", index=False)

    print("\nVERDICT")
    if all_within and hi > 0.5 and lo > 0.5:
        print("  -> The channel-level permutation p is invalid (channels "
              "cluster by run,\n     ratio "
              f"{ratio:.1f}), but the qualitative result SURVIVES correct "
              "inference:\n     separation holds in every run, and the "
              "run-level 95% CI excludes chance.")
    else:
        print("  -> Does not survive: either the effect fails within a run, "
              "or the\n     run-level CI does not exclude chance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
