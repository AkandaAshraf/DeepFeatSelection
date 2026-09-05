"""POST HOC, labelled. Two questions the aggregate cannot answer."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr

print("=== 1. WAS THE REDUNDANCY TEST ACTUALLY HARD? ===")
print("make_system with redundancy=2 appends ONE duplicate each for sources 0")
print("and 1 only. Sources 2,3,4 have NO duplicate, so they should keep their")
print("outflow and prop up the aggregate. The sharp test is WITHIN a cell:")
print("do the DUPLICATED sources specifically lose outflow?\n")
rows = []
for nz in (0.0, 0.05):
    for seed in (0,1,2):
        z0 = np.load(f"ExpOutput/unified/raw_nz{nz}_r0_s{seed}.npz")
        z2 = np.load(f"ExpOutput/unified/raw_nz{nz}_r2_s{seed}.npz")
        for arm, key in (("UNIFIED","outflow_u"), ("ADD","outflow_a")):
            a0, a2 = z0[key][:5], z2[key][:5]
            # rank sources within each cell so levels are comparable
            r0 = pd.Series(a0).rank().values; r2 = pd.Series(a2).rank().values
            for s in range(5):
                rows.append(dict(noise=nz, seed=seed, arm=arm, src=s,
                                 is_dup=s < 2, rank_r0=r0[s], rank_r2=r2[s],
                                 raw_r0=a0[s], raw_r2=a2[s]))
t = pd.DataFrame(rows)
for arm, g in t.groupby("arm"):
    print(f"  {arm}: mean outflow RANK among the 5 sources (5 = highest)")
    p = g.groupby("is_dup")[["rank_r0","rank_r2"]].mean()
    print("     " + p.round(2).to_string().replace("\n","\n     "))
    dup = g[g.is_dup]
    non = g[~g.is_dup]
    print(f"     duplicated sources: rank {dup.rank_r0.mean():.2f} -> {dup.rank_r2.mean():.2f}"
          f"   change {dup.rank_r2.mean()-dup.rank_r0.mean():+.2f}")
    print(f"     untouched sources:  rank {non.rank_r0.mean():.2f} -> {non.rank_r2.mean():.2f}"
          f"   change {non.rank_r2.mean()-non.rank_r0.mean():+.2f}\n")

print("=== 2. WHY DOES UNIFIED INFLOW COLLAPSE? ===")
print("N5 held at R2 0.88-0.97, so the model forecasts well. Is the unified")
print("inflow measuring the same thing as ridge inflow, only noisier?\n")
for nz in (0.0,):
    for seed in (0,1,2):
        z = np.load(f"ExpOutput/unified/raw_nz{nz}_r0_s{seed}.npz")
        iu, ir = z["inflow_u"], z["inflow_r"]
        rho = spearmanr(iu, ir).statistic
        # does unified inflow instead track how well the SHARED decoder does?
        rho2 = spearmanr(iu, z["r2_full"]).statistic
        rho3 = spearmanr(iu, z["r2_skip"]).statistic
        print(f"  seed {seed}: rho(unified, ridge) {rho:+.3f}   "
              f"rho(unified, decoder R2) {rho2:+.3f}   "
              f"rho(unified, skip R2) {rho3:+.3f}")
print("\n  A weak correlation with ridge plus a strong one with the shared")
print("  decoder's own accuracy means the unified inflow reports where the")
print("  ONE shared decoder chose to spend capacity, not what each target")
print("  gains from the rest of the system.")
