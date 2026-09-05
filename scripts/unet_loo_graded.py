import sys, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from scipy.stats import spearmanr, wilcoxon
from boundary_map import E, embed, make_system, poly3, ridge_r2
ARMS = ["ADD","MASK-SKIP","NOMASK-SKIP","MASK-NOSKIP","NOMASK-NOSKIP"]
# pool ALL sources across the 6 cells into one regression instead of
# averaging six Spearmans each computed on five points
pool = {a: [] for a in ARMS}
for nz in (0.0, 0.05):
    for seed in (0,1,2):
        z = np.load(f"ExpOutput/unet_loo/raw_nz{nz}_s{seed}.npz")
        V, n_src = 30, 5
        rng = np.random.default_rng(seed)
        _ = rng.uniform(0.2,0.8,V); _ = rng.uniform(3.6,3.9,V)
        parent = rng.integers(0, n_src, V-n_src)
        nch = np.array([int((parent==s).sum()) for s in range(n_src)])
        for a in ARMS:
            sc = z[a][:n_src]
            # rank within cell so cells are comparable before pooling
            r = pd.Series(sc).rank().values
            for i in range(n_src):
                pool[a].append((r[i], nch[i]))
print("POST HOC. Pooled across 6 cells: 30 source-observations, ranked within")
print("cell so levels are comparable. Does the score grade with child count?\n")
print(f"{'arm':16s} {'rho':>7s} {'p':>8s}   n")
for a in ARMS:
    arr = np.array(pool[a]); rho, p = spearmanr(arr[:,0], arr[:,1])
    print(f"{a:16s} {rho:+7.3f} {p:8.4f}   {len(arr)}")
print("\nchild counts seen:", sorted(set(int(v) for a in ARMS for _, v in pool[a])))
print("\nAND THE TENSION THAT MATTERS:")
d = pd.read_csv("ExpOutput/unet_loo/cells.csv")
z0 = d.groupby("arm").agg(AP=("avg_precision","mean"), R2=("model_r2","mean"))
arr = np.array(pool["NOMASK-NOSKIP"]); rn,_ = spearmanr(arr[:,0], arr[:,1])
arr = np.array(pool["MASK-SKIP"]); rm,_ = spearmanr(arr[:,0], arr[:,1])
print(f"  NOMASK-NOSKIP  AP {z0.loc['NOMASK-NOSKIP','AP']:.3f}  forecast R2 "
      f"{z0.loc['NOMASK-NOSKIP','R2']:.3f}  grades with children rho {rn:+.3f}")
print(f"  MASK-SKIP      AP {z0.loc['MASK-SKIP','AP']:.3f}  forecast R2 "
      f"{z0.loc['MASK-SKIP','R2']:.3f}  grades with children rho {rm:+.3f}")
print("\n  The arm with the HIGHEST average precision is the one that cannot")
print("  forecast and whose score carries no information about how much a")
print("  source drives. That is what U5 was written to catch.")
