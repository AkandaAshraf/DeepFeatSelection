"""POST HOC, labelled. Does the LOO score measure OUTFLOW, or a proxy?

Two ways it could look right for the wrong reason:
  (a) sources differ from driven channels in variance / self-predictability,
      and the model is simply more sensitive to removing such a channel;
  (b) it separates classes but carries no information about HOW MUCH a
      source drives.

The generator's parent assignment is recoverable by replaying the seed's
draws, so (b) has a positive control: a source with more children should
score higher. That is a graded prediction no class-difference confound
produces.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from scipy.stats import spearmanr
from boundary_map import E, embed, make_system, poly3, ridge_r2

ARMS = ["ADD", "MASK-SKIP", "NOMASK-SKIP", "MASK-NOSKIP", "NOMASK-NOSKIP"]
rows = []
for nz in (0.0, 0.05):
    for seed in (0, 1, 2):
        z = np.load(f"ExpOutput/unet_loo/raw_nz{nz}_s{seed}.npz")
        x, isd, iss = make_system(4000, 30, 0.20, 0, seed)
        V = 30; n_src = max(3, V // 6)
        # replay the generator's draws in order to recover parent[]
        rng = np.random.default_rng(seed)
        _ = rng.uniform(0.2, 0.8, V); _ = rng.uniform(3.6, 3.9, V)
        parent = rng.integers(0, n_src, V - n_src)
        n_children = np.zeros(V)
        for s in range(n_src):
            n_children[s] = int((parent == s).sum())
        if nz:
            x = x + nz*np.random.default_rng(seed+777).standard_normal(x.shape)
        emb = embed(x); m = emb.shape[0]
        a, bnd = int(.6*m), int(.8*m)
        tr = slice(0,a); tr_i = np.arange(0,a-1); te_i = np.arange(bnd,m-1)
        mu, sd = emb[tr].mean(0), emb[tr].std(0)+1e-12
        zs = np.clip(np.nan_to_num((emb-mu)/sd),-20,20).astype(np.float32)
        lead = zs[:, [j*E for j in range(V)]]
        self_r2 = np.array([ridge_r2(poly3(zs[:,q*E:(q+1)*E])[tr_i], lead[tr_i+1,q],
                                     poly3(zs[:,q*E:(q+1)*E])[te_i], lead[te_i+1,q])
                            for q in range(V)])
        var = x.var(0)
        for arm in ARMS:
            sc = z[arm]
            rows.append(dict(noise=nz, seed=seed, arm=arm,
                rho_selfr2=spearmanr(sc, self_r2).statistic,
                rho_var=spearmanr(sc, var).statistic,
                # positive control: among SOURCES only, does score track children?
                rho_children=spearmanr(sc[:n_src], n_children[:n_src]).statistic,
                spread_children=float(n_children[:n_src].std())))
t = pd.DataFrame(rows)
print(__doc__)
print("(a) CONFOUND: does the score track self-R2 or variance across all channels?")
print("   " + t.groupby("arm")[["rho_selfr2","rho_var"]].mean().reindex(ARMS)
      .round(3).to_string().replace("\n","\n   "))
print("\n(b) POSITIVE CONTROL: among sources only, does score track child count?")
print(f"   child-count spread across sources, mean sd = {t.spread_children.mean():.2f}"
      f"  (zero spread would make this test empty)")
print("   " + t.groupby("arm").rho_children.agg(['mean','count']).reindex(ARMS)
      .round(3).to_string().replace("\n","\n   "))
print("\n   per noise level:")
print("   " + t.pivot_table(index="noise", columns="arm", values="rho_children")[ARMS]
      .round(3).to_string().replace("\n","\n   "))
