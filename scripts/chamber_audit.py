"""Audit of the chamber source-detection run: the checks that voided it.

Findings recorded in paper/chamber_source_protocol.md, Amendment V1-V6.
Diagnostics only - no autoencoder is trained and no outflow value is
produced here.

    python scripts/chamber_audit.py
"""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from source_outflow_gate import E, embed, poly2, ridge_r2
from source_outflow_coupling import coupled

SRC = ["hatch", "pot_1", "pot_2"]
SEN = ["load_in","load_out","current_in","current_out","rpm_in","rpm_out",
       "pressure_upwind","pressure_downwind","pressure_ambient",
       "pressure_intake","mic","signal_1","signal_2"]
COLS = SRC + SEN

def self_r2(x, names):
    """Exactly the committed diagnostic, with poly2 (poly3 does not exist)."""
    z = (x - x.mean(0)) / (x.std(0) + 1e-12)
    emb = embed(z); m = emb.shape[0]
    a, b = int(.6*m), int(.8*m)
    tr, te = np.arange(0, a-1), np.arange(b, m-1)
    mu, sd = emb[:a].mean(0), emb[:a].std(0) + 1e-12
    zs = np.clip((emb-mu)/sd, -20, 20).astype(np.float32)
    lead = zs[:, [j*E for j in range(len(names))]]
    return np.array([ridge_r2(poly2(zs[:, q*E:(q+1)*E])[tr], lead[tr+1, q],
                              poly2(zs[:, q*E:(q+1)*E])[te], lead[te+1, q])
                     for q in range(len(names))])

# ---------- CHECK 1: what is actually in the 28 "random-walk runs" -------
files = sorted(glob.glob("Data/causalchamber/wt_walks_v1/*.csv"))
fam = lambda f: ("actuators_rw" if "actuators_random" in f else
                 "loads_hatch_mix" if "loads_hatch" in f else "regime_jumps")
rows = []
for f in files:
    d = pd.read_csv(f)
    if not set(COLS).issubset(d.columns):  continue
    x = d[COLS].to_numpy(float)
    s = x.std(0)
    rows.append({"file": Path(f).name, "family": fam(f), "n": len(x),
                 **{f"sd_{c}": s[COLS.index(c)] for c in SRC}})
D = pd.DataFrame(rows)
print("CHECK 1  composition of the 'random-walk' dataset")
g = D.groupby("family").agg(runs=("n","size"), samples=("n","sum"),
                            sd_hatch=("sd_hatch","median"),
                            sd_pot1=("sd_pot_1","median"),
                            sd_pot2=("sd_pot_2","median")).round(4)
print("   "+g.to_string().replace("\n","\n   "))
tot = D.n.sum()
print(f"   total {tot:,};  regime_jumps share of concatenated arm "
      f"{D[D.family=='regime_jumps'].n.sum()/tot:.1%}")
dead = D[(D[[f'sd_{c}' for c in SRC]] < 1e-9).any(axis=1)]
print(f"   runs with a CONSTANT source column: {len(dead)}")
for _, r in dead.iterrows():
    print(f"     {r.file:36s} sd(hatch,pot1,pot2)="
          f"{r.sd_hatch:.3g},{r.sd_pot_1:.3g},{r.sd_pot_2:.3g}")

# ---------- CHECK 2: self-R2 of the SOURCES, never reported --------------
print("\nCHECK 2  self-R2 of the ACTUATORS (the write-up reported sensors only)")
for f in ["actuators_random_walk_2.csv","regime_jumps_single.csv"]:
    d = pd.read_csv(f"Data/causalchamber/wt_walks_v1/{f}")
    r = self_r2(d[COLS].to_numpy(float), COLS)
    print(f"   {f:28s} sources {np.round(r[:3],4)}  "
          f"sensor median {np.median(r[3:]):.4f}")

# ---------- CHECK 3: does the 0.95 rule fire where outflow WORKS? --------
print("\nCHECK 3  the same diagnostic on the SYNTHETIC systems where outflow works")
names = ["source"]*3 + ["sink"]*6 + ["iso"]*6
for c in (0.05, 0.35, 0.5, 0.7):
    v = []
    for s in (0, 1, 2):
        x, role = coupled(coupling=c, seed=s)
        v.append(self_r2(x, names))
    v = np.array(v)
    snk = np.median(v[:, 3:9]); src = np.median(v[:, :3])
    print(f"   coupling {c:<5} sink self-R2 {snk:.4f}  source self-R2 {src:.4f}"
          f"   -> diagnostic says "
          f"{'NEAR-SYNCHRONOUS (uninformative)' if snk > .95 else 'not synchronous'}")
