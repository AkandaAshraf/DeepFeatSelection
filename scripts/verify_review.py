"""Independent re-derivation of the numbers an adversarial review disputed.

Nothing here is taken on trust from the review: every figure is recomputed
from the primary files so the corrections written into the paper rest on this
script rather than on a report.

    python scripts/verify_review.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent))
from bottleneck_membership import build_system, build_system_hetero  # noqa: E402
from network_scale import random_dag, simulate  # noqa: E402

V, MEMBERS, N, COUPLING = 1000, 100, 2000, 0.3
SEEDS = [("seed0", 0, False, "ExpOutput/excess_poly/excess_consensus.npy"),
         ("seed1", 1, False, "ExpOutput/excess_s1/excess_consensus.npy"),
         ("seed2", 2, False, "ExpOutput/excess_s2/excess_consensus.npy"),
         ("hetero", 0, True, "ExpOutput/excess_het/excess_consensus.npy")]
BAR = "=" * 78


def degrees(seed):
    n_edges = max(MEMBERS, int(1.2 * MEMBERS))
    for attempt in range(20):
        rng = np.random.default_rng(seed + 100 * attempt)
        cand = random_dag(MEMBERS, n_edges, rng)
        try:
            simulate(N, cand, MEMBERS, COUPLING, seed)
            ind, outd = np.zeros(MEMBERS, int), np.zeros(MEMBERS, int)
            for i, j in cand:
                outd[i] += 1
                ind[j] += 1
            return ind, outd
        except ValueError:
            continue
    raise RuntimeError(seed)


print(BAR)
print("1. CCM COST: is the per-pair time an ordered or unordered measurement?")
print(BAR)
m = pd.read_csv("ExpOutput/membership/membership.csv")
row = m[m.method.astype(str).str.contains("ccm", case=False, na=False)]
print(row[["seed", "method", "prauc", "auroc", "seconds"]].to_string(index=False))
full = m[m.method == "ccm_full"]
secs = float(full.seconds.iloc[0])   # ccm_full, NOT the ccm_pca arm
v_all = 31                       # V=30 plus the ghost, per the benchmark
calls = v_all * (v_all - 1)      # the loop calls ccm() for every ORDERED pair
per_call = secs / calls
print(f"\nccm_full: {secs:.2f} s over {calls} calls -> {per_call:.4f} s per call")
print("Each ccm(i,j) call already returns BOTH directions (x_causes_y and")
print("y_causes_x), but the loop also calls ccm(j,i), so the benchmark does")
print("twice the necessary work. An efficient scan needs v(v-1)/2 calls.")
for label, v in (("V=1e4", 10_000), ("worm V=131", 131),
                 ("climate V=10512", 10_512), ("fish V=71721", 71_721)):
    need = v * (v - 1) / 2 * per_call / 3600      # calls actually required
    asrep = v * v * per_call / 3600               # what the paper projected
    print(f"  {label:18s} required {need:>12,.0f} h | as projected "
          f"{asrep:>12,.0f} h")

print("\n" + BAR)
print("2. IN-DEGREE: does it separate detected from missed driven channels?")
print(BAR)
flag_ind, miss_ind, allrows = [], [], []
for label, seed, hetero, path in SEEDS:
    ex = np.load(path)
    ok = pd.read_csv("ExpOutput/recall/ghost_tail.csv")
    ok = ok[(ok.system == label) & (ok.self_predictability > 0.99)]
    thr = max(0.0, float(ok.excess.max()))
    ind, outd = degrees(seed)
    driven = ind > 0
    flagged = ex[:MEMBERS] > thr
    flag_ind += list(ind[driven & flagged])
    miss_ind += list(ind[driven & ~flagged])
    allrows.append({"system": label, "driven": int(driven.sum()),
                    "flagged_driven": int((driven & flagged).sum()),
                    "indeg0": int((~driven).sum())})
f, mi = np.array(flag_ind), np.array(miss_ind)
print(pd.DataFrame(allrows).to_string(index=False))
print(f"\nflagged driven   n={len(f):3d}  mean in-degree {f.mean():.3f}")
print(f"undetected driven n={len(mi):3d}  mean in-degree {mi.mean():.3f}")
print(f"all driven        n={len(f)+len(mi):3d}  mean in-degree "
      f"{np.concatenate([f,mi]).mean():.3f}")
u = mannwhitneyu(f, mi, alternative="greater")
print(f"Mann-Whitney (flagged > missed): U={u.statistic:.0f}, p={u.pvalue:.3f}")
print(f"generator constant n_edges/MEMBERS = {max(MEMBERS,int(1.2*MEMBERS))}/"
      f"{MEMBERS} = {max(MEMBERS,int(1.2*MEMBERS))/MEMBERS:.2f}  <-- the 1.20")
for d in (1, 2):
    nd = int((np.concatenate([f, mi]) == d).sum())
    fd = int((f == d).sum())
    print(f"  recall at in-degree {d}: {fd}/{nd} = {fd/nd:.3f}")

print("\n" + BAR)
print("3. IS THE HETEROGENEOUS POOL A DISTINCT SYSTEM?")
print(BAR)
xa, ta = build_system(V, MEMBERS, N, COUPLING, 0)
xb, tb = build_system_hetero(V, MEMBERS, N, COUPLING, 0)
same = np.array_equal(xa[:, :MEMBERS], xb[:, :MEMBERS])
print(f"web channels identical: {same}  "
      f"max|diff| {np.abs(xa[:, :MEMBERS]-xb[:, :MEMBERS]).max():.3g}")
print(f"truth vectors identical: {np.array_equal(ta[:MEMBERS], tb[:MEMBERS])}")
print(f"loner channels identical: {np.array_equal(xa[:, MEMBERS:], xb[:, MEMBERS:])}")
r = pd.read_csv("ExpOutput/recall/corrected_roles.csv")
print("\npooled role counts as currently reported (4 systems):")
print(r.groupby("role").n.sum().to_string())
distinct = r[r.system != "hetero"]
print("\ndistinct generating graphs only (3 systems):")
print(distinct.groupby("role").agg(n=("n", "sum"),
                                   flagged=("flagged", "sum")).to_string())
dd = distinct.groupby("role").agg(n=("n", "sum"), flagged=("flagged", "sum"))
print(f"recall on driven, 3 distinct graphs: "
      f"{int(dd.loc['driven','flagged'])}/{int(dd.loc['driven','n'])} = "
      f"{dd.loc['driven','flagged']/dd.loc['driven','n']:.3f}")
t = pd.read_csv("ExpOutput/recall/corrected_threshold.csv")
sub = t[t.system != "hetero"]
tp, fp = int(sub.true_pos.sum()), int(sub.false_pos.sum())
print(f"precision on 3 distinct graphs: {tp}/{tp+fp} = {tp/(tp+fp):.3f}")

print("\n" + BAR)
print("4. FALSE ALARMS ON THE SYNTHETIC TABLE")
print(BAR)
for label, seed, hetero, path in SEEDS:
    ex = np.load(path)
    builder = build_system_hetero if hetero else build_system
    _, truth = builder(V, MEMBERS, N, COUPLING, seed)
    ta_ = np.append(truth, False)
    g = float(ex[-1])
    nm = ex[MEMBERS:V]
    order = np.argsort(-ex)
    top20 = order[:20]
    mem_top20 = [float(ex[i]) for i in top20 if ta_[i]]
    print(f"{label:7s} ghost {g:+.3e} | non-members above signed ghost "
          f"{int((nm > g).sum()):3d} | above |ghost| {int((nm > abs(g)).sum()):3d}"
          f" | strictly >0 {int((nm > 0).sum()):3d}")
    print(f"        max non-member {nm.max():+.3e} | weakest top-20 member "
          f"{min(mem_top20):+.3e} | ratio "
          f"{min(mem_top20)/max(nm.max(),1e-12):.2f}x")

print("\n" + BAR)
print("5. SELF-R2 ON THE REAL DEPLOYMENTS (Prop 1's premise)")
print(BAR)
rows = []
for pat, tag in [("ExpOutput/celegans_excess/worm*_excess.csv", "worm WT"),
                 ("ExpOutput/celegans_excess_heldout/worm*_excess.csv", "worm held-out"),
                 ("ExpOutput/celegans_excess_avahiscl/worm*_excess.csv", "worm AVA-HisCl")]:
    for f in sorted(Path().glob(pat)):
        d = pd.read_csv(f)
        if "self_r2" not in d.columns:
            continue
        s = d.self_r2
        rows.append({"dataset": tag, "file": f.name, "n_ch": len(s),
                     "median": s.median(), "max": s.max(),
                     "frac>0.9": float((s > 0.9).mean()),
                     "frac>0.99": float((s > 0.99).mean())})
for f, tag in [("ExpOutput/eeg_excess/channels.csv", "EEG"),
               ("ExpOutput/climate_excess/cells.csv", "climate")]:
    p = Path(f)
    if p.exists():
        d = pd.read_csv(p)
        if "self_r2" in d.columns:
            s = d.self_r2
            rows.append({"dataset": tag, "file": p.name, "n_ch": len(s),
                         "median": s.median(), "max": s.max(),
                         "frac>0.9": float((s > 0.9).mean()),
                         "frac>0.99": float((s > 0.99).mean())})
sr = pd.DataFrame(rows)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", "{:.4f}".format)
print(sr.to_string(index=False) if len(sr) else "no self_r2 columns found")
if len(sr):
    print(f"\nOVERALL: {len(sr)} real recordings, "
          f"median self-R2 range {sr['median'].min():.3f}-{sr['median'].max():.3f}, "
          f"max observed {sr['max'].max():.3f}, "
          f"total channels above 0.99: "
          f"{int((sr['frac>0.99']*sr.n_ch).sum())}")

print("\n" + BAR)
print("6. WORM GHOST VALUES (Table 2 range)")
print(BAR)
for pat, tag in [("ExpOutput/celegans_excess/worm*_excess.csv", "WT"),
                 ("ExpOutput/celegans_excess_heldout/worm*_excess.csv", "held-out"),
                 ("ExpOutput/celegans_excess_avahiscl/worm*_excess.csv", "AVA-HisCl")]:
    vals = []
    for f in sorted(Path().glob(pat)):
        d = pd.read_csv(f)
        col = "name" if "name" in d.columns else d.columns[0]
        gh = d[d[col].astype(str).str.contains("ghost", case=False, na=False)]
        if len(gh) and "excess" in d.columns:
            vals.append((f.name, float(gh.excess.iloc[0])))
    if vals:
        print(f"{tag}: " + ", ".join(f"{n.split('_')[0]} {v:+.4f}" for n, v in vals))
        print(f"      range {min(v for _, v in vals):+.4f} to "
              f"{max(v for _, v in vals):+.4f}")
