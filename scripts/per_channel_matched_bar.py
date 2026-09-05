"""POST HOC, labelled as such. The adversarial control the protocol lacked.

SELF-Q flags more channels than GLOBAL-MAX. If its gain is only that its bar
sits lower, then lowering the GLOBAL bar to flag the SAME NUMBER of channels
must reproduce it. If SELF-Q still wins at a matched flag count, the gain is
per-channel structure, not bar height.
"""
import glob, os, re
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

rows = []
for f in sorted(glob.glob("ExpOutput/per_channel_null/raw_*.npz")):
    m = re.search(r"raw_V(\d+)_nz([\d.]+)_s(\d+)", os.path.basename(f))
    V, nz, seed = int(m[1]), float(m[2]), int(m[3])
    z = np.load(f)
    ex, null, dr, sr = z["excess"], z["null"], z["is_driven"], z["is_source"]
    thr_b = np.maximum(0.0, np.quantile(null, 30/31, axis=1))
    fq = ex > thr_b
    k = int(fq.sum())
    if k == 0:
        continue
    # GLOBAL bar lowered until it flags exactly k channels: top-k by excess
    fg = np.zeros_like(fq)
    fg[np.argsort(-ex)[:k]] = True
    def sc(fl):
        tp = int((fl & dr).sum())
        return (tp / max(int(dr.sum()), 1),
                float((fl & sr).sum() / max(int(sr.sum()), 1)))
    rq, sq = sc(fq)
    rg, sg = sc(fg)
    rows.append(dict(V=V, noise=nz, seed=seed, k=k,
                     rec_selfq=rq, rec_topk=rg, src_selfq=sq, src_topk=sg))
t = pd.DataFrame(rows)
print(__doc__)
print(f"{len(t)} cells, flag count matched exactly in each.\n")
print("MATCHED-COUNT COMPARISON (same number flagged; only WHICH differs)")
g = t.groupby("noise").agg(rec_selfq=("rec_selfq","mean"), rec_topk=("rec_topk","mean"),
                           src_selfq=("src_selfq","mean"), src_topk=("src_topk","mean"))
print("   " + g.round(3).to_string().replace("\n","\n   "))
print()
w = (t.rec_selfq > t.rec_topk).sum(); l = (t.rec_selfq < t.rec_topk).sum()
print(f"   recall:    SELF-Q better in {w}, worse in {l}, tied in {len(t)-w-l}")
w2 = (t.src_selfq < t.src_topk).sum(); l2 = (t.src_selfq > t.src_topk).sum()
print(f"   source FP: SELF-Q better in {w2}, worse in {l2}, tied in {len(t)-w2-l2}")
for k_, a, b in (("recall","rec_selfq","rec_topk"), ("source_fp","src_selfq","src_topk")):
    d_ = t[a] - t[b]; nz_ = d_ != 0
    if nz_.sum() > 5:
        print(f"   Wilcoxon {k_:10s} n={nz_.sum():2d}  p={wilcoxon(t[a][nz_], t[b][nz_]).pvalue:.4f}")
print()
print("READ AGAINST OUR OWN INTEREST: if these are indistinguishable, the")
print("per-channel null is an expensive way to lower a threshold, and the")
print("58x null cost (78.4s vs 0.3s) buys ranking that top-k already had.")
