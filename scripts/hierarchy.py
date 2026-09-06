"""Nested codes: at which scale does a variable's drive live?

Pre-registration: paper/hierarchy_protocol.md, committed before this was
written or run. EXPLORATORY -- nothing is adopted from this run.

The flat merge of 2026-09-05 was rejected because one shared decoder
allocating capacity across every output reported where it spent that capacity
rather than what each target gained. Here only the CODES are nested; every
level keeps its own per-target ridge readout, so Rule 127 holds by
construction.

  level 1  own lags of q
  level 2  code over the MODULE containing q, with q's columns zeroed
  level 3  the global code over all V, as the incumbent computes it

  excess_2 = R2[own+module] - R2[own]
  excess_3 = R2[own+module+system] - R2[own+module]

Module-level exclusion is the only asymmetry against FLAT, because q is about
a sixth of its own module code but only a thirtieth of the system code, and
system-level exclusion was measured to change nothing.

    python scripts/hierarchy.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from boundary_map import (BATCH, DEV, E, EPOCHS, MASK, embed,  # noqa: E402
                          make_system, poly3, ridge_r2)
from wormwideweb_gate import MaskedAE  # noqa: E402

OUT = Path("ExpOutput/hierarchy")
N, COUPLING = 4000, 0.20
WIDTHS = (30, 60)
NOISES = (0.0, 0.05)
SEEDS = (0, 1, 2)
MOD_EPOCHS = 12          # module encoders are small; fewer passes suffice


def train_code(zs, tr, d_in, b, seed, epochs=EPOCHS):
    """One masked autoencoder, returning the net and full-length codes."""
    v = d_in // E
    torch.manual_seed(seed)
    net = MaskedAE(d_in, b).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(seed)
    ztr = torch.as_tensor(zs[tr], device=DEV)
    for _ in range(epochs):
        perm = torch.randperm(ztr.shape[0], generator=g)
        for i in range(0, len(perm), BATCH):
            bt = ztr[perm[i:i + BATCH]]
            msk = torch.rand(bt.shape[0], v, device=DEV) < MASK
            mc = msk.repeat_interleave(E, dim=1)
            loss = ((net(bt.masked_fill(mc, 0.0)) - bt)[mc] ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return net, net.enc(torch.as_tensor(zs, device=DEV)).cpu().numpy()


def modules_for(arm, x_raw, Vt, m, parent, n_src, rng):
    """Each variable's module id. Only HIER-TRUE may see ground truth."""
    if arm == "HIER-RAND":
        lab = np.arange(Vt) % m
        rng.shuffle(lab)
        return lab
    if arm == "HIER-TRUE":
        lab = np.empty(Vt, int)
        lab[:n_src] = np.arange(n_src) % m
        lab[n_src:] = [lab[p] for p in parent]
        return lab
    d = np.diff(x_raw, axis=0)
    c = np.nan_to_num(np.corrcoef(d.T), nan=0.0)
    return AgglomerativeClustering(
        n_clusters=m, metric="precomputed", linkage="average"
    ).fit_predict(1.0 - np.abs(c))


def cell(V, noise, seed):
    x, is_driven, is_source = make_system(N, V, COUPLING, 0, seed)
    Vt = x.shape[1]
    n_src = max(3, V // 6)
    rng_g = np.random.default_rng(seed)
    _ = rng_g.uniform(0.2, 0.8, V)
    _ = rng_g.uniform(3.6, 3.9, V)
    parent = rng_g.integers(0, n_src, V - n_src)   # replayed ground truth
    if noise:
        x = x + noise * np.random.default_rng(seed + 777).standard_normal(
            x.shape)

    emb = embed(x)
    mrows = emb.shape[0]
    a, bnd = int(0.6 * mrows), int(0.8 * mrows)
    tr = slice(0, a)
    tr_i, te_i = np.arange(0, a - 1), np.arange(bnd, mrows - 1)
    mu, sd = emb[tr].mean(0), emb[tr].std(0) + 1e-12
    zs = np.clip(np.nan_to_num((emb - mu) / sd), -20, 20).astype(np.float32)
    lead = zs[:, [j * E for j in range(Vt)]]
    own = [poly3(zs[:, q * E:(q + 1) * E]) for q in range(Vt)]
    base = np.array([ridge_r2(own[q][tr_i], lead[tr_i + 1, q],
                              own[q][te_i], lead[te_i + 1, q])
                     for q in range(Vt)])

    # ---- system code, shared by FLAT and every hierarchical arm
    t0 = time.time()
    _, sys_code = train_code(zs, tr, zs.shape[1], 2 * V, seed * 100)
    t_sys = time.time() - t0

    flat = np.array([
        ridge_r2(np.hstack([own[q][tr_i], sys_code[tr_i]]), lead[tr_i + 1, q],
                 np.hstack([own[q][te_i], sys_code[te_i]]), lead[te_i + 1, q])
        - base[q] for q in range(Vt)])

    rows = [dict(V=V, noise=noise, seed=seed, arm="FLAT",
                 ap_source=float(average_precision_score(is_source, -flat)),
                 ap_driven=float(average_precision_score(is_driven, flat)),
                 loc_auroc=np.nan, loc_ap=np.nan, loc_base=np.nan,
                 loc_lift=np.nan, n_pos=0, n_neg=0, mod_share=np.nan,
                 mean_e2_driven=np.nan, mean_e3_driven=np.nan, secs=t_sys)]
    raw = {"FLAT": flat, "is_driven": is_driven, "is_source": is_source,
           "parent": parent}

    m = max(2, V // 6)
    for arm in ("HIER-CLUST", "HIER-RAND", "HIER-TRUE"):
        t0 = time.time()
        lab = modules_for(arm, x, Vt, m, parent, n_src,
                          np.random.default_rng(seed + 31))
        e2 = np.zeros(Vt)
        e3 = np.zeros(Vt)
        for mod in range(m):
            members = np.where(lab == mod)[0]
            if len(members) == 0:
                continue
            cols = np.concatenate([np.arange(j * E, (j + 1) * E)
                                   for j in members])
            zsm = zs[:, cols]
            net, _ = train_code(zsm, tr, zsm.shape[1],
                                max(4, 2 * len(members)),
                                seed * 1000 + mod, epochs=MOD_EPOCHS)
            for q in members:
                zq = zsm.copy()               # zero q before its own module
                loc = int(np.where(members == q)[0][0])
                zq[:, loc * E:(loc + 1) * E] = 0.0
                with torch.no_grad():
                    cq = net.enc(
                        torch.as_tensor(zq, device=DEV)).cpu().numpy()
                r_om = ridge_r2(
                    np.hstack([own[q][tr_i], cq[tr_i]]), lead[tr_i + 1, q],
                    np.hstack([own[q][te_i], cq[te_i]]), lead[te_i + 1, q])
                r_oms = ridge_r2(
                    np.hstack([own[q][tr_i], cq[tr_i], sys_code[tr_i]]),
                    lead[tr_i + 1, q],
                    np.hstack([own[q][te_i], cq[te_i], sys_code[te_i]]),
                    lead[te_i + 1, q])
                e2[q] = r_om - base[q]
                e3[q] = r_oms - r_om
        total = e2 + e3
        drv = np.where(is_driven)[0]
        same = np.array([lab[q] == lab[parent[q - n_src]] for q in drv])
        loc_base = float(same.mean())
        if 0 < same.mean() < 1:
            sc = (e2 - e3)[drv]
            loc_auroc = float(roc_auc_score(same, sc))   # PRIMARY: chance 0.5
            loc_ap = float(average_precision_score(same, sc))
            loc_lift = loc_ap / loc_base
        else:
            # the oracle puts every parent in-module: no negatives to score
            loc_auroc = loc_ap = loc_lift = np.nan
        tot2, tot3 = float(e2[is_driven].mean()), float(e3[is_driven].mean())
        denom = tot2 + tot3
        mod_share = float(tot2 / denom) if denom > 0 else np.nan
        rows.append(dict(
            V=V, noise=noise, seed=seed, arm=arm,
            ap_source=float(average_precision_score(is_source, -total)),
            ap_driven=float(average_precision_score(is_driven, total)),
            loc_auroc=loc_auroc, loc_ap=loc_ap, loc_base=loc_base,
            loc_lift=loc_lift, n_pos=int(same.sum()),
            n_neg=int((~same).sum()), mod_share=mod_share,
            mean_e2_driven=tot2, mean_e3_driven=tot3,
            secs=time.time() - t0))
        raw[arm + "_e2"] = e2
        raw[arm + "_e3"] = e3
        raw[arm + "_lab"] = lab
    np.savez_compressed(OUT / f"raw_V{V}_nz{noise}_s{seed}.npz", **raw)
    return rows


ARMS = ["FLAT", "HIER-CLUST", "HIER-RAND", "HIER-TRUE"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEV}   EXPLORATORY, no adoption")
    print(f"detection: average precision on the MINORITY class, "
          f"base rate {1/6:.3f}\n")
    recs, t0 = [], time.time()
    for V in WIDTHS:
        for nz in NOISES:
            for s in SEEDS:
                recs += cell(V, nz, s)
                print(f"  V={V} noise={nz:<5} seed={s}   "
                      f"({(time.time()-t0)/60:.1f}m)", flush=True)
                pd.DataFrame(recs).to_csv(OUT / "cells.csv", index=False)
    d = pd.DataFrame(recs)
    print(f"\n({(time.time()-t0)/60:.1f} min)\n")

    print("DETECTION: average precision, sources positive (chance 0.167)")
    print("   " + d.pivot_table(index=["V", "noise"], columns="arm",
                                values="ap_source")[ARMS].round(3)
          .to_string().replace("\n", "\n   "))

    h = d[d.arm != "FLAT"]
    print("\nLOCALISATION: parent is in my module, among driven channels")
    print("   per-cell base rate (varies by arm, not assumed):")
    print("   " + h.pivot_table(index=["V", "noise"], columns="arm",
                                values="loc_base").round(3)
          .to_string().replace("\n", "\n   "))
    print("\n   LIFT over that base rate (bar = 1.5x):")
    print("   " + h.pivot_table(index=["V", "noise"], columns="arm",
                                values="loc_lift").round(2)
          .to_string().replace("\n", "\n   "))

    print("\nH5 GUARD per arm: mean module-level excess on driven channels")
    print("   " + h.pivot_table(index="arm", values="mean_e2_driven")
          .round(5).to_string().replace("\n", "\n   "))
    print("\nH6 descriptive: system-level excess once the module is present")
    print("   " + h.pivot_table(index="arm", values="mean_e3_driven")
          .round(5).to_string().replace("\n", "\n   "))

    ap = d.groupby("arm").ap_source.mean()
    lift = h.groupby("arm").loc_auroc.mean()
    share = h.groupby("arm").mod_share.mean()
    e2m = h.groupby("arm").mean_e2_driven.mean()
    h1 = bool(ap["HIER-CLUST"] >= ap["FLAT"] - 0.05)
    h5 = {a: bool(e2m[a] > 0) for a in ARMS[1:]}
    h2 = bool(lift["HIER-CLUST"] >= 0.65)
    h3 = bool(lift["HIER-RAND"] >= 0.65)
    h4 = bool(share["HIER-RAND"] < share["HIER-CLUST"] < share["HIER-TRUE"])

    print(f"\nH1 DISQUALIFYING  HIER-CLUST {ap['HIER-CLUST']:.3f} vs FLAT "
          f"{ap['FLAT']:.3f} (tol 0.05)   -> {'HOLDS' if h1 else 'FAILS'}")
    print("H5 GUARD per arm  module level informative: "
          + "  ".join(f"{a}={'ok' if v else 'INERT'}" for a, v in h5.items()))
    print(f"H2 DECISIVE       HIER-CLUST localisation AUROC "
          f"{lift['HIER-CLUST']:.3f} vs bar 0.65   "
          f"-> {'HOLDS' if h2 else 'FAILS'}")
    print(f"H3 CONTROL        HIER-RAND AUROC {lift['HIER-RAND']:.3f}   -> "
          + ("ALSO CLEARS: H2 is an artefact" if h3 else "stays below: good"))
    print("MODULE SHARE      predicted ordering RAND < CLUST < TRUE: "
          f"{share['HIER-RAND']:.3f} / {share['HIER-CLUST']:.3f} / "
          f"{share['HIER-TRUE']:.3f}  -> {'HOLDS' if h4 else 'FAILS'}")

    print("\nVERDICT (rule fixed before running)")
    if not h1:
        print("   -> REJECTED. The hierarchy costs detection.")
    elif h2 and h3:
        print("   -> EMPTY. Random modules localise as well as clustered")
        print("      ones, so the split produces a number that does not")
        print("      track structure. The localisation claim is refused.")
    elif not h2 and not h4:
        print("   -> CLOSED. Localisation fails and the module share does not")
        print("      even order with module quality. The line closes.")
    elif h2:
        print("   -> OPEN. Licenses one powered pre-registration with a")
        print("      cluster count chosen without knowing the answer.")
        print("      Nothing enters the manuscript.")
    else:
        print("   -> CLOSED for the deployable arm; the oracle clears, so")
        print("      the limit is clustering rather than the idea.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
