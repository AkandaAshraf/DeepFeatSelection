"""Make saturation a number: a per-channel gate with a positive control.

Pre-registration: paper/saturation_gate_protocol.md, committed before this
was written or run.

Source blindness is a property of the saturated regime, not of the method
(Rules 71-72), and no ghost statistic catches the failure (Rules 73-74). The
quantity that would catch it - self-R2 - is directly observable. This fits a
per-channel threshold s* on a discovery grid and applies it unchanged to
held-out systems of different width, coupling and seed.

    python scripts/saturation_gate.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from boundary_map import (BATCH, BOTTLENECK, DEV, DONOR_R2, E,  # noqa: E402
                          EPOCHS, MASK, MIN_DONORS, MODELS, N_GHOSTS,
                          embed, make_system, poly3, ridge_r2)
from wormwideweb_gate import MaskedAE  # noqa: E402

OUT = Path("ExpOutput/saturation_gate")
N = 4000
DISCOVERY = dict(V=15, coupling=0.35, seeds=(0, 1, 2),
                 noise=(0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.20, 0.40))
HELDOUT = [dict(V=30, coupling=0.20, seeds=(10, 11, 12),
                noise=(0.0, 0.02, 0.05, 0.10, 0.30)),
           dict(V=30, coupling=0.50, seeds=(10, 11, 12),
                noise=(0.0, 0.02, 0.05, 0.10, 0.30))]
FP_BAR, KEEP_BAR = 0.05, 0.80
S_GRID = np.round(np.arange(0.50, 1.0, 0.01), 2)


def run_cell(V, coupling, seed, obs_noise):
    """boundary_map.run_cell with observation noise, returning per channel."""
    x, is_driven, is_source = make_system(N, V, coupling, 0, seed)
    if obs_noise:
        rng = np.random.default_rng(seed + 777)
        x = x + obs_noise * rng.standard_normal(x.shape)
    Vt = x.shape[1]
    emb = embed(x)
    m = emb.shape[0]
    a, b = int(0.6 * m), int(0.8 * m)
    tr, tr_i, te_i = slice(0, a), np.arange(0, a - 1), np.arange(b, m - 1)
    mu, sd = emb[tr].mean(0), emb[tr].std(0) + 1e-12
    zs = np.clip(np.nan_to_num((emb - mu) / sd), -20, 20).astype(np.float32)
    lead = zs[:, [j * E for j in range(Vt)]]
    feats = [poly3(zs[:, q * E:(q + 1) * E]) for q in range(Vt)]

    self_r2 = np.array([ridge_r2(f[tr_i], lead[tr_i + 1, q],
                                 f[te_i], lead[te_i + 1, q])
                        for q, f in enumerate(feats)])

    ztr, zfull = torch.as_tensor(zs[tr], device=DEV), torch.as_tensor(zs, device=DEV)
    codes = []
    for mm in range(MODELS):
        torch.manual_seed(seed * 100 + mm)
        net = MaskedAE(zs.shape[1], BOTTLENECK).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        g = torch.Generator().manual_seed(seed * 100 + mm)
        for _ in range(EPOCHS):
            perm = torch.randperm(ztr.shape[0], generator=g)
            for i in range(0, len(perm), BATCH):
                bt = ztr[perm[i:i + BATCH]]
                msk = torch.rand(bt.shape[0], Vt, device=DEV) < MASK
                mc = msk.repeat_interleave(E, dim=1)
                loss = ((net(bt.masked_fill(mc, 0.0)) - bt)[mc] ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            codes.append(net.enc(zfull).cpu().numpy())

    def excess_of(f, target):
        base = ridge_r2(f[tr_i], target[tr_i + 1], f[te_i], target[te_i + 1])
        return float(np.mean([
            ridge_r2(np.hstack([f[tr_i], c[tr_i]]), target[tr_i + 1],
                     np.hstack([f[te_i], c[te_i]]), target[te_i + 1]) - base
            for c in codes]))

    excess = np.array([excess_of(feats[q], lead[:, q]) for q in range(Vt)])

    rng = np.random.default_rng(seed + 4242)
    qual = np.where(self_r2 > DONOR_R2)[0]
    fallback = len(qual) < MIN_DONORS
    pool = np.arange(Vt) if fallback else qual
    donors = rng.choice(pool, size=min(N_GHOSTS, len(pool)), replace=False)
    ghosts = np.array([
        excess_of(poly3(np.roll(zs[:, d * E:(d + 1) * E], s_, axis=0)),
                  np.roll(lead[:, d], s_))
        for d, s_ in zip(donors, rng.integers(m // 4, 3 * m // 4, len(donors)))])
    thr = max(0.0, float(ghosts.max()))
    return dict(self_r2=self_r2, excess=excess, flagged=excess > thr,
                is_driven=is_driven, is_source=is_source,
                ghost_med=float(np.median(ghosts)),
                ghost_max=float(ghosts.max()), donor_fallback=fallback)


def sweep(cfg, tag):
    rows, chans = [], []
    for nz in cfg["noise"]:
        for s in cfg["seeds"]:
            t = time.time()
            c = run_cell(cfg["V"], cfg["coupling"], s, nz)
            src, drv, fl, sr = (c["is_source"], c["is_driven"],
                                c["flagged"], c["self_r2"])
            rows.append({
                "set": tag, "V": cfg["V"], "coupling": cfg["coupling"],
                "seed": s, "noise": nz, "secs": time.time() - t,
                "self_r2_med": float(np.median(sr)),
                "sat_at_0.9": float((sr > 0.9).mean()),
                "source_fp": float((fl & src).sum() / max(src.sum(), 1)),
                "recall": float((fl & drv).sum() / max(drv.sum(), 1)),
                "ghost_med": c["ghost_med"], "ghost_max": c["ghost_max"],
                "g3_pass": bool(c["ghost_med"] <= 0.005),
                "donor_fallback": c["donor_fallback"]})
            chans.append(pd.DataFrame({
                "set": tag, "V": cfg["V"], "coupling": cfg["coupling"],
                "seed": s, "noise": nz, "self_r2": sr, "excess": c["excess"],
                "flagged": fl, "is_source": src, "is_driven": drv}))
            r = rows[-1]
            print(f"  {tag:9s} V={cfg['V']} c={cfg['coupling']} nz={nz:<5} "
                  f"s={s}  selfR2 {r['self_r2_med']:.3f}  "
                  f"srcFP {r['source_fp']:.2f}  recall {r['recall']:.2f}  "
                  f"G3 {'pass' if r['g3_pass'] else 'FAIL'}  "
                  f"({r['secs']:.0f}s)", flush=True)
    return pd.DataFrame(rows), pd.concat(chans, ignore_index=True)


def gate_curve(ch):
    """Pooled source FP and TP-keep as a function of the per-channel bar."""
    src, drv, fl = ch.is_source.values, ch.is_driven.values, ch.flagged.values
    sr = ch.self_r2.values
    tp0 = int((fl & drv).sum())
    out = []
    for s in S_GRID:
        keep = sr >= s
        n_src = int((src & keep).sum())
        out.append({"s": s,
                    "source_fp": float((fl & src & keep).sum() / n_src)
                    if n_src else np.nan,
                    "tp_keep": float((fl & drv & keep).sum() / tp0)
                    if tp0 else np.nan,
                    "n_source_retained": n_src})
    return pd.DataFrame(out)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEV}   b={BOTTLENECK}  epochs={EPOCHS}  n={N}")
    print(f"bars: source FP <= {FP_BAR}, TP keep >= {KEEP_BAR}\n")
    t0 = time.time()

    print("DISCOVERY")
    d_sum, d_ch = sweep(DISCOVERY, "discovery")
    h_sums, h_chs = [], []
    print("\nHELD-OUT")
    for i, cfg in enumerate(HELDOUT):
        a, b = sweep(cfg, f"heldout{i+1}")
        h_sums.append(a); h_chs.append(b)
    summ = pd.concat([d_sum] + h_sums, ignore_index=True)
    summ.to_csv(OUT / "cells.csv", index=False)
    d_ch.to_csv(OUT / "channels_discovery.csv", index=False)
    pd.concat(h_chs, ignore_index=True).to_csv(
        OUT / "channels_heldout.csv", index=False)
    print(f"\n({(time.time()-t0)/60:.1f} min)\n")

    # ---- S1 ------------------------------------------------------------
    print("S1  do source false positives rise as saturation falls?")
    g = d_sum.groupby("noise").agg(selfR2=("self_r2_med", "median"),
                                   srcFP=("source_fp", "median"),
                                   recall=("recall", "median"),
                                   g3=("g3_pass", "all")).round(3)
    print("   " + g.to_string().replace("\n", "\n   "))
    s1 = g.srcFP.max() > 0.05 and g.srcFP.iloc[0] <= 0.05
    print(f"   -> {'REPRODUCES' if s1 else 'DOES NOT REPRODUCE - experiment VOID'}")
    if not s1:
        return 0

    # ---- S2, S3: fit s* on discovery only -------------------------------
    cur = gate_curve(d_ch)
    cur.to_csv(OUT / "curve_discovery.csv", index=False)
    print("\nS2  pooled discovery source FP against the per-channel bar")
    show = cur[cur.s.isin([0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99])]
    print("   " + show.to_string(index=False).replace("\n", "\n   "))
    ok = cur[(cur.source_fp <= FP_BAR) & (cur.n_source_retained > 0)]
    if ok.empty:
        print(f"   -> NO s reaches source FP <= {FP_BAR}. No gate exists.")
        return 0
    s_star = float(ok.s.min())
    row = cur[cur.s == s_star].iloc[0]
    print(f"   -> s* = {s_star:.2f}  (source FP {row.source_fp:.3f}, "
          f"TP keep {row.tp_keep:.3f})")

    print(f"\nS3  POSITIVE CONTROL: TP keep at s* must be >= {KEEP_BAR}")
    s3 = row.tp_keep >= KEEP_BAR
    print(f"   TP keep {row.tp_keep:.3f}  -> {'PASS' if s3 else 'FAIL'}")
    if not s3:
        cand = cur[(cur.source_fp <= FP_BAR) & (cur.tp_keep >= KEEP_BAR)]
        print("   no s satisfies both bars" if cand.empty else
              f"   both bars met at s = {cand.s.min():.2f}; NOT adopted, "
              "s* was declared as the smallest s meeting S2")

    # ---- S4: transfer, decisive -----------------------------------------
    print(f"\nS4  DECISIVE: does s* = {s_star:.2f} transfer, unchanged?")
    verdict = []
    for i, hc in enumerate(h_chs):
        c = gate_curve(hc)
        c.to_csv(OUT / f"curve_heldout{i+1}.csv", index=False)
        r = c[c.s == s_star].iloc[0]
        cfg = HELDOUT[i]
        okk = r.source_fp <= FP_BAR and r.tp_keep >= KEEP_BAR
        verdict.append(okk)
        pre = float((hc.flagged & hc.is_source).sum()
                    / max(int(hc.is_source.sum()), 1))
        print(f"   V={cfg['V']} c={cfg['coupling']}:  ungated source FP "
              f"{pre:.3f}  ->  gated {r.source_fp:.3f}   "
              f"TP keep {r.tp_keep:.3f}   {'PASS' if okk else 'FAIL'}")

    # ---- S5, S6 ---------------------------------------------------------
    r09 = cur[cur.s == 0.90].iloc[0]
    print(f"\nS5  the inherited 0.9 (no prediction was made): source FP "
          f"{r09.source_fp:.3f}, TP keep {r09.tp_keep:.3f}")
    print(f"   s* = {s_star:.2f}, so 0.9 is "
          f"{'above' if 0.9 > s_star else 'below' if 0.9 < s_star else 'exactly'}"
          " the fitted operating point")
    bad = d_sum[d_sum.source_fp > 0.05]
    print(f"\nS6  G3 in the cells where the gate is needed "
          f"(source FP > 0.05, {len(bad)} cells): "
          f"{int(bad.g3_pass.sum())}/{len(bad)} PASS G3")
    print(f"   donor fallback triggered in "
          f"{int(summ.donor_fallback.sum())}/{len(summ)} cells")

    print("\nVERDICT (rule fixed before running)")
    if all(verdict) and s3:
        print(f"   -> s* = {s_star:.2f} TRANSFERS. Saturation is a number: a "
              "per-channel bar\n      that removes the source false positives "
              "and keeps the detections,\n      on systems of different width, "
              "coupling and seed.")
    elif not s3:
        print("   -> NO USABLE GATE at the declared bars: the threshold that "
              "removes source\n      false positives also removes the "
              "detections. Reported as such.")
    else:
        print("   -> DOES NOT TRANSFER. s* is a property of the discovery "
              "system, not of\n      the method. Not re-fitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
