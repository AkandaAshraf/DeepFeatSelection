"""Should the null be per channel rather than one number?

Pre-registration: paper/per_channel_null_protocol.md, committed before this
was written or run.

The deployed rule collapses 30 donor ghosts to their maximum and applies that
single bar to every channel, whatever its noise, saturation, autocorrelation
or redundancy. No channel gets a p-value, so nothing controls a false
discovery rate. This makes each channel its own null: shift q's own series S
times and score each copy exactly as the real channel is scored, so the null
inherits q's own properties jointly rather than through a named covariate.

Three arms share the SAME encoders and the SAME excess array; only the
decision rule differs, so the comparison is exactly paired.

    python scripts/per_channel_null.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from boundary_map import (BATCH, DEV, DONOR_R2, E, EPOCHS, MASK,  # noqa: E402
                          MIN_DONORS, MODELS, N_GHOSTS, embed, make_system,
                          poly3, ridge_r2)
from wormwideweb_gate import MaskedAE  # noqa: E402

OUT = Path("ExpOutput/per_channel_null")
N, COUPLING, REDUNDANCY = 4000, 0.20, 0
WIDTHS = (30, 60, 120)
NOISES = (0.0, 0.05, 0.30)
SEEDS = (0, 1, 2)
S_SHIFTS = 99                  # min attainable p = 1/(1+99) = 0.01
FDR_Q = 0.10
ARM_B_QUANTILE = 30 / 31       # matches max-of-30's expected exceedance level
LOCK_AC = 0.9                  # max|ac| over lags 1..50 flags a locked channel


def max_abs_ac(v, max_lag=50):
    """Largest |autocorrelation| over lags 1..max_lag. Locked -> near 1."""
    v = np.asarray(v, float)
    v = v - v.mean()
    den = float(v @ v)
    if den <= 0:
        return 0.0
    return float(max(abs(v[:-k] @ v[k:]) / den
                     for k in range(1, min(max_lag, len(v) - 1) + 1)))


def bh_reject(p, q):
    """Benjamini-Hochberg step-up. Returns a boolean rejection mask."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    passed = p[order] <= q * (np.arange(1, m + 1) / m)
    out = np.zeros(m, bool)
    if passed.any():
        out[order[:np.nonzero(passed)[0].max() + 1]] = True
    return out


def cell(V, noise, seed):
    x, is_driven, is_source = make_system(N, V, COUPLING, REDUNDANCY, seed)
    Vt = x.shape[1]
    # P5 is a property of the GENERATOR, so it is measured on the clean
    # series. Observation noise crushes autocorrelation and would hide the
    # locked channels exactly where the diagnostic is needed.
    locked = np.array([max_abs_ac(x[:, q]) >= LOCK_AC for q in range(Vt)])
    if noise:
        x = x + noise * np.random.default_rng(seed + 777).standard_normal(
            x.shape)

    emb = embed(x)
    m = emb.shape[0]
    a, bnd = int(0.6 * m), int(0.8 * m)
    tr, tr_i, te_i = slice(0, a), np.arange(0, a - 1), np.arange(bnd, m - 1)
    mu, sd = emb[tr].mean(0), emb[tr].std(0) + 1e-12
    zs = np.clip(np.nan_to_num((emb - mu) / sd), -20, 20).astype(np.float32)
    lead = zs[:, [j * E for j in range(Vt)]]
    feats = [poly3(zs[:, q * E:(q + 1) * E]) for q in range(Vt)]

    b = 2 * V                                       # capacity law
    t0 = time.time()
    ztr, zfull = torch.as_tensor(zs[tr], device=DEV), torch.as_tensor(zs,
                                                                     device=DEV)
    codes = []
    for mm in range(MODELS):
        torch.manual_seed(seed * 100 + mm)
        net = MaskedAE(zs.shape[1], b).to(DEV)
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
    t_enc = time.time() - t0

    def excess_of(f, target):
        base = ridge_r2(f[tr_i], target[tr_i + 1], f[te_i], target[te_i + 1])
        return float(np.mean([
            ridge_r2(np.hstack([f[tr_i], c[tr_i]]), target[tr_i + 1],
                     np.hstack([f[te_i], c[te_i]]), target[te_i + 1]) - base
            for c in codes]))

    self_r2 = np.array([ridge_r2(f[tr_i], lead[tr_i + 1, q],
                                 f[te_i], lead[te_i + 1, q])
                        for q, f in enumerate(feats)])
    excess = np.array([excess_of(feats[q], lead[:, q]) for q in range(Vt)])

    # ---- arm A: the incumbent donor panel, unchanged
    t0 = time.time()
    rng = np.random.default_rng(seed + 4242)
    qual = np.where(self_r2 > DONOR_R2)[0]
    pool = np.arange(Vt) if len(qual) < MIN_DONORS else qual
    donors = rng.choice(pool, size=min(N_GHOSTS, len(pool)), replace=False)
    ghosts = np.array([
        excess_of(poly3(np.roll(zs[:, d * E:(d + 1) * E], s_, axis=0)),
                  np.roll(lead[:, d], s_))
        for d, s_ in zip(donors, rng.integers(m // 4, 3 * m // 4, len(donors)))
    ])
    thr_a = max(0.0, float(ghosts.max()))
    t_a = time.time() - t0

    # ---- arms B and C: each channel is its own null, ONE shift set shared
    t0 = time.time()
    rng_s = np.random.default_rng(seed + 9191)
    null = np.empty((Vt, S_SHIFTS))
    for q in range(Vt):
        blk = zs[:, q * E:(q + 1) * E]
        for j, s_ in enumerate(rng_s.integers(m // 4, 3 * m // 4, S_SHIFTS)):
            null[q, j] = excess_of(poly3(np.roll(blk, s_, axis=0)),
                                   np.roll(lead[:, q], s_))
    t_bc = time.time() - t0

    thr_b = np.maximum(0.0, np.quantile(null, ARM_B_QUANTILE, axis=1))
    pvals = (1 + (null >= excess[:, None]).sum(1)) / (1 + S_SHIFTS)

    np.savez_compressed(
        OUT / f"raw_V{V}_nz{noise}_s{seed}.npz", excess=excess, null=null,
        ghosts=ghosts, self_r2=self_r2, is_driven=is_driven,
        is_source=is_source, locked=locked, pvals=pvals)

    rows = []
    for arm, flagged, secs in (
            ("GLOBAL-MAX", excess > thr_a, t_a),
            ("SELF-Q", excess > thr_b, t_bc),
            ("SELF-BH", bh_reject(pvals, FDR_Q), t_bc)):
        tp = int((flagged & is_driven).sum())
        rows.append({
            "V": V, "noise": noise, "seed": seed, "arm": arm, "b": b,
            "channels": Vt,
            "recall": tp / max(int(is_driven.sum()), 1),
            "precision": tp / max(int(flagged.sum()), 1),
            "source_fp": float((flagged & is_source).sum()
                               / max(int(is_source.sum()), 1)),
            "n_flagged": int(flagged.sum()),
            "self_r2_med": float(np.median(self_r2)),
            "ghost_med": float(np.median(ghosts)),
            "encoder_secs": t_enc, "null_secs": secs,
            # P5 diagnostic: is the per-channel null inflated where locked?
            "frac_locked": float(locked.mean()),
            "null_q_locked": (float(np.median(np.quantile(
                null[locked], ARM_B_QUANTILE, axis=1))) if locked.any()
                else np.nan),
            "null_q_free": (float(np.median(np.quantile(
                null[~locked], ARM_B_QUANTILE, axis=1))) if (~locked).any()
                else np.nan),
        })
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEV}   S={S_SHIFTS} shifts/channel   b=2V   "
          f"arms: GLOBAL-MAX / SELF-Q / SELF-BH")
    print(f"{len(WIDTHS)*len(NOISES)*len(SEEDS)} cells x 3 arms\n")
    rows, t0 = [], time.time()
    for V in WIDTHS:
        for nz in NOISES:
            for s in SEEDS:
                rows += cell(V, nz, s)
                r = rows[-3:]
                print(f"  V={V:<4} nz={nz:<5} s={s}  " + "  ".join(
                    f"{x['arm'][:6]} rec {x['recall']:.2f}/srcFP "
                    f"{x['source_fp']:.2f}" for x in r)
                    + f"   ({(time.time()-t0)/60:.1f}m)", flush=True)
                pd.DataFrame(rows).to_csv(OUT / "cells.csv", index=False)
    d = pd.DataFrame(rows)
    print(f"\n({(time.time()-t0)/60:.1f} min)\n")

    order = ["GLOBAL-MAX", "SELF-Q", "SELF-BH"]
    print("SOURCE FALSE POSITIVES by noise (lower is better)")
    print("   " + d.pivot_table(index="noise", columns="arm",
                                values="source_fp")[order].round(3)
          .to_string().replace("\n", "\n   "))
    print("\nRECALL by noise")
    print("   " + d.pivot_table(index="noise", columns="arm", values="recall")
          [order].round(3).to_string().replace("\n", "\n   "))
    print("\nPRECISION by noise")
    print("   " + d.pivot_table(index="noise", columns="arm",
                                values="precision")[order].round(3)
          .to_string().replace("\n", "\n   "))

    hard = d[d.noise == 0.30].groupby("arm").source_fp.mean()
    easy = d[d.noise == 0.0].groupby("arm").recall.mean()
    prec0 = d[d.noise == 0.0].groupby("arm").precision.min()

    p1 = bool(hard["SELF-Q"] < hard["GLOBAL-MAX"]
              and hard["SELF-BH"] < hard["GLOBAL-MAX"])
    p2 = bool(easy["SELF-Q"] >= easy["GLOBAL-MAX"] - 0.05)
    p4 = bool(prec0.min() >= 0.95)

    print(f"\nP1 DECISIVE at noise 0.30, mean source FP: "
          f"GLOBAL-MAX {hard['GLOBAL-MAX']:.3f}  SELF-Q {hard['SELF-Q']:.3f}"
          f"  SELF-BH {hard['SELF-BH']:.3f}   -> {'HOLDS' if p1 else 'FAILS'}")
    print(f"P2 DECISIVE at noise 0.0, mean recall: "
          f"GLOBAL-MAX {easy['GLOBAL-MAX']:.3f}  SELF-Q {easy['SELF-Q']:.3f}"
          f"   -> {'HOLDS' if p2 else 'FAILS'}")
    print(f"P4 DISQUALIFYING min precision at noise 0.0: {prec0.min():.3f}"
          f"   -> {'HOLDS' if p4 else 'FAILS'}")

    print("\nP5 PHASE-LOCK DIAGNOSTIC (threat to validity, declared)")
    print(f"   locked fraction, median over cells: {d.frac_locked.median():.3f}")
    sub = d[d.arm == "SELF-Q"].dropna(subset=["null_q_locked", "null_q_free"])
    if len(sub):
        print(f"   per-channel null bar, locked   {sub.null_q_locked.median():+.5f}")
        print(f"   per-channel null bar, unlocked {sub.null_q_free.median():+.5f}")
        infl = sub.null_q_locked.median() - sub.null_q_free.median()
        print(f"   inflation on locked channels: {infl:+.5f}"
              + ("   <- self-null INVALID for locked channels"
                 if infl > 0.005 else "   (below the 0.005 ghost bar)"))

    print("\nP6 cost")
    for arm, g in d.groupby("arm"):
        print(f"   {arm:11s} encoder {g.encoder_secs.median():6.1f}s   "
              f"null {g.null_secs.median():7.1f}s")

    print("\nVERDICT (rule fixed before running)")
    if not p4:
        print("   -> REJECT. Precision broke at noise 0.0; the global bar stands.")
    elif not p1:
        print("   -> NULL. The per-channel null does not repair cross-regime\n"
              "      miscalibration. Incumbent stands.")
    elif not p2:
        print("   -> P1 holds but P2 fails: the calibration costs recall in the\n"
              "      easy regime. Reported as a trade, not an adoption.")
    else:
        best = "SELF-Q" if hard["SELF-Q"] <= hard["SELF-BH"] else "SELF-BH"
        print(f"   -> ADOPT {best}, subject to the P5 diagnostic above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
