"""A forecasting U-Net with per-variable skips, ablated one input at a time.

Pre-registration: paper/unet_loo_protocol.md, committed before this was
written or run. EXPLORATORY -- nothing is adopted from this run.

The readout is leave-one-out on a trained model, which this project closed
twice (Mechanism 1's collapse to AUROC 0.500 on three worms; Rule 63's
rejection of a leave-one-out dual). Both are named in the protocol. What is
new is the per-variable skip: if variable k's own history reaches k's own
output directly, the bottleneck never carries it, so ablating j can damage k
only through a genuine cross-variable path. That is the conditioning the
rejected version lacked.

Masking is an ARM, not an assumption: training with random input removal is
what teaches a model to route around a missing input.

Primary metric is AVERAGE PRECISION with sources positive (base rate 5/30 =
0.167), not AUROC, because the false-positive rate of an imbalanced problem
is diluted by the large negative class.

    python scripts/unet_loo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from boundary_map import DEV, E, embed, make_system, poly3, ridge_r2  # noqa: E402

OUT = Path("ExpOutput/unet_loo")
N, V, COUPLING, REDUNDANCY = 4000, 30, 0.20, 0
NOISES = (0.0, 0.05)
SEEDS = (0, 1, 2)
HID, BOT = 128, 60           # BOT = 2V, the capacity law
STEPS, EVAL_EVERY, LR = 800, 20, 3e-3
MASK_P = 0.25                # matches the deployed masked autoencoder


class ForecastUNet(nn.Module):
    """State at t -> state at t+1 for every variable at once.

    With `skip`, each variable's own lags reach its own output through a
    private path, so the bottleneck is free to carry only cross-variable
    information. Without it, every output depends on the bottleneck alone.
    """

    def __init__(self, v, e, hid=HID, bot=BOT, skip=True):
        super().__init__()
        self.v, self.e, self.skip = v, e, skip
        self.enc = nn.Sequential(nn.Linear(v * e, hid), nn.Tanh(),
                                 nn.Linear(hid, bot))
        self.dec = nn.Sequential(nn.Linear(bot, hid), nn.Tanh(),
                                 nn.Linear(hid, v))
        if skip:
            # one private 2-layer path per variable, applied batched
            self.s1 = nn.Parameter(torch.randn(v, e, 8) * 0.3)
            self.s1b = nn.Parameter(torch.zeros(v, 1, 8))
            self.s2 = nn.Parameter(torch.randn(v, 8, 1) * 0.3)
            self.s2b = nn.Parameter(torch.zeros(v, 1, 1))

    def forward(self, z):
        out = self.dec(self.enc(z))
        if not self.skip:
            return out
        # z is (n, v*e) -> (v, n, e) so each variable gets its own weights
        zz = z.view(-1, self.v, self.e).permute(1, 0, 2)
        h = torch.tanh(torch.baddbmm(self.s1b, zz, self.s1))
        own = torch.baddbmm(self.s2b, h, self.s2).squeeze(-1).T
        return out + own


def r2_cols(pred, y):
    """Per-column R2, as numpy."""
    ss = ((y - pred) ** 2).sum(0)
    tot = ((y - y.mean(0)) ** 2).sum(0)
    return np.where(tot > 0, 1.0 - ss / np.maximum(tot, 1e-12), 0.0)


def train_unet(ztr, ytr, zva, yva, v, skip, masked, seed):
    torch.manual_seed(seed)
    net = ForecastUNet(v, E, skip=skip).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    g = torch.Generator(device=DEV if DEV == "cpu" else "cpu")
    g.manual_seed(seed)
    best, bstate = -1e9, None
    for st in range(STEPS):
        inp = ztr
        if masked:
            m = (torch.rand(ztr.shape[0], v, generator=g).to(DEV) < MASK_P)
            inp = ztr.masked_fill(m.repeat_interleave(E, dim=1), 0.0)
        loss = ((net(inp) - ytr) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if st % EVAL_EVERY == 0 or st == STEPS - 1:
            with torch.no_grad():
                rv = -float(((net(zva) - yva) ** 2).mean())
            if rv > best:
                best = rv
                bstate = {k: t.clone() for k, t in net.state_dict().items()}
    if bstate is not None:
        net.load_state_dict(bstate)
    net.eval()
    return net


def cell(noise, seed):
    x, is_driven, is_source = make_system(N, V, COUPLING, REDUNDANCY, seed)
    if noise:
        x = x + noise * np.random.default_rng(seed + 777).standard_normal(
            x.shape)
    Vt = x.shape[1]
    emb = embed(x)
    m = emb.shape[0]
    a, bnd = int(0.6 * m), int(0.8 * m)
    tr = slice(0, a)
    tr_i, va_i, te_i = (np.arange(0, a - 1), np.arange(a, bnd - 1),
                        np.arange(bnd, m - 1))
    mu, sd = emb[tr].mean(0), emb[tr].std(0) + 1e-12
    zs = np.clip(np.nan_to_num((emb - mu) / sd), -20, 20).astype(np.float32)
    lead = zs[:, [j * E for j in range(Vt)]].astype(np.float32)

    T = lambda v_: torch.as_tensor(v_, device=DEV)  # noqa: E731
    ztr, zva, zte = T(zs[tr_i]), T(zs[va_i]), T(zs[te_i])
    ytr, yva = T(lead[tr_i + 1]), T(lead[va_i + 1])
    yte_np = lead[te_i + 1]

    rows = []

    # ---- ADD reference: conditional additive outflow, pairwise ridge
    t0 = time.time()
    own = [poly3(zs[:, k * E:(k + 1) * E]) for k in range(Vt)]
    base = np.array([ridge_r2(own[k][tr_i], lead[tr_i + 1, k],
                              own[k][te_i], lead[te_i + 1, k])
                     for k in range(Vt)])
    add = np.zeros(Vt)
    for j in range(Vt):
        gains = []
        for k in range(Vt):
            if k == j:
                continue
            f_tr = np.hstack([own[k][tr_i], own[j][tr_i]])
            f_te = np.hstack([own[k][te_i], own[j][te_i]])
            gains.append(ridge_r2(f_tr, lead[tr_i + 1, k],
                                  f_te, lead[te_i + 1, k]) - base[k])
        add[j] = float(np.mean(gains))
    rows.append(("ADD", add, np.nan, time.time() - t0))

    # ---- four U-Net arms
    for skip in (True, False):
        for masked in (True, False):
            tag = f"{'MASK' if masked else 'NOMASK'}-{'SKIP' if skip else 'NOSKIP'}"
            t0 = time.time()
            net = train_unet(ztr, ytr, zva, yva, Vt, skip, masked,
                             seed * 100 + int(skip) * 2 + int(masked))
            with torch.no_grad():
                full = net(zte).cpu().numpy()
            r2_full = r2_cols(full, yte_np)
            score = np.zeros(Vt)
            for j in range(Vt):
                with torch.no_grad():
                    zab = zte.clone()
                    zab[:, j * E:(j + 1) * E] = 0.0
                    abl = net(zab).cpu().numpy()
                dmg = r2_full - r2_cols(abl, yte_np)
                score[j] = float(np.mean(np.delete(dmg, j)))   # k != j
            rows.append((tag, score, float(np.mean(r2_full)), time.time() - t0))

    out = []
    for tag, score, r2m, secs in rows:
        out.append(dict(
            noise=noise, seed=seed, arm=tag,
            avg_precision=float(average_precision_score(is_source, score)),
            auroc=float(roc_auc_score(is_source, score)),
            base_rate=float(is_source.mean()),
            src_mean=float(score[is_source].mean()),
            drv_mean=float(score[is_driven].mean()),
            model_r2=r2m, secs=secs))
    return out, rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEV}   V={V}  bottleneck={BOT}   EXPLORATORY, no adoption")
    print(f"primary metric: average precision, base rate {1/6:.3f}\n")
    recs, t0 = [], time.time()
    for nz in NOISES:
        for s in SEEDS:
            r, raw = cell(nz, s)
            recs += r
            np.savez_compressed(OUT / f"raw_nz{nz}_s{s}.npz",
                                **{t: sc for t, sc, _, _ in raw})
            print(f"  noise={nz:<5} seed={s}  " + "  ".join(
                f"{d['arm'][:11]} AP {d['avg_precision']:.2f}" for d in r)
                + f"   ({(time.time()-t0)/60:.1f}m)", flush=True)
            pd.DataFrame(recs).to_csv(OUT / "cells.csv", index=False)
    d = pd.DataFrame(recs)
    print(f"\n({(time.time()-t0)/60:.1f} min)\n")

    ARMS = ["ADD", "MASK-SKIP", "NOMASK-SKIP", "MASK-NOSKIP", "NOMASK-NOSKIP"]
    br = float(d.base_rate.iloc[0])
    print(f"AVERAGE PRECISION   (chance = base rate = {br:.3f}; "
          f"bar = 2x = {2*br:.3f})")
    ap = d.pivot_table(index="noise", columns="arm", values="avg_precision")
    print("   " + ap[ARMS].round(3).to_string().replace("\n", "\n   "))
    print("\nLIFT over base rate")
    print("   " + (ap[ARMS] / br).round(2).to_string().replace("\n", "\n   "))
    print("\nAUROC (secondary, for comparability only)")
    print("   " + d.pivot_table(index="noise", columns="arm", values="auroc")
          [ARMS].round(3).to_string().replace("\n", "\n   "))
    print("\nU5 GUARD: model held-out R2 on its own forecasting task")
    print("   " + d[d.arm != "ADD"].pivot_table(
        index="noise", columns="arm", values="model_r2").round(3)
        .to_string().replace("\n", "\n   "))

    z = d[d.noise == 0.0].groupby("arm")
    apz, r2z = z.avg_precision.mean(), z.model_r2.mean()
    bar = 2 * br
    u4 = bool(apz["ADD"] > bar)
    u5 = bool(r2z.drop("ADD").min() > 0.5)
    unets = [a for a in ARMS if a != "ADD"]
    u1 = bool(max(apz[a] for a in unets) > bar)
    u2 = bool(apz["NOMASK-SKIP"] > apz["MASK-SKIP"]
              and apz["NOMASK-NOSKIP"] > apz["MASK-NOSKIP"])
    u3 = bool(apz["MASK-SKIP"] > apz["MASK-NOSKIP"]
              and apz["NOMASK-SKIP"] > apz["NOMASK-NOSKIP"])

    print(f"\nU4 GUARD  ADD reference AP {apz['ADD']:.3f} vs bar {bar:.3f}"
          f"   -> {'HOLDS' if u4 else 'FAILS'}")
    print(f"U5 GUARD  min U-Net forecast R2 {r2z.drop('ADD').min():.3f}"
          f"   -> {'HOLDS' if u5 else 'FAILS'}")
    print(f"U1 DECISIVE  best U-Net AP {max(apz[a] for a in unets):.3f} vs "
          f"bar {bar:.3f}   -> {'HOLDS' if u1 else 'FAILS'}")
    print(f"U2 DECISIVE  NOMASK beats MASK   -> {'HOLDS' if u2 else 'FAILS'}")
    print(f"   SKIP   {apz['MASK-SKIP']:.3f} -> {apz['NOMASK-SKIP']:.3f}")
    print(f"   NOSKIP {apz['MASK-NOSKIP']:.3f} -> {apz['NOMASK-NOSKIP']:.3f}")
    print(f"U3 DIRECTIONAL  SKIP beats NOSKIP -> {'HOLDS' if u3 else 'FAILS'}")

    print("\nVERDICT (rule fixed before running)")
    if not (u4 and u5):
        print("   -> UNINFORMATIVE. A guard failed: the instrument, not the")
        print("      hypothesis, is what was measured.")
    elif u1:
        print("   -> OPEN. Leave-one-out clears twice the base rate at this")
        print("      architecture. Licenses a powered pre-registration only;")
        print("      nothing is adopted and nothing enters the manuscript.")
    else:
        print("   -> CLOSED. Leave-one-out is dead at this architecture too,")
        print("      the skip does not rescue it, and the line closes with a")
        print("      third independent measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
