"""One model, both statistics, and the redundancy axis that should kill it.

Pre-registration: paper/unified_protocol.md, committed before this was
written or run. EXPLORATORY -- nothing is adopted from this run.

The ResNet shortcut and the U-Net skip are the same construction at different
scales, and the masked autoencoder is the encoder inside both. So:

  stage 1  per-variable skip paths ONLY, an own-history forecaster for every
           variable, then FROZEN. The ResNet run established that a frozen
           two-stage fit is what stops the optimiser underfitting the skips
           and letting the bottleneck manufacture inflow on autonomous
           channels.
  stage 2  bottleneck and decoder, masked training, added to frozen skips.

  inflow(q)  = R2_q(full) - R2_q(skip alone)
  outflow(j) = mean over k != j of [ R2_k(full) - R2_k(input j zeroed) ]

Both statistics are scored on the MINORITY class (sources, base rate 0.167).
Driven-positive average precision has a base rate of 0.833 and a lift ceiling
of 1.2x, so it cannot demonstrate discrimination and is secondary only.

    python scripts/unified.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from boundary_map import (BATCH, DEV, E, EPOCHS, MASK, embed,  # noqa: E402
                          make_system, poly3, ridge_r2)
from wormwideweb_gate import MaskedAE  # noqa: E402

OUT = Path("ExpOutput/unified")
N, V, COUPLING = 4000, 30, 0.20
NOISES = (0.0, 0.05)
REDUNDANCIES = (0, 2)
SEEDS = (0, 1, 2)
HID, BOT = 128, 2 * V
S1_STEPS, S2_STEPS, EVAL_EVERY, LR = 600, 800, 20, 3e-3
MASK_P = 0.25


class SkipStack(nn.Module):
    """One private two-layer path per variable, applied batched."""

    def __init__(self, v, e, h=8):
        super().__init__()
        self.v, self.e = v, e
        self.w1 = nn.Parameter(torch.randn(v, e, h) * 0.3)
        self.b1 = nn.Parameter(torch.zeros(v, 1, h))
        self.w2 = nn.Parameter(torch.randn(v, h, 1) * 0.3)
        self.b2 = nn.Parameter(torch.zeros(v, 1, 1))

    def forward(self, z):
        zz = z.view(-1, self.v, self.e).permute(1, 0, 2)
        h = torch.tanh(torch.baddbmm(self.b1, zz, self.w1))
        return torch.baddbmm(self.b2, h, self.w2).squeeze(-1).T


class Bottleneck(nn.Module):
    def __init__(self, d_in, v, hid=HID, bot=BOT):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, hid), nn.Tanh(),
                                 nn.Linear(hid, bot))
        self.dec = nn.Sequential(nn.Linear(bot, hid), nn.Tanh(),
                                 nn.Linear(hid, v))

    def forward(self, z):
        return self.dec(self.enc(z))


def r2_cols(pred, y):
    ss = ((y - pred) ** 2).sum(0)
    tot = ((y - y.mean(0)) ** 2).sum(0)
    return np.where(tot > 0, 1.0 - ss / np.maximum(tot, 1e-12), 0.0)


def fit(module, fwd, ztr, ytr, zva, yva, steps, seed, v=None, masked=False):
    """Full-batch Adam, early stopped on the reserved validation segment."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(module.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed)
    best, bstate = -1e9, None
    for st in range(steps):
        inp = ztr
        if masked:
            m = (torch.rand(ztr.shape[0], v, generator=g) < MASK_P).to(DEV)
            inp = ztr.masked_fill(m.repeat_interleave(E, dim=1), 0.0)
        loss = ((fwd(inp) - ytr) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if st % EVAL_EVERY == 0 or st == steps - 1:
            with torch.no_grad():
                rv = -float(((fwd(zva) - yva) ** 2).mean())
            if rv > best:
                best = rv
                bstate = {k: t.clone() for k, t in module.state_dict().items()}
    if bstate is not None:
        module.load_state_dict(bstate)
    module.eval()


def cell(noise, red, seed):
    x, is_driven, is_source = make_system(N, V, COUPLING, red, seed)
    Vt = x.shape[1]
    n_src = max(3, V // 6)
    # replay the generator's draws to recover the parent map (Rule 125 control)
    rng = np.random.default_rng(seed)
    _ = rng.uniform(0.2, 0.8, V)
    _ = rng.uniform(3.6, 3.9, V)
    parent = rng.integers(0, n_src, V - n_src)
    n_children = np.zeros(Vt)
    for s in range(n_src):
        n_children[s] = int((parent == s).sum())
    if noise:
        x = x + noise * np.random.default_rng(seed + 777).standard_normal(
            x.shape)

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
    yte = lead[te_i + 1]

    # ================= UNIFIED: stage 1 frozen, then stage 2
    t0 = time.time()
    skip = SkipStack(Vt, E).to(DEV)
    fit(skip, skip, ztr, ytr, zva, yva, S1_STEPS, seed * 7)
    for p in skip.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        s_tr, s_va, s_te = skip(ztr), skip(zva), skip(zte)
        r2_skip = r2_cols(s_te.cpu().numpy(), yte)

    bott = Bottleneck(zs.shape[1], Vt).to(DEV)

    def full_tr(inp):
        return skip(inp) + bott(inp)

    fit(bott, full_tr, ztr, ytr, zva, yva, S2_STEPS, seed * 7 + 1,
        v=Vt, masked=True)
    with torch.no_grad():
        full = (skip(zte) + bott(zte)).cpu().numpy()
    r2_full = r2_cols(full, yte)
    inflow_u = r2_full - r2_skip
    outflow_u = np.zeros(Vt)
    for j in range(Vt):
        with torch.no_grad():
            zab = zte.clone()
            zab[:, j * E:(j + 1) * E] = 0.0
            abl = (skip(zab) + bott(zab)).cpu().numpy()
        dmg = r2_full - r2_cols(abl, yte)
        outflow_u[j] = float(np.mean(np.delete(dmg, j)))
    t_uni = time.time() - t0

    # ================= RIDGE incumbent inflow
    t0 = time.time()
    torch.manual_seed(seed * 100)
    ae = MaskedAE(zs.shape[1], BOT).to(DEV)
    opt = torch.optim.Adam(ae.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(seed * 100)
    ztr_ae = T(zs[tr])
    for _ in range(EPOCHS):
        perm = torch.randperm(ztr_ae.shape[0], generator=g)
        for i in range(0, len(perm), BATCH):
            bt = ztr_ae[perm[i:i + BATCH]]
            msk = torch.rand(bt.shape[0], Vt, device=DEV) < MASK
            mc = msk.repeat_interleave(E, dim=1)
            loss = ((ae(bt.masked_fill(mc, 0.0)) - bt)[mc] ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        code = ae.enc(T(zs)).cpu().numpy()
    own = [poly3(zs[:, k * E:(k + 1) * E]) for k in range(Vt)]
    base = np.array([ridge_r2(own[k][tr_i], lead[tr_i + 1, k],
                              own[k][te_i], lead[te_i + 1, k])
                     for k in range(Vt)])
    inflow_r = np.array([
        ridge_r2(np.hstack([own[k][tr_i], code[tr_i]]), lead[tr_i + 1, k],
                 np.hstack([own[k][te_i], code[te_i]]), lead[te_i + 1, k])
        - base[k] for k in range(Vt)])
    t_ridge = time.time() - t0

    # ================= ADD incumbent-style additive outflow
    t0 = time.time()
    outflow_a = np.zeros(Vt)
    for j in range(Vt):
        gains = []
        for k in range(Vt):
            if k == j:
                continue
            gains.append(ridge_r2(
                np.hstack([own[k][tr_i], own[j][tr_i]]), lead[tr_i + 1, k],
                np.hstack([own[k][te_i], own[j][te_i]]), lead[te_i + 1, k])
                - base[k])
        outflow_a[j] = float(np.mean(gains))
    t_add = time.time() - t0

    np.savez_compressed(
        OUT / f"raw_nz{noise}_r{red}_s{seed}.npz",
        inflow_u=inflow_u, outflow_u=outflow_u, inflow_r=inflow_r,
        outflow_a=outflow_a, is_driven=is_driven, is_source=is_source,
        n_children=n_children, r2_full=r2_full, r2_skip=r2_skip)

    def score(name, kind, s, secs, r2m):
        # PRIMARY: average precision for the MINORITY class (sources).
        # Inflow ranks sources LAST, so it is scored on its negation.
        v_ = -s if kind == "inflow" else s
        return dict(
            noise=noise, red=red, seed=seed, arm=name, kind=kind,
            ap_source=float(average_precision_score(is_source, v_)),
            auroc_source=float(roc_auc_score(is_source, v_)),
            base_rate=float(is_source.mean()),
            # SECONDARY: driven-positive, base rate 0.833, ceiling 1.2x
            ap_driven=float(average_precision_score(is_driven, s))
            if kind == "inflow" else np.nan,
            rho_children=float(spearmanr(s[:n_src],
                                         n_children[:n_src]).statistic)
            if kind == "outflow" else np.nan,
            model_r2=r2m, secs=secs)

    return [
        score("UNIFIED", "inflow", inflow_u, t_uni, float(np.mean(r2_full))),
        score("UNIFIED", "outflow", outflow_u, t_uni, float(np.mean(r2_full))),
        score("RIDGE", "inflow", inflow_r, t_ridge, np.nan),
        score("ADD", "outflow", outflow_a, t_add, np.nan),
    ]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEV}   V={V}  bottleneck={BOT}   EXPLORATORY, no adoption")
    print(f"primary: average precision on the MINORITY class, base rate "
          f"{1/6:.3f}, bar {2/6:.3f}\n")
    recs, t0 = [], time.time()
    for red in REDUNDANCIES:
        for nz in NOISES:
            for s in SEEDS:
                recs += cell(nz, red, s)
                print(f"  red={red} noise={nz:<5} seed={s}   "
                      f"({(time.time()-t0)/60:.1f}m)", flush=True)
                pd.DataFrame(recs).to_csv(OUT / "cells.csv", index=False)
    d = pd.DataFrame(recs)
    print(f"\n({(time.time()-t0)/60:.1f} min)\n")
    br, bar = 1 / 6, 2 / 6

    print(f"AVERAGE PRECISION, sources positive (chance {br:.3f}, bar {bar:.3f})")
    piv = d.pivot_table(index=["red", "noise"], columns=["kind", "arm"],
                        values="ap_source")
    print("   " + piv.round(3).to_string().replace("\n", "\n   "))

    print("\nN5 GUARD per arm: unified model held-out forecast R2")
    print("   " + d[d.arm == "UNIFIED"].pivot_table(
        index=["red", "noise"], values="model_r2").round(3)
        .to_string().replace("\n", "\n   "))

    z = d[(d.red == 0) & (d.noise == 0.0)]
    inf_u = z[(z.arm == "UNIFIED") & (z.kind == "inflow")].ap_source.mean()
    inf_r = z[z.arm == "RIDGE"].ap_source.mean()
    n5 = bool(d[d.arm == "UNIFIED"].groupby(["red", "noise"])
              .model_r2.mean().min() > 0.5)
    n1 = bool(inf_u >= inf_r - 0.05)
    o = d[(d.kind == "outflow") & (d.arm == "UNIFIED")].groupby("red")
    out0, out2 = o.ap_source.mean()[0], o.ap_source.mean()[2]
    n2 = bool(out0 > bar)
    n3_fall = bool(out2 < out0)
    n3_kill = bool(out2 < br + 0.05)

    print(f"\nN1 DISQUALIFYING  unified inflow AP {inf_u:.3f} vs ridge "
          f"{inf_r:.3f} (tolerance 0.05)   -> {'HOLDS' if n1 else 'FAILS'}")
    print(f"N5 GUARD          min unified forecast R2 "
          f"{d[d.arm=='UNIFIED'].groupby(['red','noise']).model_r2.mean().min():.3f}"
          f"   -> {'HOLDS' if n5 else 'FAILS'}")
    print(f"N2 DECISIVE       outflow AP at redundancy 0 = {out0:.3f} vs bar "
          f"{bar:.3f}   -> {'HOLDS' if n2 else 'FAILS'}")
    print(f"N3 DECISIVE       outflow AP {out0:.3f} -> {out2:.3f} under "
          f"redundancy   -> {'FALLS as predicted' if n3_fall else 'does NOT fall'}")
    print(f"                  kill condition (within 0.05 of base rate): "
          f"{'FIRES' if n3_kill else 'does not fire'}")

    print("\nN4 GRADED CONTROLS, declared in advance")
    gc = d[d.kind == "outflow"].pivot_table(index=["red"], columns="arm",
                                            values="rho_children")
    print("   outflow vs child count among sources:")
    print("   " + gc.round(3).to_string().replace("\n", "\n   "))

    print("\nVERDICT (rule fixed before running)")
    if not n1:
        print("   -> REJECTED. The combination costs inflow; the two")
        print("      statistics stay on separate machinery.")
    elif n3_kill:
        print("   -> CLOSED. Ablation outflow dies under redundancy, and the")
        print("      unification keeps only its inflow half, which the ResNet")
        print("      run already showed buys nothing. The line closes.")
    elif n2:
        print("   -> OPEN. Licenses one powered pre-registration at more")
        print("      widths and a second family. Nothing enters the manuscript.")
    else:
        print("   -> CLOSED. Outflow does not clear the bar even without")
        print("      redundancy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
