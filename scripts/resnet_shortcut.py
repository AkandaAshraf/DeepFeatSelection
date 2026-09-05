"""A residual shortcut self-baseline, and taking the target out of the code.

Pre-registration: paper/resnet_shortcut_protocol.md, committed before this was
written or run. EXPLORATORY -- nothing is adopted from this run.

Two verified defects, crossed so each is attributable:

  the self-baseline is fixed degree-3 polynomial ridge, which is not dense in
  anything, so Proposition 1's saturation premise can fail on an autonomous
  channel and nothing then bounds its excess above zero;

  the code is computed from the joint state of ALL variables, so the joint
  readout holds a learned representation of the target's own lags that the
  polynomial baseline does not have.

The masked autoencoder trains with random channel subsets zeroed, so a code
computed with the target's columns zeroed is in distribution for the SAME
encoder. Exclusion therefore costs one forward pass per target, not a
retrained encoder, and the amortisation survives.

The own trunk is trained first and FROZEN before any code branch is fit.
Trained jointly, the optimiser could underfit the own path and let the code
branch carry prediction that own lags could have supplied, manufacturing
excess on autonomous channels. X3 is the guard on that.

    python scripts/resnet_shortcut.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from boundary_map import (BATCH, DEV, E, EPOCHS, MASK, embed,  # noqa: E402
                          make_system, poly3, ridge_r2)
from wormwideweb_gate import MaskedAE  # noqa: E402

OUT = Path("ExpOutput/resnet_shortcut")
N, V, COUPLING, REDUNDANCY = 4000, 30, 0.20, 0
NOISES = (0.0, 0.30)
SEEDS = (0, 1, 2)
H = 32                      # trunk width; deliberately small
TRUNK_STEPS = 600          # full-batch steps, early stopped
BRANCH_STEPS = 600
EVAL_EVERY = 20
LR = 1e-2


class ResTrunk(nn.Module):
    """Own-lag trunk with a residual block, plus its own linear head."""

    def __init__(self, d_in, h=H):
        super().__init__()
        self.inp = nn.Linear(d_in, h)
        self.b1 = nn.Sequential(nn.Tanh(), nn.Linear(h, h),
                                nn.Tanh(), nn.Linear(h, h))
        self.head = nn.Linear(h, 1)

    def feats(self, x):
        z = self.inp(x)
        return z + self.b1(z)            # the shortcut inside the trunk

    def forward(self, x):
        return self.head(self.feats(x)).squeeze(-1)


class CodeBranch(nn.Module):
    """Adds the code. With shortcut, output is own_pred + delta."""

    def __init__(self, d_feat, d_code, h=H, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.net = nn.Sequential(nn.Linear(d_feat + d_code, h), nn.Tanh(),
                                 nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))

    def forward(self, feats, code, own_pred):
        d = self.net(torch.cat([feats, code], dim=1)).squeeze(-1)
        return own_pred + d if self.shortcut else d


def r2(pred, y):
    ss = float(((y - pred) ** 2).sum())
    tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss / tot if tot > 0 else 0.0


def fit(module, params, xs_tr, y_tr, xs_va, y_va, fwd, steps, seed):
    """Full-batch Adam with EARLY STOPPING on the reserved validation
    segment, restoring the best-validation weights.

    Without this the trunk overfits within ~50 steps: measured on one
    channel, test R2 peaks at 0.165 and decays to 0.02 by step 1350 while
    train loss keeps falling. "Trained to convergence" means best
    generalisation, not lowest training loss, and the 0.6/0.2/0.2 split
    reserves the middle fifth for exactly this. Weight decay was tried and
    does far less than early stopping."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(params, lr=LR)
    best, bstate = -1e9, None
    for st in range(steps):
        loss = ((fwd(xs_tr) - y_tr) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if st % EVAL_EVERY == 0 or st == steps - 1:
            with torch.no_grad():
                rv = r2(fwd(xs_va), y_va)
            if rv > best:
                best = rv
                bstate = {k: v.clone() for k, v in module.state_dict().items()}
    if bstate is not None:
        module.load_state_dict(bstate)
    return best


def zscore(a, tr_rows):
    """Standardise columns on the train rows only. Networks need this; the
    ridge arm does not, because its penalty absorbs the scale."""
    mu = a[tr_rows].mean(0)
    sd = a[tr_rows].std(0) + 1e-8
    return ((a - mu) / sd).astype(np.float32)


def cell(noise, seed):
    x, is_driven, is_source = make_system(N, V, COUPLING, REDUNDANCY, seed)
    if noise:
        x = x + noise * np.random.default_rng(seed + 777).standard_normal(
            x.shape)
    Vt = x.shape[1]
    emb = embed(x)
    m = emb.shape[0]
    a, bnd = int(0.6 * m), int(0.8 * m)
    tr, tr_i = slice(0, a), np.arange(0, a - 1)
    va_i, te_i = np.arange(a, bnd - 1), np.arange(bnd, m - 1)
    mu, sd = emb[tr].mean(0), emb[tr].std(0) + 1e-12
    zs = np.clip(np.nan_to_num((emb - mu) / sd), -20, 20).astype(np.float32)
    lead = zs[:, [j * E for j in range(Vt)]]

    # ---- one encoder for the whole cell, shared by every arm
    b = 2 * V
    torch.manual_seed(seed * 100)
    net = MaskedAE(zs.shape[1], b).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(seed * 100)
    ztr = torch.as_tensor(zs[tr], device=DEV)
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
    zfull = torch.as_tensor(zs, device=DEV)
    with torch.no_grad():
        code_all = net.enc(zfull).cpu().numpy()

    rows = []
    for q in range(Vt):
        own_np = poly3(zs[:, q * E:(q + 1) * E]).astype(np.float32)
        y_np = lead[:, q].astype(np.float32)
        # code with THIS target's columns zeroed, same encoder
        with torch.no_grad():
            zq = zfull.clone()
            zq[:, q * E:(q + 1) * E] = 0.0
            code_ex = net.enc(zq).cpu().numpy()

        # incumbent: poly3 ridge, self and joint
        r2_ridge_own = ridge_r2(own_np[tr_i], y_np[tr_i + 1],
                                own_np[te_i], y_np[te_i + 1])
        ex_ridge = ridge_r2(
            np.hstack([own_np[tr_i], code_all[tr_i]]), y_np[tr_i + 1],
            np.hstack([own_np[te_i], code_all[te_i]]), y_np[te_i + 1]
        ) - r2_ridge_own

        T = lambda v: torch.as_tensor(v, device=DEV)  # noqa: E731
        # The trunk takes the RAW own lags and learns its own nonlinearity.
        # Feeding it poly3 of data clipped at +-20 spans several orders of
        # magnitude and does not train; that configuration failed the X3
        # guard and was corrected before X1 or X2 were read.
        raw_np = zscore(zs[:, q * E:(q + 1) * E], tr_i)
        own_tr, own_va, own_te = (T(raw_np[tr_i]), T(raw_np[va_i]),
                                  T(raw_np[te_i]))
        y_tr, y_va, y_te = (T(y_np[tr_i + 1]), T(y_np[va_i + 1]),
                            T(y_np[te_i + 1]))

        # ---- stage 1: own trunk alone, then FROZEN
        torch.manual_seed(seed * 1000 + q)
        trunk = ResTrunk(raw_np.shape[1]).to(DEV)
        fit(trunk, list(trunk.parameters()), [own_tr], y_tr, [own_va], y_va,
            lambda xs: trunk(xs[0]), TRUNK_STEPS, seed * 1000 + q)
        for p in trunk.parameters():
            p.requires_grad_(False)
        trunk.eval()
        with torch.no_grad():
            f_tr, f_va, f_te = (trunk.feats(own_tr), trunk.feats(own_va),
                                trunk.feats(own_te))
            op_tr, op_va, op_te = trunk(own_tr), trunk(own_va), trunk(own_te)
            r2_trunk_own = r2(op_te, y_te)

        # ---- stage 2: four code branches on the frozen trunk
        arm_ex = {}
        for tag, cd, sc in (("RES-SC-INCL", code_all, True),
                            ("RES-SC-EXCL", code_ex, True),
                            ("RES-NOSC-INCL", code_all, False),
                            ("RES-NOSC-EXCL", code_ex, False)):
            cz = zscore(cd, tr_i)          # networks need scaled inputs
            c_tr, c_va, c_te = T(cz[tr_i]), T(cz[va_i]), T(cz[te_i])
            torch.manual_seed(seed * 1000 + q)
            br = CodeBranch(f_tr.shape[1], c_tr.shape[1], shortcut=sc).to(DEV)
            fit(br, list(br.parameters()), [f_tr, c_tr, op_tr], y_tr,
                [f_va, c_va, op_va], y_va,
                lambda xs: br(xs[0], xs[1], xs[2]), BRANCH_STEPS,
                seed * 1000 + q)
            with torch.no_grad():
                arm_ex[tag] = r2(br(f_te, c_te, op_te), y_te) - r2_trunk_own

        rows.append(dict(noise=noise, seed=seed, q=q,
                         is_driven=bool(is_driven[q]),
                         is_source=bool(is_source[q]),
                         r2_ridge_own=r2_ridge_own, r2_trunk_own=r2_trunk_own,
                         **{"RIDGE-INCL": ex_ridge}, **arm_ex))
    return rows


ARMS = ["RIDGE-INCL", "RES-SC-INCL", "RES-SC-EXCL",
        "RES-NOSC-INCL", "RES-NOSC-EXCL"]


def summarise(d):
    from sklearn.metrics import roc_auc_score
    out = []
    for (nz, sd), g in d.groupby(["noise", "seed"]):
        k = int(g.is_driven.sum())
        for arm in ARMS:
            s = g[g.is_source][arm]
            top = g.nlargest(k, arm)
            out.append(dict(
                noise=nz, seed=sd, arm=arm,
                src_mean=float(s.mean()), src_max=float(s.max()),
                auroc=float(roc_auc_score(g.is_driven, g[arm]))
                if g.is_driven.nunique() > 1 else np.nan,
                src_fp_topk=float(top.is_source.sum()
                                  / max(int(g.is_source.sum()), 1)),
                recall_topk=float(top.is_driven.sum() / max(k, 1))))
    return pd.DataFrame(out)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEV}   V={V}  b={2*V}  H={H}   EXPLORATORY, no adoption")
    print(f"{len(NOISES)*len(SEEDS)} cells x {len(ARMS)} arms\n")
    rows, t0 = [], time.time()
    for nz in NOISES:
        for s in SEEDS:
            rows += cell(nz, s)
            print(f"  noise={nz:<5} seed={s}  ({(time.time()-t0)/60:.1f}m)",
                  flush=True)
            pd.DataFrame(rows).to_csv(OUT / "channels.csv", index=False)
    d = pd.DataFrame(rows)
    t = summarise(d)
    t.to_csv(OUT / "cells.csv", index=False)
    print(f"\n({(time.time()-t0)/60:.1f} min)\n")

    print("X3 GUARD: is the learned trunk a stronger self-baseline than poly3?")
    for nz, g in d.groupby("noise"):
        print(f"   noise {nz:<5} poly3 R2_own {g.r2_ridge_own.mean():.4f}   "
              f"trunk R2_own {g.r2_trunk_own.mean():.4f}   "
              f"trunk wins {100*(g.r2_trunk_own > g.r2_ridge_own).mean():.0f}%")
    x3 = bool(d.groupby("noise").apply(
        lambda g: g.r2_trunk_own.mean() > g.r2_ridge_own.mean()).all())
    print(f"   -> X3 {'HOLDS' if x3 else 'FAILS: run is INCONCLUSIVE'}\n")

    print("MEAN EXCESS ON SOURCES (autonomous by construction; guarantee <= 0)")
    print("   " + t.pivot_table(index="noise", columns="arm",
                                values="src_mean")[ARMS].round(4)
          .to_string().replace("\n", "\n   "))
    print("\nSOURCE FP AT MATCHED TOP-K")
    print("   " + t.pivot_table(index="noise", columns="arm",
                                values="src_fp_topk")[ARMS].round(3)
          .to_string().replace("\n", "\n   "))
    print("\nAUROC driven vs source (X4: no prediction)")
    print("   " + t.pivot_table(index="noise", columns="arm",
                                values="auroc")[ARMS].round(3)
          .to_string().replace("\n", "\n   "))

    hard = t[t.noise == 0.30].groupby("arm")
    sm, sf = hard.src_mean.mean(), hard.src_fp_topk.mean()
    x1 = bool(sm["RES-SC-EXCL"] < sm["RES-SC-INCL"]
              and sm["RES-NOSC-EXCL"] < sm["RES-NOSC-INCL"])
    x2 = bool(sf["RES-SC-EXCL"] < sf["RES-SC-INCL"]
              and sf["RES-NOSC-EXCL"] < sf["RES-NOSC-INCL"])
    print(f"\nX1 at noise 0.30, mean source excess:")
    print(f"   SC   incl {sm['RES-SC-INCL']:+.4f} -> excl "
          f"{sm['RES-SC-EXCL']:+.4f}")
    print(f"   NOSC incl {sm['RES-NOSC-INCL']:+.4f} -> excl "
          f"{sm['RES-NOSC-EXCL']:+.4f}")
    print(f"   -> X1 {'HOLDS' if x1 else 'FAILS'}")
    print(f"X2 at noise 0.30, source FP at matched top-k:")
    print(f"   SC   incl {sf['RES-SC-INCL']:.3f} -> excl "
          f"{sf['RES-SC-EXCL']:.3f}")
    print(f"   NOSC incl {sf['RES-NOSC-INCL']:.3f} -> excl "
          f"{sf['RES-NOSC-EXCL']:.3f}")
    print(f"   -> X2 {'HOLDS' if x2 else 'FAILS'}")
    print("\nNO ADOPTION RULE. Exploratory by declaration; a result in the")
    print("predicted direction licenses a powered pre-registration only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
