"""Bottleneck width, and continuous embeddings versus binary encodings.

Pre-registration: paper/bottleneck_protocol.md, committed before this was
written.

The boundary map found detections capped at 9-27 channels regardless of V;
the Proposition 2 decomposition found 80-92% of the readout gap is
compression. Both point at the fixed 32-wide bottleneck and neither tested
widening it. The binary arm separates capacity from geometry: b bits versus
b reals at the same width.

    python scripts/bottleneck_scaling.py           # full grid
    python scripts/bottleneck_scaling.py --quick   # one cell, for checks
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from boundary_map import (DEV, DONOR_R2, E, MASK, MIN_DONORS, N_GHOSTS,  # noqa
                          BATCH, EPOCHS, embed, make_system, poly3, ridge_r2)

OUT = Path("ExpOutput/bottleneck")
N, COUPLING = 4000, 0.20
V_VALUES = (30, 60, 120)
B_VALUES = (8, 16, 32, 64, 128)
CODES = ("float", "binary")
SEEDS = (0, 1, 2)
MODELS = 2


class SignSTE(torch.autograd.Function):
    """Hard sign with a straight-through gradient: forward is +-1, backward
    passes the gradient unchanged inside the linear region."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (x.abs() <= 1).float()      # clipped straight-through


class AE(torch.nn.Module):
    """Masked autoencoder with a float or binary bottleneck."""

    def __init__(self, d_in: int, b: int, code: str = "float"):
        super().__init__()
        h = max(2 * b, 64)
        self.code = code
        self.enc = torch.nn.Sequential(
            torch.nn.Linear(d_in, h), torch.nn.Tanh(), torch.nn.Linear(h, b))
        self.dec = torch.nn.Sequential(
            torch.nn.Linear(b, h), torch.nn.Tanh(), torch.nn.Linear(h, d_in))

    def encode(self, z):
        c = self.enc(z)
        return SignSTE.apply(c) if self.code == "binary" else c

    def forward(self, z):
        return self.dec(self.encode(z))


def run_cell(V, b, code, seed):
    x, is_driven, is_source = make_system(N, V, COUPLING, 0, seed)
    Vt = x.shape[1]
    emb = embed(x)
    m = emb.shape[0]
    a, bb = int(0.6 * m), int(0.8 * m)
    tr = slice(0, a)
    tr_i = np.arange(0, a - 1)
    te_i = np.arange(bb, m - 1)
    mu, sd = emb[tr].mean(0), emb[tr].std(0) + 1e-12
    zs = np.clip(np.nan_to_num((emb - mu) / sd), -20, 20).astype(np.float32)
    lead = zs[:, [j * E for j in range(Vt)]]
    feats = [poly3(zs[:, q * E:(q + 1) * E]) for q in range(Vt)]
    self_r2 = np.array([ridge_r2(f[tr_i], lead[tr_i + 1, q],
                                 f[te_i], lead[te_i + 1, q])
                        for q, f in enumerate(feats)])

    ztr = torch.as_tensor(zs[tr], device=DEV)
    zfull = torch.as_tensor(zs, device=DEV)
    codes, dead = [], 0
    for mm in range(MODELS):
        torch.manual_seed(seed * 100 + mm)
        net = AE(zs.shape[1], b, code).to(DEV)
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
            c = net.encode(zfull).cpu().numpy()
        # a binary code can collapse: count dimensions that never vary
        dead += int((c.std(0) < 1e-9).sum())
        codes.append(c)

    def excess_of(f, target):
        base = ridge_r2(f[tr_i], target[tr_i + 1], f[te_i], target[te_i + 1])
        return float(np.mean([
            ridge_r2(np.hstack([f[tr_i], c[tr_i]]), target[tr_i + 1],
                     np.hstack([f[te_i], c[te_i]]), target[te_i + 1]) - base
            for c in codes]))

    excess = np.array([excess_of(feats[q], lead[:, q]) for q in range(Vt)])
    rng = np.random.default_rng(seed + 4242)
    qual = np.where(self_r2 > DONOR_R2)[0]
    pool = np.arange(Vt) if len(qual) < MIN_DONORS else qual
    donors = rng.choice(pool, size=min(N_GHOSTS, len(pool)), replace=False)
    gh = []
    for d in donors:
        s = int(rng.integers(m // 4, 3 * m // 4))
        gz = np.roll(zs[:, d * E:(d + 1) * E], s, axis=0)
        gh.append(excess_of(poly3(gz), np.roll(lead[:, d], s)))
    gh = np.array(gh)
    thr = max(0.0, float(gh.max()))
    fl = excess > thr
    tp = int((fl & is_driven).sum())
    return {
        "V": V, "b": b, "code": code, "seed": seed,
        "precision": tp / max(int(fl.sum()), 1),
        "recall": tp / max(int(is_driven.sum()), 1),
        "n_flagged": int(fl.sum()), "n_driven": int(is_driven.sum()),
        "ghost_med": float(np.median(gh)), "ghost_max": float(gh.max()),
        "saturation": float((self_r2 > 0.9).mean()),
        "source_fp": float((fl & is_source).sum()
                           / max(int(is_source.sum()), 1)),
        "dead_dims": dead / MODELS,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "bottleneck.csv"
    rows = pd.read_csv(dest).to_dict("records") if dest.exists() else []
    done = {(r["V"], r["b"], r["code"], r["seed"]) for r in rows}

    cells = ([(60, 32, "float", 0)] if args.quick else
             [(v, b, c, s) for v in V_VALUES for b in B_VALUES
              for c in CODES for s in SEEDS])
    cells = [c for c in cells if c not in done]
    print(f"device: {DEV}   cells to run: {len(cells)}"
          f"   ({len(rows)} cached)\n")

    t0 = time.time()
    for i, (v, b, c, s) in enumerate(cells, 1):
        try:
            r = run_cell(v, b, c, s)
        except Exception as exc:
            print(f"  [{i}/{len(cells)}] V={v} b={b} {c} s={s} "
                  f"FAILED: {str(exc)[:60]}", flush=True)
            continue
        rows.append(r)
        pd.DataFrame(rows).to_csv(dest, index=False)
        print(f"  [{i}/{len(cells)}] V={v:<4} b={b:<4} {c:<7} s={s}  "
              f"prec {r['precision']:.2f}  rec {r['recall']:.2f}  "
              f"flagged {r['n_flagged']:>3}/{r['n_driven']}  "
              f"ghost {r['ghost_max']:+.4f}  srcFP {r['source_fp']:.2f}"
              f"{'  dead ' + str(int(r['dead_dims'])) if r['dead_dims'] else ''}",
              flush=True)
    print(f"\n({(time.time()-t0)/60:.1f} min) -> {dest}")

    d = pd.DataFrame(rows)
    if d.empty:
        return 0
    print("\nBS1/BS2  RECALL by width and V (median over seeds)")
    for code in CODES:
        print(f"\n  code = {code}")
        piv = (d[d.code == code].groupby(["V", "b"]).recall.median()
               .unstack().round(2))
        print("   " + piv.to_string().replace("\n", "\n   "))
    print("\nBS3/BS4  binary vs float at matched width (median recall)")
    cmp = (d.groupby(["b", "code"]).recall.median().unstack().round(2))
    print("   " + cmp.to_string().replace("\n", "\n   "))
    print("\nBS5/BS6  ghost and sources by width")
    g = (d.groupby(["code", "b"])
         .agg(ghost_max=("ghost_max", "mean"),
              src_fp=("source_fp", "max"),
              prec=("precision", "median"),
              dead=("dead_dims", "mean")).round(4))
    print("   " + g.to_string().replace("\n", "\n   "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
