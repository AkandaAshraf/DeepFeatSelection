"""Does the Proposition 2 decomposition hold as the system grows?

The single-system result (scripts/prop2_gap.py) found the gap is ~77%
readout, ~23% compression, which would say the estimator is worth changing.
That was measured at V = 15 with a 32-dimensional bottleneck - barely any
compression at all. The question that decides whether the finding transfers
to the scale MACE is built for is whether the compression share grows with V.

Stated before running: we expect the compression share to RISE with V,
because the bottleneck is fixed while the system grows. If it rises steeply,
the readout fix helps least exactly where MACE is most useful, and the
recommendation weakens accordingly.

    python scripts/prop2_gap_sweep.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from prop2_gap import (BOTTLENECK, DEV, E, MASK, MODELS, OUT, SEED,  # noqa
                       BATCH, EPOCHS, driven_system, embed, interact, poly3,
                       ridge_r2)
from wormwideweb_gate import MaskedAE  # noqa: E402


def one_system(n_drivers, n_driven, coupling=0.30, seed=0, n=6000):
    x, is_driven = driven_system(n=n, n_drivers=n_drivers,
                                 n_driven=n_driven, coupling=coupling,
                                 seed=seed)
    ghost_src = int(np.where(is_driven)[0][0])
    xg = np.concatenate(
        [x, np.roll(x[:, [ghost_src]], len(x) // 3, axis=0)], axis=1)
    V = xg.shape[1]
    driven = np.append(is_driven, False)

    emb = embed(xg)
    nn = emb.shape[0]
    a, b = int(0.6 * nn), int(0.8 * nn)
    tr = slice(0, a)
    tr_i = np.arange(0, a - 1)
    te_i = np.arange(b, nn - 1)
    mu, sd = emb[tr].mean(0), emb[tr].std(0) + 1e-12
    zs = np.clip((emb - mu) / sd, -20, 20).astype(np.float32)
    lead = zs[:, [j * E for j in range(V)]]
    own_lin = [zs[:, q * E:(q + 1) * E] for q in range(V)]
    feats = [poly3(o) for o in own_lin]

    ztr = torch.as_tensor(zs[tr], device=DEV)
    zfull = torch.as_tensor(zs, device=DEV)
    codes = []
    for m in range(MODELS):
        torch.manual_seed(SEED + m)
        net = MaskedAE(zs.shape[1], BOTTLENECK).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        g = torch.Generator().manual_seed(SEED + m)
        for _ in range(EPOCHS):
            perm = torch.randperm(ztr.shape[0], generator=g)
            for i in range(0, len(perm), BATCH):
                bt = ztr[perm[i:i + BATCH]]
                msk = torch.rand(bt.shape[0], V, device=DEV) < MASK
                mc = msk.repeat_interleave(E, dim=1)
                loss = ((net(bt.masked_fill(mc, 0.0)) - bt)[mc] ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            codes.append(net.enc(zfull).cpu().numpy())

    rows = []
    # the oracle is O(V) features per channel, so cap it for large V by
    # sampling other channels - reported, not hidden
    rng = np.random.default_rng(0)
    for q in range(V):
        y_tr, y_te = lead[tr_i + 1, q], lead[te_i + 1, q]
        base = ridge_r2(feats[q][tr_i], y_tr, feats[q][te_i], y_te)
        pool = [p for p in range(V) if p != q]
        if len(pool) > 40:
            pool = list(rng.choice(pool, 40, replace=False))
        others = np.hstack([feats[p] for p in pool])
        oracle = ridge_r2(np.hstack([feats[q][tr_i], others[tr_i]]), y_tr,
                          np.hstack([feats[q][te_i], others[te_i]]), y_te)
        aff, itr = [], []
        for c in codes:
            aff.append(ridge_r2(
                np.hstack([feats[q][tr_i], c[tr_i]]), y_tr,
                np.hstack([feats[q][te_i], c[te_i]]), y_te))
            ix = interact(own_lin[q], c)
            itr.append(ridge_r2(
                np.hstack([feats[q][tr_i], c[tr_i], ix[tr_i]]), y_tr,
                np.hstack([feats[q][te_i], c[te_i], ix[te_i]]), y_te))
        rows.append({"driven": bool(driven[q]), "ghost": q == V - 1,
                     "aff": float(np.mean(aff)) - base,
                     "itc": float(np.mean(itr)) - base,
                     "orc": oracle - base})
    d = pd.DataFrame(rows)
    dr = d[d.driven & ~d.ghost]
    gh = d[d.ghost].iloc[0]
    aff, itc, orc = dr.aff.mean(), dr.itc.mean(), dr.orc.mean()
    total = orc - aff
    return {"V": V - 1, "coupling": coupling,
            "affine": aff, "interact": itc, "oracle": orc, "gap": total,
            "readout_share": (itc - aff) / total if total > 1e-9 else np.nan,
            "ghost_affine": gh.aff, "ghost_interact": gh.itc}


def main() -> int:
    print(f"device: {DEV}")
    print("Stated before running: compression share is expected to RISE "
          "with V,\nbecause the bottleneck is fixed at "
          f"{BOTTLENECK} while the system grows.\n")
    out = []
    t0 = time.time()
    for nd, nk in ((4, 10), (8, 22), (15, 45), (25, 75)):
        r = one_system(nd, nk)
        out.append(r)
        print(f"  V={r['V']:4d}  gap {r['gap']:+.4f}   "
              f"readout {100*r['readout_share']:5.1f}%   "
              f"compression {100*(1-r['readout_share']):5.1f}%   "
              f"ghost aff {r['ghost_affine']:+.4f} -> int "
              f"{r['ghost_interact']:+.4f}", flush=True)
    print(f"\n  ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(out)
    df.to_csv(OUT / "gap_sweep.csv", index=False)
    print("\nDoes the readout share hold up as V grows?")
    lo, hi = df.readout_share.iloc[0], df.readout_share.iloc[-1]
    print(f"  V={df.V.iloc[0]}: {100*lo:.1f}%   ->   "
          f"V={df.V.iloc[-1]}: {100*hi:.1f}%")
    print(f"  -> {'HOLDS' if hi > 0.5 else 'DEGRADES - the fix helps least at scale'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
