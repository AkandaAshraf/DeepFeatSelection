"""Can channel duplication manufacture drivenness?

Pre-registration: paper/duplicate_channel_protocol.md, committed before this
was written.

Isolated channels are coupled to nothing, so their excess should be zero. If
the system contains a near-copy of an isolated channel, the code carries the
copy and the copy predicts the original's next step. The original may then be
flagged as driven for no reason but duplication.

The ghost cannot detect this: it is a circularly SHIFTED copy, which destroys
the simultaneous relationship, while a duplicate preserves it.

    python scripts/duplicate_channel.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from boundary_map import (BOTTLENECK, DEV, DONOR_R2, E, EPOCHS, MASK,  # noqa
                          MIN_DONORS, MODELS, N_GHOSTS, BATCH, embed, poly3,
                          ridge_r2)
from wormwideweb_gate import MaskedAE  # noqa: E402

OUT = Path("ExpOutput/duplicate_channel")
V, N, COUPLING = 60, 4000, 0.20
K_VALUES = (0, 1, 2, 4)
NOISE_VALUES = (0.01, 0.05, 0.20)
SEEDS = (0, 1, 2)
N_DUP_TARGETS = 5      # isolated channels given copies


def make_system(k, noise, seed):
    """Drivers, driven channels, isolated channels, plus k copies of some
    isolated channels. Copies are appended and excluded from all truth sets."""
    rng = np.random.default_rng(seed)
    n_src = 10
    n_iso = 15
    n_drv = V - n_src - n_iso
    x = np.zeros((N, V))
    x[0] = rng.uniform(0.2, 0.8, V)
    r = rng.uniform(3.6, 3.9, V)
    parent = rng.integers(0, n_src, n_drv)
    for t in range(N - 1):
        s = x[t, :n_src]
        x[t + 1, :n_src] = np.clip(r[:n_src] * s * (1 - s), 0, 1)
        d = x[t, n_src:n_src + n_drv]
        x[t + 1, n_src:n_src + n_drv] = np.clip(
            r[n_src:n_src + n_drv] * d * (1 - d)
            + COUPLING * x[t, parent] * (1 - d), 0, 1)
        i = x[t, n_src + n_drv:]
        x[t + 1, n_src + n_drv:] = np.clip(
            r[n_src + n_drv:] * i * (1 - i), 0, 1)

    iso_idx = np.arange(n_src + n_drv, V)
    dup_targets = iso_idx[:N_DUP_TARGETS]      # these get copies
    plain_iso = iso_idx[N_DUP_TARGETS:]        # these do not

    cols = [x]
    if k:
        for q in dup_targets:
            for _ in range(k):
                cols.append((x[:, q] + noise * rng.standard_normal(N))[:, None])
    xx = np.concatenate(cols, axis=1)
    xx = xx + 0.005 * rng.standard_normal(xx.shape)

    Vt = xx.shape[1]
    is_source = np.zeros(Vt, bool)
    is_source[:n_src] = True
    is_driven = np.zeros(Vt, bool)
    is_driven[n_src:n_src + n_drv] = True
    is_copy = np.zeros(Vt, bool)
    is_copy[V:] = True
    return xx, is_source, is_driven, is_copy, dup_targets, plain_iso


def scan(x, seed):
    Vt = x.shape[1]
    emb = embed(x)
    m = emb.shape[0]
    a, b = int(0.6 * m), int(0.8 * m)
    tr = slice(0, a)
    tr_i = np.arange(0, a - 1)
    te_i = np.arange(b, m - 1)
    mu, sd = emb[tr].mean(0), emb[tr].std(0) + 1e-12
    zs = np.clip(np.nan_to_num((emb - mu) / sd), -20, 20).astype(np.float32)
    lead = zs[:, [j * E for j in range(Vt)]]
    feats = [poly3(zs[:, q * E:(q + 1) * E]) for q in range(Vt)]
    self_r2 = np.array([ridge_r2(f[tr_i], lead[tr_i + 1, q],
                                 f[te_i], lead[te_i + 1, q])
                        for q, f in enumerate(feats)])

    ztr = torch.as_tensor(zs[tr], device=DEV)
    zfull = torch.as_tensor(zs, device=DEV)
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
    pool = np.arange(Vt) if len(qual) < MIN_DONORS else qual
    donors = rng.choice(pool, size=min(N_GHOSTS, len(pool)), replace=False)
    ghosts = []
    for d in donors:
        s = int(rng.integers(m // 4, 3 * m // 4))
        gz = np.roll(zs[:, d * E:(d + 1) * E], s, axis=0)
        ghosts.append(excess_of(poly3(gz), np.roll(lead[:, d], s)))
    return excess, np.array(ghosts)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEV}   V={V} n={N} coupling={COUPLING}")
    print("duplicated vs plain isolated channels; both coupled to nothing\n")
    rows = []
    t0 = time.time()
    cells = [(k, nz) for k in K_VALUES for nz in
             (NOISE_VALUES if k else (0.0,))]
    for k, nz in cells:
        for seed in SEEDS:
            x, is_src, is_drv, is_cp, dup_t, plain = make_system(k, nz, seed)
            ex, gh = scan(x, seed)
            thr = max(0.0, float(gh.max()))
            fl = ex > thr
            rows.append({
                "k": k, "noise": nz, "seed": seed, "channels": x.shape[1],
                "flag_dup_iso": float(fl[dup_t].mean()),
                "flag_plain_iso": float(fl[plain].mean()),
                "excess_dup_iso": float(np.median(ex[dup_t])),
                "excess_plain_iso": float(np.median(ex[plain])),
                "flag_source": float(fl[is_src].mean()),
                "flag_driven": float(fl[is_drv].mean()),
                "ghost_max": float(gh.max()),
                "ghost_med": float(np.median(gh)),
                "threshold": thr,
            })
            print(f"  k={k} noise={nz:<5} seed={seed}  "
                  f"dup-iso flagged {rows[-1]['flag_dup_iso']:.2f}  "
                  f"plain-iso {rows[-1]['flag_plain_iso']:.2f}  "
                  f"src {rows[-1]['flag_source']:.2f}  "
                  f"ghost {rows[-1]['ghost_max']:+.4f}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "duplicate_channel.csv", index=False)
    print(f"\n({(time.time()-t0)/60:.1f} min)\n")

    print("D1  DOES DUPLICATION MANUFACTURE DRIVENNESS?")
    print(f"  {'k':<4}{'noise':<8}{'dup-iso flagged':>17}{'plain-iso':>12}")
    for (k, nz), g in d.groupby(["k", "noise"]):
        print(f"  {k:<4}{nz:<8}{g.flag_dup_iso.mean():>17.2f}"
              f"{g.flag_plain_iso.mean():>12.2f}")
    base = d[d.k == 0].flag_plain_iso.mean()
    worst = d[d.k > 0].flag_dup_iso.max()
    print(f"  -> {'HAZARD CONFIRMED' if worst > base + 0.1 else 'NO HAZARD'}"
          f"  (worst duplicated flag rate {worst:.2f} vs baseline {base:.2f})")

    print("\nD3  DOES THE GHOST CATCH IT?")
    for k in K_VALUES:
        g = d[d.k == k]
        print(f"  k={k}  ghost max {g.ghost_max.mean():+.4f}   "
              f"ghost median {g.ghost_med.mean():+.4f}")
    print("  -> a flat ghost while flag rates rise means the built-in")
    print("     falsification test is BLIND to this failure mode")

    print("\nD4  ARE SOURCES AFFECTED?")
    print(f"  max source flag rate over all cells: {d.flag_source.max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
