"""Do the TensorFlow and PyTorch pipelines produce the same excess?

Pre-registration: paper/tf_torch_fidelity_protocol.md.

The paper's published results came from TensorFlow; everything since comes
from the PyTorch reimplementation. Same systems, same seeds, same
hyperparameters, both pipelines, compared per channel.

    python scripts/tf_torch_fidelity.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")   # TF on CPU, fair to both

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from boundary_map import embed, poly3  # noqa: E402

OUT = Path("ExpOutput/tf_torch_fidelity")
E, TAU, ALPHA = 3, 1, 1.0
BOTTLENECK, MASK, EPOCHS, BATCH, MODELS = 32, 0.25, 20, 64, 2
V, N, COUPLING = 30, 4000, 0.25
N_GHOSTS = 20


def system(seed):
    rng = np.random.default_rng(seed)
    n_src = 6
    x = np.zeros((N, V))
    x[0] = rng.uniform(0.2, 0.8, V)
    r = rng.uniform(3.6, 3.9, V)
    parent = rng.integers(0, n_src, V - n_src)
    for t in range(N - 1):
        s = x[t, :n_src]
        x[t + 1, :n_src] = np.clip(r[:n_src] * s * (1 - s), 0, 1)
        d = x[t, n_src:]
        x[t + 1, n_src:] = np.clip(
            r[n_src:] * d * (1 - d) + COUPLING * x[t, parent] * (1 - d), 0, 1)
    x += 0.005 * rng.standard_normal((N, V))
    is_driven = np.zeros(V, bool)
    is_driven[n_src:] = True
    return x, is_driven


def prepare(x):
    emb = embed(x)
    m = emb.shape[0]
    a, b = int(0.6 * m), int(0.8 * m)
    tr = slice(0, a)
    tr_i = np.arange(0, a - 1)
    te_i = np.arange(b, m - 1)
    mu, sd = emb[tr].mean(0), emb[tr].std(0) + 1e-12
    zs = np.clip((emb - mu) / sd, -20, 20).astype(np.float32)
    return zs, tr, tr_i, te_i, m


def ridge_np(Xtr, ytr, Xte, yte) -> float:
    """Numpy ridge, shared by both arms so only the ENCODER differs."""
    A = Xtr.T @ Xtr + ALPHA * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ ytr)
    err = float(((Xte @ w - yte) ** 2).mean())
    return max(0.0, 1.0 - err / (float(yte.var()) + 1e-12))


def codes_torch(zs, Vt, tr, seed):
    import torch
    from wormwideweb_gate import MaskedAE
    out = []
    ztr = torch.as_tensor(zs[tr])
    zfull = torch.as_tensor(zs)
    for mm in range(MODELS):
        torch.manual_seed(seed * 100 + mm)
        net = MaskedAE(zs.shape[1], BOTTLENECK)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        g = torch.Generator().manual_seed(seed * 100 + mm)
        for _ in range(EPOCHS):
            perm = torch.randperm(ztr.shape[0], generator=g)
            for i in range(0, len(perm), BATCH):
                bt = ztr[perm[i:i + BATCH]]
                msk = torch.rand(bt.shape[0], Vt) < MASK
                mc = msk.repeat_interleave(E, dim=1)
                loss = ((net(bt.masked_fill(mc, 0.0)) - bt)[mc] ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            out.append(net.enc(zfull).numpy())
    return out


def codes_tf(zs, Vt, tr, seed):
    import keras
    import tensorflow as tf
    out = []
    ztr = zs[tr]
    for mm in range(MODELS):
        keras.utils.set_random_seed(seed * 100 + mm)
        d = zs.shape[1]
        h = max(2 * BOTTLENECK, 64)
        enc = keras.Sequential([keras.layers.Input(shape=(d,)),
                                keras.layers.Dense(h, activation="tanh"),
                                keras.layers.Dense(BOTTLENECK)])
        dec = keras.Sequential([keras.layers.Input(shape=(BOTTLENECK,)),
                                keras.layers.Dense(h, activation="tanh"),
                                keras.layers.Dense(d)])
        opt = keras.optimizers.Adam(3e-3)
        rng = np.random.default_rng(seed * 100 + mm)
        for _ in range(EPOCHS):
            perm = rng.permutation(len(ztr))
            for i in range(0, len(perm), BATCH):
                bt = tf.constant(ztr[perm[i:i + BATCH]])
                msk = rng.random((bt.shape[0], Vt)) < MASK
                mc = tf.constant(np.repeat(msk, E, axis=1))
                with tf.GradientTape() as tape:
                    inp = tf.where(mc, tf.zeros_like(bt), bt)
                    rec = dec(enc(inp))
                    loss = tf.reduce_mean(
                        tf.boolean_mask((rec - bt) ** 2, mc))
                gr = tape.gradient(
                    loss, enc.trainable_variables + dec.trainable_variables)
                opt.apply_gradients(
                    zip(gr, enc.trainable_variables + dec.trainable_variables))
        out.append(enc(tf.constant(zs)).numpy())
    return out


def score(zs, codes, tr_i, te_i, m, Vt, seed):
    lead = zs[:, [j * E for j in range(Vt)]]
    feats = [poly3(zs[:, q * E:(q + 1) * E]) for q in range(Vt)]

    def excess_of(f, target):
        base = ridge_np(f[tr_i], target[tr_i + 1], f[te_i], target[te_i + 1])
        return float(np.mean([
            ridge_np(np.hstack([f[tr_i], c[tr_i]]), target[tr_i + 1],
                     np.hstack([f[te_i], c[te_i]]), target[te_i + 1]) - base
            for c in codes]))

    ex = np.array([excess_of(feats[q], lead[:, q]) for q in range(Vt)])
    rng = np.random.default_rng(seed + 4242)
    donors = rng.choice(Vt, size=min(N_GHOSTS, Vt), replace=False)
    gh = []
    for d in donors:
        s = int(rng.integers(m // 4, 3 * m // 4))
        gz = np.roll(zs[:, d * E:(d + 1) * E], s, axis=0)
        gh.append(excess_of(poly3(gz), np.roll(lead[:, d], s)))
    return ex, np.array(gh)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in (0, 1, 2):
        x, is_driven = system(seed)
        zs, tr, tr_i, te_i, m = prepare(x)
        t0 = time.time()
        ex_t, gh_t = score(zs, codes_torch(zs, V, tr, seed), tr_i, te_i, m, V, seed)
        t_torch = time.time() - t0
        t0 = time.time()
        ex_f, gh_f = score(zs, codes_tf(zs, V, tr, seed), tr_i, te_i, m, V, seed)
        t_tf = time.time() - t0

        fl_t = ex_t > max(0.0, gh_t.max())
        fl_f = ex_f > max(0.0, gh_f.max())
        inter = int((fl_t & fl_f).sum())
        union = int((fl_t | fl_f).sum())
        rows.append({
            "seed": seed,
            "spearman": float(stats.spearmanr(ex_t, ex_f)[0]),
            "pearson": float(stats.pearsonr(ex_t, ex_f)[0]),
            "mad": float(np.abs(ex_t - ex_f).mean()),
            "jaccard": inter / union if union else 1.0,
            "n_flag_torch": int(fl_t.sum()), "n_flag_tf": int(fl_f.sum()),
            "ghost_max_torch": float(gh_t.max()),
            "ghost_max_tf": float(gh_f.max()),
            "ghost_med_torch": float(np.median(gh_t)),
            "ghost_med_tf": float(np.median(gh_f)),
            "secs_torch": round(t_torch, 1), "secs_tf": round(t_tf, 1),
        })
        r = rows[-1]
        print(f"  seed {seed}: spearman {r['spearman']:.3f}  "
              f"jaccard {r['jaccard']:.2f}  MAD {r['mad']:.4f}  "
              f"flagged {r['n_flag_torch']}/{r['n_flag_tf']}  "
              f"({t_torch:.0f}s torch / {t_tf:.0f}s tf)", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "fidelity.csv", index=False)
    print()
    print(f"F1  Spearman  median {d.spearman.median():.3f}   "
          f"{'PASS' if d.spearman.median() >= 0.90 else 'FAIL'} (>= 0.90)")
    print(f"F2  Jaccard   median {d.jaccard.median():.3f}   "
          f"{'PASS' if d.jaccard.median() >= 0.70 else 'FAIL'} (>= 0.70)")
    print(f"F3  ghost max torch {d.ghost_max_torch.mean():+.4f}  "
          f"tf {d.ghost_max_tf.mean():+.4f}")
    print(f"F4  MAD in excess: {d.mad.median():.4f}  "
          f"(effect sizes reported are 0.01-0.30)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
