"""Fish feasibility: the excess detector at V=71,721 (ZAPBench slice, GPU).

First contact with vertebrate scale. Frames 5638-7879 (open loop, rotation,
dark; n=2241), all neurons. Self-contained -- no repo imports -- because this
runs under the WSL GPU venv whose import chain is untested; only keras,
numpy, sklearn required.

Feasibility questions, in order:
1. Does the 215k-dim encoder train at all on the 3070 (memory, wall-clock)?
2. Does the ghost stay pinned at zero at V=71k on real vertebrate calcium?
3. What does a readout cost per neuron (to plan the full 71k x 8-model run)?

Design notes. Embedding is lag-major -- [Z_t | Z_t-1 | Z_t-2], three V-wide
blocks; neuron j's coordinates are {j, V+j, 2V+j} -- which builds the joint
matrix as three shifted views instead of a 71k-loop of per-neuron embeds.
tau=1 at the slice's ~1 Hz volumetric rate (calcium-appropriate; declared,
not tuned). Readout on a DECLARED subsample: 5,000 neurons drawn with seed
0, plus the ghost. Everything else is the validated pipeline: z-score,
first-difference, poly-3 self-baseline, ridge alpha 1.0, excess = joint
minus self on the held-out tail.

    (WSL) python scripts/zapbench_feasibility.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

E = 3
TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2


def poly3(own: np.ndarray) -> np.ndarray:
    cols = [own]
    e = own.shape[1]
    for i in range(e):
        for j in range(i, e):
            cols.append((own[:, i] * own[:, j])[:, None])
    for i in range(e):
        for j in range(i, e):
            for k in range(j, e):
                cols.append((own[:, i] * own[:, j] * own[:, k])[:, None])
    return np.hstack(cols)


def r2_clamped(pred, truth):
    err = float(np.mean((pred - truth) ** 2))
    return max(0.0, 1.0 - err / (float(np.var(truth)) + 1e-12))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", default="Data/zapbench/traces_5638_7879.npy")
    p.add_argument("--bottleneck", type=int, default=64)
    p.add_argument("--units", type=int, default=64)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--n-models", type=int, default=2)
    p.add_argument("--readout-sample", type=int, default=5000)
    p.add_argument("--outdir", default="ExpOutput/zapbench_feas")
    args = p.parse_args()

    import keras
    print("devices:", [d.name for d in
                       __import__("tensorflow").config.list_physical_devices()])

    t0 = time.time()
    x = np.load(args.traces)                       # (t, V) float32
    n_t, V = x.shape
    mu, sd = x.mean(0), x.std(0) + 1e-12
    z = ((x - mu) / sd).astype(np.float32)
    del x, mu, sd                                  # host RAM is the scarce resource
    z = np.diff(z, axis=0)
    # Ghost: shifted copy of a random neuron, appended as column V.
    rng = np.random.default_rng(7331)
    donor = int(rng.integers(0, V))
    ghost_col = np.roll(z[:, donor], int(rng.integers(len(z)//4, 3*len(z)//4)))
    z = np.hstack([z, ghost_col[:, None]])
    v_all = V + 1
    # Lag-major joint state: [Z_t | Z_{t-1} | Z_{t-2}].
    zs = np.hstack([z[2:], z[1:-1], z[:-2]]).astype(np.float32)
    n = len(zs)
    a = int(TRAIN_FRACTION * n); b = int((TRAIN_FRACTION + VAL_FRACTION) * n)
    tr, va, te = slice(0, a - E), slice(a, b - E), slice(b, n)
    smu, ssd = zs[tr].mean(0), zs[tr].std(0) + 1e-12
    zs -= smu                                      # in place: no second 1.9 GB copy
    zs /= ssd
    print(f"joint state {zs.shape} ({zs.nbytes/1e9:.2f} GB) "
          f"built in {time.time()-t0:.0f}s")

    D = zs.shape[1]
    mask_frac = 0.25

    class FishAE(keras.Model):
        def __init__(self):
            super().__init__()
            self.encoder = keras.Sequential([
                keras.layers.Input(shape=(D,)),
                keras.layers.Dense(args.units, activation="tanh"),
                keras.layers.Dense(args.bottleneck),
            ])
            self.decoder = keras.Sequential([
                keras.layers.Input(shape=(args.bottleneck,)),
                keras.layers.Dense(args.units, activation="tanh"),
                keras.layers.Dense(D),
            ])

        def call(self, inp, training=False):
            if training:
                keep = keras.ops.cast(
                    keras.random.uniform((keras.ops.shape(inp)[0], v_all))
                    > mask_frac, inp.dtype)
                keep = keras.ops.tile(keep, (1, E))     # lag-major layout
                corrupted = inp * keep
                out = self.decoder(self.encoder(corrupted))
                return out * (1.0 - keep) + inp * keep  # masked-only loss
            return self.decoder(self.encoder(inp))

    outdir = Path(args.outdir)
    (outdir / "models").mkdir(parents=True, exist_ok=True)

    codes = []
    for m_i in range(args.n_models):
        t0 = time.time()
        keras.utils.set_random_seed(3000 + m_i)
        model = FishAE()
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-3))
        model.fit(zs[tr], zs[tr], epochs=args.epochs, batch_size=16,
                  shuffle=True, verbose=0)
        model.save_weights(outdir / "models" / f"m{m_i}.weights.h5")
        # predict() ships the WHOLE array to the GPU as one tensor (1.9 GB)
        # and dies post-training; encode in host-side chunks instead.
        code = np.concatenate([
            np.asarray(model.encoder(zs[i:i + 128]))
            for i in range(0, len(zs), 128)])
        codes.append(code)
        print(f"model {m_i}: trained+encoded in {time.time()-t0:.0f}s")

    # Readout on the declared subsample + ghost.
    if args.readout_sample >= V or args.readout_sample == 0:
        sample = list(range(V))
    else:
        sample = list(rng.choice(V, size=args.readout_sample, replace=False))
    sample.append(V)                                   # the ghost
    lead = z[2:]                                       # aligned with zs rows
    tr_idx = np.arange(tr.start, tr.stop - 1)
    te_idx = np.arange(te.start, n - 1)

    from sklearn.linear_model import Ridge

    def own_feats(q):
        return np.column_stack([zs[:, q], zs[:, v_all + q], zs[:, 2*v_all + q]])

    def fit_pair(q, extra):
        own_p = poly3(own_feats(q))
        src_tr, src_te = own_p[tr_idx], own_p[te_idx]
        if extra is not None:
            src_tr = np.hstack([src_tr, extra[tr_idx]])
            src_te = np.hstack([src_te, extra[te_idx]])
        m = Ridge(alpha=1.0)
        m.fit(src_tr, lead[tr_idx + 1, q])
        return r2_clamped(m.predict(src_te), lead[te_idx + 1, q])

    t0 = time.time()
    self_r2 = {q: fit_pair(q, None) for q in sample}
    t_self = time.time() - t0
    excess = {}
    t0 = time.time()
    for q in sample:
        vals = [fit_pair(q, code) - self_r2[q] for code in codes]
        excess[q] = float(np.mean(vals))
    t_joint = time.time() - t0
    per_neuron = (t_self + t_joint) / len(sample)

    ex = np.array([excess[q] for q in sample[:-1]])
    ghost_ex = excess[V]
    order = np.argsort(-ex)
    np.save(outdir / "excess_sample.npy",
            np.column_stack([sample[:-1], ex]))

    print("\n" + "=" * 76)
    print(f"FISH FEASIBILITY ({args.readout_sample} of {V} neurons, "
          f"{args.n_models} models)")
    print("=" * 76)
    print(f"ghost excess: {ghost_ex:+.4f}")
    print(f"sample excess: mean {ex.mean():+.4f}  p95 {np.percentile(ex,95):+.4f}"
          f"  max {ex.max():+.4f}  frac>0.01 {float((ex>0.01).mean()):.3f}")
    print(f"self-baseline mean {np.mean([self_r2[q] for q in sample[:-1]]):.3f}")
    print(f"readout cost {per_neuron*1e3:.1f} ms/neuron "
          f"-> full 71,721 x {args.n_models} models ~ "
          f"{per_neuron*V/60:.0f} min")
    print(f"top sample-neuron excess values: "
          f"{[f'{ex[i]:+.3f}' for i in order[:8]]}")
    print(f"wrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
