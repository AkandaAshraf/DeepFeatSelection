"""Which regions does the atmosphere drive? Daily SLP, 77 years, 10,512 cells.

The climate deployment, pre-registered in paper/validation_protocol.md before
the data was opened. NCEP/NCAR Reanalysis-1 daily sea-level pressure,
1948-2024, 2.5-degree grid (73 x 144 = 10,512 cells), n ~ 28,000 days --
both dimensions inside the validated envelope, and the substrate where every
classical scan is unaffordable.

Pre-registered:
1. Ghost ~ 0 (validity gate).
2. THE SUNSPOT CHANNEL ~ 0: the SILSO daily sunspot number rides along as a
   real exogenous forcing -- a true root, and roots are invisible by design.
   The chamber's actuators and the worm's silenced AVA approximated this;
   solar forcing is the cleanest natural version that exists.
3. The driven core is spatially CLUSTERED (nearest-neighbour test on the
   sphere vs random cells), not scattered.
Exploratory, no prior claim: WHERE the core sits -- reported as zonal/land
summaries for others to interpret.

Declared: day-of-year climatology removed per cell, z-scored, first-
differenced; E=3, tau=1 day, lag-major; 4-encoder ensemble (128 units,
bottleneck 64 -- the GPU has headroom tonight); poly-3 ridge excess.

    (WSL) python scripts/climate_excess.py
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


def load_slp(root: Path, y0: int, y1: int):
    import netCDF4
    chunks, days = [], []
    for yr in range(y0, y1 + 1):
        ds = netCDF4.Dataset(root / f"slp.{yr}.nc")
        v = np.asarray(ds.variables["slp"][:], dtype=np.float32)
        chunks.append(v.reshape(v.shape[0], -1))
        t = ds.variables["time"]
        days.extend(netCDF4.num2date(t[:], t.units))
        if yr == y0:
            lat = np.asarray(ds.variables["lat"][:])
            lon = np.asarray(ds.variables["lon"][:])
        ds.close()
    x = np.concatenate(chunks, axis=0)
    doy = np.array([min(d.timetuple().tm_yday, 365) for d in days])
    dates = [(d.year, d.month, d.day) for d in days]
    return x, doy, dates, lat, lon


def load_sunspots(path: Path, dates) -> np.ndarray:
    table = {}
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 5:
            continue
        try:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            v = float(parts[4])
        except ValueError:
            continue
        table[(y, m, d)] = v if v >= 0 else np.nan
    s = np.array([table.get(t, np.nan) for t in dates], dtype=np.float64)
    # Missing early values: forward-fill; the channel is a control, not a claim.
    idx = np.where(~np.isnan(s))[0]
    s = np.interp(np.arange(len(s)), idx, s[idx])
    return s


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="Data/climate")
    p.add_argument("--y0", type=int, default=1948)
    p.add_argument("--y1", type=int, default=2024)
    p.add_argument("--units", type=int, default=128)
    p.add_argument("--bottleneck", type=int, default=64)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--n-models", type=int, default=4)
    p.add_argument("--outdir", default="ExpOutput/climate_excess")
    args = p.parse_args()

    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # HARD cap, growth disabled, sized to ~80% of the free
        # VRAM (8 GB card minus ~700 MB display/system = ~7.4 GB free ->
        # 6 GB cap). Fixed allocation, never grows.
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=6144)])
    import keras

    t0 = time.time()
    root = Path(args.root)
    x, doy, dates, lat, lon = load_slp(root, args.y0, args.y1)
    n_t, V = x.shape
    print(f"slp {x.shape} loaded in {time.time()-t0:.0f}s")

    # Deseasonalise per cell by day-of-year climatology, then z-score.
    for d in range(1, 366):
        m = doy == d
        x[m] -= x[m].mean(axis=0, keepdims=True)
    x /= (x.std(axis=0, keepdims=True) + 1e-6)

    sun = load_sunspots(root / "sunspots_daily.csv", dates)
    sun = (sun - sun.mean()) / (sun.std() + 1e-12)
    x = np.hstack([x, sun[:, None].astype(np.float32)])
    sun_idx = V
    z = np.diff(x, axis=0)
    del x

    rng = np.random.default_rng(7331)
    donor = int(rng.integers(0, V))
    ghost_col = np.roll(z[:, donor], int(rng.integers(len(z)//4, 3*len(z)//4)))
    z = np.hstack([z, ghost_col[:, None]])
    v_all = V + 2                       # + sunspots + ghost
    ghost_idx = V + 1

    zs = np.hstack([z[2:], z[1:-1], z[:-2]]).astype(np.float32)
    n = len(zs)
    a = int(TRAIN_FRACTION * n); b = int((TRAIN_FRACTION + VAL_FRACTION) * n)
    tr, va, te = slice(0, a - E), slice(a, b - E), slice(b, n)
    smu, ssd = zs[tr].mean(0), zs[tr].std(0) + 1e-12
    zs -= smu
    zs /= ssd
    D = zs.shape[1]
    print(f"joint state {zs.shape} ({zs.nbytes/1e9:.2f} GB)")

    mask_frac = 0.25

    class AE(keras.Model):
        def __init__(self):
            super().__init__()
            self.encoder = keras.Sequential([
                keras.layers.Input(shape=(D,)),
                keras.layers.Dense(args.units, activation="tanh"),
                keras.layers.Dense(args.bottleneck)])
            self.decoder = keras.Sequential([
                keras.layers.Input(shape=(args.bottleneck,)),
                keras.layers.Dense(args.units, activation="tanh"),
                keras.layers.Dense(D)])

        def call(self, inp, training=False):
            if training:
                keep = keras.ops.cast(
                    keras.random.uniform((keras.ops.shape(inp)[0], v_all))
                    > mask_frac, inp.dtype)
                keep = keras.ops.tile(keep, (1, E))
                out = self.decoder(self.encoder(inp * keep))
                return out * (1.0 - keep) + inp * keep
            return self.decoder(self.encoder(inp))

    outdir = Path(args.outdir)
    (outdir / "models").mkdir(parents=True, exist_ok=True)
    codes = []
    for m_i in range(args.n_models):
        t0 = time.time()
        keras.utils.set_random_seed(4000 + m_i)
        model = AE()
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-3))
        # tf.data fast path: one host-side tensor of the train segment
        # (~2 GB, inside the envelope), shuffled + batched + prefetched so
        # the GPU is compute-bound instead of feed-starved. The map to
        # (b, b) shares the tensor -- no target duplication.
        ds = (tf.data.Dataset.from_tensor_slices(zs[tr.start:tr.stop])
              .shuffle(4096, reshuffle_each_iteration=True)
              .batch(128)
              .map(lambda bch: (bch, bch))
              .prefetch(tf.data.AUTOTUNE))
        model.fit(ds, epochs=args.epochs, verbose=0)
        model.save_weights(outdir / "models" / f"m{m_i}.weights.h5")
        code = np.concatenate([np.asarray(model.encoder(zs[i:i + 256]))
                               for i in range(0, n, 256)])
        codes.append(code)
        print(f"model {m_i}: {time.time()-t0:.0f}s")

    from sklearn.linear_model import Ridge
    lead = z[2:]
    tr_idx = np.arange(tr.start, tr.stop - 1)
    te_idx = np.arange(te.start, n - 1)

    def fit_pair(q, extra):
        own = np.column_stack([zs[:, q], zs[:, v_all + q], zs[:, 2*v_all + q]])
        own_p = poly3(own)
        src_tr, src_te = own_p[tr_idx], own_p[te_idx]
        if extra is not None:
            src_tr = np.hstack([src_tr, extra[tr_idx]])
            src_te = np.hstack([src_te, extra[te_idx]])
        m = Ridge(alpha=1.0)
        m.fit(src_tr, lead[tr_idx + 1, q])
        return r2_clamped(m.predict(src_te), lead[te_idx + 1, q])

    t0 = time.time()
    excess = np.empty(v_all)
    for q in range(v_all):
        s = fit_pair(q, None)
        excess[q] = np.mean([fit_pair(q, c) for c in codes]) - s
        if q % 2000 == 1999:
            print(f"  readout {q+1}/{v_all} ({time.time()-t0:.0f}s)")
    np.save(outdir / "excess.npy", excess)
    np.save(outdir / "latlon.npy",
            np.array(np.meshgrid(lat, lon, indexing="ij")).reshape(2, -1))

    cells = excess[:V]
    print("\n" + "=" * 78)
    print("CLIMATE DRIVENNESS (77 years daily SLP, 10,512 cells)")
    print("=" * 78)
    print(f"1. ghost excess:   {excess[ghost_idx]:+.4f}")
    print(f"2. SUNSPOT excess: {excess[sun_idx]:+.4f}   "
          f"(root-invisibility on real exogenous forcing)")
    print(f"cells: mean {cells.mean():+.4f}  p95 {np.percentile(cells,95):+.4f}"
          f"  max {cells.max():+.4f}  frac>0.01 {(cells>0.01).mean():.3f}")

    # 3. spatial clustering on the unit sphere.
    la = np.repeat(lat, len(lon)); lo = np.tile(lon, len(lat))
    xyz = np.column_stack([np.cos(np.radians(la)) * np.cos(np.radians(lo)),
                           np.cos(np.radians(la)) * np.sin(np.radians(lo)),
                           np.sin(np.radians(la))])
    from scipy.spatial import cKDTree
    top = np.argsort(-cells)[:500]
    def med_nn(ids):
        t = cKDTree(xyz[ids]); d, _ = t.query(xyz[ids], k=2)
        return float(np.median(d[:, 1]))
    rng2 = np.random.default_rng(0)
    null = [med_nn(rng2.choice(V, 500, replace=False)) for _ in range(20)]
    print(f"3. clustering: top-500 median NN {med_nn(top):.4f} vs "
          f"random {np.mean(null):.4f} +/- {np.std(null):.4f}")
    print("\nzonal distribution of top-500 (exploratory):")
    for lo_b, hi_b in ((-90,-60),(-60,-30),(-30,0),(0,30),(30,60),(60,90)):
        frac = float(((la[top] >= lo_b) & (la[top] < hi_b)).mean())
        print(f"  lat {lo_b:+03d}..{hi_b:+03d}: {frac:.2f}")
    print(f"\nwrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
