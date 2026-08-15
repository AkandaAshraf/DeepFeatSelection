"""Does a genuinely narrow bottleneck detect system membership at scale?

The audit that motivated this: across every prior MLP experiment the hidden
layer was as wide as or WIDER than the input (32 units over 8-18 input dims on
pairs; an incidental 2.25x squeeze on the worm). The compressibility
hypothesis -- information forced through a code much smaller than the system,
so that only shared structure survives -- was never actually tested. Every
negative so far judged uncompressed small MLPs.

The target, restated: with thousands to hundreds of thousands of
variables, the question a researcher can actually ask is not the full V x V
edge matrix but "is THIS variable causally coupled to the system at all?" --
one-vs-rest membership. Pairwise CCM needs V runs per query and V^2 for a
scan; a masked autoencoder answers every query from ONE model. This script
tests whether that works where truth is known, and how it depends on the
compression ratio D/b, which is swept deliberately for the first time.

DESIGN
- System: V logistic maps; a minority sub-web wired as a sparse DAG at
  coupling 0.3 (members), the rest fully autonomous (non-members). Truth =
  has at least one edge. A circularly shifted ghost channel is appended as a
  known non-member with realistic marginals.
- ONE masked autoencoder per bottleneck width: input all V channels
  (delay-embedded, D = V*E dims), a single linear bottleneck of width b, and
  during training a random subset of channels is zeroed at the input each
  batch (masked-AE style) so the model learns to fill channels in from the
  code. No per-variable models, no refits.
- MEMBERSHIP SCORE, absolute not differenced (so the maturity/redundancy
  collapse of LOCO differences does not apply): mask X's channels at input,
  held-out r2 of recreating X's channels at the output. Members share
  dynamics with the web and survive the squeeze; autonomous chaos does not.
- Bottleneck sweep b in {2,4,8,16,32,64} against D, i.e. compression ratios
  from ~1x to ~45x at V=30 -- the axis the hypothesis lives on.
- Checkpoints at {5,25,50,100} epochs: prediction, on record -- the absolute
  readout should NOT collapse with maturity, unlike LOCO differences.
- Baselines: full pairwise CCM membership (max rho over all partners; V-1
  runs per query -- wall-clock reported and projected to V=10^4), and
  CCM-vs-PCA (X cross-mapped against the top principal components of the
  rest -- the cheap classical way to summarise a system; if this matches the
  AE, depth bought nothing again).

    python scripts/bottleneck_membership.py --v 30 --members 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from chamber_detect import r2_of  # noqa: E402
from network_scale import random_dag, simulate  # noqa: E402
from deepfeatselect.ccm import ccm, time_delay_embed  # noqa: E402

E = 3
TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2


def build_system(v: int, members: int, n: int, coupling: float, seed: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """V series; the first ``members`` form a sparse web, the rest are
    autonomous. Membership truth vector returned alongside."""
    rng = np.random.default_rng(seed)
    n_edges = max(members, int(1.2 * members))
    for attempt in range(20):
        rng_a = np.random.default_rng(seed + 100 * attempt)
        edges = random_dag(members, n_edges, rng_a)
        try:
            web = simulate(n, edges, members, coupling, seed)
            break
        except ValueError:
            continue
    lone = np.empty((n, v - members))
    r = rng.uniform(3.6, 3.8, size=v - members)
    x = rng.uniform(0.2, 0.8, size=v - members)
    burn = 500
    series = np.empty((burn + n, v - members))
    series[0] = x
    for t in range(burn + n - 1):
        series[t + 1] = r * series[t] * (1.0 - series[t])
    lone = series[burn:]
    joined = np.hstack([web, lone])
    truth = np.array([True] * members + [False] * (v - members))
    # Edge case: a member drawn with no edges is a non-member in truth.
    deg = np.zeros(members, int)
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1
    truth[:members] = deg > 0
    return joined, truth


def build_system_hetero(v: int, members: int, n: int, coupling: float,
                        seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Members = logistic web as before; loners drawn from FOUR families.

    The typicality confound lived on loner homogeneity: one family, one
    manifold, one prior. Here the non-members split evenly across logistic
    maps, sine maps, tent maps and AR(1) noise, so no single family prior
    exists to collapse onto. The AR(1) quarter is deliberately stochastic:
    its noise floor is irreducible for the self-model AND the code, so its
    excess must sit at zero -- a per-family control.
    """
    rng = np.random.default_rng(seed)
    n_edges = max(members, int(1.2 * members))
    for attempt in range(20):
        rng_a = np.random.default_rng(seed + 100 * attempt)
        edges = random_dag(members, n_edges, rng_a)
        try:
            web = simulate(n, edges, members, coupling, seed)
            break
        except ValueError:
            continue
    n_lone = v - members
    burn = 500
    series = np.empty((burn + n, n_lone))
    kind = np.arange(n_lone) % 4
    r_log = rng.uniform(3.6, 3.8, n_lone)
    r_sin = rng.uniform(0.85, 0.99, n_lone)
    r_tent = rng.uniform(1.7, 1.95, n_lone)
    a_ar = rng.uniform(0.5, 0.9, n_lone)
    noise = rng.standard_normal((burn + n, n_lone)) * 0.3
    series[0] = rng.uniform(0.2, 0.8, n_lone)
    for t in range(burn + n - 1):
        x = series[t]
        nxt = np.where(kind == 0, r_log * x * (1 - x),
              np.where(kind == 1, r_sin * np.sin(np.pi * np.clip(x, 0, 1)),
              np.where(kind == 2, r_tent * np.minimum(np.clip(x, 0, 1),
                                                      1 - np.clip(x, 0, 1)),
                       a_ar * x + noise[t])))
        series[t + 1] = nxt
    lone = series[burn:]
    joined = np.hstack([web, lone])
    truth = np.array([True] * members + [False] * n_lone)
    deg = np.zeros(members, int)
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1
    truth[:members] = deg > 0
    return joined, truth


def splits_for(n: int) -> tuple[slice, slice, slice]:
    a = int(TRAIN_FRACTION * n)
    b = int((TRAIN_FRACTION + VAL_FRACTION) * n)
    return slice(0, a - E), slice(a, b - E), slice(b, n)


class MaskedAE(keras.Model):
    """Autoencoder with a hard linear bottleneck and train-time channel masking.

    Masking teaches the decoder to fill any channel in from the code, which is
    what makes ONE model answer every membership query at test time.
    """

    def __init__(self, v: int, e: int, b: int, mask_frac: float = 0.25,
                 mask_mode: str = "zero", loss_on_masked_only: bool = False):
        super().__init__()
        self.v, self.e, self.mask_frac = v, e, mask_frac
        self.loss_on_masked_only = loss_on_masked_only
        # "zero" fills the masked slot with the mean (zero after z-scoring),
        # which is a weakly INFORMATIVE observation the decoder can lean on.
        # "uniform" fills it with U(-2,2) noise: genuinely uninformative, so
        # recreation can only come from the code. The
        # ablation between the two measures how much the zero-mask was
        # handing the decoder for free.
        self.mask_mode = mask_mode
        d = v * e
        self.encoder = keras.Sequential([
            keras.layers.Input(shape=(d,)),
            keras.layers.Dense(max(2 * b, 64), activation="tanh"),
            keras.layers.Dense(b),
        ])
        self.decoder = keras.Sequential([
            keras.layers.Input(shape=(b,)),
            keras.layers.Dense(max(2 * b, 64), activation="tanh"),
            keras.layers.Dense(d),
        ])

    def call(self, x, training=False):
        if training:
            keep = keras.ops.cast(
                keras.random.uniform((keras.ops.shape(x)[0], self.v))
                > self.mask_frac, x.dtype)
            keep = keras.ops.repeat(keep, self.e, axis=1)
            corrupted = x * keep
            if self.mask_mode == "uniform":
                noise = keras.random.uniform(keras.ops.shape(x), -2.0, 2.0,
                                             dtype=x.dtype)
                corrupted = corrupted + noise * (1.0 - keep)
            out = self.decoder(self.encoder(corrupted))
            if self.loss_on_masked_only:
                # The MAE recipe (He et al.): gradient only from masked
                # positions, so the code specialises in CROSS-channel
                # information -- the causal content -- rather than
                # self-compression of the visible channels, which is the
                # easiest way to cut all-channel MSE and exactly the
                # generic-manifold artifact the ghost control exposed at
                # V=1000. Implemented by passing gradients only where
                # keep==0: visible positions contribute their target value
                # (zero loss) instead of their prediction.
                out = out * (1.0 - keep) + x * keep
            return out
        return self.decoder(self.encoder(x))


def membership_scores(model: MaskedAE, zs: np.ndarray, te: slice,
                      v: int, rng: np.random.Generator | None = None
                      ) -> np.ndarray:
    """Per-variable held-out recreation r2 with that variable masked at input.

    Under uniform masking the test-time fill is averaged over TEN independent
    noise draws, stacked into one forward pass per
    variable so the averaging costs one predict call, not ten.
    """
    out = np.empty(v)
    base = zs[te]
    draws = 10 if model.mask_mode == "uniform" else 1
    for x in range(v):
        stacked = np.tile(base, (draws, 1))
        for d in range(draws):
            block = slice(d * len(base), (d + 1) * len(base))
            if model.mask_mode == "uniform":
                stacked[block, x * E:(x + 1) * E] = rng.uniform(
                    -2.0, 2.0, size=(len(base), E)).astype(base.dtype)
            else:
                stacked[block, x * E:(x + 1) * E] = 0.0
        pred = model.predict(stacked, verbose=0)
        r2s = [r2_of(pred[d * len(base):(d + 1) * len(base),
                          x * E:(x + 1) * E],
                     base[:, x * E:(x + 1) * E]) for d in range(draws)]
        out[x] = float(np.mean(r2s))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--v", type=int, default=30)
    p.add_argument("--members", type=int, default=10)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--coupling", type=float, default=0.3)
    p.add_argument("--bottlenecks", type=int, nargs="+",
                   default=[2, 4, 8, 16, 32, 64])
    p.add_argument("--epochs-grid", type=int, nargs="+",
                   default=[5, 25, 50, 100])
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--mask-mode", choices=["zero", "uniform"], default="zero")
    p.add_argument("--masked-loss", action="store_true",
                   help="MAE-style: loss on masked positions only.")
    p.add_argument("--skip-full-ccm", action="store_true",
                   help="At large V the full pairwise scan is the thing that "
                        "does not scale; skip it and run only CCM-vs-PCA.")
    p.add_argument("--outdir", default="ExpOutput/membership")
    args = p.parse_args()
    grid = sorted(args.epochs_grid)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    for seed in range(args.seeds):
        x, truth = build_system(args.v, args.members, args.n, args.coupling,
                                seed)
        v = args.v
        mats = [time_delay_embed(x[:, j], E)[0] for j in range(v)]
        n = min(len(m) for m in mats)
        joint = np.hstack([m[:n] for m in mats]).astype("float64")
        # Ghost channel: shifted copy of a member -- must score as non-member.
        rng = np.random.default_rng(seed + 7331)
        donor = int(rng.integers(0, args.members))
        ghost = np.roll(joint[:, donor * E:(donor + 1) * E],
                        int(rng.integers(n // 4, 3 * n // 4)), axis=0)
        joint = np.hstack([joint, ghost])
        v_all = v + 1
        truth_all = np.append(truth, False)

        tr, va, te = splits_for(n)
        mu, sd = joint[tr].mean(0), joint[tr].std(0) + 1e-12
        zs = ((joint - mu) / sd).astype("float32")
        D = zs.shape[1]
        base = float(truth_all.mean())
        print(f"\nseed {seed}: V={v_all} (with ghost), D={D}, "
              f"{int(truth_all.sum())} members, baseline {base:.3f}")

        for b in args.bottlenecks:
            if b >= D:
                continue
            keras.utils.set_random_seed(seed)
            model = MaskedAE(v_all, E, b, mask_mode=args.mask_mode,
                             loss_on_masked_only=args.masked_loss)
            model.compile(loss="mse", optimizer=keras.optimizers.Adam(3e-3))
            t0 = time.time()
            done = 0
            for e_stop in grid:
                model.fit(zs[tr], zs[tr], validation_data=(zs[va], zs[va]),
                          epochs=e_stop - done, batch_size=64, shuffle=True,
                          verbose=0)
                done = e_stop
                # Weights persist so any future readout (different fill,
                # more draws, new statistic) is an evaluation, not a retrain.
                wdir = outdir / "models"
                wdir.mkdir(exist_ok=True)
                model.save_weights(
                    wdir / f"s{seed}_b{b}_{args.mask_mode}_e{e_stop}.weights.h5")
                scores = membership_scores(model, zs, te, v_all,
                                           np.random.default_rng(seed + 99))
                rows.append({
                    "seed": seed,
                    "method": f"ae_b{b}_{args.mask_mode}"
                              + ("_mloss" if args.masked_loss else ""),
                    "bottleneck": b,
                    "compression": D / b, "epochs": e_stop,
                    "prauc": average_precision_score(truth_all, scores),
                    "auroc": roc_auc_score(truth_all, scores),
                    "ghost_score": scores[-1],
                    "member_mean": float(scores[truth_all].mean()),
                    "lone_mean": float(scores[~truth_all].mean()),
                    "seconds": time.time() - t0})
                r = rows[-1]
                print(f"  b={b:<3} ({r['compression']:.0f}x) e={e_stop:<4}"
                      f" ap {r['prauc']:.3f} auroc {r['auroc']:.3f} "
                      f"members {r['member_mean']:+.3f} lone "
                      f"{r['lone_mean']:+.3f} ghost {r['ghost_score']:+.3f}")

        # --- CCM-vs-PCA: the scalable classical baseline ---
        t0 = time.time()
        pca_scores = np.empty(v_all)
        for q in range(v_all):
            others = np.delete(zs[:, :], slice(q * E, (q + 1) * E), axis=1)
            code = PCA(n_components=min(8, others.shape[1])).fit(
                others[tr]).transform(others)[:, 0]
            r = ccm(joint[:, q * E], code.astype(np.float64), E=E, seed=seed)
            pca_scores[q] = max(r.x_causes_y.rho_at_max_lib,
                                r.y_causes_x.rho_at_max_lib)
        rows.append({"seed": seed, "method": "ccm_pca", "bottleneck": np.nan,
                     "compression": np.nan, "epochs": np.nan,
                     "prauc": average_precision_score(truth_all, pca_scores),
                     "auroc": roc_auc_score(truth_all, pca_scores),
                     "ghost_score": pca_scores[-1],
                     "member_mean": float(pca_scores[truth_all].mean()),
                     "lone_mean": float(pca_scores[~truth_all].mean()),
                     "seconds": time.time() - t0})
        print(f"  ccm_pca: ap {rows[-1]['prauc']:.3f} "
              f"auroc {rows[-1]['auroc']:.3f} ({rows[-1]['seconds']:.0f}s)")

        if not args.skip_full_ccm:
            t0 = time.time()
            full_scores = np.zeros(v_all)
            raw = [joint[:, q * E] for q in range(v_all)]
            for i in range(v_all):
                best = -np.inf
                for j in range(v_all):
                    if i == j:
                        continue
                    r = ccm(raw[i], raw[j], E=E, seed=seed)
                    best = max(best, r.x_causes_y.rho_at_max_lib,
                               r.y_causes_x.rho_at_max_lib)
                full_scores[i] = best
            secs = time.time() - t0
            pairs = v_all * (v_all - 1)
            rows.append({"seed": seed, "method": "ccm_full",
                         "bottleneck": np.nan, "compression": np.nan,
                         "epochs": np.nan,
                         "prauc": average_precision_score(truth_all, full_scores),
                         "auroc": roc_auc_score(truth_all, full_scores),
                         "ghost_score": full_scores[-1],
                         "member_mean": float(full_scores[truth_all].mean()),
                         "lone_mean": float(full_scores[~truth_all].mean()),
                         "seconds": secs})
            per_pair = secs / pairs
            print(f"  ccm_full: ap {rows[-1]['prauc']:.3f} auroc "
                  f"{rows[-1]['auroc']:.3f} ({secs:.0f}s; {per_pair*1e3:.0f} "
                  f"ms/pair -> projected {per_pair * 1e8 / 3600:.0f} h for "
                  f"V=10,000 full scan)")

    frame = pd.DataFrame(rows)
    frame.to_csv(outdir / "membership.csv", index=False)
    print("\n" + "=" * 90)
    print("MEMBERSHIP DETECTION BY COMPRESSION RATIO (mean over seeds)")
    print("=" * 90)
    with pd.option_context("display.float_format", "{:.3f}".format,
                           "display.width", 160):
        print(frame.groupby(["method", "epochs"], dropna=False)[
            ["compression", "prauc", "auroc", "member_mean", "lone_mean",
             "ghost_score"]].mean().to_string())
    print(f"\nwrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
