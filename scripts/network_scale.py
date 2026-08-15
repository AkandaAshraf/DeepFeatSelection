"""Does a learned estimator earn its place when the network gets wide?

Every head-to-head so far has ended the same way: on pairs and on a three-node
chain, local geometry (kNN, simplex/CCM) estimates the causal imprint better
than anything trained.  This script builds the regime where local geometry
should genuinely starve and asks the question once more there.

Ten logistic maps wired as a random sparse DAG.  Detection is by recreation
with leave-one-source-out: to score the undirected link {U, V}, recreate U's
delay embedding from the other nine nodes' embeddings, with and without V.  The
conditioning input is 18-dimensional, and that dimension is the point -- a kNN
regressor's neighbourhoods thin out exponentially with conditioning dimension,
while a trained map's sample-efficiency should not degrade the same way.  If
the learned estimator is ever going to win, it is here; if it loses here too,
the negative is close to complete.

Both operators get the identical structure: for each target U, one full fit on
all nine sources, then nine refits each dropping one source V.

    gain_V(U) = r2_full(U) - r2_minus_V(U)
    link score {U,V} = max(gain_V(U), gain_U(V))     # detection, not orientation

The max over directions is applied identically to every pair, controls
included, so nothing is flipped post hoc.  Pairwise CCM (max of the two
directional rhos) runs on the same series as the incumbent: it never faces the
conditioning problem, but it pays the indirect-path tax instead -- on a DAG,
2-hop-connected non-adjacent pairs carry transitive imprint that a pairwise
test has no way to discount.

Scoring, fixed a priori: positives = true edges as undirected pairs; negatives
= non-adjacent pairs, with 2-hop pairs reported separately from farther ones.
AUROC over the link scores, per method.  A matched NULL NETWORK -- zero edges,
same rng sequence -- calibrates the noise floor: its maximum link score is the
threshold any real detection has to clear, and its AUROC has no positives by
construction so it contributes the floor, not a score.

Guard note: recreation of U from the *other* nodes is expected to fail (r2 near
zero) when no other node carries information about U -- for example any node
with neither parents nor children.  That is signal, not an unlearnable arm, so
full-model r2 is reported per target but does not gate.

    python scripts/network_scale.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

from deepfeatselect.ccm import ccm, time_delay_embed

TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2
E = 2          # per node; a logistic map is one-dimensional, two lags suffice
BURN_IN = 500
# 3.8 ceiling, matching the verified three-node generators: at r=3.9 a node
# with two parents' summed drive sends the growth term negative and the map
# leaves [0,1] (measured: divergence at coupling 0.35, seed 0).
R_LOW, R_HIGH = 3.6, 3.8
MAX_IN_DEGREE = 2   # keeps the summed coupling term inside the divergence-safe
                    # range measured for this map form (diverges above ~1.0)


def random_dag(n_nodes: int, n_edges: int, rng: np.random.Generator
               ) -> list[tuple[int, int]]:
    """A random DAG in topological order: edges only run low index -> high.

    In-degree is capped so the summed coupling never leaves the regime the
    divergence probe showed to be safe for this map family.
    """
    candidates = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    rng.shuffle(candidates)
    edges: list[tuple[int, int]] = []
    in_deg = np.zeros(n_nodes, dtype=int)
    for i, j in candidates:
        if len(edges) == n_edges:
            break
        if in_deg[j] < MAX_IN_DEGREE:
            edges.append((i, j))
            in_deg[j] += 1
    return sorted(edges)


def simulate(n: int, edges: list[tuple[int, int]], n_nodes: int,
             coupling: float, seed: int) -> np.ndarray:
    """N coupled logistic maps in the coupled_logistic functional form.

    Each coupling enters as the driver's value subtracted inside the response's
    growth term, exactly as in deepfeatselect.synthetic.  The rng call sequence
    is identical whatever the edge list, so the zero-edge null network is the
    same system minus only the coupling -- a matched control by construction.

    Raises ValueError on divergence rather than clipping; clipped series would
    fabricate dynamics that were never simulated.
    """
    rng = np.random.default_rng(seed)
    r = rng.uniform(R_LOW, R_HIGH, size=n_nodes)
    total = BURN_IN + n
    x = np.empty((total, n_nodes), dtype=np.float64)
    x[0] = rng.uniform(0.1, 0.9, size=n_nodes)
    parents: list[list[int]] = [[] for _ in range(n_nodes)]
    for i, j in edges:
        parents[j].append(i)

    for t in range(total - 1):
        for j in range(n_nodes):
            drive = sum(coupling * x[t, p] for p in parents[j])
            x[t + 1, j] = x[t, j] * (r[j] - r[j] * x[t, j] - drive)

    if not np.isfinite(x).all() or x.min() < 0.0 or x.max() > 1.0:
        raise ValueError(f"network diverged at coupling={coupling}")
    return x[BURN_IN:]


def embed_all(x: np.ndarray) -> np.ndarray:
    """Per-node delay embeddings, time-aligned: (n_points, n_nodes * E)."""
    mats = [time_delay_embed(x[:, j], E)[0] for j in range(x.shape[1])]
    n = min(len(m) for m in mats)
    return np.hstack([m[:n] for m in mats]).astype("float32")


def contiguous_splits(n: int, embargo: int) -> tuple[slice, slice, slice]:
    a = int(TRAIN_FRACTION * n)
    b = int((TRAIN_FRACTION + VAL_FRACTION) * n)
    return slice(0, a - embargo), slice(a, b - embargo), slice(b, n)


def node_cols(j: int) -> slice:
    return slice(j * E, (j + 1) * E)


def r2_of(pred: np.ndarray, truth: np.ndarray) -> float:
    """Held-out r2, clamped at zero.

    Below zero the model is worse than predicting the mean, i.e. it cannot
    recreate the target at all, and differences between two such values are
    optimisation noise, not information.  Unclamped, the smoke run produced a
    null-network "gain" of +1.20 -- larger than the whole r2 scale -- from a
    drop of one node moving r2 from -3.2 to -4.4.  Clamping bounds every gain
    by what was actually recreated.  Declared before the full run.
    """
    err = float(np.mean((pred - truth) ** 2))
    return max(0.0, 1.0 - err / (float(np.var(truth)) + 1e-12))


def fit_mlp(src: np.ndarray, dst: np.ndarray, splits, seed: int, args) -> float:
    """Held-out r2 of one small recreation model.  Short training by design."""
    tr, va, te = splits
    keras.utils.set_random_seed(seed)
    m = keras.Sequential([
        keras.layers.Input(shape=(src.shape[1],)),
        keras.layers.Dense(args.units, activation="tanh"),
        keras.layers.Dense(args.units, activation="tanh"),
        keras.layers.Dense(dst.shape[1]),
    ])
    m.compile(loss="mse", optimizer=keras.optimizers.Adam(3e-3))
    m.fit(src[tr], dst[tr], validation_data=(src[va], dst[va]),
          epochs=args.epochs, batch_size=64, shuffle=True, verbose=0,
          callbacks=[keras.callbacks.EarlyStopping(
              monitor="val_loss", patience=args.patience,
              restore_best_weights=True)])
    return r2_of(m.predict(src[te], verbose=0), dst[te])


def fit_knn(src: np.ndarray, dst: np.ndarray, splits) -> float:
    tr, _, te = splits
    m = KNeighborsRegressor(n_neighbors=5)
    m.fit(src[tr], dst[tr])
    return r2_of(m.predict(src[te]), dst[te])


def loco_gains(z: np.ndarray, n_nodes: int, operator: str, seed: int,
               args) -> tuple[np.ndarray, np.ndarray]:
    """gain[v, u] = r2_full(u) - r2_minus_v(u), plus the full r2 per target.

    Standardisation from train statistics only; the same splits serve every
    fit, so operators and ablations differ in nothing but the input columns.
    """
    splits = contiguous_splits(len(z), embargo=E)
    mu, sd = z[splits[0]].mean(axis=0), z[splits[0]].std(axis=0) + 1e-12
    zs = (z - mu) / sd

    gains = np.full((n_nodes, n_nodes), np.nan)
    full_r2 = np.empty(n_nodes)
    for u in range(n_nodes):
        others = [v for v in range(n_nodes) if v != u]
        dst = zs[:, node_cols(u)]

        def run(sources: list[int]) -> float:
            src = np.hstack([zs[:, node_cols(v)] for v in sources])
            if operator == "mlp":
                return fit_mlp(src, dst, splits, seed, args)
            return fit_knn(src, dst, splits)

        base = run(others)
        full_r2[u] = base
        for v in others:
            gains[v, u] = base - run([w for w in others if w != v])
    return gains, full_r2


def ccm_scores(x: np.ndarray, seed: int) -> np.ndarray:
    """score[i, j] for i < j: max of the two directional cross-map rhos."""
    n_nodes = x.shape[1]
    out = np.full((n_nodes, n_nodes), np.nan)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            r = ccm(x[:, i], x[:, j], E=3, seed=seed)
            out[i, j] = max(r.x_causes_y.rho_at_max_lib,
                            r.y_causes_x.rho_at_max_lib)
    return out


def pair_table(edges: list[tuple[int, int]], n_nodes: int,
               scores: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per unordered pair with its truth class and every method's score."""
    edge_set = {tuple(sorted(e)) for e in edges}
    adjacency: dict[int, set[int]] = {i: set() for i in range(n_nodes)}
    for i, j in edge_set:
        adjacency[i].add(j)
        adjacency[j].add(i)

    rows = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if (i, j) in edge_set:
                klass = "edge"
            elif adjacency[i] & adjacency[j]:
                klass = "two_hop"
            else:
                klass = "far"
            row = {"i": i, "j": j, "klass": klass}
            for name, mat in scores.items():
                if name == "ccm":
                    row[name] = mat[i, j]
                else:
                    row[name] = max(mat[i, j], mat[j, i])
            rows.append(row)
    return pd.DataFrame(rows)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUROC without sklearn, robust to small counts."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--nodes", type=int, default=10)
    p.add_argument("--edges", type=int, default=12)
    p.add_argument("--coupling", type=float, default=0.35)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--units", type=int, default=32)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--outdir", default="ExpOutput/network_scale")
    args = p.parse_args()

    all_pairs, summaries = [], []
    for seed in range(args.seeds):
        # A drawn graph can still diverge; discard and redraw rather than
        # clip, and say so -- the ensemble is conditioned on stability, which
        # is honest as long as it is visible.
        for attempt in range(20):
            rng = np.random.default_rng(1000 + seed + 100 * attempt)
            edges = random_dag(args.nodes, args.edges, rng)
            try:
                simulate(args.n, edges, args.nodes, args.coupling, seed)
                break
            except ValueError:
                print(f"  seed {seed}: graph attempt {attempt} diverged, redrawing")
        for label, edge_list in (("network", edges), ("null", [])):
            x = simulate(args.n, edge_list, args.nodes, args.coupling, seed)
            z = embed_all(x)

            t0 = time.time()
            knn_g, knn_full = loco_gains(z, args.nodes, "knn", seed, args)
            knn_s = time.time() - t0
            t0 = time.time()
            mlp_g, mlp_full = loco_gains(z, args.nodes, "mlp", seed, args)
            mlp_s = time.time() - t0
            t0 = time.time()
            ccm_m = ccm_scores(x, seed)
            ccm_s = time.time() - t0

            pairs = pair_table(edges, args.nodes,
                               {"knn": knn_g, "mlp": mlp_g, "ccm": ccm_m})
            pairs["system"] = label
            pairs["seed"] = seed
            all_pairs.append(pairs)
            summaries.append({
                "system": label, "seed": seed, "n_edges": len(edge_list),
                "knn_full_r2_mean": float(knn_full.mean()),
                "mlp_full_r2_mean": float(mlp_full.mean()),
                "knn_seconds": knn_s, "mlp_seconds": mlp_s,
                "ccm_seconds": ccm_s,
            })
            print(f"  seed {seed} {label}: knn {knn_s:.0f}s  mlp {mlp_s:.0f}s "
                  f"ccm {ccm_s:.0f}s  (full r2: knn {knn_full.mean():+.3f} "
                  f"mlp {mlp_full.mean():+.3f})")

    frame = pd.concat(all_pairs, ignore_index=True)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outdir / "pairs.csv", index=False)
    pd.DataFrame(summaries).to_csv(outdir / "summary.csv", index=False)

    print("\n" + "=" * 96)
    print(f"LINK DETECTION ON A {args.nodes}-NODE DAG "
          f"({args.edges} edges, coupling {args.coupling})")
    print("=" * 96)
    net = frame[frame.system == "network"]
    nul = frame[frame.system == "null"]
    for m in ("knn", "mlp", "ccm"):
        pos = net[net.klass == "edge"][m].to_numpy()
        neg_all = net[net.klass != "edge"][m].to_numpy()
        neg_2h = net[net.klass == "two_hop"][m].to_numpy()
        floor = nul[m].max()
        above = float((pos > floor).mean())
        print(f"  {m:<4} AUROC(all) {auroc(pos, neg_all):.3f}   "
              f"AUROC(vs 2-hop) {auroc(pos, neg_2h):.3f}   "
              f"null floor {floor:+.3f}   edges above floor {above:.0%}")
    print("\n  AUROC(vs 2-hop) is the discriminating number: 2-hop pairs carry")
    print("  transitive imprint, so a pairwise test pays a tax there that a")
    print("  conditional one should not.")
    print(f"\nwrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
