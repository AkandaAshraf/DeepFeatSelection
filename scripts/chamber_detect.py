"""Causal detection on the wind-tunnel chamber: real physics, known graph.

First contact with data this project did not generate. The wind tunnel of
Gamella, Peters & Buhlmann (Nature Machine Intelligence 2025) is a physical
device whose 32-variable causal graph is known by construction and verified by
intervention; ``wt_walks_v1`` drives its actuators with random walks and
records everything at speed. Every choice below was fixed in
``paper/validation_protocol.md`` before this script produced a number.

Methods, frozen from the internal validation:

* MLP-LOCO recreation gain (primary), kNN-LOCO (cheap control) -- recreate
  node U's delay embedding from the other varying nodes', with and without V.
  Directed score U->V = gain of V in recreating U: the effect carries the
  cause's imprint, so the downstream variable helps recreate the upstream one.
* Pairwise CCM at E=3: score U->V = rho of recovering U from V's embedding.
* PCMCI (tigramite, ParCorr, tau_max=3): the field standard, run locally;
  score U->V = max |val| over lags.

Null calibration by GHOST SOURCES (protocol's declared adaptation): each
target's LOCO round includes a circularly shifted copy of a randomly chosen
real variable.  A ghost keeps marginals and autocorrelation but loses temporal
alignment, so gain attributed to it is noise; the floor is the max ghost gain.

Free false-positive counter, declared in advance: actuators (hatch, pot_1,
pot_2, load_in, load_out, and the osr_*/v_* settings) are exogenous -- driven
by the experiment script, not by the tunnel -- so ANY confident incoming edge
to an actuator is a countable error, no judgement involved.

    python scripts/chamber_detect.py --experiments actuators_random_walk_1
"""

from __future__ import annotations

import argparse
import io
import contextlib
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

from deepfeatselect.ccm import ccm, time_delay_embed

TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2
E = 3          # per variable, fixed for every arm; declared in the protocol
TAU_MAX = 3    # PCMCI's lag horizon, matched to E
MIN_STD = 1e-6  # below this a column is constant in this experiment: excluded


def load_experiment(name: str, root: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One experiment's dataframe plus the ground-truth adjacency, quietly.

    The package prints a citation banner and download progress on import and
    load; neither belongs in experiment output.
    """
    import truststore
    truststore.inject_into_ssl()
    with contextlib.redirect_stdout(io.StringIO()):
        from causalchamber.datasets import Dataset
        import causalchamber.ground_truth as gt
        data = Dataset("wt_walks_v1", root=root, download=True)
        frame = data.get_experiment(name).as_pandas_dataframe()
        graph = gt.graph(chamber="wt", configuration="standard")
    return frame, graph


def varying_variables(frame: pd.DataFrame, graph: pd.DataFrame) -> list[str]:
    """Ground-truth variables that actually move in this experiment."""
    return [v for v in graph.columns
            if v in frame.columns and float(frame[v].std()) > MIN_STD]


def embed_columns(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    """Time-aligned per-variable delay embeddings, standardised later."""
    mats = [time_delay_embed(frame[v].to_numpy(dtype=np.float64), E)[0]
            for v in names]
    n = min(len(m) for m in mats)
    return np.hstack([m[:n] for m in mats]).astype("float64")


def contiguous_splits(n: int, embargo: int) -> tuple[slice, slice, slice]:
    a = int(TRAIN_FRACTION * n)
    b = int((TRAIN_FRACTION + VAL_FRACTION) * n)
    return slice(0, a - embargo), slice(a, b - embargo), slice(b, n)


def var_cols(i: int) -> slice:
    return slice(i * E, (i + 1) * E)


def r2_of(pred: np.ndarray, truth: np.ndarray) -> float:
    """Clamped at zero: below it the target was not recreated at all."""
    err = float(np.mean((pred - truth) ** 2))
    return max(0.0, 1.0 - err / (float(np.var(truth)) + 1e-12))


def fit_mlp(src, dst, splits, seed, args) -> float:
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


def fit_knn(src, dst, splits) -> float:
    tr, _, te = splits
    m = KNeighborsRegressor(n_neighbors=5)
    m.fit(src[tr], dst[tr])
    return r2_of(m.predict(src[te]), dst[te])


def loco_with_ghost(z: np.ndarray, names: list[str], operator: str,
                    seed: int, args) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """LOCO gains plus one ghost-source null draw per target.

    The ghost is a circular shift of a real variable's embedding: same
    marginals and autocorrelation, no temporal alignment.  It sits in the
    input like any other source, so its LOCO gain is measured by exactly the
    machinery being calibrated.
    """
    splits = contiguous_splits(len(z), embargo=E)
    mu, sd = z[splits[0]].mean(axis=0), z[splits[0]].std(axis=0) + 1e-12
    zs = ((z - mu) / sd).astype("float32")
    V = len(names)
    rng = np.random.default_rng(seed + 7331)

    gains = np.full((V, V), np.nan)   # gains[v, u]: gain of v recreating u
    full_r2 = np.empty(V)
    ghost_gains: list[float] = []

    for u in range(V):
        others = [v for v in range(V) if v != u]
        donor = int(rng.choice(others))
        shift = int(rng.integers(len(zs) // 4, 3 * len(zs) // 4))
        ghost = np.roll(zs[:, var_cols(donor)], shift, axis=0)
        dst = zs[:, var_cols(u)]

        def run(sources: list[int], extra: np.ndarray | None = None) -> float:
            parts = [zs[:, var_cols(v)] for v in sources]
            if extra is not None:
                parts.append(extra)
            src = np.hstack(parts)
            if operator == "mlp":
                return fit_mlp(src, dst, splits, seed, args)
            return fit_knn(src, dst, splits)

        base = run(others, extra=ghost)
        full_r2[u] = base
        for v in others:
            gains[v, u] = base - run([w for w in others if w != v], extra=ghost)
        # The ghost's own LOCO: drop it, keep every real source.
        ghost_gains.append(base - run(others, extra=None))

    return gains, full_r2, ghost_gains


def ccm_matrix(frame: pd.DataFrame, names: list[str], seed: int) -> np.ndarray:
    """score[u, v] = rho of recovering u from v's manifold = evidence u->v."""
    V = len(names)
    out = np.full((V, V), np.nan)
    series = {v: frame[v].to_numpy(dtype=np.float64) for v in names}
    for i in range(V):
        for j in range(i + 1, V):
            r = ccm(series[names[i]], series[names[j]], E=E, seed=seed)
            out[i, j] = r.x_causes_y.rho_at_max_lib
            out[j, i] = r.y_causes_x.rho_at_max_lib
    return out


def pcmci_matrix(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    """PCMCI with ParCorr; score[u, v] = max |val| over lags for u -> v."""
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    import tigramite.data_processing as pp

    data = frame[names].to_numpy(dtype=np.float64)
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-12)
    dataframe = pp.DataFrame(data, var_names=names)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    result = pcmci.run_pcmci(tau_max=TAU_MAX, pc_alpha=None)
    # val_matrix[i, j, tau]: dependence of j at lag 0 on i at lag -tau.
    val = np.abs(result["val_matrix"])[:, :, 1:]   # exclude lag 0: undirected
    return val.max(axis=2)


def evaluate(scores: dict[str, np.ndarray], names: list[str],
             graph: pd.DataFrame, actuators: set[str],
             floors: dict[str, float]) -> pd.DataFrame:
    """AUROC, calibrated detection, and actuator false positives per method."""
    V = len(names)
    truth = np.array([[bool(graph.loc[u, v]) for v in names] for u in names])
    rows = []
    for method, mat in scores.items():
        pos, neg, act_fp = [], [], 0
        floor = floors.get(method)
        for i in range(V):
            for j in range(V):
                if i == j or np.isnan(mat[i, j]):
                    continue
                (pos if truth[i, j] else neg).append(mat[i, j])
                if (floor is not None and names[j] in actuators
                        and mat[i, j] > floor):
                    act_fp += 1
        ranks = pd.Series(pos + neg).rank().to_numpy()
        auroc = ((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)) if pos and neg else np.nan)
        detected = (float(np.mean([s > floor for s in pos]))
                    if floor is not None and pos else np.nan)
        rows.append({"method": method, "auroc": float(auroc),
                     "n_edges": len(pos), "floor": floor,
                     "edges_above_floor": detected,
                     "actuator_incoming_fp": act_fp})
    return pd.DataFrame(rows)


ACTUATORS = {"hatch", "pot_1", "pot_2", "load_in", "load_out",
             "osr_1", "osr_2", "osr_mic", "osr_in", "osr_out", "osr_upwind",
             "osr_downwind", "osr_ambient", "osr_intake",
             "v_1", "v_2", "v_mic", "v_in", "v_out"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--experiments", nargs="+",
                   default=["actuators_random_walk_1", "actuators_random_walk_2",
                            "actuators_random_walk_3"])
    p.add_argument("--root", default="Data/causalchamber")
    p.add_argument("--units", type=int, default=32)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="ExpOutput/chamber")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_tables = []

    for name in args.experiments:
        frame, graph = load_experiment(name, args.root)
        names = varying_variables(frame, graph)
        print(f"\n{name}: {frame.shape[0]} rows, {len(names)} varying "
              f"variables of {len(graph.columns)}: {names}")
        sub = graph.loc[names, names]
        n_edges = int(sub.to_numpy().sum())
        print(f"  ground-truth edges among varying variables: {n_edges}")
        if n_edges == 0:
            print("  nothing to detect here; skipping")
            continue

        z = embed_columns(frame, names)

        t0 = time.time()
        knn_g, knn_r2, knn_ghost = loco_with_ghost(z, names, "knn", args.seed, args)
        print(f"  knn LOCO {time.time()-t0:.0f}s  (mean full r2 {knn_r2.mean():.3f})")
        t0 = time.time()
        mlp_g, mlp_r2, mlp_ghost = loco_with_ghost(z, names, "mlp", args.seed, args)
        print(f"  mlp LOCO {time.time()-t0:.0f}s  (mean full r2 {mlp_r2.mean():.3f})")
        t0 = time.time()
        ccm_m = ccm_matrix(frame, names, args.seed)
        print(f"  ccm {time.time()-t0:.0f}s")
        t0 = time.time()
        pcmci_m = pcmci_matrix(frame, names)
        print(f"  pcmci {time.time()-t0:.0f}s")

        scores = {"knn": knn_g, "mlp": mlp_g, "ccm": ccm_m, "pcmci": pcmci_m}
        floors = {"knn": max(knn_ghost), "mlp": max(mlp_ghost)}
        table = evaluate(scores, names, graph, ACTUATORS, floors)
        table["experiment"] = name
        all_tables.append(table)
        with pd.option_context("display.float_format", "{:.3f}".format,
                               "display.width", 160):
            print(table.to_string(index=False))

        for method, mat in scores.items():
            pd.DataFrame(mat, index=names, columns=names).to_csv(
                outdir / f"{name}_{method}.csv")

    if all_tables:
        combined = pd.concat(all_tables, ignore_index=True)
        combined.to_csv(outdir / "evaluation.csv", index=False)
        print("\n" + "=" * 88)
        print("POOLED OVER EXPERIMENTS")
        print("=" * 88)
        with pd.option_context("display.float_format", "{:.3f}".format):
            print(combined.groupby("method")[
                ["auroc", "edges_above_floor", "actuator_incoming_fp"]
            ].mean().to_string())
        print(f"\nwrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
