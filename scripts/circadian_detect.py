"""Tier 2: the mammalian circadian clock, interventional truth, open data.

GSE11923 (Hughes et al. 2009): mouse liver transcriptome, 48 consecutive
hourly samples (CT18-CT65), the highest-resolution standard circadian time
course. Ground truth is the TTFL core circuit as established by decades of
knockout genetics -- interventional evidence, not observational inference --
with the edge list fixed in ``paper/validation_protocol.md`` before this data
was inspected.

Why this dataset despite n=48: real transcriptomic time courses ARE this
short. The validity map predicts the classical arms (CCM, PCMCI, kNN) carry
detection at this size and the MLP starves; measuring that honestly on the
best available real case is the point of Tier 2, not an inconvenience.

Choices, all pre-declared: probes averaged per gene; series z-scored, then
FIRST-DIFFERENCED (the maturity experiment showed circular-shift ghosts are
invalid on nonstationary series); embedding E=3, tau=3 h (tau=1 on a 24-h
oscillation sampled hourly gives nearly collinear coordinates); PCMCI tau_max
= 6 h, the embedding span; Arntl is the hub (out-degree 8), so per the
maturity mechanism its LOCO edges are predicted fragile and MLP results are
reported at every checkpoint rather than one training length.

    python scripts/circadian_detect.py
"""

from __future__ import annotations

import argparse
import gzip
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

sys.path.insert(0, str(Path(__file__).parent))
from maturity_sweep import fit_checkpointed  # noqa: E402
from chamber_detect import r2_of  # noqa: E402
from deepfeatselect.ccm import ccm, time_delay_embed  # noqa: E402

GENES = ["Arntl", "Per1", "Per2", "Per3", "Cry1", "Cry2",
         "Nr1d1", "Nr1d2", "Rora", "Dbp"]

# The TTFL core, transcript-level reading; activation and repression both
# count as edges (the methods detect, they do not sign).
EDGES = ([("Arntl", g) for g in
          ("Per1", "Per2", "Per3", "Cry1", "Cry2", "Nr1d1", "Nr1d2", "Dbp")]
         + [("Nr1d1", "Arntl"), ("Nr1d2", "Arntl"), ("Rora", "Arntl")]
         + [(r, t) for r in ("Per1", "Per2", "Cry1", "Cry2")
            for t in ("Nr1d1", "Dbp")])

E = 3
TAU = 3
TAU_MAX_PCMCI = 6
TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2


def parse_geo(root: Path) -> pd.DataFrame:
    """Clock-gene expression matrix (genes x 48 timepoints), probe-averaged.

    Cached to CSV so the gz parsing happens once.  Sample order was verified
    against !Sample_title: CT18 through CT65, consecutive hours.
    """
    cache = root / "clock_genes.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0)

    probes: dict[str, str] = {}
    with gzip.open(root / "GPL1261.annot.gz", "rt", errors="replace") as f:
        header_seen = False
        for line in f:
            if not header_seen:
                header_seen = line.startswith("ID\t")
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 2 and parts[2] in GENES:
                probes[parts[0]] = parts[2]

    rows: dict[str, list[np.ndarray]] = {g: [] for g in GENES}
    with gzip.open(root / "GSE11923_series_matrix.txt.gz", "rt") as f:
        in_table = False
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                next(f)
                continue
            if not in_table or line.startswith("!series_matrix_table_end"):
                continue
            parts = line.rstrip("\n").split("\t")
            probe = parts[0].strip('"')
            if probe in probes:
                rows[probes[probe]].append(
                    np.array([float(x) for x in parts[1:]]))

    matrix = pd.DataFrame(
        {g: np.mean(rows[g], axis=0) for g in GENES if rows[g]}).T
    missing = [g for g in GENES if not rows[g]]
    if missing:
        print(f"EXCLUDED (no probes found): {missing}")
    matrix.to_csv(cache)
    return matrix


def splits_for(n: int) -> tuple[slice, slice, slice]:
    a = int(TRAIN_FRACTION * n)
    b = int((TRAIN_FRACTION + VAL_FRACTION) * n)
    return slice(0, a - E), slice(a, b - E), slice(b, n)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="Data/circadian")
    p.add_argument("--epochs-grid", type=int, nargs="+",
                   default=[2, 5, 10, 25, 50])
    p.add_argument("--units", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="ExpOutput/circadian")
    args = p.parse_args()
    grid = sorted(args.epochs_grid)

    matrix = parse_geo(Path(args.root))
    names = list(matrix.index)
    V = len(names)
    truth = np.zeros((V, V), dtype=bool)
    for a, b in EDGES:
        if a in names and b in names:
            truth[names.index(a), names.index(b)] = True
    print(f"{V} genes, {int(truth.sum())} ground-truth edges, "
          f"{matrix.shape[1]} timepoints")

    # z-score per gene, then first-difference: the stationarity lesson.
    z = ((matrix.T - matrix.T.mean()) / (matrix.T.std() + 1e-12)).to_numpy()
    z = np.diff(z, axis=0)

    # Per-gene delay embeddings at (E, TAU), time-aligned.
    mats = [time_delay_embed(z[:, j], E, tau=TAU)[0] for j in range(V)]
    n = min(len(m) for m in mats)
    joint = np.hstack([m[:n] for m in mats]).astype("float64")
    tr, va, te = splits_for(n)
    mu, sd = joint[tr].mean(axis=0), joint[tr].std(axis=0) + 1e-12
    zs = ((joint - mu) / sd).astype("float32")
    print(f"{n} embedded points -> train {tr.stop} / stop {va.stop - va.start}"
          f" / test {n - te.start}   (declared: tiny, and reported as such)")

    cols = lambda j: slice(j * E, (j + 1) * E)
    rng = np.random.default_rng(args.seed + 7331)
    splits = (tr, va, te)

    # --- MLP LOCO, checkpointed across maturity; ghost null per target ---
    mlp_gains = {e: np.full((V, V), np.nan) for e in grid}
    mlp_ghosts = {e: [] for e in grid}
    knn_gains = np.full((V, V), np.nan)
    knn_ghosts = []
    t0 = time.time()
    for u in range(V):
        others = [v for v in range(V) if v != u]
        donor = int(rng.choice(others))
        shift = int(rng.integers(n // 4, 3 * n // 4))
        ghost = np.roll(zs[:, cols(donor)], shift, axis=0)
        dst = zs[:, cols(u)]

        def mlp_run(sources, extra=None):
            parts = [zs[:, cols(v)] for v in sources]
            if extra is not None:
                parts.append(extra)
            return fit_checkpointed(np.hstack(parts), dst, splits, args.seed,
                                    args.units, grid)

        def knn_run(sources, extra=None):
            parts = [zs[:, cols(v)] for v in sources]
            if extra is not None:
                parts.append(extra)
            src = np.hstack(parts)
            m = KNeighborsRegressor(n_neighbors=5)
            m.fit(src[tr], dst[tr])
            return r2_of(m.predict(src[te]), dst[te])

        base = mlp_run(others, extra=ghost)
        no_ghost = mlp_run(others, extra=None)
        kbase = knn_run(others, extra=ghost)
        knn_ghosts.append(kbase - knn_run(others, extra=None))
        for e in grid:
            mlp_ghosts[e].append(base[e] - no_ghost[e])
        for v in others:
            without = mlp_run([w for w in others if w != v], extra=ghost)
            for e in grid:
                mlp_gains[e][v, u] = base[e] - without[e]
            knn_gains[v, u] = kbase - knn_run(
                [w for w in others if w != v], extra=ghost)
    print(f"LOCO (all maturities + knn) in {time.time()-t0:.0f}s")

    # --- CCM, directed: score[i, j] = rho recovering i from j ---
    ccm_m = np.full((V, V), np.nan)
    raw = {g: matrix.T[g].to_numpy() for g in names}
    for i in range(V):
        for j in range(i + 1, V):
            r = ccm(np.diff(raw[names[i]]), np.diff(raw[names[j]]),
                    E=E, tau=TAU, seed=args.seed)
            ccm_m[i, j] = r.x_causes_y.rho_at_max_lib
            ccm_m[j, i] = r.y_causes_x.rho_at_max_lib

    # --- PCMCI ---
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    import tigramite.data_processing as pp
    dataframe = pp.DataFrame(z.copy(), var_names=names)
    res = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(),
                verbosity=0).run_pcmci(tau_max=TAU_MAX_PCMCI, pc_alpha=None)
    pcmci_m = np.abs(res["val_matrix"])[:, :, 1:].max(axis=2)

    def auroc_of(mat, transpose_loco: bool) -> float:
        pos, neg = [], []
        for i in range(V):
            for j in range(V):
                if i == j or np.isnan(mat[i, j] if not transpose_loco
                                      else mat[j, i]):
                    continue
                s = mat[j, i] if transpose_loco else mat[i, j]
                (pos if truth[i, j] else neg).append(s)
        ranks = pd.Series(pos + neg).rank().to_numpy()
        return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                     / (len(pos) * len(neg)))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [{"method": "knn", "epochs": np.nan,
             "auroc": auroc_of(knn_gains, True),
             "floor": max(knn_ghosts)},
            {"method": "ccm", "epochs": np.nan,
             "auroc": auroc_of(ccm_m, False), "floor": np.nan},
            {"method": "pcmci", "epochs": np.nan,
             "auroc": auroc_of(pcmci_m, False), "floor": np.nan}]
    for e in grid:
        rows.append({"method": "mlp", "epochs": e,
                     "auroc": auroc_of(mlp_gains[e], True),
                     "floor": max(mlp_ghosts[e])})
    table = pd.DataFrame(rows)
    table.to_csv(outdir / "circadian_eval.csv", index=False)
    for e in grid:
        pd.DataFrame(mlp_gains[e], index=names, columns=names).to_csv(
            outdir / f"mlp_gains_epoch{e}.csv")
    pd.DataFrame(ccm_m, index=names, columns=names).to_csv(outdir / "ccm.csv")
    pd.DataFrame(pcmci_m, index=names, columns=names).to_csv(
        outdir / "pcmci.csv")

    print("\n" + "=" * 78)
    print(f"CIRCADIAN CLOCK DETECTION  ({int(truth.sum())} TTFL edges, "
          f"{V * (V - 1)} candidate pairs, n={matrix.shape[1]})")
    print("=" * 78)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string(index=False))

    # The Arntl stratum: the hub whose edges the maturity mechanism predicts
    # fragile for LOCO. Mean gain on Arntl-> edges per checkpoint.
    ai = names.index("Arntl")
    arntl_children = [names.index(b) for a, b in EDGES
                      if a == "Arntl" and b in names]
    print("\nArntl (hub) mean LOCO gain per maturity:")
    for e in grid:
        gains = [mlp_gains[e][j, ai] for j in arntl_children]
        print(f"  epochs {e:>3}: {np.mean(gains):+.4f}")
    print(f"\nwrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
