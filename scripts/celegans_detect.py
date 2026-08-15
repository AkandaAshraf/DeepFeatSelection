"""The application test: C. elegans whole-brain imaging against the connectome.

Everything here was fixed in ``paper/validation_protocol.md`` (Tier 2b) before
this script produced a number. The guiding question: does the
conditional MLP find something in real neural data that CCM, kNN-LOCO and
PCMCI miss — concretely, does it reject mediated (2-hop) pairs that the
classical methods falsely report — or does it add nothing?

Data: Kato et al. 2015, WT_NoStim (five immobilised worms, spontaneous
dynamics). Ground truth: the anatomical connectome (Cook-derived edge list;
chemical directed, gap junctions bidirectional). Truth is anatomical
possibility: precision against anatomy is the deliverable; anatomical edges
silent in this behavioural state are expected, so recall is reported but not
claimed.

    python scripts/celegans_detect.py --worms 0 1 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsRegressor

sys.path.insert(0, str(Path(__file__).parent))
from maturity_sweep import fit_checkpointed  # noqa: E402
from chamber_detect import r2_of  # noqa: E402
from deepfeatselect.ccm import ccm, time_delay_embed  # noqa: E402

E = 3
TAU = 3          # ~1 s at 2.9 Hz: the calcium timescale
TAU_MAX_PCMCI = 6
MAX_NEURONS = 25  # per protocol: alphabetical cap, never variance-ranked
TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2


def _decode_name(f: h5py.File, ref) -> str:
    """MATLAB cell-of-strings; worm 3 nests references one level deeper."""
    obj = f[ref]
    data = obj[()]
    if isinstance(data, np.ndarray) and data.dtype == object:
        return _decode_name(f, data.flatten()[0])
    if data.dtype.kind in ("u", "i"):
        return "".join(chr(int(c)) for c in np.asarray(data).flatten())
    return ""


def load_worm(path: Path, worm: int) -> tuple[np.ndarray, list[str], float]:
    f = h5py.File(path, "r")
    g = f["WT_NoStim"]
    traces = np.asarray(f[g["deltaFOverF_bc"][worm, 0]])  # (neurons, t)
    name_refs = f[g["NeuronNames"][worm, 0]]
    names = [_decode_name(f, name_refs[i, 0]) for i in range(name_refs.shape[0])]
    fps = float(np.asarray(f[g["fps"][worm, 0]]).flatten()[0])
    return traces, names, fps


def load_connectome(path: Path) -> tuple[set[tuple[str, str]], set[str]]:
    """Directed anatomical edges. Gap junctions count both ways."""
    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]
    edges: set[tuple[str, str]] = set()
    nodes: set[str] = set()
    for _, r in d.iterrows():
        a, b = str(r["Source"]).strip(), str(r["Target"]).strip()
        nodes.update((a, b))
        edges.add((a, b))
        if str(r["Type"]).strip().lower().startswith("electrical"):
            edges.add((b, a))
    return edges, nodes


def splits_for(n: int) -> tuple[slice, slice, slice]:
    a = int(TRAIN_FRACTION * n)
    b = int((TRAIN_FRACTION + VAL_FRACTION) * n)
    return slice(0, a - E), slice(a, b - E), slice(b, n)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--worms", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--root", default="Data/celegans")
    p.add_argument("--epochs-grid", type=int, nargs="+",
                   default=[2, 5, 10, 25, 50])
    p.add_argument("--units", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-neurons", type=int, default=MAX_NEURONS)
    p.add_argument("--outdir", default="ExpOutput/celegans")
    args = p.parse_args()
    grid = sorted(args.epochs_grid)

    root = Path(args.root)
    edges_all, connectome_nodes = load_connectome(root / "herm_full_edgelist.csv")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    eval_rows, indirect_rows = [], []

    for worm in args.worms:
        traces, names, fps = load_worm(root / "WT_NoStim.mat", worm)
        identified = sorted(n for n in set(names)
                            if n and not n.isdigit() and n in connectome_nodes)
        keep = identified[:args.max_neurons]
        idx = [names.index(n) for n in keep]
        V = len(keep)
        x = traces[idx].T                       # (t, V)
        truth = np.array([[(a, b) in edges_all for b in keep] for a in keep])
        adj = truth | truth.T
        print(f"\nworm {worm}: {x.shape[0]} timepoints @ {fps:.2f} fps, "
              f"{len(identified)} identified in connectome, scoring {V} "
              f"({int(truth.sum())} directed anatomical edges)")

        # z-score, difference, embed at (E, TAU) — the standing pipeline.
        z = (x - x.mean(0)) / (x.std(0) + 1e-12)
        z = np.diff(z, axis=0)
        mats = [time_delay_embed(z[:, j], E, tau=TAU)[0] for j in range(V)]
        n = min(len(m) for m in mats)
        joint = np.hstack([m[:n] for m in mats]).astype("float64")
        tr, va, te = splits_for(n)
        mu, sd = joint[tr].mean(0), joint[tr].std(0) + 1e-12
        zs = ((joint - mu) / sd).astype("float32")
        cols = lambda j: slice(j * E, (j + 1) * E)
        splits = (tr, va, te)
        rng = np.random.default_rng(args.seed + 7331)

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
                return fit_checkpointed(np.hstack(parts), dst, splits,
                                        args.seed, args.units, grid)

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
            if u % 5 == 4:
                print(f"  target {u + 1}/{V} done ({time.time()-t0:.0f}s)")
        print(f"  LOCO all maturities + knn: {time.time()-t0:.0f}s")

        t0 = time.time()
        ccm_m = np.full((V, V), np.nan)
        for i in range(V):
            for j in range(i + 1, V):
                r = ccm(z[:, i], z[:, j], E=E, tau=TAU, seed=args.seed)
                ccm_m[i, j] = r.x_causes_y.rho_at_max_lib
                ccm_m[j, i] = r.y_causes_x.rho_at_max_lib
        print(f"  ccm: {time.time()-t0:.0f}s")

        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr
        import tigramite.data_processing as pp
        t0 = time.time()
        res = PCMCI(dataframe=pp.DataFrame(z.copy(), var_names=keep),
                    cond_ind_test=ParCorr(), verbosity=0).run_pcmci(
                        tau_max=TAU_MAX_PCMCI, pc_alpha=None)
        pcmci_m = np.abs(res["val_matrix"])[:, :, 1:].max(axis=2)
        print(f"  pcmci: {time.time()-t0:.0f}s")

        def evaluate(mat, transpose_loco, method, floor=np.nan):
            y, s = [], []
            for i in range(V):
                for j in range(V):
                    if i == j:
                        continue
                    v = mat[j, i] if transpose_loco else mat[i, j]
                    if np.isnan(v):
                        continue
                    y.append(truth[i, j])
                    s.append(v)
            eval_rows.append({
                "worm": worm, "method": method,
                "prauc": average_precision_score(y, s),
                "auroc": roc_auc_score(y, s),
                "baseline": float(np.mean(y)), "floor": floor})

        evaluate(knn_gains, True, "knn", max(knn_ghosts))
        evaluate(ccm_m, False, "ccm")
        evaluate(pcmci_m, False, "pcmci")
        for e in grid:
            evaluate(mlp_gains[e], True, f"mlp_e{e}", max(mlp_ghosts[e]))

        # THE DELIVERABLE STRATUM: non-adjacent pairs with a 2-hop anatomical
        # route. For each, every method's score percentile among its own
        # non-edge scores — high percentile = the method reports this
        # mediated pair as if it were a link.
        def pct(mat, transpose_loco):
            vals = {}
            allneg = []
            for i in range(V):
                for j in range(V):
                    if i == j:
                        continue
                    v = mat[j, i] if transpose_loco else mat[i, j]
                    if not truth[i, j] and not np.isnan(v):
                        allneg.append(v)
            allneg = np.sort(allneg)
            for i in range(V):
                for j in range(V):
                    if i == j or truth[i, j]:
                        continue
                    v = mat[j, i] if transpose_loco else mat[i, j]
                    if not np.isnan(v):
                        vals[(i, j)] = float(np.searchsorted(allneg, v)
                                             / len(allneg))
            return vals

        pcts = {"knn": pct(knn_gains, True), "ccm": pct(ccm_m, False),
                "pcmci": pct(pcmci_m, False)}
        best_e = grid[len(grid) // 2]
        pcts["mlp"] = pct(mlp_gains[best_e], True)
        for i in range(V):
            for j in range(V):
                if i == j or truth[i, j] or not adj[i].any():
                    continue
                mediated = (not adj[i, j]) and bool((adj[i] & adj[j]).any())
                if not mediated:
                    continue
                row = {"worm": worm, "source": keep[i], "target": keep[j]}
                for m in ("knn", "ccm", "pcmci", "mlp"):
                    row[f"{m}_pct"] = pcts[m].get((i, j), np.nan)
                indirect_rows.append(row)

        for e in grid:
            pd.DataFrame(mlp_gains[e], index=keep, columns=keep).to_csv(
                outdir / f"worm{worm}_mlp_e{e}.csv")
        pd.DataFrame(knn_gains, index=keep, columns=keep).to_csv(
            outdir / f"worm{worm}_knn.csv")
        pd.DataFrame(ccm_m, index=keep, columns=keep).to_csv(
            outdir / f"worm{worm}_ccm.csv")
        pd.DataFrame(pcmci_m, index=keep, columns=keep).to_csv(
            outdir / f"worm{worm}_pcmci.csv")

    ev = pd.DataFrame(eval_rows)
    ind = pd.DataFrame(indirect_rows)
    ev.to_csv(outdir / "evaluation.csv", index=False)
    ind.to_csv(outdir / "indirect_pairs.csv", index=False)

    print("\n" + "=" * 84)
    print("EVALUATION (precision against anatomy is the deliverable metric)")
    print("=" * 84)
    with pd.option_context("display.float_format", "{:.3f}".format,
                           "display.width", 140):
        print(ev.pivot_table(index="method",
                             values=["prauc", "auroc", "baseline"],
                             aggfunc="mean").to_string())

    print("\nMEDIATED (2-hop, non-adjacent) pairs where classical methods rank")
    print("high but the MLP does not — the named deliverable:")
    if len(ind):
        classical = ind[["knn_pct", "ccm_pct", "pcmci_pct"]].min(axis=1)
        candidates = ind[(classical > 0.8) & (ind.mlp_pct < 0.5)]
        with pd.option_context("display.float_format", "{:.2f}".format):
            print(candidates.head(20).to_string(index=False)
                  if len(candidates) else
                  "  none found under the declared rule (>0.8 all classical, <0.5 MLP)")
    print(f"\nwrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
