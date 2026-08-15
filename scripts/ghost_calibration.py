"""Can the ghost calibrate a detection threshold, and how many ghosts does
that take?

Two questions the paper leaves open.

1.  Recall accounting. The truth label used throughout marks a channel as a
    member if it has ANY edge in the generating DAG, in or out. A pure
    source, with out-edges but no in-edges, is therefore labelled positive
    while receiving no drive at all, and a drivenness statistic is required
    to be silent on it. Recall must be reported against the driven subset,
    not against the label, or the method is charged for obeying its own
    semantics.

2.  A usable threshold. A single ghost's magnitude separates members from
    non-members perfectly on the homogeneous systems but admits 43 false
    positives on the heterogeneous pool, where autonomous channels come
    from four families and one ghost cannot represent the null for all of
    them. This script scores many ghosts, drawn from donors spanning the
    channel population, and asks whether the maximum over them is a
    threshold that holds in both cases.

Extra ghosts are scored on the SAVED encoders with no retraining: a ghost
enters only the two ridge readouts, never the encoder input. The original
ghost is rescored here as a correctness check on that shortcut; its
recomputed value must match the archived one.

    python scripts/ghost_calibration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent))
from bottleneck_membership import (E, MaskedAE, build_system,  # noqa: E402
                                   build_system_hetero, splits_for)
from excess_membership import poly_own, r2_clamped  # noqa: E402
from network_scale import random_dag, simulate  # noqa: E402
from deepfeatselect.ccm import time_delay_embed  # noqa: E402

V, MEMBERS, N, COUPLING, BOTTLENECK, DEGREE = 1000, 100, 2000, 0.3, 128, 2
N_GHOSTS = 200

CASES = [
    ("seed 0", 0, False, "ExpOutput/ensemble/models",
     "ExpOutput/excess_poly/excess_consensus.npy"),
    ("seed 1", 1, False, "ExpOutput/ensemble_s1/models",
     "ExpOutput/excess_s1/excess_consensus.npy"),
    ("seed 2", 2, False, "ExpOutput/ensemble_s2/models",
     "ExpOutput/excess_s2/excess_consensus.npy"),
    ("hetero", 0, True, "ExpOutput/ensemble_het/models",
     "ExpOutput/excess_het/excess_consensus.npy"),
]


def degrees(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """In- and out-degree of the accepted generating DAG for this seed."""
    n_edges = max(MEMBERS, int(1.2 * MEMBERS))
    for attempt in range(20):
        rng = np.random.default_rng(seed + 100 * attempt)
        cand = random_dag(MEMBERS, n_edges, rng)
        try:
            simulate(N, cand, MEMBERS, COUPLING, seed)
            ind = np.zeros(MEMBERS, int)
            outd = np.zeros(MEMBERS, int)
            for i, j in cand:
                outd[i] += 1
                ind[j] += 1
            return ind, outd
        except ValueError:
            continue
    raise RuntimeError(f"no non-divergent graph for seed {seed}")


def run_case(label, seed, hetero, models_dir, archived_path):
    builder = build_system_hetero if hetero else build_system
    x, truth = builder(V, MEMBERS, N, COUPLING, seed)
    mats = [time_delay_embed(x[:, j], E)[0] for j in range(V)]
    n = min(len(m) for m in mats)
    joint = np.hstack([m[:n] for m in mats]).astype("float64")

    # The one ghost the archived run used, reproduced exactly.
    rng = np.random.default_rng(seed + 7331)
    donor = int(rng.integers(0, MEMBERS))
    shift = int(rng.integers(n // 4, 3 * n // 4))
    orig_ghost = np.roll(joint[:, donor * E:(donor + 1) * E], shift, axis=0)
    joint_all = np.hstack([joint, orig_ghost])
    v_all = V + 1

    tr, _, te = splits_for(n)
    mu, sd = joint_all[tr].mean(0), joint_all[tr].std(0) + 1e-12
    zs = ((joint_all - mu) / sd).astype("float32")
    lead = zs[:, [j * E for j in range(v_all)]]
    tr_idx = np.arange(tr.start, tr.stop - 1)
    te_idx = np.arange(te.start, n - 1)

    def fit_pair(own, extra, target, alpha=1.0):
        own_p = poly_own(own, DEGREE)
        a, b = own_p[tr_idx], own_p[te_idx]
        if extra is not None:
            a = np.hstack([a, extra[tr_idx]])
            b = np.hstack([b, extra[te_idx]])
        m = Ridge(alpha=alpha)
        m.fit(a, target[tr_idx + 1])
        return r2_clamped(m.predict(b), target[te_idx + 1])

    # Extra ghosts: donors spread across the whole channel population, so the
    # null is represented for every family present, not only the members.
    grng = np.random.default_rng(seed + 4242)
    donors = grng.choice(V, size=min(N_GHOSTS, V), replace=False)
    gz, glead = [], []
    for d in donors:
        raw = np.roll(joint[:, d * E:(d + 1) * E],
                      int(grng.integers(n // 4, 3 * n // 4)), axis=0)
        z = ((raw - raw[tr].mean(0)) / (raw[tr].std(0) + 1e-12)).astype("float32")
        gz.append(z)
        glead.append(z[:, 0])

    models = sorted(Path(models_dir).glob("m*.weights.h5"))
    self_orig = fit_pair(zs[:, V * E:(V + 1) * E], None, lead[:, V])
    self_g = [fit_pair(z, None, l) for z, l in zip(gz, glead)]

    ex_orig, ex_g = [], []
    for f in models:
        m = MaskedAE(v_all, E, BOTTLENECK, mask_mode="zero",
                     loss_on_masked_only=True)
        m(zs[:2])
        m.load_weights(f)
        code = m.encoder.predict(zs, verbose=0, batch_size=4096)
        ex_orig.append(fit_pair(zs[:, V * E:(V + 1) * E], code,
                                lead[:, V]) - self_orig)
        ex_g.append([fit_pair(z, code, l) - s
                     for z, l, s in zip(gz, glead, self_g)])

    recomputed = float(np.mean(ex_orig))
    archived = float(np.load(archived_path)[-1])
    ghosts = np.mean(np.array(ex_g), axis=0)
    return dict(label=label, seed=seed, hetero=hetero, archived=archived,
                recomputed=recomputed, ghosts=ghosts, truth=truth,
                excess=np.load(archived_path))


def main() -> int:
    pd.set_option("display.width", 170)
    pd.set_option("display.float_format", "{:.6f}".format)
    rows, acc = [], []
    out_dir = Path("ExpOutput/recall")
    out_dir.mkdir(parents=True, exist_ok=True)

    for case in CASES:
        r = run_case(*case)
        ex, truth = r["excess"], r["truth"]
        truth_all = np.append(truth, False)
        g = r["ghosts"]
        np.save(out_dir / f"ghosts_{r['label'].replace(' ', '')}.npy", g)

        # Detection is one-sided, so the null's UPPER tail sets the
        # threshold. The lower tail is the finite-sample cost of appending an
        # uninformative code and carries no detection information.
        thr1 = abs(float(ex[-1]))                 # single archived ghost
        rules = [("1 ghost |g|", thr1),
                 ("panel max", float(g.max())),
                 ("panel p99", float(np.quantile(g, 0.99))),
                 ("panel p95", float(np.quantile(g, 0.95)))]

        print(f"\n--- {r['label']} --- archived ghost {r['archived']:+.6f}, "
              f"recomputed {r['recomputed']:+.6f} "
              f"(delta {abs(r['archived']-r['recomputed']):.2e})")
        print(f"    {len(g)} ghosts: min {g.min():+.6f}  p05 "
              f"{np.quantile(g,0.05):+.6f}  median {np.median(g):+.6f}  "
              f"p95 {np.quantile(g,0.95):+.6f}  max {g.max():+.6f}")
        print(f"    fraction of ghosts above zero: {(g > 0).mean():.3f}")

        for name, thr in rules:
            sel = ex > thr
            tp = int((sel & truth_all).sum())
            fp = int((sel & ~truth_all).sum())
            rows.append({"system": r["label"], "rule": name, "threshold": thr,
                         "true_pos": tp, "false_pos": fp,
                         "precision": tp / max(tp + fp, 1),
                         "recall_vs_label": tp / int(truth_all.sum())})

        # Role accounting: label positive means any edge; drivenness needs an
        # in-edge, so pure sources are labelled positive but unfindable.
        ind, outd = degrees(r["seed"])
        role = np.where(ind > 0, "driven",
                        np.where(outd > 0, "source", "isolated"))
        flag = ex[:MEMBERS] > float(g.max())
        for nm in ("driven", "source", "isolated"):
            s = role == nm
            if s.any():
                acc.append({"system": r["label"], "role": nm, "n": int(s.sum()),
                            "labelled_positive": int(truth[:MEMBERS][s].sum()),
                            "flagged": int(flag[s].sum()),
                            "recall": float(flag[s].mean())})

    frame = pd.DataFrame(rows)
    accf = pd.DataFrame(acc)
    out = out_dir

    frame.to_csv(out / "ghost_panel_threshold.csv", index=False)
    accf.to_csv(out / "role_accounting.csv", index=False)

    print("\n" + "=" * 92)
    print("THRESHOLD FROM ONE GHOST vs A PANEL OF GHOSTS")
    print("=" * 92)
    print(frame.to_string(index=False))
    print("\npooled by rule:")
    p = frame.groupby("rule", sort=False).agg(
        true_pos=("true_pos", "sum"), false_pos=("false_pos", "sum"))
    p["precision"] = p.true_pos / (p.true_pos + p.false_pos)
    print(p.to_string())

    print("\n" + "=" * 92)
    print("RECALL BY ROLE IN THE GENERATING GRAPH (panel threshold)")
    print("=" * 92)
    print(accf.to_string(index=False))
    print("\npooled by role:")
    pr = accf.groupby("role").agg(n=("n", "sum"),
                                  labelled_positive=("labelled_positive", "sum"),
                                  flagged=("flagged", "sum"))
    pr["recall"] = pr.flagged / pr.n
    print(pr.to_string())
    d = pr.loc["driven"]
    print(f"\nrecall among genuinely driven channels: "
          f"{int(d.flagged)}/{int(d.n)} = {d.flagged/d.n:.0%}")
    print(f"wrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
