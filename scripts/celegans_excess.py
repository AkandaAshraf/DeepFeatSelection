"""The excess statistic meets real neurons: who is driven by the worm's brain?

The scale detector's first real-data cell. On synthetic systems the
excess-over-self readout found the driven core of a 1000-variable system with
30/30 top-10 precision across three seeds and survived a heterogeneous pool
with zero false alarms. Real neural data poses the two questions synthetic
cannot: does a FLEXIBLE self-baseline (no exact function class exists here)
still refuse false positives, and does the statistic's "driven by the
system" semantics say something biologically real?

A brain has no loners -- every neuron is recurrently embedded -- so binary
membership becomes graded drivenness, and the falsifiable structure comes
from biology. PRE-REGISTERED PREDICTIONS, fixed before any number:

1. GHOST ~ 0. A circularly shifted copy of a real neuron carries realistic
   marginals and no temporal alignment; its excess must pin at zero. This is
   the load-bearing control and needs no biological ground truth at all.
2. SENSORY < COMMAND/MOTOR. In WT_NoStim the environment is deliberately
   constant, so sensory neurons' natural drivers are silent, while command
   interneurons and motor neurons are driven by the internal motor-command
   cycle Kato et al. described. Classes assigned from canonical WormAtlas
   identities by name, fixed here in the script before running.
3. EXCESS correlates positively with weighted anatomical in-degree (full
   connectome, all potential parents) among identified neurons -- graded,
   expected modest: many parents are unrecorded, though recurrence means the
   recorded state carries their influence indirectly.

Pipeline per worm (0-2): all recorded neurons (identified or not), z-scored,
first-differenced, delay-embedded (E=3, tau=3); ghost appended; 8-encoder
ensemble trained fresh (b=32 -- the worm manifold is famously low-dimensional);
excess = r2(next | own poly-3 lags + code) - r2(next | own poly-3 lags),
ridge readouts, consensus over encoders.

    python scripts/celegans_excess.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent))
import h5py  # noqa: E402
from bottleneck_membership import MaskedAE  # noqa: E402
from celegans_detect import load_worm, load_connectome, _decode_name  # noqa: E402


def load_worm_any(path, worm: int):
    """Kato-repo loader for BOTH schemas: WT_NoStim uses deltaFOverF_bc /
    NeuronNames; WT_Stim and AVA_HisCl use traces / IDs with empty strings for
    unidentified cells. Unidentified names become their index digits, matching
    the WT convention the rest of the pipeline classifies on."""
    f = h5py.File(path, "r")
    key = [k for k in f.keys() if k != "#refs#"][0]
    g = f[key]
    tr_key = "deltaFOverF_bc" if "deltaFOverF_bc" in g else "traces"
    id_key = "NeuronNames" if "NeuronNames" in g else "IDs"
    traces = np.asarray(f[g[tr_key][worm, 0]])
    refs = f[g[id_key][worm, 0]]
    names = []
    for i in range(refs.shape[0]):
        nm = _decode_name(f, refs[i, 0]).replace(chr(0), "").strip()
        names.append(nm if nm else str(i))
    fps = float(np.asarray(f[g["fps"][worm, 0]]).flatten()[0])
    return traces, names, fps
from excess_membership import poly_own, r2_clamped  # noqa: E402
from deepfeatselect.ccm import time_delay_embed  # noqa: E402

E = 3
TAU = 3
BOTTLENECK = 32
N_MODELS = 8
EPOCHS = 25
TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2

# Canonical classes, by name root (L/R/D/V and numeric suffixes stripped).
SENSORY = {"OLQ", "URY", "URX", "URA", "URB", "ASK", "ASI", "ASG", "ASH",
           "ASJ", "ASE", "AWA", "AWB", "AWC", "AFD", "ADF", "ADL", "AQR",
           "BAG", "FLP", "IL1", "IL2", "OLL", "CEP", "ADE", "ALM", "AVM"}
COMMAND_MOTOR = {"AVA", "AVB", "AVE", "AVD", "RIM", "RIB", "RIA", "AIB",
                 "AIY", "AIA", "AIZ", "RIS", "RID", "RIV", "RIF", "RME",
                 "RMD", "SMD", "SMB", "SIB", "SIA", "SAB", "VB", "DB",
                 "VA", "DA", "VD", "DD", "AS", "PVC", "AVF", "AVJ"}


def name_root(name: str) -> str:
    root = name
    while root and (root[-1].isdigit() or root[-1] in "LRDV"):
        stripped = root[:-1]
        if stripped in SENSORY or stripped in COMMAND_MOTOR:
            return stripped
        root = stripped
    return name


def classify(name: str) -> str:
    root = name_root(name)
    if root in SENSORY:
        return "sensory"
    if root in COMMAND_MOTOR:
        return "command_motor"
    return "other"


def splits_for(n: int) -> tuple[slice, slice, slice]:
    a = int(TRAIN_FRACTION * n)
    b = int((TRAIN_FRACTION + VAL_FRACTION) * n)
    return slice(0, a - E), slice(a, b - E), slice(b, n)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--worms", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--root", default="Data/celegans")
    p.add_argument("--mat", default="WT_NoStim.mat")
    p.add_argument("--poly-degree", type=int, default=3)
    p.add_argument("--outdir", default="ExpOutput/celegans_excess")
    args = p.parse_args()

    edges, connectome_nodes = load_connectome(
        Path(args.root) / "herm_full_edgelist.csv")
    in_weight: dict[str, int] = {}
    for a, b in edges:
        in_weight[b] = in_weight.get(b, 0) + 1

    outdir = Path(args.outdir)
    (outdir / "models").mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for worm in args.worms:
        traces, names, fps = load_worm_any(Path(args.root) / args.mat, worm)
        V = traces.shape[0]
        x = traces.T                                      # (t, V)
        z = (x - x.mean(0)) / (x.std(0) + 1e-12)
        z = np.diff(z, axis=0)
        mats = [time_delay_embed(z[:, j], E, tau=TAU)[0] for j in range(V)]
        n = min(len(m) for m in mats)
        joint = np.hstack([m[:n] for m in mats]).astype("float64")
        rng = np.random.default_rng(worm + 7331)
        donor = int(rng.integers(0, V))
        ghost = np.roll(joint[:, donor * E:(donor + 1) * E],
                        int(rng.integers(n // 4, 3 * n // 4)), axis=0)
        joint = np.hstack([joint, ghost])
        v_all = V + 1
        tr, va, te = splits_for(n)
        mu, sd = joint[tr].mean(0), joint[tr].std(0) + 1e-12
        zs = ((joint - mu) / sd).astype("float32")
        print(f"\nworm {worm}: {V} neurons ({sum(1 for nm in names if not nm.isdigit())}"
              f" identified), {n} embedded points @ {fps:.2f} fps")

        lead = zs[:, [j * E for j in range(v_all)]]
        tr_idx = np.arange(tr.start, tr.stop - 1)
        te_idx = np.arange(te.start, n - 1)

        def fit_pair(own, extra, target):
            own_p = poly_own(own, args.poly_degree)
            src_tr, src_te = own_p[tr_idx], own_p[te_idx]
            if extra is not None:
                src_tr = np.hstack([src_tr, extra[tr_idx]])
                src_te = np.hstack([src_te, extra[te_idx]])
            m = Ridge(alpha=1.0)
            m.fit(src_tr, target[tr_idx + 1])
            return r2_clamped(m.predict(src_te), target[te_idx + 1])

        self_r2 = np.empty(v_all)
        for q in range(v_all):
            self_r2[q] = fit_pair(zs[:, q * E:(q + 1) * E], None, lead[:, q])
        print(f"  self baselines: mean {self_r2[:V].mean():.3f}  "
              f"ghost {self_r2[-1]:.3f}")

        excess_all = []
        for m_i in range(N_MODELS):
            keras.utils.set_random_seed(2000 + m_i)
            model = MaskedAE(v_all, E, BOTTLENECK, mask_mode="zero",
                             loss_on_masked_only=True)
            model.compile(loss="mse", optimizer=keras.optimizers.Adam(3e-3))
            model.fit(zs[tr], zs[tr], validation_data=(zs[va], zs[va]),
                      epochs=EPOCHS, batch_size=64, shuffle=True, verbose=0)
            model.save_weights(outdir / "models" /
                               f"w{worm}_m{m_i}.weights.h5")
            code = model.encoder.predict(zs, verbose=0, batch_size=4096)
            ex = np.empty(v_all)
            for q in range(v_all):
                ex[q] = fit_pair(zs[:, q * E:(q + 1) * E], code,
                                 lead[:, q]) - self_r2[q]
            excess_all.append(ex)
        excess = np.mean(excess_all, axis=0)

        frame = pd.DataFrame({
            "neuron": names + ["GHOST"],
            "identified": [not nm.isdigit() for nm in names] + [False],
            "class": [classify(nm) if not nm.isdigit() else "unidentified"
                      for nm in names] + ["ghost"],
            "excess": excess, "self_r2": self_r2,
            "in_weight": [in_weight.get(nm, np.nan) for nm in names] + [np.nan],
        })
        frame.to_csv(outdir / f"worm{worm}_excess.csv", index=False)

        ghost_ex = excess[-1]
        sens = frame[frame["class"] == "sensory"].excess
        cm = frame[frame["class"] == "command_motor"].excess
        ident = frame[frame.identified & frame.in_weight.notna()]
        r_indeg = (np.corrcoef(ident.in_weight, ident.excess)[0, 1]
                   if len(ident) > 5 else np.nan)
        top10 = frame.nlargest(10, "excess")[["neuron", "class", "excess"]]

        print(f"  PREDICTION 1  ghost excess {ghost_ex:+.4f}")
        print(f"  PREDICTION 2  sensory {sens.mean():+.4f} (n={len(sens)})  "
              f"vs command/motor {cm.mean():+.4f} (n={len(cm)})")
        print(f"  PREDICTION 3  corr(excess, anatomical in-weight) "
              f"{r_indeg:+.3f}  (n={len(ident)})")
        print("  top-10 by excess:")
        for _, r in top10.iterrows():
            print(f"    {r.neuron:<10} {r['class']:<13} {r.excess:+.4f}")

        summary_rows.append({
            "worm": worm, "ghost": float(ghost_ex),
            "sensory_mean": float(sens.mean()),
            "command_motor_mean": float(cm.mean()),
            "sens_lt_cm": bool(sens.mean() < cm.mean()),
            "corr_in_weight": float(r_indeg),
            "n_sensory": len(sens), "n_command_motor": len(cm)})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "summary.csv", index=False)
    print("\n" + "=" * 80)
    print("SUMMARY ACROSS WORMS")
    print("=" * 80)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary.to_string(index=False))
    print(f"\nwrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
