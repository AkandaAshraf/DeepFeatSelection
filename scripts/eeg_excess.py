"""Drivenness of the EEG network, interictal versus ictal (CHB-MIT chb01).

The clinical deployment, pre-registered in paper/validation_protocol.md
before the data was opened, under the caveat that governs every output here:
SEIZURE-ONSET ZONES ARE SOURCES AND SOURCES ARE INVISIBLE TO THIS STATISTIC.
The tool maps the spread network -- which channels the brain-wide discharge
drives -- never the origin. Any use of these maps must carry that sentence.

Pre-registered predictions:
1. Ghost ~ 0 in every window (validity gate; no biology needed).
2. Drivenness CONCENTRATION (top-4 share and Gini of clamped-positive
   excess across the 23 channels) is higher in ictal windows than
   interictal ones: a seizure recruits the network into a driven regime.
3. The ictal channel-drivenness pattern is REPRODUCIBLE across chb01's
   seizures (mean pairwise Spearman of per-channel excess across seizure
   windows, compared to interictal-window pairs).

Declared constants (sensitivity reported, never tuned): 256 Hz mean-pooled
4x to 64 Hz; E=3, tau=1; windows are the annotated seizure span padded
symmetrically to at least 2,000 samples; interictal windows are the same
length drawn >= 5 minutes clear of any seizure, plus windows from the
seizure-free records; 4-encoder ensembles per window (V=24 with ghost is
tiny -- CPU, seconds); poly-3 ridge excess, ghost appended, standing
pipeline throughout.

    python scripts/eeg_excess.py
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import pyedflib
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent))
from bottleneck_membership import MaskedAE  # noqa: E402
from excess_membership import poly_own, r2_clamped  # noqa: E402
from deepfeatselect.ccm import time_delay_embed  # noqa: E402

E = 3
TAU = 1
DOWNSAMPLE = 4          # 256 -> 64 Hz
MIN_WINDOW = 2000       # samples at 64 Hz (~31 s)
N_MODELS = 4
EPOCHS = 25
BOTTLENECK = 8          # 23 channels; the joint state is small
TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2


def read_record(path: Path) -> tuple[np.ndarray, list[str], float]:
    f = pyedflib.EdfReader(str(path))
    labels = f.getSignalLabels()
    keep = [i for i, l in enumerate(labels) if l and l != "-"]
    sig = np.stack([f.readSignal(i) for i in keep], axis=1)
    fs = f.getSampleFrequency(keep[0])
    f._close()
    n = (len(sig) // DOWNSAMPLE) * DOWNSAMPLE
    sig = sig[:n].reshape(-1, DOWNSAMPLE, sig.shape[1]).mean(axis=1)
    return sig.astype(np.float64), [labels[i] for i in keep], fs / DOWNSAMPLE


def parse_seizures(summary: str) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    blocks = re.findall(r"File Name: (chb01_\d+\.edf).*?Number of Seizures"
                        r" in File: (\d+)(.*?)(?=File Name:|\Z)", summary, re.S)
    for name, n, rest in blocks:
        if int(n) > 0:
            out[name] = [(int(a), int(b)) for a, b in re.findall(
                r"Seizure Start Time: (\d+) seconds.*?Seizure End Time:"
                r" (\d+) seconds", rest, re.S)]
    return out


def splits_for(n: int) -> tuple[slice, slice, slice]:
    a = int(TRAIN_FRACTION * n)
    b = int((TRAIN_FRACTION + VAL_FRACTION) * n)
    return slice(0, a - E), slice(a, b - E), slice(b, n)


def window_excess(seg: np.ndarray, seed0: int) -> tuple[np.ndarray, float]:
    """Per-channel consensus excess for one window; returns (excess, ghost)."""
    V = seg.shape[1]
    z = (seg - seg.mean(0)) / (seg.std(0) + 1e-12)
    z = np.diff(z, axis=0)
    mats = [time_delay_embed(z[:, j], E, tau=TAU)[0] for j in range(V)]
    n = min(len(m) for m in mats)
    joint = np.hstack([m[:n] for m in mats])
    rng = np.random.default_rng(seed0 + 7331)
    donor = int(rng.integers(0, V))
    ghost = np.roll(joint[:, donor * E:(donor + 1) * E],
                    int(rng.integers(n // 4, 3 * n // 4)), axis=0)
    joint = np.hstack([joint, ghost])
    v_all = V + 1
    tr, va, te = splits_for(n)
    mu, sd = joint[tr].mean(0), joint[tr].std(0) + 1e-12
    zs = ((joint - mu) / sd).astype("float32")

    lead = zs[:, [j * E for j in range(v_all)]]
    tr_idx = np.arange(tr.start, tr.stop - 1)
    te_idx = np.arange(te.start, n - 1)

    def fit_pair(q, extra):
        own_p = poly_own(zs[:, q * E:(q + 1) * E], 3)
        src_tr, src_te = own_p[tr_idx], own_p[te_idx]
        if extra is not None:
            src_tr = np.hstack([src_tr, extra[tr_idx]])
            src_te = np.hstack([src_te, extra[te_idx]])
        m = Ridge(alpha=1.0)
        m.fit(src_tr, lead[tr_idx + 1, q])
        return r2_clamped(m.predict(src_te), lead[te_idx + 1, q])

    self_r2 = np.array([fit_pair(q, None) for q in range(v_all)])
    excess_all = []
    for m_i in range(N_MODELS):
        keras.utils.set_random_seed(seed0 * 100 + m_i)
        model = MaskedAE(v_all, E, BOTTLENECK, mask_mode="zero",
                         loss_on_masked_only=True)
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(3e-3))
        model.fit(zs[tr], zs[tr], epochs=EPOCHS, batch_size=64,
                  shuffle=True, verbose=0)
        code = model.encoder.predict(zs, verbose=0, batch_size=1024)
        excess_all.append(np.array([fit_pair(q, code) - self_r2[q]
                                    for q in range(v_all)]))
    ex = np.mean(excess_all, axis=0)
    return ex[:V], float(ex[V])


def concentration(ex: np.ndarray) -> dict[str, float]:
    v = np.clip(ex, 0, None)
    total = v.sum()
    if total <= 0:
        return {"top4_share": 0.0, "gini": 0.0}
    s = np.sort(v)[::-1]
    idx = np.arange(1, len(v) + 1)
    gini = float((2 * (idx * np.sort(v)).sum()) / (len(v) * total)
                 - (len(v) + 1) / len(v))
    return {"top4_share": float(s[:4].sum() / total), "gini": gini}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="Data/eeg")
    p.add_argument("--outdir", default="ExpOutput/eeg_excess")
    args = p.parse_args()
    root = Path(args.root)
    seizures = parse_seizures((root / "chb01-summary.txt").read_text())

    fs64 = 64.0
    rows, per_channel = [], []
    seed0 = 0
    for path in sorted(root.glob("chb01_*.edf")):
        sig, labels, fs = read_record(path)
        name = path.name
        spans = seizures.get(name, [])
        windows = []
        for (s0, s1) in spans:
            a, b = int(s0 * fs64), int(s1 * fs64)
            if b - a < MIN_WINDOW:
                pad = (MIN_WINDOW - (b - a)) // 2 + 1
                a, b = max(0, a - pad), min(len(sig), b + pad)
            windows.append(("ictal", a, b))
        # Interictal window(s): same length as the record's first seizure
        # window (or MIN_WINDOW), >= 5 min clear of every seizure.
        length = (windows[0][2] - windows[0][1]) if windows else MIN_WINDOW
        clear = int(300 * fs64)
        cand = 0
        while cand + length < len(sig):
            ok = all(not (cand < b + clear and cand + length > a - clear)
                     for _, a, b in windows)
            if ok:
                windows.append(("interictal", cand, cand + length))
                break
            cand += length
        for kind, a, b in windows:
            seed0 += 1
            t0 = time.time()
            ex, ghost = window_excess(sig[a:b], seed0)
            c = concentration(ex)
            rows.append({"record": name, "kind": kind, "start_s": a / fs64,
                         "len_s": (b - a) / fs64, "ghost": ghost, **c,
                         "mean_excess": float(ex.mean()),
                         "seconds": time.time() - t0})
            for lab, v in zip(labels, ex):
                per_channel.append({"record": name, "kind": kind,
                                    "start_s": a / fs64, "channel": lab,
                                    "excess": float(v)})
            r = rows[-1]
            print(f"{name} {kind:<11} [{r['start_s']:6.0f}s +{r['len_s']:3.0f}s] "
                  f"ghost {ghost:+.4f} top4 {c['top4_share']:.3f} "
                  f"gini {c['gini']:.3f} ({r['seconds']:.0f}s)")

    frame = pd.DataFrame(rows)
    chan = pd.DataFrame(per_channel)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outdir / "windows.csv", index=False)
    chan.to_csv(outdir / "channels.csv", index=False)

    print("\n" + "=" * 78)
    print("PREDICTIONS  (source-invisibility caveat: this maps SPREAD, not origin)")
    print("=" * 78)
    ict = frame[frame.kind == "ictal"]
    inter = frame[frame.kind == "interictal"]
    print(f"1. ghost: max |ghost| = {frame.ghost.abs().max():.4f}")
    print(f"2. concentration: ictal top4 {ict.top4_share.mean():.3f} "
          f"(n={len(ict)}) vs interictal {inter.top4_share.mean():.3f} "
          f"(n={len(inter)});  gini {ict.gini.mean():.3f} vs {inter.gini.mean():.3f}")

    piv = chan[chan.kind == "ictal"].pivot_table(
        index="channel", columns=["record", "start_s"], values="excess")
    if piv.shape[1] >= 2:
        cors = []
        cols = list(piv.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                cors.append(piv[cols[i]].corr(piv[cols[j]], method="spearman"))
        print(f"3. cross-seizure reproducibility: mean pairwise Spearman "
              f"{np.mean(cors):.3f} over {len(cors)} pairs")
        top = piv.mean(axis=1).sort_values(ascending=False)
        print("\n   most-driven channels during seizures (mean excess):")
        for ch, v in top.head(6).items():
            print(f"     {ch:<12} {v:+.4f}")
    print(f"\nwrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
