"""Corpus scan: MACE drivenness fingerprints across all 91 WormWideWeb
freely-moving recordings.

DECLARED BEFORE EXECUTION (2026-08-16), after the three-recording gate check
and before any other recording was opened. The dataset index was frozen in
Data/wormwideweb/corpus_index.json before this header was written:
35 baseline, 30 heat, 8 reFed, 8 sickness, 7 patchEncounter, 3 gfp.

Pipeline: identical to scripts/wormwideweb_gate.py (constants imported from
it). One upgrade, licensed by the gate result that 10-15% of channels exceed
self-R^2 0.9 on this platform: GHOST DONORS ARE FILTERED to channels with
self-R^2 > 0.9 where at least 8 such channels exist (panel up to 50 donors,
shifts from the middle half); otherwise uniform donors are used and the
dataset is flagged fallback=True. This is the corrected donor rule of the
paper, applicable on real data for the first time.

Per-dataset detection threshold, fixed now:  thr = max(0, ghost panel max).
Platform artifact floor, fixed now: FLOOR = max over the 3 GFP recordings of
their top channel excess. A channel "clears" if excess > max(thr, FLOOR).

PRE-REGISTERED PREDICTIONS:

  P1 (primary). Pooled across the 35 baseline recordings, among labelled
     channels (neuron_class field, confidence >= 2), the channels that clear
     are enriched for the command/motor ensemble, using VERBATIM the
     COMMAND_MOTOR class set already declared in scripts/celegans_excess.py
     for the immobilised worms. Test: one-sided hypergeometric, p < 0.05.
     This asks whether the paper's central worm claim replicates in FREELY
     MOVING animals under a reimplemented (torch) pipeline.

  P2. The GFP floor is materially positive (> 0.02): shared motion
     masquerades as drivenness and must be subtracted from every claim.

  P3. The non-baseline arms (heat, sickness, reFed, patchEncounter) are
     EXPLORATORY. Descriptive statistics only; no claims, no tests.

Outputs: ExpOutput/wormwideweb/scan_summary.csv, per-channel CSVs under
ExpOutput/wormwideweb/channels/, enrichment.json, and wall-clock totals
(speed is a headline claim; it is measured).

    python scripts/wormwideweb_corpus.py
"""

from __future__ import annotations

import bz2
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from wormwideweb_gate import (ALPHA, E, MODELS, N_GHOSTS, TAU, TRAIN_F, VAL_F,
                              MaskedAE, embed, poly3, to_dev, train_codes)
import torch

ROOT = Path("Data/wormwideweb")
OUT = Path("ExpOutput/wormwideweb")
SEED = 0
CONF_MIN = 2.0
SELF_DONOR = 0.9
MIN_DONORS = 8

# Verbatim from scripts/celegans_excess.py (declared for the immobilised worms
# before any freely-moving data was seen).
COMMAND_MOTOR = {"AVA", "AVB", "AVE", "AVD", "RIM", "RIB", "RIA", "AIB",
                 "AIY", "AIA", "AIZ", "RIS", "RID", "RIV", "RIF", "RME",
                 "RMD", "SMD", "SMB", "SIB", "SIA", "SAB", "VB", "DB",
                 "VA", "DA", "VD", "DD", "AS", "PVC", "AVF", "AVJ"}


def load_full(uid: str):
    d = json.loads(bz2.open(ROOT / f"{uid}.json.bz2").read())
    x = np.asarray(d["gcamp"]["trace_array"], dtype=np.float64).T
    dt = float(d["timing"]["mean_timestep"])
    labels = {}
    for k, v in d.get("label", {}).items():
        try:
            idx = int(k)
        except ValueError:
            continue
        labels[idx] = (str(v.get("label", "")), str(v.get("neuron_class", "")),
                       float(v.get("confidence") or 0))
    return x, dt, labels


def solve_r2(feats_tr, y_tr, feats_te, y_te) -> float:
    X, y, Xe, ye = map(to_dev, (feats_tr, y_tr, feats_te, y_te))
    A = X.T @ X + ALPHA * torch.eye(X.shape[1], device=X.device)
    w = torch.linalg.solve(A, X.T @ y)
    err = float(((Xe @ w - ye) ** 2).mean())
    return max(0.0, 1.0 - err / (float(ye.var()) + 1e-12))


def scan(uid: str):
    x, dt, labels = load_full(uid)
    T, V = x.shape
    span = (E - 1) * TAU
    emb = embed(x)
    n = emb.shape[0]
    a = int(TRAIN_F * n)
    b = int((TRAIN_F + VAL_F) * n)
    tr = slice(0, a - span)
    te = slice(b, n - 1)
    mu, sd = emb[tr].mean(0), emb[tr].std(0) + 1e-12
    zs = ((emb - mu) / sd).astype(np.float32)
    lead = zs[:, [j * E for j in range(V)]]
    tr_i = np.arange(tr.start, tr.stop - 1)
    te_i = np.arange(te.start, n - 1)

    feats_all = [poly3(zs[:, q * E:(q + 1) * E]) for q in range(V)]
    self_r2 = np.array([solve_r2(f[tr_i], lead[tr_i + 1, q], f[te_i],
                                 lead[te_i + 1, q])
                        for q, f in enumerate(feats_all)])

    codes = train_codes(zs, V, tr)

    def excess_of(feats, target):
        base = solve_r2(feats[tr_i], target[tr_i + 1],
                        feats[te_i], target[te_i + 1])
        vals = []
        for c in codes:
            vals.append(solve_r2(np.hstack([feats[tr_i], c[tr_i]]),
                                 target[tr_i + 1],
                                 np.hstack([feats[te_i], c[te_i]]),
                                 target[te_i + 1]) - base)
        return float(np.mean(vals))

    excess = np.array([excess_of(feats_all[q], lead[:, q]) for q in range(V)])

    rng = np.random.default_rng(SEED + 4242)
    qual = np.where(self_r2 > SELF_DONOR)[0]
    fallback = len(qual) < MIN_DONORS
    pool = np.arange(V) if fallback else qual
    donors = rng.choice(pool, size=min(N_GHOSTS, len(pool)), replace=False)
    ghosts = []
    for dnr in donors:
        s = int(rng.integers(n // 4, 3 * n // 4))
        gz = np.roll(zs[:, dnr * E:(dnr + 1) * E], s, axis=0)
        ghosts.append(excess_of(poly3(gz), np.roll(lead[:, dnr], s)))
    ghosts = np.array(ghosts)
    thr = max(0.0, float(ghosts.max()))

    rows = []
    for q in range(V):
        lab, cls, conf = labels.get(q, ("", "", 0.0))
        rows.append({"channel": q, "label": lab, "neuron_class": cls,
                     "confidence": conf, "self_r2": self_r2[q],
                     "excess": excess[q]})
    return (pd.DataFrame(rows),
            dict(uid=uid, V=V, T=T, n=n, dt=dt,
                 self_med=float(np.median(self_r2)),
                 self_max=float(self_r2.max()),
                 self_f90=float((self_r2 > 0.9).mean()),
                 donor_fallback=bool(fallback), n_donors=int(len(donors)),
                 ghost_med=float(np.median(ghosts)),
                 ghost_max=float(ghosts.max()), thr=thr,
                 ex_top=float(excess.max()),
                 n_above_thr=int((excess > thr).sum())))


def main() -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    idx = json.load(open(ROOT / "corpus_index.json"))
    (OUT / "channels").mkdir(parents=True, exist_ok=True)

    t_all = time.time()
    summaries = []
    per_channel = {}
    for i, o in enumerate(idx, 1):
        t0 = time.time()
        try:
            frame, summ = scan(o["id"])
        except Exception as exc:
            print(f"[{i:2d}/{len(idx)}] {o['id']}  ERROR {exc}")
            continue
        summ["kind"] = o["kind"]
        summ["seconds"] = round(time.time() - t0, 1)
        summaries.append(summ)
        per_channel[o["id"]] = frame
        frame.to_csv(OUT / "channels" / f"{o['id']}.csv", index=False)
        print(f"[{i:2d}/{len(idx)}] {o['id']:38s} {o['kind']:14s} "
              f"V={summ['V']:3d} top={summ['ex_top']:+.3f} "
              f"thr={summ['thr']:+.4f} n>thr={summ['n_above_thr']:3d} "
              f"{summ['seconds']:.0f}s")

    sm = pd.DataFrame(summaries)
    sm.to_csv(OUT / "scan_summary.csv", index=False)

    # ---- FLOOR from the GFP arm (rule fixed in header) --------------------
    gfp = sm[sm.kind == "gfp"]
    floor = float(gfp.ex_top.max()) if len(gfp) else float("nan")
    print(f"\nGFP artifact floor: {floor:+.4f} "
          f"(from {len(gfp)} activity-free recordings)")

    # ---- P1: command/motor enrichment among clearing channels -------------
    lab_rows = []
    for o in idx:
        if o["kind"] != "baseline" or o["id"] not in per_channel:
            continue
        s = sm[sm.uid == o["id"]].iloc[0]
        f = per_channel[o["id"]]
        f = f[(f.neuron_class != "") & (f.confidence >= CONF_MIN)].copy()
        f["clears"] = f.excess > max(s.thr, floor)
        f["command"] = f.neuron_class.isin(COMMAND_MOTOR)
        lab_rows.append(f)
    pool = pd.concat(lab_rows)
    M = len(pool)
    K = int(pool.command.sum())
    N = int(pool.clears.sum())
    k = int((pool.clears & pool.command).sum())
    from scipy.stats import hypergeom
    p = float(hypergeom.sf(k - 1, M, K, N)) if N else float("nan")

    print("\nP1 command/motor enrichment (baseline, labelled, conf>=2):")
    print(f"  labelled channels pooled: {M}  command-class: {K} "
          f"({K/M:.1%} base rate)")
    print(f"  clearing max(thr, floor): {N}  of which command-class: {k} "
          f"({k/max(N,1):.1%})")
    print(f"  one-sided hypergeometric p = {p:.4g}   "
          f"({'PASS' if p < 0.05 else 'FAIL'} at 0.05)")
    top_ids = (pool[pool.clears].neuron_class.value_counts().head(12))
    print("  clearing classes:", dict(top_ids))

    json.dump({"floor": floor, "M": M, "K": K, "N": N, "k": k, "p": p},
              open(OUT / "enrichment.json", "w"), indent=1)

    print("\nby condition:")
    print(sm.groupby("kind")[["V", "self_med", "thr", "ex_top", "n_above_thr"]
                             ].mean().round(4).to_string())
    print(f"\nTOTAL wall-clock: {(time.time()-t_all)/60:.1f} min for "
          f"{len(sm)} recordings on {dev}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
