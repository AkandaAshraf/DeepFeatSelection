"""Gate check: can intracranial EEG carry MACE?

Pre-registration: paper/ieeg_protocol.md, written before any recording was
downloaded. This header restates the gate; the protocol is authoritative.

Two interictal SEEG subjects from OpenNeuro ds003876. Channel tables are
parsed for NAME and TYPE ONLY; no SOZ, resection or outcome column is read,
displayed or stored (declared in the protocol, enforced here by selecting
columns explicitly).

Gates: G1 length (expected pass by construction, recorded); G2 saturation
(the open question: do field potentials saturate the self-baseline where
calcium could not?); G3 ghost panel (stationarity); G4 shared-reference
floor: the pipeline runs twice per subject, common-average referenced and
raw-referenced, and the difference bounds the reference artifact, since a
shared reference is shared signal exactly as shared motion was in the
freely-moving worm.

Constants per protocol: decimate 1024 -> 256 Hz by mean-pooling 4; middle
120 s segment; E=3, tau=1; first-difference; 0.6/0.2/0.2 contiguous splits,
embargo E; degree-3 self-baseline, ridge alpha 1; MaskedAE M=4, b=32, mask
0.25, Adam 3e-3, batch 64, 25 epochs; 50-donor ghost panel, donors filtered
to self-R2 > 0.9 when at least 8 qualify (fallback flagged).

    python scripts/ieeg_gate.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import torch
from wormwideweb_gate import MaskedAE  # architecture reused verbatim

DATA = Path("Data/ieeg876")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
DECIM, SEG_SECONDS = 4, 120
E, TAU, DEGREE, ALPHA = 3, 1, 3, 1.0
BOTTLENECK, MASK, EPOCHS, BATCH, MODELS = 32, 0.25, 25, 64, 4
N_GHOSTS, DONOR_R2, MIN_DONORS = 50, 0.9, 8
SUBJECTS = ["NIH1", "NIH2"]
BAR = "=" * 78


def read_edf(path: Path):
    """Minimal EDF reader: header + int16 records -> (T, V) float64, labels."""
    with open(path, "rb") as f:
        hdr = f.read(256)
        n_recs = int(hdr[236:244])
        rec_dur = float(hdr[244:252])
        ns = int(hdr[252:256])
        sig = f.read(ns * 256)

        def field(off, ln):
            return [sig[off * ns + i * ln: off * ns + i * ln + ln]
                    .decode("ascii", "replace").strip() for i in range(ns)]
        labels = field(0, 16)
        phys_min = np.array([float(x) for x in field(0, 8)]) if False else None
        # proper offsets per EDF spec
        lab = [sig[i * 16:(i + 1) * 16].decode("ascii", "replace").strip()
               for i in range(ns)]
        base = ns * 16 + ns * 80 + ns * 8
        pmin = np.array([float(sig[base + i * 8: base + (i + 1) * 8]) for i in range(ns)])
        pmax = np.array([float(sig[base + ns * 8 + i * 8: base + ns * 8 + (i + 1) * 8]) for i in range(ns)])
        dmin = np.array([float(sig[base + 2 * ns * 8 + i * 8: base + 2 * ns * 8 + (i + 1) * 8]) for i in range(ns)])
        dmax = np.array([float(sig[base + 3 * ns * 8 + i * 8: base + 3 * ns * 8 + (i + 1) * 8]) for i in range(ns)])
        spr_off = ns * 16 + ns * 80 + ns * 8 + 4 * ns * 8 + ns * 80
        spr = np.array([int(sig[spr_off + i * 8: spr_off + (i + 1) * 8]) for i in range(ns)])
        gain = (pmax - pmin) / (dmax - dmin + 1e-12)
        raw = np.fromfile(f, dtype="<i2")
    per_rec = spr.sum()
    n_recs = min(n_recs, len(raw) // per_rec)
    raw = raw[: n_recs * per_rec].reshape(n_recs, per_rec)
    out = np.empty((n_recs * spr.max(), ns), dtype=np.float64)
    # channels may have differing rates; keep only the modal rate
    modal = int(pd.Series(spr).mode()[0])
    keep_ch = [i for i in range(ns) if spr[i] == modal]
    T = n_recs * modal
    out = np.empty((T, len(keep_ch)))
    offs = np.concatenate([[0], np.cumsum(spr)])
    for j, i in enumerate(keep_ch):
        out[:, j] = (raw[:, offs[i]:offs[i + 1]].reshape(-1) * gain[i]
                     + pmin[i] - dmin[i] * gain[i])
    fs = modal / rec_dur
    return out, [lab[i] for i in keep_ch], fs


def norm_label(l: str) -> str:
    """EDF labels look like "EEG RAI1-G2" / "POL RAI1"; the tsv says "RAI1"."""
    l = l.replace("POL ", "").replace("EEG ", "").strip()
    return l.split("-")[0].strip()


def good_channels(sub: str, labels: list[str]) -> list[int]:
    """Indices of SEEG data channels. Reads ONLY name and type columns."""
    tsv = pd.read_csv(DATA / f"sub-{sub}_channels.tsv", sep="\t",
                      usecols=lambda c: c in ("name", "type"))
    want = set(tsv[tsv.type.str.upper().isin(["SEEG", "ECOG"])].name)
    return [i for i, l in enumerate(labels)
            if norm_label(l) in want or l in want]


def montage(x: np.ndarray, labels: list[str], mode: str):
    """raw: as recorded. car: common average. bipolar: within-shaft adjacent
    contact differences (RAI2-RAI1), the standard SEEG local derivation that
    cancels both the shared reference and far-field volume conduction."""
    if mode == "car":
        return x - x.mean(1, keepdims=True), list(labels)
    if mode == "raw":
        return x, list(labels)
    import re as _re
    shafts: dict[str, list[tuple[int, int]]] = {}
    for i, l in enumerate(labels):
        m = _re.match(r"([A-Za-z']+)(\d+)$", l)
        if m:
            shafts.setdefault(m.group(1), []).append((int(m.group(2)), i))
    cols, labs = [], []
    for shaft, contacts in sorted(shafts.items()):
        contacts.sort()
        for (n1, i1), (n2, i2) in zip(contacts, contacts[1:]):
            if n2 == n1 + 1:
                cols.append(x[:, i2] - x[:, i1])
                labs.append(f"{shaft}{n2}-{shaft}{n1}")
    return np.stack(cols, axis=1), labs


def preprocess(x: np.ndarray, labels: list[str], fs: float, mode: str):
    T = (x.shape[0] // DECIM) * DECIM
    x = x[:T].reshape(-1, DECIM, x.shape[1]).mean(1)      # decimate
    fs2 = fs / DECIM
    seg = int(SEG_SECONDS * fs2)
    mid = x.shape[0] // 2
    x = x[mid - seg // 2: mid + seg // 2]
    # guard: drop flat or broken channels before any statistics
    alive = x.std(0) > 1e-6
    x = x[:, alive]
    labels = [l for l, a in zip(labels, alive) if a]
    x, labels = montage(x, labels, mode)
    x = np.diff(x, axis=0)                                # first difference
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    # a channel can go degenerate only after differencing (clipped/held DACs)
    x = x[:, x.std(0) > 1e-9]
    return x, fs2


def embed(x: np.ndarray) -> np.ndarray:
    span = (E - 1) * TAU
    n = x.shape[0] - span
    return np.concatenate(
        [np.stack([x[span - k * TAU: span - k * TAU + n, j] for k in range(E)],
                  axis=1) for j in range(x.shape[1])], axis=1)


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


def ridge(Xtr, ytr, Xte, yte) -> float:
    # float64: raw-referenced iEEG gives near-collinear lag features whose
    # Gram matrix reaches ~1e12, where float32 swallows the ridge entirely.
    Xt = torch.as_tensor(Xtr, dtype=torch.float64, device=DEV)
    yt = torch.as_tensor(ytr, dtype=torch.float64, device=DEV)
    Xe = torch.as_tensor(Xte, dtype=torch.float64, device=DEV)
    ye = torch.as_tensor(yte, dtype=torch.float64, device=DEV)
    A = Xt.T @ Xt + ALPHA * torch.eye(Xt.shape[1], device=DEV,
                                      dtype=torch.float64)
    w = torch.linalg.solve(A, Xt.T @ yt)
    err = float(((Xe @ w - ye) ** 2).mean())
    return max(0.0, 1.0 - err / (float(ye.var()) + 1e-12))


def train_codes(zs: np.ndarray, v: int, tr: slice) -> list[np.ndarray]:
    ztr = torch.as_tensor(zs[tr], device=DEV)
    zfull = torch.as_tensor(zs, device=DEV)
    codes = []
    for m in range(MODELS):
        torch.manual_seed(SEED + m)
        net = MaskedAE(zs.shape[1], BOTTLENECK).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        g = torch.Generator().manual_seed(SEED + m)
        for _ in range(EPOCHS):
            perm = torch.randperm(ztr.shape[0], generator=g)
            for i in range(0, len(perm), BATCH):
                b = ztr[perm[i:i + BATCH]]
                mask = torch.rand(b.shape[0], v, device=DEV) < MASK
                mc = mask.repeat_interleave(E, dim=1)
                out = net(b.masked_fill(mc, 0.0))
                loss = ((out - b)[mc] ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            codes.append(net.enc(zfull).cpu().numpy())
    return codes


def arm(x: np.ndarray, tag: str) -> dict:
    t0 = time.time()
    emb = embed(x)
    n = emb.shape[0]
    a, b = int(0.6 * n), int(0.8 * n)
    span = (E - 1) * TAU
    tr = slice(0, a - span)
    tr_i = np.arange(tr.start, tr.stop - 1)
    te_i = np.arange(b, n - 1)
    mu, sd = emb[tr].mean(0), emb[tr].std(0) + 1e-12
    zs = np.clip(np.nan_to_num((emb - mu) / sd), -20, 20).astype(np.float32)
    V = x.shape[1]
    lead = zs[:, [j * E for j in range(V)]]

    feats = [poly3(zs[:, q * E:(q + 1) * E]) for q in range(V)]
    self_r2 = np.array([ridge(f[tr_i], lead[tr_i + 1, q], f[te_i],
                              lead[te_i + 1, q]) for q, f in enumerate(feats)])
    codes = train_codes(zs, V, tr)

    def excess_of(f, target):
        base = ridge(f[tr_i], target[tr_i + 1], f[te_i], target[te_i + 1])
        vals = [ridge(np.hstack([f[tr_i], c[tr_i]]), target[tr_i + 1],
                      np.hstack([f[te_i], c[te_i]]), target[te_i + 1]) - base
                for c in codes]
        return float(np.mean(vals))

    excess = np.array([excess_of(feats[q], lead[:, q]) for q in range(V)])

    rng = np.random.default_rng(SEED + 4242)
    qual = np.where(self_r2 > DONOR_R2)[0]
    fallback = len(qual) < MIN_DONORS
    pool = np.arange(V) if fallback else qual
    donors = rng.choice(pool, size=min(N_GHOSTS, len(pool)), replace=False)
    ghosts = []
    for d in donors:
        s = int(rng.integers(n // 4, 3 * n // 4))
        gz = np.roll(zs[:, d * E:(d + 1) * E], s, axis=0)
        ghosts.append(excess_of(poly3(gz), np.roll(lead[:, d], s)))
    ghosts = np.array(ghosts)

    return dict(tag=tag, V=V, n=n,
                self_med=float(np.median(self_r2)),
                self_max=float(self_r2.max()),
                self_f90=float((self_r2 > 0.9).mean()),
                donor_fallback=bool(fallback),
                g_med=float(np.median(ghosts)), g_max=float(ghosts.max()),
                g_pos=float((ghosts > 0).mean()),
                ex_top=float(np.sort(excess)[-1]),
                ex_top5=np.sort(excess)[-5:][::-1].round(4).tolist(),
                n_above_gmax=int((excess > max(0.0, ghosts.max())).sum()),
                secs=round(time.time() - t0, 1))


def main() -> int:
    print(f"device: {DEV}")
    subs = sorted(f.name.split("_")[0][4:]
                  for f in DATA.glob("sub-*_run-01_ieeg.edf")
                  if (DATA / f"{f.name.split('_')[0]}_channels.tsv").exists())
    print(f"cohort: {len(subs)} subjects {subs}")
    # Resumable: each (subject, montage) row is written as it completes, so an
    # interruption costs one arm rather than the whole cohort. A 29-subject
    # cohort is hours of GPU and the first attempt was lost to a power cut.
    dest = Path("ExpOutput/ieeg_gate_cohort.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    done, rows = set(), []
    if dest.exists():
        rows = pd.read_csv(dest).to_dict("records")
        done = {(r["sub"], r["tag"]) for r in rows}
        print(f"resuming: {len(done)} arms already complete")
    for sub in subs:
        if all((sub, m) in done for m in ("car", "raw", "bipolar")):
            print(f"[sub-{sub}] complete, skipping")
            continue
        x, labels, fs = read_edf(DATA / f"sub-{sub}_run-01_ieeg.edf")
        idx = good_channels(sub, labels)
        x = x[:, idx]
        labels_n = [norm_label(labels[i]) for i in idx]
        print(f"\n{BAR}\n[sub-{sub}]  {x.shape[1]} data channels, fs={fs:.0f} Hz, "
              f"{x.shape[0]/fs:.0f} s total")
        for mode in ("car", "raw", "bipolar"):
            if (sub, mode) in done:
                continue
            xp, fs2 = preprocess(x, labels_n, fs, mode)
            r = arm(xp, mode)
            r["sub"] = sub
            rows.append(r)
            pd.DataFrame([{k: v for k, v in q.items() if k != "ex_top5"}
                          for q in rows]).to_csv(dest, index=False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  [{r['tag']:8s}] n={r['n']}  "
                  f"self-R2 med {r['self_med']:.3f} max {r['self_max']:.3f} "
                  f"frac>0.9 {r['self_f90']:.2f} "
                  f"{'(donor fallback)' if r['donor_fallback'] else ''}")
            print(f"             ghosts: med {r['g_med']:+.4f} max {r['g_max']:+.4f} "
                  f"frac>0 {r['g_pos']:.2f}")
            print(f"             excess: top {r['ex_top']:+.4f} top5 {r['ex_top5']} "
                  f"channels>ghostmax: {r['n_above_gmax']}/{r['V']}  "
                  f"({r['secs']}s)")
    print(f"cohort table: {dest} ({len(rows)} arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
