"""Gate check: can WormWideWeb freely-moving recordings carry MACE at all?

DECLARED BEFORE EXECUTION (2026-08-16), per the standing protocol. Everything
below was fixed after inspecting only one dataset's shape (151 x 1600) and no
activity values.

Datasets (downloaded before this header was written, values unseen):
  baseline  atanas_kim_2023-2023-01-23-01
  baseline  atanas_kim_2023-2022-06-14-01
  GFP       atanas_kim_2023-2022-01-07-03   <- activity-INDEPENDENT fluorophore

Pipeline, matched to the validated worm deployment wherever possible:
  per-channel standardisation on train statistics, first-differencing,
  E=3; tau=2 frames (dt ~0.60 s so tau*dt ~1.2 s, nearest to the ~1 s
  convention declared for calcium at 2.9 Hz); contiguous 0.6/0.2/0.2 split
  with an embedding-span embargo; degree-3 polynomial self-baseline; ridge
  alpha=1; R^2 clamped at zero. Encoder: masked AE, tanh(max(2b,64)) ->
  linear bottleneck b=32 -> tanh, mask fraction 0.25, loss on masked
  positions only, Adam 3e-3, batch 64, 25 epochs, M=4 ensemble. Ghost
  panel: 50 donors, circular shifts drawn from the middle half.

THE GATES, with predictions fixed now:

  G1 LENGTH. n=1600 < 2000, the validated floor. Declared marginal before
     any computation; nothing below can promote the data above this.

  G2 SATURATION. Median and max self-R^2, fraction above 0.9. Prediction:
     far from saturation (immobilised worms gave max 0.35), so Prop 1's
     guarantee will NOT hold here and the ghost is descriptive only.

  G3 STATIONARITY. Freely-moving worms switch behavioural states, which is
     nonstationarity by construction. Prediction: the ghost panel median
     sits visibly above zero on the baseline recordings, i.e. the paper's
     own discard rule fires. If instead the panel is clean, freely-moving
     data is usable and that is worth knowing.

  G4 ARTIFACT FLOOR (GFP). In a GFP worm every trace is motion, not
     activity. Prediction: the GFP recording shows positive "excess" for
     some channels (shared motion is predictable from other channels), and
     the size of that excess defines the platform's artifact floor. Any
     future MACE claim on this platform must exceed the GFP floor, not
     zero. If GFP excess is ~zero, the floor is negligible.

  VERDICT RULE, fixed now: the platform is usable for mutant phenotyping
  only if G3 passes on baseline recordings AND the baseline top-channel
  excess exceeds both its own ghost panel maximum and the GFP floor.
  Otherwise the honest conclusion is that freely-moving WormWideWeb data
  cannot carry MACE as-is, and immobilised datasets are the fallback.

Wall-clock is reported per stage: speed is a headline claim of the method
and is measured, not asserted.

    python scripts/wormwideweb_gate.py
"""

from __future__ import annotations

import bz2
import json
import time
from pathlib import Path

import numpy as np

try:
    import torch
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:  # torch still installing; numpy fallback keeps the gate honest
    torch = None
    DEV = "numpy"

ROOT = Path("Data/wormwideweb")
SETS = [("baseline-1", "atanas_kim_2023-2023-01-23-01"),
        ("baseline-2", "atanas_kim_2023-2022-06-14-01"),
        ("GFP", "atanas_kim_2023-2022-01-07-03")]
E, TAU, DEGREE, ALPHA = 3, 2, 3, 1.0
BOTTLENECK, MASK_FRAC, EPOCHS, BATCH, MODELS = 32, 0.25, 25, 64, 4
N_GHOSTS = 50
TRAIN_F, VAL_F = 0.6, 0.2
SEED = 0
BAR = "=" * 78


def load(uid: str):
    d = json.loads(bz2.open(ROOT / f"{uid}.json.bz2").read())
    x = np.asarray(d["gcamp"]["trace_array"], dtype=np.float64).T  # (T, V)
    dt = float(d["timing"]["mean_timestep"])
    return x, dt


def embed(x: np.ndarray) -> np.ndarray:
    """(T, V) -> (n, V*E) delay embedding with lag TAU, channel-major blocks."""
    T = x.shape[0]
    span = (E - 1) * TAU
    n = T - span
    cols = [x[span - k * TAU: span - k * TAU + n, :] for k in range(E)]
    return np.concatenate([np.stack([c[:, j] for c in cols], axis=1)
                           for j in range(x.shape[1])], axis=1)


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


def to_dev(a):
    return torch.as_tensor(a, dtype=torch.float32, device=DEV)


def ridge_r2(feats: np.ndarray, target: np.ndarray, tr, te) -> float:
    """Closed-form ridge on train rows, clamped R^2 on held-out tail."""
    X, y = feats[tr], target[tr]
    Xt, yt = feats[te], target[te]
    if torch is not None:
        X, y, Xt, yt = map(to_dev, (X, y, Xt, yt))
        A = X.T @ X + ALPHA * torch.eye(X.shape[1], device=X.device)
        w = torch.linalg.solve(A, X.T @ y)
        err = float(((Xt @ w - yt) ** 2).mean())
        var = float(yt.var())
    else:
        A = X.T @ X + ALPHA * np.eye(X.shape[1])
        w = np.linalg.solve(A, X.T @ y)
        err = float(np.mean((Xt @ w - yt) ** 2))
        var = float(yt.var())
    return max(0.0, 1.0 - err / (var + 1e-12))


class MaskedAE(torch.nn.Module if torch else object):
    def __init__(self, d_in: int, b: int):
        super().__init__()
        h = max(2 * b, 64)
        self.enc = torch.nn.Sequential(
            torch.nn.Linear(d_in, h), torch.nn.Tanh(), torch.nn.Linear(h, b))
        self.dec = torch.nn.Sequential(
            torch.nn.Linear(b, h), torch.nn.Tanh(), torch.nn.Linear(h, d_in))

    def forward(self, z):
        return self.dec(self.enc(z))


def train_codes(zs: np.ndarray, v: int, tr) -> list[np.ndarray]:
    """M masked AEs on the train rows; return full-length codes."""
    g = torch.Generator(device="cpu").manual_seed(SEED)
    ztr = to_dev(zs[tr])
    zfull = to_dev(zs)
    codes = []
    for m in range(MODELS):
        torch.manual_seed(SEED + m)
        net = MaskedAE(zs.shape[1], BOTTLENECK).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        for _ in range(EPOCHS):
            perm = torch.randperm(ztr.shape[0], generator=g)
            for i in range(0, len(perm), BATCH):
                batch = ztr[perm[i:i + BATCH]]
                mask = (torch.rand(batch.shape[0], v, device=DEV) < MASK_FRAC)
                mask_cols = mask.repeat_interleave(E, dim=1)
                corrupted = batch.masked_fill(mask_cols, 0.0)
                out = net(corrupted)
                loss = ((out - batch)[mask_cols] ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            codes.append(net.enc(zfull).cpu().numpy())
    return codes


def run(tag: str, uid: str):
    t0 = time.time()
    x, dt = load(uid)
    T, V = x.shape
    span = (E - 1) * TAU
    emb = embed(x)
    n = emb.shape[0]
    a = int(TRAIN_F * n); b = int((TRAIN_F + VAL_F) * n)
    tr = slice(0, a - span); te = slice(b, n - 1)
    mu, sd = emb[tr].mean(0), emb[tr].std(0) + 1e-12
    zs = ((emb - mu) / sd).astype(np.float32)
    lead = zs[:, [j * E for j in range(V)]]
    tr_i = np.arange(tr.start, tr.stop - 1)
    te_i = np.arange(te.start, n - 1)
    t_load = time.time() - t0

    # --- G2: self-baselines ------------------------------------------------
    t0 = time.time()
    self_r2 = np.empty(V)
    for q in range(V):
        own = zs[:, q * E:(q + 1) * E]
        feats = poly3(own)
        Xtr, ytr = feats[tr_i], lead[tr_i + 1, q]
        Xte, yte = feats[te_i], lead[te_i + 1, q]
        if torch is not None:
            Xt, yt2, Xe, ye = map(to_dev, (Xtr, ytr, Xte, yte))
            A = Xt.T @ Xt + ALPHA * torch.eye(Xt.shape[1], device=Xt.device)
            w = torch.linalg.solve(A, Xt.T @ yt2)
            err = float(((Xe @ w - ye) ** 2).mean()); var = float(ye.var())
        else:
            A = Xtr.T @ Xtr + ALPHA * np.eye(Xtr.shape[1])
            w = np.linalg.solve(A, Xtr.T @ ytr)
            err = float(np.mean((Xte @ w - yte) ** 2)); var = float(yte.var())
        self_r2[q] = max(0.0, 1.0 - err / (var + 1e-12))
    t_self = time.time() - t0

    # --- codes -------------------------------------------------------------
    t0 = time.time()
    codes = train_codes(zs, V, tr)
    t_train = time.time() - t0

    # --- excess for every channel + ghost panel ----------------------------
    t0 = time.time()
    rng = np.random.default_rng(SEED + 4242)
    donors = rng.choice(V, size=min(N_GHOSTS, V), replace=False)
    shifts = rng.integers(n // 4, 3 * n // 4, size=len(donors))

    def excess_of(own_block: np.ndarray, target: np.ndarray) -> float:
        feats = poly3(own_block)
        exs = []
        base = None
        for c in codes:
            Xtr = np.hstack([feats[tr_i], c[tr_i]])
            Xte = np.hstack([feats[te_i], c[te_i]])
            if base is None:
                A0, y0 = feats[tr_i], target[tr_i + 1]
                if torch is not None:
                    Xt, yt2 = to_dev(A0), to_dev(y0)
                    M0 = Xt.T @ Xt + ALPHA * torch.eye(Xt.shape[1], device=Xt.device)
                    w0 = torch.linalg.solve(M0, Xt.T @ yt2)
                    Xe, ye = to_dev(feats[te_i]), to_dev(target[te_i + 1])
                    err = float(((Xe @ w0 - ye) ** 2).mean()); var = float(ye.var())
                else:
                    M0 = A0.T @ A0 + ALPHA * np.eye(A0.shape[1])
                    w0 = np.linalg.solve(M0, A0.T @ y0)
                    err = float(np.mean((feats[te_i] @ w0 - target[te_i + 1]) ** 2))
                    var = float(np.var(target[te_i + 1]))
                base = max(0.0, 1.0 - err / (var + 1e-12))
            if torch is not None:
                Xt, yt2 = to_dev(Xtr), to_dev(target[tr_i + 1])
                M1 = Xt.T @ Xt + ALPHA * torch.eye(Xt.shape[1], device=Xt.device)
                w1 = torch.linalg.solve(M1, Xt.T @ yt2)
                Xe, ye = to_dev(Xte), to_dev(target[te_i + 1])
                err = float(((Xe @ w1 - ye) ** 2).mean()); var = float(ye.var())
            else:
                M1 = Xtr.T @ Xtr + ALPHA * np.eye(Xtr.shape[1])
                w1 = np.linalg.solve(M1, Xtr.T @ target[tr_i + 1])
                err = float(np.mean((Xte @ w1 - target[te_i + 1]) ** 2))
                var = float(np.var(target[te_i + 1]))
            exs.append(max(0.0, 1.0 - err / (var + 1e-12)) - base)
        return float(np.mean(exs))

    excess = np.array([excess_of(zs[:, q * E:(q + 1) * E], lead[:, q])
                       for q in range(V)])
    ghosts = []
    for d, s in zip(donors, shifts):
        gz = np.roll(zs[:, d * E:(d + 1) * E], int(s), axis=0)
        gt = np.roll(lead[:, d], int(s))
        ghosts.append(excess_of(gz, gt))
    ghosts = np.array(ghosts)
    t_scan = time.time() - t0

    return dict(tag=tag, uid=uid, T=T, V=V, dt=dt, n=n,
                self_med=float(np.median(self_r2)), self_max=float(self_r2.max()),
                self_f90=float((self_r2 > 0.9).mean()),
                ex_top=float(np.sort(excess)[-1]),
                ex_top5=np.sort(excess)[-5:][::-1].round(4).tolist(),
                ex_pos=int((excess > 0).sum()),
                g_med=float(np.median(ghosts)), g_p95=float(np.quantile(ghosts, .95)),
                g_max=float(ghosts.max()), g_pos=float((ghosts > 0).mean()),
                t_load=t_load, t_self=t_self, t_train=t_train, t_scan=t_scan)


def main() -> int:
    print(f"device: {DEV}" + (f" ({torch.cuda.get_device_name(0)})"
                              if torch is not None and DEV == "cuda" else ""))
    rows = [run(tag, uid) for tag, uid in SETS]

    print("\n" + BAR)
    print("GATE CHECK: WormWideWeb freely-moving recordings")
    print(BAR)
    for r in rows:
        print(f"\n[{r['tag']}]  {r['uid']}   V={r['V']}  T={r['T']}  "
              f"dt={r['dt']:.2f}s  n={r['n']}")
        print(f"  G1 length      : n={r['n']} vs validated floor 2000  "
              f"{'MARGINAL' if r['n'] < 2000 else 'ok'}")
        print(f"  G2 saturation  : self-R2 median {r['self_med']:.3f}  "
              f"max {r['self_max']:.3f}  frac>0.9 {r['self_f90']:.3f}")
        print(f"  G3 ghost panel : median {r['g_med']:+.4f}  p95 {r['g_p95']:+.4f}  "
              f"max {r['g_max']:+.4f}  frac>0 {r['g_pos']:.2f}")
        print(f"  excess         : top {r['ex_top']:+.4f}  top5 {r['ex_top5']}  "
              f"channels>0: {r['ex_pos']}/{r['V']}")
        print(f"  wall-clock     : load {r['t_load']:.1f}s  self {r['t_self']:.1f}s  "
              f"train {r['t_train']:.1f}s  scan+ghosts {r['t_scan']:.1f}s")

    base = [r for r in rows if r["tag"].startswith("baseline")]
    gfp = [r for r in rows if r["tag"] == "GFP"][0]
    print("\n" + BAR)
    print("VERDICT (rule fixed in header before execution)")
    print(BAR)
    floor = gfp["ex_top"]
    print(f"GFP artifact floor (top excess in activity-free worm): {floor:+.4f}")
    for r in base:
        ok3 = r["g_med"] <= 0
        okx = r["ex_top"] > max(r["g_max"], floor)
        print(f"  {r['tag']}: G3 {'pass' if ok3 else 'FAIL'} "
              f"(ghost median {r['g_med']:+.4f});  top excess "
              f"{r['ex_top']:+.4f} vs max(ghost {r['g_max']:+.4f}, "
              f"GFP floor {floor:+.4f}) -> {'clears' if okx else 'DOES NOT clear'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
