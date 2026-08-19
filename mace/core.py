"""MACE: Masked-Autoencoder Conditional Excess.

Scores each channel of a multivariate time series by drivenness: the gain in
one-step predictability when a learned code of the entire remaining system is
added to a flexible model of the channel's own history. Every scan carries
its own controls; there is no way to obtain scores without them.

    import mace
    result = mace.scan(X)          # X: (timepoints, channels), n >= 2000
    print(result.summary())
    result.to_frame().to_csv("drivenness.csv")

Semantics, stated up front (see the paper for the full account):
  - MACE detects DRIVEN channels. Pure sources are invisible by design.
  - High precision, low recall: absence from the result is not evidence
    that a channel is autonomous.
  - The ghost guarantee is licensed only where the self-baseline saturates
    (self-R^2 near 1); elsewhere the ghost panel is a descriptive noise
    scale, and the report says which regime you are in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import torch
except ImportError as _exc:                                   # pragma: no cover
    torch = None
    _TORCH_ERR = _exc


@dataclass
class MaceConfig:
    """Declared defaults; every value mirrors the released deployments."""
    e: int = 3                 # embedding dimension
    tau: int = 1               # embedding delay, in samples
    degree: int = 3            # polynomial self-baseline order (2 or 3)
    alpha: float = 1.0         # ridge penalty
    bottleneck: int = 32       # code width b
    models: int = 4            # ensemble size M
    mask: float = 0.25         # masked-AE corruption fraction
    epochs: int = 25
    batch: int = 64
    ghosts: int = 50           # ghost panel size
    donor_r2: float = 0.9      # self-R^2 required of a ghost donor
    min_donors: int = 8        # below this, fall back to uniform (flagged)
    train_frac: float = 0.6
    val_frac: float = 0.2
    difference: bool = True    # first-difference before analysis
    zclip: float = 20.0
    seed: int = 0
    device: Optional[str] = None   # None -> cuda if available


@dataclass
class ScanResult:
    """Scores plus the controls that license (or refuse to license) them."""
    excess: np.ndarray            # per-channel consensus excess
    self_r2: np.ndarray           # per-channel self-baseline R^2
    ghosts: np.ndarray            # ghost-panel excess values
    n: int
    V: int
    donor_fallback: bool          # True -> no saturating donors; panel is
    #                               descriptive, not a calibrated null
    config: MaceConfig = field(repr=False, default=None)
    channel_names: Optional[list] = None

    # ---- derived ----------------------------------------------------------
    @property
    def threshold(self) -> float:
        """max(0, ghost panel max): the detection bar a channel must clear."""
        return max(0.0, float(self.ghosts.max()))

    @property
    def driven(self) -> np.ndarray:
        return self.excess > self.threshold

    @property
    def saturated(self) -> bool:
        return not self.donor_fallback

    @property
    def stationary(self) -> bool:
        """Ghost median clearly above zero flags non-stationarity."""
        return float(np.median(self.ghosts)) <= 0.005

    def gate(self) -> dict:
        return {
            "G1_length": {"n": self.n, "validated_floor": 2000,
                          "verdict": "ok" if self.n >= 2000 else "MARGINAL"},
            "G2_saturation": {
                "self_r2_median": float(np.median(self.self_r2)),
                "self_r2_max": float(self.self_r2.max()),
                "frac_above_donor": float((self.self_r2
                                           > self.config.donor_r2).mean()),
                "verdict": ("theory licensed" if self.saturated else
                            "EMPIRICAL ONLY: ghost is a noise scale, not a "
                            "calibrated null")},
            "G3_stationarity": {
                "ghost_median": float(np.median(self.ghosts)),
                "ghost_max": float(self.ghosts.max()),
                "frac_positive": float((self.ghosts > 0).mean()),
                "verdict": ("ok" if self.stationary else
                            "FAIL: discard this segment, do not threshold")},
        }

    def summary(self) -> str:
        g = self.gate()
        lines = [
            f"MACE scan: V={self.V} channels, n={self.n} samples",
            f"  G1 length      : n={self.n} "
            f"({g['G1_length']['verdict']})",
            f"  G2 saturation  : self-R2 med "
            f"{g['G2_saturation']['self_r2_median']:.3f} max "
            f"{g['G2_saturation']['self_r2_max']:.3f} "
            f"-> {g['G2_saturation']['verdict']}",
            f"  G3 ghost panel : median "
            f"{g['G3_stationarity']['ghost_median']:+.4f} max "
            f"{g['G3_stationarity']['ghost_max']:+.4f} "
            f"-> {g['G3_stationarity']['verdict']}",
            f"  threshold      : {self.threshold:+.4f}"
            f"  driven: {int(self.driven.sum())}/{self.V}",
            "  reminders      : sources are invisible by design; absence is",
            "                   NOT evidence a channel is autonomous.",
        ]
        order = np.argsort(-self.excess)[:10]
        names = (self.channel_names if self.channel_names is not None
                 else [str(i) for i in range(self.V)])
        lines.append("  top channels   : " + ", ".join(
            f"{names[i]}({self.excess[i]:+.3f})" for i in order))
        return "\n".join(lines)

    def to_frame(self):
        import pandas as pd
        names = (self.channel_names if self.channel_names is not None
                 else list(range(self.V)))
        return pd.DataFrame({
            "channel": names, "excess": self.excess,
            "self_r2": self.self_r2, "driven": self.driven,
        }).sort_values("excess", ascending=False)


# ---------------------------------------------------------------------------


def _poly(own: np.ndarray, degree: int) -> np.ndarray:
    cols = [own]
    e = own.shape[1]
    for i in range(e):
        for j in range(i, e):
            cols.append((own[:, i] * own[:, j])[:, None])
    if degree >= 3:
        for i in range(e):
            for j in range(i, e):
                for k in range(j, e):
                    cols.append((own[:, i] * own[:, j] * own[:, k])[:, None])
    return np.hstack(cols)


class _MaskedAE(torch.nn.Module if torch else object):
    def __init__(self, d_in: int, b: int):
        super().__init__()
        h = max(2 * b, 64)
        self.enc = torch.nn.Sequential(
            torch.nn.Linear(d_in, h), torch.nn.Tanh(), torch.nn.Linear(h, b))
        self.dec = torch.nn.Sequential(
            torch.nn.Linear(b, h), torch.nn.Tanh(), torch.nn.Linear(h, d_in))

    def forward(self, z):
        return self.dec(self.enc(z))


def scan(X: np.ndarray, config: MaceConfig | None = None,
         channel_names: Optional[list] = None) -> ScanResult:
    """Run the full MACE pipeline, controls included, on (n, V) data."""
    if torch is None:                                        # pragma: no cover
        raise ImportError(
            "mace requires torch; install with pip install -e .[mace]"
        ) from _TORCH_ERR
    cfg = config or MaceConfig()
    dev = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < X.shape[1]:
        raise ValueError("X must be (timepoints, channels) with more "
                         "timepoints than channels")
    if cfg.difference:
        X = np.diff(X, axis=0)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = X[:, X.std(0) > 1e-12]
    T, V = X.shape

    # delay embedding, channel-major blocks
    span = (cfg.e - 1) * cfg.tau
    n = T - span
    emb = np.concatenate(
        [np.stack([X[span - k * cfg.tau: span - k * cfg.tau + n, j]
                   for k in range(cfg.e)], axis=1) for j in range(V)], axis=1)

    a = int(cfg.train_frac * n)
    b_end = int((cfg.train_frac + cfg.val_frac) * n)
    tr = slice(0, a - span)
    tr_i = np.arange(tr.start, tr.stop - 1)
    te_i = np.arange(b_end, n - 1)
    mu, sd = emb[tr].mean(0), emb[tr].std(0) + 1e-12
    zs = np.clip(np.nan_to_num((emb - mu) / sd),
                 -cfg.zclip, cfg.zclip).astype(np.float32)
    lead = zs[:, [j * cfg.e for j in range(V)]]

    def ridge(Xtr, ytr, Xte, yte) -> float:
        Xt = torch.as_tensor(Xtr, dtype=torch.float64, device=dev)
        yt = torch.as_tensor(ytr, dtype=torch.float64, device=dev)
        Xe = torch.as_tensor(Xte, dtype=torch.float64, device=dev)
        ye = torch.as_tensor(yte, dtype=torch.float64, device=dev)
        A = Xt.T @ Xt + cfg.alpha * torch.eye(Xt.shape[1], device=dev,
                                              dtype=torch.float64)
        w = torch.linalg.solve(A, Xt.T @ yt)
        err = float(((Xe @ w - ye) ** 2).mean())
        return max(0.0, 1.0 - err / (float(ye.var()) + 1e-12))

    feats = [_poly(zs[:, q * cfg.e:(q + 1) * cfg.e], cfg.degree)
             for q in range(V)]
    self_r2 = np.array([ridge(f[tr_i], lead[tr_i + 1, q],
                              f[te_i], lead[te_i + 1, q])
                        for q, f in enumerate(feats)])

    # ---- encoder ensemble -------------------------------------------------
    ztr = torch.as_tensor(zs[tr], device=dev)
    zfull = torch.as_tensor(zs, device=dev)
    codes = []
    for m in range(cfg.models):
        torch.manual_seed(cfg.seed + m)
        net = _MaskedAE(zs.shape[1], cfg.bottleneck).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        g = torch.Generator().manual_seed(cfg.seed + m)
        for _ in range(cfg.epochs):
            perm = torch.randperm(ztr.shape[0], generator=g)
            for i in range(0, len(perm), cfg.batch):
                bt = ztr[perm[i:i + cfg.batch]]
                msk = torch.rand(bt.shape[0], V, device=dev) < cfg.mask
                mc = msk.repeat_interleave(cfg.e, dim=1)
                out = net(bt.masked_fill(mc, 0.0))
                loss = ((out - bt)[mc] ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            codes.append(net.enc(zfull).cpu().numpy())

    def excess_of(f: np.ndarray, target: np.ndarray) -> float:
        base = ridge(f[tr_i], target[tr_i + 1], f[te_i], target[te_i + 1])
        return float(np.mean(
            [ridge(np.hstack([f[tr_i], c[tr_i]]), target[tr_i + 1],
                   np.hstack([f[te_i], c[te_i]]), target[te_i + 1]) - base
             for c in codes]))

    excess = np.array([excess_of(feats[q], lead[:, q]) for q in range(V)])

    # ---- ghost panel with the donor rule ----------------------------------
    rng = np.random.default_rng(cfg.seed + 4242)
    qual = np.where(self_r2 > cfg.donor_r2)[0]
    fallback = len(qual) < cfg.min_donors
    pool = np.arange(V) if fallback else qual
    donors = rng.choice(pool, size=min(cfg.ghosts, len(pool)), replace=False)
    ghosts = []
    for d in donors:
        s = int(rng.integers(n // 4, 3 * n // 4))
        gz = np.roll(zs[:, d * cfg.e:(d + 1) * cfg.e], s, axis=0)
        ghosts.append(excess_of(_poly(gz, cfg.degree),
                                np.roll(lead[:, d], s)))

    return ScanResult(excess=excess, self_r2=self_r2,
                      ghosts=np.array(ghosts), n=n, V=V,
                      donor_fallback=bool(fallback), config=cfg,
                      channel_names=channel_names)
