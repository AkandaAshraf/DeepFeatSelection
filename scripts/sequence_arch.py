"""Does more architecture rescue deprivation scoring at weak coupling?

``sequence_causal.py`` found that a bidirectional LSTM does not separate coupled
from uncoupled logistic maps until coupling reaches 0.08, where CCM already
separates cleanly at 0.01 -- the weakest strength tested -- in a fiftieth of the
time.  The obvious response is that the network needed more capacity.

The guard columns argue otherwise, and the argument is worth stating before
adding anything, because it predicts that capacity alone will make things worse.
The forecaster reached an R^2 of 0.998 to 1.000.  Its held-out error was 0.001,
and zeroing the driver's channel moved that to 0.002.  A deprivation score is a
*difference of two losses*, so when the autonomous term already explains
essentially everything, the quantity being differenced is a rounding error on
top of a rounding error.  A stronger model forecasts the autonomous term better
still, drives the baseline lower, and shrinks the very gap the score is built
from.  Capacity is not the binding constraint; the readout is.

So this varies both, and separates them:

**Architectures.**  ``bilstm`` is the incumbent, already bidirectional.
``bilstm_attn`` encodes each channel separately with a shared bidirectional LSTM
and then attends over channels.  ``mha_attn`` replaces the recurrent encoder
with multi-head self-attention over the window.  The last two are the "attention
based system"; the first is carried unchanged as the reference.

**Readouts.**  ``deprivation`` is the incumbent difference-of-losses.
``attention`` reads the weight the model puts on the source channel directly.
That second one is the point of the exercise: it is a quantity the model
computes internally rather than a difference between two near-identical numbers,
so it does not degrade as the baseline error approaches zero.  If the diagnosis
above is right, attention should hold at couplings where deprivation collapses,
and doing so under the *same* architecture is what distinguishes a readout
effect from a capacity effect.

**Horizon.**  One-step prediction of a near-deterministic map is dominated by
the autonomous term.  Over ``h`` steps the trajectories separate, error
accumulates, and the coupling's contribution grows relative to it.  Horizon is
therefore the third axis, and it tests the diagnosis rather than the
architecture.

Orientation is fixed in advance for both readouts: more coupling should mean a
larger deprivation delta and more attention mass on the source.  Nothing is
flipped to improve a score.

    python scripts/sequence_arch.py --couplings 0.0 0.01 0.02 0.04 --horizons 1 5
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import pandas as pd

from deepfeatselect.model import FeatureGate
from deepfeatselect.synthetic import coupled_logistic

# Fractions of the series used for fitting, for early stopping, and for the
# reported error.  Three disjoint contiguous segments: the early-stopping
# segment is selected on, so it cannot also serve as the held-out measurement.
TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2

# Below this held-out R^2 the forecaster did not beat the mean, and any score
# derived from it is noise regardless of how it was computed.
MIN_FORECAST_R2 = 0.05


@dataclass(frozen=True)
class Fit:
    """Held-out error with the variance it has to be judged against."""

    mse: float
    target_variance: float
    attention: np.ndarray | None = None
    gate: np.ndarray | None = None

    @property
    def r2(self) -> float:
        return 1.0 - self.mse / (self.target_variance + 1e-12)

    @property
    def learned(self) -> bool:
        return self.r2 >= MIN_FORECAST_R2


def windows(series: np.ndarray, lag: int, horizon: int
            ) -> tuple[np.ndarray, np.ndarray]:
    """Windows of length ``lag`` and the value ``horizon`` steps past their end.

    At ``horizon=1`` this is the one-step problem the incumbent solved almost
    perfectly.  Larger horizons are strictly harder for the autonomous term,
    which is the point: they give the coupling room to matter.
    """
    n = len(series)
    stop = n - lag - horizon + 1
    x = np.stack([series[i:i + lag] for i in range(stop)], axis=0)
    y = series[lag + horizon - 1:lag + horizon - 1 + stop]
    return x.astype("float32"), y.astype("float32")


def _channel_gate(inputs, channels: int):
    """Input-dependent attention over channels, applied as a gate.

    An earlier version pooled per-channel encodings by their attention weights.
    That is readable but it cannot forecast: pooling admits only weighted *sums*
    of per-channel features, so nothing downstream can combine two channels, and
    the measured consequence was a model whose error IMPROVED when the source
    channel was removed.  Gating instead leaves a full multivariate sequence for
    the encoder to mix, so the architecture can still fit the map while the
    weights stay directly readable.

    The weights are rescaled by the channel count so a uniform gate is the
    identity.  Without that the softmax shrinks every channel by ``1/channels``
    and the encoder has to undo a constant attenuation, which is a gradient
    problem rather than an attention one.
    """
    per_channel = keras.layers.Permute((2, 1))(inputs)
    scores = keras.layers.Dense(1)(per_channel)
    weights = keras.layers.Softmax(axis=1, name="channel_attention")(scores)
    gate = keras.layers.Permute((2, 1))(weights)
    scaled = keras.layers.Lambda(lambda t: t * float(channels),
                                 output_shape=lambda s: s)(gate)
    return keras.layers.Multiply()([inputs, scaled]), weights


def build_model(arch: str, lag: int, channels: int, units: int, dropout: float,
                gate_threshold: float = 0.0):
    """Return ``(model, attention_model, gate_layer)``; the last two may be None.

    All three see exactly the same input tensor and are trained identically, so
    a difference between them is a difference of architecture and not of
    protocol.
    """
    inputs = keras.layers.Input(shape=(lag, channels))
    gate_layer = None

    if arch == "bilstm":
        # The incumbent, unchanged. Already bidirectional; it mixes channels in
        # the first layer, so it has no per-channel quantity to report.
        h = keras.layers.Bidirectional(keras.layers.LSTM(units))(inputs)
        if dropout:
            h = keras.layers.Dropout(dropout)(h)
        h = keras.layers.Dense(units, activation="relu")(h)
        out, weights = keras.layers.Dense(1)(h), None

    elif arch == "bilstm_attn":
        # The incumbent's encoder, with a readable channel gate in front of it.
        # Identical to ``bilstm`` downstream, so any difference between the two
        # is attributable to the gate alone.
        gated, weights = _channel_gate(inputs, channels)
        h = keras.layers.Bidirectional(keras.layers.LSTM(units))(gated)
        if dropout:
            h = keras.layers.Dropout(dropout)(h)
        h = keras.layers.Dense(units, activation="relu")(h)
        out = keras.layers.Dense(1)(h)

    elif arch == "mha_attn":
        # Same gate, self-attention over the window in place of recurrence, so
        # the two attention arms differ only in their encoder.
        gated, weights = _channel_gate(inputs, channels)
        attended = keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=max(8, units // 4), dropout=dropout)(gated, gated)
        attended = keras.layers.LayerNormalization()(
            keras.layers.Add()([attended, gated]))
        # Flatten rather than average-pool. Mean pooling over the window
        # discards position, and the first version of this arm measured the
        # consequence: R^2 of 0.019 to 0.031, no better than the mean. Forecasting
        # a map needs the most recent state most, and averaging throws away which
        # state was most recent.
        h = keras.layers.Flatten()(attended)
        if dropout:
            h = keras.layers.Dropout(dropout)(h)
        h = keras.layers.Dense(units, activation="relu")(h)
        out = keras.layers.Dense(1)(h)

    elif arch == "bilstm_gate":
        # This project's own FeatureGate, applied to sequence channels rather
        # than tabular columns. It differs from the softmax gate in the way that
        # matters here: a softmax has no pressure to leave uniform, since uniform
        # is a perfectly good solution, and the smoke test found it sitting at
        # 0.498-0.502 in every condition. A proximally-penalised gate is pushed
        # toward zero and a channel survives only by paying for itself.
        gate_layer = FeatureGate(groups=np.arange(channels), l1=0.0,
                                 prox_threshold=gate_threshold)
        h = keras.layers.Bidirectional(keras.layers.LSTM(units))(gate_layer(inputs))
        if dropout:
            h = keras.layers.Dropout(dropout)(h)
        h = keras.layers.Dense(units, activation="relu")(h)
        out, weights = keras.layers.Dense(1)(h), None

    else:
        raise ValueError(f"unknown architecture {arch!r}")

    model = keras.Model(inputs, out)
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(3e-3))
    attention_model = keras.Model(inputs, weights) if weights is not None else None
    return model, attention_model, gate_layer


def fit_once(x, y, arch: str, seed: int, args) -> Fit:
    """Train on the first segment, early-stop on the second, report on the third.

    Windows overlap, so the splits are contiguous and each seam is embargoed by
    ``lag`` windows -- without that, windows straddling a boundary share
    timesteps, and their targets, with the segment before it.
    """
    keras.utils.set_random_seed(seed)
    lag, n = x.shape[1], len(x)
    train_end = int(TRAIN_FRACTION * n)
    val_end = int((TRAIN_FRACTION + VAL_FRACTION) * n)

    model, attention_model, gate_layer = build_model(
        arch, lag, x.shape[2], args.units, args.dropout, args.gate_threshold)
    stopper = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True)
    model.fit(
        x[:train_end - lag], y[:train_end - lag],
        validation_data=(x[train_end:val_end - lag], y[train_end:val_end - lag]),
        epochs=args.epochs, batch_size=64, shuffle=True, verbose=0,
        callbacks=[stopper])

    attention = None
    if attention_model is not None:
        # Measured on the same held-out segment as the error, so the two numbers
        # describe the same model on the same data.
        attention = np.asarray(
            attention_model.predict(x[val_end:], verbose=0)).reshape(len(x[val_end:]), -1)

    return Fit(mse=float(model.evaluate(x[val_end:], y[val_end:], verbose=0)),
               target_variance=float(np.var(y[val_end:])),
               attention=attention,
               gate=gate_layer.gate_values() if gate_layer is not None else None)


def score_pair(series: np.ndarray, arch: str, seed: int, horizon: int, args) -> dict:
    """Both readouts for predicting channel 1 (y) from a window of both channels.

    The base fit serves both: its loss anchors the deprivation difference and its
    attention weights are the second readout, so the comparison between readouts
    is on one model rather than two.
    """
    x, y_all = windows(series, args.lag, horizon)
    y = y_all[:, 1:2]

    t0 = time.time()
    base = fit_once(x, y, arch, seed, args)
    ablated_x = x.copy()
    ablated_x[:, :, 0] = 0.0
    ablated = fit_once(ablated_x, y, arch, seed, args)
    seconds = time.time() - t0

    deprivation = (ablated.mse - base.mse) / (base.mse + 1e-12)
    attention_on_source = (float(base.attention[:, 0].mean())
                           if base.attention is not None else np.nan)

    # Reported as a share rather than a level. The documented failure of an L1
    # gate without the hierarchy constraint is that every gate drifts DOWN
    # together while the layer above grows to compensate, which reorders nothing;
    # a share is invariant to that uniform shrinkage, so it still separates
    # "the source was dropped" from "everything shrank".
    gate_share = np.nan
    if base.gate is not None:
        total = float(base.gate.sum())
        gate_share = float(base.gate[0] / total) if total > 1e-9 else 0.0

    return {
        "deprivation": deprivation,
        "attention_on_source": attention_on_source,
        "base_mse": base.mse, "ablated_mse": ablated.mse,
        "base_r2": base.r2, "ablated_r2": ablated.r2,
        "gate_share_on_source": gate_share,
        "gate_source": float(base.gate[0]) if base.gate is not None else np.nan,
        "gate_target": float(base.gate[1]) if base.gate is not None else np.nan,
        "learned": base.learned,
        "seconds": seconds,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--lag", type=int, default=8)
    p.add_argument("--units", type=int, default=32)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--architectures", nargs="+",
                   default=["bilstm", "bilstm_attn", "mha_attn", "bilstm_gate"])
    p.add_argument("--couplings", type=float, nargs="+",
                   default=[0.0, 0.01, 0.02, 0.04],
                   help="0.0 is the control; 0.01-0.04 is where the incumbent lost.")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 5])
    p.add_argument("--gate-threshold", type=float, default=1.5e-3,
                   help="Per-step proximal shrinkage for the bilstm_gate arm. "
                        "This has to be calibrated against the STEP BUDGET, not "
                        "chosen for looking small: the gate starts at 1.0 and "
                        "shrinks by at most threshold*steps, so at 2e-4 over the "
                        "~1140 steps of a default run the most it can lose is "
                        "0.23 and a useless channel can never reach zero. The "
                        "measured symptom was a gate share of 0.462 against 0.463 "
                        "between control and coupled -- no separation, because "
                        "nothing could move far enough to separate.")
    p.add_argument("--outdir", default="ExpOutput/sequence_arch")
    args = p.parse_args()

    rows = []
    for arch in args.architectures:
        for horizon in args.horizons:
            for coupling in args.couplings:
                for seed in range(args.seeds):
                    system = coupled_logistic(n=args.n, coupling_x_to_y=coupling,
                                              seed=seed)
                    xs = np.asarray(system["x"], dtype=np.float64)
                    ys = np.asarray(system["y"], dtype=np.float64)
                    series = np.column_stack([
                        (xs - xs.mean()) / (xs.std() + 1e-12),
                        (ys - ys.mean()) / (ys.std() + 1e-12)])

                    row = score_pair(series, arch, seed, horizon, args)
                    row.update(arch=arch, horizon=horizon, coupling=coupling,
                               seed=seed, coupled=coupling > 0)
                    rows.append(row)
                    print(f"  {arch:<12} h={horizon} c={coupling:<5g} seed={seed}: "
                          f"depriv {row['deprivation']:+8.3f}  attn "
                          f"{row['attention_on_source']:.3f}  gate "
                          f"{row['gate_share_on_source']:.3f}  base_r2 "
                          f"{row['base_r2']:.4f}  ({row['seconds']:.0f}s)")

    frame = pd.DataFrame(rows)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outdir / "sequence_arch.csv", index=False)

    print("\n" + "=" * 104)
    print("DOES EITHER READOUT SEPARATE COUPLED FROM CONTROL AT WEAK COUPLING?")
    print("=" * 104)
    print("  Separation is the honest test used throughout this project: the")
    print("  WEAKEST coupled arm must exceed the STRONGEST control arm. A mean")
    print("  difference with overlapping ranges is not a detection.\n")

    for readout in ("deprivation", "attention_on_source", "gate_share_on_source"):
        print(f"\n  --- {readout} ---")
        for arch in args.architectures:
            for horizon in args.horizons:
                sub = frame[(frame.arch == arch) & (frame.horizon == horizon)]
                if sub[readout].isna().all():
                    print(f"    {arch:<12} h={horizon}: not available for this architecture")
                    continue
                control = sub[~sub.coupled][readout]
                if control.empty:
                    continue
                ceiling = control.max()
                verdict = []
                for c in sorted(sub[sub.coupled].coupling.unique()):
                    vals = sub[sub.coupling == c][readout]
                    verdict.append(f"c={c:g}:{'SEP' if vals.min() > ceiling else '---'}")
                learned = sub.learned.mean()
                print(f"    {arch:<12} h={horizon}: control_max {ceiling:+8.3f}   "
                      + "  ".join(verdict) + f"   (fitted {learned:.0%})")

    print("\n" + "=" * 104)
    print("WHY: THE DYNAMIC RANGE OF THE DEPRIVATION DIFFERENCE")
    print("=" * 104)
    with pd.option_context("display.float_format", "{:.5f}".format, "display.width", 200):
        print(frame.groupby(["arch", "horizon"])[
            ["base_mse", "ablated_mse", "base_r2"]].mean().to_string())
    print("\n  A base_mse near zero means the autonomous term explains the target")
    print("  on its own, so the difference the deprivation score is built from is")
    print("  taken between two numbers that are both nearly zero.")

    print(f"\nwrote {outdir}/sequence_arch.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
