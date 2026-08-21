# A source-detection complement to MACE

Working note, 2026-08-20. Nothing here is a result. It is a derivation and a
set of predictions that a synthetic gate can falsify, written before that
gate is run.

## The problem

MACE detects **driven** variables. Its excess

```
    excess(q) = R²[ x_q(t+1) | φ(x_q lags) ⊕ z ] − R²[ x_q(t+1) | φ(x_q lags) ]
```

is zero in population for a variable whose own history determines its next
step. That is true of two very different things:

- an **isolated autonomous** variable, coupled to nothing, and
- a **pure source**, which drives others and receives from none.

Both score zero. MACE cannot tell them apart, and says so: *absence is not
autonomy*. The cost of that blindness is concrete. The seizure onset zone is
a source by clinical definition, so the entire iEEG study had to be framed
around SOZ **depletion** rather than detection. A complement that separates
sources from isolated variables would convert "we can say where it is not"
into "we can say where it is".

## What the dual has to look like

Excess asks: *what does the system add to the prediction of q?* The dual
asks: *what does q add to the prediction of the system?*

Write `z(t)` for the learned code of the system state. The naive dual is
leave-one-out degradation: retrain without q and measure how much worse
everything else is predicted. Two objections kill it.

**Objection 1 — cost.** It needs V codes rather than one. MACE's whole
claim is one shared code and V cheap readouts, giving linear cost; a
per-variable retrain restores the quadratic-ish cost that made the pairwise
methods infeasible. At V = 71,721 it is simply not available.

**Objection 2 — it is the failure mode this project already catalogued.**
Leave-one-out on a *trained* model is a difference-based importance score,
and Mechanism 1 of the companion paper is that such scores collapse to zero
under redundancy as the model matures and learns alternative routes. The
naive dual would reproduce, exactly, the pathology the method was built to
avoid. It must be rejected on the project's own evidence.

## The proposed statistic

Condition the *system's* future on its own past, and ask what q adds:

```
    outflow(q) = R²[ z₋q(t+1) | ψ(z₋q lags) ⊕ φ(x_q lags) ]
               − R²[ z₋q(t+1) | ψ(z₋q lags) ]
```

where `z₋q` is the code with channel q withheld. This is the mirror image of
excess: excess conditions on q's own history and adds the code; outflow
conditions on the code's own history and adds q.

**Why `z₋q` and not `z`.** This is the step where the first version of this
note was wrong. `z(t)` is an encoding of the whole system *including q*, so
`ψ(z lags)` already carries q's history, and the increment from adding
`φ(x_q lags)` would be near zero for every q, source or not. The statistic
would be vacuous — a control that cannot fail, in the ledger's terms.
Withholding q from the code is not a refinement; it is what makes the
quantity non-trivial.

**Why this stays linear in V.** `z₋q` is obtained by masking channel q at
the encoder input, not by retraining. The encoder is a *masked* autoencoder,
trained with 25% of channels masked at every step, so evaluating it with one
channel masked is in-distribution rather than extrapolation. The cost is V
forward passes plus V ridge solves of size (E·degree) × b. Linear, and the
same shape as MACE's readout stage.

## What it should do, stated before testing

The pair (excess, outflow) gives four quadrants:

| | low outflow | high outflow |
|---|---|---|
| **low excess** | isolated / autonomous | **source** |
| **high excess** | driven sink | hub (both) |

Predictions, to be checked on synthetic systems with known ground truth:

- **S1.** On a system with designated pure sources, sources score high
  outflow and near-zero excess; sinks score the reverse.
- **S2.** The ghost channel — a circularly shifted copy of real data — must
  score outflow ≈ 0, as it does for excess. If it does not, the statistic is
  reading typicality rather than influence (Mechanism 2) and is dead.
- **S3.** Outflow must **not** collapse as the autoencoder matures. Excess is
  stable across training because it is a conditional-information increment
  rather than a model-attribution difference; outflow is constructed the
  same way and should inherit that stability. If outflow decays with training
  epochs on a fixed system, it is a difference-based importance score in
  disguise and must be discarded (Mechanism 1).

## Limits, expected in advance

Redundancy will cost recall, symmetrically with MACE. If q is a source but
another channel carries the same information into the code, masking q alone
leaves that information intact and outflow(q) ≈ 0. The expected profile is
therefore the same as MACE's: **high precision, low recall, silence
uninformative**. A source detector that claimed high recall under redundancy
would be claiming something Takens forbids.

Masking one channel may also fail to remove its information when the encoder
has learned a redundant representation, which is the same limitation seen
from the other side.

## What would falsify this

Any of: the ghost scoring non-zero (S2); outflow decaying with training
epochs (S3); or sources and isolated variables not separating on a synthetic
system where the ground truth is designed (S1). Each is checked before the
statistic is applied to any real recording.

## If it survives

The immediate application is the iEEG cohort already scanned: the SOZ is a
source, so it should show **high outflow and low excess** — the opposite
corner from the driven core. That is a directional prediction on data whose
labels are already open, so it would be post-hoc and must be declared as
such; the honest use is as a hypothesis to be pre-registered on a held-out
cohort, not as a confirmation.

---

## Result of the gate (2026-08-20): the statistic FAILS S1

Run: `scripts/source_outflow_gate.py`, 15-variable system with 3 designed
sources, 6 sinks, 6 isolated variables, n = 4000.

**First implementation was wrong, and the ghost caught it.** The baseline
carried a single time point of the code while the added term carried E = 3
lags plus polynomial terms, so any channel improved the fit merely by
supplying temporal depth the baseline lacked. Source, isolated AND ghost all
scored ~ +0.026 - identical, which is what a statistic measuring lag
structure rather than influence looks like. The baseline now uses a delay
embedding of the code, matching the added term in depth.

**After the fix:**

| check | result | verdict |
|---|---|---|
| S2 ghost | outflow +0.0040 (was +0.0262) | **PASS** |
| S3 maturity | +0.0054 / +0.0051 / +0.0064 at 5 / 15 / 40 epochs | **PASS**, no decay |
| S1 separation | source +0.0060, isolated +0.0037, **sink +0.0062** | **FAIL** |

The ghost test and the maturity test pass, so the construction is not
reading typicality and is not a difference-based importance score in
disguise. Both of the companion paper's mechanisms are avoided.

**Why S1 fails, and why it is not a tuning problem.** Sinks score as high as
sources (+0.0062 vs +0.0060). A sink is driven by a source, so the sink's
history carries a proxy of the source's history, which predicts every other
sink's future. The statistic therefore cannot separate *influence* from
*shared-driver correlation*: it reports the confounded path as outflow. This
is precisely the mediated-false-positive failure the companion paper
attributes to PCMCI with partial correlation near synchrony, arrived at from
the opposite direction. Raising thresholds or extending training does not
touch it, because the confound is in the estimand, not the estimator.

**Status: rejected as formulated.** Not carried to any real data.

**What a next attempt would have to solve.** The baseline `psi(z_-q lags)`
is meant to screen off the common driver, and fails to because the code is
lossy: a 16-dimensional bottleneck does not fully capture the state, so
residual driver information leaks into q's increment. A viable version needs
either a baseline that provably screens off the common cause, or an estimand
defined on interventional rather than observational differences - and the
second is not available in recorded data, which is the whole difficulty.

The four-quadrant map (inflow x outflow) remains the right *shape* for an
answer. This particular outflow is not the right statistic to put on the
second axis.
