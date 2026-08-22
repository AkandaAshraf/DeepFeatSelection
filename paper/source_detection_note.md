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

---

## What the literature says (deep-research pass, 2026-08-20)

A five-angle survey with adversarial verification of each claim (100 agents
completed, 3 verification agents lost to connection errors). Two findings
change how the rejection above should be read.

### 1. Nobody has linear cost in V, and the one apparent exception is padding

Across every method whose primary source survived verification -- Amortized
Causal Discovery, neural Granger causality (cMLP/cLSTM/cRNN), PCMCI/PCMCI+,
oCSE, Large Causal Models -- **none achieves sub-quadratic cost in the number
of variables.** ACD's encoder propagates over a fully connected graph
emitting a directed latent per ordered pair, Theta(V^2) per forward pass;
neural Granger fits p networks each over all p pasts; PCMCI computes
N^2*tau_max p-values by construction; oCSE loops over every remaining node
per added parent per target.

Critically, **"amortized" in this literature indexes over samples, not over
variables.** ACD trains one model that infers graphs for previously unseen
samples without refitting -- the saving is per-dataset optimisation, not
per-pair enumeration.

The single reported case of runtime "independent of input dimensionality"
(Large Causal Models) is constant only because inputs are padded to a frozen
Vmax = 12, with a head whose dominant term is Theta(Vmax^2 * l_max).

So the niche MACE occupies -- a per-variable score at linear cost -- is
genuinely unoccupied, and the survey also notes that **none of these methods
produces a per-variable source score at all**: source status has to be read
off row/column asymmetries of an inferred edge tensor.

### 2. The common-driver problem is the field's problem, not just ours

ACD, neural Granger causality, LCM and PCMCI/PCMCI+ **all assume causal
sufficiency**. Where the driver is unobserved, a sink carrying a lagged proxy
of it is reported as a link and -- for lagged pairs, where time order forces
orientation -- as a directed source->sink edge. That is exactly the error our
outflow made. oCSE's proved no-false-positive property is likewise
conditional on a Markov assumption over the *observed* node set.

Only LPCMCI, in the FCI family, has an output space able to represent an
unobserved common driver: the PAG's bidirected edge. It screens by marking
X<->Y rather than by resolving a unique source -- that is, the honest answer
it can give is "these two share something unmeasured", not "this one is the
source".

### 3. Why our dual failed where PCMCI would not have

This is the part worth keeping. In our synthetic system the driver **was
observed** -- the sources were channels in the system. PCMCI conditions on
the actual candidate drivers and would have screened the sink correctly.
Our outflow conditioned on the *code*, a 16-dimensional lossy compression,
and a bottleneck is by construction not a sufficient statistic for the state.
Residual driver information leaked into the sink's increment.

That exposes an asymmetry between the original statistic and its dual:

- **Excess works at linear cost** because the conditioning set is the
  variable's OWN history, which is exact and complete. The code enters only
  as an additive predictor, and needs to be a useful summary, not a
  sufficient statistic.
- **Outflow needs the code to BE the conditioning set**, because screening
  off the common driver is the whole job. A lossy bottleneck cannot do that.

**Linear cost and common-driver screening are therefore in tension, and the
tension is structural rather than an implementation defect.** Screening
requires conditioning on the candidate drivers themselves, which is what
forces O(V^2); amortising through a shared compressed code is precisely what
gives up the completeness screening requires.

That is the barrier a second attempt has to address, and it is not a small
one. It is also a defensible reason for MACE to remain a drivenness detector
and to keep saying so plainly, rather than reaching for a source claim it
cannot support.

---

## Retry at sufficient capacity (declared 2026-08-22, before running)

The bottleneck experiment changed the premise this rejection rested on.

The rejection above diagnosed the cause precisely: outflow conditions on the
CODE, and a 16-dimensional code is not a sufficient statistic for the state,
so residual driver information leaks into a sink's increment and the sink
scores like a source. The literature pass then argued the barrier was
structural - linear cost and confounder screening in tension, because
amortising through a bottleneck surrenders the completeness screening needs.

Both arguments assumed a TIGHT bottleneck. The width experiment
(paper/bottleneck_protocol.md) finds MACE needs b of order 2V regardless -
a compression of only 1.5x against the 3V delay embedding. If the code must
be nearly as wide as the input anyway, it is close to sufficient, and the
tension may be much smaller than argued.

The original run used b = 16 at V = 15, where the new rule gives b ~ 32. It
was under-provisioned by about a factor of two, in precisely the way that
produces the failure observed.

### What is tested, fixed now

The identical gate (S1 separation, S2 ghost, S3 maturity) at b in
{16, 32, 64, 128}, three seeds, with b = 16 retained as the published
reference point.

  R1  Sources separate from ISOLATED channels at sufficient b. The original
      gap was +0.0023 and failed.
  R2  THE DECISIVE ONE. Sources separate from SINKS. The original failure
      was sinks (+0.0062) scoring as high as sources (+0.0060). If capacity
      was the cause, this gap opens as b grows. If it stays closed at
      b = 128, the confounding is structural as the literature suggests and
      the rejection stands permanently.
  R3  The ghost stays near zero at every width, as it did at b = 16 after
      the baseline-depth fix.
  R4  NO PREDICTION on whether any width makes the statistic usable.
      Separation appearing is necessary, not sufficient; it would still need
      the maturity check and a real-data test.

### The rule, fixed now

The rejection is lifted only if R2 holds AND R3 holds. A source-sink gap
that appears without the ghost staying clean is a capacity-enabled artefact,
not a working statistic.

If R2 fails at every width, this is recorded as CONFIRMED REJECTED: the
barrier is the confound, not the compression, and the deep-research
conclusion stands.
