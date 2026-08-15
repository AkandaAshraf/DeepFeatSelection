# Does the shape of a training curve recover causal role?

Experimental design, for review before running. *Draft v0.1.*

---

## 1. Question

Every importance measure tried in this project ranks a **child** of the target
first — LOCO, permutation, mutual information, random forest, SAGE, the gated
network, and the deprivation probe. On `nonlinear_scm`, `x_effect` takes rank 1
of 9 under all of them. That is the Markov-blanket ceiling: prediction cannot
separate a cause from an effect.

Every one of those measures reads a network at a **single point** — its final
loss, its converged weights, its endpoint importance. This design asks whether
the **trajectory** carries information the endpoint does not, and specifically
whether any property of the loss curve separates causes from effects.

## 2. Where the idea comes from

Simplex projection does not judge an embedding by peak prediction skill. It
judges it by whether skill **decays smoothly** as the prediction horizon grows:
smooth decay means the attractor has been unfolded correctly, erratic or flat
decay means the embedding is wrong, whatever the peak number says. The *shape*
of the curve diagnoses the setup; the best single value does not.

The analogue here: a training run's final loss says how well the network ended
up fitting. The curve says how it got there — and two networks can arrive at the
same loss by very different routes.

## 3. Hypothesis

**H-timing.** A network deprived of a genuine **cause** must re-derive that
cause's contribution from the remaining inputs before it can fit at all. That is
work, and it should appear as a *later* inflection — a longer initial plateau,
a delayed steepest descent. A network deprived of an **effect** loses a cheap
shortcut but retains direct access to the causes, so it should begin descending
sooner even if it ends in a similar place.

If so, **timing statistics separate cause from effect where final loss does
not**, and the separation is mechanistic rather than incidental.

Committed predictions, before running:

* **P1.** `final_loss` does *not* separate cause from effect (AUROC near 0.5, or
  inverted). Consistent with every result so far.
* **P2.** At least one timing statistic (`t_half`, `epoch_of_max_slope`) beats
  `final_loss` on cause-vs-effect on both systems.
* **P3.** Most statistics separate cause from *noise*. This is a sanity check,
  not a finding; failure here means the setup is broken.
* **P4.** If P2 fails on both systems, the loss trajectory carries no causal
  information beyond its endpoint, and this line of attack closes. That is a
  clean negative and worth having.
* **P5.** The strong form, and the one that would matter: some statistic drives
  **P(effect ranks first)** below its null on both systems. Seven methods have
  scored 1.0 on this; anything below the null is a result, and only this
  prediction corresponds to a practical improvement rather than a better AUROC.

## 4. Testbeds

Two systems with known roles, chosen because they fail differently.

**`nonlinear_scm`** — 9 features. Causes `z`, `x_cause1`, `x_cause2`; effect
`x_effect` (a child of the target); confounded `x_conf1`, `x_conf2`; irrelevant
`x_noise1..3`. Leave-one-out is *not* zero here — `x_effect` scores 0.51 — so
this tests the Markov-blanket ceiling, i.e. orientation.

**`redundancy_demo`** — 4 features. Cause `driver`; effects `proxy_cos`,
`proxy_sin`; irrelevant `unrelated`. Leave-one-out is *exactly* zero for all
three informative features, so this tests whether the trajectory sees anything
at all where the endpoint is blind by theorem.

## 5. Protocol

* Fixed epoch budget, **no early stopping**. Arms must be measured over the same
  window or the curve shapes are not comparable — the confound that corrupted
  the first L1 sweep.
* Small network (16 units, 2 layers), **no dropout or activation noise**: both
  would smear the trajectory with their own randomness, which is the quantity
  being measured.
* Paired by seed. Every statistic is reported as `ablated − full` at the same
  seed, so initialisation is differenced out.

### 5.1 A sample-size problem, and the fix

`nonlinear_scm` has **3 causes and 1 effect**. An AUROC computed over 3 versus 1
takes only four possible values, which is not a measurement. Repeating with more
*training* seeds does not help, because the feature roles stay fixed and the
comparison is still 3 against 1.

The fix is to resample the **system**, not just the initialisation: draw S
independent instances of the SCM with different data seeds, run the full ablation
sweep on each, and pool. That yields 3·S causes against 1·S effects and makes the
AUROC meaningful. Recommended S = 8, giving 24 versus 8.

This must be settled before the run; the version of this experiment that varies
only the training seed would produce a number that looks like evidence and is not.

## 6. Candidate statistics

Fourteen summaries per curve, grouped by what they could capture. Several are
near-duplicates by construction — the aim is to find which *formulation*
discriminates, not to bet on one in advance.

| group | statistics |
|---|---|
| endpoint | `final_loss` (incumbent baseline), `total_drop`, `area` |
| rate | `slope_early`, `slope_mid`, `slope_late`, `slope_max` |
| timing | `epoch_of_max_slope`, `t_half`, `t_ninety` |
| shape | `monotone_frac`, `smoothness`, `curvature`, `plateau_frac` |

`smoothness` is total variation divided by total descent: ≈1 for a clean
monotone fall, ≫1 for an erratic one. `t_half` and `t_ninety` are the first
epochs reaching 50% and 90% of the eventual total descent, expressed as a
fraction of the budget.

## 7. Scoring

Two metrics, because they answer different questions and neither is sufficient.

### 7.1 Ranking quality: AUROC

Per statistic, AUROC at ranking one role above another:

1. **cause vs effect**, `nonlinear_scm` — the hard comparison
2. **cause vs irrelevant**, `nonlinear_scm` — sanity
3. **cause vs effect**, `redundancy_demo` — where LOCO is exactly zero

Direction is not assumed. A statistic that ranks causes reliably *below* effects
is equally informative, so the reported quantity is distance from chance,
`max(AUC, 1 − AUC)`, with the raw value retained.

**AUROC rather than PR-AUC**, because the class balance differs across the three
comparisons — 24:8, 24:24 and 8:16 at S = 8, so positive rates of 75%, 50% and
33%. PR-AUC's baseline *is* the positive rate, so its values would not be
comparable across comparisons and a difference in class balance would read as a
difference in performance. AUROC's baseline is 0.5 throughout. PR-AUC is reported
alongside for completeness but is not used for cross-comparison claims.

### 7.2 The metric that matches the failure: rank of the effect

AUROC scores the whole ordering, but the failure this project keeps hitting is
located precisely at **rank 1**: `x_effect` takes the top position under LOCO,
permutation, mutual information, random forest, SAGE, the gated network and the
deprivation probe — seven for seven. A statistic could reach AUROC 0.70 and still
put the effect first, which would be no practical improvement at all.

So the headline metric is

> **P(the effect ranks first)**, over the S system draws.

Its null is **not** 0.5 and must be stated per system:

| system | roles in the ranking | null P(effect first) | current methods |
|---|---|---|---|
| `nonlinear_scm` | 3 causes, 1 effect, 2 confounded, 3 irrelevant | **1/9 ≈ 0.11** for a uniformly random column; **1/4** if restricted to the four causally relevant roles | 1.0 |
| `redundancy_demo` | 1 cause, 2 effects, 1 irrelevant | **1/2** among informative features | 1.0 |

Reported as a binomial proportion over S draws with an exact interval.

A note on the tempting alternative: *"does a **cause** rank first?"* has a null of
**0.75** on `nonlinear_scm`, because three of the four causally relevant features
are causes and random ranking therefore looks strong. That metric is only
discriminating on `redundancy_demo`, where its null is 1/3. Any rank-based claim
must quote its null rather than assume 0.5.

### 7.3 Baseline

`final_loss` is the incumbent on both metrics. Anything that does not beat it on
comparison 1, or does not move P(effect first) below its null, has found nothing.

## 8. Threats

* **Multiple comparisons.** 14 statistics × 3 comparisons = 42 tests. Some will
  clear any threshold by chance. Treat a hit as real only if it replicates across
  *both* systems, and prefer a statistic with a mechanism behind it over the
  best number.
* **Correlated statistics.** The 14 are not independent; `t_half` and
  `epoch_of_max_slope` will move together. Report the correlation among them so
  a family of near-duplicates is not mistaken for repeated confirmation.
* **Confounding with usable information.** The deprivation probe already tracks
  how cheaply a feature can be read, and the effect is the cheapest feature here.
  Any statistic that separates cause from effect must be checked against that
  alternative reading — the reversal manipulation from the companion draft is
  the tool, since it changes cheapness while holding causal structure fixed.
* **Convergence.** If arms differ in how far training has progressed at the
  budget's end, timing statistics partly measure that. Reporting `slope_late`
  and `plateau_frac` alongside makes it visible rather than hidden.

## 9. Cost

With S = 8 system draws: `nonlinear_scm` is 10 arms × 8 draws, `redundancy_demo`
is 5 arms × 8 draws, at 150 epochs on a 16-unit network — roughly 120 small
trainings, on the order of 20–30 minutes on CPU.

## 10. Outcome either way

A positive result gives the first statistic in this project that separates cause
from effect, and a mechanism for why. A negative result closes off trajectory
analysis as a route to orientation, which is worth knowing before it is built
into anything larger.
