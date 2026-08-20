# DeepFeatSelection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21988145.svg)](https://doi.org/10.5281/zenodo.21988145)

Feature selection by putting a learnable, non-negative gate in front of every
input feature, training a network end to end, and reading the gates back as
importances.

Originally written in 2021 against TensorFlow 2.4. This version brings it up to
TensorFlow 2.20 / Keras 3 and fixes the parts of the method that made the
importances hard to defend. The 2021 code is preserved unchanged in
[`legacy/`](legacy/).

**Looking for the findings?** [FINDINGS.md](FINDINGS.md) consolidates every
result in this repository — including the negative ones — in plain language.

**Looking for the causal-detection paper?** The repository also holds a second,
independent line of work on identifying which variables of a large dynamical
system are driven by the system. See
[MACE](#mace-detecting-driven-variables-in-large-dynamical-systems) below, or go
straight to the preprint: [`paper/excess_paper.pdf`](paper/excess_paper.pdf).

---

## MACE: detecting driven variables in large dynamical systems

This repository also contains a second, self-contained line of work. **MACE**
(Masked-Autoencoder Conditional Excess) answers a different question from the
feature-selection tool above: given many simultaneously recorded time series,
**which variables are driven by the rest of the system, and which evolve
autonomously?**

📄 **Preprint: [`paper/excess_paper.pdf`](paper/excess_paper.pdf)**

The statistic is the gain in one-step predictability of a variable when a
learned low-dimensional code of the entire remaining system is added to a
flexible model of that variable's own history. Because the code is shared
across variables and each readout is a ridge regression, a complete scan of
71,721 variables runs in minutes on a consumer laptop, against roughly 800,000
GPU-hours for the pairwise equivalent. Every scan embeds a **ghost channel**, a
circularly shifted copy of real data connected to nothing by construction, so
each analysis carries its own falsification test and needs no ground truth.

**Limits, stated up front.** MACE measures drivenness only, so pure sources are
invisible by design. It reaches precision 0.95 at 28% recall: it finds the
strongly driven variables and stays quiet about the rest, so **a variable's
absence from the result is not evidence that it is autonomous.** Validity is
established down to n ≈ 2,000 samples.

### Run it on your own data

```bash
pip install -e .[mace]
```

```python
import numpy as np, mace
X = np.load("recording.npy")      # (timepoints, channels), n >= 2000
result = mace.scan(X)
print(result.summary())             # gate verdicts, ghost panel, top channels
result.to_frame().to_csv("drivenness.csv")
```

Or from the shell: `mace-scan data.csv --out drivenness.csv`.

The scan refuses to be quiet about its own validity: every result carries the
three-gate report (length, saturation, stationarity), the donor-filtered ghost
panel that sets the detection threshold, and the standing reminders — sources
are invisible by design, and absence from the result is not evidence a channel
is autonomous. If the ghost panel says the segment is non-stationary, believe
it and discard the segment; that rule is written from experience recorded in
the ledger.

### Reproducing the results

Every number in the paper is derived from the scripts below and re-checked by
`scripts/audit_paper_numbers.py`, which asserts each published figure against
the files in `ExpOutput/`.

| script | what it produces |
|---|---|
| `scripts/bottleneck_membership.py` | the synthetic coupled-map systems and the masked autoencoder |
| `scripts/excess_membership.py` | the core statistic; top-k precision at V = 1001 |
| `scripts/recall_analysis.py` | the precision/recall operating curve |
| `scripts/ghost_calibration.py`, `ghost_tail.py`, `ghost_corrected.py` | the ghost panel, the donor requirement, and the calibrated threshold |
| `scripts/readout_class_gap.py` | the gap between the estimator and the quantity it identifies |
| `scripts/celegans_excess.py` | *C. elegans* wild-type and AVA-silenced scans |
| `scripts/intervention_null.py` | the three per-cell nulls for the intervention |
| `scripts/zapbench_feasibility.py` | the 71,721-neuron zebrafish scan |
| `scripts/eeg_excess.py`, `eeg_concentration_null.py` | clinical EEG, and the clamp-free re-analysis |
| `scripts/climate_excess.py` | 77 years of sea-level pressure |
| `scripts/paper_figures.py`, `fig_recall.py` | all figures, from primary data |
| `scripts/audit_paper_numbers.py` | re-derives every published number and checks it appears in the PDF |

### Protocol and negative results

- [`paper/causal_detection_log.md`](paper/causal_detection_log.md) — the full
  experiment ledger, including every negative result, every voided finding, and
  the standing protocol rules each failure produced.
- [`paper/validation_protocol.md`](paper/validation_protocol.md) — the
  pre-registrations, fixed before the corresponding data was opened.

Findings that did not survive scrutiny are marked VOID in the ledger rather
than removed, and superseded entries carry forward-pointers to whatever
overturned them.

Datasets are public and require no registration. They are not tracked here;
the scripts fetch them.

---

## Quick start

```bash
pip install -e .
deepfeatselect --n-models 20
```

Everything is a flag — no interactive prompt — so runs can be scripted and swept:

```bash
deepfeatselect --data Data/processed.cleveland.data \
               --task binary --n-models 50 --l1-gate 1e-2 \
               --outdir ExpOutput
```

Two CSVs land in `--outdir`: `runs.csv` (one row per trained model: seed,
held-out metrics, gate per feature) and `importance.csv` (the aggregated
ranking).

### GPU

TensorFlow dropped native Windows GPU support at 2.11, so on Windows the GPU
path is WSL2:

```bash
python3.12 -m venv ~/venvs/dfs
~/venvs/dfs/bin/pip install "tensorflow[and-cuda]>=2.17,<2.21"
```

The `[and-cuda]` extra keeps a version-matched CUDA runtime and cuDNN inside the
venv, so nothing is installed system-wide. Verify with:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Worth saying plainly: **on this dataset the GPU does not help.** 297 rows and a
3x128 MLP means roughly three steps per epoch, where kernel-launch overhead
dominates and the card idles. The cost here is training N models in sequence, so
`--workers` (multiple models at once) buys far more than the GPU does. The GPU
path exists for when the method is pointed at something larger.

---

## How it works

Each feature gets one non-negative scalar gate. Train, then read the gates:
larger gate means the network leaned on that feature more.

```
x ──▶ FeatureGate (one non-negative weight per feature, L1-penalised)
        │
        ▼
      Dense(128) ─▶ Dense(128) ─▶ Dense(128)   (ReLU, dropout, weight decay)
        │
        ▼
      sigmoid / softmax
```

The original built this as 13 separate `Dense(1, use_bias=False)` layers whose
outputs were concatenated. [`FeatureGate`](deepfeatselect/model.py) is the same
computation vectorised into one layer, which makes two things possible that were
awkward before: an L1 penalty on the gate vector, and one gate shared across all
one-hot columns of a categorical feature.

### Why the L1 penalty is the important part

In the original there was nothing pushing an unimportant gate toward zero. The
network could leave a gate large and zero the matching column of the first dense
layer instead — gate magnitude and downstream weights are freely
interchangeable, so the ranking was only weakly identified.

Getting from that observation to a penalty that actually selects took three
steps, and the two failed ones are more instructive than the fix.

**1. L1 in the loss selects nothing.** The obvious move is a `regularizers.L1` on
the gate vector. On this dataset it selected nothing at any strength — by
`l1=0.1` held-out AUC was already degrading while all 13 features still carried
near-identical gate mass.

The reason is arithmetic. 178 training rows at batch 64 is three steps an epoch,
and early stopping fires around epoch 60 — roughly 180 updates. At `lr=1e-3` a
gate can move about 0.18 in total, so one initialised at 1.0 cannot reach zero
however large the penalty. A loss-based L1 approaches zero asymptotically in any
case; it never lands there.

**2. So apply L1 proximally.** Soft-thresholding `max(w - lr·λ, 0)` after every
update is the standard treatment for L1 in deep networks, and it yields exact
zeros. Keras constraints run at precisely the right moment, so it is a nine-line
[`NonNegSoftThreshold`](deepfeatselect/model.py). Because a gate now travels
about `lr·λ` per step, useful `--l1-gate` values are far larger than L1 strengths
usually look — order 0.1–10, not 1e-4. It is a per-step shrinkage, not a loss
coefficient. `--no-proximal` restores the loss-based version.

**3. That shrank every gate equally, which is still not selection.** Gates fell
from 1.0 to roughly 0.24–0.32 and stalled — all thirteen, *uniformly*. This is
the interchangeability problem biting from the other side: as gates shrink, the
first layer grows to compensate, output is unchanged, and every gate feels the
same push-back. Weight decay at `1e-3` is far too cheap to prevent it.

So the first layer is tied to its gate directly.
[`HierarchyProjection`](deepfeatselect/model.py) projects after every batch so
that column `j` obeys `max|W[j,:]| ≤ M · gate[feature(j)]`. A gate of exactly zero
now forces that feature's weights to zero, making it genuinely unreachable rather
than merely quiet. This is the hierarchy constraint from
[LassoNet](https://jmlr.org/papers/v22/20-848.html), applied as an alternating
projection. `--no-hierarchy` turns it off.

`M` only binds when it is comparable to the first layer's weight scale. Glorot
initialisation here gives `max|W| ≈ 0.2`, so an initial `M=10` never engaged and
the projection was silently a no-op — worth knowing before tuning it. Default
`1.0`.

Each of these is a test rather than a claim:
[`test_l1_penalty_actually_sparsifies_the_gates`](tests/test_experiment.py) checks
the penalty concentrates gate mass on genuinely informative features,
[`test_proximal_gate_reaches_exact_zero`](tests/test_model.py) checks gates reach
exactly zero on pure noise, and
[`test_hierarchy_projection_zeroes_weights_under_a_closed_gate`](tests/test_model.py)
checks a closed gate really does disable its feature.

---

## What changed from 2021, and why

### Method

| | 2021 | now |
|---|---|---|
| Gate identifiability | no penalty; gate/first-layer weights interchangeable | proximal L1 on gates + hierarchy constraint tying first-layer weights to their gate |
| Validation split | `validation_split=0.5` | stratified 60/20/20 train/val/test |
| Scaling | L2-normalised over the whole dataset | `StandardScaler` fitted on train only |
| Categoricals | integer codes fed in as magnitudes | one-hot, with one shared gate per feature |
| Target | 5 independent sigmoids | softmax (multiclass) or single sigmoid (binary) |
| Reported result | importances alone | importances *and* held-out AUC/F1/balanced accuracy |
| Uncertainty | mean over runs | bootstrap CI, rank stability, selection frequency |

Four of these deserve a note:

**`validation_split=0.5` was not a random split.** Keras takes the *last*
fraction of the data *before* shuffling. On 297 rows of a file with structure in
its ordering, that is a skewed, non-random holdout. Splits are now stratified, so
every class is represented in each split.

**Scaling leaked.** L2 normalisation was computed across the full dataset,
validation rows included, before any split. The scaler is now fitted on the
training split alone.

**`cp`, `restecg`, `slope` and `thal` are nominal.** Feeding `thal ∈ {3,6,7}` in
as a number asserts that 6 sits midway between 3 and 7, which is meaningless.
They are one-hot encoded now. Because that turns one feature into several
columns, each feature's columns are scaled by `1/sqrt(group_size)` so a
four-level categorical does not collect four times the gate mass of a continuous
feature purely for being wider — the normalisation group-lasso applies to its
blocks.

**Five independent sigmoids let the model call a patient severity 1 and 4 at
once.** The levels are mutually exclusive, so multiclass now uses a softmax.

### Defaults

The default task is now `--task binary` (disease absent vs present) rather than
the original 5-class severity. The 5-class problem is still there via
`--task multiclass`, but the numbers show why it is not the default — see below.

### Code

- `input()` prompt replaced with `argparse`, so the experiment can be scripted.
- 156 lines of vendored TensorFlow `EarlyStopping` deleted; stock
  `EarlyStopping(restore_best_weights=True)` does the same job. It had been
  copied in only to bolt on model saving.
- `ModelSaverCallback` deleted. It never updated `prior_monitor_val`, so from
  epoch 1 it compared against epoch 0 forever, and its `if_max=True` branch was
  nested inside the `not if_max` branch and could never execute.
- Dead `monitor_values` list removed; it hardcoded five classes and was unused.
- `np.float`, `np.Inf`, `Adam(lr=)` and the private `tensorflow.python.keras`
  imports all updated — every one of them is a hard error on current versions.
- `requirements.txt` (which had `numpy=1.20.1`, a single `=`, and so could never
  install) replaced with `pyproject.toml`.
- Trained models, logs and result CSVs untracked and `.gitignore`d; the 2021
  outputs are kept in [`legacy/results/`](legacy/results/).
- Test suite added, since every claim above should be checkable.

---

## Results

20 models, `--task binary`, defaults as shipped. Each model draws its own
stratified split and its own initialisation, so the spread is sampling
variability, not just seed noise.

| metric | mean | std | min | max |
|---|---|---|---|---|
| test AUC | 0.899 | 0.036 | 0.830 | 0.973 |
| test F1 | 0.814 | 0.042 | 0.750 | 0.909 |
| balanced accuracy | 0.827 | 0.040 | 0.766 | 0.915 |
| accuracy | 0.829 | 0.041 | 0.767 | 0.917 |

That is in the range usually reported for Cleveland, so the gates are being read
off models that actually work.

Ranking, best first, with mean rank across the 20 runs:

| feature | importance | 95% CI | mean rank | rank std |
|---|---|---|---|---|
| `ca` | 0.0797 | 0.0789–0.0806 | 2.6 | 1.9 |
| `thal` | 0.0789 | 0.0785–0.0794 | 3.2 | 2.0 |
| `cp` | 0.0789 | 0.0778–0.0800 | 4.5 | 3.5 |
| `oldpeak` | 0.0777 | 0.0772–0.0782 | 5.5 | 2.4 |
| `sex` | 0.0777 | 0.0770–0.0785 | 6.5 | 3.2 |
| `exang` | 0.0774 | 0.0769–0.0779 | 5.5 | 2.8 |
| `thalach` | 0.0771 | 0.0765–0.0778 | 5.9 | 2.9 |
| `slope` | 0.0771 | 0.0765–0.0778 | 6.8 | 2.4 |
| `restecg` | 0.0758 | 0.0752–0.0764 | 9.1 | 2.2 |
| `age` | 0.0752 | 0.0745–0.0758 | 10.0 | 2.5 |
| `fbs` | 0.0749 | 0.0741–0.0760 | 10.8 | 2.8 |
| `trestbps` | 0.0749 | 0.0741–0.0756 | 10.4 | 2.2 |
| `chol` | 0.0746 | 0.0736–0.0754 | 10.7 | 2.6 |

`ca`, `thal` and `cp` at the top is the standard finding on this dataset, and
`chol`, `trestbps` and `fbs` at the bottom is equally unsurprising. So the
ranking is picking up something real.

### What this does not do

Two limitations, stated plainly, because the table above flatters the method if
read alone.

**Under the default settings it ranks but does not select.** `selected_frac` is
1.0 for every feature — nothing is eliminated. The importances span 0.0746 to
0.0797, a 7% spread around the uniform value of 1/13 = 0.0769. Early stopping
fires around epoch 54, and roughly 300 update steps is not enough shrinkage for
the penalty to close any gate. To get actual selection, spend a fixed budget
instead of stopping early:

```bash
deepfeatselect --n-models 20 --epochs 120 --patience 1000000
```

At `--l1-gate 1.0` that eliminated 4 of 13 features and gave the best held-out
AUC seen in any configuration (0.907).

**The sparsity path is not monotone**, so `--l1-gate` cannot be tuned by the
usual reasoning that more penalty means fewer features:

| `--l1-gate` | features retained | test AUC |
|---|---|---|
| 0.0 | 13 | 0.896 |
| 1.0 | **9** | **0.907** |
| 2.0 | 13 | 0.901 |
| 3.0 | 13 | 0.903 |
| 5.0 | 13 | 0.899 |

The reason is a feedback loop. Push the penalty hard and the gates collapse
quickly; the hierarchy projection then clamps the first layer small, the
network's output shrinks, the loss gradient grows, and every gate is pushed back
up *together*. The equilibrium is uniform gates more or less regardless of
penalty strength. So a stronger penalty can select less than a weaker one.

Both of these come back to 178 training rows against a ~50k-parameter network.
The mechanism is sound — on synthetic data with a known answer it isolates the
informative features cleanly, which is what
[`test_l1_penalty_actually_sparsifies_the_gates`](tests/test_experiment.py)
checks — but this dataset is too small to support a stable sparsity path. Treat
the output as a ranking, and treat the ordering *within* the top group as
uncertain: rank standard deviations of 2–3 places on 13 features mean the top
cluster is identifiable while the exact order inside it is not.

The obvious next step, and the one that would settle whether any of this beats
simpler methods, is a baseline comparison against mutual information,
permutation importance and L1 logistic regression. That is not in the repo yet.

---

## Reproducing

```bash
pip install -e ".[dev]"
pytest -m "not slow"       # fast unit tests
pytest                     # includes the L1 sparsification check
python scripts/l1_sweep.py --n-models 5
```

## Docker

```bash
docker compose run --rm experiment --n-models 50    # GPU
docker compose run --rm cpu        --n-models 50    # CPU only
```

## Data

[UCI Heart Disease, Cleveland](https://archive.ics.uci.edu/ml/datasets/Heart+Disease).
303 rows, 13 attributes, 6 rows dropped for missing values, leaving 297. Any CSV
in the same layout works via `--data` and `--attributes`.

## Licence

Code (`scripts/`, `mace/`, `deepfeatselect/`, `tests/`) is licensed under
**Apache-2.0**; the paper, protocols, ledger and prose are **CC BY 4.0**,
matching the preprint as deposited. See [LICENSE](LICENSE),
[LICENSE-CC-BY-4.0](LICENSE-CC-BY-4.0) and [NOTICE](NOTICE).

Both are deliberately permissive so that clinical and academic institutions
can actually use this. What the work is *for*, and what would count as a
misuse of it, is stated separately and without binding force in
[ETHICS.md](ETHICS.md).
