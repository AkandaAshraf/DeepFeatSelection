# Pre-registration: source detection on a real physical system

Declared 2026-08-22, before any excess or outflow was computed on this data.
Column names and variance were inspected to identify which variables are
actuators and which are sensors; no statistic was run.

## Why this dataset

Three synthetic tests have now passed: the statistic is unconfounded at
sufficient code width, strong enough to use above coupling ~0.5, and not a
difference-based importance score in that regime. All three were coupled
logistic maps. This project's own record says that is where the real work
starts: the intracranial EEG result passed every synthetic and within-cohort
check and then failed a pre-registered replication.

The causal chamber (Gamella et al., Nature Machine Intelligence 2025) is a
built physical apparatus in which the experimenter SETS certain variables.
Ground truth is therefore structural rather than annotated: no expert opinion
is involved in knowing that a potentiometer setting is not caused by a
pressure reading.

Dataset: wt_walks_v1, wind tunnel, 28 random-walk runs of 1,016 samples.

## Ground truth, fixed by the apparatus

  SOURCES (set by the experimenter's random walk, driven by nothing in the
  system):        hatch, pot_1, pot_2

  DRIVEN (physical consequences):  load_in, load_out, current_in,
  current_out, rpm_in, rpm_out, pressure_upwind, pressure_downwind,
  pressure_ambient, pressure_intake, mic, signal_1, signal_2

V = 16. Constant columns and metadata (timestamp, counter, flag,
intervention, osr_*, v_*, res_*) are excluded.

Code width b = 32, following the b ~ 2V rule established this week.

## Two arms, because n is awkward

Each run is 1,016 samples, BELOW the method's validated floor of ~2,000.
Both arms are reported and must agree.

  PER-RUN       28 independent runs at n = 1,016, below the floor. Declared
                as such; treated as 28 replicates.
  CONCATENATED  all runs joined, n = 28,448, above the floor, with 27
                discontinuities. At E = 3 each boundary corrupts 2 samples,
                so 54 of 28,448 samples (0.2%) are affected. Declared, not
                repaired.

## Predictions, fixed now

  CH1  The three actuators show HIGH outflow and LOW inflow. Low inflow is
       the already-published behaviour - MACE is blind to sources - so the
       new content is the outflow.
  CH2  DECISIVE. Actuator outflow exceeds sensor outflow, and exceeds the
       ghost by at least 0.01, the bar used throughout this line.
  CH3  The ghost stays clean, panel median at or below 0.005.
  CH4  Both arms agree on the ORDERING of actuators against sensors, whatever
       the absolute values.
  CH5  NO PREDICTION on inflow for sensors. Whether MACE detects the driven
       variables here is a separate question and not what this tests.

## The risk that could defeat this for a legitimate reason

The chamber may respond faster than it is sampled. If a fan reaches its new
speed within one sampling interval, the system state at time t already
encodes the actuator at time t, the actuator's history adds nothing beyond
the system's own, and outflow vanishes. That is the synchrony failure the
boundary map found at strong coupling, arriving through sampling rate rather
than coupling strength.

If CH2 fails, this alternative must be distinguished from a genuine failure
of the statistic before the line is closed. The diagnostic declared now: if
the sensors' own history already predicts them almost perfectly (self-R2
above 0.95), the system is effectively synchronous at this sampling rate and
the test is uninformative rather than negative.

## The rule, fixed now

  PASSES        CH2, CH3 and CH4 all hold. The statistic detects known
                sources in a real physical system, and the next step is a
                second real dataset of a different kind.
  UNINFORMATIVE CH2 fails AND the sensors are near-synchronous by the
                diagnostic above. Reported as such; no claim either way.
  FAILS         CH2 fails and the sensors are NOT near-synchronous. The
                statistic works on synthetic data and not on a real system
                whose sources are known with certainty. The line closes, and
                that failure is reported as the headline.

## Void conditions

Void if the ground-truth assignment is changed after seeing a result, if
only one arm is reported, or if the synchrony diagnostic is invoked without
being computed.

---

## Result (2026-08-22): UNINFORMATIVE, by the rule declared in advance

### A protocol error to disclose first

This document states "28 random-walk runs of 1,016 samples". That is wrong.
It was written after inspecting only the first file. The runs are 1,016 to
320,000 samples, median 10,000, total 891,016: 27 of 28 are comfortably
ABOVE the validated floor, not below it. The two-arm design was therefore
unnecessary, though harmless. No prediction was affected; the premise
describing the data was simply inaccurate.

### The result is null

  PER-RUN (28 runs)     source outflow -0.0005   sensor outflow -0.0004
                        gap -0.0000, positive in 13 of 28 runs - chance
  CONCATENATED          source outflow -0.0002   sensor outflow +0.0002
                        gap -0.0004
  ghost                 -0.0007 and -0.0013, clean

CH2 fails: actuator outflow does not exceed sensor outflow and does not
clear the ghost by the 0.01 bar.

### The declared diagnostic fires

Sensor self-baseline R2, median over sensors:

  single run, n = 10,000        0.9930
  five runs concatenated        0.9980

Both far above the 0.95 threshold declared for near-synchrony. Per sensor on
the concatenated data:

  rpm_in 0.998   rpm_out 0.998   load_in 0.997   load_out 0.998
  pressure_upwind 0.998   pressure_downwind 0.999   pressure_ambient 0.999
  pressure_intake 0.999   current_in 0.898   current_out 0.896
  mic 0.003   signal_1 0.200   signal_2 0.226

Ten of thirteen sensors are essentially deterministic from their own history
at this sampling rate. The chamber settles faster than it is sampled, so the
system state at time t already encodes the actuator setting at time t and
the actuator's history adds nothing beyond it.

### Verdict

By the rule fixed before running: CH2 fails AND the sensors are
near-synchronous, so this is UNINFORMATIVE. NO CLAIM EITHER WAY. The
statistic has not been shown to work on real data, and it has not been shown
to fail. The test did not happen.

That is the least satisfying of the three declared outcomes and it is the
correct one. Had the diagnostic not been declared in advance, this null
would have been readable as either a failure of the statistic or an excuse
for one.

### What is NOT done here

Three sensors are not synchronous - mic (0.003), signal_1 (0.200) and
signal_2 (0.226). Restricting the analysis to those three after seeing which
ones failed the diagnostic is precisely the post-hoc subgroup hunt this
project refuses. It is not run. If it is worth doing it needs its own
pre-registration, and the honest version would pre-specify the
non-synchronous subset from the diagnostic BEFORE computing any outflow.

### What the next dataset needs

A real system sampled FAST relative to its own dynamics, so that a driven
variable's own history does not already determine its next value. The
chamber is a good apparatus and the wrong sampling rate. Higher-rate chamber
recordings, if they exist, would test the same structure properly.

---

## Amendment (2026-08-22, same day): the result above is VOID

An adversarial audit of the write-up immediately above found that its verdict
rests on a diagnostic that does not discriminate, applied to a dataset that
does not match this pre-registration. Every number in the Result section
stands as computed; what it was taken to mean does not. Diagnostics below
train no autoencoder and produce no outflow value.

### V1  The 0.95 synchrony threshold does not separate the two cases

The same diagnostic, same code path, run on the SYNTHETIC systems where
outflow demonstrably works:

  coupling   sink self-R2   outflow margin over ghost   diagnostic says
  0.05         0.9865              0.0003               near-synchronous
  0.35         0.9694              0.0019               near-synchronous
  0.50         0.9586              0.0107  CLEARS BAR   near-synchronous
  0.70         0.9348              0.0103  CLEARS BAR   not synchronous

At coupling 0.50 the statistic clears the 0.01 bar that REOPENED this line,
and the diagnostic calls that same system near-synchronous. A threshold that
fires where the method works cannot license the inference "near-synchronous,
therefore the test did not happen".

The rule was applied exactly as written, so nothing was chosen after the
fact. The rule itself was miscalibrated: because high self-R2 is generic in
smooth or densely sampled data, the UNINFORMATIVE branch absorbs nearly
every real dataset and the FAILS branch was close to unreachable. A
pre-registration whose decisive negative outcome cannot be reached is not a
pre-registration of anything.

### V2  Ground truth is invalid in 10 of the 28 runs

In all ten loads_hatch_mix runs, pot_1 and pot_2 are EXACTLY constant
(sd = 0). Two of the three declared sources are not sources there; they are
constants. Standardisation maps them to all-zero columns, so their outflow
is identically zero and the median over three sources is exactly zero by
construction. The saved output confirms it: src_outflow = +0.000000 in all
ten.

Those ten runs supply 8 of the 13 "positive" gaps. The headline "positive in
13 of 28 runs, chance" is arithmetic over a set in which ten runs cannot
produce anything else.

On the 18 runs where all three sources actually vary:

  gap median -0.00057, positive in 5 of 18

which is worse for the statistic than the number reported, not better. The
correction does not rescue anything; it removes an artefact that happened to
flatter the null.

### V3  The dataset is not the dataset this document describes

Not "28 random-walk runs". Three families:

  actuators_random_walk    16 runs   151,016 samples
  loads_hatch_mix          10 runs   100,000 samples   (pot_1, pot_2 constant)
  regime_jumps              2 runs   640,000 samples

The two regime-jump runs are 71.8% of the concatenated arm, which is one of
the two arms this protocol requires to agree. That arm is therefore mostly
an experiment family this pre-registration never mentions. The chronological
60/20/20 split also puts train and test in different regimes there: hatch
self-R2 is 0.000 on regime_jumps_single.

The earlier disclosure, that run lengths were misdescribed, understated this.
The composition was wrong, not just the lengths.

### V4  The mechanism given for the null is backwards

Actuator self-R2 was never reported. It is 0.9986, 0.9989, 0.9919 - as high
as the sensors'. The actuators are driven by nothing inside the chamber, so
their near-perfect self-prediction cannot be the system "already encoding"
them. It is dense sampling of a slowly moving random walk.

High self-R2 is therefore a signature of oversampling, and the Result
section's reading - "the chamber settles faster than it is sampled" - does
not follow. Its prescription, that the next dataset be sampled FAST relative
to its dynamics, is backwards: this one already is. What is needed is a
driver that moves enough per sample to leave novelty in the target, which is
a property of the excitation, not of the sampling rate alone.

### V5  Code does not implement the registered protocol

  CH3 is registered at ghost panel median <= 0.005; scripts/chamber_source.py
      tests <= 0.02, four times looser. The outcome is unaffected (ghost was
      0.0007 and 0.0013) but the committed gate is not the registered one.
  scripts/chamber_source.py:122 imports poly3, which does not exist in
      source_outflow_gate. The committed script CRASHES at the diagnostic;
      the reported self-R2 numbers came from a separate ad-hoc run using
      poly2. A committed artefact cannot reproduce a reported number.
  The per-sensor table is labelled "concatenated data" but the figure beside
      it says five runs concatenated - not the 28-run concatenation the
      verdict rule refers to.
  CH1 is never evaluated anywhere.

### V6  The configuration is below the capacity at which outflow works

The chamber ran b = 32 at V = 17 including the ghost, about 2V. Every
synthetic test that reopened this line - the capacity retry, the coupling
sweep, the maturity check - ran b = 64 at V = 16, about 4V. The real-data
test used half the relative code width of the tests that established the
statistic. The bottleneck study showed capacity is the binding constraint on
detection, so this null is confounded with under-capacity and cannot be read
as a property of the data.

### Corrected status

VOID, not uninformative. The dataset does not satisfy this document's own
description of it, the declared ground truth fails in ten of twenty-eight
runs, and the analysis ran below the capacity at which the statistic is known
to work. No verdict of any kind is supported, and "uninformative" is itself a
claim - that physics excused the method - which the evidence does not carry.

What a valid test needs, fixed now so it cannot be chosen later: one
experiment family, every declared source actually varying in every run,
b = 4V matching the tests that reopened the line, a split that does not cross
a regime boundary, and a fitness-for-purpose check on the DATA - is there any
lagged driver-to-target information at this sampling rate at all - decided
BEFORE the statistic is run and reported whatever it says.
