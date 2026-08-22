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
