# Pre-registration: is a dataset fit to test a lagged-influence statistic?

Declared 2026-08-23, before the quantity below was computed on any data.

## Why this exists

The chamber source-detection run was voided because its escape hatch - a
self-R2 threshold of 0.95 declared in advance - fired on the synthetic system
at coupling 0.50, where outflow demonstrably works. The threshold was
declared but never calibrated, so invoking it was a claim rather than a
check (Rule 79).

This document is that rule applied to the thing the chamber run should have
tested first: whether a candidate dataset carries any lagged driver-to-target
information at its own sampling rate. If it does not, no statistic of this
family can succeed on it, and the dataset is disqualified on evidence rather
than on assumption. Rule 78 said to check; this specifies what the check is.

## The quantity

Linear, no autoencoder, no code, no outflow. For each target channel q:

  base     R2[ x_q(t+1) | poly2(own delay embedding of x_q) ]
  with_d   R2[ x_q(t+1) | poly2(own) (+) delay embedding of the DRIVERS ]
  lag_info(q) = with_d - base

reported as the median over targets. Same embedding (E = 3, TAU = 1), same
ridge (ALPHA = 1.0), same chronological 60/20/20 split, same code path as the
deployed gate. This is a property of the DATA. It says nothing about MACE.

GHOST CONTROL: the identical computation with the driver block circularly
shifted by n/3. It must sit at or below 0.002. A ghost above that means the
split or the embedding is leaking and the check itself is void.

## The threshold, calibrated rather than declared

The reference is the coupled logistic system used throughout this line, where
it is already established which couplings the statistic works at:

  coupling  0.05  0.15  0.30  outflow margin below the 0.01 bar - FAILS
  coupling  0.50  0.70              margin 0.0107, 0.0103 - CLEARS

lag_info is computed at all five couplings, three seeds each, giving the
value the method needs. Write L50 for the median at coupling 0.50, the
weakest system on which outflow is known to work, and L30 for coupling 0.30,
the strongest on which it is known to fail.

## The rule, fixed now

  QUALIFIES     chamber lag_info >= L50. The data carries at least as much
                lagged driver information as the weakest system where the
                statistic works. A null there would be about the method, so
                the clean re-run is worth its cost.
  DISQUALIFIED  chamber lag_info <= L30. The data carries no more lagged
                information than systems where the statistic is known to
                fail. No re-run. The chamber is excluded on evidence, and
                this check becomes the pre-flight gate for every future
                candidate dataset.
  AMBIGUOUS     strictly between L30 and L50. Reported as such, and treated
                as DISQUALIFIED for the purpose of spending an hour of
                compute, because a test that starts ambiguous cannot end
                decisive.

## Data

The 16 actuators_random_walk runs only: one experiment family, all three
declared sources vary in every run, no regime boundary inside any run
(Rules 80, 81). The loads_hatch_mix runs are excluded because pot_1 and
pot_2 are constant there, and the regime_jumps runs because a chronological
split crosses a regime boundary. Both exclusions are made now, on grounds
already documented, and not after seeing any result.

Drivers: hatch, pot_1, pot_2. Targets: the 13 sensors. Per run, then the
median across the 16 runs.

## Void conditions

Void if the threshold is moved after the chamber number is seen, if the
ghost control exceeds 0.002, if runs are added or dropped after a result, or
if the outcome is reported for only one of the two systems.

---

## Result (2026-08-23): the chamber is DISQUALIFIED on evidence

### Reference, computed first

  coupling   lag_info    ghost     outflow status (already established)
  0.05       +0.0007    -0.0000    fails
  0.15       +0.0044    -0.0001    fails
  0.30       +0.0114    -0.0001    fails          <- L30, the floor
  0.50       +0.0136    -0.0002    WORKS          <- L50, the pass mark
  0.70       +0.0189    -0.0001    WORKS

lag_info rises monotonically with coupling and the ghost sits at zero
throughout, so the quantity measures what it was built to measure. The two
couplings where outflow clears the 0.01 bar are the two highest values.

### Candidate

  16 actuators_random_walk runs, one family, all three sources varying

  lag_info median  -0.0000   range [-0.0020, +0.0000]
  ghost    median  -0.0000   range [-0.0007, +0.0000]

### Verdict

DISQUALIFIED. The chamber sits at zero - not merely below the L30 floor but
below coupling 0.05, the weakest system tested, by an order of magnitude.
The maximum over all 16 runs is +0.0000. At this sampling rate the actuators'
history adds nothing to any sensor beyond that sensor's own history.

There is no lagged influence in this dataset for any statistic of this family
to find. The chamber was never a test of the method, and it could not have
been. That was true before any autoencoder was trained.

### What this settles, and what it does not

It settles the chamber: no re-run at b = 4V, no re-registration, no further
compute. The exclusion now rests on a measured absence rather than on the
uncalibrated self-R2 threshold that voided the earlier attempt.

It also vindicates the earlier conclusion while confirming the audit's charge
against its reasoning. "The test did not happen" was right. The self-R2
diagnostic was not what showed it and could not have shown it, since that
same diagnostic fires at coupling 0.50 where the method works. Right answer,
wrong instrument, now with the right one.

LIMIT, stated rather than left implicit: lag_info is linear in a poly2
expansion of the delay embedding. A purely nonlinear lagged dependence
invisible to that expansion would not be detected. The check is not a proof
of absence. What licenses its use here is that the same expansion registers
+0.0114 to +0.0189 on the systems where the method works, so it is sensitive
at exactly the scale that matters, and the chamber returns zero against it.

### Standing use

This becomes the pre-flight gate for every future candidate dataset in this
line. A dataset is not selected until it has been run and has cleared L50,
and the number is reported whatever it says.

---

## Addendum (2026-08-23): decimation, and a corrected ground truth

Declared before any lag_info was computed on the new candidate datasets.

### A ground-truth error in the voided chamber run, disclosed

paper/chamber_source_protocol.md listed load_in and load_out as DRIVEN
sensors. They are settable fan loads, and they VARY in every family of
wt_walks_v1:

  actuators_rw     hatch 9.16  pot_1 28.92  pot_2 66.23  load_in 0.28  load_out 0.18
  loads_hatch_mix  hatch 22.50 pot_1  0.00  pot_2  0.00  load_in 0.35  load_out 0.50
  regime_jumps     hatch  0.00 pot_1 67.52  pot_2 78.47  load_in 0.29  load_out 0.29

So two of the five actuators were on the wrong side of the truth set, and
hatch is additionally constant in the regime_jumps runs - which is why its
self-R2 read 0.000 there. This is the fifth defect in that run and does not
change its VOID status. It is recorded because any new test on this apparatus
must not inherit it.

CORRECTED, and fixed now for every chamber dataset:

  SETTABLE (sources)  hatch, pot_1, pot_2, load_in, load_out
  MEASURED (driven)   current_in, current_out, rpm_in, rpm_out,
                      pressure_upwind, pressure_downwind, pressure_ambient,
                      pressure_intake, mic, signal_1, signal_2
  EXCLUDED            timestamp, config, counter, flag, intervention,
                      osr_*, v_*, res_*

Per dataset, a settable variable counts as a source only if it actually
varies there (Rule 80). Constants are excluded from the truth set, not
scored as zeros.

### Decimation, because sampling rate is a choice and not a fact

The wind tunnel is sampled far faster than its actuators move: actuator
lag-1 autocorrelation is 0.99 in every dataset inspected. That is the
oversampling Rule 78 warns about, and it is not a property of the apparatus
but of the rate at which its log is read. Taking every m-th sample makes the
actuator's motion fast relative to the sampling interval.

Choosing m to maximise a RESULT would be fitting. Choosing m so that the DATA
carries lagged influence at all is dataset qualification, and it is decided
by the gate, before any statistic is run, and reported in full.

  GRID          m in {1, 2, 5, 10, 20, 50}
  MINIMUM n     after decimation a run must retain at least 2,000 samples,
                the validated floor; shorter cells are dropped and reported
                as dropped.
  RULE          a dataset QUALIFIES at the SMALLEST m whose lag_info reaches
                L50 = +0.0136. If no m reaches it, the dataset is
                DISQUALIFIED.
  REPORTING     every candidate at every m, whatever the numbers say, with
                the ghost control beside each.

### Candidates, fixed now

  wt_walks_v1            already disqualified at m = 1; re-screened across the
                         grid with the corrected truth set
  wt_changepoints_v1     abrupt actuator steps
  wt_bernoulli_v1        randomised fan loads
  wt_intake_impulse_v1   impulse responses

Void if the grid or the candidate list changes after any number is seen, if a
dataset is selected at other than the smallest qualifying m, or if any
candidate screened is left unreported.

---

## Screen result (2026-08-23): two of four chamber datasets qualify

Reference recomputed with the identical code path: L30 = +0.0114,
L50 = +0.0136. Unchanged, as required.

  wt_walks_v1            m=1  +0.0002 | m=2 +0.0007 | m=5 +0.0018
                         m=10 +0.0158 QUALIFIES | m=20 +0.0162 | m=50 +0.0405
                         at m>=10 only 2 runs retain 2,000 samples: the two
                         320k-sample regime_jumps runs. 4 live sources there
                         (hatch is constant in them).
  wt_changepoints_v1     m=1 +0.0000, and no run retains 2,000 samples at any
                         higher m. DISQUALIFIED.
  wt_bernoulli_v1        +0.0001, +0.0000, +0.0000, +0.0001, +0.0004, +0.0015
                         across the grid. DISQUALIFIED. Despite the name, the
                         randomised loads are held for many samples: their
                         lag-1 autocorrelation is 0.99.
  wt_intake_impulse_v1   m=1 +0.0008 | m=2 +0.0015 | m=5 +0.0083
                         m=10 +0.0224 QUALIFIES | m=20 +0.0254 | m=50 +0.0000
                         5 runs of 250,000 samples, 25,000 after decimation.
                         2 live sources: hatch and load_in.

Ghost at or below +0.0011 in every cell of every dataset.

The earlier verdict on wt_walks_v1 stands as given: at the sampling rate its
logs are distributed in, it carries no lagged influence. Decimation is not a
rescue of that result but a different question - whether the APPARATUS can
carry lagged influence at some rate - and the answer for two of these
datasets is yes.

The m=50 collapse for the impulse dataset, from +0.0254 to +0.0000, is
consistent with the transient being aliased away entirely at that spacing.
No claim is made about it.

### A selection rule that was NOT declared

The script broke the tie between the two qualifying datasets by taking the
larger lag_info. The protocol declared how a dataset qualifies but not how to
choose among several that do, so that tie-break is an undeclared rule. It is
not used. BOTH qualifying datasets go forward, and both are reported.

---

## Addendum (2026-08-24): the light tunnel

Declared before any lag_info was computed on this data. Column names, run
lengths and per-column variance were inspected across ALL files of ALL five
datasets (Rules 81, 94); no statistic has been run.

### Why

The wind-tunnel result (marginal outflow AUC 0.916) rests on one apparatus
and one experimental protocol. The light tunnel is the second apparatus of
the same chamber family: physically unrelated dynamics (LEDs and polarisers
against fans and pressure), same structural ground truth. It is the declared
replication target, and the gate must qualify its data first.

### Ground truth, fixed now for every light-tunnel dataset

  SETTABLE (sources)  red, green, blue, pol_1, pol_2
  MEASURED (driven)   current, angle_1, angle_2, ir_1, vis_1, ir_2, vis_2,
                      ir_3, vis_3, l_11, l_12, l_21, l_22, l_31, l_32
  EXCLUDED            timestamp, config, counter, flag, intervention,
                      osr_*, v_*, diode_*, t_*, camera, v_board, v_reg
                      (sensor-configuration parameters and metadata, the
                      same class excluded for the wind tunnel)

pol_1/pol_2 are commanded polariser angles; angle_1/angle_2 are the measured
angles and are consequences. current is the LED current draw, a consequence
of red/green/blue.

Per run, a settable counts as a source only if it varies there (Rule 80).
NEW, symmetric rule declared now: a MEASURED column constant in a run is
excluded from the target set for that run - a structural zero is not
evidence about lagged influence. In the inspected data the six wall
photodiodes l_* are constant in two of the three walks runs.

### Candidates and their composition, from all files and all rows

  lt_walks_v1                    3 runs: actuators_white n=20,000 (all five
                                 sources vary), color_mix n=10,000 (red,
                                 green, blue only), smooth_polarizers
                                 n=10,000 (red, green, blue, pol_1)
  lt_interventions_standard_v1   59 runs: 58 of n=1,000 (below the 2,000
                                 floor at every decimation) plus
                                 uniform_reference n=10,000 (all five vary)
  lt_test_v1                     4 calibration runs, n=3,128-12,000, mostly
                                 sensor-config interventions
  lt_malus_v1                    12 runs of n=1,000 - expected to drop
                                 entirely at the floor
  lt_validate_v1                 29 runs of n=50-1,000 - expected to drop
                                 entirely at the floor

All five are screened and every cell is reported, including the expected
empty ones. The screen is scripts/lt_screen.py, which reuses the wind-tunnel
machinery unchanged except for the variable assignment above.

### Rule, unchanged

Same grid m in {1, 2, 5, 10, 20, 50}, same 2,000-sample floor, same
calibrated pass mark L50 recomputed by the identical code path, ghost beside
every cell, qualification at the smallest clearing m, every qualifying
dataset goes forward. If any dataset qualifies, the replication test gets
its own pre-registration BEFORE any outflow is computed, with the MARGINAL
statistic as primary this time - that is what the wind tunnel validated -
and the conditional variant reported beside it under its declared envelope
(fifteen measured consequences per five sources predicts conditioning loses
here too).

Void if the truth assignment changes after any result, if a candidate is
dropped from the report, or if the replication is run without its own
pre-registration.

### Light-tunnel screen result (2026-08-24): DISQUALIFIED at every decimation

Reference recomputed unchanged (L30 +0.0114, L50 +0.0136). Every cell:

  lt_walks_v1                    m=1/2/5: 3 runs, lag_info +0.0000 (max over
                                 runs +0.0010/+0.0013/+0.0025); m=10: 1 run,
                                 +0.0000; m>=20 below the floor
  lt_test_v1                     +0.0000 in every surviving cell
  lt_interventions_standard_v1   +0.0000 in every surviving cell (only the
                                 10,000-sample reference run survives)
  lt_malus_v1, lt_validate_v1    no run retains 2,000 samples at any m

Ghost at zero throughout. Best cell observed anywhere: +0.0025, a quarter of
the L30 floor.

THE PHYSICAL READING, and why decimation cannot rescue this apparatus where
it rescued the wind tunnel. The wind tunnel carries lagged influence because
rotors and air have inertia: a fan-load step takes real time to appear in
rpm and pressure, so there exists a sampling rate at which the actuator's
history adds information, and decimation can reach it. In the light tunnel
the actuator-to-sensor path is light: an LED setting appears in the
photodiodes within the same sample at ANY logging rate the apparatus can
achieve. Decimation changes the sampling of the dynamics; it cannot create
dynamics that the physics does not have.

The one plausibly lagged pathway - the polariser servos, where commanded
pol_1/pol_2 must reach measured angle_1/angle_2 through a motor - is two of
fifteen targets and cannot move the declared median. Restricting the gate to
those two targets after seeing this result would be the post-hoc subgroup
move this project refuses; if a servo-only test is ever worth running it
needs its own pre-registration naming angle_1/angle_2 as the only targets
BEFORE any lag_info is computed, and it would be a far narrower test (two
commanded angles against two measured ones) than the replication this screen
was looking for.

CONSEQUENCE: the wind-tunnel outflow result cannot be replicated on the
light tunnel. The replication needs a different physical system with
experimenter-set drivers AND internal dynamics slower than its sampling -
mechanical, thermal or chemical, not optical.

---

## Addendum (2026-08-24): the NIST UR5 robot arm

Declared before any lag_info was computed on this data. All 18 files parsed
and inspected for composition (Rules 81, 94); no statistic has been run.

### Why this dataset

The light tunnel fell to Rule 96: optical propagation is within-sample at
any logging rate. A robot arm is the mechanical case: commanded joint
trajectories reach measured positions through servo loops, link inertia and
motor electrical dynamics, and reach joint temperatures through thermal
mass. The NIST degradation dataset (Universal Robots UR5, NIST PHM
programme, public, no registration) logs TARGET and ACTUAL signals at
125 Hz while the arm repeatedly runs a preprogrammed trajectory with
randomly-selected stop points, under varied speed (half/full), payload
(1.6/4.5 lb) and thermal state (cold-start arm included).

### Ground truth, fixed now

  SOURCES   the six TARGET_JOINT_POSITIONS: the preprogrammed trajectory,
            exogenous to the physical plant by construction
  DRIVEN    ACTUAL_JOINT_POSITIONS (6), ACTUAL_JOINT_VELOCITIES (6),
            ACTUAL_JOINT_CURRENT (6), JOINT_CONTROL_CURRENT (6),
            CARTESIAN_COORD_TOOL (6), TCP_FORCE (6), JOINT_TEMP (6)
  EXCLUDED  TARGET velocities, accelerations, currents and torques. These
            are deterministic functions of the same preprogrammed
            trajectory (interpolator derivatives and the controller's
            feedforward model): not independent sources, not physical
            consequences. Declaring them sources would stack five copies of
            one exogenous signal; declaring them driven would put commanded
            quantities in the consequence set, the error Rule 80's chamber
            correction exists to prevent.

Per run, a source column counts only if it varies there (one target joint
is held constant in every run inspected); a driven column constant in a run
leaves the target set (the symmetric rule of the light-tunnel addendum).

### Composition, from all rows of all files

18 runs: half/full speed x 1.6/4.5 lb x 3 repetitions (12) plus cold-start
half/full x 4.5 lb x 3 (6). Row counts 6,487-10,440 (0.9-1.4 min each),
total 153,658 samples. Target-position lag-1 autocorrelation is 1.0000 in
every run - heavily oversampled, as the wind tunnel was, so the decimation
grid is where qualification will be decided. At m=5 most runs fall below
the 2,000-sample floor; the informative cells are expected at m=1 and m=2.
Joint temperatures span 23-37 C across runs.

### The risk named in advance

A UR5 position servo tracks within milliseconds. If the tracking loop
closes within one 8 ms sample, actual position may be synchronous with its
target - the light-tunnel failure arriving through control bandwidth rather
than optics. Currents, TCP force and temperatures have their own slower
dynamics, so the outcome is genuinely open; that is what the gate is for,
and either verdict is informative about the gate as well as the data.

### Rule, unchanged

Same grid m in {1, 2, 5, 10, 20, 50}, same 2,000-sample floor, same
calibrated L50 recomputed by the identical code path, ghost beside every
cell, every cell reported. Screen: scripts/nist_screen.py. If the dataset
qualifies, the replication test gets its own pre-registration BEFORE any
outflow is computed, marginal statistic primary, conditional reported
beside it under its declared envelope (about 40 driven channels against at
most five varying sources predicts conditioning loses).

Void if the truth assignment changes after any result, if any of the 18
runs is dropped from the report, or if the replication runs without its own
pre-registration.

### NIST UR5 screen result (2026-08-24): DISQUALIFIED

Reference unchanged (L30 +0.0114, L50 +0.0136). Every cell:

  m=1   18 runs   lag_info +0.0000  (max over runs +0.0001)  ghost -0.0000
  m=2   18 runs   lag_info +0.0001  (max +0.0002)            ghost -0.0000
  m=5    5 runs   lag_info +0.0003  (max +0.0003)            ghost -0.0001
  m>=10 no run retains 2,000 samples

Best cell +0.0003, forty times below the L50 pass mark and thirty below the
L30 floor.

The risk declared in the addendum fired, and with a second jaw. At 8 ms per
sample the UR5's position servo closes within one sample, so actual
position is synchronous with its target - the light-tunnel failure through
control bandwidth. The slower dynamics that might have carried lag at
deeper decimation (thermal mass, minutes-scale; trajectory-level inertia)
are unreachable: the runs are 0.9-1.4 minutes, so any m above 5 drops every
run below the 2,000-sample floor. The dataset is disqualified not because
the arm lacks lagged physics but because its recordings are too short to
decimate to the rate where that physics lives.

That is a third distinct disqualification mode, and it is checkable by
arithmetic BEFORE downloading anything: the run length must satisfy
n >= floor x (dynamics timescale / sampling interval). A candidate whose
slow dynamics live at seconds needs runs of at least floor x (seconds /
sample interval) samples. The UR5 runs fail that inequality for every
timescale slower than its servo.

CONSEQUENCE: three screens, three modes. The wind tunnel qualified
(dynamics present, decimation reached them). The light tunnel cannot carry
lag at any rate (physics). The UR5 arm plausibly carries lag but was not
recorded long enough to reach it (run length). The replication target must
have set drivers, dynamics slower than sampling, AND recordings long enough
to decimate into that regime.

---

## Addendum (2026-08-24): the NASA PCoE randomized battery usage data

Declared before any lag_info was computed on this data. All 28 cells of all
seven archives parsed and inspected for composition (Rules 81, 94); no
statistic has been run.

### Why this dataset

Rule 97 asks for set drivers, dynamics slower than sampling, and recordings
long enough to decimate into that regime. Battery cycling under a RANDOMIZED
load satisfies all three by construction: the applied current is redrawn at
random from a declared distribution every 60-300 s (exogenous excitation, the
same class as the wind tunnel's random walk), terminal voltage responds
through RC polarisation at seconds and through state of charge - an integral
of the whole current history - and cell temperature responds through thermal
mass at minutes. Recordings span 98-204 days per cell at 1 s within active
steps. Zenodo mirror of the NASA PCoE set, CC-open, no registration.

### Ground truth, fixed now

  SOURCE   applied current (set by the randomized protocol)
  DRIVEN   terminal voltage, cell temperature
  V = 3 per cell, plus the gate's shifted-driver ghost.

EXOGENEITY CAVEAT, declared rather than hidden: two regimes couple current
back to the cell state - the constant-voltage tail of each charge (charger
feedback) and the 3.2 V discharge cutoff that truncates a random-walk step
early. The random-walk selection itself is exogenous. Both regimes stay in
the series and are disclosed; excising them after inspection would be
selection.

### Base grid, declared

The native sampling is mixed-rate: 1 Hz inside active steps, 60 s in
charges and long rests, so contiguous 1 Hz stretches never reach the 2,000
floor. The base grid is DT = 60 s - the excitation's own timescale and the
sparse blocks' native rate - built by linear interpolation within recording
spans and SPLIT at voids longer than 300 s (the ~13 reference-cycle outages
per cell). Segments with at least 2,000 grid points enter; the decimation
grid m in {1, 2, 5, 10, 20, 50} then spans 1-50 minutes.

### Composition, from all rows of all files

  family (4 cells each)            span/cell   RW step   usable pts/cell
  Uniform charge+discharge          ~147 d      300 s      165k-168k
  Uniform discharge (room T)        ~98-157 d   300 s      137k-220k
  Uniform variable charge (room T)  ~154 d      300 s       96k-97k
  Skewed high 40C                   ~99 d        60 s      139k-140k
  Skewed high (room T)              ~201 d      5-60 s     254k-276k
  Skewed low 40C                    ~99 d        60 s      140k
  Skewed low (room T)               ~204 d      229k-281k (60 s steps)

Total ~4.7 million usable grid points over 28 cells. Two cells (RW18, RW19)
have median random-walk step durations of 13 s and 5 s - shorter than the
base grid, so their driver moves within one grid sample; declared here,
reported per family, not excluded.

### Rule, unchanged

Same calibrated pass mark L50 recomputed by the identical code path, ghost
beside every cell, every decimation reported. Screen:
scripts/battery_screen.py. If the dataset qualifies, the replication test
gets its own pre-registration BEFORE any outflow is computed - marginal
statistic primary (what the wind tunnel validated), conditional reported
beside it under its declared envelope, with the note that V = 3 makes this
the smallest panel the statistic has faced.

Void if the truth assignment changes after any result, if any of the 28
cells is dropped from the report, if the base grid or void bound moves after
any lag_info is seen, or if the replication runs without its own
pre-registration.

### Battery screen result (2026-08-24): DISQUALIFIED at the declared grid,
### with a disclosed design error and a licensed follow-up

Reference unchanged (L30 +0.0114, L50 +0.0136). Aggregate, 28 cells:

  m=1  +0.0010   m=2  +0.0018   m=5  +0.0100  [cell range -0.0004,+0.0192]
  m>=10: no cell retains 2,000 samples. Ghost at or below 0.0008 throughout.

VERDICT by the rule fixed in advance: DISQUALIFIED. Best median +0.0100
against L50 +0.0136.

DESIGN ERROR, disclosed: Rule 97's inequality was applied to the recording
SPAN (98-204 days) when it binds at the SEGMENT (~8 days between reference
outages, ~12,000 grid points). 12,000 / 2,000 = 6, so decimations above
m=5-6 were unreachable BY ARITHMETIC VISIBLE IN THE COMPOSITION SWEEP, and
the pre-registered grid to m=50 was half dead on arrival. The screen is
valid for the cells it could reach; the error is that the reachable range
was knowable in advance and was not checked.

DESCRIPTION, claimed as nothing: lag_info rises monotonically toward the
truncation point in every family. At m=5 the skewed-high-40C family sits at
median +0.0136 - exactly the pass mark - with 2 of 4 cells clearing it, and
every family's trajectory is still ascending where the samples run out.
This is the NIST failure mode softened: not hopeless physics, but a grid
that cannot reach the timescale where the dynamics live (thermal and
state-of-charge effects at tens of minutes to hours).

ANOMALY, reported and unexplained: all four cells of the uniform
charge+discharge family (RW9-12) show large NEGATIVE lag_info at m=1 and
m=2 (-0.13 to -0.17) with clean ghosts - adding driver lags actively hurts
held-out prediction. A negative of this size is not noise; no explanation
is offered and no claim is made.

LICENSED FOLLOW-UP, declared now with its motivation admitted: any coarser
regrid is post-hoc motivated by the ascent above, so it cannot simply be
re-declared and re-run. The honest design is a split-sample
pre-registration: fix the new base grid from the PHYSICS (thermal and SOC
timescales, not from these numbers), tune nothing, and evaluate on a
DISCOVERY set of 4 named cells with the verdict taken on the remaining 24
held-out cells only. Without that split, a qualifying result would be
grid-shopping.

---

## Addendum (2026-08-25): battery split-sample re-screen

Declared before scripts/battery_screen2.py ran. The motivation is admitted:
the first battery screen showed lag_info ascending toward its truncation
edge, so ANY regrid is post-hoc motivated. This design contains that in
three ways, all fixed now.

1. THE GRID COMES FROM PHYSICS, NOT FROM THE NUMBERS. An 18650 cell in
   air has a thermal time constant of roughly 10-30 minutes; resolving
   lagged influence around that timescale needs a sampling interval of a
   fraction of it. Base grid DT = 300 s. This value is also the random-walk
   step duration of the uniform protocol families - protocol knowledge, not
   result knowledge. Rule 98 arithmetic, computed IN ADVANCE this time:
   segments of ~12,000 minutes give ~2,400 grid points at DT = 300 s, so
   only m = 1 is reachable and only m = 1 is declared. No dead grid cells.

2. DISCOVERY AND VERDICT ARE DISJOINT CELLS. Discovery, named now: RW25
   (skewed high 40C), RW3 (uniform discharge room T), RW13 (skewed low room
   T), RW9 (uniform charge+discharge - the anomaly family). Discovery cells
   may be examined freely and carry no evidential weight. The verdict is
   the median lag_info over the 24 HELD-OUT cells, against the unchanged
   L50 = +0.0136, ghost beside every cell. No cell is excluded; the anomaly
   family's held-out cells stay in.

3. THIS IS THE FINAL SCREEN OF THIS DATASET. If the held-out median fails,
   the dataset is disqualified and no third grid will be sought.

Void if the grid or the discovery set changes after any number is seen, if
any held-out cell is dropped, or if a third grid is proposed after a
failure.

### Split-sample re-screen result (2026-08-25): DISQUALIFIED. Final.

Reference unchanged (L30 +0.0114, L50 +0.0136). Base grid 300 s, m = 1, as
declared.

  HELD-OUT VERDICT   median +0.0102 over 17 cells  [+0.0047, +0.0192]
                     ghost -0.0004
  DISCOVERY          median +0.0035 over 3 cells (informational, no weight)

  per family, held-out          median    cells clearing L50
  Uniform discharge (room T)   +0.0072          0/3
  Skewed high 40C              +0.0133          1/3
  Skewed high (room T)         +0.0101          0/4
  Skewed low 40C               +0.0105          0/4
  Skewed low (room T)          +0.0063          0/3

+0.0102 against +0.0136. DISQUALIFIED. As declared, this is final for this
dataset and no third grid will be sought.

A SECOND ARITHMETIC ERROR, disclosed. The addendum promised 24 held-out and
4 discovery cells; the run reports 17 and 3. Eight cells produced no segment
reaching 2,000 points at the 300 s grid - the entire uniform charge+discharge
family (RW9-12, longest segments 1,102-1,311 points) and the entire uniform
variable-charge family (RW1, RW2, RW7, RW8, longest 1,384-1,550). Their
protocol interleaves reference cycles more often, so their segments are
shorter, and 300 s x 2,000 = ~7 days exceeds what those families ever run
uninterrupted. This is Rule 98 again, one level finer: the inequality binds
per FAMILY, and I checked it against the pooled segment length rather than
the shortest family's. It is the same class of error I had just written a
rule about.

The dropped cells were not chosen: the criterion was fixed in advance and
applied identically to all 28. But two consequences must be stated. The
declared cell counts were wrong, and RW9 - a discovery cell, and the
anomaly family - vanished with them, so the -0.13 to -0.17 anomaly of the
first screen remains unexplained and is now untestable at this grid. Both
are reported rather than repaired.

WHAT THE TWO SCREENS SAY TOGETHER. At 60 s the reachable median was +0.0100;
at 300 s, with independent held-out cells, +0.0102. Coarsening five-fold
moved the statistic by 0.0002 and did not approach the pass mark, which is
evidence that the first screen's ascent was the approach to a plateau below
L50 rather than a truncated climb toward it. The dataset carries lagged
influence - the values are far above zero and the ghost is clean - but not
enough to reach the level at which outflow is known to work.

CONSEQUENCE: four screens, four verdicts. Wind tunnel QUALIFIED. Light
tunnel disqualified by physics (Rule 96). UR5 disqualified by run length
(Rule 97). Battery disqualified on the quantity itself, at two independently
declared grids, with a clean ghost - the first candidate to fail because the
influence is genuinely too weak rather than because the recording could not
express it.

---

## Addendum (2026-08-25): the PRONTO multiphase flow facility

Declared before any lag_info was computed on this data. All four test-day
process files parsed and inspected; the reachable-grid arithmetic is done
BELOW, in advance, per test-day (Rules 97, 98, 99).

### Why, and a forecast that was wrong

Cranfield's 2-inch multiphase flow rig (Stief et al. 2019; Zenodo 1341583,
CC-BY). Operators command air and water flow SET POINTS; the rig answers
through valves, pumps, two-phase transport up a riser and a separator - real
transport delay. DeltaV logs every process variable at 1 Hz CONTINUOUSLY for
a whole test day, so both the set point changes and their transients are in
the record.

A forecast made from the technical report - that the 5-7 minute flow-regime
settling time forces a grid coarser than the sample floor allows - is
recorded here as WRONG, before it could quietly disappear. It confused two
timescales. Flow-regime stabilisation is minutes, but the SET POINTS
themselves move on a scale of seconds (mean holds of 2-27 s, thousands of
changes per day), and the control loops answer at that scale. The fast
timescale is testable at 1 Hz even though the slow one is not, and the
inspection below is what corrected the forecast.

### Ground truth, fixed now

  SOURCES   FIC302/PID1/SP.CV, FIC301/PID1/SP.CV, FIC102/PID1/SP.CV,
            FIC101/PID1/SP.CV - the four commanded flow set points
  DRIVEN    the 22 measured process variables: air and water flow
            transmitters, air/water temperature and density, mixture-zone,
            riser-outlet, top-separator and three-phase pressures,
            separator gas and liquid flows, three level indicators, and the
            controllers' PROCESS VALUES (PV.CV)
  EXCLUDED  the controllers' OUTPUT values - FIC302/301/102/101/PID1/OUT.CV,
            PIC501/PID1/OUT.CV, LVC502-SR/PID1/OUT.CV (valve openings).
            These are computed by feedback from SP and PV: neither exogenous
            nor pure physical consequences, the same class excluded as the
            UR5's target velocities.

A NAMING TRAP, disclosed because it would silently corrupt the truth set:
DeltaV writes a bare transmitter's reading as "FT305/OUT.CV". That is a
MEASUREMENT. Only "/PID1/OUT.CV" is a controller output. Classifying by the
"OUT.CV" suffix alone would move fourteen sensors into the excluded set.

Per test-day, a source counts only if it varies there; a driven column
constant in a day leaves the target set.

### Composition and the reachable grid, computed in advance

  file               rows     span    setpoint changes   grids >= 2,000
  0626Testday5.csv   3,601    1.0 h   0 (all four flat)  1 s only
  0907Testday2.csv  18,601    5.2 h   13,645             1 s
  0911Testday3.csv  25,201    6.3 h   15,347             1 s, 10 s
  0912Testday4.csv  14,401    4.0 h   8,412              1 s

Declared grid: m in {1, 2, 5, 10}. Nothing coarser is declared because
nothing coarser is reachable - at 60 s the longest day yields 420 samples
against a floor of 2,000. This is Rule 98/99 applied before the fact rather
than after it, and m=10 is expected to retain 0911Testday3 alone.

0626Testday5 has NO varying set point and is therefore excluded from the
source set by Rule 80 - reported, not silently dropped.

### Rule, unchanged

Same calibrated L50 recomputed by the identical code path, ghost beside
every cell, every reachable cell reported. Screen: scripts/pronto_screen.py.
If it qualifies, the replication test gets its own pre-registration before
any outflow is computed, marginal statistic primary.

Void if the truth assignment changes after any result, if a test-day is
dropped from the report, or if a grid coarser than the declared arithmetic
allows is introduced after a failure.

### PRONTO screen result (2026-08-25): DISQUALIFIED, and negative throughout

Reference unchanged (L30 +0.0114, L50 +0.0136). Every reachable cell:

  m=1    3 days   lag_info -0.0001  (max -0.0000)  ghost -0.0000
  m=2    3 days   lag_info -0.0009  (max -0.0002)  ghost -0.0000
  m=5    3 days   lag_info -0.0014  (max -0.0007)  ghost -0.0001
  m=10   1 day    lag_info -0.0199  (max -0.0199)  ghost -0.0003

DISQUALIFIED. Not merely below the pass mark: NEGATIVE at every grid, and
increasingly so as the grid coarsens. Adding the commanded set points'
history makes held-out prediction of the measured process variables WORSE.

A CORRECTION to this addendum's own composition table: it recorded
0626Testday5 as having flat set points, excluded by Rule 80. Wrong. That
file has a different column set entirely - 69 columns, and NO SP.CV channels
at all. It was never a candidate for this truth assignment, and the reason
stated was the wrong one. The exclusion stands; its justification is
corrected.

THE LIKELY MECHANISM, offered as a reading and not a claim. A source
contributes a poly2 expansion of its three delay lags, so three varying set
points add 54 features to a 9-feature own-lag baseline. Where those features
carry no predictive information, a ridge at fixed regularisation pays a
held-out penalty for carrying them, and the penalty grows as decimation
shrinks the sample count - which is the observed pattern, monotone from
-0.0001 at 25,000 samples to -0.0199 at 2,500. On this reading a negative
lag_info means "no lagged information, plus the cost of asking", and the
statistic is behaving correctly. Testing that would need a feature-count
control, which is not run here.

This is the second dataset to return negative values - the battery's uniform
charge+discharge family gave -0.13 to -0.17 - and the two share the
structure of many added driver features against a well-predicted target. The
battery anomaly is no longer isolated, and no explanation is claimed for
either.

WHY THIS APPARATUS FAILS. The rig has genuine transport delay, but the
control loops close on their set points fast relative to 1 Hz, so the
measured flows track the commands within a sample; and the slow flow-regime
dynamics that would carry lag live at 5-7 minutes, where the sample floor
puts a 60 s grid out of reach (420 samples against 2,000 on the longest
day). Squeezed from both sides - the UR5 mode, on a chemical plant.
