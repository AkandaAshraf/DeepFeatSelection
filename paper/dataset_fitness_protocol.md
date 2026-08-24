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
