# Pre-registration: does scaling alpha recover the V=500/1000 signal loss?

Declared 2026-09-08. This revises the same-day draft after independent
review found four defects, listed below, before the script exists.
EXPLORATORY. No verdict word licensed; this document does not draw one.

## Revision note, kept rather than silently edited

The first draft is superseded by this version, which fixes:

  1. UNSUPPORTED CAUSAL WORDING. The draft called the fixed-alpha
     mechanism "the leading cause of the V=500/1000 collapse" and spoke of
     a synthetic result "confirming it on the real pipeline". The
     diagnosis this protocol follows from explicitly does NOT establish a
     cause -- it reports a candidate mechanism, checked on synthetic noise
     columns, not on the real learned system code. Both phrases are
     replaced below with language the diagnosis itself would sign off on.
  2. AN UNSATISFIABLE VOID CONDITION. The draft required "boundary_map.
     ridge_r2 called unmodified" for every arm, but ridge_r2's penalty is a
     MODULE-LEVEL CONSTANT, not a function parameter -- there is no way to
     call the unmodified function at two different alpha values. Fixed by
     specifying a local, alpha-parameterised reimplementation of the exact
     same formula, with an explicit equivalence check against the real
     ridge_r2 at alpha=1.0 as a pre-flight step, rather than a condition
     that could never have held.
  3. NO DECLARED RESOURCE CAP. The draft estimated "expected seconds
     total" but declared no cap and no stop rule. Fixed below.
  4. AN EMPIRICALLY-MEASURED, SEED-DEPENDENT CEILING. The draft's "p=1
     ceiling" was one noisy draw from an earlier ad hoc script. The
     population ceiling for this signal is exact and analytic; fixed
     below, with a per-seed empirical p=1 control retained alongside it
     for a finite-sample reference point.

## What this tests

The 2026-09-08 diagnosis (paper/large_system_protocol.md) found that
hierarchy_repair.py's e2 term (a small, V-independent ridge) stays stable
from V=120 to V=1000 while its e3 term (a ridge conditioned on the full,
V-scaled system code) crosses zero and inverts. It reports, as a CANDIDATE
mechanism rather than an established cause, that boundary_map.ridge_r2's
fixed penalty (ALPHA = 1.0, unconditional on feature count) can produce
exactly this pattern: a synthetic check showed a fixed-strength signal
driven to the helper's own zero-clamp once diluted among a few hundred
additional noise-like columns, at large_system's actual n_train = 2397.

This is the cheapest test that could move that candidate: does selecting
alpha properly (on a validation split, never on test) recover a KNOWN,
planted signal at the SAME (n_train, p) shapes large_system actually used,
purely synthetically, before any GPU time is spent testing whether the
same fix does anything for the real, learned system code. A positive
result here would make the candidate worth testing on the real pipeline in
a SEPARATE, later pre-registration. It would not, by itself, show that
fixed alpha is what actually happened in the real run.

## Design

CPU only, forced explicitly regardless of local CUDA availability -- this
step deliberately has NO encoder and NO GPU work at all, which is the
strongest form of isolating READOUT regularisation from ENCODER capacity:
with no encoder in the design, any recovery can only come from the
readout's own alpha, not from a confounding change in what an encoder
learned.

  SIGNAL      y = s*x0 + sqrt(1-s^2)*noise, x0 and noise independent unit
              Gaussians, s fixed at 0.15. The population R2 of the best
              possible fit on x0 alone is EXACTLY s^2 = 0.0225 (not
              measured, derived: Var(y)=1, best residual variance is
              1-s^2, R2=1-(1-s^2)/1=s^2). This is the ceiling every arm is
              read against.
  SPLIT       0.6 / 0.2 / 0.2 train / validation / test, matching this
              project's standard split convention. n_train = 2397
              (large_system's actual value at every V), n_val = 799,
              n_test = 799.
  P           {1, 20, 100, 500, 1000, 2000}. p=1 is a per-seed EMPIRICAL
              reference control (the finite-sample ceiling, alongside the
              analytic one above), reported descriptively, not compared
              across arms since there is nothing to regularise
              differently at one feature. {20,...,2000} are the tested
              cells, spanning large_system's actual V=500 (p~1019) and
              V=1000 (p~2019) shapes.
  SEEDS       0-4, five synthetic draws per (p, arm) cell.

  RIDGE FORMULA, a local function used by every arm, taking alpha as an
  explicit parameter: A = Xtr^T Xtr + alpha*I; w = solve(A, Xtr^T ytr);
  R2 = max(0, 1 - test_err/test_var). Identical arithmetic to
  boundary_map.ridge_r2 with alpha as a parameter instead of a module
  constant. PRE-FLIGHT CORRECTNESS CHECK, run and printed PASS/FAIL before
  any declared cell: on one fixed synthetic draw, this local function at
  alpha=1.0 must match boundary_map.ridge_r2's own output to within 1e-9.
  A FAIL stops the run before any declared cell is computed.

  ARMS, alpha selected on VALIDATION ONLY where selection is involved,
  R2 always REPORTED on the held-out TEST split, never on validation
    ALPHA-FIXED     alpha = 1.0, the incumbent value, via the local
                    formula above (verified equivalent to the real
                    ridge_r2 by the pre-flight check). No selection.
    ALPHA-SCALED    alpha = p, a rule-of-thumb scaling that grows with
                    feature count the way the incumbent's does not. No
                    selection.
    ALPHA-VAL       alpha chosen from a log-spaced grid
                    {0.1, 1, 10, 100, 1000, 10000, 100000} by lowest
                    squared error on the VALIDATION split only, then that
                    single chosen alpha is refit on train and reported on
                    test. Validation is touched only for this choice; test
                    is touched only for reporting.

30 (p x seed) cells x 3 arms (p=1's control cell reported separately, not
multiplied by arm since alpha is nearly inert there) = under 100 ridge
fits total.

## Resource caps, fixed now, stop rather than shrink on breach

  RUNTIME   <= 5 minutes wall-clock for the entire script, pre-flight
            included. This is a CPU-only, at-most-2000-feature ridge grid;
            5 minutes is generous headroom over the expected single-digit
            seconds.
  RSS       <= 500 MB process RSS at any point, checked after every cell.
  DISK      CORRECTED, self-contradictory as first written: no file under
            Data/, and no file under any PRIOR experiment's output
            directory (ExpOutput/large_system/ in particular, whose
            results this run must not read, write, or overwrite), is
            touched. This run's OWN new, declared output is exempted from
            that restriction by construction: ExpOutput/ridge_alpha_scaling/
            cells.csv, ONE compact summary CSV under 50 KB (one row per
            p/seed/arm/p=1-control cell: p, seed, arm, alpha_used, r2) --
            no raw arrays, no per-cell model objects, no cache files.
            PYTHONDONTWRITEBYTECODE=1 for the run itself.

A breach of any cap: record which cap, at what value, at which cell; stop
immediately; report the failure. Does not retry with a smaller grid, fewer
seeds, or a narrower P range. Shrinking scope on a breach is what this
document forbids, matching the large_system precedent's own stop rule.

## Predictions, fixed now

  A1  DECISIVE. At p=2000 (matching V=1000's actual own+sys_code width),
      ALPHA-VAL's mean test R2, pooled over 5 seeds, is at least 50% of
      the analytic ceiling (>= 0.01125).
  A2  At p=2000, ALPHA-SCALED's mean test R2 is strictly greater than
      zero. Deliberately weak and directional: the rule-of-thumb scaling
      is a cheap first pass, not expected to be optimal.
  A3  NO PREDICTION on the exact p at which ALPHA-FIXED's R2 first reaches
      the zero clamp; the diagnosis check already measured that near p=100
      for the same signal strength, and reproducing it here is a
      consistency check, not a new prediction.
  A4  GUARD. ALPHA-FIXED's mean R2 at p=100, this run, is 0.000 (the
      clamp), on fresh seeds. If it is not, the reimplementation used here
      differs from the diagnosis check in some way not yet identified, and
      the run is UNINFORMATIVE until reconciled.

## The rule, fixed now

NONE, and no adoption is possible from this run alone, by design -- every
outcome below tests the mechanism on synthetic noise, not the real system
code, and NONE of them authorises retraining an encoder or spending GPU
time. A later, GPU-touching follow-up needs its OWN separate
pre-registration regardless of what this run shows; nothing here starts
it automatically.

  A CANDIDATE THAT SURVIVED THIS TEST   A1 and A4 hold. The fixed-alpha
      mechanism recovers a known synthetic signal under proper validation
      selection at the real pipeline's own feature-count shape. This
      makes the candidate worth testing on the real, learned system code
      in a future, separately pre-registered run -- described here only
      as a pointer, not authorised: freeze one trained encoder per (V,
      seed) exactly as large_system.py produces today, and score that
      SAME frozen code under each alpha arm above, so any recovery is
      attributable to the readout alone. That future run is NOT part of
      this document and NOT licensed by this result.
  A CANDIDATE THAT DID NOT SURVIVE   A1 fails with A4 intact. Selecting
      alpha properly does not recover the signal even on controlled
      synthetic noise at the real pipeline's own shape. This weakens the
      fixed-alpha candidate materially; it does not need to have been the
      sole possible contributor to have been worth ruling out this
      cheaply, and the diagnosis's lower-ranked candidate (separate
      per-block ridge fits) becomes the next thing worth a cheap test,
      not a real retrain.
  UNINFORMATIVE   A4 fails, or any resource cap breaches. The instrument,
      not the mechanism, is what was measured.

## Void conditions

Void if any file under Data/, or under any prior experiment's output
directory including ExpOutput/large_system/, is read or written by the
script -- this run's own declared new output,
ExpOutput/ridge_alpha_scaling/cells.csv, is exempted by construction, per
the DISK cap correction above; if the local ridge formula's arithmetic
differs from
boundary_map.ridge_r2's own formula in anything but taking alpha as a
parameter (checked by the pre-flight equivalence test, not asserted); if
the grid, seeds, signal strength, alpha choices or resource caps change
after any result is seen; if A1 is judged anywhere other than p=2000; if
the P range or seed count is expanded after seeing a partial result --
this run is fixed in scope from the moment it starts, not adaptively
grown; or if any conclusion about the REAL system code, rather than this
synthetic test, is drawn from this run's result alone.
