# Pre-registration: does scaling alpha recover the V=500/1000 signal loss?

Declared 2026-09-08, before the script exists. NOT RUN as part of this
pre-registration; drafted per the scale-diagnosis task's own requirement
that a future scientific run needs a pre-registration in place first.
EXPLORATORY. No verdict word licensed; this document does not draw one.

## What this tests

The 2026-09-08 post-hoc diagnosis appended to
paper/large_system_protocol.md found, from data already on disk plus one
synthetic check using boundary_map.ridge_r2 unmodified, that a fixed
penalty (ALPHA = 1.0, unconditional on feature count) drives held-out R2 to
the helper's own zero-clamp once a fixed-strength signal is diluted among
as few as 100-500 additional noise-like columns, at large_system's actual
n_train = 2397. This is offered as the leading cause of the V=500/1000
collapse. It is a hypothesis about a GENERIC ridge property, checked in
isolation on synthetic noise columns; whether it explains the REAL system
code's failure, which is learned and correlated rather than i.i.d. noise,
is untested.

This is the cheapest test that could move that hypothesis: does raising or
scaling alpha recover a KNOWN, planted signal at the SAME (n_train, p)
shapes large_system actually used, purely synthetically, before any GPU
time is spent confirming it on the real pipeline.

## Design

CPU only. No encoder trained, no dataset touched, no file under Data/ or
ExpOutput/large_system/ read or written. This step deliberately has NO
encoder at all, which is the strongest form of isolating READOUT
regularization from ENCODER capacity: with no encoder in the design, any
recovery can only come from the readout's own alpha, not from a
confounding change in what the encoder learned.

  SIGNAL      y = s*x0 + noise, s fixed at 0.15 (matches the diagnosis
              check), x0 one genuinely informative column, noise ~
              N(0, 1-s^2), giving a p=1 ceiling R2 around 0.03.
  SPLIT       0.6 / 0.2 / 0.2 train / validation / test, matching this
              project's standard split (boundary_map.py and large_system.py
              both use it). n_train = 2397, matching large_system's actual
              value at every V tested; n_val and n_test scaled the same way.
  P           {20, 100, 500, 1000, 2000} -- spans the diagnosis check's
              range, including large_system's actual V=500 (p~1019) and
              V=1000 (p~2019) shapes
  SEEDS       0-4, five synthetic draws per (p, arm) cell -- this is a
              pure-noise synthetic generator, cheap enough to afford more
              seeds than the real pipeline, and the diagnosis check used
              only one draw per p

  ARMS, alpha selected on VALIDATION ONLY where selection is involved,
  R2 always REPORTED on the held-out TEST split, never on validation
    ALPHA-FIXED     alpha = 1.0, the incumbent, unchanged from
                    boundary_map.ridge_r2. No selection; reported on test.
    ALPHA-SCALED    alpha = p (a common rule-of-thumb scaling; grows with
                    feature count the way the incumbent's does not). No
                    selection; reported on test.
    ALPHA-VAL       alpha chosen from a log-spaced grid by lowest error on
                    the VALIDATION split only, then that single chosen
                    alpha is refit on train and reported on test. The
                    validation split is never touched for anything but
                    this choice, and test is never touched until reporting.

25 (p x seed) cells x 3 arms = 75 ridge fits, CPU, expected seconds total.

## Predictions, fixed now

  A1  DECISIVE. At p=2000 (matching V=1000's actual own+sys_code width),
      ALPHA-VAL recovers TEST-set held-out R2 to within 50% of the p=1
      ceiling (about 0.014), pooled over 5 seeds. If it does not, alpha
      alone does not explain the collapse and this candidate mechanism is
      WRONG, not merely incomplete -- it remains a candidate, not the
      established cause, either way (see the diagnosis's own hedge above).
  A2  At p=2000, ALPHA-SCALED recovers PART of the signal (nonzero mean
      test R2) but is not required to match ALPHA-VAL. This is deliberately
      a weak, directional prediction: the rule-of-thumb scaling is a cheap
      first pass, not expected to be optimal.
  A3  NO PREDICTION on the exact p at which ALPHA-FIXED's R2 first reaches
      the zero clamp; the diagnosis check already measured that at p=100
      for the SAME signal strength, and refitting it here is a
      consistency check, not a new prediction.
  A4  GUARD. ALPHA-FIXED reproduces the diagnosis check's own numbers
      (R2 = 0 by p=100) within this run, on fresh seeds. If it does not,
      something about this reimplementation differs from the diagnosis
      check and the run is UNINFORMATIVE until reconciled.

## The rule, fixed now

NONE, and no adoption is possible from this run alone, by design -- it
tests a mechanism on synthetic noise, not the real system code. Outcomes:

  RECOVERED     A1 and A4 hold. Licenses the CONTINGENT step already named
                in the diagnosis, itself requiring its own pre-registration
                before any GPU time is spent, designed to keep isolating
                the readout from the encoder: train ONE encoder per (V,
                seed) exactly as large_system.py does today, freeze it, and
                score that SAME frozen system code under each alpha arm
                above (selected on validation, reported on test). Every
                arm sees an identical encoder; only the readout's
                regularisation differs between arms, so any recovery is
                attributable to the readout and not to a confounded change
                in what the encoder learned. Not a full V=500/1000
                large_system rerun -- the readout stage only, reusing the
                same generator/seeds/splits.
  NOT RECOVERED A1 fails with A4 intact. The fixed-alpha hypothesis is
                wrong or insufficient as the sole explanation; the
                diagnosis's ranked list moves to its next-lowest item
                (separate per-block ridge fits) rather than to a retrain.
  UNINFORMATIVE A4 fails. The instrument, not the hypothesis, is what was
                measured.

## Void conditions

Void if any file under Data/ or ExpOutput/ is read or written by the
script; if boundary_map.ridge_r2 itself is modified rather than called
unmodified; if the grid, seeds, signal strength or alpha choices change
after any result is seen; or if A1 is judged anywhere other than p=2000.
