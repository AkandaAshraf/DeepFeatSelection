# Pre-registration: should the null be per channel rather than one number?

Declared 2026-09-05, before the experiment was written or run.

## The proposal this REPLACES, and why it was abandoned unrun

The improvement first proposed was: regress ghost excess on the donor's own
self-R2 across the panel, then predict each channel's expected null from its
own self-R2 and score the deviation. It was motivated by a real observation
in the crossed-saturation cells -- Spearman(self_r2_med, ghost_max) = -0.507,
n = 90, holding within every width.

Three checks against existing outputs killed it before it was written.

  1. THE CORRELATION IS A NOISE CONFOUND. self_r2_med in that run is a
     downstream reading of the observation-noise manipulation. Controlling
     noise and width, the partial rank correlation is +0.169, p = 0.11 --
     not significant, and the SIGN FLIPS. Source FP tracks noise (rho
     +0.448, p = 1e-5) more strongly than it tracks the ghost bar (+0.177,
     n.s.). The regression would have fitted the confound.
  2. THE DONOR PANEL HAS NO LEVERAGE. Donors are filtered to self-R2 > 0.9
     by construction, so across all 1485 recoverable donor-ghost pairs in
     ExpOutput/boundary_map the regressor spans [0.952, 1.000]. Predicting
     the null for a channel at self-R2 = 0.3 is extrapolation from a
     0.048-wide window.
  3. THE LEDGER HAD ALREADY CLOSED IT, TWICE. 2026-08-23: "the same self-R2
     carries different risk at different width, so a one-dimensional
     per-channel saturation bar cannot exist in the proposed form."
     2026-09-02: "a per-channel gate on self-R2 alone stays dead, but the
     second dimension is REDUNDANCY, not width."

Recorded because proposing it again should cost nothing to refute.

## The defect that survives all three checks

The defect is not which covariate the bar should use. It is that there is no
per-channel evidence at all:

```
thr = max(0, ghosts.max())          # ONE number, applied to every channel
flagged = excess > thr
```

Thirty ghosts are collapsed to their maximum and that single bar is applied
to every channel whatever its noise, saturation, autocorrelation or
redundancy. No channel gets a p-value, so there is nothing to control a
false-discovery rate with and nothing to rank confidence by.

That matters for the stated research strategy. If the plan is many cheap
screens in the hope of a few real detections, then uncalibrated per-channel
evidence makes volume actively harmful: more screens multiply false
positives instead of accumulating rare truths. A calibrated per-channel
p-value is the thing that makes running more experiments pay.

## The design

Make each channel its own null. For channel q, circularly shift q's own
series S times and score each shifted copy exactly as the real channel is
scored. The resulting null inherits q's noise, autocorrelation, saturation
and redundancy jointly, without anyone having to name which dimension
matters -- which is precisely what the closed self-R2 gate could not do.

Three arms share the SAME encoders and the SAME excess array. Only the
decision rule differs, so the comparison is exactly paired and any
difference is attributable to the rule alone.

  A  GLOBAL-MAX   thr = max(0, max over 30 donor ghosts), donors filtered
                  self-R2 > 0.9 with the existing fallback. THE INCUMBENT.
  B  SELF-Q       per channel, thr_q = max(0, Q(null_q, 30/31)).
  C  SELF-BH      per channel, p_q = (1 + #{null_q >= excess_q}) / (1 + S),
                  then Benjamini-Hochberg at FDR q = 0.10.

B is deliberately NOT the maximum of the per-channel null. The maximum of 30
i.i.d. draws sits at the 30/31 = 0.968 quantile in expectation, so scoring
the per-channel null at q = 0.968 matches the incumbent's effective
exceedance level. Taking a max over S = 99 draws instead would raise the bar
mechanically, and B would lose recall for a reason that has nothing to do
with being per-channel. This choice is fixed here, before any result.

  S       99 shifts, drawn in [m/4, 3m/4]. Minimum attainable p is 0.01,
          which BH needs; 30 would floor p at 0.032.
  WIDTH   V in {30, 60, 120}
  NOISE   observation noise in {0.0, 0.05, 0.30}, added as in
          crossed_saturation.py. This axis is mandatory, not decoration:
          at the boundary map's centre the incumbent's source FP is already
          0.000, so the centre cannot show whether calibration helps. The
          crossed run puts incumbent source FP at 0.000 / 0.172 / 0.232
          across these three levels.
  SEEDS   0, 1, 2
  b       2V, the capacity law, so no arm is capacity-crippled and
          differences are attributable to the decision rule.
  SYSTEM  boundary_map.make_system, n = 4000, coupling 0.20, redundancy 0.

27 cells x 3 arms.

## Predictions, fixed now

  P1  DECISIVE. At noise 0.30, both self-null arms reduce the source
      false-positive rate against GLOBAL-MAX, pooled over widths and seeds.
      This is the entire point: the incumbent bar is calibrated in one
      regime and applied in another.
  P2  DECISIVE. At noise 0.0, arm B's recall is not more than 0.05 below
      arm A's. Per-channel calibration that costs recall in the easy regime
      is not free, and the matched quantile above is what makes this a fair
      test rather than a rigged one.
  P3  NO PREDICTION on B versus C. Quantile bar against FDR control is a
      second axis and predicting it either way would be storytelling.
  P4  DISQUALIFYING. At noise 0.0, precision stays >= 0.95 in every arm. A
      calibration that buys source blindness by flagging nothing, or by
      flagging everything, is rejected whatever it does to P1.
  P5  PHASE-LOCK DIAGNOSTIC, declared as a threat to validity. The
      generator draws r ~ U(3.6, 3.9), which contains the locked windows
      the 2026-09-03 audit identified, so roughly 12% of channels are
      near-periodic. A circular shift of a near-periodic series can
      REALIGN, which would make the self-null carry real structure and
      inflate it. Channels with max|autocorrelation| over lags 1..50 above
      0.9 are flagged and their null levels reported separately. If locked
      channels' self-nulls are inflated, the self-null is INVALID for them
      and that is reported as a scope limit, not buried.
  P6  Wall-clock recorded per arm. The per-channel null costs S/30 times
      more ridge solves than the panel; if that is not roughly what is
      observed, the accounting is wrong and must be corrected.

## The rule, fixed now

  ADOPT   P1 holds and P2 and P4 hold. The per-channel null becomes the
          default, the arm that wins on P1 without losing P2 is named, and
          every deployed recall figure is re-derived under it.
  REJECT  P4 fails. Recorded as an attempted improvement that broke
          precision, and the global bar stands.
  NULL    P1 fails with P4 intact. The incumbent bar is not miscalibrated
          across regimes in a way a per-channel null repairs; reported as a
          negative result and the incumbent stands.
  SCOPED  P1 holds but P5 shows locked-channel inflation large enough to
          account for it. Then the gain is a generator artefact and is
          reported as such, not adopted.

## Void conditions

Void if the three arms are computed from different encoders or different
excess arrays; if S, the grid, the seeds, the FDR level or arm B's 30/31
quantile change after any result is seen; if the per-channel null for an arm
is built from a different shift set than the one scored; if P1 is judged
anywhere other than noise 0.30; or if the phase-lock diagnostic is dropped.

---

## Result (2026-09-05): REJECT by the declared rule. The gain was mostly bar height.

27 cells x 3 arms, 61.1 min, S = 99 shifts per channel.

  source FP by noise      GLOBAL-MAX   SELF-Q   SELF-BH
    0.00                     0.000      0.061     0.233
    0.05                     0.128      0.083     0.133
    0.30                     0.244      0.161     0.150

  recall by noise         GLOBAL-MAX   SELF-Q   SELF-BH
    0.00                     0.812      0.959     0.980
    0.05                     0.389      0.422     0.599
    0.30                     0.092      0.121     0.112

  P1 HOLDS. At noise 0.30 both self-null arms cut source false positives
     against the incumbent, 0.244 -> 0.161 and 0.150, at every width.
  P2 HOLDS, and in the opposite direction to the risk it guarded against:
     SELF-Q's recall at noise 0.0 is ABOVE the incumbent's, 0.959 to 0.812.
  P4 FAILS. Minimum precision at noise 0.0 is 0.917, below the 0.95 floor.
  P5 locked fraction 0.100; per-channel null bar -0.00008 on locked channels
     against -0.00177 on free ones, an inflation of +0.00169. Real, in the
     predicted direction, and below the 0.005 ghost bar. SCOPED does not fire.
  P6 null cost 78.4 s against 0.3 s, a factor of 261 rather than the S/30 =
     3.3 predicted. The prediction was wrong: the incumbent scores 30 donors
     while the per-channel null scores 99 shifts for EVERY channel, so the
     true ratio is 99V/30, which at V = 60 is 198. P6's arithmetic was
     stated wrongly in the protocol and is corrected here.

### The verdict is REJECT and it stands

P4 failed, so the declared rule fires REJECT. Nothing is adopted on this run.

P4 failed **entirely because of SELF-BH**, and for a reason that is our own
fault at declaration time rather than a fact about the method:

  arm          min precision at noise 0.0    cells below 0.95
  GLOBAL-MAX             1.000                    0 of 9
  SELF-Q                 0.957                    0 of 9
  SELF-BH                0.917                    4 of 9

SELF-BH's realised false-discovery rate at noise 0.0 was 0.045, comfortably
inside its declared bound of 0.10. Controlling FDR at 0.10 permits precision
as low as 0.90, so a 0.95 precision floor and an FDR-0.10 arm are
incompatible by construction: SELF-BH could fail P4 while behaving exactly
as specified, and did. The two clauses should never have been declared
together.

That does NOT license adopting SELF-Q on this run. The protocol says every
arm; one arm failed; the verdict is REJECT. SELF-Q needs its own
pre-registration on fresh seeds with a coherent precision floor.

### The adversarial control the protocol should have declared

POST HOC, labelled as such, scripts/per_channel_matched_bar.py.

SELF-Q flags more channels than the incumbent (56.0 against 47.9 at noise
0.0). If that is the whole story, then lowering the global bar until it
flags the SAME NUMBER must reproduce SELF-Q. Matching the count exactly in
every cell and taking the top-k by excess:

  matched flag count      SELF-Q   top-k global      SELF-Q   top-k global
    noise                 recall     recall         source FP  source FP
    0.00                   0.959      0.963           0.061      0.039
    0.05                   0.422      0.413           0.083      0.128
    0.30                   0.121      0.104           0.161      0.244

**Most of the headline recall gain is bar height, not per-channel structure.**
At noise 0.0 the matched global bar is slightly BETTER than SELF-Q on both
metrics. In 19 of 27 cells the per-channel null selects exactly the same set
as top-k. It changes the selection in only 8 cells, and there it wins 7-1 on
recall (Wilcoxon p = 0.039, n = 8) and 7-1 on source FP (p = 0.10, n = 8),
concentrated in the noisy cells where source blindness is the problem.

### What this actually establishes

  - The incumbent's global maximum is TOO CONSERVATIVE. It is a
    family-wise bar taken across 30 donor channels, and applying it as a
    per-channel bar costs recall the ranking had already earned. This is
    the real defect and it is now measured: 0.812 against 0.963 at a
    matched count.
  - MACE's RANKING of channels by excess is already good. The per-channel
    null adds little to it, changing the selected set in under a third of
    cells.
  - The per-channel null's genuine contribution is that it CHOOSES how many
    to flag without a tuning constant. Top-k is not a deployable rule --
    k is exactly what is unknown in deployment -- so the matched-count
    control bounds the claim without replacing the method.
  - At 261x the null cost, that contribution is not yet worth its price.
    A cheaper route to the same threshold selection, such as a lower
    declared quantile of the existing donor panel, is the obvious next
    pre-registration and has not been run.

### Not established

Three seeds, one generating family, one coupling, redundancy 0. The
matched-count control is post hoc and its two Wilcoxon tests rest on the 8
cells where the arms disagree. The phase-lock inflation is real but small at
this noise range and was not probed at the locked windows deliberately.
