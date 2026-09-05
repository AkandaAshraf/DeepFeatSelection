"""What does the screen actually buy, at the pessimistic end of its interval?

The method's claim is triage, not identification: observational structure
concentrates an expensive interventional search. That claim is only useful
if it comes with a number a practitioner can plan against, and the number
they can plan against is the LOWER BOUND, not the point estimate.

This derives, per deployment, from primary output files:

  base rate    prevalence of the target class in that population
  precision    hit rate inside the shortlist
  ENRICHMENT   precision / base rate -- the multiplier on search efficiency
  recall       fraction of the target class the shortlist recovers
  experiments per discovery, with and without the screen

Every quantity is reported at its lower confidence bound where an interval
exists. Deployments where the screen did NOT enrich are included; they are
the reason the summary interval is wide.

    python scripts/enrichment_table.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("ExpOutput/enrichment")
ROWS = []


def add(domain, base, base_lo, prec, prec_lo, recall, recall_lo, source,
        enriched):
    """One deployment. *_lo are pessimistic bounds; None where unmeasured."""
    def lift(p, b):
        return (p / b) if (p is not None and b) else None
    ROWS.append({
        "domain": domain, "base_rate": base, "precision": prec,
        "enrichment": lift(prec, base),
        # pessimistic: worst precision against the LARGEST plausible base rate
        "enrichment_lo": lift(prec_lo, base_lo if base_lo else base),
        "recall": recall, "recall_lo": recall_lo,
        "expts_per_hit_blind": (1 / base) if base else None,
        "expts_per_hit_screened": (1 / prec) if prec else None,
        "expts_per_hit_screened_lo": (1 / prec_lo) if prec_lo else None,
        "enriched": enriched, "source": source,
    })


# ---- 1. DepMap: interventional equivalence given observational redundancy
# Pre-registered B=100 intervals are primary (B=1000 agreed; see ledger).
b = pd.read_csv("ExpOutput/depmap_calibration/bootstrap_ci_b1000.csv")
base_row = b.iloc[0]
add("DepMap CRISPR (gene pairs)",
    base=0.0042, base_lo=0.0048,          # widest base -> most pessimistic lift
    prec=0.170, prec_lo=0.097,            # ceiling 17.0% [9.7, 27.3]
    recall=None, recall_lo=None,          # not defined: no enumerable positive set
    source="ledger B=100 primary; bootstrap_ci_b1000.csv agrees",
    enriched=True)

# ---- 2. Causal chamber: actuators vs sensors, ground truth by construction
d = pd.read_csv("ExpOutput/real_conditional/channels_wt_intake_impulse_v1.csv")
n_src = int((d.role == "source").sum())
n_tot = int(d.role.isin(["source", "sensor"]).sum())
base_ch = n_src / n_tot
# shortlist = top n_src by marginal outflow, pooled over runs
top = d[d.role.isin(["source", "sensor"])].nlargest(n_src, "A1")
prec_ch = float((top.role == "source").mean())
# pessimistic: worst single run's top-2 precision
per_run = []
for r, g in d[d.role.isin(["source", "sensor"])].groupby("run"):
    k = int((g.role == "source").sum())
    per_run.append(float((g.nlargest(k, "A1").role == "source").mean()))
add("Causal chamber (actuators)",
    base=base_ch, base_lo=base_ch,
    prec=prec_ch, prec_lo=float(np.min(per_run)),
    recall=prec_ch, recall_lo=float(np.min(per_run)),   # top-k: recall==precision
    source="channels_wt_intake_impulse_v1.csv, top-k over 5 runs",
    enriched=True)

# ---- 3. Synthetic boundary map: driven detection, truth by construction
bm = pd.read_csv("ExpOutput/boundary_map/boundary_map.csv")
sl = bm[(bm.n == 4000) & (bm.coupling == 0.2) & (bm.redundancy == 0)]
# base rate = driven fraction; recover it from recall and flagged counts
base_bm = float(np.median((sl.n_flagged / sl.recall.replace(0, np.nan)) / sl.V))
add("Synthetic driven-variable scan",
    base=base_bm, base_lo=base_bm,
    prec=float(sl.precision.median()), prec_lo=float(sl.precision.min()),
    recall=float(sl.recall.median()), recall_lo=float(sl.recall.min()),
    source="boundary_map.csv, V sweep at centre",
    enriched=True)

# ---- 4. Worm, freely moving: pre-registered replication that did NOT enrich
add("C. elegans, freely moving",
    base=0.33, base_lo=0.33,
    prec=0.33, prec_lo=0.33,              # 2 of 6 cleared = the base rate, p=0.64
    recall=None, recall_lo=None,
    source="ledger 2026-08-16 corpus scan, 91 recordings",
    enriched=False)

# ---- 5. iEEG SOZ, held-out cohort: replication that failed, wrong direction
add("iEEG seizure-onset zone",
    base=None, base_lo=None,
    prec=None, prec_lo=None,
    recall=None, recall_lo=None,
    source="ledger 2026-08-20 replication, z=-1.455 (wrong direction)",
    enriched=False)

t = pd.DataFrame(ROWS)
OUT.mkdir(parents=True, exist_ok=True)
t.to_csv(OUT / "enrichment.csv", index=False)


def fmt(v, spec=".2f", pct=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "  --"
    return f"{v*100:{spec}}%" if pct else f"{v:{spec}}"


print("ENRICHMENT BY DEPLOYMENT — lower bounds are what a planner can rely on\n")
hdr = (f"{'deployment':30s} {'base':>7s} {'prec':>7s} {'lift':>7s} "
       f"{'LIFT_LO':>8s} {'recall':>7s} {'REC_LO':>7s}")
print(hdr)
print("-" * len(hdr))
for _, r in t.iterrows():
    print(f"{r.domain[:30]:30s} {fmt(r.base_rate,'.2f',True):>7s} "
          f"{fmt(r.precision,'.1f',True):>7s} {fmt(r.enrichment,'.1f'):>7s} "
          f"{fmt(r.enrichment_lo,'.1f'):>8s} "
          f"{fmt(r.recall,'.2f'):>7s} {fmt(r.recall_lo,'.2f'):>7s}")

print("\nEXPERIMENTS PER DISCOVERY (DepMap, the only enumerable case)")
dm = t.iloc[0]
print(f"   blind search       {dm.expts_per_hit_blind:6.0f}")
print(f"   screened, point    {dm.expts_per_hit_screened:6.1f}")
print(f"   screened, WORST    {dm.expts_per_hit_screened_lo:6.1f}   <- plan against this")

# A deployment can only demonstrate triage if the base rate leaves room:
# at base p the maximum attainable lift is 1/p. Where that ceiling is near
# 1, the deployment is uninformative about enrichment, not a success.
t["max_possible_lift"] = t.base_rate.map(
    lambda b: (1 / b) if b else None)
t["informative"] = t.max_possible_lift.map(
    lambda m: bool(m is not None and m >= 2.0))

print("\nSTRUCTURAL CEILING ON LIFT (1 / base rate)")
for _, r in t.iterrows():
    m = r.max_possible_lift
    print(f"   {r.domain[:30]:30s} "
          f"{'ceiling ' + format(m, '.1f') + 'x' if m else 'base unknown':>16s}"
          f"   {'' if r.informative else '<- cannot show triage'}")

inf = t[t.informative]
ok = int(inf.enriched.sum())
n = len(inf)
# Jeffreys interval on the success rate across deployments
from scipy.stats import beta                                  # noqa: E402
lo, hi = beta.ppf([0.025, 0.975], ok + 0.5, n - ok + 0.5)
print(f"\nAMONG DEPLOYMENTS THAT COULD SHOW TRIAGE: {ok} of {n} enriched")
print(f"   rate {ok/n:.2f}, Jeffreys 95% interval [{lo:.2f}, {hi:.2f}]")
print("   The interval is wide because the sample is 5 and the deployments")
print("   are not exchangeable draws; the carve-up is a judgement call.")
print("\n   The honest planning statement: on a new domain, treat enrichment")
print("   as roughly a coin flip until the fitness gate and a base-rate")
print("   estimate say otherwise. Screening is minutes; the gate is free.")
print(f"\nwritten: {OUT / 'enrichment.csv'}")
