"""Audit of the coupled-logistic generator behind every outflow experiment.

Written 2026-09-03 after the reopen and ratio-sweep runs, as a post-hoc
diagnostic. Nothing here is a pre-registered result; the pre-registered
follow-up is paper/clean_generator_protocol.md.

Five checks, each reproducing a number quoted in the ledger entry of
2026-09-03:

  1  r-range scan: what fraction of r ~ U(3.7, 3.9) gives a phase-locked
     (periodic or band-periodic) source, and where the windows are.
  2  per-run locked-source counts for the sink-bar (3/6/6) and reopen
     (2/11/2) runs, from the same seeds, and the sensitivities restricted to
     runs with no locked source.
  3  the paper's own fitness-gate quantity (dR2 from adding the parent's
     lags) for sinks of locked versus chaotic parents.
  4  the fraction of sinks that spend >30% of steps at the clip boundary,
     by coupling.
  5  orphan sources (no sink assigned) in the ratio sweep's cells.

    python scripts/generator_audit.py
"""

from __future__ import annotations

import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import source_outflow_gate as G  # noqa: E402
from source_outflow_coupling import coupled  # noqa: E402

OUT = Path("ExpOutput/generator_audit")
LOCK_AC, LOCK_LYAP, MAXLAG = 0.90, 0.05, 12
NL = chr(10)


# ------------------------------------------------------- lock detection

def orbit_stats(r, x0=0.4, burn=500, n=3000):
    """Lyapunov exponent and max |autocorrelation| over lags 1..MAXLAG."""
    x = x0
    for _ in range(burn):
        x = r * x * (1 - x)
    s, lam = np.empty(n), 0.0
    for t in range(n):
        lam += np.log(abs(r * (1 - 2 * x)) + 1e-300)
        x = r * x * (1 - x)
        s[t] = x
    return lam / n, max_ac(s)


def max_ac(s):
    s = s - s.mean()
    return max(abs(np.corrcoef(s[:-L], s[L:])[0, 1])
               for L in range(1, MAXLAG + 1))


def is_locked_r(r):
    lam, ac = orbit_stats(r)
    return lam <= LOCK_LYAP or ac >= LOCK_AC


def locked_sources(x, role):
    """Observed-series test: a source is locked if max |ac| >= LOCK_AC."""
    return np.array([max_ac(x[:, j]) >= LOCK_AC
                     for j in np.flatnonzero(role == "source")])


def draws(seed, shape):
    """Replay the generator's RNG to recover r and the parent assignment."""
    rng = np.random.default_rng(seed)
    V = sum(shape.values())
    rng.uniform(0.2, 0.8, V)
    r_src = rng.uniform(3.7, 3.9, shape["n_src"])
    rng.uniform(3.7, 3.9, shape["n_iso"])
    rng.uniform(3.5, 3.7, shape["n_sink"])
    parent = rng.integers(0, shape["n_src"], shape["n_sink"])
    return r_src, parent


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 2), round(c + h, 2)


def indent(text, pad="      "):
    return text.replace(NL, NL + pad)


# ---------------------------------------------------------------- checks

def check1_scan():
    print("1  r-range scan, r in U(3.7, 3.9), 4001 points")
    rs = np.linspace(3.7, 3.9, 4001)
    st = np.array([orbit_stats(r) for r in rs])
    rej = (st[:, 0] <= LOCK_LYAP) | (st[:, 1] >= LOCK_AC)
    segs, start, prev = [], None, rs[0]
    for r, flag in zip(rs, rej):
        if flag and start is None:
            start = r
        if not flag and start is not None:
            segs.append((start, prev))
            start = None
        prev = r
    if start is not None:
        segs.append((start, rs[-1]))
    wide = [(round(a, 4), round(b, 4)) for a, b in segs if b - a >= 0.001]
    print(f"   rejected fraction {rej.mean():.3f}  "
          f"(lyapunov<={LOCK_LYAP} or max|ac|>={LOCK_AC} over lags 1..{MAXLAG})")
    print(f"   windows wider than 0.001: {wide}")
    for k in (1, 2, 3, 6, 12):
        print(f"   P(>=1 locked source | n_src={k}) = {1-(1-rej.mean())**k:.2f}")
    pd.DataFrame({"r": rs, "lyapunov": st[:, 0], "max_ac": st[:, 1],
                  "locked": rej}).to_csv(OUT / "r_scan.csv", index=False)
    return float(rej.mean())


def check2_runs():
    print(NL + "2  locked sources per run, and sensitivity restricted to clean runs")
    rows = []
    specs = [("sink_bar 3/6/6", "ExpOutput/sink_bar/runs.csv",
              dict(n_src=3, n_sink=6, n_iso=6), "calibration"),
             ("reopen 2/11/2", "ExpOutput/chamber_shape_reopen/runs.csv",
              dict(n_src=2, n_sink=11, n_iso=2), "cal")]
    for name, path, shape, calname in specs:
        d = pd.read_csv(path)
        if "coupling" in d:
            d = d[d.coupling == 0.5]
        nl = []
        for s in d.seed:
            x, role = coupled(coupling=0.5, seed=s, **shape)
            nl.append(int(locked_sources(x, role).sum()))
        d = d.assign(n_locked=nl, experiment=name)
        rows.append(d)
        cal, te = d[d.set == calname], d[d.set != calname]
        bar = float(np.quantile(cal.sink, 0.95))
        bar_c = float(np.quantile(cal[cal.n_locked == 0].sink, 0.95))
        tc = te[te.n_locked == 0]
        k0 = int((te.source > bar).sum())
        kc = int((tc.source > bar).sum())
        kcc = int((tc.source > bar_c).sum())
        dead = d.source < 0.002
        print(f"   {name}: runs with >=1 locked source "
              f"{int((d.n_locked > 0).sum())}/{len(d)}; dead runs (source<0.002) "
              f"{int(dead.sum())}, of which with a locked source "
              f"{int((dead & (d.n_locked > 0)).sum())}")
        tab = d.groupby("n_locked").agg(
            n=("seed", "size"), source_med=("source", "median"),
            sink_med=("sink", "median")).round(5).to_string()
        print("      " + indent(tab))
        print(f"      declared:             bar {bar:+.5f}  sens {k0}/{len(te)} "
              f"= {k0/len(te):.2f} {wilson(k0, len(te))}")
        print(f"      clean test, same bar: bar {bar:+.5f}  sens {kc}/{len(tc)} "
              f"= {kc/len(tc):.2f} {wilson(kc, len(tc))}")
        print(f"      clean cal and test:   bar {bar_c:+.5f}  sens {kcc}/{len(tc)} "
              f"= {kcc/len(tc):.2f} {wilson(kcc, len(tc))}   "
              f"(clean cal n={int((cal.n_locked == 0).sum())})")
    pd.concat(rows).to_csv(OUT / "locked_runs.csv", index=False)


def check3_gate(seeds=range(800, 830), shape=None):
    shape = shape or dict(n_src=2, n_sink=11, n_iso=2)
    print(NL + "3  the paper's fitness-gate quantity on the reopen test seeds")
    print("   dR2[sink(t+1) | own lags + parent lags] - R2[own lags], poly2, E=3")
    rows = []
    for s in seeds:
        x, role = coupled(coupling=0.5, seed=s, **shape)
        _, par = draws(s, shape)
        locked = locked_sources(x, role)
        emb = G.embed(x)
        m = emb.shape[0]
        a, b = int(0.6 * m), int(0.8 * m)
        tr_i, te_i = np.arange(0, a - 1), np.arange(b, m - 1)
        mu, sd = emb[:a].mean(0), emb[:a].std(0) + 1e-12
        zs = np.clip((emb - mu) / sd, -20, 20)
        lead = zs[:, [j * G.E for j in range(x.shape[1])]]
        feats = [G.poly2(zs[:, q * G.E:(q + 1) * G.E])
                 for q in range(x.shape[1])]
        ns = shape["n_src"]
        for i, q in enumerate(range(ns, ns + shape["n_sink"])):
            p = par[i]
            own = G.ridge_r2(feats[q][tr_i], lead[tr_i + 1, q],
                             feats[q][te_i], lead[te_i + 1, q])
            both = G.ridge_r2(
                np.hstack([feats[q][tr_i], feats[p][tr_i]]), lead[tr_i + 1, q],
                np.hstack([feats[q][te_i], feats[p][te_i]]), lead[te_i + 1, q])
            rows.append(dict(seed=s, sink=q, parent=p,
                             parent_locked=bool(locked[p]),
                             own_r2=own, dR2=both - own))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "gate_by_parent.csv", index=False)
    tab = d.groupby("parent_locked").agg(
        n=("dR2", "size"), own_r2_med=("own_r2", "median"),
        dR2_med=("dR2", "median"),
        dR2_q25=("dR2", lambda v: v.quantile(.25)),
        dR2_q75=("dR2", lambda v: v.quantile(.75))).round(4).to_string()
    print("   " + indent(tab, "   "))
    print("   calibrated gate: L50 = +0.0136, L30 = +0.0114")


def check4_clip(shape=None, seeds=range(800, 830)):
    shape = shape or dict(n_src=2, n_sink=11, n_iso=2)
    print(NL + "4  sinks at the clip boundary (>30% of steps at x<0.01 or x>0.99)")
    for c in (0.30, 0.50, 0.70):
        fr = []
        for s in seeds:
            x, role = coupled(coupling=c, seed=s, **shape)
            snk = x[:, role == "sink"]
            fr.append((((snk > 0.99).mean(0) + (snk < 0.01).mean(0)) > 0.3).mean())
        fr = np.array(fr)
        print(f"   coupling {c:.2f}: mean fraction of clipped sinks {fr.mean():.2f}"
              f"   runs with every sink clipped {int((fr == 1).sum())}/{len(seeds)}")


def check5_orphans():
    print(NL + "5  orphan sources in the ratio sweep (a source with no sink assigned)")
    print("   cell        orphans/run   frac   ratio -> effective   locked/run")
    for ns, nk in [(12, 12), (6, 12), (4, 12), (3, 15), (2, 16), (1, 11)]:
        shape = dict(n_src=ns, n_sink=nk, n_iso=25 - ns - nk)
        o, nl = [], []
        for s in range(1000, 1020):
            r_src, par = draws(s, shape)
            o.append(int((np.bincount(par, minlength=ns) == 0).sum()))
            nl.append(int(sum(is_locked_r(r) for r in r_src)))
        eff = nk / (ns - np.mean(o))
        print(f"   {ns:>2}/{nk:>2}/{shape['n_iso']:>2}    {np.mean(o):.2f} (max {max(o)})"
              f"    {np.mean(o)/ns:.2f}   {nk/ns:4.1f} -> {eff:4.1f}          "
              f"{np.mean(nl):.2f}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    check1_scan()
    check2_runs()
    check3_gate()
    check4_clip()
    check5_orphans()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
