"""Forecast-horizon adequacy check.

Pre-registration: paper/forecast_horizon_check_protocol.md (c2079c6 + pre-run
amendment 7789591). For each target channel and horizon h=1..10, fit a direct
h-step model on train rows and score unclipped R2 on test rows. A (model,
target) is adequate iff (a) s(1) >= 0.10 and s(1) > persistence(1),
(b) s(h+1) <= s(h) + 0.02 for all h, (c) s(1) - s(10) >= 0.05.

    python scripts/forecast_horizon_check.py
    python scripts/forecast_horizon_check.py --family1b   # family1_generator_fix_protocol.md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import parent_screening as PS  # noqa: E402
import parent_screening_arms as PA  # noqa: E402
from boundary_map import poly3  # noqa: E402

OUT = Path("ExpOutput/forecast_horizon_check")
H, EMAX, E3, V, N = 10, 6, 3, 24, 4000
SEEDS = {"family1": (26001, 26002, 26003, 26004),
         "family2": (27001, 27002, 27003, 27004)}
NOISE = {"family1": 26091, "family2": 27091}
MODELS = ["M1_SIMPLEX", "M2_LINEAR", "M3_POLY3", "M4_MASKED_AE", "M5_GROUP_AE",
          "C_LEAK"]
S1_MIN, TOL, DROP = 0.10, 0.02, 0.05


def split_rows(n):
    t = np.arange(EMAX - 1, n - H)
    m, emb = len(t), EMAX + H
    a, b = int(0.6 * m), int(0.8 * m)
    return t[:a - emb], t[a:b - emb], t[b:]


def r2(pred, y):
    return float(1.0 - np.mean((pred - y) ** 2) / (np.var(y, ddof=1) + 1e-12))


def lags(x, rows, E, end_shift=0):
    return np.stack([x[rows + end_shift - k] for k in range(E - 1, -1, -1)], 1)


def simplex(x, tr, ev, E, h):
    lib = lags(x, tr, E)
    tree = cKDTree(lib)
    d, idx = tree.query(lags(x, ev, E), k=E + 1)
    d = np.atleast_2d(d)
    idx = np.atleast_2d(idx)
    w = np.exp(-d / np.maximum(d[:, :1], 1e-12))
    return (w * x[tr[idx] + h]).sum(1) / w.sum(1)


def ridge_curve(feats_fn, x, tr, te):
    out = []
    for h in range(1, H + 1):
        Ftr, Fte = feats_fn(tr, h), feats_fn(te, h)
        out.append(PS.ridge_r2_val(Ftr, x[tr + h], Fte, x[te + h])[0])
    return out


def run_system(x_obs, guard):
    n = len(x_obs)
    tr, va, te = split_rows(n)
    raw_end = tr[-1] + H + 1
    mu, sd = x_obs[:raw_end].mean(0), x_obs[:raw_end].std(0) + 1e-12
    xs = (x_obs - mu) / sd
    all_rows = np.arange(EMAX - 1, n - H)
    pos = {int(t): i for i, t in enumerate(all_rows)}
    tr_i = np.array([pos[int(t)] for t in tr])

    labels = PS.cluster_size_capped(xs[:raw_end])
    groups = {g: sorted(np.where(labels == g)[0].tolist())
              for g in sorted(set(labels.tolist()))}
    gcode = {}
    for gid, members in groups.items():
        guard.check_light("group encoder")
        z = np.concatenate([lags(xs[:, j], all_rows, E3) for j in members],
                           1).astype(np.float32)
        net = PS.train_group_encoder(z, tr_i, len(members), seed=gid,
                                     guard=guard)
        code = PS.group_code(net, z, None)
        for j in members:
            gcode[j] = code
    res = []
    for q in range(V):
        guard.check_light(f"target {q}")
        x = xs[:, q]
        curves, info = {}, {}
        vals = [r2(simplex(x, tr, va, E, 1), x[va + 1]) for E in range(1, EMAX + 1)]
        Ebest = int(np.argmax(vals)) + 1
        info["E_simplex"] = Ebest
        curves["M1_SIMPLEX"] = [r2(simplex(x, tr, te, Ebest, h), x[te + h])
                                for h in range(1, H + 1)]
        curves["M2_LINEAR"] = ridge_curve(lambda r, h: lags(x, r, E3), x, tr, te)
        curves["M3_POLY3"] = ridge_curve(lambda r, h: poly3(lags(x, r, E3)),
                                         x, tr, te)
        z1 = lags(x, all_rows, E3).astype(np.float32)
        net = PS.train_group_encoder(z1, tr_i, 1, seed=1000 + q, guard=guard)
        c1 = PS.group_code(net, z1, None)
        curves["M4_MASKED_AE"] = ridge_curve(
            lambda r, h: np.hstack([poly3(lags(x, r, E3)),
                                    c1[[pos[int(t)] for t in r]]]), x, tr, te)
        curves["M5_GROUP_AE"] = ridge_curve(
            lambda r, h: np.hstack([poly3(lags(x, r, E3)),
                                    gcode[q][[pos[int(t)] for t in r]]]),
            x, tr, te)
        curves["C_LEAK"] = ridge_curve(
            lambda r, h: poly3(lags(x, r, E3, end_shift=h - 1)), x, tr, te)
        info["persistence"] = [r2(x[te], x[te + h]) for h in range(1, H + 1)]
        res.append(dict(q=q, curves=curves, **info))
    return res, [len(v) for v in groups.values()]


def verdict(s, pers):
    a = s[0] >= S1_MIN and s[0] > pers[0]
    b = all(s[i + 1] <= s[i] + TOL for i in range(H - 1))
    c = s[0] - s[-1] >= DROP
    return a, b, c


def main():
    global OUT, SEEDS, NOISE
    f1b = "--family1b" in sys.argv
    if f1b:
        import generators_v2 as G2
        OUT = Path("ExpOutput/forecast_horizon_check_f1b")
        SEEDS, NOISE = {"family1b": (28001, 28002, 28003, 28004)}, {"family1b": 28091}
    OUT.mkdir(parents=True, exist_ok=True)
    PA._take_lock("forecast horizon check (V=24, 10 systems, small encoders)")
    guard = PA.ResourceGuard(1800, out_dir=OUT)
    rows, t0 = [], time.perf_counter()
    try:
        guard.check_full("start")
        jobs = []
        gens = ((("family1b", G2.family1b_generate),) if f1b else
                (("family1", PS.family1_generate), ("family2", PS.family2_generate)))
        for fam, gen in gens:
            for s in SEEDS[fam]:
                jobs.append((fam, s, "dynamical", lambda g=gen, s=s: g(V, N, s)))
            jobs.append((fam, NOISE[fam], "noise", lambda s=NOISE[fam]:
                         dict(x_obs=np.random.default_rng(s).standard_normal((N, V)))))
        acc = {}
        for fam, seed, kind, make in jobs:
            sysd = make()
            res, sizes = run_system(sysd["x_obs"], guard)
            if f1b and kind == "dynamical":
                acc[seed] = acceptance(sysd)
            for r in res:
                for mname in MODELS:
                    s = r["curves"][mname]
                    a, b, c = verdict(s, r["persistence"])
                    rows.append(dict(family=fam, seed=seed, kind=kind, q=r["q"],
                                     model=mname, skill=s,
                                     persistence=r["persistence"],
                                     E_simplex=r["E_simplex"], a=a, b=b, c=c,
                                     adequate=a and b and c))
            print(f"{fam} {seed} {kind}: done ({time.perf_counter()-t0:.0f}s, "
                  f"groups {sorted(sizes, reverse=True)[:6]}...)", flush=True)
            (OUT / "rows.json").write_text(json.dumps(rows))
    finally:
        PA.LOCK.unlink(missing_ok=True)
    report(rows, guard)
    if f1b:
        report_acceptance(rows, acc)


def acceptance(sysd):
    """Generator-level checks on DRIVEN channels (family1_generator_fix_protocol)."""
    x, parent = sysd["x_obs"], sysd["parent"]
    z = (x - x.mean(0)) / (x.std(0) + 1e-12)
    driven = [q for q in parent if parent[q]]
    ac2 = [np.corrcoef(z[:-2, q], z[2:, q])[0, 1] for q in driven]
    p2left = ((z[2:] - z[:-2]).var(0) / 2)[driven]
    P, Nn = [], []
    for q in driven:
        pa = {j for j, _ in parent[q]}
        for j in range(z.shape[1]):
            if j == q:
                continue
            m = max(abs(np.corrcoef(z[:-d, j], z[d:, q])[0, 1]) for d in (1, 2, 3))
            (P if j in pa else Nn).append(m)
    return dict(driven=driven, ac2_median=float(np.median(ac2)),
                p2_var_left=float(np.median(p2left)),
                parent_corr=float(np.median(P)), nonparent_corr=float(np.median(Nn)))


def report_acceptance(rows, acc):
    fam = "family1b"
    m1 = [r for r in rows if r["family"] == fam and r["kind"] == "dynamical"
          and r["model"] == "M1_SIMPLEX" and r["q"] in acc[r["seed"]]["driven"]]
    rate = float(np.mean([r["adequate"] for r in m1]))
    ac2 = float(np.median([a["ac2_median"] for a in acc.values()]))
    p2 = float(np.median([a["p2_var_left"] for a in acc.values()]))
    sep = all(a["parent_corr"] > a["nonparent_corr"] for a in acc.values())
    ok = rate >= 0.80 and p2 >= 0.20 and sep
    print()
    print("FAMILY 1b ACCEPTANCE (driven channels)")
    print(f"  M1 simplex adequate on driven: {rate:.2f} (need >=0.80)")
    print(f"  period-2 variance left on driven: {p2:.3f} (need >=0.20; defective "
          f"generator 0.006)")
    print(f"  median lag-2 autocorrelation (reported only): {ac2:+.3f}")
    for s, a in acc.items():
        print(f"  seed {s}: parent |corr| {a['parent_corr']:.3f} vs non-parent "
              f"{a['nonparent_corr']:.3f}")
    print(f"  parents above non-parents in every seed: {sep}")
    print(f"  FAMILY 1b {'ACCEPTED' if ok else 'REJECTED'}")
    (OUT / "acceptance.json").write_text(json.dumps(dict(
        accepted=bool(ok), m1_driven_adequate=rate, ac2_median=ac2, p2_var_left=p2,
        per_seed={str(k): v for k, v in acc.items()}), indent=1, default=float))


def report(rows, guard):
    ok_controls = True
    print("\nCONTROLS")
    for fam in SEEDS:
        for m in MODELS:
            nz = [r for r in rows if r["family"] == fam and r["kind"] == "noise"
                  and r["model"] == m]
            rate = np.mean([not r["a"] for r in nz])
            ok = rate >= 0.95
            ok_controls &= ok
            print(f"  noise {fam} {m:13s} fails (a): {rate:.2f} (need >=0.95) "
                  f"{'ok' if ok else 'VIOLATED'}")
        dyn = lambda m: [r for r in rows if r["family"] == fam  # noqa: E731
                         and r["kind"] == "dynamical" and r["model"] == m]
        m3a = {(r["seed"], r["q"]) for r in dyn("M3_POLY3") if r["a"]}
        leak = [r for r in dyn("C_LEAK") if (r["seed"], r["q"]) in m3a]
        rate = np.mean([not r["c"] for r in leak]) if leak else float("nan")
        ok = leak and rate >= 0.80
        ok_controls &= bool(ok)
        print(f"  leak  {fam} C_LEAK fails (c) where M3 learns: {rate:.2f} "
              f"of {len(leak)} (need >=0.80) {'ok' if ok else 'VIOLATED'}")
    summary = dict(controls_ok=bool(ok_controls), families={})
    print("\nMODEL ADEQUACY (dynamical targets, 96 per family; bar 0.80)")
    for fam in SEEDS:
        summary["families"][fam] = {}
        for m in MODELS[:-1]:
            d = [r for r in rows if r["family"] == fam and r["kind"] == "dynamical"
                 and r["model"] == m]
            rate = np.mean([r["adequate"] for r in d])
            med = np.median([r["skill"] for r in d], axis=0)
            fa = sum(not r["a"] for r in d)
            fb = sum(r["a"] and not r["b"] for r in d)
            fc = sum(r["a"] and r["b"] and not r["c"] for r in d)
            summary["families"][fam][m] = dict(
                adequate=float(rate), n=len(d), fail_a=fa, fail_b_only=fb,
                fail_c_only=fc, median_curve=med.tolist())
            verdict_word = ("ADEQUATE" if rate >= 0.80 else "not adequate") \
                if ok_controls else "(controls void)"
            print(f"  {fam} {m:13s} adequate {rate:.2f}  fail a/b/c "
                  f"{fa}/{fb}/{fc}  median s(h) "
                  f"{' '.join(f'{v:+.2f}' for v in med)}  -> {verdict_word}")
        Es = [r["E_simplex"] for r in rows if r["family"] == fam and
              r["kind"] == "dynamical" and r["model"] == "M1_SIMPLEX"]
        print(f"  {fam} simplex chosen E counts: "
              f"{dict(zip(*np.unique(Es, return_counts=True)))}")
    summary["guard"] = guard.snapshot()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"\nelapsed {guard.elapsed():.0f}s, peak tree RSS "
          f"{guard.peak_tree_rss:.0f}MB, min free {guard.min_free:.0f}MB")


if __name__ == "__main__":
    main()
