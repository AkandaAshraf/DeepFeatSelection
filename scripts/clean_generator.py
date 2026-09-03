"""The outflow closure re-tested on a generator with no phase-locked sources.

Pre-registration: paper/clean_generator_protocol.md, committed before this
was run. coupled_clean() is coupled() with every U(3.7, 3.9) draw
rejection-sampled against generator_audit.is_locked_r; nothing else in the
generator or the pipeline changes. one() is sink_bar.one with the shape
parameterised and per-channel rows kept.

    python scripts/clean_generator.py
"""

from __future__ import annotations

import sys
import time
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import source_outflow_gate as G  # noqa: E402
from generator_audit import is_locked_r, max_ac  # noqa: E402

OUT = Path("ExpOutput/clean_generator")
ALPHA, COUPLING, EPOCHS = 0.05, 0.50, 25
BAR_TO_BEAT = 0.80
DEAD, MAX_DEAD = 0.002, 3
GHOST_MAX = 0.05
SHAPES = {
    "3/6/6": (dict(n_src=3, n_sink=6, n_iso=6),
              range(1100, 1130), range(1200, 1230), "0.53 (0.5-0.7)"),
    "2/11/2": (dict(n_src=2, n_sink=11, n_iso=2),
               range(1300, 1330), range(1400, 1430), "0.83"),
}
AUDIT_BAR = {"3/6/6": 0.00557, "2/11/2": 0.01585}   # clean-subset bars


def draw_chaotic(rng, k):
    """U(3.7, 3.9) with phase-locked windows rejected."""
    out = []
    while len(out) < k:
        r = float(rng.uniform(3.7, 3.9))
        if not is_locked_r(r):
            out.append(r)
    return np.array(out)


def coupled_clean(n=4000, n_src=3, n_sink=6, n_iso=6, coupling=0.35, seed=0):
    """coupled() with the source and isolated r draws rejection-sampled.

    Returns x, role, r (per channel) and parent (per channel, -1 if none).
    """
    rng = np.random.default_rng(seed)
    V = n_src + n_sink + n_iso
    x = np.zeros((n, V))
    x[0] = rng.uniform(0.2, 0.8, V)
    r_src = draw_chaotic(rng, n_src)
    r_iso = draw_chaotic(rng, n_iso)
    r_snk = rng.uniform(3.5, 3.7, n_sink)
    parent = rng.integers(0, n_src, n_sink)
    for t in range(n - 1):
        s = x[t, :n_src]
        x[t + 1, :n_src] = np.clip(r_src * s * (1 - s), 0, 1)
        k = x[t, n_src:n_src + n_sink]
        x[t + 1, n_src:n_src + n_sink] = np.clip(
            r_snk * k * (1 - k) + coupling * x[t, parent] * (1 - k), 0, 1)
        i = x[t, n_src + n_sink:]
        x[t + 1, n_src + n_sink:] = np.clip(r_iso * i * (1 - i), 0, 1)
    x += 0.01 * rng.standard_normal((n, V))
    role = np.array(["source"] * n_src + ["sink"] * n_sink
                    + ["isolated"] * n_iso)
    r = np.concatenate([r_src, r_snk, r_iso])
    par = np.full(V, -1)
    par[n_src:n_src + n_sink] = parent
    return x, role, r, par


def auc(pos, neg):
    """Identical to error_metrics.auc; see scripts/test_auc_identical.py."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"[{c - h:.2f}, {c + h:.2f}]"


def one(name, shape, seed, tag):
    V = sum(shape.values()) + 1                      # + ghost
    G.BOTTLENECK, G.SEED = 4 * V, seed
    x, role, r, par = coupled_clean(coupling=COUPLING, seed=seed, **shape)
    exc, out = G.analyse(x, epochs=EPOCHS)
    ghost = float(out[-1])
    out = out[:-1]
    src, snk = out[role == "source"], out[role == "sink"]
    n_assigned = np.bincount(par[par >= 0], minlength=shape["n_src"])
    chans = [{"shape": name, "set": tag, "seed": seed, "channel": j,
              "role": role[j], "r": float(r[j]), "parent": int(par[j]),
              "n_sinks": int(n_assigned[j]) if role[j] == "source" else -1,
              "outflow": float(out[j]), "inflow": float(exc[j]),
              "max_ac": float(max_ac(x[:, j]))}
             for j in range(len(role))]
    run = {"shape": name, "set": tag, "seed": seed, "b": 4 * V,
           "source": float(np.median(src)), "sink": float(np.median(snk)),
           "iso": float(np.median(out[role == "isolated"])), "ghost": ghost,
           "auc": auc(src, snk),
           "n_orphan": int((n_assigned == 0).sum()),
           "src_max_ac": float(max(max_ac(x[:, j])
                                   for j in np.flatnonzero(role == "source")))}
    return run, chans


def sweep(name, shape, seeds, tag, t0):
    runs, chans = [], []
    for s in seeds:
        run, ch = one(name, shape, s, tag)
        runs.append(run)
        chans.extend(ch)
        print(f"  {name:>6} {tag:<4} seed {s}  src {run['source']:+.5f}  "
              f"snk {run['sink']:+.5f}  ghost {run['ghost']:+.5f}  "
              f"auc {run['auc']:.3f}  ({(time.time()-t0)/60:.1f} min)",
              flush=True)
    return pd.DataFrame(runs), pd.DataFrame(chans)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("clean generator: no phase-locked sources or isolated channels")
    print(f"alpha={ALPHA}  coupling={COUPLING}  b=4V  epochs={EPOCHS}  "
          f"models={G.MODELS}  bar to beat {BAR_TO_BEAT}\n")
    t0 = time.time()
    runs, chans, summ = [], [], {}
    for name, (shape, cal_seeds, test_seeds, dirty) in SHAPES.items():
        cal, cc = sweep(name, shape, cal_seeds, "cal", t0)
        test, tc = sweep(name, shape, test_seeds, "test", t0)
        runs += [cal, test]
        chans += [cc, tc]
        both = pd.concat([cal, test])
        bar = float(np.quantile(cal.sink, 1 - ALPHA))
        k = int((test.source > bar).sum())
        summ[name] = {
            "bar": bar, "k": k, "n": len(test), "sens": k / len(test),
            "auc": float(test.auc.median()),
            "source_med": float(test.source.median()),
            "sink_med": float(test.sink.median()),
            "dead": int((both.source < DEAD).sum()),
            "ghost_clear": float((both.ghost > bar).mean()),
            "src_max_ac": float(both.src_max_ac.max()),
            "orphan_runs": int((both.n_orphan > 0).sum()), "dirty": dirty}
        print(flush=True)
    pd.concat(runs).to_csv(OUT / "runs.csv", index=False)
    pd.concat(chans).to_csv(OUT / "channels.csv", index=False)
    pd.DataFrame(summ).T.to_csv(OUT / "summary.csv")
    print(f"({(time.time()-t0)/60:.1f} min)\n")

    for name, s in summ.items():
        print(f"{name:>6}: bar {s['bar']:+.5f} (audit clean-subset bar "
              f"{AUDIT_BAR[name]:+.5f})   sens {s['k']}/{s['n']} = "
              f"{s['sens']:.2f} {wilson(s['k'], s['n'])}   AUC {s['auc']:.3f}"
              f"\n        source med {s['source_med']:+.5f}  sink med "
              f"{s['sink_med']:+.5f}   dead {s['dead']}/60   ghost clears bar "
              f"{s['ghost_clear']:.2f}   max source |ac| {s['src_max_ac']:.2f}"
              f"   runs with an orphan {s['orphan_runs']}/60"
              f"\n        dirty generator, same shape: {s['dirty']}")

    a, c = summ["3/6/6"], summ["2/11/2"]
    q1, q2 = a["sens"] >= BAR_TO_BEAT, c["sens"] >= BAR_TO_BEAT
    q3 = c["source_med"] > a["source_med"] and c["auc"] > a["auc"]
    q4 = max(a["dead"], c["dead"]) > MAX_DEAD
    gap = c["sens"] - a["sens"]
    q5 = gap < 0.30
    q6 = max(a["ghost_clear"], c["ghost_clear"]) > GHOST_MAX

    print(f"\nQ1  DECISIVE 3/6/6 sens {a['sens']:.2f} vs {BAR_TO_BEAT}: "
          f"{'HOLDS' if q1 else 'FAILS'}")
    print(f"Q2  2/11/2 sens {c['sens']:.2f} vs {BAR_TO_BEAT}: "
          f"{'HOLDS' if q2 else 'FAILS'}")
    print(f"Q3  shape effect, threshold-free: source med {a['source_med']:+.5f} "
          f"-> {c['source_med']:+.5f}, AUC {a['auc']:.3f} -> {c['auc']:.3f}: "
          f"{'PERSISTS' if q3 else 'does NOT persist'}")
    print(f"Q4  dead runs {a['dead']}/60 and {c['dead']}/60 vs max {MAX_DEAD}: "
          f"{'FIRES' if q4 else 'does not fire'}")
    print(f"Q5  sensitivity gap {gap:+.2f} vs dirty +0.30: "
          f"{'SHRINKS' if q5 else 'does NOT shrink'}")
    print(f"Q6  ghost clears bar {a['ghost_clear']:.2f} / {c['ghost_clear']:.2f}"
          f" vs {GHOST_MAX}: {'VOID' if q6 else 'clean'}")

    print("\nVERDICT (rule fixed before running; one branch per outcome)")
    if q6:
        print("   -> VOID. The ghost clears the bar; the pipeline leaks.")
    elif q4:
        print("   -> NO VERDICT. Dead runs persist without a locked source; "
              "the locking\n      explanation is incomplete. Diagnosis next, "
              "not another bar.")
    elif q1 and q2:
        print("   -> CLOSURE WAS AN ARTEFACT. Both shapes clear 0.80 on a "
              "generator with chaotic\n      sources. Section 12's closure "
              "is withdrawn: it measured the generator, not the\n      "
              "statistic. The 2026-09-02 shape result is re-described as "
              "obtained on the\n      defective generator, with Q3 beside it.")
    elif (not q1) and q2:
        print("   -> CLOSURE STANDS WITH A SHAPE CONDITION. Locking "
              "contributed to the 3/6/6 failure\n      but did not cause it; "
              "the reopen is confirmed on a clean generator; the shape\n"
              "      effect is real.")
    elif q1 and not q2:
        print("   -> ANOMALOUS. 3/6/6 clears and 2/11/2 does not; the audit's "
              "clean subsets were not\n      representative. Both shape "
              "results are re-examined before any closure verdict.")
    else:
        print("   -> CLOSED ON THE CLEAN GENERATOR. The locking explanation "
              "is wrong; the closure\n      stands and the reopen's 0.83 is "
              "downgraded to seed-block fortune.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
