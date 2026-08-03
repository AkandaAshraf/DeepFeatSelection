"""Orientation panel: RESIT verdicts against ground truth on the project's systems.

Every pair's truth is known from the generator, and the committed expectations
include the refusals: an orientation method that never says "cannot tell" is
not usable, so the panel scores honesty as well as accuracy.

    python scripts/orient_pairs.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from deepfeatselect.anm import anm_orient
from deepfeatselect.synthetic import nonlinear_scm, redundancy_demo


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--stride", type=int, default=3,
                   help="thinning for the map series; the HSIC permutation test "
                        "assumes exchangeable rows and the chaotic map is "
                        "autocorrelated over a few steps")
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="ExpOutput")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    system = redundancy_demo(n=args.n, seed=args.seed)
    x = np.asarray(system["x"], dtype=np.float64)[:: args.stride]
    names = list(system["feature_names"])
    driver = x[:, names.index("driver")]
    proxy_cos = x[:, names.index("proxy_cos")]
    proxy_sin = x[:, names.index("proxy_sin")]
    unrelated = x[:, names.index("unrelated")]

    def noisy(v: np.ndarray) -> np.ndarray:
        return v + args.noise * rng.standard_normal(len(v))

    scm = nonlinear_scm(n=len(driver), seed=args.seed)
    col = {name: scm.x[:, i] for i, name in enumerate(scm.feature_names)}

    # (label, x, y, ground truth, committed expectation)
    panel = [
        ("driver -> proxy_cos + noise", driver, noisy(proxy_cos),
         "x->y", "x->y"),
        ("driver -> proxy_sin + noise", driver, noisy(proxy_sin),
         "x->y", "x->y"),
        ("proxy_cos+n <-> proxy_sin+n (common cause u)", noisy(proxy_cos), noisy(proxy_sin),
         "confounded", "no ANM in either direction"),
        ("driver -> proxy_cos (noiseless, 2-to-1)", driver, proxy_cos,
         "x->y", "x->y (deterministic)"),
        ("driver -> 2*driver-1 (noiseless bijection)", driver, 2.0 * driver - 1.0,
         "x->y", "deterministic bijection: unidentifiable"),
        ("driver <-> unrelated map", driver, unrelated,
         "none", "independent"),
        ("linear-Gaussian control", *(lambda a: (a, 0.8 * a + 0.6 * rng.standard_normal(len(a))))(
            rng.standard_normal(len(driver))),
         "x->y", "undecided: both admissible"),
        ("scm: z -> x_conf1 (tanh + noise)", col["z"], col["x_conf1"],
         "x->y", "x->y"),
        ("scm: z -> x_conf2 (square + noise)", col["z"], col["x_conf2"],
         "x->y", "x->y"),
    ]

    rows = []
    for label, a, b, truth, expected in panel:
        result = anm_orient(a, b, seed=args.seed)
        # A refusal that the generator says is forced (bijection, confounding,
        # linear-Gaussian) counts as honest, not as a miss.
        ok = result.verdict == expected
        rows.append({
            "pair": label,
            "truth": truth,
            "expected": expected,
            "verdict": result.verdict,
            "ok": ok,
            "p_fwd": result.p_forward,
            "p_bwd": result.p_backward,
            "rvar_fwd": result.residual_ratio_forward,
            "rvar_bwd": result.residual_ratio_backward,
        })
        print(f"{'OK ' if ok else 'MISS'} {label}")
        print(f"     truth={truth}  verdict={result.verdict}")
        print(f"     p_fwd={result.p_forward:.3f} p_bwd={result.p_backward:.3f} "
              f"rvar_fwd={result.residual_ratio_forward:.2e} "
              f"rvar_bwd={result.residual_ratio_backward:.2e}")

    df = pd.DataFrame(rows)
    n_ok = int(df.ok.sum())
    print(f"\n{n_ok}/{len(df)} panel entries matched the committed expectation")

    from pathlib import Path
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "orientation_panel.csv", index=False)
    print(f"wrote {outdir}\\orientation_panel.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
