"""Controls for the Perturb-seq calibration, declared in the protocol.

The protocol makes these gating: if same-complex pairs do not show similar
transcriptomic responses, the similarity measure is wrong and the study is
VOID until it is fixed. That rule is what caught the Pearson/proximity error
in the DepMap study, where canonical proteasome subunits scored 0.04.

    python scripts/perturbseq_controls.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from depmap_calibration import log  # noqa: E402
from perturbseq_calibration import PS, load_perturbseq  # noqa: E402

OUT = Path("ExpOutput/perturbseq")
CORUM = Path("ExpOutput/depmap_calibration/control_corum.csv")
NEG = Path("ExpOutput/depmap_calibration/control_negative.csv")
TAU = 0.2          # the declared calibration point


def cosine_for_pairs(pairs, R, own_col, index):
    """Cosine between two knockdown responses, both on-target columns zeroed."""
    out = []
    for a, b in pairs:
        if a not in index or b not in index:
            continue
        ia, ib = index[a], index[b]
        va, vb = R[ia].copy(), R[ib].copy()
        for c in (own_col.get(a), own_col.get(b)):
            if c is not None:
                va[c] = 0.0
                vb[c] = 0.0
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na < 1e-9 or nb < 1e-9:
            continue
        out.append((a, b, float(va @ vb / (na * nb))))
    return pd.DataFrame(out, columns=["a", "b", "cosine"])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    X, targets, ps_genes = load_perturbseq()
    index = {}
    for i, t in enumerate(targets):
        index.setdefault(t, i)
    own_col = {}
    for j, g in enumerate(ps_genes):
        own_col.setdefault(g, j)

    log(f"\nTAU = {TAU} (the declared calibration point)\n")

    log("POSITIVE CONTROL: CORUM same-complex pairs")
    if not CORUM.exists():
        log("  CORUM control file missing; cannot run the gating control")
        return 1
    cp = pd.read_csv(CORUM)
    pos = cosine_for_pairs(list(zip(cp.a, cp.b)), X, own_col, index)
    pos.to_csv(OUT / "control_corum_cosine.csv", index=False)
    log(f"  pairs testable in Perturb-seq: {len(pos)} of {len(cp)}")
    if len(pos):
        log(f"  median cosine {pos.cosine.median():+.3f}   "
            f"frac > {TAU}: {(pos.cosine > TAU).mean():.3f}")
        psma = pos[(pos.a == 'PSMA1') & (pos.b == 'PSMB5')]
        if len(psma):
            log(f"  PSMA1-PSMB5: {float(psma.cosine.iloc[0]):+.3f}")

    log("\nNEGATIVE CONTROL: random pairs")
    if NEG.exists():
        ng = pd.read_csv(NEG)
        neg = cosine_for_pairs(list(zip(ng.a, ng.b)), X, own_col, index)
        neg.to_csv(OUT / "control_negative_cosine.csv", index=False)
        log(f"  pairs testable: {len(neg)}")
        if len(neg):
            log(f"  median cosine {neg.cosine.median():+.3f}   "
                f"frac > {TAU}: {(neg.cosine > TAU).mean():.4f}")

    log("\nGHOST: gene labels on the response matrix permuted")
    rng = np.random.default_rng(0)
    keys = list(index)
    perm = rng.permutation(len(keys))
    gindex = {k: index[keys[perm[i]]] for i, k in enumerate(keys)}
    gpos = cosine_for_pairs(list(zip(cp.a, cp.b)), X, own_col, gindex)
    if len(gpos):
        log(f"  CORUM pairs under permuted labels: median "
            f"{gpos.cosine.median():+.3f}   "
            f"frac > {TAU}: {(gpos.cosine > TAU).mean():.4f}")

    log("\nVERDICT")
    if len(pos) and len(neg):
        sep = (pos.cosine > TAU).mean() - (neg.cosine > TAU).mean()
        ok = (pos.cosine > TAU).mean() > 0.25 and sep > 0.2
        log(f"  positive {(pos.cosine > TAU).mean():.3f} vs negative "
            f"{(neg.cosine > TAU).mean():.4f}   separation {sep:+.3f}")
        log(f"  -> {'POSITIVE CONTROL PASSES' if ok else 'POSITIVE CONTROL FAILS - study VOID until the measure is fixed'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
