"""Redundancy audit on real CRISPR knockout data, scored against known biology.

Every other dataset in this repo is either simulated, where the ground truth is
ours to define, or real but without an independent answer to check against.
DepMap supplies both at once.  Its Chronos gene-effect matrix records how much
each cell line's growth depends on knocking out each gene, and members of the
same protein complex are co-essential -- knock out any subunit and the complex
fails the same way, so their essentiality profiles across cell lines are close
to interchangeable.

Complex membership is established biology and was not derived from this matrix,
so it is an outside standard.  If the audit's equivalence classes line up with
complexes, the instrument is recovering real structure rather than an artefact
of the encoding.

The practical reading is the same one the biomarker simulation gives, now on
real screens: if a ranking puts PSMA3 above PSMB6, that ordering is not a
statement about biology.  They are subunits of one machine, they are one
hypothesis, and which of them a particular analysis puts on top is decided by
noise in that dataset.

Data (~429 MB, gitignored):

    curl -L -o Data/depmap/CRISPRGeneEffect.csv \\
        https://ndownloader.figshare.com/files/51064667

    python scripts/depmap_audit.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from deepfeatselect.redundancy import equivalence_classes, redundancy_scores

# Complexes chosen for being unambiguous, well populated in the screens, and
# functionally unrelated to each other, so a class that merges two of them is a
# visible failure rather than a judgement call.
COMPLEXES: dict[str, list[str]] = {
    "proteasome_core": [
        "PSMA1", "PSMA2", "PSMA3", "PSMA4", "PSMA5", "PSMA6", "PSMA7",
        "PSMB1", "PSMB2", "PSMB3", "PSMB4", "PSMB5", "PSMB6", "PSMB7",
    ],
    "proteasome_lid": [
        "PSMD1", "PSMD2", "PSMD3", "PSMD4", "PSMD6", "PSMD7",
        "PSMD8", "PSMD11", "PSMD12", "PSMD13", "PSMD14",
    ],
    "exosome": [
        "EXOSC2", "EXOSC3", "EXOSC4", "EXOSC5", "EXOSC6", "EXOSC7",
        "EXOSC8", "EXOSC9", "EXOSC10",
    ],
    "coatomer_COPI": ["COPA", "COPB1", "COPB2", "COPG1", "COPZ1", "ARCN1"],
    "vATPase": [
        "ATP6V0B", "ATP6V0C", "ATP6V1A", "ATP6V1B2",
        "ATP6V1C1", "ATP6V1D", "ATP6V1E1", "ATP6V1F",
    ],
    "mediator": ["MED1", "MED4", "MED6", "MED7", "MED10", "MED11", "MED14", "MED30"],
}

# Housekeeping-ish genes that are broadly non-essential and belong to no shared
# machine: the negative control the audit must leave alone.
CONTROLS = [
    "OR2T1", "OR51E2", "KRT1", "KRT10", "DEFB1", "MYH7", "ACTN3",
    "CYP2D6", "GSTM1", "HBB", "TTN", "MUC16",
]


def load_matrix(path: Path, genes: list[str]) -> pd.DataFrame:
    """Load the gene-effect matrix and keep the requested gene columns.

    Columns are labelled ``SYMBOL (ENTREZ)``, so the symbol is parsed out. Rows
    are cell lines; a gene's column is its essentiality profile across them,
    which is the vector the audit compares.
    """
    frame = pd.read_csv(path, index_col=0)
    symbols = {c.split(" (")[0]: c for c in frame.columns}
    present = [g for g in genes if g in symbols]
    missing = sorted(set(genes) - set(present))
    if missing:
        print(f"  not in this release ({len(missing)}): {', '.join(missing)}")
    sub = frame[[symbols[g] for g in present]]
    sub.columns = present
    # Cell lines missing any gene in the panel are dropped rather than imputed:
    # imputation would manufacture exactly the redundancy under test.
    return sub.dropna()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="Data/depmap/CRISPRGeneEffect.csv")
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--outdir", default="ExpOutput/depmap")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    path = Path(args.data)
    if not path.exists():
        print(f"missing {path}\n  see the module docstring for the download command")
        return 1

    truth = {g: name for name, genes in COMPLEXES.items() for g in genes}
    truth.update({g: "control" for g in CONTROLS})
    panel = [g for genes in COMPLEXES.values() for g in genes] + CONTROLS

    print("=" * 78)
    print("DEPMAP CRISPR GENE EFFECT -- redundancy audit against known complexes")
    print("=" * 78)
    data = load_matrix(path, panel)
    print(f"  {data.shape[0]} cell lines x {data.shape[1]} genes")

    x = data.to_numpy(dtype=np.float64)
    names = list(data.columns)

    audit = redundancy_scores(x, names, seed=args.seed, threshold=args.threshold)
    audit["complex"] = audit.feature.map(truth)

    print("\nreconstructability from the other genes in the panel")
    print("-" * 78)
    by_complex = (audit.groupby("complex")
                  .agg(n=("feature", "size"),
                       mean_r2=("r2_from_others", "mean"),
                       max_r2=("r2_from_others", "max"),
                       n_redundant=("redundant", "sum"))
                  .sort_values("mean_r2", ascending=False))
    with pd.option_context("display.float_format", "{:+.4f}".format):
        print(by_complex.to_string())

    controls = audit[audit.complex == "control"]
    members = audit[audit.complex != "control"]
    print(f"\n  complex members : mean R^2 {members.r2_from_others.mean():+.4f}")
    print(f"  controls        : mean R^2 {controls.r2_from_others.mean():+.4f}")
    print(f"  separation      : {members.r2_from_others.mean() - controls.r2_from_others.mean():+.4f}")

    print("\nequivalence classes found, against known membership")
    print("-" * 78)
    classes = equivalence_classes(x, names, seed=args.seed, threshold=args.threshold)
    if not classes:
        print("  none at this threshold")
    rows = []
    for i, group in enumerate(classes, 1):
        labels = {truth.get(g, "?") for g in group}
        pure = len(labels) == 1
        label = next(iter(labels)) if pure else " + ".join(sorted(labels))
        print(f"  class {i}: {{{', '.join(sorted(group))}}}")
        print(f"           -> {label}  [{'pure' if pure else 'MIXED'}]")
        rows.append({"class": i, "size": len(group), "pure": pure,
                     "labels": label, "genes": " ".join(sorted(group))})

    if rows:
        pure_frac = sum(r["pure"] for r in rows) / len(rows)
        print(f"\n  {sum(r['pure'] for r in rows)}/{len(rows)} classes fall inside a "
              f"single complex ({pure_frac:.0%})")
        print("  A class spanning two complexes would be a false merge; a control")
        print("  appearing in any class would be a false positive.")
        in_class = {g for r in rows for g in r["genes"].split()}
        stray = sorted(in_class & set(CONTROLS))
        print(f"  controls captured: {len(stray)} {stray if stray else ''}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(outdir / "depmap_redundancy.csv", index=False)
    if rows:
        pd.DataFrame(rows).to_csv(outdir / "depmap_classes.csv", index=False)
    print(f"\nwrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
