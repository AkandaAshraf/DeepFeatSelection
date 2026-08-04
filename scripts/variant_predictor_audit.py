"""Are in-silico variant predictors independent evidence? Audited on ClinVar.

Clinical variant classification (ACMG/AMP criteria PP3 and BP4) counts computational
evidence toward a pathogenicity call, and a curator choosing between REVEL, CADD,
SIFT, PolyPhen and their neighbours may reasonably assume that agreement between
several of them strengthens the case.  For many of these scores that assumption is
arithmetically false: REVEL is *computed from* eighteen scores across thirteen
tools, SIFT and PolyPhen among them.  Counting REVEL and SIFT as separate support
counts SIFT twice.

That is the derivation case, which is where Proposition 1 actually bites.  A
companion audit on DepMap CRISPR screens found protein-complex subunits
reconstructing each other at only R^2 ~ 0.2-0.33: biologically the same machine,
but separately measured, so the redundancy is partial and importance rankings
remain identifiable.  Predictor scores are different in kind -- one is a
deterministic function of the others -- so the prediction is that they land in the
non-identifiable band and the panel supplies far fewer independent pieces of
evidence than it has columns.

Data comes from the MyVariant.info REST API rather than the 50 GB dbNSFP download,
so this runs from a laptop with no registration.

    python scripts/variant_predictor_audit.py --n-variants 4000
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from deepfeatselect.redundancy import equivalence_classes, redundancy_scores

API = "https://myvariant.info/v1/query"

# dbNSFP fields, grouped by what they are built from. The ensembles are the point:
# each is a trained combination of scores that also appear in this table.
COMPONENT_SCORES = {
    "sift": "dbnsfp.sift.score",
    "polyphen_hdiv": "dbnsfp.polyphen2.hdiv.score",
    "polyphen_hvar": "dbnsfp.polyphen2.hvar.score",
    "mutationtaster": "dbnsfp.mutationtaster.score",
    "mutationassessor": "dbnsfp.mutationassessor.score",
    "fathmm": "dbnsfp.fathmm.score",
    "provean": "dbnsfp.provean.score",
    "vest4": "dbnsfp.vest4.score",
    "lrt": "dbnsfp.lrt.score",
}
CONSERVATION_SCORES = {
    "gerp": "dbnsfp.gerp++.rs",
    "phylop_vert": "dbnsfp.phylop.100way_vertebrate.score",
    "phylop_mammal": "dbnsfp.phylop.470way_mammalian.score",
    "phastcons_vert": "dbnsfp.phastcons.100way_vertebrate.score",
    "phastcons_mammal": "dbnsfp.phastcons.470way_mammalian.score",
    "siphy": "dbnsfp.siphy_29way.logodds_score",
}
ENSEMBLE_SCORES = {
    "revel": "dbnsfp.revel.score",      # built from 13 tools, incl. most of the above
    # CADD is served as a top-level annotation, not under dbnsfp.
    "cadd": "cadd.phred",
    "metasvm": "dbnsfp.metasvm.score",
    "metalr": "dbnsfp.metalr.score",
    "mcap": "dbnsfp.m-cap.score",
    "dann": "dbnsfp.dann.score",
    "fathmm_mkl": "dbnsfp.fathmm-mkl.coding_score",
    "eigen": "dbnsfp.eigen.raw_coding",
}
ALL_SCORES = {**COMPONENT_SCORES, **CONSERVATION_SCORES, **ENSEMBLE_SCORES}
GROUP_OF = ({k: "component" for k in COMPONENT_SCORES}
            | {k: "conservation" for k in CONSERVATION_SCORES}
            | {k: "ensemble" for k in ENSEMBLE_SCORES})


def _first(value):
    """dbNSFP returns lists where a variant has several transcripts."""
    if isinstance(value, list):
        value = value[0] if value else None
    return value if isinstance(value, (int, float)) else None


def _dig(doc: dict, path: str):
    node = doc
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return _first(node)


def fetch(query: str, size: int, cache: Path, pause: float = 0.3) -> pd.DataFrame:
    """Page through MyVariant.info, keeping one row per variant.

    Cached to disk because the interesting part is the audit, not re-downloading
    the same variants while iterating on it.
    """
    if cache.exists():
        print(f"  using cached {cache}")
        return pd.read_csv(cache)

    # Whole objects rather than individual paths: a field name containing "++"
    # (gerp++) does not survive the fields parameter, and silently returns
    # nothing rather than erroring.
    fields = "dbnsfp,cadd,clinvar.rcv.clinical_significance"
    # Elasticsearch refuses from+size beyond 10000, so that is the ceiling for
    # this pagination style. It is far more than the audit needs.
    ES_WINDOW = 10_000
    rows, offset, page = [], 0, 1000
    while len(rows) < size and offset < ES_WINDOW:
        params = urllib.parse.urlencode({
            "q": query, "fields": fields,
            "size": min(page, ES_WINDOW - offset), "from": offset,
        })
        try:
            with urllib.request.urlopen(f"{API}?{params}", timeout=120) as response:
                hits = json.load(response).get("hits", [])
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"  request failed at offset {offset}: {exc}")
            break
        if not hits:
            break
        for hit in hits:
            row = {name: _dig(hit, path) for name, path in ALL_SCORES.items()}
            sig = _dig_significance(hit)
            if sig is not None:
                row["label"] = sig
                rows.append(row)
        offset += len(hits)
        print(f"  fetched {offset}, kept {len(rows)}")
        time.sleep(pause)

    frame = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)
    return frame


def _dig_significance(hit: dict) -> int | None:
    """Collapse ClinVar significance to pathogenic (1) / benign (0), else drop."""
    node = hit.get("clinvar", {}).get("rcv")
    if node is None:
        return None
    entries = node if isinstance(node, list) else [node]
    labels = {str(e.get("clinical_significance", "")).lower() for e in entries}
    text = " ".join(labels)
    pathogenic = "pathogenic" in text and "conflicting" not in text
    benign = "benign" in text and "conflicting" not in text
    if pathogenic and not benign:
        return 1
    if benign and not pathogenic:
        return 0
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-variants", type=int, default=4000)
    p.add_argument("--query", default="_exists_:dbnsfp.revel AND _exists_:clinvar",
                   help="MyVariant.info query; the default keeps variants that have "
                        "both an ensemble score and a ClinVar record")
    p.add_argument("--cache", default="Data/variants/myvariant_sample.csv")
    p.add_argument("--min-coverage", type=float, default=0.80,
                   help="drop predictors present in fewer than this fraction of variants")
    p.add_argument("--outdir", default="ExpOutput/variants")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("=" * 78)
    print("IN-SILICO VARIANT PREDICTORS -- how much independent evidence?")
    print("=" * 78)
    frame = fetch(args.query, args.n_variants, Path(args.cache))
    if frame.empty:
        print("no variants retrieved; check connectivity or the query")
        return 1

    # Drop sparsely-populated predictors before dropping rows: requiring all 23
    # at once would discard every variant if a single column were unavailable,
    # which is what happens when a field path is wrong.
    candidates = [c for c in ALL_SCORES if c in frame.columns]
    coverage = frame[candidates].notna().mean()
    scores = [c for c in candidates if coverage[c] >= args.min_coverage]
    dropped = [f"{c} ({coverage[c]:.0%})" for c in candidates if c not in scores]
    if dropped:
        print(f"  dropped for low coverage: {', '.join(dropped)}")

    frame = frame.dropna(subset=scores)
    print(f"\n  {len(frame)} variants x {len(scores)} predictors "
          f"({frame.label.mean():.1%} pathogenic)")
    if len(frame) < 200:
        print("  too few complete rows for a stable audit")
        return 1

    x = frame[scores].to_numpy(dtype=np.float64)
    audit = redundancy_scores(x, scores, seed=args.seed)
    audit["kind"] = audit.feature.map(GROUP_OF)

    print("\nreconstructability of each predictor from the others")
    print("-" * 78)
    with pd.option_context("display.float_format", "{:+.4f}".format):
        print(audit[["feature", "kind", "r2_from_others", "verdict"]].to_string(index=False))

    print("\nby kind")
    print("-" * 78)
    print(audit.groupby("kind").r2_from_others.agg(["size", "mean", "max"]).to_string())

    counts = audit.verdict.value_counts()
    print("\nverdicts")
    for verdict in ("not_identifiable", "partially_redundant", "identifiable"):
        print(f"  {verdict:<22} {counts.get(verdict, 0)}")

    classes = equivalence_classes(x, scores, seed=args.seed)
    print(f"\npairwise-interchangeable groups: {len(classes)}")
    for group in classes:
        kinds = sorted({GROUP_OF[g] for g in group})
        print(f"  {{{', '.join(sorted(group))}}}  [{', '.join(kinds)}]")
    if not classes:
        print("  none -- and that is informative rather than reassuring. Equivalence")
        print("  classes test whether two predictors reconstruct *each other*. Here the")
        print("  redundancy is many-to-one: MetaSVM is recoverable from the panel as a")
        print("  whole, not from any single partner, which is what a trained ensemble")
        print("  of the other columns looks like. Counting classes would report no")
        print("  problem while most of the table is non-identifiable.")

    # Effective dimensionality instead: the participation ratio of the correlation
    # spectrum, which counts directions actually occupied rather than columns.
    corr = np.corrcoef(x, rowvar=False)
    eigenvalues = np.clip(np.linalg.eigvalsh(corr), 0.0, None)
    participation = eigenvalues.sum() ** 2 / (eigenvalues**2).sum()
    ordered = np.sort(eigenvalues)[::-1]
    n_95 = int(np.searchsorted(np.cumsum(ordered) / ordered.sum(), 0.95) + 1)

    print(f"\nhow much independent evidence is really here?")
    print(f"  predictor columns                       {len(scores)}")
    print(f"  effective dimensions (participation)    {participation:.1f}")
    print(f"  components for 95% of variance          {n_95}")
    print("  Counting an ensemble score alongside its own inputs double-counts")
    print("  those inputs, which is the ACMG PP3/BP4 concern this quantifies.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(outdir / "variant_predictor_audit.csv", index=False)
    print(f"\nwrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
