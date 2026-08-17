"""Assemble what bioRxiv's submission form actually asks for.

bioRxiv takes a manuscript PDF rather than LaTeX source, plus metadata typed
into a web form. This collects the PDF, the plain-text abstract and a checklist
of the form fields with the answers this paper implies, so nothing has to be
reconstructed at submission time.

    python scripts/make_biorxiv_bundle.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

PDF = Path("paper/excess_paper.pdf")
ABSTRACT = Path("paper/abstract_plain.txt")
OUT = Path("paper/biorxiv")

CHECKLIST = """bioRxiv submission checklist
============================

FILES IN THIS FOLDER
  excess_paper.pdf     the manuscript, upload this
  abstract_plain.txt   paste into the Abstract field
  checklist.txt        this file

SUBJECT CATEGORY
  Neuroscience
  (Alternative: Bioinformatics. Neuroscience is the better fit: the C. elegans
  and zebrafish deployments are the flagship validations and the interventional
  test is the strongest evidence in the paper.)

TITLE
  MACE: Masked-Autoencoder Conditional Excess for scalable, self-validating
  detection of driven variables in large dynamical systems

AUTHOR
  Akanda Wahid -Ul- Ashraf
  Use this exact spelling. It matches arXiv:1905.09087 and the Google Scholar
  profile; two other spellings exist in the record and adding a third would
  fragment it further. Link an ORCID here if you have one.

LICENCE
  CC BY 4.0. This matches the LICENSE file in the repository and the licence
  line printed on page 1 of the manuscript.

TYPE
  New Results.

COMPETING INTERESTS
  None.

FUNDING
  State none unless something applies. The manuscript notes all computation ran
  on one consumer laptop with no cluster, cloud budget or grant support.

DATA AND CODE AVAILABILITY
  Suggested wording:

    All datasets are public and require no registration: whole-brain calcium
    imaging of C. elegans (Kato et al. 2015), light-sheet imaging of larval
    zebrafish, the CHB-MIT scalp EEG database, and NCEP/NCAR reanalysis
    sea-level pressure. Code, pre-registration documents and a complete
    experiment ledger including all negative results and voided attempts are
    available at https://github.com/AkandaAshraf/DeepFeatSelection

  Push the repository before submitting. The abstract promises released code
  and the manuscript prints that URL, so the link must resolve on day one.

PRIOR POSTING
  This manuscript is not posted on any other preprint server. Do not also post
  it to arXiv: maintaining two canonical copies means every future correction
  has to be made twice, and they drift.

BEFORE YOU SUBMIT
  [ ] Repository pushed and the URL resolves
  [ ] Zenodo DOI minted for the code release, if you want a fixed citation
  [ ] The four outstanding literature checks resolved:
        - whether AVB appears in Kato 2015 Fig. 5B and what its bar shows
        - the AVA->AVB entry in the Randi/Leifer 2023 functional atlas
        - whether RIM is in Uzel et al. 2022's hub set
        - Ray & Gordus 2025 on AIB being driven by AVA
  [ ] Decide whether to cite your own 2016 tourism/greenhouse-gas causality
      paper as prior work on causal detection in real time series
"""


def main() -> int:
    if not PDF.exists():
        print(f"missing {PDF}; compile the paper first")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PDF, OUT / PDF.name)
    if ABSTRACT.exists():
        shutil.copy2(ABSTRACT, OUT / ABSTRACT.name)
    (OUT / "checklist.txt").write_text(CHECKLIST, encoding="utf-8")

    print(f"bioRxiv bundle: {OUT}/")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.stat().st_size/1024:8.0f} KB  {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
