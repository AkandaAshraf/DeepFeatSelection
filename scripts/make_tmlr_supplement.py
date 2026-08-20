"""Build the anonymised supplementary archive for double-blind review.

TMLR permits up to 100 MB of supplementary material. The paper promises
reviewers the code, the pre-registrations and the full experiment ledger, so
this packages exactly those, with every author identifier scrubbed: a
supplement that names the author would deanonymise the submission and the
paper would be rejected without review.

Datasets are not included. They are public and every script downloads its
own; redistributing them would blow the size limit and is not ours to do.

    python scripts/make_tmlr_supplement.py
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "tmlr" / "supplementary.zip"

INCLUDE_DIRS = ["scripts", "mace", "deepfeatselect", "tests"]
INCLUDE_FILES = [
    "FINDINGS.md",
    "README.md",
    "pyproject.toml",
    "paper/causal_detection_log.md",
    "paper/validation_protocol.md",
    "paper/depmap_protocol.md",
    "paper/ieeg_protocol.md",
]
SKIP_SUFFIX = {".pyc", ".pdf", ".zip", ".log"}
# Publication-logistics scripts: not research, and they necessarily contain
# the identifiers this archive exists to remove.
SKIP_NAMES = {
    "make_tmlr_supplement.py", "make_biorxiv_bundle.py",
    "make_arxiv_package.py", "find_emails.py", "repo_hygiene.py",
    "abstract_plaintext.py",
}
SKIP_PARTS = {"__pycache__", ".git", ".venv", "ExpOutput", "Data"}

# Identifiers that must not survive into the archive. Order matters: longer
# strings first so the shorter ones cannot partially eat them.
SCRUB = [
    ("https://github.com/AkandaAshraf/DeepFeatSelection",
     "[repository URL withheld for review]"),
    ("github.com/AkandaAshraf/DeepFeatSelection",
     "[repository URL withheld for review]"),
    ("https://doi.org/10.5281/zenodo.21988145", "[DOI withheld for review]"),
    ("10.5281/zenodo.21988145", "[DOI withheld for review]"),
    ("akandaashraf@outlook.com", "[email withheld for review]"),
    ("Akanda Wahid-Ul-Ashraf", "[author withheld for review]"),
    ("Akanda Wahid -Ul- Ashraf", "[author withheld for review]"),
    ("Akanda Ashraf", "[author withheld for review]"),
    ("AkandaAshraf", "[author withheld for review]"),
    ("Akanda", "[author withheld for review]"),
    ("Ashraf", "[author withheld for review]"),
]
# Case-insensitive residue check after scrubbing.
FORBIDDEN = re.compile(
    r"akanda|ashraf|outlook\.com|zenodo\.21988145|bournemouth|orcid",
    re.IGNORECASE)

TEXT_SUFFIX = {".py", ".md", ".toml", ".txt", ".cff", ".yml", ".yaml", ".sh"}


def wanted(p: Path) -> bool:
    if p.suffix in SKIP_SUFFIX or p.name in SKIP_NAMES:
        return False
    return not (SKIP_PARTS & set(p.parts))


def scrub(text: str) -> str:
    for old, new in SCRUB:
        text = text.replace(old, new)
    return text


def main() -> int:
    files: list[Path] = []
    for d in INCLUDE_DIRS:
        base = ROOT / d
        if base.exists():
            files += [p for p in base.rglob("*") if p.is_file() and wanted(p)]
    for f in INCLUDE_FILES:
        p = ROOT / f
        if p.exists():
            files.append(p)

    leaks: list[str] = []
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            rel = p.relative_to(ROOT).as_posix()
            if p.suffix in TEXT_SUFFIX:
                try:
                    body = scrub(p.read_text(encoding="utf-8"))
                except UnicodeDecodeError:
                    z.write(p, rel)
                    continue
                hit = FORBIDDEN.search(body)
                if hit:
                    line = body[:hit.start()].count("\n") + 1
                    leaks.append(f"{rel}:{line}  {hit.group(0)!r}")
                z.writestr(rel, body)
            else:
                z.write(p, rel)

        z.writestr("README_SUPPLEMENT.txt",
                   "Supplementary material for an anonymous TMLR submission.\n\n"
                   "Contents\n"
                   "  scripts/    every experiment in the paper, one script per result\n"
                   "  mace/       the method as an installable package; scan() returns\n"
                   "              scores only together with their controls\n"
                   "  tests/      unit tests, CPU-only\n"
                   "  paper/      the pre-registration documents and the complete\n"
                   "              experiment ledger, including negative results and\n"
                   "              entries voided after they failed scrutiny\n"
                   "  FINDINGS.md plain-language summary of every result\n\n"
                   "Datasets are not included: all are public and each script\n"
                   "downloads its own. Author identifiers have been removed for\n"
                   "double-blind review.\n")

    print(f"wrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB, {len(files)} files)")
    if leaks:
        print("\nIDENTIFIER LEAKS -- fix before submitting:")
        for line in leaks:
            print("  " + line)
        return 1
    print("anonymity check: no author identifiers found in any text file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
