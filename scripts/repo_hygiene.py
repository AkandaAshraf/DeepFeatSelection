"""Pre-publication hygiene check on everything git tracks.

Looks for four classes of problem before a repository goes public:
assistant/process leakage, superseded claims that the corrected paper no longer
makes, secrets and personal data, and files that simply should not ship.

    python scripts/repo_hygiene.py
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

BAR = "=" * 74
TEXT_EXT = {".py", ".md", ".tex", ".txt", ".json", ".yml", ".yaml", ".cfg",
            ".toml", ".ini", ".sh", ".bib", ""}

# (label, regex, why it matters)
PATTERNS = [
    ("assistant/process leakage",
     r"\b(claude|anthropic|chatgpt|openai|copilot|gpt-4|llm-generated|"
     r"as an ai\b|language model|prompt|subagent|conversation)\b",
     "reveals how the work was produced"),
    ("superseded CCM cost",
     r"31[,{]?[,}]?700|3\.6 years",
     "the corrected figure is ~15,800 h / 1.8 years"),
    ("withdrawn per-cell claim",
     r"(neuron|cell)[-\s]by[-\s](neuron|cell)",
     "withdrawn; only VB01 survives multiplicity correction"),
    ("superseded four-systems framing",
     r"four\s+(\w+\s+)?systems",
     "three distinct graphs plus one re-analysis"),
    ("superseded in-degree claim",
     r"1\.63 against|population mean of 1\.20",
     "the effect is null, p = 0.79"),
    ("credential or key",
     r"(api[_-]?key|secret[_-]?key|password|passwd|token\s*=|"
     r"BEGIN [A-Z ]*PRIVATE KEY|ghp_[A-Za-z0-9]{20,})",
     "credential material"),
    ("email address",
     r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
     "personal contact data"),
    ("absolute local path",
     r"[A-Za-z]:[\\/]Users[\\/]|/home/[a-z]+/",
     "leaks the author's filesystem"),
]


def tracked() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
    return [Path(p) for p in out.stdout.split("\n") if p.strip()]


def main() -> int:
    files = tracked()
    print(f"{len(files)} tracked files\n")

    findings: dict[str, list[tuple[str, int, str]]] = {}
    for f in files:
        if f.suffix.lower() not in TEXT_EXT or not f.exists():
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for label, pat, _ in PATTERNS:
            for m in re.finditer(pat, text, re.I):
                ln = text[:m.start()].count("\n") + 1
                snippet = text.split("\n")[ln - 1].strip()[:88]
                findings.setdefault(label, []).append((str(f), ln, snippet))

    for label, pat, why in PATTERNS:
        hits = findings.get(label, [])
        print(BAR)
        print(f"{label.upper()}  ({len(hits)} hits) - {why}")
        print(BAR)
        if not hits:
            print("  clean")
        else:
            seen = set()
            for path, ln, snip in hits[:14]:
                key = (path, snip)
                if key in seen:
                    continue
                seen.add(key)
                print(f"  {path}:{ln}: {snip}")
            if len(hits) > 14:
                print(f"  ... and {len(hits)-14} more")
        print()

    print(BAR)
    print("LARGE TRACKED FILES (>1 MB)")
    print(BAR)
    big = [(f, f.stat().st_size) for f in files if f.exists()
           and f.stat().st_size > 1_000_000]
    for f, sz in sorted(big, key=lambda x: -x[1]):
        print(f"  {sz/1e6:6.1f} MB  {f}")
    if not big:
        print("  none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
