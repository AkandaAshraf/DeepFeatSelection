"""Emit the abstract as clean plain text for a preprint-server submission form.

Extracting from the PDF loses superscripts and re-joins hyphenated line breaks
wrongly ("10^4" becomes "104", "masked-reconstruction" becomes
"maskedreconstruction"). Converting from the LaTeX source avoids both.

    python scripts/abstract_plaintext.py
"""

from __future__ import annotations

import re
from pathlib import Path

TEX = Path("paper/excess_paper.tex")
BS = chr(92)

# LaTeX construct -> plain text, applied in order.
RULES = [
    (re.escape(BS) + r"emph\{([^}]*)\}", r"\1"),
    (re.escape(BS) + r"textbf\{([^}]*)\}", r"\1"),
    (re.escape(BS) + r"citep\{[^}]*\}", ""),
    (re.escape(BS) + r"ref\{[^}]*\}", ""),
    (r"\$" + re.escape(BS) + r"approx\$\s*", "~"),
    (re.escape(BS) + r"approx", "~"),
    (r"\$10\^4\$", "10^4"),
    (r"\$n" + re.escape(BS) + r"approx2\{,\}000\$", "n ~ 2,000"),
    (r"\$([^$]*)\$", r"\1"),                 # strip remaining math delimiters
    (r"\{,\}", ","),                          # 71{,}721 -> 71,721
    (re.escape(BS) + r"%", "%"),
    (re.escape(BS) + r"[,;]", " "),           # thin spaces
    (re.escape(BS) + r"~", " "),
    (r"``", '"'),
    (r"''", '"'),
    (r"---", "-"),
    (r"--", "-"),
    (re.escape(BS) + r"[a-zA-Z]+", ""),       # any leftover command
    (r"[{}]", ""),
    (r"C\.~elegans", "C. elegans"),      # non-breaking space in the species name
    (r"n~2,000", "n = 2,000"),            # reads badly as a tilde
    (r"~", " approx. "),                  # any remaining approx sign
    (r"\s{2,}", " "),
]


def main() -> int:
    s = TEX.read_text(encoding="utf-8")
    a = s[s.index(BS + "begin{abstract}") + len(BS + "begin{abstract}"):
          s.index(BS + "end{abstract}")]
    for pat, rep in RULES:
        a = re.sub(pat, rep, a)
    a = re.sub(r"\s+", " ", a).strip()
    a = re.sub(r"\s+([.,;:])", r"\1", a)

    words = len(a.split())
    print(f"words: {words}   characters: {len(a)}")
    print("=" * 72)
    print(a)
    out = Path("paper/abstract_plain.txt")
    out.write_text(a + "\n", encoding="utf-8")
    print("=" * 72)
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
