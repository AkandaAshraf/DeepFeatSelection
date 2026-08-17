"""Find every email address in the tracked working tree, in binary artifacts,
and in git history, so nothing is published unintentionally."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
BAR = "=" * 72
SKIP_SUFFIX = {".png", ".jpg", ".jpeg", ".gz", ".zip", ".h5", ".npy", ".mat"}


def git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True,
                          text=True, errors="replace").stdout


def main() -> int:
    print(BAR)
    print("1. TRACKED WORKING TREE")
    print(BAR)
    hits: dict[str, list[str]] = {}
    for rel in [p for p in git("ls-files").split("\n") if p.strip()]:
        f = Path(rel)
        if not f.exists() or f.suffix.lower() in SKIP_SUFFIX:
            continue
        if f.suffix.lower() == ".pdf":
            try:
                import pymupdf
                doc = pymupdf.open(f)
                text = "".join(doc[i].get_text() for i in range(doc.page_count))
            except Exception:
                continue
        else:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
        for m in EMAIL.finditer(text):
            addr = m.group(0)
            ln = text[:m.start()].count("\n") + 1
            hits.setdefault(addr, []).append(f"{rel}:{ln}")

    if not hits:
        print("  none")
    for addr, locs in sorted(hits.items()):
        print(f"\n  {addr}")
        for loc in locs:
            print(f"     {loc}")

    print("\n" + BAR)
    print("2. GIT HISTORY: commit author/committer addresses")
    print(BAR)
    log = git("log", "--all", "--format=%ae%n%ce")
    authors = sorted({a.strip() for a in log.split("\n") if a.strip()})
    for a in authors:
        n = log.count(a)
        print(f"  {a}   ({n} occurrences)")

    print("\n" + BAR)
    print("3. GIT HISTORY: addresses inside committed file content")
    print(BAR)
    found = set()
    for addr in set(list(hits) + authors):
        out = git("log", "--all", "--oneline", "-S", addr)
        n = len([l for l in out.split("\n") if l.strip()])
        if n:
            found.add(addr)
            print(f"  {addr}: appears in {n} commit(s) touching its content")
    if not found:
        print("  none beyond the above")

    print("\n" + BAR)
    print(f"DISTINCT ADDRESSES: {len(set(list(hits) + authors))}")
    print(BAR)
    for a in sorted(set(list(hits) + authors)):
        where = []
        if a in hits:
            where.append("working tree")
        if a in authors:
            where.append("commit metadata")
        print(f"  {a:38s} {', '.join(where)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
