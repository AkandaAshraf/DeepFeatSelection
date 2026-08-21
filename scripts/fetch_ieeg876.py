"""Fetch interictal SEEG subjects from OpenNeuro ds003876.

Downloads each subject's run-01 EDF, channels.tsv and sidecar JSON into
Data/ieeg876/ with absolute paths, resuming nothing (files are checked by
size and skipped when present). Channel tables are stored verbatim but are
only ever read by scripts that select the name and type columns; see
paper/ieeg_protocol.md.

    python scripts/fetch_ieeg876.py NIH3 NIH4 NIH5 NIH6
"""

from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "Data" / "ieeg876"
TASK = "interictal"          # overridden by --task for the replication
BASE = ("https://s3.amazonaws.com/openneuro.org/ds003876/sub-{s}/"
        "ses-extraoperative/ieeg/sub-{s}_ses-extraoperative_task-{task}_"
        "run-01_ieeg{ext}")
UA = {"User-Agent": "Mozilla/5.0"}


def fetch(url: str, dest: Path, min_size: int) -> str:
    if dest.exists() and dest.stat().st_size > min_size:
        return "skip"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r,                     open(dest, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if dest.stat().st_size > min_size:
                return "ok"
        except Exception as exc:
            print(f"  retry {attempt+1}: {exc}", flush=True)
            time.sleep(5)
    return "FAILED"


def main() -> int:
    args = sys.argv[1:]
    task = TASK
    if args and args[0].startswith("--task="):
        task = args.pop(0).split("=", 1)[1]
    subs = args or ["NIH3", "NIH4", "NIH5", "NIH6"]
    ROOT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for s in subs:
        for ext, name, floor in ((".edf", f"sub-{s}_{task}_run-01_ieeg.edf", 10_000_000),
                                 ("", "", 0)):
            if not name:
                continue
            print(s, fetch(BASE.format(s=s, task=task, ext=ext), ROOT / name, floor),
                  flush=True)
        for suffix, floor in (("channels.tsv", 500), ("ieeg.json", 100)):
            url = BASE.format(s=s, task=task, ext="").replace(
                "_ieeg", f"_{suffix.split('.')[0]}") \
                if suffix == "channels.tsv" else BASE.format(s=s, task=task, ext=".json")
            # channels.tsv URL differs from the EDF stem
            if suffix == "channels.tsv":
                url = ("https://s3.amazonaws.com/openneuro.org/ds003876/"
                       f"sub-{s}/ses-extraoperative/ieeg/sub-{s}_"
                       f"ses-extraoperative_task-{task}_run-01_channels.tsv")
            print(s, suffix,
                  fetch(url, ROOT / f"sub-{s}_{suffix}", floor), flush=True)
    print(f"done in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
