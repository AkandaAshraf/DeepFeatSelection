"""Download a time-slice of the ZAPBench trace matrix (zarr v3 over HTTPS).

Traces: shape (7879 time, 71721 neurons), float32, chunks (512, 512), public
GCS bucket, CC-BY 4.0, anonymous access verified. This grabs whole chunk rows
covering a frame window and assembles the exact slice to one .npy.

Default window: frames 5638-7879 -- conditions 7-9 (open loop, rotation,
dark), n=2241: inside the pipeline's validated range and containing the dark
condition, the NoStim analogue for the sensory-contrast readout.

    python scripts/zapbench_download.py
"""

from __future__ import annotations

import argparse
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

BASE = "https://storage.googleapis.com/zapbench-release/volumes/20240930/traces"
T_TOTAL, V_TOTAL = 7879, 71721
CHUNK = 512


def fetch_chunk(session, row: int, col: int) -> tuple[int, int, np.ndarray]:
    r = session.get(f"{BASE}/c/{row}/{col}", timeout=120)
    r.raise_for_status()
    # zarr v3 "bytes" codec, little-endian float32, C-order (512, 512);
    # edge chunks are stored full-size and sliced on assembly.
    arr = np.frombuffer(r.content, dtype="<f4").reshape(CHUNK, CHUNK)
    return row, col, arr


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=5638)
    p.add_argument("--stop", type=int, default=7879)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--out", default="Data/zapbench/traces_5638_7879.npy")
    args = p.parse_args()

    import truststore
    truststore.inject_into_ssl()
    import requests

    row_lo, row_hi = args.start // CHUNK, (args.stop - 1) // CHUNK
    col_hi = (V_TOTAL - 1) // CHUNK
    rows = range(row_lo, row_hi + 1)
    cols = range(col_hi + 1)
    total = len(rows) * len(cols)
    print(f"frames {args.start}-{args.stop}: chunk rows {row_lo}-{row_hi}, "
          f"{total} chunks (~{total} MB)")

    buf = np.empty(((row_hi - row_lo + 1) * CHUNK, V_TOTAL), dtype=np.float32)
    session = requests.Session()
    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = [pool.submit(fetch_chunk, session, r, c)
                   for r in rows for c in cols]
        for fut in as_completed(futures):
            row, col, arr = fut.result()
            r0 = (row - row_lo) * CHUNK
            c0 = col * CHUNK
            c1 = min(c0 + CHUNK, V_TOTAL)
            buf[r0:r0 + CHUNK, c0:c1] = arr[:, :c1 - c0]
            done += 1
            if done % 100 == 0:
                rate = done / (time.time() - t0)
                print(f"  {done}/{total} chunks "
                      f"({rate:.1f}/s, eta {(total-done)/rate:.0f}s)")

    lo = args.start - row_lo * CHUNK
    out = buf[lo:lo + (args.stop - args.start)]
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.save(dest, out)
    print(f"saved {out.shape} float32 -> {dest} "
          f"({dest.stat().st_size/1e6:.0f} MB in {time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
