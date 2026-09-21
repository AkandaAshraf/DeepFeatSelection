"""Interruption, identity, integrity and lock checks on engineering seeds."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import parent_screening_laptop as L


def check():
    os.chdir(L.ROOT)
    with tempfile.TemporaryDirectory(prefix="resume_", dir=L.ROOT / "ExpOutput") as tmp:
        base = Path(tmp)
        resumed, reference = base / "resumed", base / "reference"
        seeds = {"family1": (9930,), "family2": (9940,)}
        def command(out):
            return [sys.executable, "-u", "-c",
                    "import sys;sys.path.insert(0,'scripts');"
                    "import parent_screening_laptop as L;"
                    f"L.run(L.Path({str(out)!r}),{seeds!r},16,1200,True)"]
        with (base / "interrupted.log").open("w") as log:
            child = subprocess.Popen(command(resumed), stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 120
                target = resumed / "family1_9930_LASSO.json"
                while not target.exists():
                    if child.poll() is not None or time.monotonic() > deadline:
                        raise AssertionError("Child failed before interruption checkpoint")
                    time.sleep(.05)
                # A second process cannot enter even if it is the same runner.
                try:
                    with L.RunLock():
                        raise AssertionError("Concurrent launch accepted")
                except OSError:
                    pass
                child.kill()
                child.wait()
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()
        before = target.read_bytes(), target.stat().st_mtime_ns
        store = L.Store(resumed, L.identity(seeds, 16, 1200))
        assert store.get("runtime") > 0
        # A torn temporary write must not invalidate a committed checkpoint.
        target.with_name(target.name + ".tmp").write_text("{torn")
        with (base / "resumed.log").open("w") as log:
            subprocess.run(command(resumed), stdout=log, stderr=log, check=True, timeout=180)
        assert before == (target.read_bytes(), target.stat().st_mtime_ns)
        assert "RESUMED family1_9930_LASSO" in (base / "resumed.log").read_text()
        with (base / "reference.log").open("w") as log:
            subprocess.run(command(reference), stdout=log, stderr=log, check=True, timeout=180)
        ref = L.Store(reference, L.identity(seeds, 16, 1200))
        for fam, values in seeds.items():
            for seed in values:
                key = f"shortlists_{fam}_{seed}"
                assert store.get(key) == ref.get(key), key
        assert json.loads((resumed / "pilot_metrics.json").read_text()) == json.loads(
            (reference / "pilot_metrics.json").read_text())
        try:
            L.Store(resumed, {"wrong": True})
            raise AssertionError("Changed configuration accepted")
        except RuntimeError:
            pass
        original = target.read_text()
        damaged = json.loads(original)
        damaged["sha256"] = "0" * 64
        target.write_text(json.dumps(damaged))
        try:
            store.get("family1_9930_LASSO")
            raise AssertionError("Corrupt checkpoint accepted")
        except RuntimeError:
            pass
        target.write_text(original)
        budget = L.Store(base / "budget", {"test": True})
        budget.put("runtime", 60.)
        guard = L.DurableGuard(budget, cap=60.)
        try:
            guard.check_light("exhausted")
            raise AssertionError("Runtime cap reset on restart")
        except L.A.CapBreach:
            pass
        print("PASS: hard kill/resume; identical shortlists/metrics; skip completed arms; "
              "torn temporary file; identity/checksum refusal; concurrent lock; persistent runtime")


if __name__ == "__main__":
    check()
