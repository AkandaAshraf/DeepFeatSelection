"""Durable, explicitly resumed execution of the extended laptop protocol."""
import argparse
import base64
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import pickle
import subprocess
import time

import numpy as np
import parent_screening_arms as A

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ExpOutput/parent_screening_laptop"
SEEDS = {"family1": tuple(range(24001, 24007)),
         "family2": tuple(range(25001, 25007))}
CAP = 20 * 3600
FILES = (*A.GUARDED_FILES, "scripts/parent_screening_laptop.py",
         "scripts/resume_parent_screening.ps1",
         "paper/parent_screening_laptop_protocol.md")


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, default=lambda x: x.item())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Store:
    def __init__(self, path, identity):
        self.path, self.identity = Path(path), identity
        self.path.mkdir(parents=True, exist_ok=True)
        p = self.path / "identity.json"
        if p.exists():
            if json.loads(p.read_text()) != identity:
                raise RuntimeError("Checkpoint code/config identity mismatch")
        else:
            if any(self.path.glob("*.json")):
                raise RuntimeError("Existing output has no identity")
            atomic(p, identity)

    def get(self, name):
        p = self.path / (name + ".json")
        if not p.exists():
            return None
        envelope = json.loads(p.read_text())
        blob = base64.b64decode(envelope["data"], validate=True)
        if hashlib.sha256(blob).hexdigest() != envelope["sha256"]:
            raise RuntimeError("Checkpoint checksum mismatch: " + name)
        # Only this local runner's checksummed checkpoint files are accepted.
        return pickle.loads(blob)

    def put(self, name, value):
        blob = pickle.dumps(value, protocol=5)
        atomic(self.path / (name + ".json"), {
            "sha256": hashlib.sha256(blob).hexdigest(),
            "data": base64.b64encode(blob).decode("ascii")})

    def checkpoint(self, prefix, guard):
        def call(name, fn, timings):
            key = prefix + "_" + name
            saved = self.get(key)
            if saved is not None:
                print("RESUMED " + key, flush=True)
                result, elapsed = saved
            else:
                guard.check_light(key)
                t0 = time.perf_counter()
                print("START " + key, flush=True)
                result = fn()
                elapsed = time.perf_counter() - t0
                self.put(key, (result, elapsed))
                guard.check_full(key)
                print(f"SAVED {key} {elapsed:.1f}s", flush=True)
            timings[name] = elapsed
            return result
        return call


class DurableGuard(A.ResourceGuard):
    def __init__(self, store, cap=CAP):
        self.store = store
        self.prior = store.get("runtime") or 0.0
        self.reserved = self.prior
        self.total_cap = cap
        super().__init__(cap - self.prior, out_dir=store.path)

    def check_light(self, where):
        elapsed = self.prior + self.elapsed()
        if elapsed > self.reserved:
            self.reserved = min(self.total_cap, elapsed + 60)
            self.store.put("runtime", self.reserved)
        super().check_light(where)

    def finish(self):
        self.store.put("runtime", min(self.total_cap, self.prior + self.elapsed()))


class RunLock:
    """OS lock is released on power loss; foreign repository locks are refused."""
    def __enter__(self):
        p = ROOT / "ExpOutput/parent_screening_laptop.lock"
        p.parent.mkdir(exist_ok=True)
        self.handle = p.open("a+b")
        self.handle.seek(0)
        if not self.handle.read(1):
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            lock = ROOT / ".agent-lock"
            if lock.exists():
                data = json.loads(lock.read_text())
                if data.get("owner") != "parent_screening_laptop":
                    raise RuntimeError("Foreign training lock; refusing launch")
                # Every owner of this marker must hold the OS lock above.
                lock.unlink()
            marker = p.with_name("parent_screening_laptop.owner.json")
            atomic(marker, {"owner": "parent_screening_laptop", "pid": os.getpid()})
            # Publish a fully flushed marker without overwriting a racing owner.
            os.link(marker, lock)
            return self
        except BaseException:
            self.handle.close()
            raise

    def __exit__(self, *args):
        (ROOT / ".agent-lock").unlink(missing_ok=True)
        self.handle.close()


def identity(seeds, V, n):
    return {"seeds": {k: list(v) for k, v in seeds.items()}, "V": V, "n": n,
            "cap": CAP, "files": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                                   for f in FILES}}


def run(out=OUT, seeds=SEEDS, V=240, n=4000, engineering=False):
    os.chdir(ROOT)
    if not engineering:
        ok, info = A.clearance_state(guarded=FILES)
        if not ok:
            raise RuntimeError("Launch refused: " + info)
    with RunLock():
        store = Store(out, identity(seeds, V, n))
        status = store.get("status")
        if status and status["state"] in ("complete", "resource-limited"):
            print(json.dumps(status), flush=True)
            return status
        guard = DurableGuard(store)
        try:
            guard.check_full("start")
            guard.check_light("start")
            for fam, gen in (("family1", A.PS.family1_generate),
                             ("family2", A.PS.family2_generate)):
                for seed in seeds[fam]:
                    key = f"{fam}_{seed}"
                    if store.get("done_" + key) is not None:
                        print("RESUMED system " + key, flush=True)
                        continue
                    guard.check_light("generate " + key)
                    data = gen(V, n, seed)
                    timings = {}
                    result, _, _ = A.run_all_arms(
                        data["x_obs"], seed, guard, timings,
                        checkpoint=store.checkpoint(key, guard))
                    store.put("shortlists_" + key, A._payload(result, V))
                    store.put("evaluator_" + key, data["parent"])
                    store.put("done_" + key, {"timings": timings})
                    print("SAVED system " + key, flush=True)
            metrics = {f: {a: [] for a in A.ARMS} for f in seeds}
            for fam in seeds:
                for seed in seeds[fam]:
                    key = f"{fam}_{seed}"
                    truth = store.get("evaluator_" + key)
                    payload = store.get("shortlists_" + key)
                    for arm in A.ARMS:
                        C, unresolved = A._restore(payload[arm], V)
                        m = A.arm_metrics(C, unresolved, truth, V, A.PS.k_for(V))
                        m["seed"] = seed
                        metrics[fam][arm].append(m)
            atomic(Path(out) / "pilot_metrics.json", metrics)
            passed = A.evaluate_gate(metrics, seeds=seeds)
            status = {"state": "complete", "gate_passed": bool(passed),
                      "engineering": engineering}
            store.put("status", status)
            return status
        except A.CapBreach as exc:
            status = {"state": "resource-limited", "reason": str(exc)}
            store.put("status", status)
            print(json.dumps(status), flush=True)
            return status
        finally:
            guard.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engineering", action="store_true")
    args = parser.parse_args()
    if args.engineering:
        run(ROOT / "ExpOutput/parent_screening_resume_test",
            {"family1": (9930,), "family2": (9940,)}, 16, 1200, True)
    else:
        run()
