"""Repair of the 2026-09-06 hierarchy result: three defects, one script.

Pre-registration: paper/hierarchy_repair_protocol.md, committed before this
was written. BOUNDED DIAGNOSTIC on the ORIGINAL 12 cells and ORIGINAL seeds
-- not a reopening of the rejected architecture, not a confirmatory study.
No verdict word (adopt/reject/close/width/repaired) appears in this script's
output; see the protocol's "Language" section for what is licensed instead.

Three repairs, five arms:

  FLAT               unchanged incumbent. Fidelity-checked against the
                     archived run (raw array AND summary, not summary alone).
  HIER-CLUST-TRAIN   repair 1: module clustering fit on TRAINING RAW ROWS
                     ONLY (the same numeric cutoff the encoder's own train
                     slice uses), never on validation or test rows.
  HIER-RAND-BAL      the archived control, unchanged: balanced random
                     modules, equal sizes.
  HIER-RAND-SIZED    repair 2's control: random modules whose SIZE VECTOR
                     matches HIER-CLUST-TRAIN's in that cell (built from the
                     new train-only clustering, not the archive), targets
                     assigned to those sizes at random.
  HIER-TRUE          unchanged oracle from the true parent map.

Repair 3 (statistics) is not a code change to the readout; it is how results
are aggregated in `report()`: within-cell correlations only, no pooling
across modules or cells, no p-value from three seeds.

    python scripts/hierarchy_repair.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (adjusted_rand_score, average_precision_score,
                             roc_auc_score)
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from boundary_map import (BATCH, DEV, E, EPOCHS, MASK, embed,  # noqa: E402
                          make_system, poly3, ridge_r2)
from wormwideweb_gate import MaskedAE  # noqa: E402

ARCHIVE = Path("ExpOutput/hierarchy")           # read-only, never written
OUT = Path("ExpOutput/hierarchy_repair")        # new directory
N, COUPLING = 4000, 0.20
WIDTHS = (30, 60)
NOISES = (0.0, 0.05)
SEEDS = (0, 1, 2)
MOD_EPOCHS = 12
FLAT_AP_TOL = 0.03
FLAT_ARR_TOL = 1e-6
SHARE_DENOM_EPS = 1e-5
LOCK = ".agent-lock"

ARMS = ["FLAT", "HIER-CLUST-TRAIN", "HIER-RAND-BAL", "HIER-RAND-SIZED",
       "HIER-TRUE"]


# --------------------------------------------------------------- shared

class FlatGuardError(RuntimeError):
    """FLAT fidelity guard failed. Raised BEFORE any module training so no
    GPU time is spent on comparisons the protocol already voids."""


def _persist_guard(guard: dict) -> None:
    """Append the guard diagnostic immediately, so a failure is on disk even
    though the run stops right after."""
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "flat_guard.csv"
    df = pd.DataFrame([guard])
    df.to_csv(p, mode="a", header=not p.exists(), index=False)


def train_code(zs, tr, d_in, b, seed, epochs=EPOCHS):
    """One masked autoencoder, returning the net and full-length codes.
    Identical to the archived implementation -- nothing about the encoder
    changes in this repair."""
    v = d_in // E
    torch.manual_seed(seed)
    net = MaskedAE(d_in, b).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(seed)
    ztr = torch.as_tensor(zs[tr], device=DEV)
    for _ in range(epochs):
        perm = torch.randperm(ztr.shape[0], generator=g)
        for i in range(0, len(perm), BATCH):
            bt = ztr[perm[i:i + BATCH]]
            msk = torch.rand(bt.shape[0], v, device=DEV) < MASK
            mc = msk.repeat_interleave(E, dim=1)
            loss = ((net(bt.masked_fill(mc, 0.0)) - bt)[mc] ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return net, net.enc(torch.as_tensor(zs, device=DEV)).cpu().numpy()


# ------------------------------------------------------- module assignment

def cluster_train_only(x_raw: np.ndarray, raw_cutoff: int, m: int) -> np.ndarray:
    """Repair 1. Correlate diffs of x_raw[:raw_cutoff] ONLY -- validation and
    test raw rows never enter this computation, unlike the archived
    `modules_for`, which correlated over the full recording."""
    d = np.diff(x_raw[:raw_cutoff], axis=0)
    c = np.nan_to_num(np.corrcoef(d.T), nan=0.0)
    return AgglomerativeClustering(
        n_clusters=m, metric="precomputed", linkage="average"
    ).fit_predict(1.0 - np.abs(c))


def balanced_random(Vt: int, m: int, rng: np.random.Generator) -> np.ndarray:
    lab = np.arange(Vt) % m
    rng.shuffle(lab)
    return lab


def sized_random(sizes: np.ndarray, Vt: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Repair 2's control. A random partition whose size MULTISET matches
    `sizes` exactly: which targets get which size, and which size lands in
    which module id, are both randomised -- only the width distribution is
    matched, never a specific target's width (stated in the protocol)."""
    order = rng.permutation(len(sizes))
    perm = rng.permutation(Vt)
    lab = np.empty(Vt, int)
    pos = 0
    for new_id, orig_id in enumerate(order):
        s = int(sizes[orig_id])
        lab[perm[pos:pos + s]] = new_id
        pos += s
    return lab


def true_modules(Vt: int, m: int, parent: np.ndarray, n_src: int) -> np.ndarray:
    lab = np.empty(Vt, int)
    lab[:n_src] = np.arange(n_src) % m
    lab[n_src:] = [lab[p] for p in parent]
    return lab


def size_stats(lab: np.ndarray, Vt: int, is_driven: np.ndarray) -> dict:
    sizes = np.bincount(lab, minlength=int(lab.max()) + 1)
    per_target = sizes[lab]
    return dict(
        unweighted_mean=float(sizes.mean()),
        target_weighted=float((sizes.astype(float) ** 2).sum() / Vt),
        driven_weighted=float(per_target[is_driven].mean()),
        sizes=sizes)


# ---------------------------------------------------------- module readout

def module_readout(lab, zs, own, lead, base, tr, tr_i, te_i, sys_code, Vt,
                   seed_tag):
    """Given a partition `lab`, train one small encoder per module (module
    code, target's own columns zeroed) and compute e2/e3 for every target.
    Identical arithmetic to the archived script; only which partition is
    passed in differs by arm."""
    m = int(lab.max()) + 1
    e2 = np.zeros(Vt)
    e3 = np.zeros(Vt)
    widths = np.zeros(m, dtype=int)
    for mod in range(m):
        members = np.where(lab == mod)[0]
        if len(members) == 0:
            continue
        b = max(4, 2 * len(members))            # the archived formula
        widths[mod] = b
        cols = np.concatenate([np.arange(j * E, (j + 1) * E)
                               for j in members])
        zsm = zs[:, cols]
        net, _ = train_code(zsm, tr, zsm.shape[1], b,
                            seed_tag * 1000 + mod, epochs=MOD_EPOCHS)
        for q in members:
            zq = zsm.copy()                     # zero q before its module
            loc = int(np.where(members == q)[0][0])
            zq[:, loc * E:(loc + 1) * E] = 0.0
            with torch.no_grad():
                cq = net.enc(torch.as_tensor(zq, device=DEV)).cpu().numpy()
            r_om = ridge_r2(
                np.hstack([own[q][tr_i], cq[tr_i]]), lead[tr_i + 1, q],
                np.hstack([own[q][te_i], cq[te_i]]), lead[te_i + 1, q])
            r_oms = ridge_r2(
                np.hstack([own[q][tr_i], cq[tr_i], sys_code[tr_i]]),
                lead[tr_i + 1, q],
                np.hstack([own[q][te_i], cq[te_i], sys_code[te_i]]),
                lead[te_i + 1, q])
            e2[q] = r_om - base[q]
            e3[q] = r_oms - r_om
    return e2, e3, widths


# ------------------------------------------------------ train-only guard

def invariance_test(V: int = 30, noise: float = 0.0, seed: int = 0,
                    n_trials: int = 3) -> tuple[bool, list[dict]]:
    """Perturb every raw value AT OR AFTER the train cutoff with independent
    noise, recompute HIER-CLUST-TRAIN's clustering, and require the
    resulting partition to be IDENTICAL (ARI = 1.0) to the unperturbed run's.
    Runs on CPU only (AgglomerativeClustering), no torch/GPU touched.
    Repeated with several perturbation draws, not just one, since a single
    lucky draw passing would be weak evidence."""
    x, _, _ = make_system(N, V, COUPLING, 0, seed)
    if noise:
        x = x + noise * np.random.default_rng(seed + 777).standard_normal(
            x.shape)
    Vt = x.shape[1]
    m = max(2, V // 6)
    mrows = embed(x).shape[0]
    raw_cutoff = int(0.6 * mrows)

    base_lab = cluster_train_only(x, raw_cutoff, m)
    results = []
    all_ok = True
    for trial in range(n_trials):
        xp = x.copy()
        rng = np.random.default_rng(9000 + trial)
        xp[raw_cutoff:] += rng.standard_normal(xp[raw_cutoff:].shape) * 5.0
        pert_lab = cluster_train_only(xp, raw_cutoff, m)
        ari = float(adjusted_rand_score(base_lab, pert_lab))
        ok = ari == 1.0
        all_ok &= ok
        results.append(dict(trial=trial, ari=ari, ok=ok))
    return all_ok, results


# ------------------------------------------------------------------ cell

def cell(V: int, noise: float, seed: int) -> tuple[list[dict], dict]:
    x, is_driven, is_source = make_system(N, V, COUPLING, 0, seed)
    Vt = x.shape[1]
    n_src = max(3, V // 6)
    rng_g = np.random.default_rng(seed)
    _ = rng_g.uniform(0.2, 0.8, V)
    _ = rng_g.uniform(3.6, 3.9, V)
    parent = rng_g.integers(0, n_src, V - n_src)
    if noise:
        x = x + noise * np.random.default_rng(seed + 777).standard_normal(
            x.shape)

    emb = embed(x)
    mrows = emb.shape[0]
    a, bnd = int(0.6 * mrows), int(0.8 * mrows)
    raw_cutoff = a                     # the protocol's raw/train cutoff
    tr = slice(0, a)
    tr_i, te_i = np.arange(0, a - 1), np.arange(bnd, mrows - 1)
    mu, sd = emb[tr].mean(0), emb[tr].std(0) + 1e-12
    zs = np.clip(np.nan_to_num((emb - mu) / sd), -20, 20).astype(np.float32)
    lead = zs[:, [j * E for j in range(Vt)]]
    own = [poly3(zs[:, q * E:(q + 1) * E]) for q in range(Vt)]
    base = np.array([ridge_r2(own[q][tr_i], lead[tr_i + 1, q],
                              own[q][te_i], lead[te_i + 1, q])
                     for q in range(Vt)])

    t0 = time.time()
    _, sys_code = train_code(zs, tr, zs.shape[1], 2 * V, seed * 100)
    t_sys = time.time() - t0

    flat = np.array([
        ridge_r2(np.hstack([own[q][tr_i], sys_code[tr_i]]), lead[tr_i + 1, q],
                 np.hstack([own[q][te_i], sys_code[te_i]]), lead[te_i + 1, q])
        - base[q] for q in range(Vt)])

    # ---- FLAT FIDELITY GUARD against the archived run, same cell.
    # FAILS CLOSED: any of missing archive, shape mismatch, label mismatch,
    # nonfinite scores, AP mismatch or array mismatch raises FlatGuardError,
    # which stops this cell BEFORE any module training and makes the run
    # exit nonzero. Continuing past a failed fidelity check would spend GPU
    # time producing comparisons that are void by the protocol anyway.
    guard = dict(V=V, noise=noise, seed=seed, archive_found=False,
                ap_diff=np.nan, arr_max_abs_diff=np.nan,
                ap_ok=False, arr_ok=False, reason="")
    arch_path = ARCHIVE / f"raw_V{V}_nz{noise}_s{seed}.npz"

    def _fail(reason: str):
        guard["reason"] = reason
        _persist_guard(guard)
        raise FlatGuardError(f"V={V} noise={noise} seed={seed}: {reason}")

    if not np.all(np.isfinite(flat)):
        _fail(f"nonfinite FLAT scores: {int((~np.isfinite(flat)).sum())} of "
              f"{flat.size}")
    if not arch_path.exists():
        _fail(f"archived cell missing: {arch_path}")

    za = np.load(arch_path)
    for key in ("FLAT", "is_source", "is_driven"):
        if key not in za:
            _fail(f"archived cell lacks '{key}'")
    arch_flat = za["FLAT"]
    if arch_flat.shape != flat.shape:
        _fail(f"shape mismatch: archived {arch_flat.shape} vs new "
              f"{flat.shape}")
    if not np.array_equal(za["is_source"], is_source):
        _fail("label mismatch: is_source differs from the archived cell")
    if not np.array_equal(za["is_driven"], is_driven):
        _fail("label mismatch: is_driven differs from the archived cell")
    if not np.all(np.isfinite(arch_flat)):
        _fail("nonfinite scores in the archived FLAT array")

    arch_ap = float(average_precision_score(za["is_source"], -arch_flat))
    new_ap = float(average_precision_score(is_source, -flat))
    arr_diff = float(np.max(np.abs(flat - arch_flat)))
    ap_diff = abs(new_ap - arch_ap)
    guard.update(archive_found=True, ap_diff=ap_diff,
                 arr_max_abs_diff=arr_diff,
                 ap_ok=bool(ap_diff <= FLAT_AP_TOL),
                 arr_ok=bool(arr_diff <= FLAT_ARR_TOL))
    if not guard["ap_ok"]:
        _fail(f"AP mismatch {ap_diff:.5f} > tol {FLAT_AP_TOL}")
    if not guard["arr_ok"]:
        _fail(f"array mismatch {arr_diff:.3e} > tol {FLAT_ARR_TOL:.0e}")
    _persist_guard(guard)

    m = max(2, V // 6)
    rows = [dict(V=V, noise=noise, seed=seed, arm="FLAT",
                 ap_source=float(average_precision_score(is_source, -flat)),
                 ap_driven=float(average_precision_score(is_driven, flat)),
                 mod_share=np.nan, mean_e2_driven=np.nan,
                 mean_e3_driven=np.nan, share_excluded=False, secs=t_sys)]
    raw = {"FLAT": flat, "is_driven": is_driven, "is_source": is_source,
          "parent": parent, "raw_cutoff": raw_cutoff}
    size_rows = []
    graded_rows = []       # per-module rows for the within-cell Spearman

    # HIER-CLUST-TRAIN first: HIER-RAND-SIZED needs its size vector
    lab_clust = cluster_train_only(x, raw_cutoff, m)
    e2c, e3c, wc = module_readout(lab_clust, zs, own, lead, base, tr, tr_i,
                                  te_i, sys_code, Vt, seed)
    clust_sizes = np.bincount(lab_clust, minlength=m)

    partitions = {
        "HIER-CLUST-TRAIN": (lab_clust, e2c, e3c, wc),
    }
    t0 = time.time()
    rng = np.random.default_rng(seed + 31)
    lab_bal = balanced_random(Vt, m, rng)
    e2b, e3b, wb = module_readout(lab_bal, zs, own, lead, base, tr, tr_i,
                                  te_i, sys_code, Vt, seed)
    partitions["HIER-RAND-BAL"] = (lab_bal, e2b, e3b, wb)

    lab_sz = sized_random(clust_sizes, Vt, np.random.default_rng(seed + 41))
    e2s, e3s, ws = module_readout(lab_sz, zs, own, lead, base, tr, tr_i,
                                  te_i, sys_code, Vt, seed)
    partitions["HIER-RAND-SIZED"] = (lab_sz, e2s, e3s, ws)

    lab_true = true_modules(Vt, m, parent, n_src)
    e2t, e3t, wt = module_readout(lab_true, zs, own, lead, base, tr, tr_i,
                                  te_i, sys_code, Vt, seed)
    partitions["HIER-TRUE"] = (lab_true, e2t, e3t, wt)
    secs_hier = time.time() - t0

    for arm, (lab, e2, e3, widths) in partitions.items():
        tot2 = float(e2[is_driven].mean())
        tot3 = float(e3[is_driven].mean())
        denom = tot2 + tot3
        share_excluded = abs(denom) < SHARE_DENOM_EPS
        mod_share = np.nan if share_excluded else float(tot2 / denom)
        total = e2 + e3
        rows.append(dict(
            V=V, noise=noise, seed=seed, arm=arm,
            ap_source=float(average_precision_score(is_source, -total)),
            ap_driven=float(average_precision_score(is_driven, total)),
            mod_share=mod_share, mean_e2_driven=tot2, mean_e3_driven=tot3,
            share_excluded=share_excluded, secs=secs_hier / 4))
        sz = size_stats(lab, Vt, is_driven)
        size_rows.append(dict(V=V, noise=noise, seed=seed, arm=arm,
                              unweighted_mean=sz["unweighted_mean"],
                              target_weighted=sz["target_weighted"],
                              driven_weighted=sz["driven_weighted"],
                              n_modules=len(sz["sizes"]),
                              min_width=int(widths[widths > 0].min())
                              if (widths > 0).any() else 0,
                              max_width=int(widths.max())))
        raw[arm + "_e2"] = e2
        raw[arm + "_e3"] = e3
        raw[arm + "_lab"] = lab
        raw[arm + "_widths"] = widths

        # per-module rows for the GRADED within-cell correlation
        for mod in range(int(lab.max()) + 1):
            members = np.where(lab == mod)[0]
            drv = members[is_driven[members]]
            if len(drv) == 0:
                continue
            # drv holds DRIVEN indices only (is_driven is false for every
            # source), so every q in drv satisfies q >= n_src and
            # parent[q - n_src] is always a valid lookup.
            frac_in = float(np.mean(
                [lab[parent[q - n_src]] == mod for q in drv]))
            m_tot2, m_tot3 = float(e2[drv].mean()), float(e3[drv].mean())
            m_denom = m_tot2 + m_tot3
            excluded = abs(m_denom) < SHARE_DENOM_EPS or len(drv) < 2
            m_share = np.nan if excluded else float(m_tot2 / m_denom)
            graded_rows.append(dict(
                V=V, noise=noise, seed=seed, arm=arm, module=mod,
                n_driven=len(drv), frac_parent_in=frac_in, share=m_share,
                mod_e2=m_tot2, mod_e3=m_tot3, mod_denom=m_denom,
                excluded=excluded))

    # D2: ARI between archived HIER-CLUST (transductive) and the new
    # HIER-CLUST-TRAIN partition, descriptive only
    d2_ari = np.nan
    if arch_path.exists():
        za = np.load(arch_path)
        if "HIER-CLUST_lab" in za:
            d2_ari = float(adjusted_rand_score(za["HIER-CLUST_lab"], lab_clust))

    np.savez_compressed(OUT / f"raw_V{V}_nz{noise}_s{seed}.npz", **raw)
    return rows, dict(guard=guard, sizes=size_rows, graded=graded_rows,
                      d2_ari=d2_ari)


# ------------------------------------------------------------------ main

def run_invariance_precheck() -> bool:
    print("TRAIN-ONLY INVARIANCE PRECHECK (CPU only, run before the main "
         "script)")
    ok, trials = invariance_test()
    for t in trials:
        print(f"   trial {t['trial']}: ARI={t['ari']:.6f}  "
             f"{'PASS' if t['ok'] else 'FAIL'}")
    print(f"   -> {'ALL PASS' if ok else 'FAILURE: clustering is not '
         'train-only'}\n")
    return ok


def _proc_alive(pid: int) -> bool:
    """Liveness, FAILING CLOSED. Anything that is not a definitive 'this pid
    does not exist' is reported as alive: on Windows OpenProcess can fail
    with ACCESS_DENIED for a live process owned by another user, and
    treating that as dead would let a second run seize a held lock."""
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True                       # exists, not ours
        except OSError:
            return True                       # unknown -> assume alive
    import ctypes
    k = ctypes.windll.kernel32
    ERROR_INVALID_PARAMETER = 87              # the only "no such pid" answer
    h = k.OpenProcess(0x1000, False, pid)     # PROCESS_QUERY_LIMITED_INFO
    if not h:
        return k.GetLastError() != ERROR_INVALID_PARAMETER
    try:
        code = ctypes.c_ulong()
        if k.GetExitCodeProcess(h, ctypes.byref(code)):
            return code.value == 259          # STILL_ACTIVE
        return True
    finally:
        k.CloseHandle(h)


_LOCK_TOKEN: str | None = None                # ours, in memory only


def acquire_lock(what: str = "hierarchy_repair") -> bool:
    """Acquire .agent-lock atomically, or fail closed.

    `what` names the run in the lock file so a reader knows WHICH script
    holds it; callers importing this from another script must pass their
    own name, or the lock misreports what is live.

    Never unlinks a lock this process does not own. An earlier version
    removed a lock whose pid looked dead, which races: two contenders can
    both see it as stale, and the second unlink destroys the first's freshly
    created lock. A lock held by anyone else -- live, dead, or unreadable --
    stops the run and is left for a human to clear deliberately.

    Ownership is a random token held in memory AND written to the file, so
    release can prove the file is still the one we created; a pid alone is
    not enough, since pids are reused.
    """
    global _LOCK_TOKEN
    token = uuid.uuid4().hex
    payload = json.dumps({"pid": os.getpid(), "token": token,
                          "what": what, "started": time.time()})
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(payload)
        _LOCK_TOKEN = token
        return True
    except FileExistsError:
        pass

    try:
        info = json.loads(Path(LOCK).read_text())
    except (OSError, ValueError):
        print(f"PREFLIGHT: {LOCK} exists and is unreadable. Not starting; "
             f"clear it deliberately once you know no run is live.")
        return False

    if _LOCK_TOKEN is not None and info.get("token") == _LOCK_TOKEN:
        return True                                    # already ours

    pid = int(info.get("pid", -1))
    alive = _proc_alive(pid)
    age = (time.time() - float(info.get("started", 0))) / 60
    state = "LIVE" if alive else "not detectably alive"
    print(f"PREFLIGHT: {LOCK} held by pid {pid} ({state}, "
         f"{info.get('what')}, {age:.0f} min). NOT starting and NOT "
         f"removing it: a lock is only ever cleared deliberately, because "
         f"auto-recovery races another contender doing the same.")
    return False


def release_lock() -> None:
    """Release only if the file still carries OUR token."""
    global _LOCK_TOKEN
    if _LOCK_TOKEN is None:
        return
    try:
        info = json.loads(Path(LOCK).read_text())
        if info.get("token") == _LOCK_TOKEN:
            os.unlink(LOCK)
    except (OSError, ValueError):
        pass
    finally:
        _LOCK_TOKEN = None


# ------------------------------------------------------- per-cell bundles

def bundle_path(V, noise, seed) -> Path:
    return OUT / f"cell_V{V}_nz{noise}_s{seed}.json"


def save_bundle(V, noise, seed, rows, extra) -> None:
    """Persist EVERYTHING this cell produced, atomically. Written last, after
    the npz, so a cell counts as complete only if both exist and this
    validates -- a crash mid-write cannot leave a half-valid cell."""
    payload = {"V": V, "noise": noise, "seed": seed, "rows": rows,
               "guard": extra["guard"], "sizes": extra["sizes"],
               "graded": extra["graded"], "d2_ari": extra["d2_ari"]}
    tmp = bundle_path(V, noise, seed).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, default=float))
    os.replace(tmp, bundle_path(V, noise, seed))    # atomic


def _quarantine(p: Path, why: str) -> None:
    """Move an invalid artifact aside instead of deleting or trusting it, so
    the evidence survives for inspection and the cell is recomputed."""
    if not p.exists():
        return
    qdir = OUT / "quarantine"
    qdir.mkdir(parents=True, exist_ok=True)
    dest = qdir / f"{p.name}.{int(time.time())}"
    try:
        os.replace(p, dest)
        print(f"   QUARANTINED {p.name} -> {dest.relative_to(OUT)}: {why}")
    except OSError as exc:
        print(f"   could not quarantine {p.name}: {exc}")


def load_bundle(V, noise, seed):
    """Return a bundle only if it is complete AND self-consistent.

    Checked: the JSON parses; it names the cell actually requested; its rows
    carry exactly the expected arm set with no duplicates; the guard is
    present, finite and PASSING; the size and D2 tables exist; and the
    companion NPZ is readable with the arrays and shapes a finished cell
    must have. Anything else is quarantined and the cell recomputed --
    a partially written cell must never be mistaken for a finished one.
    """
    bp = bundle_path(V, noise, seed)
    npz = OUT / f"raw_V{V}_nz{noise}_s{seed}.npz"
    if not bp.exists():
        return None
    tag = f"V{V} nz{noise} s{seed}"

    def bad(why):
        print(f"   bundle {tag} rejected: {why}")
        _quarantine(bp, why)
        _quarantine(npz, "companion of an invalid bundle")
        return None

    try:
        b = json.loads(bp.read_text())
    except (OSError, ValueError) as exc:
        return bad(f"unreadable JSON ({exc})")

    for k in ("rows", "guard", "sizes", "graded", "d2_ari", "V", "noise",
              "seed"):
        if k not in b:
            return bad(f"missing key '{k}'")
    if (b["V"], float(b["noise"]), b["seed"]) != (V, float(noise), seed):
        return bad(f"wrong cell: bundle says V{b['V']} nz{b['noise']} "
                   f"s{b['seed']}")

    arms = [r.get("arm") for r in b["rows"]]
    if len(arms) != len(set(arms)):
        return bad(f"duplicate arm rows: {arms}")
    if set(arms) != set(ARMS):
        return bad(f"arm set mismatch: {sorted(set(arms))}")

    g = b["guard"]
    if not isinstance(g, dict) or not g.get("archive_found"):
        return bad("guard missing or archive_found false")
    if not (g.get("ap_ok") and g.get("arr_ok")):
        return bad("guard did not pass (ap_ok/arr_ok false)")
    for k in ("ap_diff", "arr_max_abs_diff"):
        v = g.get(k)
        if v is None or not np.isfinite(float(v)):
            return bad(f"guard field '{k}' is not finite: {v}")

    if not b["sizes"]:
        return bad("empty size table")
    if b["d2_ari"] is None:
        return bad("missing d2_ari")

    if not npz.exists():
        return bad("companion NPZ missing")
    # Validate inside the context, but report OUTSIDE it: quarantining while
    # the file is still open fails on Windows (the handle blocks the move).
    npz_problem = None
    try:
        with np.load(npz) as z:
            need = ["FLAT", "is_driven", "is_source", "parent", "raw_cutoff"]
            need += [f"{a}_lab" for a in ARMS if a != "FLAT"]
            missing = [k for k in need if k not in z]
            if missing:
                npz_problem = f"NPZ lacks {missing}"
            else:
                n = z["FLAT"].shape[0]
                if (z["is_driven"].shape[0] != n
                        or z["is_source"].shape[0] != n):
                    npz_problem = "NPZ label arrays disagree with FLAT length"
                elif not np.all(np.isfinite(z["FLAT"])):
                    npz_problem = "NPZ FLAT contains nonfinite values"
                else:
                    for a in ARMS:
                        if a == "FLAT":
                            continue
                        if z[f"{a}_lab"].shape[0] != n:
                            npz_problem = (f"NPZ '{a}_lab' length "
                                           f"{z[f'{a}_lab'].shape[0]} != {n}")
                            break
    except Exception as exc:                        # noqa: BLE001
        npz_problem = f"unreadable NPZ ({exc})"
    if npz_problem:
        return bad(npz_problem)

    return b


def preflight() -> bool:
    """Reboot safety. This machine has crashed mid-run, so before spending
    GPU time: report validated completed cells (they are neither recomputed
    nor overwritten) and acquire the lock atomically."""
    OUT.mkdir(parents=True, exist_ok=True)
    done = [(V, nz, s) for V in WIDTHS for nz in NOISES for s in SEEDS
            if load_bundle(V, nz, s) is not None]
    orphan = [p for p in OUT.glob("raw_V*_nz*_s*.npz")
              if not (OUT / p.name.replace("raw_", "cell_")
                      .replace(".npz", ".json")).exists()]
    if done:
        print(f"PREFLIGHT: {len(done)} of "
             f"{len(WIDTHS) * len(NOISES) * len(SEEDS)} cells already "
             f"complete and validated; they are skipped, not recomputed.")
    if orphan:
        print(f"PREFLIGHT: {len(orphan)} npz without a valid bundle "
             f"(interrupted mid-cell); those cells WILL be recomputed:")
        for p in orphan:
            print(f"   {p.name}")
    return acquire_lock()


def _run() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    if not preflight():
        return 1

    if not run_invariance_precheck():
        print("VOID per protocol: the train-only invariance test failed, "
             "so defect 1 is not repaired. Not proceeding.")
        return 1

    print(f"device: {DEV}   BOUNDED DIAGNOSTIC, original 12 cells, "
         f"original seeds\n")
    recs, guards, size_recs, graded_recs, d2_recs = [], [], [], [], []
    t0 = time.time()
    for V in WIDTHS:
        for nz in NOISES:
            for s in SEEDS:
                cached = load_bundle(V, nz, s)
                if cached is not None:
                    # Load it back into the aggregates. Skipping without
                    # loading is what made resume lossy: the CSVs were then
                    # rewritten from only the newly computed cells.
                    recs += cached["rows"]
                    guards.append(cached["guard"])
                    size_recs += cached["sizes"]
                    graded_recs += cached["graded"]
                    d2_recs.append(dict(V=V, noise=nz, seed=s,
                                        d2_ari=cached["d2_ari"]))
                    print(f"  V={V} noise={nz:<5} seed={s}   RESUMED "
                         f"(validated bundle, no retraining)", flush=True)
                    continue
                try:
                    r, extra = cell(V, nz, s)
                except FlatGuardError as exc:
                    print(f"\nFLAT FIDELITY GUARD FAILED: {exc}")
                    print("Stopped before module training. Diagnostic "
                         "persisted to flat_guard.csv. Per the protocol the "
                         "comparison is void until this is understood.")
                    return 2
                recs += r
                guards.append(extra["guard"])
                size_recs += extra["sizes"]
                graded_recs += extra["graded"]
                d2_recs.append(dict(V=V, noise=nz, seed=s,
                                    d2_ari=extra["d2_ari"]))
                save_bundle(V, nz, s, r, extra)
                g = extra["guard"]
                print(f"  V={V} noise={nz:<5} seed={s}   "
                     f"FLAT-guard[ap_ok={g['ap_ok']} arr_ok={g['arr_ok']}]"
                     f"   ({(time.time()-t0)/60:.1f}m)", flush=True)
                pd.DataFrame(recs).to_csv(OUT / "cells.csv", index=False)
                # flat_guard.csv is the append-only record written by
                # _persist_guard (including failures); this is the summary.
                pd.DataFrame(guards).to_csv(
                    OUT / "flat_guard_summary.csv", index=False)
                pd.DataFrame(size_recs).to_csv(OUT / "sizes.csv", index=False)
                pd.DataFrame(graded_recs).to_csv(OUT / "graded.csv",
                                                index=False)
                pd.DataFrame(d2_recs).to_csv(OUT / "d2_ari.csv", index=False)

    if not recs:
        print("\nNo cell results available: nothing was computed and no "
             "validated bundle was found. Not writing empty summaries.")
        return 1
    d = pd.DataFrame(recs)
    gdf = pd.DataFrame(guards)
    print(f"\n({(time.time()-t0)/60:.1f} min)\n")
    print(f"cells in summary: {len(d) // len(ARMS)} "
         f"(resumed + newly computed)")

    # Summaries are rebuilt from EVERY completed cell, resumed ones
    # included, so a restart cannot silently drop earlier rows.
    pd.DataFrame(recs).to_csv(OUT / "cells.csv", index=False)
    # Rebuilt here too, so an all-complete resume (which computes no cell
    # and so never enters the per-cell writer) still refreshes it.
    pd.DataFrame(guards).to_csv(OUT / "flat_guard_summary.csv", index=False)
    pd.DataFrame(size_recs).to_csv(OUT / "sizes.csv", index=False)
    pd.DataFrame(graded_recs).to_csv(OUT / "graded.csv", index=False)
    pd.DataFrame(d2_recs).to_csv(OUT / "d2_ari.csv", index=False)

    print("FLAT FIDELITY GUARD (against the archived run, same 12 cells)")
    found = gdf[gdf.archive_found]
    if len(found):
        print(f"   archive found for {len(found)}/{len(gdf)} cells")
        print(f"   AP guard:  {'ALL PASS' if found.ap_ok.all() else 'FAILURES PRESENT'}"
             f"  (max |diff| {found.ap_diff.max():.4f}, tol {FLAT_AP_TOL})")
        print(f"   ARR guard: {'ALL PASS' if found.arr_ok.all() else 'FAILURES PRESENT'}"
             f"  (max |diff| {found.arr_max_abs_diff.max():.2e}, tol "
             f"{FLAT_ARR_TOL:.0e})")
        if not (found.ap_ok.all() and found.arr_ok.all()):
            print("   VOID per protocol: FLAT fidelity guard failed. "
                 "Comparisons below are not meaningful until this is found.")
    else:
        print("   no archived cells found to compare against")

    print("\nDETECTION: average precision, sources positive (chance 0.167)")
    print("   " + d.pivot_table(index=["V", "noise"], columns="arm",
                                values="ap_source")[ARMS].round(3)
         .to_string().replace("\n", "\n   "))

    sdf = pd.DataFrame(size_recs)
    print("\nTARGET-WEIGHTED MODULE SIZE (first-class table, per Rule 131)")
    print("   " + sdf.groupby("arm")[["unweighted_mean", "target_weighted",
                                      "driven_weighted"]].mean()
         .round(3).to_string().replace("\n", "\n   "))
    print("\nACTUAL MODULE BOTTLENECK WIDTHS (min/max seen, per arm)")
    print("   " + sdf.groupby("arm")[["min_width", "max_width"]].agg(
        ["min", "max"]).to_string().replace("\n", "\n   "))

    h = d[d.arm != "FLAT"]
    print("\nMODULE SHARE (unbounded ratio; NaN rows are predefined "
         "exclusions)")
    # Count the EXCLUSION FLAG and the share itself, not mean_e2_driven:
    # e2 can be finite while the share is undefined (near-zero denominator),
    # so counting e2's NaNs would report zero exclusions on exactly the
    # cells the predefined rule removes.
    excl = h.groupby("arm").apply(
        lambda g: int((g.share_excluded | g.mod_share.isna()).sum()),
        include_groups=False)
    print(f"   share-undefined cells excluded, by arm: {excl.to_dict()}")
    share_piv = h.pivot_table(index=["V", "noise"], columns="arm",
                              values="mod_share")
    print("   " + (share_piv.round(3).to_string().replace("\n", "\n   ")
                   if len(share_piv) else "no defined shares to tabulate"))

    print("\nDESCRIPTIVE COMPARISON: does clustering after matching the "
         "aggregate size distribution attenuate, persist, or reverse "
         "against the size-matched random control?")
    piv = h.pivot_table(index=["V", "noise", "seed"], columns="arm",
                        values="mod_share")
    need = ["HIER-CLUST-TRAIN", "HIER-RAND-SIZED"]
    missing = [a for a in need if a not in piv.columns]
    if missing:
        print(f"   cannot compare: no defined shares for {missing}")
    else:
        comparable = piv.dropna(subset=need)
        if not len(comparable):
            print("   cannot compare: zero cells have both arms defined")
        else:
            persists = int((comparable[need[0]] > comparable[need[1]]).sum())
            reverses = int((comparable[need[0]] < comparable[need[1]]).sum())
            ties = len(comparable) - persists - reverses
            print(f"   HIER-CLUST-TRAIN > HIER-RAND-SIZED in {persists} of "
                 f"{len(comparable)} comparable cells (persists)")
            print(f"   HIER-CLUST-TRAIN < HIER-RAND-SIZED in {reverses} of "
                 f"{len(comparable)} comparable cells (reverses)")
            if ties:
                print(f"   exactly equal in {ties} cells")
    print("   No pooled test statistic, no adopt/reject/close/repaired "
         "wording -- descriptive counts only, per the protocol's Language "
         "section.")

    gr = pd.DataFrame(graded_recs)
    gr_ok = gr[~gr.excluded].dropna(subset=["frac_parent_in", "share"])
    print(f"\nGRADED (within cell, per-module exclusions: "
         f"{int(gr.excluded.sum())} of {len(gr)} modules)")

    # PER-CELL Spearman persisted with an explicit status, exclusion counts
    # and the module inputs, rather than only a printed seed median: a
    # constant-input or too-few-modules cell must be visible as such, not
    # silently absent from an aggregate.
    per_cell = []
    for (arm, V, nz, seed), gg_all in gr.groupby(["arm", "V", "noise",
                                                  "seed"]):
        gg = gg_all[~gg_all.excluded].dropna(
            subset=["frac_parent_in", "share"])
        n_excl = int(gg_all.excluded.sum()) + int(
            gg_all[~gg_all.excluded][["frac_parent_in", "share"]]
            .isna().any(axis=1).sum())
        rho, status = np.nan, ""
        if len(gg) < 3:
            status = f"too_few_modules({len(gg)})"
        elif gg.frac_parent_in.nunique() < 2:
            status = "constant_frac_parent_in"
        elif gg.share.nunique() < 2:
            status = "constant_share"
        else:
            r, _ = spearmanr(gg.frac_parent_in, gg.share)
            if np.isfinite(r):
                rho, status = float(r), "ok"
            else:
                status = "nonfinite_rho"
        per_cell.append(dict(arm=arm, V=V, noise=nz, seed=seed, rho=rho,
                             status=status, n_modules_used=len(gg),
                             n_modules_total=len(gg_all),
                             n_excluded=n_excl))
    pcdf = pd.DataFrame(per_cell)
    pcdf.to_csv(OUT / "graded_per_cell.csv", index=False)
    print("   per-cell status counts: "
         f"{pcdf.status.value_counts().to_dict()}")

    ok_cells = pcdf[pcdf.status == "ok"]
    if len(ok_cells):
        by_seed = (ok_cells.groupby(["arm", "seed"]).rho.median()
                   .reset_index())
        print("   median within-cell rho, by seed (cells with status 'ok' "
             "only):")
        print("   " + by_seed.pivot_table(index="seed", columns="arm",
                                          values="rho").round(3)
             .to_string().replace("\n", "\n   "))
    else:
        print("   no cell reached status 'ok'; nothing to summarise")
    print("   No pooled p-value across modules or seeds, per Rule 131 / "
         "repair 3. Full per-cell detail in graded_per_cell.csv; raw "
         "module e2/e3 in graded.csv.")

    d2 = pd.DataFrame(d2_recs)
    print("\nD2 (descriptive): ARI between archived transductive partition "
         "and the new train-only one, per cell")
    print(f"   median {d2.d2_ari.median():.3f}   "
         f"range [{d2.d2_ari.min():.3f}, {d2.d2_ari.max():.3f}]")

    print("\nNo adopt/reject/close/width-as-verdict/repaired language "
         "follows. This diagnostic supports only the descriptive "
         "statements above.")
    return 0




def main() -> int:
    """Wrapper so the lock is released on every exit path, including an
    exception -- a crash must not leave a lock that blocks the next run."""
    try:
        return _run()
    finally:
        release_lock()


if __name__ == "__main__":
    raise SystemExit(main())
