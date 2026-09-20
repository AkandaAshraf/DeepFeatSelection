"""Bounded-code parent screening at a fixed candidate budget.

Pre-registration: paper/parent_screening_protocol.md, committed before this
was written. This file implements the SHARED MACHINERY (graph construction,
both generator families, size-capped clustering, the intercept+validation-
selected ridge readout, group encoders, scoring, candidate-set construction)
plus STAGE A ONLY: toy correctness and feasibility checks, no large training,
no scientific result. Stage B's full baseline suite (arms 2-4 beyond RANDOM,
community-detection comparison) and Stages B/C themselves are NOT implemented
here; they are written only once Stage A's checks pass and Stage B is
separately authorised, per the protocol's own staged-investment structure.

    python scripts/parent_screening.py           # runs stage_a()
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from boundary_map import poly3  # noqa: E402  -- reused verbatim, not redefined
from hierarchy_repair import sized_random  # noqa: E402  -- reused verbatim
from wormwideweb_gate import MaskedAE  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
E = 3                          # own-lag window, matches the rest of the repo
MAX_DELAY = 3
GROUP_CAP = 8                  # deterministic max group size
ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
VAL_INTERNAL_FRAC = 0.15       # tail of TRAIN used for alpha selection
UNRESOLVED_GAIN = 0.01
MASK_P = 0.25
EPOCHS, BATCH = 20, 64         # matches boundary_map's own constants
MAX_REDRAWS = 20
N_ROOT_FRAC = 0.12


# ================================================================
# Shared DAG topology, both families
# ================================================================

def build_dag(V: int, seed: int):
    """Topological order, root set, parent map with delays, orphan-repaired.

    Returns (order, is_root, parent[q] -> list[(j, delay)], n_root).
    `order` is the permutation defining topological order; is_root is a
    boolean array over ORIGINAL variable ids.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(V)
    n_root = max(3, round(N_ROOT_FRAC * V))
    is_root = np.zeros(V, bool)
    is_root[order[:n_root]] = True

    parent: dict[int, list[tuple[int, int]]] = {q: [] for q in range(V)}
    for pos in range(n_root, V):
        q = order[pos]
        d = int(rng.integers(1, 4))
        d = min(d, pos)  # cannot exceed available earlier variables
        earlier = order[:pos]
        choose = rng.choice(earlier, size=d, replace=False)
        delays = rng.integers(1, MAX_DELAY + 1, size=d)
        parent[q] = [(int(j), int(dl)) for j, dl in zip(choose, delays)]

    # orphan repair: deterministic single pass
    children_count = np.zeros(V, int)
    for q, plist in parent.items():
        for j, _ in plist:
            children_count[j] += 1
    for pos in range(n_root):
        root = order[pos]
        if children_count[root] > 0:
            continue
        if pos + 1 >= V:
            continue  # last-in-order root has no later variable; recorded, not fatal
        heir = order[pos + 1]
        if len(parent[heir]) >= 3:
            # replace the heir's highest-index existing parent
            worst = max(range(len(parent[heir])),
                       key=lambda i: parent[heir][i][0])
            parent[heir][worst] = (int(root), int(rng.integers(1, MAX_DELAY + 1)))
        else:
            parent[heir].append((int(root), int(rng.integers(1, MAX_DELAY + 1))))
        children_count[root] += 1

    return order, is_root, parent, n_root


def orphan_free(is_root, parent) -> bool:
    """True iff every root has >=1 child, EXCEPT possibly the root that is
    last in topological order (build_dag records, does not force-fix, that
    single edge case since there is no later variable to attach it to)."""
    children = set()
    for plist in parent.values():
        for j, _ in plist:
            children.add(j)
    roots = np.where(is_root)[0]
    missing = [r for r in roots if r not in children]
    return len(missing) <= 1


# ================================================================
# Family 1: corrected sparse coupled logistic-map family
# ================================================================

def _is_locked_r(r: float) -> bool:
    """Reuses generator_audit's rule if importable; otherwise a documented
    fallback using the same definition (Lyapunov <= 0.05 or max|ac| >= 0.9
    over a short probe trajectory), so this module has no hard dependency
    on a script whose own __main__ does unrelated file I/O at import time."""
    try:
        from generator_audit import is_locked_r
        return is_locked_r(r)
    except Exception:
        x = np.empty(600)
        x[0] = 0.4
        for t in range(599):
            x[t + 1] = r * x[t] * (1 - x[t])
        s = x[100:]
        lam = np.mean(np.log(np.abs(r * (1 - 2 * s[:-1])) + 1e-12))
        d = s - s.mean()
        den = float(d @ d)
        max_ac = 0.0
        if den > 0:
            max_ac = max(abs(d[:-k] @ d[k:]) / den for k in range(1, 51))
        return bool(lam <= 0.05 or max_ac >= 0.9)


def family1_generate(V: int, n: int, seed: int, coupling: float = 0.20):
    """Returns dict with x_obs, x_clean, parent, is_root, diagnostics, or
    raises RuntimeError after MAX_REDRAWS failed validity attempts."""
    for attempt in range(MAX_REDRAWS):
        rng = np.random.default_rng(seed * 97 + attempt)
        order, is_root, parent, n_root = build_dag(V, seed * 97 + attempt)

        r = np.empty(V)
        for i in range(V):
            while True:
                cand = float(rng.uniform(3.6, 3.9))
                if not _is_locked_r(cand):
                    r[i] = cand
                    break
        eta = {q: {j: float(rng.uniform(0.85, 1.15)) for j, _ in plist}
              for q, plist in parent.items()}

        max_d = MAX_DELAY
        total = 500 + max_d + n
        x = np.empty((total, V))
        x[0] = rng.uniform(0.2, 0.8, V)
        # need max_d history before t=0 of the "kept" region; simulate from
        # a single initial row using the same recursion (self-consistent
        # burn-in, no separate warm formula needed since max_d < total)
        for t in range(total - 1):
            nxt = np.empty(V)
            for q in range(V):
                k = x[t, q]
                if is_root[q]:
                    nxt[q] = r[q] * k * (1 - k)
                else:
                    drive = np.mean([eta[q][j] * x[max(t - d + 1, 0), j]
                                     for j, d in parent[q]])
                    nxt[q] = r[q] * k * (1 - k) * (1 - coupling) \
                        + coupling * drive * (1 - k)
            x[t + 1] = np.clip(nxt, 0.0, 1.0)
        x_clean = x[500:]  # drop burn-in; length max_d + n

        if not np.all(np.isfinite(x_clean)):
            continue
        clip_frac = np.mean((x_clean <= 1e-9) | (x_clean >= 1 - 1e-9), axis=0)
        if clip_frac.max() > 0.30:
            continue
        if not orphan_free(is_root, parent):
            continue

        x_kept = x_clean[max_d:max_d + n]
        a = int(0.6 * n)
        near_sync = 0
        for q, plist in parent.items():
            for j, d in plist:
                xj = x_kept[:a - d, j]
                xq = x_kept[d:a, q]
                if len(xj) > 10 and xj.std() > 0 and xq.std() > 0:
                    c = abs(np.corrcoef(xj, xq)[0, 1])
                    if c >= 0.98:
                        near_sync += 1

        train_std = x_kept[:a].std(axis=0) + 1e-12
        noise = 0.05 * train_std * np.random.default_rng(
            seed * 97 + attempt + 5000).standard_normal(x_kept.shape)
        x_obs = x_kept + noise

        return dict(x_obs=x_obs, x_clean=x_kept, parent=parent,
                   is_root=is_root, order=order, n_root=n_root,
                   clip_frac_max=float(clip_frac.max()),
                   near_sync_edges=near_sync, attempts=attempt + 1,
                   family="family1")
    raise RuntimeError(f"family1_generate: no valid draw in {MAX_REDRAWS} "
                       f"attempts at V={V} seed={seed}")


# ================================================================
# Family 2: stable sparse nonlinear autoregressive family
# ================================================================

def _stable_ar2(rng):
    while True:
        a = float(rng.uniform(-0.6, 0.6))
        b = float(rng.uniform(-0.3, 0.3))
        if abs(a) + abs(b) >= 0.9:
            continue
        roots = np.roots([1.0, -a, -b])
        if np.all(np.abs(roots) < 1.0):
            return a, b


def family2_generate(V: int, n: int, seed: int):
    for attempt in range(MAX_REDRAWS):
        rng = np.random.default_rng(seed * 131 + attempt)
        order, is_root, parent, n_root = build_dag(V, seed * 131 + attempt)

        ab = [_stable_ar2(rng) for _ in range(V)]
        a_coef = np.array([p[0] for p in ab])
        b_coef = np.array([p[1] for p in ab])
        gamma = {q: float(rng.uniform(0.4, 0.8)) for q in range(V)
                 if not is_root[q]}
        w = {q: {j: float(rng.uniform(0.3, 0.7)) for j, _ in parent[q]}
            for q in range(V) if not is_root[q]}
        sigma = np.where(is_root, 0.7, 0.5)

        max_d = MAX_DELAY
        total = 500 + max_d + n
        x = np.zeros((total, V))
        x[0] = rng.standard_normal(V)
        x[1] = rng.standard_normal(V)
        eps = sigma * rng.standard_normal((total, V))
        blew_up = False
        for t in range(1, total - 1):
            nxt = a_coef * x[t] + b_coef * x[t - 1]
            for q in range(V):
                if not is_root[q]:
                    drive = np.mean([w[q][j] * np.tanh(x[max(t - d + 1, 0), j])
                                     for j, d in parent[q]])
                    nxt[q] += gamma[q] * drive
            nxt += eps[t + 1]
            if np.any(np.abs(nxt) > 50):
                blew_up = True
                break
            x[t + 1] = nxt
        if blew_up or not np.all(np.isfinite(x)):
            continue
        x_clean = x[500:]
        var = x_clean.var(axis=0)
        if not np.all((var > 1e-4) & (var < 100)):
            continue
        if not orphan_free(is_root, parent):
            continue

        x_kept = x_clean[max_d:max_d + n]
        a_cut = int(0.6 * n)
        train_std = x_kept[:a_cut].std(axis=0) + 1e-12
        noise = 0.05 * train_std * np.random.default_rng(
            seed * 131 + attempt + 5000).standard_normal(x_kept.shape)
        x_obs = x_kept + noise

        return dict(x_obs=x_obs, x_clean=x_kept, parent=parent,
                   is_root=is_root, order=order, n_root=n_root,
                   variance_range=(float(var.min()), float(var.max())),
                   attempts=attempt + 1, family="family2")
    raise RuntimeError(f"family2_generate: no valid draw in {MAX_REDRAWS} "
                       f"attempts at V={V} seed={seed}")


# ================================================================
# Own-lag features, splits
# ================================================================

def own_lag_window(x_obs: np.ndarray, q: int, max_delay: int = MAX_DELAY):
    """Rows aligned so row i holds x_q(t-2), x_q(t-1), x_q(t) for t = i+2,
    and the matching target is x_q(t+1) = x_obs[i+3, q]. Returns
    (own_raw[i] shape (m, E), target[i] shape (m,), t_index[i]) with
    m = len(x_obs) - max_delay - 1."""
    n = x_obs.shape[0]
    m = n - max_delay - 1
    own = np.stack([x_obs[max_delay - k: max_delay - k + m, q]
                   for k in range(E - 1, -1, -1)], axis=1)
    target = x_obs[max_delay + 1: max_delay + 1 + m, q]
    t_index = np.arange(max_delay, max_delay + m)
    return own, target, t_index


def splits_for(n_obs: int, max_delay: int = MAX_DELAY, E_: int = E):
    """Contiguous 60/20/20 on the m = n_obs - max_delay - 1 aligned rows.

    EMBARGO = E, discovered by direct enumeration
    (parent_screening_split_test.py), not assumed from another pipeline's
    differencing convention (Rule 132's lesson applied to this pipeline
    directly). The first drafted value here, and independently the first
    value in paper/parent_screening_protocol.md, both stated
    (E-1)+max_delay = 5 or (E-1)*max_delay = 6 -- WRONG, because a
    candidate group's own multi-lag structure is never differenced in and
    never reaches further back than the SAME E-length window q's own lags
    use; the only extra touch beyond that window is the single forward
    step to the t+1 target. The enumerated minimum is exactly E: a row's
    raw support is its own [t-(E-1) .. t] window unioned with {t+1}, a
    span of E+1 consecutive raw indices, and embargoing E rows at each
    seam is what the oracle-verified enumeration in
    parent_screening_split_test.py confirms closes every such overlap."""
    m = n_obs - max_delay - 1
    a = int(0.6 * m)
    b = int(0.8 * m)
    embargo = E_
    tr = np.arange(0, max(a - embargo, 1))
    va = np.arange(a, max(b - embargo, a + 1))
    te = np.arange(b, m)
    return tr, va, te, m, embargo


# ================================================================
# Deterministic size-capped clustering (train rows only)
# ================================================================

def cluster_size_capped(x_train_raw: np.ndarray, cap: int = GROUP_CAP):
    """Build the FULL average-linkage dendrogram (every scipy merge node
    unconditionally, matching scipy's own numbering exactly -- skipping a
    merge mid-build was tried first and is WRONG: scipy's later rows
    reference earlier merge-node ids by number regardless of whether this
    code decided to use them, so skipping one invalidates every later
    reference to it, discovered by an IndexError when this was first run).

    Then CUT the tree top-down: starting from the root, descend into a
    node's two children whenever its own subtree exceeds `cap`, and stop
    descending (emit one group) the moment a subtree is <= cap. This is
    deterministic given x_train_raw and produces every final group at
    size 1-cap with no declared group COUNT in advance."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    Vt = x_train_raw.shape[1]
    d = np.diff(x_train_raw, axis=0)
    with np.errstate(invalid="ignore"):
        c = np.corrcoef(d.T)
    c = np.nan_to_num(c, nan=0.0)
    np.fill_diagonal(c, 1.0)
    dist = 1.0 - np.abs(c)
    dist[dist < 0] = 0.0
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")  # shape (Vt-1, 4)

    members = {i: [i] for i in range(Vt)}
    for i, row in enumerate(Z):
        a, b = int(row[0]), int(row[1])
        members[Vt + i] = members[a] + members[b]

    root = Vt + len(Z) - 1  # scipy's final merge is always the whole tree

    groups: list[list[int]] = []

    def cut(node: int):
        subtree = members[node]
        if len(subtree) <= cap or node < Vt:
            groups.append(subtree)
            return
        a, b = int(Z[node - Vt][0]), int(Z[node - Vt][1])
        cut(a)
        cut(b)

    cut(root)

    labels = np.empty(Vt, int)
    for gid, grp in enumerate(groups):
        for m in grp:
            labels[m] = gid
    return labels


# ================================================================
# Ridge readout: intercept + train-internal validation-selected alpha
# ================================================================

def ridge_fit_predict(Xtr, ytr, alpha):
    Xt = torch.as_tensor(Xtr, dtype=torch.float64, device=DEV)
    yt = torch.as_tensor(ytr, dtype=torch.float64, device=DEV)
    ones = torch.ones((Xt.shape[0], 1), dtype=torch.float64, device=DEV)
    Xt1 = torch.cat([Xt, ones], dim=1)
    p = Xt1.shape[1]
    A = Xt1.T @ Xt1 + alpha * torch.eye(p, device=DEV, dtype=torch.float64)
    A[-1, -1] -= alpha  # do not penalise the intercept column
    w = torch.linalg.solve(A, Xt1.T @ yt)
    return w


def _r2(Xe, ye, w):
    Xe_t = torch.as_tensor(Xe, dtype=torch.float64, device=DEV)
    ye_t = torch.as_tensor(ye, dtype=torch.float64, device=DEV)
    ones = torch.ones((Xe_t.shape[0], 1), dtype=torch.float64, device=DEV)
    pred = torch.cat([Xe_t, ones], dim=1) @ w
    err = float(((pred - ye_t) ** 2).mean())
    var = float(ye_t.var()) + 1e-12
    return 1.0 - err / var  # UNCLIPPED, per protocol


def ridge_r2_val(Xtr, ytr, Xeval, yeval):
    """Select alpha on the last VAL_INTERNAL_FRAC of Xtr/ytr (temporal),
    refit on the full Xtr at that alpha, report UNCLIPPED R2 on
    (Xeval, yeval). Returns (r2, alpha_used, hit_grid_boundary)."""
    n = Xtr.shape[0]
    cut = max(int(n * (1 - VAL_INTERNAL_FRAC)), 1)
    Xi_tr, yi_tr = Xtr[:cut], ytr[:cut]
    Xi_va, yi_va = Xtr[cut:], ytr[cut:]
    if len(yi_va) < 2:
        Xi_va, yi_va = Xtr, ytr  # degenerate tiny-n fallback, both same slice
    best_a, best_err = ALPHA_GRID[0], float("inf")
    for a in ALPHA_GRID:
        w = ridge_fit_predict(Xi_tr, yi_tr, a)
        Xv = torch.as_tensor(Xi_va, dtype=torch.float64, device=DEV)
        yv = torch.as_tensor(yi_va, dtype=torch.float64, device=DEV)
        ones = torch.ones((Xv.shape[0], 1), dtype=torch.float64, device=DEV)
        pred = torch.cat([Xv, ones], dim=1) @ w
        err = float(((pred - yv) ** 2).mean())
        if err < best_err:
            best_err, best_a = err, a
    w_full = ridge_fit_predict(Xtr, ytr, best_a)
    r2 = _r2(Xeval, yeval, w_full)
    hit_boundary = best_a in (ALPHA_GRID[0], ALPHA_GRID[-1])
    return r2, best_a, hit_boundary


# ================================================================
# Group encoders (per-member masking so target exclusion is in-distribution)
# ================================================================

def group_bottleneck(group_size: int) -> int:
    return min(GROUP_CAP, 2 * group_size)


def train_group_encoder(z_group: np.ndarray, tr_idx, group_size: int, seed: int):
    b = group_bottleneck(group_size)
    d_in = z_group.shape[1]
    torch.manual_seed(seed)
    net = MaskedAE(d_in, b).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(seed)
    ztr = torch.as_tensor(z_group[tr_idx], device=DEV, dtype=torch.float32)
    for _ in range(EPOCHS):
        perm = torch.randperm(ztr.shape[0], generator=g)
        for i in range(0, len(perm), BATCH):
            bt = ztr[perm[i:i + BATCH]]
            msk = torch.rand(bt.shape[0], group_size, device=DEV) < MASK_P
            mc = msk.repeat_interleave(E, dim=1)
            loss = ((net(bt.masked_fill(mc, 0.0)) - bt)[mc] ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


def group_code(net, z_group: np.ndarray, exclude_member_pos: int | None = None):
    z = z_group.copy()
    if exclude_member_pos is not None:
        z[:, exclude_member_pos * E:(exclude_member_pos + 1) * E] = 0.0
    with torch.no_grad():
        return net.enc(torch.as_tensor(z, device=DEV, dtype=torch.float32)
                      ).cpu().numpy()


# ================================================================
# Scoring and candidate-set construction
# ================================================================

def score_target_against_group(own_feats, target, group_codes, tr, va):
    """own_feats: (m, 19) poly3 of own lags. group_codes: (m, b_G) or None
    (zero-width group). Returns (gain, alpha_used, hit_boundary)."""
    Xtr_own, ytr = own_feats[tr], target[tr]
    Xva_own, yva = own_feats[va], target[va]
    r2_own, _, _ = ridge_r2_val(Xtr_own, ytr, Xva_own, yva)
    if group_codes is None or group_codes.shape[1] == 0:
        return 0.0, None, False
    Xtr_full = np.hstack([Xtr_own, group_codes[tr]])
    Xva_full = np.hstack([Xva_own, group_codes[va]])
    r2_full, alpha, hit = ridge_r2_val(Xtr_full, ytr, Xva_full, yva)
    return r2_full - r2_own, alpha, hit


def build_candidate_set(q: int, gains: dict[int, float], groups: dict[int, list[int]],
                        k: int):
    """gains: group_id -> gain(q, group_id). groups: group_id -> member list.
    Returns (C_q sorted list, unresolved bool)."""
    if not gains or max(gains.values()) <= UNRESOLVED_GAIN:
        all_others = sorted(g for grp in groups.values() for g in grp if g != q)
        return all_others, True
    order = sorted(gains.keys(), key=lambda gid: (-gains[gid], gid))
    C: list[int] = []
    for gid in order:
        members = sorted(m for m in groups[gid] if m != q)
        if len(C) + len(members) <= k:
            C.extend(members)
        else:
            remaining = k - len(C)
            if remaining > 0:
                C.extend(members[:remaining])
            break
    return sorted(set(C)), False


def k_for(V: int) -> int:
    import math
    return max(1, math.ceil(0.10 * (V - 1)))


# ================================================================
# Metrics
# ================================================================

def retained_parent_recall(C: dict[int, list[int]], parent: dict[int, list]):
    num = sum(len(set(C[q]) & {j for j, _ in parent[q]}) for q in parent
             if parent[q])
    den = sum(len(parent[q]) for q in parent if parent[q])
    return num / den if den else float("nan")


def complete_target_coverage(C: dict[int, list[int]], parent: dict[int, list]):
    vals = [set(j for j, _ in parent[q]).issubset(set(C[q]))
           for q in parent if parent[q]]
    return float(np.mean(vals)) if vals else float("nan")


# ================================================================
# STAGE A: correctness and feasibility only. No scientific result.
# ================================================================

def _toy_orientation_system(n=2000, coupling=0.30, seed=900):
    """Hand-fixed 5-variable graph, bypassing build_dag's V>=3-root floor:
    0=A (root) -> 1=B (delay 1) -> 2=C (delay 1, the target); 3=D, 4=F
    unrelated roots. Family-1 dynamics, fixed graph, for the orientation
    check only -- excluded from every scientific claim."""
    rng = np.random.default_rng(seed)
    V = 5
    is_root = np.array([True, False, False, True, True])
    parent = {0: [], 1: [(0, 1)], 2: [(1, 1)], 3: [], 4: []}
    r = np.array([3.75, 3.75, 3.75, 3.75, 3.75])  # away from lock windows
    total = 300 + n
    x = np.empty((total, V))
    x[0] = rng.uniform(0.2, 0.8, V)
    for t in range(total - 1):
        nxt = np.empty(V)
        for q in range(V):
            k = x[t, q]
            if is_root[q]:
                nxt[q] = r[q] * k * (1 - k)
            else:
                j, d = parent[q][0]
                drive = x[max(t - d + 1, 0), j]
                nxt[q] = r[q] * k * (1 - k) * (1 - coupling) \
                    + coupling * drive * (1 - k)
        x[t + 1] = np.clip(nxt, 0.0, 1.0)
    return x[300:]


def _run_toy_pipeline(x_obs, groups, targets, seed=901):
    """Minimal end-to-end run: per-group encoder, own-lag features, ridge
    scoring for every (target, group) pair. Returns gains[q][gid].

    Every window, own or group-member, is built by own_lag_window with the
    SAME max_delay convention splits_for uses, so tr/va/te indices from
    splits_for index the SAME row space these arrays use -- a mismatch
    caught here before running (an earlier draft built group windows with
    a different t-alignment and a different row count than splits_for
    assumes, which would have applied one array's indices to another
    array's rows silently)."""
    tr, va, te, m, embargo = splits_for(len(x_obs))
    gains: dict[int, dict[int, float]] = {}
    nets = {}
    for gid, members in groups.items():
        member_windows = [own_lag_window(x_obs, j)[0] for j in members]
        assert all(w.shape[0] == m for w in member_windows)
        z_group = np.concatenate(member_windows, axis=1)  # (m, len(members)*E)
        net = train_group_encoder(z_group, tr, len(members), seed=seed + gid)
        nets[gid] = (net, z_group, members)
    for q in targets:
        own_raw, target_vals, _ = own_lag_window(x_obs, q)
        own_feats = poly3(own_raw)
        gains[q] = {}
        for gid, (net, z_group, members) in nets.items():
            excl = members.index(q) if q in members else None
            codes = group_code(net, z_group, exclude_member_pos=excl)
            gain, alpha, hit = score_target_against_group(
                own_feats, target_vals, codes, tr, va)
            gains[q][gid] = gain
    return gains


def stage_a() -> bool:
    t0 = time.time()
    ok = True
    print("=" * 66)
    print("STAGE A: correctness and feasibility only. No scientific result.")
    print("=" * 66)

    # ---- import / baseline memory footprint
    try:
        import large_system as _LS
        rss = _LS.host_rss_mb()
        print(f"\n[resource] RSS after imports: {rss:.1f} MB")
    except Exception as e:
        print(f"\n[resource] host_rss_mb unavailable ({e}); skipping")

    # ---- hypergeometric identity, numeric re-check
    from math import comb
    cases = [(10, 3, 2), (239, 24, 1), (239, 24, 2), (239, 24, 3),
             (499, 50, 2), (999, 100, 3)]
    print("\n[1] hypergeometric identity re-check")
    id_ok = True
    for V1, k, d in cases:
        a = comb(V1 - d, k - d) / comb(V1, k)
        b = comb(k, d) / comb(V1, d)
        same = abs(a - b) < 1e-9
        id_ok &= same
        print(f"    V-1={V1:4d} k={k:3d} d={d}  {a:.6f} vs {b:.6f}  "
              f"{'OK' if same else 'MISMATCH'}")
    print(f"    -> {'PASS' if id_ok else 'FAIL'}")
    ok &= id_ok

    # ---- k formula sanity
    print("\n[2] k = ceil(0.10*(V-1)) at pilot/confirmation widths")
    for V in (240, 500, 1000):
        k = k_for(V)
        print(f"    V={V:<5} k={k:<4} k/(V-1)={k/(V-1):.4f}")

    # ---- clustering determinism and size cap
    print("\n[3] size-capped clustering: determinism and max group size")
    rng = np.random.default_rng(42)
    x_toy = rng.standard_normal((300, 40))
    x_toy[:, 5:10] += x_toy[:, [5]] * 3  # force one clear correlated block
    lab1 = cluster_size_capped(x_toy)
    lab2 = cluster_size_capped(x_toy)
    deterministic = np.array_equal(lab1, lab2)
    sizes = np.bincount(lab1)
    size_ok = sizes.max() <= GROUP_CAP
    print(f"    deterministic across two runs: {deterministic}")
    print(f"    group sizes: {sorted(sizes.tolist(), reverse=True)}  "
          f"max={sizes.max()} (cap {GROUP_CAP})")
    print(f"    -> {'PASS' if deterministic and size_ok else 'FAIL'}")
    ok &= deterministic and size_ok

    # ---- sized_random size-multiset check
    print("\n[4] sized_random reproduces the exact size multiset")
    rng2 = np.random.default_rng(7)
    rand_lab = sized_random(sizes, 40, rng2)
    rand_sizes = np.bincount(rand_lab)
    multiset_ok = sorted(rand_sizes.tolist()) == sorted(sizes.tolist())
    print(f"    original sizes: {sorted(sizes.tolist())}")
    print(f"    random sizes:   {sorted(rand_sizes.tolist())}")
    print(f"    -> {'PASS' if multiset_ok else 'FAIL'}")
    ok &= multiset_ok

    # ---- split disjointness (delegates to the dedicated enumeration test)
    print("\n[5] split disjointness by enumeration")
    import subprocess
    r = subprocess.run([sys.executable, str(Path(__file__).parent /
                       "parent_screening_split_test.py")],
                       capture_output=True, text=True)
    print("    " + r.stdout.strip().replace("\n", "\n    "))
    split_ok = r.returncode == 0
    ok &= split_ok

    # ---- generator validity smoke test, both families, V=30, 3 seeds each
    print("\n[6] generator validity smoke test (V=30, engineering seeds)")
    gen_ok = True
    for fam_name, fam_fn in [("family1", family1_generate),
                             ("family2", family2_generate)]:
        for seed in (9001, 9002, 9003):
            try:
                out = fam_fn(30, 500, seed)
                print(f"    {fam_name} seed={seed}: OK, attempts={out['attempts']}, "
                      f"n_root={out['n_root']}")
            except RuntimeError as e:
                print(f"    {fam_name} seed={seed}: FAILED -- {e}")
                gen_ok = False
    print(f"    -> {'PASS' if gen_ok else 'FAIL'}")
    ok &= gen_ok

    # ---- A->B->C label orientation, on the real scoring pipeline
    print("\n[7] label-orientation check on a hand-fixed 5-variable chain")
    x_toy2 = _toy_orientation_system()
    groups = {0: [0], 1: [1], 2: [2], 3: [3, 4]}
    gains = _run_toy_pipeline(x_toy2, groups, targets=[1, 2])
    # B's true parent is A (group 0); C's true parent is B (group 1)
    b_true = gains[1][0]
    b_false = max(gains[1][3], 0.0)
    c_true = gains[2][1]
    c_false = max(gains[2][3], 0.0)
    orient_ok = (b_true > b_false) and (c_true > c_false)
    print(f"    B: gain(true parent A)={b_true:+.4f}  "
          f"gain(unrelated D,F)={b_false:+.4f}")
    print(f"    C: gain(true parent B)={c_true:+.4f}  "
          f"gain(unrelated D,F)={c_false:+.4f}")
    print(f"    -> {'PASS' if orient_ok else 'FAIL'}")
    ok &= orient_ok

    # ---- test-block perturbation invariance
    print("\n[8] test-block perturbation invariance")
    tr, va, te, m, embargo = splits_for(len(x_toy2))
    x_pert = x_toy2.copy()
    # perturb only rows whose window lies entirely within the te block
    pert_rows = np.arange(len(x_toy2) - 50, len(x_toy2))
    x_pert[pert_rows] += np.random.default_rng(123).standard_normal(
        (len(pert_rows), x_pert.shape[1])) * 10.0
    gains_pert = _run_toy_pipeline(x_pert, groups, targets=[1, 2], seed=901)
    same = all(abs(gains[q][g] - gains_pert[q][g]) < 1e-9
              for q in gains for g in gains[q])
    print(f"    gains identical after perturbing the last 50 raw rows: {same}")
    print(f"    -> {'PASS' if same else 'FAIL'}")
    ok &= same

    print(f"\n{'=' * 66}")
    print(f"STAGE A {'PASSED' if ok else 'FAILED'} in {time.time()-t0:.1f}s")
    print("No baseline arms beyond RANDOM, no stress panels, no Stage B/C "
          "widths were run. Deferred to Stage B's own authorisation.")
    print("=" * 66)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if stage_a() else 1)
