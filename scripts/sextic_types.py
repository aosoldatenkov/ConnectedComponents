#!/usr/bin/env python3
"""Nesting types of plane sextics: a random scan (volumes, pools), a type-guided exploration (MAP-Elites over
nesting types, seeded only with the scan's pools), and exactly certified integer representatives.

Types. Smooth plane sextics have nesting depth <= 2 and at most one nest (Bezout), so every type is a u 1<b>
(a empty ovals, one oval containing b empty ovals): code = 16 a + b. Trees violating this (and inconsistent trees)
are counted as "other".

Phase A (`--minutes-a`, default 70): forms drawn from the uniform measure on S^27 (1/3 of the batches) and the
  Kostlan measure (2/3), types on FaceMesh(N = 30). Per measure and type: counts, and a uniform reservoir (bottom-k
  random keys; all hits of rare types), used for the grid-error correction. Per type: a pool of the most robust
  forms, robustness = min |critical value| of the normalized form on RP^2 (0 on the discriminant); the 8 smallest
  |critical values| are stored with each pool form (closeness to oval births/deaths, for later refined searches).
Phase B (`--minutes-b`, default 45): MAP-Elites with one archive cell per type, initialized with the phase A pools.
  Children: parent + r g (g a random unit direction, r log-uniform in [rmin, rmax]), renormalized; half of the
  parents come from cells weighted by 4^ovals / sqrt(cell size), half from uniformly chosen cells. Up to `cand`
  children per type and batch are scored; children with >= 5 ovals or >= 2 inner ovals must be confirmed at
  N = 200 and have robustness >= `min_score` (thin annuli broken by the grid fake nested types, even at N = 200
  for forms practically on the discriminant); cells keep up to `cell` forms,
  chosen greedily by robustness with a minimum angle `min_angle`
  between members (otherwise a cell collapses onto copies of one parent). Phase B is biased: no volumes from it.
Phase C (finish): reservoirs reclassified at N = 150 (confusion matrices), corrected volumes with bootstrap
  intervals, a log-linear fit of the frequency by number of ovals (extrapolated to 9-11), and per type up to 3
  integer representatives: diverse robust forms (confirmed at N = 150) rounded to small integers (type checked at
  N = 100 and 200) and certified exactly in parallel CPU processes. Oval count: pencil tangents (upper bound;
  if loose, the interior pencil bound of conncomp.certify with base points at the witnesses) and separating
  polygons (lower bound), with a FLINT cross-check. Type: with k ovals proven, the k + 1 pairwise
  separated witnesses lie one in each region and the certified base point O lies in the root region N; then b is
  the number of other regions with the sign of N, and a = k - b - [b > 0]. (The depth-3 nest 1<1<1>>, the only
  other sextic scheme, has the signs of 1 u 1<1>; it is excluded by lines through the witnesses of sign -sign(N)
  meeting the curve in <= 4 points.)

Large outputs go to pools/sextic_plain/ (untracked); integer representatives to data/sextic_types/plain.json.

    uv run python scripts/sextic_types.py                     # A, B and C
    uv run python scripts/sextic_types.py --stage finish      # C only, from the saved checkpoints
"""

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch

from conncomp import DEVICE, DTYPE
from conncomp.critical import FaceCharts, critical_points
from conncomp.gpu_trees import FaceMesh
from conncomp.polynomials import monomials, normalize, sample

DEG = 6
D = len(monomials(DEG))
OTHER = 256
NC = 257
MEASURES = ["uniform", "kostlan"]
KOSTLAN = torch.tensor([math.sqrt(math.factorial(DEG) / (math.factorial(i) * math.factorial(j) * math.factorial(k)))
                        for i, j, k in monomials(DEG)], dtype=DTYPE, device=DEVICE)
NCV = 8  # smallest |critical values| stored per pool form


def type_name(code):
    if code == OTHER:
        return "other"
    a, b = divmod(int(code), 16)
    if b == 0:
        return str(a)
    return f"1<{b}>" if a == 0 else f"{a} u 1<{b}>"


def n_ovals(code):
    a, b = divmod(int(code), 16)
    return a + b + (b > 0)


def tag(code):
    return type_name(code).replace(" u ", "u").replace("<", "(").replace(">", ")")


def draw(measure, B):
    if measure == "uniform":
        return sample(DEG, B)
    return normalize(KOSTLAN[:, None] * torch.randn((D, B), dtype=DTYPE, device=DEVICE))


def type_codes(t, B):
    """Type code per form from GPUTrees (OTHER unless a valid sextic type a u 1<b>)."""
    dev = t.depth.device
    al = t.alive
    d = torch.where(al, t.depth, torch.zeros_like(t.depth))
    cnt = lambda m: torch.bincount(t.form[m], minlength=B)  # noqa: E731
    d1, d2 = cnt(al & (d == 1)), cnt(al & (d == 2))
    has_child = torch.zeros_like(al)
    deep = al & (d == 2)
    has_child[t.parent[deep]] = True
    nests = cnt(has_child & al & (d == 1))
    maxd = torch.zeros(B, dtype=torch.long, device=dev).scatter_reduce_(0, t.form, d, "amax")
    a, b = d1 - nests, d2
    valid = t.ok & (maxd <= 2) & (nests <= 1) & (a < 16) & (b < 16)
    return torch.where(valid, 16 * a + b, torch.full_like(a, OTHER))


def classify(fm, c, batch=None):
    batch = batch or max(1, int(4e8 // fm.n_loc))
    return torch.cat([type_codes(fm.trees(c[:, s : s + batch], DEG), c[:, s : s + batch].shape[1])
                      for s in range(0, c.shape[1], batch)])


def robustness(c, charts):
    """(min |critical value|, the NCV smallest |critical values|) of the normalized forms c (D, B); score 0 if
    the Morse check fails."""
    B = c.shape[1]
    crit, morse = critical_points(c, DEG, charts=charts)
    counts = torch.bincount(crit.form, minlength=B)
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(len(crit), device=c.device) - start[crit.form]
    K = max(int(counts.max()) if len(crit) else 1, NCV)
    pad = torch.full((B, K), float("inf"), dtype=DTYPE, device=c.device)
    pad[crit.form, pos] = crit.value.abs()
    cv = pad.sort(dim=1).values[:, :NCV]
    score = torch.where(morse == 1, cv[:, 0], torch.zeros_like(cv[:, 0]))
    return score, cv


# ---------------------------------------------------------------------------
# Storage: dicts of arrays keyed by type code, saved as .npz
# ---------------------------------------------------------------------------


def save_npz(path, scalars, groups):
    arrs = {k: np.asarray(v) for k, v in scalars.items()}
    for gname, g in groups.items():
        for key, tup in g.items():
            ks = "_".join(map(str, key)) if isinstance(key, tuple) else str(key)
            for i, a in enumerate(tup):
                arrs[f"{gname}__{ks}__{i}"] = a
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrs)
    tmp.replace(path)


def load_npz(path):
    z = np.load(path)
    scalars, groups = {}, {}
    for name in z.files:
        if "__" not in name:
            scalars[name] = z[name]
            continue
        gname, ks, i = name.split("__")
        key = tuple(int(x) for x in ks.split("_")) if "_" in ks else int(ks)
        groups.setdefault(gname, {}).setdefault(key, {})[int(i)] = z[name]
    groups = {g: {k: tuple(v[i] for i in range(len(v))) for k, v in d.items()} for g, d in groups.items()}
    return scalars, groups


def empty_pool():
    return (np.zeros(0), np.zeros((0, D)), np.zeros((0, NCV)))


def merge_pool(pool, score, F, cv, cap, min_angle=0.0):
    """Keep the `cap` most robust forms (score > 0); with min_angle > 0, greedily by robustness, skipping forms
    within that angle (up to sign) of an already kept one, so that cells spread out instead of collapsing."""
    s = np.concatenate([pool[0], score])
    f = np.concatenate([pool[1], F])
    c = np.concatenate([pool[2], cv])
    order = np.argsort(-s, kind="stable")
    order = order[s[order] > 0]
    if min_angle > 0 and len(order) > 1:
        U = f[order] / np.linalg.norm(f[order], axis=1, keepdims=True)
        close = np.abs(U @ U.T) > math.cos(min_angle)
        keep_mask = np.zeros(len(order), dtype=bool)
        blocked = np.zeros(len(order), dtype=bool)
        for i in range(len(order)):
            if blocked[i]:
                continue
            keep_mask[i] = True
            blocked |= close[i]
            if keep_mask.sum() == cap:
                break
        order = order[keep_mask]
    keep = order[:cap]
    return (s[keep], f[keep], c[keep])


# ---------------------------------------------------------------------------
# Phase A: random scan
# ---------------------------------------------------------------------------


def phase_a(args, run):
    path = run / "phase_a.npz"
    counts = np.zeros((len(MEASURES), NC), dtype=np.int64)
    res, pool, elapsed = {}, {}, 0.0
    if path.exists():
        sc, g = load_npz(path)
        counts, elapsed = sc["counts"], float(sc["elapsed"])
        res, pool = g.get("res", {}), g.get("pool", {})
    fm, charts = FaceMesh(args.N), FaceCharts(DEG, N=40)
    buf, nbuf = {}, {}

    def flush(code):
        if not buf.get(code):
            return
        f = torch.cat(buf[code], dim=1)
        s, cv = robustness(f, charts)
        pool[code] = merge_pool(pool.get(code, empty_pool()), s.cpu().numpy(), f.T.cpu().numpy(),
                                cv.cpu().numpy(), args.pool)
        buf[code], nbuf[code] = [], 0

    def save():
        save_npz(path, {"counts": counts, "elapsed": elapsed}, {"res": res, "pool": pool})

    t0, t_ck, it = time.time(), time.time(), int(counts.sum() // args.batch)
    start_elapsed = elapsed
    while start_elapsed + time.time() - t0 < args.minutes_a * 60:
        m = 0 if it % 3 == 0 else 1
        f = draw(MEASURES[m], args.batch)
        code = classify(fm, f, args.batch)
        counts[m] += torch.bincount(code, minlength=NC).cpu().numpy()
        key = torch.rand(args.batch, dtype=DTYPE, device=DEVICE)
        for k in torch.unique(code).tolist():
            idx = (code == k).nonzero()[:, 0]
            u0, f0 = res.get((m, k), (np.zeros(0), np.zeros((0, D))))
            thr = u0.max() if len(u0) >= args.reservoir else 2.0
            sel = idx[key[idx] < thr]
            if sel.numel():
                u = np.concatenate([u0, key[sel].cpu().numpy()])
                ff = np.concatenate([f0, f[:, sel].T.cpu().numpy()])
                o = np.argsort(u)[: args.reservoir]
                res[(m, k)] = (u[o], ff[o])
            if k != OTHER:
                take = idx[torch.randperm(idx.numel(), device=DEVICE)[: args.cand]]
                buf.setdefault(k, []).append(f[:, take])
                nbuf[k] = nbuf.get(k, 0) + take.numel()
                if nbuf[k] >= args.flush:
                    flush(k)
        it += 1
        if time.time() - t_ck > args.checkpoint * 60:
            for k in list(buf):
                flush(k)
            elapsed = start_elapsed + time.time() - t0
            save()
            t_ck = time.time()
            tot = counts.sum(0)
            top = sorted((k for k in range(NC) if tot[k] and k != OTHER), key=lambda k: (-n_ovals(k), -tot[k]))[:6]
            print(f"[A {elapsed / 60:5.1f} min] {counts.sum():,} forms ({counts.sum() / elapsed:,.0f}/s), "
                  f"{int((tot[:OTHER] > 0).sum())} types; highest: "
                  + ", ".join(f"{type_name(k)}: {tot[k]}" for k in top), flush=True)
    for k in list(buf):
        flush(k)
    elapsed = start_elapsed + time.time() - t0
    save()
    print(f"phase A done: {counts.sum():,} forms in {elapsed / 60:.1f} min, "
          f"{int((counts.sum(0)[:OTHER] > 0).sum())} types", flush=True)
    return counts, res, pool


# ---------------------------------------------------------------------------
# Phase B: MAP-Elites over nesting types
# ---------------------------------------------------------------------------


def phase_b(args, run, pool_a):
    path = run / "phase_b.npz"
    if path.exists():
        sc, g = load_npz(path)
        archive, elapsed = g.get("cell", {}), float(sc["elapsed"])
        n_children = int(sc["children"])
        first = json.loads(str(sc["first"]))
    else:
        archive = {k: v for k, v in pool_a.items() if len(v[0])}
        elapsed, n_children = 0.0, 0
        first = {type_name(k): {"minutes": 0.0, "source": "A"} for k in archive}
    fm, charts = FaceMesh(args.N), FaceCharts(DEG, N=40)
    confirm = FaceMesh(200)
    rng = np.random.default_rng()

    def save():
        save_npz(path, {"elapsed": elapsed, "children": n_children, "first": json.dumps(first)}, {"cell": archive})

    t0, t_ck = time.time(), time.time()
    start_elapsed = elapsed
    while start_elapsed + time.time() - t0 < args.minutes_b * 60:
        cells = sorted(archive)
        size = np.array([len(archive[k][0]) for k in cells], dtype=np.float64)
        ov = np.array([n_ovals(k) for k in cells], dtype=np.float64)
        w = 4.0 ** ov / np.sqrt(size)
        w /= w.sum()
        B = args.batch
        pick = np.where(rng.random(B) < 0.5, rng.choice(len(cells), B, p=w), rng.integers(0, len(cells), B))
        P = np.empty((B, D))
        for ci in np.unique(pick):
            sel = np.nonzero(pick == ci)[0]
            F = archive[cells[ci]][1]
            P[sel] = F[rng.integers(0, len(F), sel.size)]
        P = torch.as_tensor(P.T, dtype=DTYPE, device=DEVICE)
        g = torch.randn((D, B), dtype=DTYPE, device=DEVICE)
        g /= g.norm(dim=0, keepdim=True)
        r = torch.exp(torch.empty(B, dtype=DTYPE, device=DEVICE).uniform_(math.log(args.rmin), math.log(args.rmax)))
        child = normalize(P / P.norm(dim=0, keepdim=True) + r * g)
        code = classify(fm, child, B)
        n_children += B
        for k in torch.unique(code).tolist():
            if k == OTHER:
                continue
            idx = (code == k).nonzero()[:, 0]
            take = idx[torch.randperm(idx.numel(), device=DEVICE)[: args.cand]]
            f = child[:, take]
            ko = n_ovals(k)
            risky = ko >= 5 or k % 16 >= 2  # grid artifacts: thin annuli broken into "inner ovals"
            if risky:
                f = f[:, classify(confirm, f) == k]
            if f.shape[1] == 0:
                continue
            s, cv = robustness(f, charts)
            if risky:  # forms practically on the discriminant fake types even at N = 200
                keep = s >= args.min_score
                f, s, cv = f[:, keep], s[keep], cv[keep]
                if f.shape[1] == 0:
                    continue
            new = k not in archive
            cell = merge_pool(archive.get(k, empty_pool()), s.cpu().numpy(), f.T.cpu().numpy(), cv.cpu().numpy(),
                              args.cell, args.min_angle)
            if len(cell[0]):
                archive[k] = cell
                if new:
                    mins = (start_elapsed + time.time() - t0) / 60
                    first[type_name(k)] = {"minutes": round(mins, 2), "source": "B"}
                    print(f"  [B {mins:5.1f} min] new type {type_name(k)} ({ko} ovals)", flush=True)
        if time.time() - t_ck > args.checkpoint * 60:
            elapsed = start_elapsed + time.time() - t0
            save()
            t_ck = time.time()
            top = sorted(archive, key=lambda k: -n_ovals(k))[:5]
            print(f"[B {elapsed / 60:5.1f} min] {n_children:,} children ({n_children / elapsed:,.0f}/s), "
                  f"{len(archive)} types; highest: " + ", ".join(f"{type_name(k)} ({len(archive[k][0])})"
                                                               for k in top), flush=True)
    elapsed = start_elapsed + time.time() - t0
    save()
    print(f"phase B done: {n_children:,} children in {elapsed / 60:.1f} min, {len(archive)} types", flush=True)
    return archive, first, n_children, elapsed


# ---------------------------------------------------------------------------
# Phase C: volumes, integer representatives, certification
# ---------------------------------------------------------------------------


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def volumes(counts, conf, active, rng, n_boot=2000):
    """Raw and corrected fractions over the active codes; conf[i, j] = #(coarse i, reference j). Bootstrap over
    the coarse counts and the confusion rows (observed transitions only)."""
    n = counts.sum()
    T = len(active)
    cnt = counts[active]
    C = conf[np.ix_(active, active)]
    raw = cnt / n
    cond = np.where(C.sum(1, keepdims=True) > 0, C / np.maximum(C.sum(1, keepdims=True), 1), np.eye(T))
    corr = raw @ cond
    boots = np.empty((n_boot, T))
    for bi in range(n_boot):
        p = rng.dirichlet(cnt + 0.5)
        q = np.zeros((T, T))
        for i in range(T):
            sup = (C[i] > 0) | (np.arange(T) == i)
            q[i, sup] = rng.dirichlet(C[i, sup] + 0.5)
        boots[bi] = p @ q
    out = {}
    for j, k in enumerate(active):
        out[type_name(k)] = {"ovals": n_ovals(k) if k != OTHER else None, "count": int(cnt[j]),
                             "raw": float(raw[j]), "raw_ci95": wilson(int(cnt[j]), int(n)),
                             "corrected": float(corr[j]),
                             "corrected_ci95": (float(np.quantile(boots[:, j], 0.025)),
                                                float(np.quantile(boots[:, j], 0.975)))}
    return out


def tail_fit(vol):
    """Fraction by number of ovals and a weighted log-linear fit over k >= 3 (counts >= 10)."""
    by_k, cnt_k = {}, {}
    for v in vol.values():
        if v["ovals"] is None:
            continue
        by_k[v["ovals"]] = by_k.get(v["ovals"], 0.0) + v["corrected"]
        cnt_k[v["ovals"]] = cnt_k.get(v["ovals"], 0) + v["count"]
    ks = [k for k in sorted(by_k) if k >= 3 and cnt_k[k] >= 10 and by_k[k] > 0]
    fit = None
    if len(ks) >= 2:
        x = np.array(ks, dtype=float)
        y = np.log(np.array([by_k[k] for k in ks]))
        wt = np.sqrt(np.array([cnt_k[k] for k in ks], dtype=float))
        A = np.stack([np.ones_like(x), x], 1) * wt[:, None]
        coef = np.linalg.lstsq(A, y * wt, rcond=None)[0]
        fit = {"log_intercept": float(coef[0]), "log_slope": float(coef[1]), "factor_per_oval": float(np.exp(-coef[1])),
               "fitted_k": ks, "extrapolated": {str(k): float(np.exp(coef[0] + coef[1] * k)) for k in (8, 9, 10, 11)}}
    return {"by_ovals": {str(k): by_k[k] for k in sorted(by_k)}, "counts_by_ovals": {str(k): cnt_k[k]
                                                                                    for k in sorted(cnt_k)},
            "fit": fit}


def certified_type(k, sign_N, sep_signs):
    b = sum(1 for s in sep_signs if s == sign_N) - 1
    a = k - b - (1 if b > 0 else 0)
    return type_name(16 * a + b) if a >= 0 and b >= 0 else None


def certify_task(task):
    """Worker (CPU only): exact certification of one integer sextic. task = (id, coefs, k, witnesses, seed)."""
    from dataclasses import asdict

    import random

    from conncomp.certify import (certify_upper_bound, certify_upper_bound_interior, crosscheck_flint, form_poly,
                                  line_real_roots)
    from conncomp.rationalize import eval_form
    from conncomp.separation import certify_lower_bound

    tid, c, k, wit, seed = task
    t0 = time.time()
    try:
        H = form_poly(c, DEG)
        ub = certify_upper_bound(H, target=k, seed=seed)
        if ub is None or not ub.certified:  # the loop certificate (O in the root region N) is needed for the type
            return tid, {"certified": False, "reason": "upper bound", "seconds": time.time() - t0}
        lb = certify_lower_bound(H, wit)
        sep = lb.certificate
        upper, pencil, interior = ub.upper_bound, ub, None
        if upper > sep.lower_bound:  # loose (non-convex ovals): base points inside ovals, e.g. inside a nest
            interior = certify_upper_bound_interior(H, wit, target=sep.lower_bound, seed=seed)
            if interior is not None and interior.upper_bound < upper:
                upper, pencil = interior.upper_bound, interior.pencil
        flint = crosscheck_flint(H, pencil)
        ok = bool(flint["agree"]) and sep.lower_bound == upper
        out = {"certified": ok, "upper_bound": upper, "lower_bound": sep.lower_bound,
               "flint_agree": bool(flint["agree"]), "base_point": [str(v) for v in ub.base_point],
               "upper_method": "pencil" if interior is None or pencil is ub else "interior pencil",
               "seconds": time.time() - t0}
        if ok:
            sO = eval_form(c, DEG, [int(v) for v in ub.base_point])
            sO = (sO > 0) - (sO < 0)
            signs = [sep.signs[i] for i in sep.separated]
            out["type"] = certified_type(upper, sO, signs)
            out["nesting"] = {"sign_N": sO, "separated_signs": signs}
            if out["type"] == "1 u 1<1>":  # same signs as the depth-3 nest 1<1<1>>: lines through the -sign(N)
                rng = random.Random(seed)  # witnesses meeting C in <= 4 points exclude it (>= 6 for the innermost)
                lines = []
                for i in sep.separated:
                    if sep.signs[i] == sO:
                        continue
                    Q = next((Q for Q in ([rng.randint(-9, 9) for _ in range(3)] for _ in range(60))
                              if line_real_roots(H, sep.witnesses[i], Q) <= 4), None)
                    lines.append(Q)
                out["nesting"]["depth3_excluded_by_lines"] = lines
                if any(Q is None for Q in lines):
                    out["certified"], out["type"] = False, None
            out["upper"] = {kk: v for kk, v in asdict(ub).items() if kk != "resultant"}
            if out["upper_method"] == "interior pencil":
                out["upper_interior"] = {"pencil": {kk: v for kk, v in asdict(interior.pencil).items()
                                                    if kk != "resultant"},
                                         "line_point": interior.line_point, "n_line_roots": interior.n_line_roots,
                                         "upper_bound": interior.upper_bound}
            out["lower"] = asdict(sep)
        return tid, out
    except Exception as e:  # keep the pool alive
        return tid, {"certified": False, "reason": f"error: {e!r}", "seconds": time.time() - t0}


def to_float_columns(vs):
    a = np.array(vs, dtype=np.float64)
    a = a / np.abs(a).max(axis=1, keepdims=True)
    return torch.as_tensor(a.T, dtype=DTYPE, device=DEVICE)


def integer_candidates(f, scales):
    f = f / np.abs(f).max()
    out, seen = [], set()
    for s in scales:
        v = tuple(int(x) for x in np.rint(s * f))
        g = math.gcd(*v)
        v = tuple(x // g for x in v) if g > 1 else v
        if v not in seen and any(v):
            seen.add(v)
            out.append((list(v), s))
    return out


def diverse(F, n, tol=0.95):
    U = F / np.linalg.norm(F, axis=1, keepdims=True)
    keep = []
    for i in range(len(F)):
        if all(abs(U[i] @ U[j]) < tol for j in keep):
            keep.append(i)
        if len(keep) == n:
            break
    return keep


def representatives(args, run, pools):
    """Integer representatives per type: rounding, grid checks (GPU), exact certification (CPU workers)."""
    import sympy as sp

    from conncomp.polynomials import to_sympy
    from conncomp.rationalize import witnesses

    ref, mid, top = FaceMesh(150), FaceMesh(100), FaceMesh(200)
    scales = [4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400, 512, 640,
              800, 1024, 1280, 1600, 2048]
    plans = {}  # code -> list of (pool index, [(int vector, scale), ...] passing the grid checks)
    for k in sorted(pools):
        s, F, _ = pools[k]
        if k == OTHER or len(s) == 0:
            continue
        order = np.argsort(-s)
        okr = (classify(ref, torch.as_tensor(F[order].T, device=DEVICE)) == k).cpu().numpy()
        order = order[okr]
        plans[k] = []
        for i in [order[j] for j in diverse(F[order], args.cand_reps)]:
            cands = integer_candidates(F[i], scales)
            cf = to_float_columns([v for v, _ in cands])
            good = ((classify(mid, cf) == k) & (classify(top, cf) == k)).cpu().numpy()
            passing = [c for c, g in zip(cands, good) if g][:3]
            if passing:
                plans[k].append((int(i), passing))
    # rounds: attempt r uses the r-th passing scale of each candidate not yet certified
    reps = {k: [] for k in plans}
    certs = {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for attempt in range(3):
            tasks, meta = [], {}
            for k, plan in plans.items():
                if len(reps[k]) >= args.reps:
                    continue
                done = {r["pool_index"] for r in reps[k]}
                for i, passing in plan:
                    if i in done or attempt >= len(passing):
                        continue
                    v, sc = passing[attempt]
                    wit = [w["point"] for w in witnesses(v, DEG, 800)]
                    tid = len(certs) + len(tasks)
                    tasks.append((tid, v, n_ovals(k), wit, attempt))
                    meta[tid] = (k, i, v, sc)
            if not tasks:
                break
            print(f"certification round {attempt + 1}: {len(tasks)} forms on {args.workers} workers", flush=True)
            asyncs = [pool.apply_async(certify_task, (t,)) for t in tasks]
            results = {}
            for a, t in zip(asyncs, tasks):
                try:
                    tid, out = a.get(timeout=args.cert_timeout)
                except mp.TimeoutError:
                    tid, out = t[0], {"certified": False, "reason": "timeout"}
                results[tid] = out
            for tid in sorted(results):
                out = results[tid]
                k, i, v, sc = meta[tid]
                certs[tid] = out
                ok = out.get("certified") and out.get("type") == type_name(k)
                print(f"  {type_name(k):12s} pool #{i} height {max(map(abs, v))}: "
                      f"{out.get('type', out.get('reason', 'not certified'))} ({out.get('seconds', 0):.0f} s)",
                      flush=True)
                if ok and len(reps[k]) < args.reps:
                    x, y = sp.symbols("x y")
                    rep = {"type": type_name(k), "ovals": n_ovals(k), "f": v, "height": max(map(abs, v)),
                           "f_affine": str(sp.expand(to_sympy(v, DEG, x, y, 1))), "pool_index": i,
                           "robustness": float(pools[k][0][i]),
                           "certificate": {kk: out[kk] for kk in ("upper_bound", "upper_method", "lower_bound",
                                                                  "flint_agree", "base_point", "nesting")}}
                    reps[k].append(rep)
                    (run / f"cert_{tag(k)}_{len(reps[k])}.json").write_text(
                        json.dumps({"rep": rep, "certificate": out}, indent=1, default=str))
        pool.terminate()
    return {type_name(k): r for k, r in reps.items()}, {type_name(k): len(p) for k, p in plans.items()}


def finish(args, run):
    rng = np.random.default_rng(0)
    sa, ga = load_npz(run / "phase_a.npz")
    counts, res, pool_a = sa["counts"], ga.get("res", {}), ga.get("pool", {})
    sb, gb = load_npz(run / "phase_b.npz") if (run / "phase_b.npz").exists() else ({}, {})
    archive = gb.get("cell", {})
    first = json.loads(str(sb["first"])) if "first" in sb else {}
    ref = FaceMesh(150)
    summary = {"N": args.N, "phase_a": {"forms": int(counts.sum()), "minutes": float(sa["elapsed"]) / 60,
                                        "types": int((counts.sum(0)[:OTHER] > 0).sum())},
               "phase_b": {"children": int(sb.get("children", 0)), "minutes": float(sb.get("elapsed", 0)) / 60,
                           "types": len(archive), "first_found": first}, "measures": {}}
    for m, name in enumerate(MEASURES):
        conf = np.zeros((NC, NC), dtype=np.int64)
        for (mm, k), (_, f) in res.items():
            if mm == m and len(f):
                conf[k] = torch.bincount(classify(ref, torch.as_tensor(f.T, device=DEVICE)), minlength=NC).cpu().numpy()
        active = [k for k in range(NC) if counts[m, k] > 0 or conf[:, k].sum() > 0]
        vol = volumes(counts[m], conf, active, rng)
        summary["measures"][name] = {"forms": int(counts[m].sum()), "volumes": vol, "tail": tail_fit(vol),
                                     "confusion_N150": {f"{type_name(i)} -> {type_name(j)}": int(conf[i, j])
                                                        for i in active for j in active if conf[i, j]}}
        print(f"{name}: {len(active)} types; by ovals: {summary['measures'][name]['tail']['by_ovals']}", flush=True)
    # pools for the representatives: phase A pools merged with the phase B archive
    pools = dict(pool_a)
    for k, cell in archive.items():
        pools[k] = merge_pool(pools.get(k, empty_pool()), *cell, args.pool + args.cell)
    top = FaceMesh(400)  # artifacts were seen to survive N = 200

    def confirmed(g, k):
        if k not in g or len(g[k][0]) == 0:
            return 0
        return int((classify(top, torch.as_tensor(g[k][1].T, device=DEVICE), batch=8) == k).sum())

    summary["pools"] = {type_name(k): {"ovals": n_ovals(k), "phase_a": int(len(pool_a.get(k, empty_pool())[0])),
                                       "phase_b": int(len(archive.get(k, empty_pool())[0])),
                                       "phase_a_confirmed_N400": confirmed(pool_a, k),
                                       "phase_b_confirmed_N400": confirmed(archive, k),
                                       "best_robustness": float(pools[k][0].max())} for k in sorted(pools)}
    for p in summary["pools"].values():
        p["genuine"] = p["phase_a_confirmed_N400"] + p["phase_b_confirmed_N400"] > 0
    reps, n_plans = representatives(args, run, pools)
    summary["representatives"] = {t: len(r) for t, r in reps.items()}
    summary["candidates_passing_grid"] = n_plans
    (run / "summary.json").write_text(json.dumps(summary, indent=1))
    out = Path("data/sextic_types")
    out.mkdir(parents=True, exist_ok=True)
    desc = ("Integer plane sextics (coefficients in the order of conncomp.polynomials.monomials(6)) of each nesting "
            "type found; oval count certified exactly (pencil tangents + separating polygons, FLINT cross-check); "
            "nesting from the signs of the pairwise separated witnesses and of the certified base point.")
    (out / "plain.json").write_text(json.dumps({"description": desc, "representatives": reps}, indent=1))
    write_markdown(run / "summary.md", summary)
    print((run / "summary.md").read_text())


def write_markdown(path, s):
    a, b = s["phase_a"], s["phase_b"]
    lines = ["# Plane sextics: nesting types", "",
             f"Phase A: {a['forms']:,} random forms in {a['minutes']:.1f} min, {a['types']} types. "
             f"Phase B: {b['children']:,} children in {b['minutes']:.1f} min, {b['types']} types in the archive.", ""]
    for name, m in s["measures"].items():
        lines += [f"## {name} ({m['forms']:,} forms; corrected against N = 150)", "",
                  "| type | ovals | count | corrected fraction | 95% CI |", "|---|---|---|---|---|"]
        rows = sorted(m["volumes"].items(), key=lambda kv: (kv[1]["ovals"] if kv[1]["ovals"] is not None else 99,
                                                            -kv[1]["count"]))
        for t, v in rows:
            lo, hi = v["corrected_ci95"]
            lines.append(f"| {t} | {v['ovals'] if v['ovals'] is not None else '-'} | {v['count']:,} | "
                         f"{v['corrected']:.3g} | {lo:.2g} – {hi:.2g} |")
        fit = m["tail"]["fit"]
        if fit:
            ext = ", ".join(f"{k}: {v:.1e}" for k, v in fit["extrapolated"].items())
            lines += ["", f"Tail: factor {fit['factor_per_oval']:.1f} per oval (fit over k = {fit['fitted_k']}); "
                          f"extrapolated fractions {ext}."]
        lines.append("")
    lines += ["## Types, pools and certified representatives", "",
              "Pool sizes as confirmed at N = 400 / stored; types with nothing confirmed are grid artifacts.", "",
              "| type | ovals | found in | pool A | archive B | best robustness | certified |",
              "|---|---|---|---|---|---|---|"]
    for t, p in sorted(s["pools"].items(), key=lambda kv: (kv[1]["ovals"], kv[0])):
        fi = s["phase_b"]["first_found"].get(t, {})
        src = "A" if fi.get("source", "A") == "A" else f"B ({fi['minutes']:.0f} min)"
        name = t if p["genuine"] else f"~~{t}~~ (artifact)"
        lines.append(f"| {name} | {p['ovals']} | {src} | {p['phase_a_confirmed_N400']}/{p['phase_a']} | "
                     f"{p['phase_b_confirmed_N400']}/{p['phase_b']} | {p['best_robustness']:.2g} | "
                     f"{s['representatives'].get(t, 0)} |")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["all", "a", "b", "finish"], default="all")
    ap.add_argument("--minutes-a", type=float, default=70)
    ap.add_argument("--minutes-b", type=float, default=45)
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--batch", type=int, default=50000)
    ap.add_argument("--reservoir", type=int, default=2000)
    ap.add_argument("--pool", type=int, default=500, help="most robust forms per type in phase A")
    ap.add_argument("--cell", type=int, default=500, help="archive cell size in phase B")
    ap.add_argument("--cand", type=int, default=64, help="candidates scored per type and batch")
    ap.add_argument("--flush", type=int, default=4000)
    ap.add_argument("--min-score", type=float, default=3e-4,
                    help="minimum robustness of children with >= 5 ovals or >= 2 inner ovals (B)")
    ap.add_argument("--min-angle", type=float, default=0.03, help="minimum angle between members of a cell (B)")
    ap.add_argument("--rmin", type=float, default=1e-3)
    ap.add_argument("--rmax", type=float, default=1e-1)
    ap.add_argument("--checkpoint", type=float, default=10)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cand-reps", type=int, default=6, help="pool forms tried per type for representatives")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--cert-timeout", type=float, default=1800)
    ap.add_argument("--out", type=Path, default=Path("pools/sextic_plain"))
    args = ap.parse_args()
    run = args.out
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(time.time()))
    if args.stage in ("all", "a"):
        phase_a(args, run)
    if args.stage in ("all", "b"):
        _, g = load_npz(run / "phase_a.npz")
        phase_b(args, run, g.get("pool", {}))
    if args.stage in ("all", "finish"):
        finish(args, run)


if __name__ == "__main__":
    main()
