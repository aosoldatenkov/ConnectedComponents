#!/usr/bin/env python3
"""Targeted search for new nesting types of plane sextics by oval births, starting from the phase A pools of
scripts/sextic_types.py.

Mechanism. The topology of {f = 0} changes only when a critical value v of f on RP^2 crosses 0. An extremum on the
"wrong" side (a maximum with v < 0, a minimum with v > 0) gives a birth: a new empty oval around the critical point,
inside the region that contains it. More generally, pushing the value at any point q of a region R through zero
(while the other critical values keep their signs) creates a new oval inside R. The resulting type is predicted from
the region tree (sextic types a u 1<b>):
    R = root N            -> (a + 1) u 1<b>
    R = empty oval disk   -> (a - 1) u 1<1>         (only if b = 0: a sextic has at most one nest)
    R = annulus of a nest -> a u 1<b + 1>
    R = inner disk        -> 1<1<1>> (only for 1<1>; not representable here, skipped)
Saddle crossings (merges, splits, nesting changes) are tried with a low weight; their outcome is not predicted.

Events (form, point q, predicted type) come from annotating the archive: critical points (kind, value, wall distance
|v| / |grad v|) and random interior points of each region (needed for nested births inside an oval whose interior has
a single extremum, the "two-stage" birth). Events are sampled with weight 4^ovals(target) x (10 if the target type is
new) / sqrt(1 + cell size), times a preference for extrema close to their wall.

Step (batched): minimal-norm change of the unit coefficient vector c with
    f(q) -> -sign(f(q)) delta,     v_j unchanged for the nearby critical points j (|v_j| small, up to `keep`),
delta log-uniform in [dmin, dmax]; linear constraints (rows m(q) - f(q) c, m(p_j) - v_j c), two Gauss-Newton steps.
Children are classified on FaceMesh(30); risky types (>= 5 ovals or >= 2 inner ovals) must be confirmed at N = 200
and have robustness >= min_score (grid artifacts, see project notes). Archive cells keep up to `cell` forms, greedily
by robustness with a minimum angle between members. Prediction accuracy is logged per event kind.

    uv run python scripts/sextic_births.py --minutes 60
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import sextic_types as S
from conncomp import DEVICE, DTYPE
from conncomp.critical import MAX, MIN, SADDLE, FaceCharts, critical_points, monomial_values
from conncomp.gpu_trees import FaceMesh

DEG = 6
KINDS = ["extremum", "crater", "saddle", "interior", "random"]
RING = 8


def tangent_frame(p):
    """Orthonormal tangent vectors (e1, e2) at the unit points p (K, 3)."""
    a = torch.zeros_like(p)
    a[torch.arange(len(p)), p.abs().argmin(dim=1)] = 1.0
    e1 = torch.linalg.cross(p, a)
    e1 = e1 / e1.norm(dim=1, keepdim=True)
    return e1, torch.linalg.cross(p, e1)


def distance_to_curve(Fn, form, p, n_rays=8, n_radii=40, chunk=1024):
    """Angular distance from the points p (K, 3) to the curve of the forms Fn[:, form]: the first sign change along
    n_rays great-circle rays (radii log-spaced in [0.005, 1])."""
    radii = torch.logspace(math.log10(0.005), 0, n_radii, dtype=DTYPE, device=DEVICE)
    th = torch.arange(n_rays, dtype=DTYPE, device=DEVICE) * (2 * math.pi / n_rays)
    out = torch.empty(len(p), dtype=DTYPE, device=DEVICE)
    for s0 in range(0, len(p), chunk):
        q, c = p[s0 : s0 + chunk], Fn.T[form[s0 : s0 + chunk]]
        e1, e2 = tangent_frame(q)
        dirs = torch.cos(th)[None, :, None] * e1[:, None] + torch.sin(th)[None, :, None] * e2[:, None]
        pts = q[:, None, None] + torch.tan(radii)[None, None, :, None] * dirs[:, :, None]  # (k, rays, radii, 3)
        vals = (monomial_values(pts.reshape(-1, 3), DEG).reshape(len(q), -1, c.shape[1]) * c[:, None]).sum(-1)
        s_q = torch.sign((monomial_values(q, DEG) * c).sum(1))
        changed = (torch.sign(vals) != s_q[:, None]).reshape(len(q), n_rays, n_radii)
        first = torch.where(changed.any(2), changed.float().argmax(2), torch.full(changed.shape[:2], n_radii - 1,
                                                                                  device=DEVICE))
        out[s0 : s0 + chunk] = radii[first.min(1).values]
    return out


def node_of(fm, P):
    """Local face-grid node index of the points P (K, 3) (nearest grid point on the cube face)."""
    N, M = fm.N, fm.M
    axis = P.abs().argmax(dim=1)
    pa = P.gather(1, axis[:, None])[:, 0]
    side = (pa > 0).long()
    o = torch.tensor([[1, 2], [0, 2], [0, 1]], device=P.device)[axis]
    s = N / pa.abs()
    i = torch.round(P.gather(1, o[:, :1])[:, 0] * s).long().clamp(-N, N) + N
    j = torch.round(P.gather(1, o[:, 1:])[:, 0] * s).long().clamp(-N, N) + N
    return ((2 * axis + side) * M + i) * M + j


def predicted_code(code, depth, has_child):
    """Type after a birth inside a region (depth, has children) of a form of type `code`; -1 if not representable."""
    a, b = divmod(int(code), 16)
    if depth == 0:
        return 16 * (a + 1) + b if a + 1 < 16 else -1
    if depth == 1 and not has_child:
        return 16 * (a - 1) + 1 if b == 0 and a >= 1 else -1
    if depth == 1 and has_child:
        return 16 * a + b + 1 if b + 1 < 16 else -1
    return -1


class Annotation:
    """Events of a set of forms F (D, n): per form its critical points, per event the target point and prediction."""

    def __init__(self, fm, charts, F, codes, n_random=16):
        n = F.shape[1]
        Fn = F / F.norm(dim=0, keepdim=True)
        crit, morse = critical_points(Fn, DEG, charts=charts)
        crit = crit.select(morse[crit.form] == 1)
        # wall distances |v| / |grad v| on the unit sphere of coefficients
        m = monomial_values(crit.point, DEG)
        g = m - crit.value[:, None] * Fn.T[crit.form]
        d = crit.value.abs() / g.norm(dim=1).clamp(min=1e-300)
        # regions of the critical points and of random nodes
        K = len(crit)
        V = fm.n_loc
        qn = torch.randint(0, V, (n * n_random,), device=DEVICE)
        qf = torch.arange(n, device=DEVICE).repeat_interleave(n_random)
        depth, has_child, alive = self._regions(fm, Fn, torch.cat([crit.form, qf]),
                                                torch.cat([node_of(fm, crit.point), qn]))
        rdepth, rhas, ralive = depth[K:].cpu().numpy(), has_child[K:].cpu().numpy(), alive[K:].cpu().numpy()
        depth, has_child, alive = depth[:K], has_child[:K], alive[:K]
        self.F = Fn
        self.codes = codes
        # per-form critical points (for the keep constraints), padded
        counts = torch.bincount(crit.form, minlength=n)
        start = torch.cumsum(counts, 0) - counts
        pos = torch.arange(K, device=DEVICE) - start[crit.form]
        kmax = max(int(counts.max()) if K else 1, 1)
        self.cp = torch.zeros((n, kmax, 3), dtype=DTYPE, device=DEVICE)
        self.cv = torch.full((n, kmax), float("inf"), dtype=DTYPE, device=DEVICE)
        self.cp[crit.form, pos] = crit.point
        self.cv[crit.form, pos] = crit.value
        # events: wrong-side extrema (direct birth), other extrema (crater inside their region), saddles
        cd, ch, ca = depth.cpu().numpy(), has_child.cpu().numpy(), alive.cpu().numpy()
        kind, cf = crit.kind.cpu().numpy(), crit.form.cpu().numpy()
        val = crit.value.cpu().numpy()
        cod = codes.cpu().numpy()
        wrong = ((kind == MAX) & (val < 0)) | ((kind == MIN) & (val > 0))
        ext = kind != SADDLE
        pred = np.array([predicted_code(cod[cf[k]], cd[k], ch[k]) if ext[k] and ca[k] else -1 for k in range(K)])
        ev_kind = np.where(kind == SADDLE, 2, np.where(wrong, 0, 1))
        keep = (kind == SADDLE) | (pred >= 0)
        self.form = cf[keep]
        self.point = crit.point.cpu().numpy()[keep]
        self.kind = ev_kind[keep]
        self.pred = np.where(self.kind == 2, -1, pred[keep])
        self.dist = d.cpu().numpy()[keep]
        # crater radius: 0.9 x the distance from the extremum to the curve (first sign change along 8 rays)
        self.radius = np.zeros(keep.sum())
        sel = np.nonzero(self.kind == 1)[0]
        if len(sel):
            p = torch.as_tensor(self.point[sel], dtype=DTYPE, device=DEVICE)
            fp = torch.as_tensor(self.form[sel], device=DEVICE)
            self.radius[sel] = 0.9 * distance_to_curve(Fn, fp, p).cpu().numpy()
        # random interior points (pushed without a ring), one per (form, region kind); one random move per form
        rf = np.repeat(np.arange(n), n_random)
        rp = fm.points[qn].cpu().numpy()
        seen, extra = set(), []
        for k in np.random.permutation(len(rf)):
            key = (rf[k], rdepth[k], rhas[k])
            if ralive[k] and key not in seen:
                pr = predicted_code(cod[rf[k]], rdepth[k], rhas[k])
                if pr >= 0:
                    seen.add(key)
                    extra.append((rf[k], rp[k], 3, pr))
        extra += [(f, np.array([0.0, 0.0, 1.0]), 4, -1) for f in range(n)]
        self.form = np.concatenate([self.form, [e[0] for e in extra]]).astype(np.int64)
        self.point = np.concatenate([self.point, np.array([e[1] for e in extra]).reshape(-1, 3)])
        self.kind = np.concatenate([self.kind, [e[2] for e in extra]]).astype(np.int64)
        self.pred = np.concatenate([self.pred, [e[3] for e in extra]]).astype(np.int64)
        self.dist = np.concatenate([self.dist, np.full(len(extra), np.nan)])
        self.radius = np.concatenate([self.radius, np.zeros(len(extra))])

    @staticmethod
    def _regions(fm, Fn, q_form, q_node):
        """(depth, has children, alive) of the regions containing the given nodes (forms q_form)."""
        signs = fm.signs(Fn, DEG)
        L = fm.labels(signs)
        t = fm._trees_cuda(signs, L, 3, 1)
        V = fm.n_loc
        n = Fn.shape[1]
        Lf = L.reshape(-1).long()
        anti = fm.anti
        nodes = torch.arange(n * V, device=DEVICE)
        roots = nodes[Lf == nodes]
        aroot = Lf[(roots // V) * V + anti[roots % V]]
        keys = roots[roots <= aroot]  # one key component per region, in the region order of the trees
        if keys.numel() != t.form.numel():
            raise RuntimeError("region order mismatch (overflowing forms in the batch)")
        q = q_form * V + q_node
        r = Lf[q]
        ra = Lf[(r // V) * V + anti[r % V]]
        gid = torch.searchsorted(keys, torch.minimum(r, ra))
        nchild = torch.bincount(t.parent[t.parent >= 0], minlength=t.form.numel())
        return t.depth[gid], nchild[gid] > 0, t.alive[gid] & (t.depth[gid] >= 0)


def push(F, form, point, delta, cp, cv, keep, radius, ring_goal=1.0, cap=None, steps=2):
    """Batched minimal-norm steps on the unit coefficient vectors: f(q) -> -sign(f(q)) delta, with f held fixed at
    RING points at angular distance `radius` around q (craters, values scaled by ring_goal; radius 0: no ring) and
    at the nearby critical points (critical values unchanged, up to `keep`). Returns (D, P)."""
    c = F.T[form]  # (P, D) unit vectors
    P_ = len(form)
    q = torch.as_tensor(point, dtype=DTYPE, device=DEVICE)
    q = q / q.norm(dim=1, keepdim=True)
    rad = torch.as_tensor(radius, dtype=DTYPE, device=DEVICE)
    mq = monomial_values(q, DEG)
    fq0 = (mq * c).sum(1)
    # ring points
    e1, e2 = tangent_frame(q)
    th = torch.arange(RING, dtype=DTYPE, device=DEVICE) * (2 * math.pi / RING)
    ring = q[:, None] + torch.tan(rad)[:, None, None] * (torch.cos(th)[None, :, None] * e1[:, None]
                                                        + torch.sin(th)[None, :, None] * e2[:, None])
    mr = monomial_values(ring.reshape(-1, 3), DEG).reshape(P_, RING, -1)
    mr = mr / ring.norm(dim=2, keepdim=True) ** DEG  # values at the unit points
    use_r = (rad > 0)[:, None].expand(-1, RING)
    rv = (mr * c[:, None]).sum(2)
    # nearby critical points of each form (excluding the target point itself)
    pts, vals = cp[form], cv[form].clone()  # (P, k, 3), (P, k)
    same = ((pts - q[:, None]).norm(dim=2) < 1e-6) | ((pts + q[:, None]).norm(dim=2) < 1e-6)
    vals[same] = float("inf")
    order = vals.abs().argsort(dim=1)[:, :keep]
    kp = pts.gather(1, order[..., None].expand(-1, -1, 3))
    kv = vals.gather(1, order)
    tol = torch.maximum(3 * (fq0.abs() + delta), torch.full_like(delta, 1e-2))
    use_k = kv.abs() < tol[:, None]  # padded entries (inf) are never used
    kv = torch.where(use_k, kv, torch.zeros_like(kv))
    mk = monomial_values(kp.reshape(-1, 3), DEG).reshape(P_, keep, -1)
    Mx = torch.cat([mq[:, None], mr, mk], dim=1)  # (P, 1 + RING + keep, D)
    goal = torch.cat([(-torch.sign(fq0) * delta)[:, None], ring_goal * rv, kv], dim=1)
    mask = torch.cat([torch.ones_like(use_k[:, :1]), use_r, use_k], dim=1).to(DTYPE)
    eye = torch.eye(Mx.shape[1], dtype=DTYPE, device=DEVICE)
    for _ in range(steps):
        fx = (Mx * c[:, None]).sum(2)
        J = (Mx - fx[..., None] * c[:, None]) * mask[..., None]  # tangent to the unit sphere
        r = (goal - fx) * mask
        A = J @ J.transpose(1, 2)
        scale = torch.diagonal(A, dim1=1, dim2=2).amax(1).clamp(min=1e-300)
        A = A + (1e-10 * scale[:, None, None] + torch.diag_embed(1 - mask)) * eye
        lam = torch.linalg.solve(A, r[..., None])
        dc = (J.transpose(1, 2) @ lam)[..., 0]
        if cap is not None:  # continuation: at most `cap` relative change per step
            dc = dc * torch.clamp(cap / dc.norm(dim=1, keepdim=True).clamp(min=1e-300), max=1.0)
        c = c + dc
        c = c / c.norm(dim=1, keepdim=True)
    return c.T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--pairs", type=int, default=8192, help="events per step")
    ap.add_argument("--cell", type=int, default=300)
    ap.add_argument("--cand", type=int, default=64, help="children scored per type and step")
    ap.add_argument("--keep", type=int, default=12, help="nearby critical values held fixed")
    ap.add_argument("--dmin", type=float, default=1e-3)
    ap.add_argument("--dmax", type=float, default=3e-2)
    ap.add_argument("--rmin", type=float, default=1e-3, help="random moves: radius log-uniform in [rmin, rmax]")
    ap.add_argument("--rmax", type=float, default=1e-1)
    ap.add_argument("--ring-goal", type=float, default=0.5, help="crater rim values, relative to the current ones")
    ap.add_argument("--min-score", type=float, default=3e-4)
    ap.add_argument("--min-angle", type=float, default=0.03)
    ap.add_argument("--reannotate", type=int, default=20, help="steps between re-annotations of the archive")
    ap.add_argument("--checkpoint", type=float, default=5)
    ap.add_argument("--pools", type=Path, default=Path("pools/sextic_plain/phase_a.npz"))
    ap.add_argument("--out", type=Path, default=Path("pools/sextic_births"))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cand-reps", type=int, default=6)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--cert-timeout", type=float, default=1800)
    ap.add_argument("--no-certify", action="store_true")
    args = ap.parse_args()
    run = args.out
    run.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng()
    torch.manual_seed(int(time.time()))
    fm, charts, confirm = FaceMesh(30), FaceCharts(DEG, N=40), FaceMesh(200)

    def risky(k):
        return S.n_ovals(k) >= 5 or k % 16 >= 2

    def insert(k, F, cv):
        """Cells keep a random diverse sample of genuine forms (key = uniform random; robustness = cv[:, 0])."""
        archive[k] = S.merge_pool(archive.get(k, S.empty_pool()), rng.random(len(F)), F, cv, args.cell,
                                  args.min_angle)

    # --- initial archive: the phase A pools, with the artifact filters
    _, ga = S.load_npz(args.pools)
    archive = {}
    for k, (s, F, cv) in ga["pool"].items():
        if k == S.OTHER or len(s) == 0:
            continue
        ok = s >= args.min_score
        if risky(k):
            ok &= (S.classify(confirm, torch.as_tensor(F.T, device=DEVICE), batch=40) == k).cpu().numpy()
        if ok.any():
            insert(k, F[ok], cv[ok])
    first = {S.type_name(k): {"minutes": 0.0, "source": "A"} for k in archive}
    print(f"initial archive: {len(archive)} types: " + ", ".join(sorted(first, key=lambda t: (len(t), t))), flush=True)

    stats = {k: np.zeros(5, dtype=np.int64) for k in KINDS}  # tried, changed type, as predicted, accepted, new type

    def annotate():
        cells = sorted(archive)
        F = np.concatenate([archive[k][1] for k in cells])
        codes = np.concatenate([np.full(len(archive[k][0]), k) for k in cells])
        return Annotation(fm, charts, torch.as_tensor(F.T, dtype=DTYPE, device=DEVICE),
                          torch.as_tensor(codes, device=DEVICE))

    def weights(ann):
        size = {k: len(v[0]) for k, v in archive.items()}
        par = ann.codes.cpu().numpy()[ann.form]
        w = np.empty(len(ann.form))
        for i in range(len(w)):
            kd, pr = ann.kind[i], ann.pred[i]
            if pr < 0:  # saddles, random moves: outcome unknown
                base = 0.5 * 4.0 ** S.n_ovals(int(par[i])) / math.sqrt(size[int(par[i])])
            else:
                base = 4.0 ** S.n_ovals(int(pr)) * (10.0 if pr not in archive else 1.0 / math.sqrt(1 + size[pr]))
            if kd in (0, 2):
                base /= 1.0 + ann.dist[i] / 0.01
            w[i] = base
        return w / w.sum()

    t0 = time.time()
    t_ck = t0
    ann = annotate()
    w = weights(ann)
    step, children = 0, 0
    while time.time() - t0 < args.minutes * 60:
        sel = rng.choice(len(w), args.pairs, p=w)
        kinds = ann.kind[sel]
        form = torch.as_tensor(ann.form[sel], device=DEVICE)
        delta = torch.exp(torch.empty(args.pairs, dtype=DTYPE, device=DEVICE).uniform_(math.log(args.dmin),
                                                                                     math.log(args.dmax)))
        child = push(ann.F, form, ann.point[sel], delta, ann.cp, ann.cv, args.keep, ann.radius[sel], args.ring_goal)
        rnd = torch.as_tensor(kinds == 4, device=DEVICE)
        if rnd.any():
            g = torch.randn((child.shape[0], int(rnd.sum())), dtype=DTYPE, device=DEVICE)
            r = torch.exp(torch.empty(g.shape[1], dtype=DTYPE, device=DEVICE).uniform_(math.log(args.rmin),
                                                                                       math.log(args.rmax)))
            c = ann.F[:, form[rnd]] + r * g / g.norm(dim=0, keepdim=True)
            child[:, rnd] = c / c.norm(dim=0, keepdim=True)
        ok = torch.isfinite(child).all(0)
        code = torch.full((args.pairs,), S.OTHER, dtype=torch.long, device=DEVICE)
        if ok.any():
            code[ok] = S.classify(fm, child[:, ok])
        children += args.pairs
        code_np, pc_np = code.cpu().numpy(), ann.codes[form].cpu().numpy()
        for kd, name in enumerate(KINDS):
            m = kinds == kd
            stats[name][:3] += [m.sum(), (m & (code_np != pc_np)).sum(), (m & (code_np == ann.pred[sel])).sum()]
        new_cell = False
        for k in np.unique(code_np):
            if k == S.OTHER:
                continue
            idx = np.nonzero(code_np == k)[0]
            new = k not in archive
            if not new:  # existing cells: a random subset; new types: all candidates
                idx = rng.permutation(idx)[: args.cand]
            f = child[:, torch.as_tensor(idx, device=DEVICE)]
            if risky(k):
                good = (S.classify(confirm, f, batch=40) == k).cpu().numpy()
                idx, f = idx[good], f[:, torch.as_tensor(good, device=DEVICE)]
            if len(idx) == 0:
                continue
            sc, cv = S.robustness(f, charts)
            good = (sc >= args.min_score).cpu().numpy()
            idx, f, cv = idx[good], f[:, torch.as_tensor(good, device=DEVICE)], cv[torch.as_tensor(good,
                                                                                                device=DEVICE)]
            if len(idx) == 0:
                continue
            for kd, name in enumerate(KINDS):
                stats[name][3] += (kinds[idx] == kd).sum()
                if new:
                    stats[name][4] += (kinds[idx] == kd).sum()
            insert(k, f.T.cpu().numpy(), cv.cpu().numpy())
            if new:
                mins = (time.time() - t0) / 60
                via = sorted({KINDS[x] for x in kinds[idx]})
                first[S.type_name(k)] = {"minutes": round(mins, 2), "source": "births", "via": via}
                print(f"  [{mins:5.1f} min] new type {S.type_name(k)} ({S.n_ovals(k)} ovals) via {via}", flush=True)
                new_cell = True
        step += 1
        if new_cell or step % args.reannotate == 0:
            ann = annotate()
            w = weights(ann)
        if time.time() - t_ck > args.checkpoint * 60 or time.time() - t0 >= args.minutes * 60:
            t_ck = time.time()
            el = time.time() - t0
            top = sorted(archive, key=lambda k: (-S.n_ovals(k), k))[:6]
            st = "\n    ".join(f"{k:8s}: {v[0]:>10,} tried, {v[1] / max(v[0], 1):5.1%} changed, "
                               f"{v[2] / max(v[0], 1):5.1%} as predicted, {v[3]:,} accepted ({v[4]:,} new type)"
                               for k, v in stats.items())
            print(f"[{el / 60:5.1f} min] {step} steps, {children:,} children ({children / el:,.0f}/s), "
                  f"{len(archive)} types; highest: " + ", ".join(f"{S.type_name(k)} ({len(archive[k][0])})"
                                                               for k in top) + f"\n    {st}", flush=True)
            S.save_npz(run / "archive.npz", {"elapsed": el, "children": children, "first": json.dumps(first)},
                       {"cell": archive})
    el = time.time() - t0
    summary = {"minutes": el / 60, "children": children, "steps": step,
               "types": {S.type_name(k): {"ovals": S.n_ovals(k), "size": int(len(v[0])),
                                          "best_robustness": float(v[2][:, 0].max())} for k, v in archive.items()},
               "first_found": first,
               "stats": {k: dict(zip(["tried", "changed", "as_predicted", "accepted", "new_type"], map(int, v)))
                         for k, v in stats.items()}}
    S.save_npz(run / "archive.npz", {"elapsed": el, "children": children, "first": json.dumps(first)},
               {"cell": archive})
    new_types = {k: (v[2][:, 0], v[1], v[2]) for k, v in archive.items() if first[S.type_name(k)]["source"] != "A"}
    print(f"done: {children:,} children in {el / 60:.1f} min; {len(archive)} types, {len(new_types)} new: "
          + ", ".join(S.type_name(k) for k in sorted(new_types, key=lambda k: (S.n_ovals(k), k))), flush=True)
    if new_types and not args.no_certify:
        reps, _ = S.representatives(args, run, new_types)
        summary["representatives"] = {t: len(r) for t, r in reps.items()}
        Path("data/sextic_types/births.json").write_text(json.dumps(
            {"description": "Certified integer plane sextics of the types first found by scripts/sextic_births.py "
             "(format as plain.json).", "representatives": reps}, indent=1))
    (run / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
