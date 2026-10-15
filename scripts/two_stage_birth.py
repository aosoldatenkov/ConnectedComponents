#!/usr/bin/env python3
"""Two-stage births for Hessian curves H(f) of sextics f (degree-8 curves): try to add an oval to saturated forms.

A saturated form (every maximum of H positive, every minimum negative, cf. diagnose_births.py) has no extremum whose
value can be pushed through zero. A new oval needs a new extremum first. This script does both stages in one
constrained step: pick a point q inside a sign region of H, at angular distance rho from the curve, and ask for
    H(q) -> -sign(H(q)) delta              (weight w_q)
    H(ring_i) -> ring_goal * H(ring_i)      (RING points at angular radius frac * rho around q; weight w_r)
    H(p_j) -> H(p_j)                        (all critical points p_j of H; weight w_c)
for the normalized curve, by damped weighted Gauss-Newton steps on f restricted to a family (x-symmetric, xy, or full),
capped in length. Ring and critical values only need to keep their signs (with a margin): in each linearized step an
active-set iteration constrains them only where the predicted sign margin fails (others get a small weight). If H(q) is
flipped while the ring keeps its sign (a "dimple"), a new small oval lies inside the ring (inside an oval: a nested
one); in the x-family the mirror image of q gets one as well.

Dimple children are counted at N = 200 (new ovals are too small for a coarse grid); counts above the seeds' maximum
are confirmed by the exact shift test at N = 400 (hessian6_scan.shift) and appended to --out.

    uv run python scripts/two_stage_birth.py pools/hessian6_test/xy_genuine16.txt --family x --minutes 20
"""

import argparse
import collections
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import hessian6_scan as H6  # noqa: E402
import sextic_births as SB  # noqa: E402
import sextic_types as S  # noqa: E402
from diagnose_births import projector  # noqa: E402
from conncomp import DEVICE, DTYPE  # noqa: E402
from conncomp.critical import critical_points, monomial_values  # noqa: E402
from conncomp.gpu_trees import FaceMesh  # noqa: E402
from conncomp.io import load_coefs  # noqa: E402

RING = 8


def ring_points(q, rad):
    e1, e2 = SB.tangent_frame(q)
    th = torch.arange(RING, dtype=DTYPE, device=DEVICE) * (2 * math.pi / RING)
    P = q[:, None] + torch.tan(rad)[:, None, None] * (torch.cos(th)[None, :, None] * e1[:, None]
                                                     + torch.sin(th)[None, :, None] * e2[:, None])
    return P / P.norm(dim=2, keepdim=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool", type=Path)
    ap.add_argument("--n", type=int, default=300, help="first n forms of the pool (most robust first)")
    ap.add_argument("--family", choices=["x", "xy", "full"], default="x")
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--pairs", type=int, default=2048, help="(form, q, parameters) per step")
    ap.add_argument("--wq", type=float, default=100.0)
    ap.add_argument("--wr", type=float, default=10.0)
    ap.add_argument("--wc", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--inner", type=int, default=6, help="active-set iterations per linearized step")
    ap.add_argument("--margin", type=float, default=0.3, help="ring / critical values keep sign with this margin")
    ap.add_argument("--w-rest", type=float, default=0.01, help="weight of satisfied constraints (relative)")
    ap.add_argument("--cap", type=float, default=0.01, help="max relative change per Gauss-Newton step")
    ap.add_argument("--exact-dimples", action="store_true", help="shift test for every dimple child (slower)")
    ap.add_argument("--out", type=Path, default=Path("pools/hessian6_test/two_stage.txt"))
    args = ap.parse_args()
    S.set_mode("hessian6")
    SB.DEG = S.DEG
    rng = np.random.default_rng()
    F = np.asarray(load_coefs(args.pool), dtype=np.float64)[: args.n]
    f = torch.as_tensor(F.T, device=DEVICE)
    Pj = projector(args.family, f.shape[0])
    Q = torch.linalg.qr(Pj)[0][:, : int(round(float(torch.trace(Pj))))]
    f = Pj @ f  # (drops terms of degree <= 1, which do not affect H)
    f = f / f.norm(dim=0, keepdim=True)
    n = f.shape[1]
    m40, m200, m400 = FaceMesh(40), FaceMesh(200), FaceMesh(400)
    C = H6.curves(f)
    base = H6.ovals(m40, C).cpu().numpy()
    base200 = H6.ovals(m200, C).cpu().numpy()
    top = int(base.max())
    crit, morse = critical_points(C, 8)
    counts = torch.bincount(crit.form, minlength=n)
    kmax = int(counts.max())
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(len(crit), device=DEVICE) - start[crit.form]
    CP = torch.zeros((n, kmax, 3), dtype=DTYPE, device=DEVICE)
    CV = torch.zeros((n, kmax), dtype=DTYPE, device=DEVICE)
    CM = torch.zeros((n, kmax), dtype=DTYPE, device=DEVICE)  # 1 for real entries
    CP[crit.form, pos], CV[crit.form, pos], CM[crit.form, pos] = crit.point, crit.value, 1.0
    print(f"{n} seeds ({args.family} family, {Q.shape[1]} parameters); oval counts at N=40: "
          f"{dict(collections.Counter(base.tolist()))}; {kmax} critical points per form at most", flush=True)
    t0, step, tried = time.time(), 0, 0
    found = collections.Counter()
    stats, stats_dimple, dimple_200 = collections.Counter(), collections.Counter(), collections.Counter()
    dimple_exact = collections.Counter()
    while time.time() - t0 < args.minutes * 60:
        B = args.pairs
        form = torch.as_tensor(rng.integers(0, n, B), device=DEVICE)
        # candidate points: random unit vectors, kept if well inside a region (|H| above the median of the form)
        q = torch.randn((B, 3), dtype=DTYPE, device=DEVICE)
        q = q / q.norm(dim=1, keepdim=True)
        if args.family != "full" and rng.random() < 0.3:  # on the mirror axis x = 0: a single new oval
            q[:, 0] = 0.0
            q = q / q.norm(dim=1, keepdim=True)
        rho = SB.distance_to_curve(C, form, q)
        keep = rho > 0.02
        form, q, rho = form[keep], q[keep], rho[keep]
        P_ = len(form)
        if P_ == 0:
            continue
        frac = torch.as_tensor(rng.uniform(0.3, 0.9, P_), dtype=DTYPE, device=DEVICE)
        delta = torch.exp(torch.empty(P_, dtype=DTYPE, device=DEVICE).uniform_(math.log(1e-5), math.log(1e-3)))
        ring_goal = torch.as_tensor(rng.choice([1.0, 1.5, 2.0], P_), dtype=DTYPE, device=DEVICE)
        R = ring_points(q, frac * rho)
        pts = torch.cat([q[:, None], R, CP[form]], dim=1)  # (P, 1 + RING + kmax, 3)
        Mx = monomial_values(pts.reshape(-1, 3), 8).reshape(P_, pts.shape[1], -1)
        x = f.T[form].clone()
        _, v0 = SB.curve_rows(Mx, x)
        goal = torch.cat([(-torch.sign(v0[:, 0]) * delta)[:, None], ring_goal[:, None] * v0[:, 1 : 1 + RING],
                          CV[form]], dim=1)
        w = torch.cat([torch.full((P_, 1), args.wq, dtype=DTYPE, device=DEVICE),
                       torch.full((P_, RING), args.wr, dtype=DTYPE, device=DEVICE), args.wc * CM[form]], dim=1)
        k = Q.shape[1]
        eye = torch.eye(k, dtype=DTYPE, device=DEVICE)
        sgn = torch.sign(v0)  # signs to keep (ring and critical points); the target q is flipped
        margin = args.margin * v0.abs()
        for _ in range(args.steps):
            J, fx = SB.curve_rows(Mx, x)
            JQ = J @ Q  # (P, K, k)
            da = torch.zeros((P_, k), dtype=DTYPE, device=DEVICE)
            for _ in range(args.inner):  # active set: constrain ring / critical values only where the sign margin fails
                pred = fx + (JQ @ da[..., None])[..., 0]
                viol = (sgn * pred < margin) & (w[:, :] > 0)
                ww = torch.where(viol, w, args.w_rest * w)
                ww[:, 0] = args.wq
                tgt = torch.where(viol, sgn * margin, fx)
                tgt[:, 0] = goal[:, 0]
                Jw = JQ * ww[..., None]
                A = Jw.transpose(1, 2) @ Jw
                A = A + 1e-8 * A.diagonal(dim1=1, dim2=2).amax(1)[:, None, None] * eye
                da = torch.linalg.solve(A, Jw.transpose(1, 2) @ ((tgt - fx) * ww)[..., None])[..., 0]
            dx = da @ Q.T
            dx = dx * torch.clamp(args.cap / dx.norm(dim=1, keepdim=True).clamp(min=1e-300), max=1.0)
            x = x + dx
            x = x / x.norm(dim=1, keepdim=True)
        _, fx = SB.curve_rows(Mx, x)
        flipped = torch.sign(fx[:, 0]) != torch.sign(v0[:, 0])
        ring_ok = (torch.sign(fx[:, 1 : 1 + RING]) == torch.sign(v0[:, 1 : 1 + RING])).all(dim=1)
        dimple = (flipped & ring_ok).cpu().numpy()  # a new oval inside the ring (unless the curve slips between)
        nc = H6.ovals(m40, H6.curves(x.T)).cpu().numpy()
        d = nc - base[form.cpu().numpy()]
        tried += P_
        stats.update(d.tolist())
        stats_dimple.update(d[dimple].tolist())
        hi = np.nonzero(dimple)[0]  # checked at high resolution regardless of the N = 40 count (new ovals are tiny)
        if len(hi):
            n200 = H6.ovals(m200, H6.curves(x[torch.as_tensor(hi, device=DEVICE)].T)).cpu().numpy()
            dimple_200.update((n200 - base200[form.cpu().numpy()][hi]).tolist())
            if args.exact_dimples:  # exact topology (shift test) of every dimple child, not only of high counts
                ex = [int(H6.ovals(m400, H6.shift(H6.curves(x[i][:, None])[:, 0])[0][:, None])[0]) for i in hi]
                dimple_exact.update((np.array(ex) - base200[form.cpu().numpy()][hi]).tolist())
                hi = hi[np.array(ex) > top]
            else:
                hi = hi[n200 > top]
        for i in hi:
            c = H6.curves(x[i][:, None])
            n200 = int(H6.ovals(m200, c)[0])
            Hs, _, rob, _ = H6.shift(c[:, 0])
            ns = int(H6.ovals(m400, Hs[:, None])[0])
            found[(int(nc[i]), n200, ns)] += 1
            if ns > top and n200 > top:
                with open(args.out, "a") as fh:
                    fh.write(" ".join(str(v) for v in x[i].cpu().numpy()) + "\n")
                print(f"  [{(time.time() - t0) / 60:5.1f} min] {ns} ovals confirmed (N=40 {nc[i]}, N=200 {n200}; "
                      f"robustness {rob:.1e}) from seed {int(form[i])}, rho {float(rho[i]):.3f}, frac "
                      f"{float(frac[i]):.2f}, delta {float(delta[i]):.1e}", flush=True)
        step += 1
        if step % 20 == 0:
            print(f"[{(time.time() - t0) / 60:5.1f} min] {tried:,} tried; flip of H(q) achieved "
                  f"{float(flipped.double().mean()):.0%} (last batch); change of the count at N=40: "
                  f"{dict(sorted(stats.items()))}; dimples (flip + ring kept): {sum(stats_dimple.values()):,}, change "
                  f"at N=40 {dict(sorted(stats_dimple.items()))}, at N=200 {dict(sorted(dimple_200.items()))}; "
                  f"above {top}: (N40, N200, shift) {dict(found)}", flush=True)
    print(f"done: {tried:,} tried; dimples {sum(stats_dimple.values()):,}: change at N=200 "
          f"{dict(sorted(dimple_200.items()))}" + (f", exact (shift test) {dict(sorted(dimple_exact.items()))}"
                                                   if args.exact_dimples else "")
          + f"; above {top}: (N40, N200, shift) {dict(found)}")


if __name__ == "__main__":
    main()
