#!/usr/bin/env python3
"""Compare GPU seeds of critical points (conncomp.critical) with exact critical points from sympy.

For random integer forms H of degree n, the exact critical points on RP^2 (chart z = 1) solve
grad H(p) parallel to p: g1 = H_y - v H_z = 0, g2 = u H_z - H_x = 0. The resultant in v is solved
exactly (real roots), v is recovered at 60 digits, and the type comes from the chart Hessian of
h(u, v) = H(u, v, 1) (1 + u^2 + v^2)^(-n/2).

    uv run python scripts/check_critical_seeds.py --deg 6 --forms 20 --N 40 80
"""

import argparse
import collections
import random
import time

import mpmath
import numpy as np
import sympy as sp
import torch

from conncomp import DEVICE, DTYPE
from conncomp.critical import MAX, MIN, SADDLE, FaceCharts
from conncomp.polynomials import monomials

u, v = sp.symbols("u v")
NAMES = {MIN: "min", SADDLE: "saddle", MAX: "max"}


def exact_critical_points(coefs, n, dps=60):
    """Real critical points (unit vectors, type) of the form with integer coefficients, in the chart z = 1."""
    X, Y, Z = sp.symbols("X Y Z")
    H = sum(c * X**i * Y**j * Z**k for c, (i, j, k) in zip(coefs, monomials(n)))
    Hx, Hy, Hz = (sp.diff(H, w).subs({X: u, Y: v, Z: 1}) for w in (X, Y, Z))
    g1, g2 = sp.expand(Hy - v * Hz), sp.expand(u * Hz - Hx)
    R = sp.Poly(sp.resultant(g1, g2, v), u)
    h = H.subs({X: u, Y: v, Z: 1}) * (1 + u**2 + v**2) ** sp.Rational(-n, 2)
    hess = sp.lambdify((u, v), [sp.diff(h, u, 2), sp.diff(h, u, v), sp.diff(h, v, 2)], "mpmath")
    g2f = sp.lambdify((u, v), g2, "mpmath")
    mpmath.mp.dps = dps
    out = []
    for r in R.sqf_part().real_roots():
        u0 = mpmath.mpf(str(r.evalf(dps)))
        cv = sp.Poly(g1.subs(u, sp.Float(str(u0), dps)), v).all_coeffs()
        for v0 in mpmath.polyroots([mpmath.mpf(str(c)) for c in cv], maxsteps=200, extraprec=200):
            if abs(mpmath.im(v0)) > mpmath.mpf(10) ** (-dps // 3):
                continue
            v0 = mpmath.re(v0)
            scale = 1 + sum(abs(c) for c in coefs) * (1 + abs(u0) + abs(v0)) ** n
            if abs(g2f(u0, v0)) > scale * mpmath.mpf(10) ** (-dps // 2):
                continue
            a, b, c = hess(u0, v0)
            det, tr = a * c - b * b, a + c
            kind = SADDLE if det < 0 else (MAX if tr < 0 else MIN)
            p = np.array([float(u0), float(v0), 1.0])
            out.append((p / np.linalg.norm(p), kind))
    # deduplicate (multiple v for the same u are distinct points; exact duplicates are not expected)
    return out


def angular_distance(p, q):
    """Distance in RP^2 between unit vectors (rows of p) and a unit vector q."""
    d = np.abs(p @ q).clip(max=1.0)
    return np.arccos(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deg", type=int, default=6)
    ap.add_argument("--forms", type=int, default=20)
    ap.add_argument("--N", type=int, nargs="+", default=[40, 80])
    ap.add_argument("--height", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inspect", action="store_true", help="list the seeds that are not near any critical point")
    args = ap.parse_args()
    if args.inspect:
        return inspect_spurious(args.N[0], args.deg, args.forms, args.seed, args.height)
    rng = random.Random(args.seed)
    n = args.deg
    forms = [[rng.randint(-args.height, args.height) for _ in monomials(n)] for _ in range(args.forms)]
    t0 = time.time()
    truth = [exact_critical_points(c, n) for c in forms]
    print(f"exact critical points of {args.forms} random degree-{n} forms in {time.time() - t0:.0f}s "
          f"(bound n^2 - n + 1 = {n * n - n + 1}); real: {[len(t) for t in truth]}")
    morse = [sum(1 if k != SADDLE else -1 for _, k in t) for t in truth]
    print(f"Morse check (#max - #saddle + #min = 1 on RP^2): {collections.Counter(morse)}")
    coefs = torch.tensor(np.array(forms, dtype=float).T, dtype=DTYPE, device=DEVICE)
    for N in args.N:
        charts = FaceCharts(n, N=N)
        t0 = time.time()
        seeds = charts.seeds(coefs)
        torch.cuda.synchronize()
        dt = time.time() - t0
        tol = 2.5 * charts.spacing
        found, missed, wrong_type, spurious = collections.Counter(), collections.Counter(), 0, 0
        for b, crit in enumerate(truth):
            sel = (seeds.form == b).nonzero().squeeze(1)
            P = seeds.point[sel].cpu().numpy()
            K = seeds.kind[sel].cpu().numpy()
            used = np.zeros(len(P), dtype=bool)
            for p, kind in crit:
                d = angular_distance(P, p) if len(P) else np.array([])
                near = d < tol
                if near.any():
                    used |= near
                    if (K[near] == kind).any():
                        found[kind] += 1
                    else:
                        wrong_type += 1
                else:
                    missed[kind] += 1
            spurious += int((~used).sum())
        total = sum(found.values()) + sum(missed.values()) + wrong_type
        print(f"N={N} (spacing {charts.spacing:.4f}, tolerance {tol:.4f} rad): {len(seeds)} seeds "
              f"in {dt * 1000:.0f} ms | matched {sum(found.values())}/{total} "
              f"({', '.join(f'{NAMES[k]} {found[k]}/{found[k] + missed[k]}' for k in (MIN, SADDLE, MAX))}), "
              f"wrong type {wrong_type}, missed {dict((NAMES[k], m) for k, m in missed.items())}, "
              f"seeds not near any critical point {spurious}")



def inspect_spurious(N=40, n=6, forms=20, seed=0, height=100):
    """Print the seeds that are not near any exact critical point."""
    rng = random.Random(seed)
    F = [[rng.randint(-height, height) for _ in monomials(n)] for _ in range(forms)]
    truth = [exact_critical_points(c, n) for c in F]
    charts = FaceCharts(n, N=N)
    seeds = charts.seeds(torch.tensor(np.array(F, dtype=float).T, dtype=DTYPE, device=DEVICE))
    tol = 2.5 * charts.spacing
    for k in range(len(seeds)):
        b = int(seeds.form[k])
        p = seeds.point[k].cpu().numpy()
        d = min(angular_distance(np.array([q]), p)[0] for q, _ in truth[b]) if truth[b] else np.inf
        if d >= tol:
            uv = f"({float(seeds.u[k]):+.3f}, {float(seeds.v[k]):+.3f})"
            print(f"form {b} face {int(seeds.face[k])} (u, v) = {uv} {NAMES[int(seeds.kind[k])]}, "
                  f"value {float(seeds.value[k]):+.2e}, nearest critical point {d:.3f} rad")


if __name__ == "__main__":
    main()
