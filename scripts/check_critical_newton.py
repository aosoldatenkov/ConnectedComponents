#!/usr/bin/env python3
"""Evaluate the Newton refinement of critical points (conncomp.critical): accuracy against exact sympy critical
points, convergence versus iterations, fate of spurious seeds, Morse check and timing on a large random batch.

    uv run python scripts/check_critical_newton.py --deg 6 --forms 40 --batch 100000
"""

import argparse
import collections
import importlib.util
import random
import time
from pathlib import Path

import mpmath
import numpy as np
import torch

from conncomp import DEVICE, DTYPE
from conncomp.critical import MAX, MIN, SADDLE, FaceCharts, critical_points, deduplicate, morse_check, refine
from conncomp.polynomials import monomials, sample

spec = importlib.util.spec_from_file_location(
    "check_critical_seeds", Path(__file__).resolve().parent / "check_critical_seeds.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
NAMES = {MIN: "min", SADDLE: "saddle", MAX: "max"}


def exact_values(coefs, n, points):
    """Exact values H(p) at the unit points (60 digits)."""
    mpmath.mp.dps = 60
    out = []
    for p in points:
        x, y, z = (mpmath.mpf(float(t)) for t in p)
        r = mpmath.sqrt(x * x + y * y + z * z)
        terms = (c * (x / r) ** i * (y / r) ** j * (z / r) ** k for c, (i, j, k) in zip(coefs, monomials(n)))
        out.append(float(sum(terms)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deg", type=int, default=6)
    ap.add_argument("--forms", type=int, default=40)
    ap.add_argument("--batch", type=int, default=100000)
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    n, rng = args.deg, random.Random(args.seed)
    forms = [[rng.randint(-100, 100) for _ in monomials(n)] for _ in range(args.forms)]
    truth = [S.exact_critical_points(c, n) for c in forms]
    coefs = torch.tensor(np.array(forms, dtype=float).T, dtype=DTYPE, device=DEVICE)
    charts = FaceCharts(n, N=args.N)
    seeds = charts.seeds(coefs)
    n_true = sum(len(t) for t in truth)
    print(f"{args.forms} random integer degree-{n} forms: {n_true} exact real critical points, {len(seeds)} seeds")

    # --- convergence versus iterations
    print("\nconvergence versus Newton iterations (all seeds):")
    for iters in (1, 2, 3, 4, 6, 8, 12):
        crit = refine(seeds, coefs, n, iters=iters, max_step=2 * charts.spacing)
        q = np.quantile(crit.grad_norm.cpu().numpy(), [0.5, 0.9, 0.99])
        print(f"  {iters:2d} iterations: converged {crit.converged.float().mean().item():6.1%} of seeds; "
              f"|grad h| median {q[0]:.1e}, 90% {q[1]:.1e}, 99% {q[2]:.1e}")

    # --- accuracy against the exact critical points
    crit = refine(seeds, coefs, n, iters=8, max_step=2 * charts.spacing)
    uniq = deduplicate(crit, args.forms)
    errs, verr, wrong, missing, extra = [], [], 0, 0, 0
    fate = collections.Counter()
    for b, tr in enumerate(truth):
        sel = (uniq.form == b).nonzero().squeeze(1)
        P = uniq.point[sel].cpu().numpy()
        K = uniq.kind[sel].cpu().numpy()
        Vg = uniq.value[sel].cpu().numpy()
        used = np.zeros(len(P), dtype=bool)
        norm = np.linalg.norm(np.array(forms[b], dtype=float))
        exact_v = exact_values(forms[b], n, [p for p, _ in tr])
        for (p, kind), ve in zip(tr, exact_v):
            d = S.angular_distance(P, p) if len(P) else np.array([np.inf])
            j = int(np.argmin(d))
            if d[j] < 1e-6:
                used[j] = True
                errs.append(d[j])
                verr.append(abs(Vg[j] - ve / norm))
                wrong += int(K[j] != kind)
            else:
                missing += 1
        extra += int((~used).sum())
    # fate of the seeds that were not near any critical point
    tol = 2.5 * charts.spacing
    for k in range(len(seeds)):
        b = int(seeds.form[k])
        d0 = min(S.angular_distance(np.array([q]), seeds.point[k].cpu().numpy())[0] for q, _ in truth[b])
        if d0 < tol:
            continue
        if not bool(crit.converged[k]):
            fate["did not converge (dropped)"] += 1
        else:
            d1 = min(S.angular_distance(np.array([q]), crit.point[k].cpu().numpy())[0] for q, _ in truth[b])
            fate["converged to a true critical point (merged)" if d1 < 1e-6 else "converged elsewhere (extra)"] += 1
    morse = morse_check(uniq, args.forms).cpu().numpy()
    print(f"\naccuracy (8 iterations): found {len(errs)}/{n_true}, missing {missing}, extra points {extra}, "
          f"wrong type {wrong}")
    print(f"  angular error: max {max(errs):.1e}, median {np.median(errs):.1e}; "
          f"value error (normalized form): max {max(verr):.1e}")
    print(f"  spurious seeds: {dict(fate)}")
    print(f"  Morse check (= 1): {collections.Counter(morse.tolist())}")

    # --- large random batch: Morse check and timing
    print(f"\nrandom batch of {args.batch:,} forms (N = {args.N}):")
    c = sample(n, args.batch)
    critical_points(c[:, :1000], n, charts=charts)  # warm-up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    s = charts.seeds(c)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    r = refine(s, c, n, iters=8, max_step=2 * charts.spacing)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    u = deduplicate(r, args.batch)
    m = morse_check(u, args.batch)
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    counts = torch.bincount(u.form, minlength=args.batch)
    ok = (m == 1).float().mean().item()
    torch.cuda.synchronize()
    t4 = time.perf_counter()
    _, m2 = critical_points(c, n, charts=charts, max_N=320)
    torch.cuda.synchronize()
    t5 = time.perf_counter()
    ok2 = (m2 == 1).float().mean().item()
    print(f"  seeds {t1 - t0:.2f}s ({len(s) / args.batch:.1f}/form), Newton {t2 - t1:.2f}s, "
          f"dedupe+Morse {t3 - t2:.2f}s -> {args.batch / (t3 - t0):,.0f} forms/s")
    print(f"  converged {r.converged.float().mean().item():.1%} of seeds; critical points per form: "
          f"mean {counts.float().mean().item():.2f}, max {int(counts.max())} (bound {n * n - n + 1})")
    print(f"  Morse check passed for {ok:.3%} of forms; failures: "
          f"{collections.Counter(m[m != 1].tolist()).most_common(5)}")
    print(f"  with adaptive re-seeding of the failures (N doubled up to 320): passed {ok2:.4%}; "
          f"total {t5 - t4:.2f}s -> {args.batch / (t5 - t4):,.0f} forms/s")


if __name__ == "__main__":
    main()
