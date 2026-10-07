#!/usr/bin/env python3
"""Adaptive Hessian scans starting from degenerate seed forms (conncomp.seeds), one run per family.

    uv run python scripts/degenerate_scan.py --deg 5 --families lines concurrent mixed separable developable

Each run alternates perturbed seeds (eps log-uniform) with refinements of the best forms (log-uniform radius),
pre-screened on the GPU and counted exactly; candidates must reproduce their count at twice the width.
The best forms are saved to data/degenerate/cache_H_deg<d>_<family>.txt.
"""

import argparse
import collections
import time
from pathlib import Path

import numpy as np

from conncomp.euler import EulerScreen
from conncomp.io import save_coefs
from conncomp.rationalize import float_oval_counts
from conncomp.scan import adaptive_scan
from conncomp.seeds import FAMILIES, perturbed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--deg", type=int, default=5)
    p.add_argument("--families", nargs="+", default=list(FAMILIES))
    p.add_argument("--width", type=int, default=200)
    p.add_argument("--batch", type=int, default=200000)
    p.add_argument("--niter", type=int, default=60)
    p.add_argument("--cap", type=int, default=20000)
    p.add_argument("--eps", type=float, nargs=2, default=(1e-4, 1e-1))
    p.add_argument("--check-factor", type=int, default=4, help="pool members must agree at width * factor")
    p.add_argument("--out-dir", type=Path, default=Path("data/degenerate"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    screen = EulerScreen(args.deg, N=40)
    for fam in args.families:
        t0 = time.time()
        best, hist = adaptive_scan(
            args.deg, args.width, args.batch, args.niter, r=tuple(args.eps), keep=200, seed=0, screen=screen,
            slack=3, cap=args.cap, sampler=lambda n, fam=fam: perturbed(fam, args.deg, n, tuple(args.eps)),
            check_width=args.width * args.check_factor,
        )
        coefs = np.array([c for _, c in best]).T
        per_w = [float_oval_counts(coefs, args.deg, w) for w in (400, 800, 1600)]
        stable = [a if a == b == c else None for a, b, c in zip(*per_w)]
        order = sorted(range(len(best)), key=lambda j: -(stable[j] if stable[j] is not None else -1))
        save_coefs(args.out_dir / f"cache_H_deg{args.deg}_{fam}.txt", [best[j][1] for j in order])
        tail = sorted(hist.items())[-4:]
        pool = sorted(collections.Counter(s for s in stable if s is not None).items())[-3:]
        print(f"{fam:12s} | exact counts (width {args.width}) tail {tail} | pool stable at 400/800/1600: {pool} | "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
