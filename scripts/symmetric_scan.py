#!/usr/bin/env python3
"""Adaptive scans of Hessian curves of semi-invariant forms, one run per symmetry class.

    uv run python scripts/symmetric_scan.py --deg 5 --specs x x:-1 xy D3 ... --niter 30

For each symmetry: the dimension of the search space, the oval counts of the random samples, and the
best forms, re-counted at widths 200/400/800. The best forms are saved to
data/symmetric/cache_H_deg<d>_sym-<spec>.txt (input for conncomp-rationalize --symmetry <spec>).
"""

import argparse
import time
from pathlib import Path

import numpy as np

from conncomp.io import save_coefs
from conncomp.rationalize import float_oval_counts
from conncomp.scan import adaptive_scan
from conncomp.symmetry import symmetry

DEFAULT_SPECS = ["x", "x:-1", "xy", "xy:-1,1", "xy:-1,-1", "diag", "diag:-1", "C2", "C3", "C4", "C4:-1",
                 "D3", "D3:1,-1", "D4", "D4:1,-1", "D4:-1,1", "D4:-1,-1", "D5", "D5:1,-1"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--deg", type=int, default=5)
    p.add_argument("--specs", nargs="+", default=DEFAULT_SPECS)
    p.add_argument("--width", type=int, default=150)
    p.add_argument("--batch", type=int, default=10000)
    p.add_argument("--niter", type=int, default=30)
    p.add_argument("--out-dir", type=Path, default=Path("data/symmetric"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for spec in args.specs:
        try:
            sym = symmetry(spec, args.deg)
        except ValueError as e:
            print(f"{spec}: {e}")
            continue
        if sym.warnings():
            print(f"{spec:9s} dim {sym.dim:2d} | skipped: {'; '.join(sym.warnings())}", flush=True)
            continue
        t0 = time.time()
        best, hist = adaptive_scan(args.deg, args.width, args.batch, args.niter, basis=sym.basis, seed=0)
        coefs = np.array([c for _, c in best]).T
        per_w = [float_oval_counts(coefs, args.deg, w) for w in (200, 400, 800)]
        stable = [a for a, b, c in zip(*per_w) if a == b == c]
        top = max(stable) if stable else None
        out = args.out_dir / f"cache_H_deg{args.deg}_sym-{spec.replace(':', '_').replace(',', '_')}.txt"
        score = [c if a == b == c else -1 for a, b, c in zip(*per_w)]
        order = sorted(range(len(best)), key=lambda j: -score[j])
        save_coefs(out, [best[j][1] for j in order])
        common = ", ".join(f"{k}:{v}" for k, v in sorted(hist.items()) if k >= 0)
        print(f"{spec:9s} dim {sym.dim:2d} | best stable count {top} | histogram {common} | {time.time() - t0:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
