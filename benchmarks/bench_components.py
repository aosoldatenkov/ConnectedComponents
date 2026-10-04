#!/usr/bin/env python3
"""Benchmark the component labelling: single-grid calls vs. the threaded batch.

    uv run python benchmarks/bench_components.py --width 300 --samples 10000
"""

import argparse
import time

import numpy as np

from conncomp import _connected
from conncomp.grid import neighbour_pattern


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=300)
    p.add_argument("--samples", type=int, default=10000)
    p.add_argument("--single", action="store_true", help="also time the single-grid loop")
    args = p.parse_args()

    pat = neighbour_pattern(args.width)
    rng = np.random.default_rng(seed=42)
    vals = rng.random((args.samples, args.width, args.width), dtype=np.float64) * 2 - 1

    if args.single:
        start = time.perf_counter()
        results = [_connected.components(vals[i], pat) for i in range(args.samples)]
        print(f"single: {time.perf_counter() - start:.3f}s, last: {len(results[-1])} components")

    start = time.perf_counter()
    results = _connected.components_batch(vals, pat)
    print(f"batch:  {time.perf_counter() - start:.3f}s, last: {len(results[-1])} components")


if __name__ == "__main__":
    main()
