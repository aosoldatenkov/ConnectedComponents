#!/usr/bin/env python3
"""Fused per-form kernel (FaceMesh.trees_fused: row runs in shared memory) versus the multi-kernel CUDA path
(FaceMesh.trees: float32 signs, tile labelling, tree kernel): exact agreement and throughput (forms/s, from the
coefficients to the region trees).

    uv run python benchmarks/bench_fused.py
"""

import time

import torch

from conncomp.gpu_trees import FaceMesh, GPUTrees
from conncomp.hessian import hessian
from conncomp.polynomials import sample


def timed(f, reps=3):
    f()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


def main():
    torch.manual_seed(0)
    for N, B in ((30, 50000), (50, 20000), (100, 4000), (200, 1000)):
        fm = FaceMesh(N)
        for name, c in (("plane sextics", sample(6, B)), ("quintic Hessians", hessian(5, sample(5, B)))):
            chunk = max(1, int(4e7 // fm.n_loc))
            a, b = fm.trees_fused(c[:, :chunk], 6), fm.trees(c[:, :chunk], 6)
            same = all(torch.equal(getattr(a, f), getattr(b, f)) for f in GPUTrees.__dataclass_fields__)
            t_old = timed(lambda: fm.trees(c, 6))
            t_new = timed(lambda: fm.trees_fused(c, 6))
            print(f"N={N:3d} {name:16s}: multi-kernel {B / t_old:9,.0f} forms/s, fused {B / t_new:9,.0f} forms/s "
                  f"({t_old / t_new:.1f}x); identical trees: {same}", flush=True)


if __name__ == "__main__":
    main()
