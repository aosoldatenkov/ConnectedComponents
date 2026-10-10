#!/usr/bin/env python3
"""Region trees on the sphere mesh: Triton (conncomp.gpu_trees) versus the C++ baseline (region_trees_mesh).

Checks exact agreement (oval counts, types, consistency flags) and measures throughput: labelling alone, and full
trees with the CUDA tree kernel, the CUDA labels + PyTorch post-processing, Triton, and C++.

    uv run python benchmarks/bench_trees.py
"""

import time

import numpy as np
import torch

from conncomp import _connected
from conncomp.gpu_trees import FaceMesh, label_components, region_trees, to_nesting_trees
from conncomp.hessian import hessian
from conncomp.nesting import SphereMesh
from conncomp.polynomials import sample


def sync():
    torch.cuda.synchronize()


def main():
    torch.manual_seed(0)
    for N, B in ((30, 20000), (50, 10000), (100, 2000)):
        mesh = SphereMesh(N)
        fmesh = FaceMesh(N)
        tables = mesh.tables()
        for name, deg, coefs in (("quintic Hessians", 6, hessian(5, sample(5, B))), ("plain sextics", 6, sample(6, B))):
            s = mesh.signs(coefs, deg)
            region_trees(s[:100], tables)  # warm-up / compile
            sync()
            t0 = time.perf_counter()
            label_components(s, tables[0], tables[1])
            sync()
            t1 = time.perf_counter()
            g = region_trees(s, tables)
            sync()
            t2 = time.perf_counter()
            sc = s.cpu().numpy()
            t3 = time.perf_counter()
            res = _connected.region_trees_mesh(sc, mesh.nbr4, mesh.nbr8, mesh.antipode, 1, 3, -1)
            t4 = time.perf_counter()
            fs = fmesh.signs(coefs, deg)
            fmesh.trees(coefs[:, :100], deg)
            fmesh.trees(coefs[:, :100], deg, backend="torch")
            sync()
            t5 = time.perf_counter()
            for s0 in range(0, B, int(4e7 // fmesh.n_loc)):
                fmesh.labels(fs[s0 : s0 + int(4e7 // fmesh.n_loc)])
            sync()
            t6 = time.perf_counter()
            gc = fmesh.trees(coefs, deg)
            sync()
            t7 = time.perf_counter()
            fmesh.trees(coefs, deg, backend="torch")
            sync()
            t8 = time.perf_counter()
            same_c = np.mean(gc.n_ovals.cpu().numpy() == (np.diff(res[0]) - 1))
            off, root, nroots, tree = res[0], res[5], res[6], res[7]
            n_cpp = np.diff(off) - 1
            ok_cpp = (nroots == 1) & (tree == 1) & (root >= 0)
            same_n = np.mean(g.n_ovals.cpu().numpy() == n_cpp)
            same_ok = np.mean(g.ok.cpu().numpy() == ok_cpp)
            k = min(B, 2000)
            tg = to_nesting_trees(region_trees(s[:k], tables), k)
            tc = mesh.trees(coefs[:, :k], deg)
            same_t = np.mean([a.type == b.type for a, b in zip(tg, tc)])
            print(f"N={N:3d} {name:16s} B={B}: labels: CUDA {B / (t6 - t5):9,.0f}, Triton {B / (t1 - t0):7,.0f} | "
                  f"full trees: CUDA {B / (t7 - t6):8,.0f}, CUDA+torch {B / (t8 - t7):7,.0f}, "
                  f"Triton {B / (t2 - t1):7,.0f}, C++ {B / (t4 - t3):7,.0f} "
                  f"forms/s | agree with C++: Triton counts {same_n:.2%} ok {same_ok:.2%} types {same_t:.2%}, "
                  f"CUDA counts {same_c:.2%}", flush=True)


if __name__ == "__main__":
    main()
