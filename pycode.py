#!/usr/bin/python3

from Hessian import connected
import numpy as np
import time
from numba import jit
import concurrent.futures

def form_groups_np(width, vals, pat):
    labels = [[-1] * width for _ in range(width)]
    groups = {0: (0, 0)}
    n = -1
    for i in range(width):
        for j in range(width):
            v = vals[i][j]
            ll = [-1] * 12
            for k in range(12):
                d = (pat[k, 0, i, j], pat[k, 1, i, j])
                if v * vals[d[0]][d[1]] >= 0:
                    ll[k] = labels[d[0]][d[1]]
            sorted(ll)
            M = max(ll)
            if M == -1:
                n += 1
                labels[i][j] = n
                groups[n] = (n, 1)
            else:
                labels[i][j] = M
                groups[M] = (groups[M][0], groups[M][1] + 1)
                c = 1
                for k in range(1, 12):
                    if ll[k - 1] != ll[k]:
                        c += 1
                if c < 3:
                    continue
                mm = [groups[l][0] for l in ll if l != -1]
                m = min(mm)
                for k in groups:
                    if groups[k][0] in mm:
                        groups[k] = (m, groups[k][1])
    return groups

def nbr_pattern_np(width, pat):
    dirs = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    mid = (width - 1) / 2
    for i in range(width):
        for j in range(width):
            for k in range(12):
                pat[k, 0, i, j] = i
                pat[k, 1, i, j] = j
            for k, d in enumerate(dirs):
                if 0 <= i + d[0] < width and 0 <= j + d[1] < width:
                    pat[k, 0, i, j] = i + d[0]
                    pat[k, 1, i, j] = j + d[1]
            rr = (i / mid - 1)**2 + (j / mid - 1)**2
            if rr == 0:
                continue
            k0 = int(np.floor(mid * (1 + (1 - i / mid) / rr)))
            k1 = int(np.ceil(mid * (1 + (1 - i / mid) / rr)))
            l0 = int(np.floor(mid * (1 + (1 - j / mid) / rr)))
            l1 = int(np.ceil(mid * (1 + (1 - j / mid) / rr)))
            nbr = [(k0, l0), (k1, l0), (k0, l1), (k1, l1)]
            for k, d in enumerate(nbr):
                if 0 <= d[0] < width and 0 <= d[1] < width:
                    pat[k + 8, 0, i, j] = d[0]
                    pat[k + 8, 1, i, j] = d[1]
    return pat

def test_py(width, nsamples):
    pat = np.ndarray((12, 2, width, width), dtype=np.int32)
    nbr_pattern_np(width, pat)
    rng = np.random.default_rng(seed=42)
    vals = rng.random((nsamples, width, width), dtype=np.float64) * 2 - 1
    g = form_groups_np(width, vals[0], pat)
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        g = executor.map(form_groups_np, [width] * nsamples,
                         [vals[i] for i in range(nsamples)],
                         [pat] * nsamples)
    #for i in range(nsamples):
    #    g = form_groups_np(width, vals[i], pat)
    end = time.perf_counter()
    print("Elapsed = {}s".format((end - start)))

def test_cpp(width, nsamples):
    pat = np.ndarray((12, 2, width, width), dtype=np.int32)
    nbr_pattern_np(width, pat)
    rng = np.random.default_rng(seed=42)
    vals = rng.random((nsamples, width, width), dtype=np.float64) * 2 - 1
    results = []
    start = time.perf_counter()
    for i in range(nsamples):
        results.append(connected.components(vals[i], pat))
        #print(g)
    end = time.perf_counter()
    print(results[-1])
    print("Elapsed = {}s".format((end - start)))
    
def test_cpp_p(width, nsamples):
    pat = np.ndarray((12, 2, width, width), dtype=np.int32)
    nbr_pattern_np(width, pat)
    rng = np.random.default_rng(seed=42)
    vals = rng.random((nsamples, width, width), dtype=np.float64) * 2 - 1
    start = time.perf_counter()
    results = connected.components_batch(vals, pat)
    end = time.perf_counter()
    print(results[-1])
    print("Elapsed = {}s".format((end - start)))


#test_cpp(200, 1000)
test_cpp_p(300, 10000)