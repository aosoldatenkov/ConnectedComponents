#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 02:01:21 2025

@author: andrey
"""

import torch
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import connected
import curses as cr
import random
import time

dtype = torch.float64
if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator().type 
else:
    device ="cpu"
torch.set_default_device(device)

def normalize(pts):
    norm = pts.norm(p=2, dim=0)
    return pts.div_(norm)

def sample(pts):
    pts = torch.randn_like(pts, dtype=dtype)
    return normalize(pts)

def tsample(dim, size):
    bsize = min(size, 10**3)
    n = 0
    pts = None
    while n < size:
        batch = torch.randn((dim, bsize), dtype=dtype)
        norm = (batch**2).sum(axis=0)
        #le = torch.heaviside(1. - norm, torch.tensor([0.], dtype=dtype))
        ge = torch.heaviside(norm - 1e-2, torch.tensor([0.], dtype=dtype))
        ind = torch.argwhere(ge).squeeze()
        if n > 0:
            pts = torch.hstack((pts, torch.index_select(batch, 1, ind)))
        else:
            pts = torch.index_select(batch, 1, ind)
        n += ind.size()[0]
    return normalize(pts[:,:size])

def evaluate(mons, coefs, pts, vals):
    dim, psize = pts.size(dim=0), tuple(pts.size()[1:])
    csize = tuple(coefs.size()[1:])
    mon = torch.zeros(psize, dtype=dtype)
    vals.fill_(0.)
    #val = torch.zeros(csize + psize, dtype=dtype) 
    for i, m in enumerate(mons):
        mon.fill_(1.)
        for j in range(dim):
            mon = mon * pts[j].pow(m[j]) if m[j] > 0 else mon
        vals += torch.tensordot(coefs[i].reshape(csize + (1,)),
                                mon.reshape((1,) + psize), dims=1)
    return vals

def deg_to_dim(deg):
    return (deg + 1) * (deg + 2) // 2

def monomials(deg):
    return [(i, j, deg-i-j) for i in range(deg+1) for j in range(deg-i+1)]

def hessian_map(deg):
    coefs = []
    x, y, z = sp.symbols('x y z')
    f = sp.Poly(0, x, y, z)
    for i in range(deg + 1):
        for j in range(deg - i + 1):
            a = sp.symbols(f'a{i}\:{j}\:{deg-i-j}')
            coefs.append(a)
            f += a * x**i * y**j * z**(deg-i-j)
    Hf = (f.diff((x, 2)) * f.diff((y, 2)) - f.diff((x, 1), (y, 1))**2).as_dict()
    return {x: sp.Poly(Hf[x], *coefs).as_dict() for x in Hf}

def hessian(deg, coefs, hcoefs):
    hmap = hessian_map(deg)
    #hdim = deg_to_dim(2 * deg - 4)
    hmons = monomials(2 * deg - 4)
    #shape = tuple(coefs.size()[1:])
    #hcoefs = torch.zeros((hdim,) + shape, dtype=dtype)
    hcoefs.fill_(0.)
    mon = torch.ones(tuple(coefs.size()[1:]), dtype=dtype)
    for m in hmap:
        for n in hmap[m]:
            mon.fill_(1.)
            for i in range(len(n)):
                if n[i] > 0:
                    mon *= coefs[i].pow(n[i])
            hcoefs[hmons.index(m)] += float(hmap[m][n]) * mon
    return hcoefs

def nbr_pattern(width, pat):
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

def batch_scan(deg, width, nsamples, niter, lo, filtr=6):
    mons = monomials(deg)
    coefs = torch.zeros((deg_to_dim(deg), nsamples), dtype=dtype)
    vals = torch.zeros((nsamples, width, width), dtype=dtype)
    xx = torch.linspace(-1, 1, width, dtype=dtype)
    xx = xx.reshape((width, 1)) * torch.ones((1, width), dtype=dtype)
    yy = torch.linspace(-1, 1, width, dtype=dtype)
    yy = torch.ones((width, 1), dtype=dtype) * yy.reshape((1, width))
    nn = xx**2 + yy**2
    pts = torch.stack([2. * xx, 2. * yy, nn - 1.])
    pts = pts / (nn + 1.)
    
    pat = np.ndarray((12, 2, width, width), dtype=np.int32)
    nbr_pattern(width, pat)
    comp_counts = defaultdict(int)
    save_coefs = defaultdict(list)
    for i in range(niter):
        coefs = sample(coefs)
        evaluate(mons, coefs, pts, vals)
        counts = connected.components_batch(vals.cpu().numpy(), pat)
        components = defaultdict(list)
        for j in range(nsamples):
            l = len([k for k in counts[j] if k >= filtr])
            components[l].append(j)
            comp_counts[l] += 1
            if l >= lo:
                save_coefs[l].append(coefs.cpu().numpy()[:,j])
        print(i, " | " + " ".join([f"{i}: {comp_counts[i]};"
                        for i in sorted(comp_counts.keys())]))
    return save_coefs

def batch_scan_H(deg, width, nsamples, niter, lo, filtr=6, perturb=0):
    hmons = monomials(2 * deg - 4)
    coefs = torch.zeros((deg_to_dim(deg), nsamples), dtype=dtype)
    hcoefs = torch.zeros((deg_to_dim(2 * deg - 4), nsamples), dtype=dtype)
    vals = torch.zeros((nsamples, width, width), dtype=dtype)
    xx = torch.linspace(-1, 1, width, dtype=dtype)
    xx = xx.reshape((width, 1)) * torch.ones((1, width), dtype=dtype)
    yy = torch.linspace(-1, 1, width, dtype=dtype)
    yy = torch.ones((width, 1), dtype=dtype) * yy.reshape((1, width))
    nn = xx**2 + yy**2
    pts = torch.stack([2. * xx, 2. * yy, nn - 1.])
    pts = pts / (nn + 1.)
    
    pat = np.ndarray((12, 2, width, width), dtype=np.int32)
    nbr_pattern(width, pat)
    comp_counts = defaultdict(int)
    save_coefs = defaultdict(list)
    for i in range(niter):
        coefs = sample(coefs)
        hessian(deg, coefs, hcoefs)
        if perturb > 0:
            pert = torch.randn_like(hcoefs, dtype=dtype) * perturb
            hcoefs.add_(pert)
            normalize(hcoefs)
        evaluate(hmons, hcoefs, pts, vals)
        counts = connected.components_batch(vals.cpu().numpy(), pat)
        components = defaultdict(list)
        for j in range(nsamples):
            l = len([k for k in counts[j] if k >= filtr])
            components[l].append(j)
            comp_counts[l] += 1
            if l >= lo:
                save_coefs[l].append(coefs.cpu().numpy()[:,j])
        print(i, " | " + " ".join([f"{i}: {comp_counts[i]};"
                        for i in sorted(comp_counts.keys())]))
    return save_coefs

def center_scan(deg, width, center, r, nsamples, niter, lo, filtr=6):
    mons = monomials(deg)
    tcenter = torch.tensor(center, dtype=dtype).reshape((deg_to_dim(deg), 1))
    tcenter = tcenter.matmul(torch.ones((1, nsamples), dtype=dtype))
    coefs = torch.zeros((deg_to_dim(deg), nsamples), dtype=dtype)
    vals = torch.zeros((nsamples, width, width), dtype=dtype)
    xx = torch.linspace(-1, 1, width, dtype=dtype)
    xx = xx.reshape((width, 1)) * torch.ones((1, width), dtype=dtype)
    yy = torch.linspace(-1, 1, width, dtype=dtype)
    yy = torch.ones((width, 1), dtype=dtype) * yy.reshape((1, width))
    nn = xx**2 + yy**2
    pts = torch.stack([2. * xx, 2. * yy, nn - 1.])
    pts = pts / (nn + 1.)
    
    pat = np.ndarray((12, 2, width, width), dtype=np.int32)
    nbr_pattern(width, pat)
    comp_counts = defaultdict(int)
    save_coefs = defaultdict(list)
    for i in range(niter):
        coefs = torch.randn_like(tcenter, dtype=dtype) * r + tcenter
        normalize(coefs)
        evaluate(mons, coefs, pts, vals)
        counts = connected.components_batch(vals.cpu().numpy(), pat)
        components = defaultdict(list)
        for j in range(nsamples):
            l = len([k for k in counts[j] if k >= filtr])
            components[l].append(j)
            comp_counts[l] += 1
            if l >= lo:
                save_coefs[l].append(coefs.cpu().numpy()[:,j])
        print(i, " | " + " ".join([f"{i}: {comp_counts[i]};"
                        for i in sorted(comp_counts.keys())]))
    return save_coefs

def center_scan_H(deg, width, center, r, nsamples, niter, lo, filtr=6):
    hmons = monomials(2 * deg - 4)
    tcenter = torch.tensor(center, dtype=dtype).reshape((deg_to_dim(deg), 1))
    tcenter = tcenter.matmul(torch.ones((1, nsamples), dtype=dtype))
    coefs = torch.zeros((deg_to_dim(deg), nsamples), dtype=dtype)
    hcoefs = torch.zeros((deg_to_dim(2 * deg - 4), nsamples), dtype=dtype)
    vals = torch.zeros((nsamples, width, width), dtype=dtype)
    xx = torch.linspace(-1, 1, width, dtype=dtype)
    xx = xx.reshape((width, 1)) * torch.ones((1, width), dtype=dtype)
    yy = torch.linspace(-1, 1, width, dtype=dtype)
    yy = torch.ones((width, 1), dtype=dtype) * yy.reshape((1, width))
    nn = xx**2 + yy**2
    pts = torch.stack([2. * xx, 2. * yy, nn - 1.])
    pts = pts / (nn + 1.)
    
    pat = np.ndarray((12, 2, width, width), dtype=np.int32)
    nbr_pattern(width, pat)
    comp_counts = defaultdict(int)
    save_coefs = defaultdict(list)
    for i in range(niter):
        coefs = torch.randn_like(tcenter, dtype=dtype) * r + tcenter
        normalize(coefs)
        hessian(deg, coefs, hcoefs)
        evaluate(hmons, hcoefs, pts, vals)
        counts = connected.components_batch(vals.cpu().numpy(), pat)
        components = defaultdict(list)
        for j in range(nsamples):
            l = len([k for k in counts[j] if k >= filtr])
            components[l].append(j)
            comp_counts[l] += 1
            if l >= lo:
                save_coefs[l].append(coefs.cpu().numpy()[:,j])
        print(i, " | " + " ".join([f"{i}: {comp_counts[i]};"
                        for i in sorted(comp_counts.keys())]))
    return save_coefs

def make_plots(deg, width, coefs):
    n = len(coefs)
    mons = monomials(deg)
    cc = torch.tensor(coefs, dtype=dtype).T
    vals = torch.zeros((n, width, width))
    xx = torch.linspace(-1, 1, width, dtype=dtype)
    xx = xx.reshape((width, 1)) * torch.ones((1, width), dtype=dtype)
    yy = torch.linspace(-1, 1, width, dtype=dtype)
    yy = torch.ones((width, 1), dtype=dtype) * yy.reshape((1, width))
    nn = xx**2 + yy**2
    pts = torch.stack([2. * xx, 2. * yy, nn - 1.])
    pts = pts / (nn + 1.)
    evaluate(mons, cc, pts, vals)
    v = vals.cpu().numpy()
    for i in range(n):
        ind = [(a, b) for a in range(width) for b in range(width)
               if v[i, a, b] < 0]
        points = [[j[0] for j in ind], [j[1] for j in ind]]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(points[0], points[1], c='tab:blue', s=1)


def make_plots_H(deg, width, coefs):
    n = len(coefs)
    hmons = monomials(2 * deg - 4)
    cc = torch.tensor(coefs, dtype=dtype).T
    hcc = torch.zeros((deg_to_dim(2 * deg - 4), n), dtype=dtype)
    hessian(deg, cc, hcc)
    vals = torch.zeros((n, width, width))
    xx = torch.linspace(-1, 1, width, dtype=dtype)
    xx = xx.reshape((width, 1)) * torch.ones((1, width), dtype=dtype)
    yy = torch.linspace(-1, 1, width, dtype=dtype)
    yy = torch.ones((width, 1), dtype=dtype) * yy.reshape((1, width))
    nn = xx**2 + yy**2
    pts = torch.stack([2. * xx, 2. * yy, nn - 1.])
    pts = pts / (nn + 1.)
    evaluate(hmons, hcc, pts, vals)
    v = vals.cpu().numpy()
    for i in range(n):
        ind = [(a, b) for a in range(width) for b in range(width)
               if v[i, a, b] < 0]
        points = [[j[0] for j in ind], [j[1] for j in ind]]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(points[0], points[1], c='tab:blue', s=1)
        
def save_cache(fname, cache):
    with open(fname, "w") as f:
        for x in reversed(cache):
            f.write(' '.join(str(y) for y in list(x)) + '\n')

def loop(stdscr, deg, width, batch_size, filtr=3):
    cr.use_default_colors()
    stdscr.nodelay(True)
    total_samples = 0
    iteration = 0
    min_comp = 8
    max_comp_found = 0
    cache_len = 1000
    cache0 = deque([], cache_len)
    cache1 = deque([], cache_len)
    
    mons = monomials(deg)
    coefs = torch.zeros((deg_to_dim(deg), batch_size), dtype=dtype)
    vals = torch.zeros((batch_size, width, width), dtype=dtype)
    xx = torch.linspace(-1, 1, width, dtype=dtype)
    xx = xx.reshape((width, 1)) * torch.ones((1, width), dtype=dtype)
    yy = torch.linspace(-1, 1, width, dtype=dtype)
    yy = torch.ones((width, 1), dtype=dtype) * yy.reshape((1, width))
    nn = xx**2 + yy**2
    pts = torch.stack([2. * xx, 2. * yy, nn - 1.])
    pts = pts / (nn + 1.)
    pat = np.ndarray((12, 2, width, width), dtype=np.int32)
    nbr_pattern(width, pat)
    comp_counts = defaultdict(int)
    
    while True:
        stdscr.erase()
        stdscr.addstr(0, 0, f"Using {device} device")
        stdscr.addstr(1, 0, f'Iteration: {iteration};\t Total samples: {total_samples}')
        stdscr.addstr(2, 0, "Total counts: "
                      + " ".join([f"{i}: {comp_counts[i]};"
                                  for i in sorted(comp_counts.keys())]))
        stdscr.addstr(3, 0, f"Cache 0: {len(cache0)};\t Cache 1: {len(cache1)}")
        stdscr.addstr(4, 0, f"Maximal number of components found: {max_comp_found}")
        stdscr.refresh()
        dice = random.randrange(3)
        r = 1e-1
        if dice == 0 and len(cache0) > 0:
            coefs = torch.randn_like(coefs, dtype=dtype) * r
            center = cache0.popleft()
            tcenter = torch.tensor(center, dtype=dtype).reshape((deg_to_dim(deg), 1))
            coefs.add_(tcenter.matmul(torch.ones((1, batch_size), dtype=dtype)))
            normalize(coefs)
        elif dice == 1 and len(cache1) > 0:
            coefs = torch.randn_like(coefs, dtype=dtype) * r
            center = cache1[random.randrange(len(cache1))]
            tcenter = torch.tensor(center, dtype=dtype).reshape((deg_to_dim(deg), 1))
            coefs.add_(tcenter.matmul(torch.ones((1, batch_size), dtype=dtype)))
            normalize(coefs)
        else:
            coefs = sample(coefs)
        evaluate(mons, coefs, pts, vals)
        counts = connected.components_batch(vals.cpu().numpy(), pat)
        components = defaultdict(list)
        for j in range(batch_size):
            l = len([k for k in counts[j] if k >= filtr]) - 1
            components[l].append(j)
            comp_counts[l] += 1
            if l > max_comp_found:
                max_comp_found = l
            if l >= min_comp and len(cache0) < cache_len:
                cache0.append(coefs.cpu().numpy()[:,j])
            if l == max_comp_found:
                cache1.append(coefs.cpu().numpy()[:,j])
        iteration += 1
        total_samples += batch_size
        input_char = stdscr.getch(0, 0)
        if input_char in [ord('q'), ord('Q')]:
            save_cache("save_cache.txt", cache1)
            break

def loop_H(stdscr, deg, width, batch_size, filtr=3):
    cr.use_default_colors()
    stdscr.nodelay(True)
    total_samples = 0
    iteration = 0
    max_comp_found = 0
    cache_len = 1000
    cache0 = deque([], cache_len)
    cache1 = deque([], cache_len)
    
    hmons = monomials(2 * deg - 4)
    coefs = torch.zeros((deg_to_dim(deg), batch_size), dtype=dtype)
    hcoefs = torch.zeros((deg_to_dim(2 * deg - 4), batch_size), dtype=dtype)
    vals = torch.zeros((batch_size, width, width), dtype=dtype)
    xx = torch.linspace(-1, 1, width, dtype=dtype)
    xx = xx.reshape((width, 1)) * torch.ones((1, width), dtype=dtype)
    yy = torch.linspace(-1, 1, width, dtype=dtype)
    yy = torch.ones((width, 1), dtype=dtype) * yy.reshape((1, width))
    nn = xx**2 + yy**2
    pts = torch.stack([2. * xx, 2. * yy, nn - 1.])
    pts = pts / (nn + 1.)
    pat = np.ndarray((12, 2, width, width), dtype=np.int32)
    nbr_pattern(width, pat)
    comp_counts = defaultdict(int)
    
    while True:
        stdscr.erase()
        stdscr.addstr(0, 0, f"Using {device} device")
        stdscr.addstr(1, 0, f'Iteration: {iteration};\t Total samples: {total_samples}')
        stdscr.addstr(2, 0, "Total counts: "
                      + " ".join([f"{i}: {comp_counts[i]};"
                                  for i in sorted(comp_counts.keys())]))
        stdscr.addstr(3, 0, f"Cache 0: {len(cache0)};\t Cache 1: {len(cache1)}")
        stdscr.addstr(4, 0, f"Maximal number of components found: {max_comp_found}")
        stdscr.refresh()
        dice = random.randrange(3)
        r = 1e-1
        if dice == 0 and len(cache0) > 0:
            coefs = torch.randn_like(coefs, dtype=dtype) * r
            center = cache0.popleft()
            tcenter = torch.tensor(center, dtype=dtype).reshape((deg_to_dim(deg), 1))
            coefs.add_(tcenter.matmul(torch.ones((1, batch_size), dtype=dtype)))
            normalize(coefs)
        elif dice == 1 and len(cache1) > 0:
            coefs = torch.randn_like(coefs, dtype=dtype) * r
            center = cache1[random.randrange(len(cache1))]
            tcenter = torch.tensor(center, dtype=dtype).reshape((deg_to_dim(deg), 1))
            coefs.add_(tcenter.matmul(torch.ones((1, batch_size), dtype=dtype)))
            normalize(coefs)
        else:
            coefs = sample(coefs)
        hessian(deg, coefs, hcoefs)
        evaluate(hmons, hcoefs, pts, vals)
        counts = connected.components_batch(vals.cpu().numpy(), pat)
        components = defaultdict(list)
        for j in range(batch_size):
            l = len([k for k in counts[j] if k >= filtr]) - 1
            components[l].append(j)
            comp_counts[l] += 1
            if l > max_comp_found:
                max_comp_found = l
            if l == max_comp_found:
                    cache1.append(coefs.cpu().numpy()[:,j])
            elif l >= max_comp_found - 2 and len(cache0) < cache_len:
                cache0.append(coefs.cpu().numpy()[:,j])
        iteration += 1
        total_samples += batch_size
        input_char = stdscr.getch(0, 0)
        if input_char in [ord('q'), ord('Q')]:
            save_cache("save_cache_H.txt", cache1)
            break

random.seed()
try:
    cr.wrapper(loop_H, 6, 100, 10000)
except Exception as e:
    cr.endwin()
    print(f"An error occurred: {e}")
