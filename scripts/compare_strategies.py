#!/usr/bin/env python3
"""Compare search strategies near the discriminant for degree-5 Hessians, at equal wall-clock budget.

Strategies (iterated; children that keep the parent's count join the pool, so the search can drift on a level):
  RND      random perturbations, radius log-uniform in [1e-3, 1e-1] (relative to |f|)
  RND-DEG  parents weighted by degeneracy (walls within tau); radius = 3rd-nearest wall distance * logU[0.3, 3]
  WALL-1   cross one random wall among the 6 nearest (any type)
  WALL-k   cross a random subset of the 8 nearest walls simultaneously (Gauss-Newton on several constraints)
Children are counted at width 400; counts above the start level are confirmed at widths 800 and 1600 (near deep
strata of the discriminant, agreement at 400/800 alone is not enough).

    uv run python scripts/compare_strategies.py --start data/strategies/start_9.txt --level 9 --seconds 120
"""

import argparse
import collections
import time

import numpy as np
import torch

from conncomp import DEVICE, DTYPE
from conncomp.critical import critical_points, cross_walls, cross_walls_multi, degeneracy, wall_distances
from conncomp.hessian import hessian
from conncomp.io import load_coefs, save_coefs
from conncomp.polynomials import normalize, perturb
from conncomp.scan import Experiment

DEG = 5
E400, E800, E1600 = None, None, None


def counts(f, exp):
    return torch.tensor(exp.count(f, 3), device=f.device) - 1


def nearest_walls(f, k):
    """Critical points of the forms and, per form, the indices of the k nearest walls (padded with -1)."""
    crit, _ = critical_points(hessian(DEG, f), 2 * DEG - 4)
    d, _ = wall_distances(crit, f, DEG)
    B = f.shape[1]
    order = torch.argsort(d)
    order = order[torch.argsort(crit.form[order], stable=True)]
    cnt = torch.bincount(crit.form, minlength=B)
    start = torch.cumsum(cnt, 0) - cnt
    pos = torch.arange(len(crit), device=f.device) - start[crit.form[order]]
    near = torch.full((B, k), -1, dtype=torch.long, device=f.device)
    keep = pos < k
    near[crit.form[order][keep], pos[keep]] = order[keep]
    return crit, d, near


class Pool:
    def __init__(self, f, level, size=400):
        self.f, self.level, self.size = f, level, size
        self.n = counts(f, E400)

    def add(self, children, n):
        keep = n >= self.level
        if keep.any():
            self.f = torch.cat([self.f, children[:, keep]], 1)[:, -self.size:]
            self.n = torch.cat([self.n, n[keep]])[-self.size:]

    def parents(self, m, weights=None):
        p = torch.ones(self.f.shape[1], device=DEVICE) if weights is None else weights
        return torch.multinomial(p / p.sum(), m, replacement=True)


def step_rnd(pool, batch, gen):
    idx = pool.parents(batch // 20)
    out = []
    for i in idx.tolist():
        r = 10 ** (np.random.uniform(-3, -1))
        out.append(perturb(pool.f[:, i], 20, r))
    return torch.cat(out, 1)


def step_rnd_deg(pool, batch, gen, tau=1e-2):
    crit, d, near = nearest_walls(pool.f, 3)
    close, _, soft = degeneracy(crit, d, pool.f.shape[1], tau)
    scale = torch.where(near[:, 2] >= 0, d[near[:, 2].clamp(min=0)], torch.full_like(soft, 1e-2))
    idx = pool.parents(batch // 20, weights=soft + 1e-3)
    out = []
    for i in idx.tolist():
        r = float(scale[i]) * 10 ** np.random.uniform(np.log10(0.3), np.log10(3))
        out.append(perturb(pool.f[:, i], 20, r))
    return torch.cat(out, 1)


def step_wall1(pool, batch, gen):
    idx = pool.parents(min(batch, 2000))
    f = pool.f[:, idx]
    crit, d, near = nearest_walls(f, 6)
    pick = near[torch.arange(f.shape[1], device=DEVICE), torch.randint(0, 6, (f.shape[1],), device=DEVICE)]
    ok = pick >= 0
    target = crit.select(pick[ok])
    f_new, _, _, _ = cross_walls(target, f / f.norm(dim=0, keepdim=True), DEG)
    return normalize(f_new)


def step_wallk(pool, batch, gen):
    idx = pool.parents(min(batch, 2000))
    f = pool.f[:, idx]
    crit, d, near = nearest_walls(f, 8)
    choose = (torch.rand(near.shape, device=DEVICE) < 0.5) & (near >= 0)
    sel = torch.zeros(len(crit), dtype=torch.bool, device=DEVICE)
    sel[near[choose]] = True
    f_new, _ = cross_walls_multi(crit, f, DEG, sel)
    return normalize(f_new)


STRATEGIES = {"RND": step_rnd, "RND-DEG": step_rnd_deg, "WALL-1": step_wall1, "WALL-k": step_wallk}


def run(name, start, level, seconds, batch):
    torch.manual_seed(0)
    np.random.seed(0)
    pool = Pool(start.clone(), level)
    t0, evals, changes, found = time.time(), 0, collections.Counter(), []
    while time.time() - t0 < seconds:
        children = STRATEGIES[name](pool, batch, None)
        n = counts(children, E400)
        evals += children.shape[1]
        changes.update((n - level).clamp(-3, 3).tolist())
        up = (n > level).nonzero().squeeze(1)
        if up.numel():
            n8 = counts(children[:, up], E800)
            up = up[n8 == n[up]]
            n16 = counts(children[:, up], E1600) if up.numel() else n8[:0]
            conf = up[n16 == n[up]]
            found += [(int(n[j]), children[:, j].cpu().numpy()) for j in conf.tolist()]
        pool.add(children, n)
    dt = time.time() - t0
    best = max((k for k, _ in found), default=level)
    print(f"{name:8s} {evals:8d} children in {dt:.0f}s ({evals / dt:7.0f}/s) | count change vs {level}: "
          f"{dict(sorted(changes.items()))} | confirmed > {level}: {len(found)} (best {best}) | "
          f"{len(found) / dt * 60:.2f} per minute", flush=True)
    return found


def main():
    global E400, E800, E1600
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--batch", type=int, default=20000)
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    args = ap.parse_args()
    E400, E800, E1600 = Experiment(DEG, 400, True), Experiment(DEG, 800, True), Experiment(DEG, 1600, True)
    start = torch.as_tensor(np.array(load_coefs(args.start)).T, dtype=DTYPE, device=DEVICE)
    start = normalize(start)
    n = counts(start, E400)
    start = start[:, n == args.level]
    print(f"start set: {start.shape[1]} forms with {args.level} ovals; budget {args.seconds:.0f}s per strategy")
    for name in args.strategies:
        found = run(name, start, args.level, args.seconds, args.batch)
        if found:
            save_coefs(f"data/strategies/found_{name}_from{args.level}.txt", [c for _, c in found])


if __name__ == "__main__":
    main()
