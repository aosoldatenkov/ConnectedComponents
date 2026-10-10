#!/usr/bin/env python3
"""Validate gradients of critical values (conncomp.critical) by finite differences, and try single wall crossings.

1. Plain sextics (parameters = coefficients of the curve), quintic Hessians (parameters = coefficients of f), and
   the x -> -x symmetric family (parameters = basis coordinates): perturb by eps * delta, re-locate the critical
   points (track), compare the change of each critical value with eps * grad . delta.
2. Wall crossing for quintic Hessians: for the critical point with the smallest |v|, step
   theta <- theta - (1 + eta) v grad / |grad|^2 and check that v changes sign; oval counts before and after.

    uv run python scripts/check_critical_gradients.py
"""

import collections

import numpy as np
import torch

from conncomp import DEVICE, DTYPE
from conncomp.critical import (MAX, MIN, SADDLE, critical_points, cross_walls, hessian_value_gradients, track,
                               value_gradients)
from conncomp.hessian import hessian
from conncomp.polynomials import sample
from conncomp.scan import Experiment
from conncomp.symmetry import symmetry

NAMES = {MIN: "min", SADDLE: "saddle", MAX: "max"}


def fd_report(name, crit, grad, values_after, eps_list):
    print(f"{name}: {len(crit)} critical points")
    for eps, (vnew, conv, delta_dot) in zip(eps_list, values_after):
        pred = eps * delta_dot
        ok = conv
        rel = ((vnew - crit.value) - pred).abs()[ok] / pred.abs()[ok].clamp(min=1e-300)
        q = np.quantile(rel.cpu().numpy(), [0.5, 0.9, 0.99])
        print(f"  eps={eps:.0e}: tracked {conv.float().mean().item():.2%}; relative error of the first-order "
              f"prediction: median {q[0]:.1e}, 90% {q[1]:.1e}, 99% {q[2]:.1e}")


def check_plain(B=2000, deg=6, eps_list=(1e-3, 1e-4, 1e-5, 1e-6)):
    torch.manual_seed(0)
    c = sample(deg, B)
    crit, _ = critical_points(c, deg)
    grad = value_gradients(crit, c, deg)
    delta = torch.randn_like(c)
    delta /= delta.norm(dim=0, keepdim=True)
    out = []
    for eps in eps_list:
        new = track(crit, c + eps * delta, deg)
        out.append((new.value, new.converged, (grad * delta.T[crit.form]).sum(1)))
    fd_report(f"plain degree-{deg} forms", crit, grad, out, eps_list)


def check_hessian(B=2000, deg=5, eps_list=(1e-3, 1e-4, 1e-5, 1e-6), basis=None, name=None):
    torch.manual_seed(1)
    f = sample(deg, B, basis=basis)
    H = hessian(deg, f)
    crit, _ = critical_points(H, 2 * deg - 4)
    grad = hessian_value_gradients(crit, f, deg, basis=basis)
    if basis is None:
        delta = torch.randn_like(f)
        delta /= delta.norm(dim=0, keepdim=True)
        step, coords = delta, delta.T[crit.form]
    else:
        Bt = torch.as_tensor(np.asarray(basis, dtype=np.float64), dtype=DTYPE, device=DEVICE)
        da = torch.randn((Bt.shape[1], B), dtype=DTYPE, device=DEVICE)
        da /= da.norm(dim=0, keepdim=True)
        step, coords = Bt @ da, da.T[crit.form]
    out = []
    for eps in eps_list:
        new = track(crit, hessian(deg, f + eps * step), 2 * deg - 4)
        out.append((new.value, new.converged, (grad * coords).sum(1)))
    fd_report(name or f"Hessians of degree-{deg} forms", crit, grad, out, eps_list)


def wall_crossings(B=50000, keep=2000, deg=5, width=800):
    """Cross the closest wall (estimated distance |v| / |grad v|) of the `keep` forms closest to a wall."""
    torch.manual_seed(2)
    f = sample(deg, B)
    crit, _ = critical_points(hessian(deg, f), 2 * deg - 4)
    g = hessian_value_gradients(crit, f, deg)
    dist = crit.value.abs() / g.norm(dim=1) / f.norm(dim=0)[crit.form]
    order = torch.argsort(dist)
    first = torch.full((B,), len(crit), dtype=torch.long, device=DEVICE)
    first.scatter_reduce_(0, crit.form[order], order, reduce="amin", include_self=True)
    target = crit.select(first[first < len(crit)])
    target = target.select(torch.argsort(dist[first[first < len(crit)]])[:keep])
    dist = dist[first[first < len(crit)]]
    f_new, after, crossed, rel = cross_walls(target, f, deg)
    exp = Experiment(deg, width, True)
    before = torch.tensor(exp.count(f[:, target.form], 3)) - 1
    after_n = torch.tensor(exp.count(f_new, 3)) - 1
    print(f"\nwall crossings for the {len(target)} of {B} quintic Hessians closest to a wall (target v = -v0):")
    print(f"  estimated distance to the wall: median {torch.median(torch.sort(dist).values[:keep]).item():.1e}; "
          f"actual relative parameter change: median {torch.median(rel).item():.1e}")
    print(f"  critical value crossed zero: {crossed.float().mean().item():.1%}; "
          f"|v after / v before| median {torch.median((after.value / target.value).abs()).item():.3f}")
    c = crossed.cpu()
    for k in (MIN, SADDLE, MAX):
        sel = (target.kind.cpu() == k) & c
        side = torch.sign(target.value.cpu())[sel]
        for sgn, label in ((1, "v > 0"), (-1, "v < 0")):
            ch = collections.Counter((after_n - before)[sel][side == sgn].tolist())
            if ch:
                print(f"  {NAMES[k]:6s} {label} ({sum(ch.values())} crossings): change of the oval count "
                      f"{dict(sorted(ch.items()))}")


if __name__ == "__main__":
    import sys

    if "--crossings-only" not in sys.argv:
        check_plain()
        check_hessian()
        check_hessian(basis=symmetry("x", 5).basis, name="Hessians, x -> -x symmetric family (basis coordinates)")
    wall_crossings()
