#!/usr/bin/env python3
"""Targeted births of nested ovals for plane curves f = 0 with non-nested ovals (e.g. 10 ovals -> 9 u 1<1>).

For each form: the sign s of the outside region N (the sign class with a single region), interiors have sign -s.
Extrema inside interiors whose critical value has sign -s and that point towards s (max with v < 0 if -s < 0,
min with v > 0 if -s > 0) mark walls whose crossing creates a nested oval. The closest such walls (estimated
distance |v| / |grad v|) are crossed by Newton steps on v = -overshoot * v0 along grad v (plain-curve gradients),
and the results are re-counted, with the number of regions per sign class.

    uv run python scripts/nest_birth.py --level 10 --per-form 3
"""

import argparse
import collections
import glob

import numpy as np
import torch

from conncomp import DEVICE, DTYPE
from conncomp.certify import certify_upper_bound, form_poly
from conncomp.components import component_labels
from conncomp.critical import MAX, MIN, Critical, critical_points, track, value_gradients
from conncomp.io import load_coefs, save_coefs
from conncomp.polynomials import normalize
from conncomp.scan import Experiment


def region_signs(e, f):
    """Numbers of regions (>= 3 pixels) of each sign class of the form f (column) on the grid of e."""
    v = e.values(f[:, None])[0].cpu().numpy()
    labels, sizes = component_labels(v, e.pat)
    out = collections.Counter()
    for i, s in enumerate(sizes):
        if s >= 3:
            pix = tuple(np.argwhere(labels == i)[0])
            out[1 if v[pix] >= 0 else -1] += 1
    return out


def cross(crit, cols, deg, overshoot=1.0, iters=8):
    """Newton on v = -overshoot v0 along the gradient (parameters = coefficients of the curve); one point per form."""
    v0 = crit.value.clone()
    cur = crit
    for _ in range(iters):
        g = value_gradients(cur, cols, deg)
        step = ((-overshoot * v0 - cur.value) / (g * g).sum(1).clamp(min=1e-300))[:, None] * g
        cols = cols + step.T
        cur = track(cur, cols, deg)
    crossed = (torch.sign(cur.value) == -torch.sign(v0)) & cur.converged
    return cols, crossed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deg", type=int, default=6)
    ap.add_argument("--level", type=int, default=10)
    ap.add_argument("--per-form", type=int, default=3)
    ap.add_argument("--overshoot", type=float, nargs="+", default=[0.3, 1.0])
    args = ap.parse_args()
    d = args.deg
    files = sorted(glob.glob("data/cache_f_deg6_*.txt")) + sorted(glob.glob("data/strategies/rnd_plain*.txt"))
    forms = [c for fn in files for c in load_coefs(fn)]
    F = normalize(torch.as_tensor(np.array(forms).T, dtype=DTYPE, device=DEVICE))
    e400, e800, e1600 = (Experiment(d, w, False) for w in (400, 800, 1600))
    n = [torch.tensor(e.count(F, 3)) - 1 for e in (e400, e800, e1600)]
    sel = ((n[0] == args.level) & (n[1] == args.level) & (n[2] == args.level)).nonzero().squeeze(1)
    X = F[:, sel].T.cpu().numpy()
    keep = []
    for i, x in enumerate(X):
        if all(min(np.linalg.norm(x - X[j]), np.linalg.norm(x + X[j])) > 1e-3 for j in keep):
            keep.append(i)
    F = F[:, sel[keep]]
    B = F.shape[1]
    print(f"{B} distinct forms with {args.level} ovals (stable at 400/800/1600)")
    s_out = torch.tensor([max(region_signs(e800, F[:, b]).items(), key=lambda kv: -kv[1] if kv[1] == 1 else 0)[0]
                          if 1 in region_signs(e800, F[:, b]).values() else 0 for b in range(B)], device=DEVICE)
    crit, morse = critical_points(F, d)
    g = value_gradients(crit, F, d)
    dist = crit.value.abs() / g.norm(dim=1)
    inner = -s_out[crit.form]
    want = ((inner < 0) & (crit.kind == MAX) & (crit.value < 0)) | ((inner > 0) & (crit.kind == MIN) & (crit.value > 0))
    want &= s_out[crit.form] != 0
    print(f"candidate walls (extrema inside ovals pointing outwards): {int(want.sum())} in "
          f"{int(torch.unique(crit.form[want]).numel())} forms; distances: median "
          f"{torch.median(dist[want]).item():.2e}" if want.any() else "no candidate walls")
    if not want.any():
        return
    idx = want.nonzero().squeeze(1)
    idx = idx[torch.argsort(dist[idx])]
    idx = idx[torch.argsort(crit.form[idx], stable=True)]
    rank = torch.zeros_like(idx)
    for k in range(1, len(idx)):
        rank[k] = rank[k - 1] + 1 if crit.form[idx[k]] == crit.form[idx[k - 1]] else 0
    idx = idx[rank < args.per_form]
    target = crit.select(idx)
    cols0 = F[:, target.form]
    t2 = Critical(*[getattr(target, k).clone() for k in Critical.__dataclass_fields__])
    t2.form = torch.arange(len(target), device=DEVICE)
    results, found = collections.Counter(), []
    for os in args.overshoot:
        cols, crossed = cross(t2, cols0.clone(), d, overshoot=os)
        cols = normalize(cols)
        c4, c8, c16 = (torch.tensor(e.count(cols, 3)) - 1 for e in (e400, e800, e1600))
        for j in range(len(target)):
            if not bool(crossed[j]):
                results[(os, "not crossed")] += 1
                continue
            key = (int(c4[j]), int(c8[j]), int(c16[j]))
            results[(os, key)] += 1
            if key[2] >= args.level + 1 and key[1] == key[2]:
                found.append(cols[:, j].cpu().numpy())
    for (os, key), c in sorted(results.items(), key=lambda kv: str(kv[0])):
        print(f"  overshoot {os}: counts at 400/800/1600 {key}: {c}")
    if found:
        save_coefs("data/strategies/nest_birth_found.txt", found)
        for f in found[:5]:
            T = torch.as_tensor(f[:, None], device=DEVICE)
            ints = [int(v) for v in np.rint(f / np.abs(f).max() * 10**9)]
            ub = certify_upper_bound(form_poly(ints, d), target=args.level + 1, n_base=3, n_frames=3)
            print(f"  candidate: regions per sign {dict(region_signs(e1600, T[:, 0]))}, exact upper bound "
                  f"{ub.upper_bound if ub and ub.certified else None}")


if __name__ == "__main__":
    main()
