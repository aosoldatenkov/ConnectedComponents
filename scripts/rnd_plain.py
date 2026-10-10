#!/usr/bin/env python3
"""RND strategy for plane curves f = 0 (no Hessian): iterated random perturbations of a pool of forms with many
ovals, parents weighted by 4^count, radius log-uniform in [rmin, rmax] (relative to |f|).

Children with count >= max(parent count, admit) join the pool (counts >= 10 must agree at width 800 first).
Candidates above the target level are confirmed at widths 800 and 1600 and saved; then checked with the exact
upper bound (pencil tangents).

    uv run python scripts/rnd_plain.py --deg 6 --minutes 10 --target 11
"""

import argparse
import collections
import glob
import time

import numpy as np
import torch

from conncomp import DEVICE, DTYPE
from conncomp.certify import certify_upper_bound, form_poly
from conncomp.io import load_coefs, save_coefs
from conncomp.polynomials import normalize, perturb
from conncomp.scan import Experiment


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deg", type=int, default=6)
    ap.add_argument("--pool", nargs="+", default=sorted(glob.glob("data/cache_f_deg6_*.txt")))
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--batch", type=int, default=20000)
    ap.add_argument("--rmin", type=float, default=1e-4)
    ap.add_argument("--rmax", type=float, default=1e-1)
    ap.add_argument("--admit", type=int, default=9)
    ap.add_argument("--target", type=int, default=11)
    ap.add_argument("--max-pool", type=int, default=3000)
    ap.add_argument("--out", default="data/strategies/rnd_plain_found.txt")
    args = ap.parse_args()
    d = args.deg
    e400, e800, e1600 = (Experiment(d, w, False) for w in (400, 800, 1600))

    def count(f, e):
        return torch.tensor(e.count(f, 3), device=DEVICE) - 1

    forms = [c for fn in args.pool for c in load_coefs(fn)]
    F = normalize(torch.as_tensor(np.array(forms).T, dtype=DTYPE, device=DEVICE))
    harnack = (d - 1) * (d - 2) // 2 + 1
    n4, n8 = count(F, e400), count(F, e800)
    sel = ((n4 == n8) & (n4 >= 8) & (n4 <= harnack)).nonzero().squeeze(1)
    hi = sel[n4[sel] >= 10]
    if hi.numel():  # high counts must also agree at width 1600 (near-singular forms fool coarser grids)
        bad = hi[count(F[:, hi], e1600) != n4[hi]]
        sel = sel[~torch.isin(sel, bad)]
    X = F[:, sel].T.cpu().numpy()
    keep = []
    for i, x in enumerate(X):
        if all(min(np.linalg.norm(x - X[j]), np.linalg.norm(x + X[j])) > 1e-3 for j in keep):
            keep.append(i)
    pool_f = F[:, sel[keep]]
    pool_n = n4[sel[keep]]
    levels = dict(sorted(collections.Counter(pool_n.tolist()).items()))
    print(f"pool: {pool_f.shape[1]} distinct forms with stable counts {levels}", flush=True)

    t0, evals, changes, found = time.time(), 0, collections.Counter(), []
    rounds = 0
    while time.time() - t0 < args.minutes * 60:
        w = 4.0 ** (pool_n.double() - pool_n.max())
        idx = torch.multinomial(w / w.sum(), args.batch // 20, replacement=True)
        children, parent_n = [], []
        for i in idx.tolist():
            r = 10 ** np.random.uniform(np.log10(args.rmin), np.log10(args.rmax))
            children.append(perturb(pool_f[:, i], 20, r))
            parent_n.append(int(pool_n[i]))
        C = torch.cat(children, 1)
        pn = torch.tensor(np.repeat(parent_n, 20), device=DEVICE)
        n = count(C, e400)
        evals += C.shape[1]
        changes.update((n - pn).clamp(-3, 3).tolist())
        adm = ((n >= pn) & (n >= args.admit) & (n <= harnack)).nonzero().squeeze(1)
        if adm.numel():
            hi = adm[n[adm] >= 10]
            ok = torch.ones(adm.numel(), dtype=torch.bool, device=DEVICE)
            if hi.numel():
                n8h = count(C[:, hi], e800)
                ok[torch.isin(adm, hi[n8h != n[hi]])] = False
            adm = adm[ok]
            pool_f = torch.cat([pool_f, C[:, adm]], 1)
            pool_n = torch.cat([pool_n, n[adm]])
            if pool_n.numel() > args.max_pool:  # keep by level first (highest), then by recency
                order = torch.arange(pool_n.numel(), device=DEVICE)
                key = pool_n.double() * pool_n.numel() + order.double()
                keep = torch.sort(torch.argsort(key, descending=True)[: args.max_pool]).values
                pool_f, pool_n = pool_f[:, keep], pool_n[keep]
        up = (n >= args.target).nonzero().squeeze(1)
        if up.numel():
            n8 = count(C[:, up], e800)
            up = up[n8 == n[up]]
            if up.numel():
                n16 = count(C[:, up], e1600)
                found += [(int(n[j]), C[:, j].cpu().numpy()) for j, k in zip(up.tolist(), n16.tolist()) if k == n[j]]
        rounds += 1
        if rounds % 50 == 0:
            levels = dict(sorted(collections.Counter(pool_n.tolist()).items()))
            print(f"  {time.time() - t0:5.0f}s: {evals:,} children; pool levels {levels}; "
                  f"candidates >= {args.target}: {len(found)}", flush=True)
    dt = time.time() - t0
    print(f"{evals:,} children in {dt:.0f}s ({evals / dt:,.0f}/s); count change vs parent: "
          f"{dict(sorted(changes.items()))}; grid-confirmed candidates >= {args.target}: {len(found)}", flush=True)
    if found:
        save_coefs(args.out, [c for _, c in found])
        X = np.array([c for _, c in found])
        distinct = []
        for i in range(len(X)):
            if all(min(np.linalg.norm(X[i] - X[j]), np.linalg.norm(X[i] + X[j])) > 1e-3 for j in distinct):
                distinct.append(i)
        res = collections.Counter()
        for j in distinct[:10]:
            ints = [int(v) for v in np.rint(X[j] / np.abs(X[j]).max() * 10**9)]
            ub = certify_upper_bound(form_poly(ints, d), target=args.target, n_base=3, n_frames=3)
            res["no base point" if ub is None else (ub.upper_bound if ub.certified else "singular")] += 1
        print(f"{len(distinct)} distinct candidates; exact upper bounds of up to 10 of them: {dict(res)}")


if __name__ == "__main__":
    main()
