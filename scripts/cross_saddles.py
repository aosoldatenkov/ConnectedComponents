#!/usr/bin/env python3
"""Push the nearest saddle walls of the Hessian curves H(f) of a pool of sextics through zero and count the ovals.

For each form, the `per` saddles with the smallest estimated wall distance in the parameter space S (full, x-family
or xy-family, see diagnose_births.py) are crossed one at a time with conncomp.critical.cross_walls (Newton steps
along the gradient projected onto S, re-tracking the saddle, overshoot factor). Children are counted on FaceMesh(200)
and with the exact shift test at N = 400; the forms with more ovals than the pool's maximum are saved.

    uv run python scripts/cross_saddles.py pools/hessian6_test/xy_genuine16.txt --n 300 --space x
"""

import argparse
import collections
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import hessian6_scan as H6  # noqa: E402
from diagnose_births import projector  # noqa: E402
from conncomp import DEVICE  # noqa: E402
from conncomp.critical import SADDLE, critical_points, cross_walls, hessian_value_gradients  # noqa: E402
from conncomp.gpu_trees import FaceMesh  # noqa: E402
from conncomp.io import load_coefs, save_coefs  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool", type=Path)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--per", type=int, default=20, help="nearest saddles crossed per form")
    ap.add_argument("--space", choices=["full", "x", "xy"], default="x")
    ap.add_argument("--overshoot", type=float, nargs="+", default=[1.0, 3.0])
    ap.add_argument("--out", type=Path, default=Path("pools/hessian6_test/crossed"))
    args = ap.parse_args()
    F = np.asarray(load_coefs(args.pool), dtype=np.float64)[: args.n]
    f = torch.as_tensor(F.T, device=DEVICE)
    f = f / f.norm(dim=0, keepdim=True)
    D, n = f.shape
    P = projector(args.space, D)
    Q = torch.linalg.qr(P)[0][:, : int(round(float(torch.trace(P))))]  # orthonormal basis of S (D, k)
    m200, m400 = FaceMesh(200), FaceMesh(400)
    base = H6.ovals(m200, H6.curves(f)).cpu().numpy()
    top = int(base.max())
    crit, morse = critical_points(H6.curves(f), 8)
    crit = crit.select((morse[crit.form] == 1) & (crit.kind == SADDLE))
    g = hessian_value_gradients(crit, f, 6)
    d = crit.value.abs() / (g @ P).norm(dim=1).clamp(min=1e-300)
    sel = []
    for k in range(n):  # nearest `per` saddles of each form
        idx = torch.nonzero(crit.form == k)[:, 0]
        sel.append(idx[torch.argsort(d[idx])[: args.per]])
    tgt = crit.select(torch.cat(sel))
    args.out.mkdir(parents=True, exist_ok=True)
    for over in args.overshoot:
        res = []  # cross_walls steps along the basis Q (orthonormal basis of S): f + Q da
        for s0 in range(0, len(tgt), 4000):
            t = tgt.select(torch.arange(s0, min(s0 + 4000, len(tgt)), device=DEVICE))
            cols, cur, crossed, rel = cross_walls(t, f, 6, overshoot=over, basis=Q)
            res.append((cols[:, crossed], t.form[crossed]))
        cols = torch.cat([r[0] for r in res], dim=1)
        par = torch.cat([r[1] for r in res])
        cols = cols / cols.norm(dim=0, keepdim=True)
        n200 = H6.ovals(m200, H6.curves(cols)).cpu().numpy()
        delta = n200 - base[par.cpu().numpy()]
        print(f"space {args.space}, overshoot {over}: {len(tgt)} saddle walls, {cols.shape[1]} crossed; "
              f"change of the oval count at N=200: {dict(sorted(collections.Counter(delta.tolist()).items()))}",
              flush=True)
        hi = np.nonzero(n200 > top)[0]
        conf = []
        for i in hi:
            Hs, _, rob, _ = H6.shift(H6.curves(cols[:, i : i + 1])[:, 0])
            ns = int(H6.ovals(m400, Hs[:, None])[0])
            conf.append(ns)
        if len(hi):
            print(f"   {len(hi)} children above {top} at N=200; shift test at N=400: "
                  f"{dict(collections.Counter(conf))}", flush=True)
            good = [i for i, ns in zip(hi, conf) if ns > top]
            if good:
                save_coefs(args.out / f"{args.space}_over{over:g}_gt{top}.txt", cols[:, good].T.cpu().numpy())


if __name__ == "__main__":
    main()
