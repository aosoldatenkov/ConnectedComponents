#!/usr/bin/env python3
"""Random scan of Hessian curves H(f) of sextics f (curves of degree 8): oval counts with the fused CUDA kernel.

Forms f from the uniform measure on S^27 and the Kostlan measure (alternating batches); per measure a histogram of
the oval counts (FaceMesh(N), default N = 40); all forms with at least `keep` ovals are stored. At the end, the forms
with the most ovals are re-checked: oval counts at N = 200 and 400, the exact shift test (H + c (x^2+y^2+z^2)^4 with
sign(c) = sign of the smallest critical value and |c| below all critical values of the opposite sign: isotopic, so
the same topology with wider necks), robustness and nesting types.

Bertrand-Brugalle: (d - 2)^2 = 16 compact ovals in R^2 for d = 6; Harnack bound for degree 8: 22.

    uv run python scripts/hessian6_scan.py --minutes 10
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import sympy as sp
import torch

from conncomp import DEVICE, DTYPE
from conncomp.critical import critical_points
from conncomp.gpu_trees import FaceMesh, to_nesting_trees
from conncomp.hessian import hessian
from conncomp.polynomials import from_sympy, monomials, normalize, sample

DEGF, DEGH = 6, 8
D = len(monomials(DEGF))
KOSTLAN = torch.tensor([math.sqrt(math.factorial(DEGF) / (math.factorial(i) * math.factorial(j) * math.factorial(k)))
                        for i, j, k in monomials(DEGF)], dtype=DTYPE, device=DEVICE)


def draw(measure, B):
    if measure == "uniform":
        return sample(DEGF, B)
    return normalize(KOSTLAN[:, None] * torch.randn((D, B), dtype=DTYPE, device=DEVICE))


def curves(f):
    H = hessian(DEGF, f)
    return H / H.norm(dim=0, keepdim=True)


def ovals(fm, H):
    """Oval counts (-1 if the region tree is inconsistent) of the curves H (D_H, B)."""
    t = fm.trees_fused(H, DEGH)
    return torch.where(t.ok, t.n_ovals, torch.full_like(t.n_ovals, -1))


def shift(H):
    """The exact shift test: H + c r^8 (see the module docstring); returns (shifted curve, c, robustness of H)."""
    X, Y, Z = sp.symbols("x y z")
    r8 = torch.tensor([float(v) for v in from_sympy((X**2 + Y**2 + Z**2) ** 4, DEGH, X, Y, Z)], dtype=DTYPE,
                      device=DEVICE)
    crit, morse = critical_points(H[:, None], DEGH)
    v = crit.value
    t = v[v.abs().argmin()]
    opp = v[torch.sign(v) != torch.sign(t)].abs()
    c = float(torch.sign(t) * 0.5 * opp.min()) if len(opp) else 0.0
    return H + c * r8, c, float(v.abs().min()), int(morse[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--batch", type=int, default=20000)
    ap.add_argument("--keep", type=int, default=12, help="store all forms with at least this many ovals")
    ap.add_argument("--check", type=int, default=30, help="forms with the most ovals re-checked at the end")
    ap.add_argument("--out", type=Path, default=Path("pools/hessian6_test"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(time.time()))
    fm = FaceMesh(args.N)
    # sanity check: the earlier certified 14-oval forms
    from conncomp.io import load_coefs

    ref = torch.as_tensor(np.asarray(load_coefs("data/save_cache_H.txt"), dtype=np.float64)[:10].T, device=DEVICE)
    print("earlier 14-oval forms (data/save_cache_H.txt) at N =", args.N, ":",
          ovals(fm, curves(ref)).tolist(), flush=True)

    hist = {m: np.zeros(64, dtype=np.int64) for m in ("uniform", "kostlan")}
    kept_f, kept_n, kept_m = [], [], []
    t0, t_log, it = time.time(), time.time(), 0
    while time.time() - t0 < args.minutes * 60:
        m = "uniform" if it % 2 == 0 else "kostlan"
        f = draw(m, args.batch)
        n = ovals(fm, curves(f))
        hist[m] += torch.bincount(n.clamp(min=0, max=63), minlength=64).cpu().numpy()
        hi = (n >= args.keep).nonzero()[:, 0]
        if hi.numel():
            kept_f.append(f[:, hi].T.cpu().numpy())
            kept_n.append(n[hi].cpu().numpy())
            kept_m += [m] * hi.numel()
        it += 1
        if time.time() - t_log > 120:
            t_log = time.time()
            tot = sum(h.sum() for h in hist.values())
            top = max(k for h in hist.values() for k in np.nonzero(h)[0])
            print(f"[{(time.time() - t0) / 60:4.1f} min] {tot:,} forms ({tot / (time.time() - t0):,.0f}/s); max "
                  f"{top} ovals; counts >= 12: " + ", ".join(f"{k}: {sum(h[k] for h in hist.values())}"
                                                       for k in range(12, top + 1)), flush=True)
    el = time.time() - t0
    F = np.concatenate(kept_f) if kept_f else np.zeros((0, D))
    nk = np.concatenate(kept_n) if kept_n else np.zeros(0, dtype=np.int64)
    np.savez(args.out / "kept.npz", f=F, n=nk, measure=np.array(kept_m))
    tot = sum(h.sum() for h in hist.values())
    print(f"done: {tot:,} forms in {el / 60:.1f} min ({tot / el:,.0f}/s)", flush=True)
    for m, h in hist.items():
        nz = np.nonzero(h)[0]
        print(f"  {m}: " + ", ".join(f"{k}: {h[k] / h.sum():.3g}" for k in nz), flush=True)

    # re-check the forms with the most ovals
    order = np.argsort(-nk, kind="stable")[: args.check]
    m200, m400 = FaceMesh(200), FaceMesh(400)
    rows = []
    for i in order:
        f = torch.as_tensor(F[i][:, None], device=DEVICE)
        H = curves(f)[:, 0]
        Hs, c, rob, morse = shift(H)
        n200, n400 = int(ovals(m200, H[:, None])[0]), int(ovals(m400, H[:, None])[0])
        ns = int(ovals(m400, Hs[:, None])[0])
        typ = to_nesting_trees(m400.trees(Hs[:, None], DEGH), 1)[0].type
        rows.append({"index": int(i), "measure": kept_m[i], "ovals_N40": int(nk[i]), "ovals_N200": n200,
                     "ovals_N400": n400, "ovals_shifted_N400": ns, "type": typ, "robustness": rob, "shift": c,
                     "morse": morse})
        print(f"  #{i} ({kept_m[i]}): N={args.N} {nk[i]}, N=200 {n200}, N=400 {n400}, shift test {ns} ({typ}); "
              f"robustness {rob:.1e}", flush=True)
    (args.out / "summary.json").write_text(json.dumps({"forms": int(tot), "minutes": el / 60, "N": args.N,
                                                       "histograms": {m: h.tolist() for m, h in hist.items()},
                                                       "checked": rows}, indent=1))


if __name__ == "__main__":
    main()
