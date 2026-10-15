#!/usr/bin/env python3
"""Where could a new oval be born? Wall distances of the Hessian curves H(f) of a pool of sextics f (degree-8 curves),
restricted to symmetric families.

For each form: the critical points of H (unit coefficient vector) on RP^2. A birth wall is an extremum on the wrong side
(a maximum with value v < 0 or a minimum with v > 0): pushing v through 0 creates a small new oval around the point.
Its estimated distance in a parameter space S (unit f) is |v| / |P_S grad_f v|, where P_S is the orthogonal projection
onto S: the full space, the x-family (f(-x, y) = f(x, y)) and the xy-family (f = F(x^2, y^2)), spans of
conncomp.symmetry bases. Saddles are reported too (merges/splits; whether they add an oval depends on the topology).
Critical points on the lines x = 0, y = 0, z = 0 are tagged: under a perturbation odd under a reflection fixing the
point, its value moves only at second order, so the linear estimate is infinite there.

    uv run python scripts/diagnose_births.py pools/hessian6_test/xy_genuine16.txt --n 200
"""

import argparse
import collections
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import hessian6_scan as H6  # noqa: E402
from conncomp import DEVICE, DTYPE  # noqa: E402
from conncomp.critical import MAX, MIN, SADDLE, critical_points, hessian_value_gradients  # noqa: E402
from conncomp.io import load_coefs  # noqa: E402
from conncomp.symmetry import symmetry  # noqa: E402

SPACES = ("full", "x", "xy")


def projector(spec, D):
    if spec == "full":
        return torch.eye(D, dtype=DTYPE, device=DEVICE)
    B = torch.as_tensor(np.asarray(symmetry(spec, 6).basis, dtype=np.float64), device=DEVICE)
    Q, _ = torch.linalg.qr(B)
    return Q @ Q.T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool", type=Path)
    ap.add_argument("--n", type=int, default=200, help="first n forms of the pool")
    ap.add_argument("--axis-tol", type=float, default=1e-6)
    args = ap.parse_args()
    F = np.asarray(load_coefs(args.pool), dtype=np.float64)[: args.n]
    f = torch.as_tensor(F.T, device=DEVICE)
    f = f / f.norm(dim=0, keepdim=True)
    D = f.shape[0]
    P = {s: projector(s, D) for s in SPACES}
    C = H6.curves(f)
    crit, morse = critical_points(C, 8)
    ok = morse[crit.form] == 1
    crit = crit.select(ok)
    g = hessian_value_gradients(crit, f, 6)  # (K, D) with respect to the unit forms f
    dist = {s: crit.value.abs() / (g @ P[s]).norm(dim=1).clamp(min=1e-300) for s in SPACES}
    p = crit.point.abs()
    tag = np.array(["x=0" if a < args.axis_tol else "y=0" if b < args.axis_tol else "z=0" if c < args.axis_tol
                    else "generic" for a, b, c in p.cpu().numpy()])
    kind, val = crit.kind.cpu().numpy(), crit.value.cpu().numpy()
    birth = ((kind == MAX) & (val < 0)) | ((kind == MIN) & (val > 0))
    sad = kind == SADDLE
    form = crit.form.cpu().numpy()
    d = {s: dist[s].cpu().numpy() for s in SPACES}
    n = F.shape[0]
    print(f"{n} forms ({int((morse == 1).sum())} pass the Morse check); {len(kind)} critical points, "
          f"{birth.sum()} birth extrema, {sad.sum()} saddles")
    print("birth extrema by location:", dict(collections.Counter(tag[birth].tolist())))
    for what, m in (("birth extrema", birth), ("saddles", sad)):
        print(f"\n{what}: nearest wall per form (median over forms; quantiles of all events)")
        for s in SPACES:
            per = np.full(n, np.inf)
            np.minimum.at(per, form[m], d[s][m])
            fin = d[s][m][np.isfinite(d[s][m]) & (d[s][m] < 1e3)]
            print(f"  {s:4s}: per-form nearest median {np.median(per):.2e}, best {per.min():.2e}; "
                  f"events with d < 1e-3: {(d[s][m] < 1e-3).sum()}, < 1e-2: {(d[s][m] < 1e-2).sum()}"
                  + (f"; 10% quantile {np.quantile(fin, 0.1):.2e}" if len(fin) else ""))
    # the closest birth walls reachable within the x-family but not within the xy-family
    gain = d["xy"] / d["x"]
    order = np.argsort(d["x"] + np.where(birth, 0, np.inf))
    print("\nclosest birth walls in the x-family (form, location, kind, value, d_x, d_xy, d_full):")
    for i in order[:12]:
        if not birth[i]:
            break
        print(f"  form {form[i]:3d} {tag[i]:7s} {['min', 'saddle', 'max'][kind[i]]:6s} v={val[i]:+.1e} "
              f"d_x={d['x'][i]:.1e} d_xy={d['xy'][i]:.1e} d_full={d['full'][i]:.1e} (xy/x {gain[i]:.1f})")


if __name__ == "__main__":
    main()
