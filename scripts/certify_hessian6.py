#!/usr/bin/env python3
"""Exact certification of oval counts of Hessian curves H(f) of sextics f (curves of degree 8) from coefficient files.

For each selected form: drop the terms of degree <= 1 in x, y (they do not affect H), round s f / max|f| to integers
for increasing scales s and keep the smallest scale whose exact Hessian has the expected oval count on the sphere mesh
(N = 200 and 400) and in the exact shift test; then certify in parallel CPU processes: upper bound by pencil tangents
(the interior pencil bound if loose), lower bound by separating polygons around the witnesses, FLINT cross-check.
Certificates (verifier format, "curve": "hessian") go to --out. Forms without a witness in every region are
skipped; each certification is limited to --timeout seconds.

    uv run python scripts/certify_hessian6.py output/cache2_H_6.txt --indices 3 2 338 --ovals 15
"""

import argparse
import json
import math
import multiprocessing as mp
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

DEGF, DEGH = 6, 8


def certify_task(task):
    """Worker (CPU only): (id, integer f, integer H, expected ovals, witnesses, seed) -> result dict."""
    from conncomp.certify import (certify_upper_bound, certify_upper_bound_interior, crosscheck_flint, form_poly)
    from conncomp.separation import certify_lower_bound

    tid, f, h, k, wit, seed = task
    t0 = time.time()
    try:
        H = form_poly(h, DEGH)
        ub = certify_upper_bound(H, target=k, seed=seed)
        if ub is None or not ub.certified:
            return tid, {"certified": False, "reason": "upper bound (no outside loop)", "seconds": time.time() - t0}
        lb = certify_lower_bound(H, wit)
        sep = lb.certificate
        upper, pencil, interior = ub.upper_bound, ub, None
        if upper > sep.lower_bound:
            interior = certify_upper_bound_interior(H, wit, target=sep.lower_bound, seed=seed)
            if interior is not None and interior.upper_bound < upper:
                upper, pencil = interior.upper_bound, interior.pencil
        flint = crosscheck_flint(H, pencil)
        ok = bool(flint["agree"]) and sep.lower_bound == upper
        cert = {"deg": DEGF, "hessian_deg": DEGH, "curve": "hessian", "f": f, "hessian": h,
                "proven_ovals": upper if ok else None, "upper": {"base_point": ub.base_point, "M": ub.M,
                                                                  "loop": ub.loop},
                "witnesses": [{"point": w} for w in sep.witnesses],
                "lower": {"paths": [{"polygon": p["polygon"], "ell": p["ell"]} for p in asdict(sep)["paths"]
                                    if p["ok"]]}}
        if interior is not None and pencil is interior.pencil:
            cert["upper_interior"] = {"M": interior.pencil.M, "line_point": interior.line_point}
        return tid, {"certified": ok, "upper_bound": upper, "lower_bound": sep.lower_bound,
                     "flint_agree": bool(flint["agree"]), "seconds": time.time() - t0, "cert": cert}
    except Exception as e:
        return tid, {"certified": False, "reason": f"error: {e!r}", "seconds": time.time() - t0}


def main():
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    import hessian6_scan as H6
    from conncomp import DEVICE, DTYPE
    from conncomp.gpu_trees import FaceMesh
    from conncomp.io import load_coefs
    from conncomp.polynomials import monomials
    from conncomp.rationalize import exact_hessian, witnesses

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path)
    ap.add_argument("--indices", type=int, nargs="+", required=True)
    ap.add_argument("--ovals", type=int, required=True)
    ap.add_argument("--max-scale-log2", type=int, default=24)
    ap.add_argument("--witness-width", type=int, default=1600)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--timeout", type=float, default=600, help="seconds per form")
    ap.add_argument("--out", type=Path, default=Path("data/hessian6/certificates"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    F = np.asarray(load_coefs(args.file), dtype=np.float64)
    low = np.array([i + j >= 2 for i, j, _ in monomials(DEGF)])
    m200, m400 = FaceMesh(200), FaceMesh(400)
    tasks, meta = [], {}
    for idx in args.indices:
        f = F[idx] * low
        f = f / np.abs(f).max()
        found = None
        for e in range(4, args.max_scale_log2 + 1):
            for s in (2 ** e, 3 * 2 ** (e - 1)):
                v = [int(x) for x in np.rint(s * f)]
                g = math.gcd(*v)
                v = [x // g for x in v] if g > 1 else v
                h = exact_hessian(DEGF, v)
                hm = max(abs(x) for x in h)
                Hc = torch.tensor([x / hm for x in h], dtype=DTYPE, device=DEVICE)
                Hc = Hc / Hc.norm()
                n = [int(H6.ovals(m, Hc[:, None])[0]) for m in (m200, m400)]
                if n != [args.ovals] * 2:
                    continue
                Hs, _, rob, _ = H6.shift(Hc)
                if int(H6.ovals(m400, Hs[:, None])[0]) == args.ovals:
                    found = (v, h, s, rob)
                    break
            if found:
                break
        if found is None:
            print(f"form {idx}: no rounding up to 2^{args.max_scale_log2} keeps {args.ovals} ovals", flush=True)
            continue
        v, h, s, rob = found
        wit = [w["point"] for w in witnesses(h, DEGH, args.witness_width)]
        print(f"form {idx}: scale {s}, height of f {max(map(abs, v))}, robustness {rob:.1e}, {len(wit)} witnesses",
              flush=True)
        if len(wit) < args.ovals + 1:  # some region has no witness: the lower bound cannot reach the target
            print(f"form {idx}: skipped (needs {args.ovals + 1} witnesses; try a larger --witness-width)", flush=True)
            continue
        tasks.append((len(tasks), v, h, args.ovals, wit, 0))
        meta[len(tasks) - 1] = idx
    if not tasks:
        return
    with mp.get_context("spawn").Pool(min(args.workers, len(tasks))) as pool:
        asyncs = [pool.apply_async(certify_task, (t,)) for t in tasks]
        for a, t in zip(asyncs, tasks):
            try:
                tid, out = a.get(timeout=args.timeout)
            except mp.TimeoutError:
                tid, out = t[0], {"certified": False, "reason": "timeout"}
            idx = meta[tid]
            print(f"form {idx}: certified {out.get('certified')}, bounds {out.get('lower_bound')} <= ovals <= "
                  f"{out.get('upper_bound')}, {out.get('reason', '')} ({out.get('seconds', 0):.0f} s)", flush=True)
            if out.get("certified"):
                (args.out / f"{args.file.stem}_{idx}.json").write_text(json.dumps(out["cert"]))
        pool.terminate()


if __name__ == "__main__":
    main()
