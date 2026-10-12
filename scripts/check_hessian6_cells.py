#!/usr/bin/env python3
"""Re-check the high cells of a degree-8 births archive (sextic_births.py --kind hessian6): oval counts at N = 200 and
400 and the exact shift test (hessian6_scan.shift) for every form with at least --min ovals; writes the confirmed forms
(coefficients of f, one per line) to <archive dir>/confirmed_<n>.txt.

    uv run python scripts/check_hessian6_cells.py pools/hessian6_births/archive.npz --min 15
"""

import argparse
import collections
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import hessian6_scan as H6  # noqa: E402
import sextic_types as S  # noqa: E402
from conncomp import DEVICE  # noqa: E402
from conncomp.gpu_trees import FaceMesh  # noqa: E402
from conncomp.io import save_coefs  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", type=Path)
    ap.add_argument("--min", type=int, default=15)
    args = ap.parse_args()
    _, g = S.load_npz(args.archive)
    m200, m400 = FaceMesh(200), FaceMesh(400)
    for k in sorted(g["cell"], reverse=True):
        if k < args.min or k == S.OTHER:
            continue
        F = g["cell"][k][1]
        C = H6.curves(torch.as_tensor(F.T, device=DEVICE))
        n200 = H6.ovals(m200, C).cpu().numpy()
        n400 = H6.ovals(m400, C).cpu().numpy()
        shifted, rob = [], []
        for i in range(len(F)):
            Hs, _, r, _ = H6.shift(C[:, i])
            shifted.append(int(H6.ovals(m400, Hs[:, None])[0]))
            rob.append(r)
        shifted = np.array(shifted)
        cnt = [dict(collections.Counter(x.tolist())) for x in (n200, n400, shifted)]
        print(f"cell {k}: {len(F)} forms; N=200 {cnt[0]}, N=400 {cnt[1]}, shift test {cnt[2]}", flush=True)
        for n in sorted(set(shifted.tolist())):
            if n >= args.min:
                sel = np.nonzero(shifted == n)[0]
                sel = sel[np.argsort(-np.array(rob)[sel])]
                save_coefs(args.archive.parent / f"confirmed_{n}.txt", F[sel])
                print(f"   confirmed {n}: {len(sel)} forms (robustness up to {max(np.array(rob)[sel]):.1e})")


if __name__ == "__main__":
    main()
