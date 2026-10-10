#!/usr/bin/env python3
"""Build start sets for the strategy comparison: degree-5 forms whose Hessian curves have exactly 8 (or 9) ovals
(stable at widths 200 and 400), plus random forms; print degeneracy statistics (walls within distance tau)."""

import numpy as np
import torch

from conncomp.critical import critical_points, degeneracy, wall_distances
from conncomp.euler import EulerScreen, select_candidates
from conncomp.hessian import hessian
from conncomp.io import load_coefs, save_coefs
from conncomp.polynomials import sample
from conncomp.scan import Experiment

DEG = 5


def stable_counts(f, widths=(200, 400)):
    counts = [torch.tensor(Experiment(DEG, w, True).count(f, 3)) - 1 for w in widths]
    return torch.where(counts[0] == counts[1], counts[0], torch.tensor(-1))


def stats(name, f):
    crit, _ = critical_points(hessian(DEG, f), 2 * DEG - 4)
    d, _ = wall_distances(crit, f, DEG)
    out = []
    for tau in (3e-3, 1e-2, 3e-2):
        close, good, soft = degeneracy(crit, d, f.shape[1], tau)
        out.append(f"tau={tau:.0e}: walls {close.float().mean():.2f} (good {good.float().mean():.2f})")
    dmin = torch.full((f.shape[1],), float("inf"), dtype=d.dtype, device=d.device).scatter_reduce(
        0, crit.form, d, reduce="amin")
    print(f"{name:14s} ({f.shape[1]} forms): mean number of walls within tau: {'; '.join(out)}; "
          f"median nearest wall {torch.median(dmin).item():.1e}")


def main():
    torch.manual_seed(0)
    screen = EulerScreen(DEG, N=40)
    eights, nines = [], []
    for _ in range(10):
        c = sample(DEG, 1_000_000)
        c = c[:, select_candidates(screen.estimate(c), 7, 100000)]
        n = stable_counts(c)
        eights.append(c[:, n == 8].cpu())
        nines.append(c[:, n == 9].cpu())
    e8, e9 = torch.cat(eights, 1), torch.cat(nines, 1)
    save_coefs("data/strategies/start_8.txt", e8.T.numpy())
    known9 = np.array(load_coefs("data/degenerate/cache_H_deg5_mixed.txt")[:100]).T
    print(f"from 10M random forms: {e8.shape[1]} with 8 ovals, {e9.shape[1]} with 9 (stable at 200/400)")
    stats("random", sample(DEG, 3000))
    stats("8 ovals", e8[:, :3000].cuda())
    if e9.shape[1]:
        stats("9 ovals (rnd)", e9.cuda())
    stats("9 ovals (seeds)", torch.as_tensor(known9, device="cuda"))


if __name__ == "__main__":
    main()
