"""Randomized scans over the space of ternary forms, recording component counts."""

from collections import defaultdict

import torch

from conncomp.components import count_components
from conncomp.grid import neighbour_pattern, sphere_points
from conncomp.hessian import hessian
from conncomp.polynomials import evaluate, monomial_basis, monomials, normalize, perturb, sample


class Experiment:
    """Precomputed grid data for evaluating forms of degree `deg` (or their Hessians) on RP^2."""

    def __init__(self, deg, width, use_hessian=True):
        self.deg = deg
        self.width = width
        self.use_hessian = use_hessian
        self.curve_deg = 2 * deg - 4 if use_hessian else deg
        self.basis = monomial_basis(monomials(self.curve_deg), sphere_points(width))
        self.pat = neighbour_pattern(width)

    def curve_coefs(self, coefs, noise=0.0):
        """Coefficients of the curve being studied: f itself or H(f), optionally perturbed by `noise`."""
        if not self.use_hessian:
            return coefs
        h = hessian(self.deg, coefs)
        if noise > 0:
            h.add_(torch.randn_like(h) * noise)
            normalize(h)
        return h

    def values(self, coefs, noise=0.0):
        """Values of the curve's equation on the grid; shape (N, width, width)."""
        return evaluate(self.basis, self.curve_coefs(coefs, noise))

    def count(self, coefs, min_size, noise=0.0):
        """Number of sign components (with at least `min_size` pixels) for each form in the batch."""
        return count_components(self.values(coefs, noise), self.pat, min_size)


def _scan(exp, draw, nsamples, niter, lo, filtr, noise=0.0):
    comp_counts = defaultdict(int)
    save_coefs = defaultdict(list)
    for it in range(niter):
        coefs = draw()
        counts = exp.count(coefs, filtr, noise)
        cpu_coefs = coefs.cpu().numpy()
        for j, l in enumerate(counts):
            comp_counts[l] += 1
            if l >= lo:
                save_coefs[l].append(cpu_coefs[:, j])
        print(it, " | " + " ".join(f"{k}: {comp_counts[k]};" for k in sorted(comp_counts)))
    return save_coefs


def batch_scan(deg, width, nsamples, niter, lo, filtr=6, use_hessian=True, noise=0.0):
    """Sample uniformly random forms; return {component count: [coefficient vectors]} for counts >= lo.

    With `noise` > 0 (Hessian mode only), the Hessian coefficients are perturbed before counting.
    """
    exp = Experiment(deg, width, use_hessian)
    return _scan(exp, lambda: sample(deg, nsamples), nsamples, niter, lo, filtr, noise)


def center_scan(deg, width, center, r, nsamples, niter, lo, filtr=6, use_hessian=True):
    """Like `batch_scan`, but sample Gaussian perturbations of radius `r` around `center`."""
    exp = Experiment(deg, width, use_hessian)
    return _scan(exp, lambda: perturb(center, nsamples, r), nsamples, niter, lo, filtr)
