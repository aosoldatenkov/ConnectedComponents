"""Randomized scans over the space of ternary forms, recording component counts."""

from collections import defaultdict

import numpy as np
import torch

from conncomp.components import CountPipeline, count_components
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

    def signs(self, coefs, noise=0.0, chunk=None):
        """uint8 sign classes (1 where the value is >= 0) on the grid, shape (N, width, width).

        Evaluated in chunks of `chunk` forms (default: about 8e7 grid values per chunk), so the float64
        values of the whole batch never need to be in memory at once.
        """
        chunk = chunk or max(16, int(8e7 // (self.width * self.width)))
        cc = self.curve_coefs(coefs, noise)
        out = torch.empty((cc.shape[1], self.width, self.width), dtype=torch.uint8, device=cc.device)
        for s in range(0, cc.shape[1], chunk):
            out[s : s + chunk] = evaluate(self.basis, cc[:, s : s + chunk]) >= 0
        return out

    def nesting(self, coefs, min_size=3, noise=0.0):
        """Region trees (oval counts and nesting types, conncomp.nesting) for each form in the batch."""
        from conncomp.nesting import nesting_trees

        return nesting_trees(self.signs(coefs, noise).cpu().numpy(), self.pat, min_size)

    def count(self, coefs, min_size, noise=0.0):
        """Number of sign components (with at least `min_size` pixels) for each form in the batch."""
        return count_components(self.signs(coefs, noise).cpu().numpy(), self.pat, min_size)


def _scan(exp, draw, nsamples, niter, lo, filtr, noise=0.0):
    comp_counts = defaultdict(int)
    save_coefs = defaultdict(list)
    pipe = CountPipeline(exp.pat, filtr)

    def process(counts, coefs, it):
        cpu_coefs = coefs.cpu().numpy()
        for j, l in enumerate(counts):
            comp_counts[l] += 1
            if l >= lo:
                save_coefs[l].append(cpu_coefs[:, j])
        print(it, " | " + " ".join(f"{k}: {comp_counts[k]};" for k in sorted(comp_counts)))

    for it in range(niter):
        coefs = draw()
        pipe.submit(exp.signs(coefs, noise), (coefs, it))
        for counts, (c, i) in pipe.results():
            process(counts, c, i)
    for counts, (c, i) in pipe.results(0):
        process(counts, c, i)
    pipe.close()
    return save_coefs


def batch_scan(deg, width, nsamples, niter, lo, filtr=6, use_hessian=True, noise=0.0, basis=None):
    """Sample uniformly random forms; return {component count: [coefficient vectors]} for counts >= lo.

    With `noise` > 0 (Hessian mode only), the Hessian coefficients are perturbed before counting.
    With `basis` (e.g. conncomp.symmetry.symmetry("D3", deg).basis), only forms in its span are sampled.
    """
    exp = Experiment(deg, width, use_hessian)
    return _scan(exp, lambda: sample(deg, nsamples, basis=basis), nsamples, niter, lo, filtr, noise)


def center_scan(deg, width, center, r, nsamples, niter, lo, filtr=6, use_hessian=True, basis=None):
    """Like `batch_scan`, but sample Gaussian perturbations of radius `r` around `center`."""
    exp = Experiment(deg, width, use_hessian)
    return _scan(exp, lambda: perturb(center, nsamples, r, basis=basis), nsamples, niter, lo, filtr)


def adaptive_scan(deg, width, batch, niter, filtr=3, use_hessian=True, basis=None, r=0.1, keep=100, seed=None,
                  check_width=None, screen=None, slack=2, cap=50000, sampler=None):
    """Non-interactive version of the search loop of conncomp.search.

    Odd iterations perturb (radius r, within the span of `basis`) a randomly chosen form among the
    best ones found so far; even iterations sample new random forms, from `sampler(n)` if given
    (e.g. degenerate seeds, conncomp.seeds.perturbed). r may be a pair (lo, hi): then each batch
    uses a log-uniform radius in that range. Returns (best, histogram):
    best is a list of up to `keep` pairs (ovals, coefficient vector), sorted by decreasing oval
    count; histogram counts the oval numbers of all forms evaluated. A form only enters `best` if
    its count is reproduced at `check_width` (default 2 * width), which filters out pixel noise of
    nearly degenerate curves.

    With `screen` (conncomp.euler.EulerScreen), each batch is screened on the GPU and only the forms
    with estimate >= (best count so far - slack), at most `cap`, are counted exactly; the histogram
    then only covers those.
    """
    from collections import Counter
    import random

    rng = random.Random(seed)
    exp = Experiment(deg, width, use_hessian)
    check = Experiment(deg, check_width or 2 * width, use_hessian)
    best, hist = [], Counter()
    pipe = CountPipeline(exp.pat, filtr)

    def process(counts, coefs):
        nonlocal best
        ovals = [n - 1 for n in counts]
        hist.update(ovals)
        cpu = coefs.cpu().numpy()
        top = sorted(range(len(ovals)), key=lambda j: -ovals[j])[:keep]
        recount = [n - 1 for n in check.count(coefs[:, top], filtr)]
        new = [(ovals[j], cpu[:, j]) for j, o2 in zip(top, recount) if o2 == ovals[j]]
        best = sorted(best + new, key=lambda p: -p[0])[:keep]

    # the batch generated in iteration it uses the pool as known after iteration it - 2 (pipelining)
    for it in range(niter):
        if best and it % 2:
            radius = r if not isinstance(r, tuple) else 10 ** rng.uniform(*np.log10(r))
            coefs = perturb(rng.choice(best)[1], batch, radius, basis=basis)
        else:
            coefs = sampler(batch) if sampler is not None else sample(deg, batch, basis=basis)
        if screen is not None:
            from conncomp.euler import select_candidates

            top_count = best[0][0] if best else 0
            coefs = coefs[:, select_candidates(screen.estimate(coefs), top_count - slack, cap)]
        if coefs.shape[1]:
            pipe.submit(exp.signs(coefs), coefs)
        for counts, c in pipe.results():
            process(counts, c)
    for counts, c in pipe.results(0):
        process(counts, c)
    pipe.close()
    return best, hist
