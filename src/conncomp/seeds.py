"""Degenerate seed forms and their multi-scale perturbations.

Curves with many ovals are classically obtained by small perturbations of very degenerate curves
(Harnack, Hilbert, Viro). For Hessian curves, the corresponding seeds are forms f whose Hessian is
degenerate:

  lines       f = l_1 ... l_d                (Ortiz-Rodriguez: (d-1)(d-2)/2 ovals)
  concurrent  products of lines with triple points (H is singular at a triple point of f)
  mixed       products of random conics, cubics and lines
  separable   f = phi(x) + psi(y), H = phi''(x) psi''(y): a grid of lines, nodes smoothed by eps^2 H(g)
  developable f = phi(l) for a linear form l (H = 0)

All seeds are randomly placed (a random projective change of coordinates is applied, except for
the affine families separable/developable) and normalized; `perturbed` adds eps * g with g random
and eps log-uniform.
"""

from functools import lru_cache

import torch

from conncomp import DEVICE, DTYPE
from conncomp.polynomials import monomials, normalize, sample


@lru_cache(maxsize=None)
def _mul_table(d1, d2):
    m1, m2, m3 = monomials(d1), monomials(d2), monomials(d1 + d2)
    pos = {m: k for k, m in enumerate(m3)}
    I, J, K = zip(*[(i, j, pos[(a[0] + b[0], a[1] + b[1], a[2] + b[2])])
                    for i, a in enumerate(m1) for j, b in enumerate(m2)])
    return tuple(torch.tensor(v, device=DEVICE) for v in (I, J, K)), len(m3)


def poly_mul(a, d1, b, d2):
    """Product of batches of forms: a (dim(d1), n), b (dim(d2), n) -> (dim(d1 + d2), n)."""
    (I, J, K), n = _mul_table(d1, d2)
    out = torch.zeros((n, a.shape[1]), dtype=DTYPE, device=a.device)
    out.index_add_(0, K, a[I] * b[J])
    return out


def product(factors):
    """Product of a list of (coefs, degree) batches."""
    c, d = factors[0]
    for c2, d2 in factors[1:]:
        c, d = poly_mul(c, d, c2, d2), d + d2
    return c, d


def univariate(coefs, var, deg):
    """Forms sum_k coefs[k] * var^k * z^(deg - k) for var in {"x", "y"}; coefs (deg + 1, n)."""
    out = torch.zeros((len(monomials(deg)), coefs.shape[1]), dtype=DTYPE, device=coefs.device)
    for k in range(deg + 1):
        m = (k, 0, deg - k) if var == "x" else (0, k, deg - k)
        out[monomials(deg).index(m)] = coefs[k]
    return out


def _lines_through(p, n):
    """Random lines through the points p (3, n): l = cross(p, q) for random q."""
    q = torch.randn((3, n), dtype=DTYPE, device=DEVICE)
    return torch.linalg.cross(p.T, q.T).T


def _as_linear(l):
    """Linear forms (3, n) as coefficient vectors over monomials(1) = (z, y, x)."""
    order = [m for m in monomials(1)]  # [(0,0,1), (0,1,0), (1,0,0)]
    idx = {(1, 0, 0): 0, (0, 1, 0): 1, (0, 0, 1): 2}
    return torch.stack([l[idx[m]] for m in order])


def seed_lines(deg, n):
    return product([(sample(1, n), 1) for _ in range(deg)])[0]


def seed_concurrent(deg, n):
    """d lines, in groups of 3 through common points (two triple points for d = 5, 6)."""
    factors, used = [], 0
    while deg - used >= 3:
        p = torch.randn((3, n), dtype=DTYPE, device=DEVICE)
        factors += [(_as_linear(_lines_through(p, n)), 1) for _ in range(3)]
        used += 3
    factors += [(sample(1, n), 1) for _ in range(deg - used)]
    return product(factors)[0]


def seed_mixed(deg, n):
    """Products of random lower-degree forms (random partition of deg into parts 1, 2, 3)."""
    parts = {5: [(3, 2), (2, 2, 1), (3, 1, 1), (2, 1, 1, 1)], 6: [(3, 3), (2, 2, 2), (3, 2, 1), (2, 2, 1, 1)]}
    options = parts.get(deg, [(2,) * (deg // 2) + (1,) * (deg % 2)])
    out = torch.empty((len(monomials(deg)), n), dtype=DTYPE, device=DEVICE)
    choice = torch.randint(len(options), (n,), device=DEVICE)
    for k, part in enumerate(options):
        idx = (choice == k).nonzero().squeeze(1)
        if idx.numel():
            out[:, idx] = product([(sample(d, idx.numel()), d) for d in part])[0]
    return out


def seed_separable(deg, n):
    """f = phi(x) + psi(y) with random univariate phi, psi; then H = phi''(x) psi''(y)."""
    phi = univariate(torch.randn((deg + 1, n), dtype=DTYPE, device=DEVICE), "x", deg)
    psi = univariate(torch.randn((deg + 1, n), dtype=DTYPE, device=DEVICE), "y", deg)
    return phi + psi


def seed_developable(deg, n):
    """f = phi(l) with l = a x + b y + c z random and phi random; then H = 0."""
    l = sample(1, n)
    phi = torch.randn((deg + 1, n), dtype=DTYPE, device=DEVICE)
    out = torch.zeros((len(monomials(deg)), n), dtype=DTYPE, device=DEVICE)
    z = torch.zeros((3, n), dtype=DTYPE, device=DEVICE)
    z[monomials(1).index((0, 0, 1))] = 1.0
    lp, zp = [torch.ones((1, n), dtype=DTYPE, device=DEVICE)], [torch.ones((1, n), dtype=DTYPE, device=DEVICE)]
    for k in range(1, deg + 1):
        lp.append(poly_mul(lp[-1], k - 1, l, 1))
        zp.append(poly_mul(zp[-1], k - 1, z, 1))
    for k in range(deg + 1):
        out += phi[k] * poly_mul(lp[k], k, zp[deg - k], deg - k)
    return out


FAMILIES = {
    "lines": seed_lines,
    "concurrent": seed_concurrent,
    "mixed": seed_mixed,
    "separable": seed_separable,
    "developable": seed_developable,
}


def seeds(family, deg, n):
    """Normalized seed forms of a family, shape (dim(deg), n)."""
    return normalize(FAMILIES[family](deg, n))


def perturbed(family, deg, n, eps_range=(1e-4, 1e-1)):
    """seed + eps * g with g uniform on the unit sphere and eps log-uniform in eps_range, normalized."""
    lo, hi = torch.log10(torch.tensor(eps_range, dtype=DTYPE))
    eps = 10 ** (lo + (hi - lo) * torch.rand(n, dtype=DTYPE, device=DEVICE))
    return normalize(seeds(family, deg, n) + eps * sample(deg, n))
