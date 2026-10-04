"""Ternary forms: monomial bookkeeping, random sampling and batched evaluation.

A batch of N forms of degree d is stored as a coefficient tensor of shape
(deg_to_dim(d), N); row i is the coefficient of monomials(d)[i].
"""

import sympy as sp
import torch

from conncomp import DEVICE, DTYPE


def deg_to_dim(deg):
    """Dimension of the space of ternary forms of degree `deg`."""
    return (deg + 1) * (deg + 2) // 2


def monomials(deg):
    """Exponent triples (i, j, k) of x^i y^j z^k, i + j + k = deg, in the canonical order."""
    return [(i, j, deg - i - j) for i in range(deg + 1) for j in range(deg - i + 1)]


def normalize(coefs):
    """Normalize each column (one form per column) to unit L2 norm, in place."""
    return coefs.div_(coefs.norm(p=2, dim=0))


def sample(deg, n, device=DEVICE):
    """N forms of degree `deg` drawn uniformly from the unit sphere in coefficient space."""
    return normalize(torch.randn((deg_to_dim(deg), n), dtype=DTYPE, device=device))


def perturb(center, n, r):
    """N normalized Gaussian perturbations of radius `r` around the coefficient vector `center`."""
    c = torch.as_tensor(center, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
    return normalize(torch.randn((c.shape[0], n), dtype=DTYPE, device=DEVICE) * r + c)


def monomial_basis(mons, pts):
    """Values of the monomials `mons` at `pts` (shape (3, *grid)); returns shape (len(mons), *grid)."""
    return torch.stack([pts[0].pow(i) * pts[1].pow(j) * pts[2].pow(k) for i, j, k in mons])


def evaluate(basis, coefs):
    """Evaluate a batch of forms on a grid.

    basis: (D, *grid) from `monomial_basis`; coefs: (D, N). Returns (N, *grid).
    """
    return torch.tensordot(coefs.T, basis, dims=1)


def to_sympy(coefs, deg, x=None, y=None, z=None):
    """The form with coefficient vector `coefs` as a sympy expression."""
    if x is None:
        x, y, z = sp.symbols("x y z")
    return sum(float(c) * x**i * y**j * z**k for c, (i, j, k) in zip(coefs, monomials(deg)))
