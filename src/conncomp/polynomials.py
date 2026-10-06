"""Ternary forms: monomial bookkeeping, random sampling and batched evaluation.

A batch of N forms of degree d is stored as a coefficient tensor of shape
(deg_to_dim(d), N); row i is the coefficient of monomials(d)[i].
"""

import numpy as np
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


def _basis_tensor(basis, device=DEVICE):
    """Float tensor (D, k) of the basis columns, each scaled to unit norm."""
    B = torch.as_tensor(np.asarray(basis, dtype=np.float64), dtype=DTYPE, device=device)
    return B / B.norm(dim=0, keepdim=True)


def sample(deg, n, device=DEVICE, basis=None):
    """N forms of degree `deg` drawn uniformly from the unit sphere in coefficient space.

    With `basis` (integer matrix (D, k), e.g. conncomp.symmetry.Symmetry.basis), the forms are
    random combinations B a of the (normalized) basis columns instead.
    """
    if basis is None:
        return normalize(torch.randn((deg_to_dim(deg), n), dtype=DTYPE, device=device))
    B = _basis_tensor(basis, device)
    return normalize(B @ torch.randn((B.shape[1], n), dtype=DTYPE, device=device))


def perturb(center, n, r, basis=None):
    """N normalized Gaussian perturbations of radius `r` around the coefficient vector `center`.

    With `basis`, the perturbations stay in the span of the basis (`center` should lie in it).
    """
    c = torch.as_tensor(center, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
    if basis is None:
        return normalize(torch.randn((c.shape[0], n), dtype=DTYPE, device=DEVICE) * r + c)
    B = _basis_tensor(basis)
    a = torch.linalg.pinv(B) @ c
    return normalize(B @ (a + r * torch.randn((B.shape[1], n), dtype=DTYPE, device=DEVICE)))


def monomial_basis(mons, pts):
    """Values of the monomials `mons` at `pts` (shape (3, *grid)); returns shape (len(mons), *grid)."""
    return torch.stack([pts[0].pow(i) * pts[1].pow(j) * pts[2].pow(k) for i, j, k in mons])


def evaluate(basis, coefs):
    """Evaluate a batch of forms on a grid.

    basis: (D, *grid) from `monomial_basis`; coefs: (D, N). Returns (N, *grid).
    """
    return torch.tensordot(coefs.T, basis, dims=1)


def to_sympy(coefs, deg, x=None, y=None, z=None):
    """The form with coefficient vector `coefs` as a sympy expression (exact for int/Fraction coefficients).

    Pass z=1 to get the affine polynomial f(x, y, 1).
    """
    if x is None:
        x, y, z = sp.symbols("x y z")
    conv = float if isinstance(coefs[0], float) else sp.sympify
    return sum(conv(c) * x**i * y**j * z**k for c, (i, j, k) in zip(coefs, monomials(deg)))


def from_sympy(expr, deg, x=None, y=None, z=None):
    """Coefficient vector (exact sympy numbers) of a polynomial in x, y (affine) or x, y, z (homogeneous).

    An affine polynomial f(x, y) of degree <= deg is homogenized with z.
    """
    if x is None:
        x, y, z = sp.symbols("x y z")
    d = sp.Poly(sp.expand(expr), x, y, z).as_dict()
    coefs = {}
    for (i, j, k), c in d.items():
        if k == 0 and i + j < deg:  # affine term: homogenize
            k = deg - i - j
        if i + j + k != deg:
            raise ValueError(f"term x^{i} y^{j} z^{k} does not fit degree {deg}")
        coefs[(i, j, k)] = c
    return [coefs.get(m, sp.Integer(0)) for m in monomials(deg)]
