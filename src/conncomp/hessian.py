"""The Hessian form H(f) = f_xx f_yy - f_xy^2 of a ternary form f of degree d.

H(f) is a form of degree 2d - 4 whose coefficients are quadratic polynomials
in the coefficients of f. The map is derived symbolically once per degree and
then applied to batches of coefficient vectors.
"""

from functools import lru_cache

import sympy as sp
import torch

from conncomp.polynomials import monomials


@lru_cache(maxsize=None)
def hessian_map(deg):
    """{hessian monomial exponents: {coefficient exponents: integer factor}}.

    The coefficient exponents are indexed in the order of `monomials(deg)`.
    """
    x, y, z = sp.symbols("x y z")
    coefs = [sp.Symbol(f"a_{i}_{j}_{k}") for i, j, k in monomials(deg)]
    f = sp.Poly(sum(a * x**i * y**j * z**k for a, (i, j, k) in zip(coefs, monomials(deg))), x, y, z)
    Hf = (f.diff((x, 2)) * f.diff((y, 2)) - f.diff((x, 1), (y, 1)) ** 2).as_dict()
    return {m: sp.Poly(Hf[m], *coefs).as_dict() for m in Hf}


def hessian(deg, coefs, out=None):
    """Coefficients of H(f) for a batch of forms f.

    coefs: (deg_to_dim(deg), N). Returns (deg_to_dim(2*deg - 4), N), written into `out` if given.
    """
    hmap = hessian_map(deg)
    hmons = monomials(2 * deg - 4)
    if out is None:
        out = torch.zeros((len(hmons),) + tuple(coefs.shape[1:]), dtype=coefs.dtype, device=coefs.device)
    out.fill_(0.0)
    mon = torch.ones(tuple(coefs.shape[1:]), dtype=coefs.dtype, device=coefs.device)
    for m, terms in hmap.items():
        row = out[hmons.index(m)]
        for n, factor in terms.items():
            mon.fill_(1.0)
            for i, e in enumerate(n):
                if e > 0:
                    mon *= coefs[i].pow(e)
            row += float(factor) * mon
    return out
