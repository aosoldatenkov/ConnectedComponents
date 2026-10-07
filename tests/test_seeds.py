import sympy as sp
import torch

from conncomp.hessian import hessian
from conncomp.polynomials import monomials, to_sympy
from conncomp.seeds import FAMILIES, perturbed, poly_mul, seeds

x, y, z = sp.symbols("x y z")


def test_shapes_and_normalization():
    for fam in FAMILIES:
        s = seeds(fam, 5, 7)
        assert s.shape == (len(monomials(5)), 7)
        assert torch.allclose(s.norm(dim=0), torch.ones(7, dtype=s.dtype, device=s.device))
    assert perturbed("lines", 5, 4).shape == (21, 4)


def test_poly_mul_matches_sympy():
    a, b = torch.randn(6, 1, dtype=torch.float64), torch.randn(3, 1, dtype=torch.float64)
    a, b = a.to(seeds("lines", 2, 1).device), b.to(seeds("lines", 2, 1).device)
    prod = poly_mul(a, 2, b, 1)[:, 0].cpu().numpy()
    expected = sp.Poly(sp.expand(to_sympy(a[:, 0].cpu().numpy(), 2) * to_sympy(b[:, 0].cpu().numpy(), 1)), x, y, z)
    for c, m in zip(prod, monomials(3)):
        assert abs(c - float(expected.coeff_monomial(x**m[0] * y**m[1] * z**m[2]))) < 1e-12


def test_degenerate_hessians():
    f = seeds("separable", 5, 1)
    fx = to_sympy(f[:, 0].cpu().numpy(), 5, x, y, 1)
    Hs = sp.expand(sp.diff(fx, x, 2) * sp.diff(fx, y, 2) - sp.diff(fx, x, y) ** 2)
    assert sp.expand(Hs - sp.diff(fx, x, 2) * sp.diff(fx, y, 2)) == 0 or abs(
        float(sp.Poly(sp.expand(Hs - sp.diff(fx, x, 2) * sp.diff(fx, y, 2)), x, y).max_norm())) < 1e-10
    d = seeds("developable", 5, 3)
    assert hessian(5, d).abs().max() < 1e-10  # H = 0 for f = phi(l)
