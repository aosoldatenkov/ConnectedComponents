import numpy as np
import pytest
import sympy as sp
import torch

from conncomp import DEVICE, DTYPE, _connected
from conncomp.components import count_components
from conncomp.grid import neighbour_pattern, sphere_points
from conncomp.hessian import hessian
from conncomp.polynomials import deg_to_dim, evaluate, monomial_basis, monomials, sample, to_sympy

WIDTH = 101


@pytest.fixture(scope="module")
def grid():
    return sphere_points(WIDTH), neighbour_pattern(WIDTH)


def coefs_of(expr, deg):
    """Coefficient vector (as a (D, 1) tensor) of a sympy form in x, y, z."""
    x, y, z = sp.symbols("x y z")
    d = sp.Poly(expr, x, y, z).as_dict()
    c = [float(d.get(m, 0)) for m in monomials(deg)]
    return torch.tensor(c, dtype=DTYPE, device=DEVICE).reshape(-1, 1)


def count(expr, deg, grid):
    pts, pat = grid
    vals = evaluate(monomial_basis(monomials(deg), pts), coefs_of(expr, deg))
    return count_components(vals, pat, min_size=3)[0]


def test_sphere_points_on_unit_sphere(grid):
    pts, _ = grid
    assert torch.allclose(pts.norm(dim=0), torch.ones(WIDTH, WIDTH, dtype=DTYPE, device=pts.device))


def test_definite_form_has_one_region(grid):
    x, y, z = sp.symbols("x y z")
    assert count(x**2 + y**2 + z**2, 2, grid) == 1


@pytest.mark.parametrize("r2", [sp.Rational(1, 4), 1, 4])
def test_one_oval(grid, r2):
    # x^2 + y^2 = r^2 z^2 is one oval in RP^2 (seen near the centre or across the antipodal gluing)
    x, y, z = sp.symbols("x y z")
    assert count(x**2 + y**2 - r2 * z**2, 2, grid) == 2


def test_nested_ovals(grid):
    x, y, z = sp.symbols("x y z")
    f = (x**2 + y**2 - z**2 / 4) * (x**2 + y**2 - 4 * z**2)
    assert count(f, 4, grid) == 3


def test_disjoint_ovals(grid):
    # Two disjoint (non-nested) ellipses: 3 regions
    x, y, z = sp.symbols("x y z")
    f = ((x - z / 2) ** 2 + 4 * y**2 - z**2 / 16) * ((x + z / 2) ** 2 + 4 * y**2 - z**2 / 16)
    assert count(f, 4, grid) == 3


def test_single_and_batch_agree(grid):
    _, pat = grid
    rng = np.random.default_rng(0)
    vals = rng.standard_normal((4, WIDTH, WIDTH))
    batch = _connected.components_batch(vals, pat, 2)
    for i in range(4):
        assert sorted(_connected.components(vals[i], pat)) == sorted(batch[i])
        assert sum(batch[i]) == WIDTH * WIDTH


def test_empty_batch(grid):
    _, pat = grid
    assert _connected.components_batch(np.zeros((0, WIDTH, WIDTH)), pat) == []


@pytest.mark.parametrize("deg", [3, 4, 5])
def test_hessian_matches_sympy(deg):
    x, y, z = sp.symbols("x y z")
    c = sample(deg, 1)
    h = hessian(deg, c)
    assert h.shape == (deg_to_dim(2 * deg - 4), 1)
    f = sp.Poly(to_sympy(c[:, 0].cpu().numpy(), deg, x, y, z), x, y, z)
    H = (f.diff((0, 2)) * f.diff((1, 2)) - f.diff((0, 1), (1, 1)) ** 2).as_dict()
    expected = torch.tensor([float(H.get(m, 0)) for m in monomials(2 * deg - 4)], dtype=DTYPE)
    assert torch.allclose(h[:, 0].cpu(), expected, atol=1e-12)
