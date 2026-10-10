import numpy as np
import pytest
import sympy as sp
import torch

from conncomp import DEVICE, DTYPE
from conncomp.hessian import hessian
from conncomp.nesting import SphereMesh
from conncomp.polynomials import from_sympy, sample
from conncomp.scan import Experiment

x, y = sp.symbols("x y")
R = sp.Rational
CASES = [
    ("empty", x**2 + y**2 + 1, 2, "0"),
    ("circle", x**2 + y**2 - 1, 2, "1"),
    ("two disjoint", (x**2 + y**2 - R(1, 4)) * ((x - 2) ** 2 + y**2 - R(1, 4)), 4, "2"),
    ("nested pair", (x**2 + y**2 - R(1, 4)) * (x**2 + y**2 - 1), 4, "1<1>"),
    ("nest + separate", (x**2 + y**2 - R(1, 4)) * (x**2 + y**2 - 1) * ((x - 3) ** 2 + y**2 - R(1, 4)), 6, "1 u 1<1>"),
    ("three nested", (x**2 + y**2 - R(1, 16)) * (x**2 + y**2 - R(1, 4)) * (x**2 + y**2 - 1), 6, "1<1<1>>"),
    ("big around two",
     (x**2 + y**2 - 4) * ((x - R(1, 2)) ** 2 + y**2 - R(1, 9)) * ((x + R(1, 2)) ** 2 + y**2 - R(1, 9)), 6, "1<2>"),
]


@pytest.fixture(scope="module")
def mesh():
    return SphereMesh(60)


def coefs(expr, deg):
    return torch.tensor([float(v) for v in from_sympy(sp.expand(expr), deg)], dtype=DTYPE, device=DEVICE)[:, None]


@pytest.mark.parametrize("name,expr,deg,expected", CASES, ids=[c[0] for c in CASES])
def test_known_types(mesh, name, expr, deg, expected):
    c = coefs(expr, deg)
    t_grid = Experiment(deg, 300, False).nesting(c)[0]
    t_mesh = mesh.trees(c, deg)[0]
    assert t_grid.ok and t_mesh.ok
    assert t_grid.type == expected and t_mesh.type == expected
    assert t_mesh.n_ovals == len(t_mesh.size) - 1
    assert t_mesh.outside_sign == (1 if name != "empty" else 1)


def test_mesh_tables():
    m = SphereMesh(8)
    assert m.V == 24 * 64 + 2
    assert np.all(m.antipode[m.antipode] == np.arange(m.V))
    deg4 = (m.nbr4 >= 0).sum(1)
    assert set(deg4.tolist()) <= {3, 4}  # cube corners have valence 3


def test_mesh_trees_consistent_on_random_hessians(mesh):
    torch.manual_seed(0)
    f = sample(5, 2000)
    trees = mesh.trees(hessian(5, f), 6)
    assert all(t.ok for t in trees)
