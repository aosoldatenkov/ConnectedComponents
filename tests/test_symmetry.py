import numpy as np
import pytest

from conncomp.hessian import hessian
from conncomp.polynomials import perturb, sample
from conncomp.rationalize import candidates
from conncomp.symmetry import Symmetry, _action_matrix, is_semi_invariant, symmetry

SPECS = ["x", "x:-1", "y", "xy", "xy:-1,1", "diag", "diag:-1", "C2", "C3", "C4:-1", "D3", "D3:1,-1", "D4", "D5",
         "C6:-1"]


def test_dimensions_degree_5():
    # 18 monomials of degree 5 affect the Hessian; x -> -x splits them by the parity of the x-degree
    assert symmetry("x", 5).dim == 10 and symmetry("x:-1", 5).dim == 8
    assert symmetry("xy", 5).dim == 5
    assert symmetry("D3", 5).dim == 4 and symmetry("D3:1,-1", 5).dim == 2
    assert symmetry("D2", 5).dim == symmetry("xy", 5).dim  # the same (Klein) group


@pytest.mark.parametrize("spec", SPECS)
def test_basis_is_semi_invariant(spec):
    sym = symmetry(spec, 5)
    assert sym.basis.dtype == np.int64
    assert all(is_semi_invariant(col, 5, sym) for col in sym.basis.T)
    assert np.linalg.matrix_rank(sym.basis.astype(float)) == sym.dim


@pytest.mark.parametrize("spec", ["x:-1", "diag", "D3", "D3:1,-1", "C4:-1"])
def test_hessian_is_invariant(spec):
    sym = symmetry(spec, 5)
    f = sample(5, 3, basis=sym.basis)
    H = hessian(5, f).cpu().numpy()
    hsym = Symmetry(sym.name, sym.generators, [1] * len(sym.generators), sym.basis, 6)  # chi = 1 on H
    for col in H.T:
        assert is_semi_invariant(col, 6, hsym, tol=1e-8)


def test_perturb_stays_in_span():
    sym = symmetry("D3", 5)
    c = sample(5, 1, basis=sym.basis)[:, 0]
    for col in perturb(c, 4, 0.1, basis=sym.basis).cpu().numpy().T:
        assert is_semi_invariant(col, 5, sym)


def test_symmetric_rational_candidates():
    sym = symmetry("D3", 5)
    c = sample(5, 1, basis=sym.basis)[:, 0].cpu().numpy()
    for ints, _ in candidates(c, "round", max_scale=30, basis=sym.basis):
        assert is_semi_invariant(ints, 5, sym)


def test_invalid_specs():
    with pytest.raises(ValueError):
        symmetry("Q8", 5)
    with pytest.raises(ValueError):
        symmetry("x:1,1", 5)
    with pytest.raises(ValueError):
        symmetry("D3:1,2", 5)


def test_action_matrix_is_a_representation():
    g = symmetry("D3", 4).generators[0]
    A = np.array(_action_matrix(g, 4).evalf(), dtype=float)
    assert np.allclose(np.linalg.matrix_power(A, 3), np.eye(A.shape[0]))


def test_degeneracy_warnings():
    assert symmetry("xy:-1,-1", 5).effective_degree == 4  # f = x y g(x^2, y^2)
    assert symmetry("C4:-1", 5).effective_degree == 4
    assert symmetry("D3:1,-1", 5).singular_at_origin  # f = y (3 x^2 - y^2) (...)
    assert not symmetry("x", 5).warnings() and not symmetry("D3", 5).warnings()


def test_bases_for_plain_curves():
    # for the curve f = 0 itself the terms of x, y-degree <= 1 are kept
    assert symmetry("xy", 6, for_hessian=False).dim == 10  # f = F(x^2, y^2, z^2)
    assert symmetry("x", 6, for_hessian=False).dim == 16 and symmetry("x", 6).dim == 16 - 2
    s = symmetry("D3", 6, for_hessian=False)
    assert all(is_semi_invariant(col, 6, s) for col in s.basis.T)
