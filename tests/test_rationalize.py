import math
from fractions import Fraction

import numpy as np
import pytest
import sympy as sp
import torch

from conncomp import DEVICE, DTYPE, _connected
from conncomp.components import component_labels
from conncomp.grid import neighbour_pattern, sphere_points
from conncomp.hessian import hessian
from conncomp.polynomials import from_sympy, monomials
from conncomp.rationalize import (
    candidates,
    cf_approx,
    convergents,
    eval_form,
    exact_hessian,
    grid_point,
    oval_counts,
    rationalize,
    witnesses,
)

x, y = sp.symbols("x y")

# Examples from Ortiz-Rodriguez & Sottile, "Real Hessian curves"
PAPER = {
    "quartic": (
        4,
        4,
        -2 * y**2 + 2 * x * y + 12 * x**2 + 10 * y**3 + 3 * x * y**2 - 10 * x**2 * y - 13 * x**3
        - 11 * y**4 + 6 * x * y**3 + 9 * x**2 * y**2 - 2 * x**3 * y - x**4,
    ),
    "quintic": (
        5,
        8,
        4 * y**2 + x * y - 6 * x**2 - 25 * y**3 + 24 * x * y**2 + 15 * x**2 * y - 33 * x**3 + y**4
        - 3 * x * y**3 + 15 * x**2 * y**2 - 19 * x**3 * y - 26 * x**4 + 33 * y**5 - 2 * x * y**4
        - 23 * x**2 * y**3 - 30 * x**3 * y**2 - 26 * x**4 * y + 31 * x**5,
    ),
}


def paper_form(name):
    deg, ovals, f = PAPER[name]
    return deg, ovals, [int(v) for v in from_sympy(f, deg)]


def test_convergents_of_pi():
    cs = list(convergents(math.pi))
    assert cs[:4] == [Fraction(3), Fraction(22, 7), Fraction(333, 106), Fraction(355, 113)]
    assert cs[-1] == Fraction(math.pi)
    assert cf_approx(math.pi, 1e-6) == Fraction(355, 113)


def test_candidates_are_primitive_and_proportional():
    c = np.array([0.5, -0.25, 0.125])
    for mode in ("cf", "round"):
        cands = [ints for ints, _ in candidates(c, mode, max_scale=10)]
        assert [4, -2, 1] in cands
        assert all(math.gcd(*ints) == 1 for ints in cands)


def test_exact_hessian_matches_float():
    deg, _, f = paper_form("quintic")
    h = exact_hessian(deg, f)
    hf = hessian(deg, torch.tensor(f, dtype=DTYPE, device=DEVICE).reshape(-1, 1))[:, 0].cpu().numpy()
    ratio = hf[np.nonzero(h)] / np.array(h, dtype=float)[np.nonzero(h)]
    assert np.allclose(ratio, ratio[0])


def test_paper_quartic_hessian_value():
    # The paper's h is Hess(f) / -4, with h(-2, 0) = -7068
    deg, _, f = paper_form("quartic")
    assert eval_form(exact_hessian(deg, f), 4, (-2, 0, 1)) * -4 == -7068


@pytest.mark.parametrize("name", ["quartic", "quintic"])
def test_paper_oval_counts(name):
    deg, ovals, f = paper_form(name)
    assert oval_counts([exact_hessian(deg, f)], 2 * deg - 4, 300)[0] == ovals


def test_component_labels_consistent_with_sizes():
    rng = np.random.default_rng(1)
    W = 51
    pat = neighbour_pattern(W)
    vals = rng.standard_normal((W, W))
    labels, sizes = component_labels(vals, pat)
    assert sizes == _connected.components(vals, pat)
    assert np.bincount(labels.ravel()).tolist() == sizes


def test_grid_point_is_proportional_to_sphere_point():
    W = 37
    pts = sphere_points(W, device="cpu").numpy()
    for i, j in [(0, 0), (5, 30), (18, 18), (36, 1)]:
        p = np.array(grid_point(i, j, W), dtype=float)
        assert np.allclose(np.cross(p, pts[:, i, j]), 0, atol=1e-9 * np.abs(p).max())


def test_witnesses_paper_quartic():
    deg, ovals, f = paper_form("quartic")
    wit = witnesses(exact_hessian(deg, f), 4, 300)
    assert len(wit) == ovals + 1
    assert all(w["exact_sign"] == w["sign"] for w in wit)


def test_rationalize_recovers_perturbed_paper_quartic():
    deg, ovals, f = paper_form("quartic")
    rng = np.random.default_rng(2)
    c = np.array(f, dtype=float) + rng.standard_normal(len(f)) * 1e-3
    res = rationalize(c / np.linalg.norm(c), deg, widths=(200, 300), mode="round")
    assert res is not None
    assert set(res["ovals"].values()) == {ovals}
    assert oval_counts([res["hessian"]], 4, 300)[0] == ovals
    assert all(v == 0 for v, (i, j, _) in zip(res["f"], monomials(deg)) if i + j < 2)
