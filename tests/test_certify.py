from fractions import Fraction as F

import pytest
import sympy as sp

from conncomp.certify import (
    X,
    Y,
    Z,
    _y_coeffs,
    certify_line,
    certify_path,
    certify_segment,
    certify_upper_bound,
    crosscheck_flint,
    find_outside_loops,
    flint_real_root_count,
    form_poly,
    pencil_tangents,
    subresultant_coeff,
    verify_outside_loop,
    x,
    y,
)
from conncomp.polynomials import from_sympy
from conncomp.rationalize import eval_form, exact_hessian

from test_rationalize import PAPER

CIRCLE = sp.Poly(X**2 + Y**2 - Z**2, X, Y, Z)


def paper_hessian(name):
    deg, ovals, f = PAPER[name]
    h = exact_hessian(deg, [int(v) for v in from_sympy(f, deg)])
    return form_poly(h, 2 * deg - 4), h, ovals


def test_subresultant_resultant_matches_sympy():
    f = 3 * y**4 + (x + 1) * y**3 - x**2 * y + 2 * x - 5
    g = sp.diff(f, y)
    dom = sp.ZZ[x]
    R = subresultant_coeff(_y_coeffs(f, dom), _y_coeffs(g, dom), 0, 0, dom)
    assert sp.Poly(dom.to_sympy(R), x) == sp.Poly(sp.resultant(f, g, y), x)


def test_segments_and_lines_on_circle():
    inside = certify_segment(CIRCLE, [0, 0, 1], [F(1, 2), 0, 1])
    assert inside.ok and inside.sign == -1
    crossing = certify_segment(CIRCLE, [0, 0, 1], [2, 0, 1])
    assert not crossing.ok and crossing.n_roots == 1
    # tangent segment: touches the circle at (1, 0)
    assert not certify_segment(CIRCLE, [1, -1, 1], [1, 1, 1]).ok
    # (-3, 0, -1) is the point (3, 0): this arc runs 2 -> +inf -> -inf -> 3 and crosses the circle twice
    assert certify_segment(CIRCLE, [2, 0, 1], [-3, 0, -1]).n_roots == 2
    # (3, 0, -1) is the point (-3, 0): this arc runs 2 -> +inf -> -inf -> -3 and misses the circle
    assert certify_segment(CIRCLE, [2, 0, 1], [3, 0, -1]).ok
    assert certify_segment(CIRCLE, [2, 0, 1], [3, 0, 1]).ok  # the short arc, also outside
    # the straight segment from (2, 0) to (-3, 0) crosses the circle twice
    assert certify_segment(CIRCLE, [2, 0, 1], [-3, 0, 1]).n_roots == 2
    assert certify_line(CIRCLE, [1, 0, 0], [0, 1, 0]).ok  # line at infinity
    assert certify_line(CIRCLE, [2, 0, 1], [0, 1, 0]).ok  # x = 2
    assert not certify_line(CIRCLE, [F(1, 2), 0, 1], [0, 1, 0]).ok  # x = 1/2
    square = [[2, 2, 1], [-2, 2, 1], [-2, -2, 1], [2, -2, 1]]
    assert all(c.ok and c.sign == 1 for c in certify_path(CIRCLE, square, closed=True))
    # a segment lying on a component of the curve
    assert certify_segment(sp.Poly(X * Y, X, Y, Z), [0, 0, 1], [0, 1, 1]).n_roots == -1


def test_paper_theorem_2_proof():
    """Ortiz-Rodriguez & Sottile, Theorem 2: four lines avoiding the Hessian curve separate 4 points."""
    H, h, _ = paper_hessian("quartic")
    lines = [
        ([1, 0, 0], [0, 1, 0]),  # line at infinity
        ([0, F(3, 4), 1], [1, F(-1, 2), 0]),  # y = 3/4 - x/2
        ([0, F(-1, 4), 1], [1, F(1, 2), 0]),  # y = x/2 - 1/4
        ([F(-2, 5), 0, 1], [0, 1, 0]),  # x = -2/5
    ]
    assert all(certify_line(H, P, Q).ok for P, Q in lines)
    # h = Hess / -4 is negative at the points, so Hess is positive there
    for p in [(-2, 0, 1), (0, 1, 5), (2, 2, 1), (2, -1, 1)]:
        assert eval_form(h, 4, p) > 0


@pytest.mark.parametrize("name", ["quartic", "quintic"])
def test_paper_upper_bounds(name):
    H, _, ovals = paper_hessian(name)
    cert = certify_upper_bound(H, target=ovals)
    assert cert.certified and verify_outside_loop(H, cert.loop)
    assert cert.upper_bound == ovals
    assert crosscheck_flint(H, cert)["agree"]


def test_singular_curve_detected():
    # two circles meeting transversally at two real points
    H = sp.Poly((X**2 + Y**2 - Z**2) * ((X - Z) ** 2 + Y**2 - Z**2), X, Y, Z)
    # (it also has two complex nodes at [1 : +-i : 0], which share a pencil line for the second frame)
    for P, O, R in (([1, -1, 0], [2, 1, 1], [0, 1, 3]), ([2, 1, 1], [1, -1, 0], [1, 0, 2])):
        cert = pencil_tangents(H, O, P, R)
        assert cert.valid and not cert.smooth


def test_pencil_tangents_circle():
    # O = (2, 0) outside the circle: two tangents. The line OP (the x-axis) crosses the circle,
    # which does not matter for the pencil count.
    cert = pencil_tangents(CIRCLE, [2, 0, 1], [0, 0, 1], [0, 1, 0])
    assert cert.valid and cert.smooth and (cert.n_affine, cert.n_line, cert.upper_bound) == (2, 0, 1)
    # O = (1, 1): the line OP is x = 1, tangent at (1, 0); the other tangency point is (0, 1)
    cert = pencil_tangents(CIRCLE, [1, 1, 1], [1, 0, 1], [0, 0, 1])
    assert (cert.n_affine, cert.n_line, cert.upper_bound) == (1, 1, 1)
    assert crosscheck_flint(CIRCLE, cert)["agree"]
    # O on the curve
    assert not pencil_tangents(CIRCLE, [1, 0, 1], [0, 0, 1], [0, 1, 0]).valid


def test_outside_loops_circle():
    (O, loop), = find_outside_loops(CIRCLE, base_point=[2, 0, 1], N=20)
    assert O == [2, 0, 1] and verify_outside_loop(CIRCLE, loop)
    assert find_outside_loops(CIRCLE, base_point=[0, 0, 1], N=20) == []  # inside the oval
    # a closed (contractible) loop, and an "outside" loop crossing the circle, are rejected
    square = [[2, 2, 1], [-2, 2, 1], [-2, -2, 1], [2, -2, 1], [2, 2, 1]]
    assert not verify_outside_loop(CIRCLE, square)
    assert verify_outside_loop(CIRCLE, [[2, 0, 1], [0, 1, 0], [-2, 0, -1]])  # the line x = 2
    assert not verify_outside_loop(CIRCLE, [[2, 0, 1], [1, 0, 0], [-2, 0, -1]])  # the line y = 0


def test_two_circles_bound():
    H = sp.Poly((X**2 + Y**2 - Z**2) * ((X - 3 * Z) ** 2 + Y**2 - Z**2), X, Y, Z)
    cert = certify_upper_bound(H, target=2)
    assert cert.certified and cert.upper_bound == 2 and cert.n_tangent_points == 4



def test_flint_real_root_count():
    assert flint_real_root_count([-2, 0, 1]) == 2  # x^2 - 2
    assert flint_real_root_count([-2, 0, -1, 0, 1]) == 2  # (x^2 - 2)(x^2 + 1)
    assert flint_real_root_count([1, -2, 1]) == 1  # (x - 1)^2
