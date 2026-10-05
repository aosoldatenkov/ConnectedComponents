import random
from fractions import Fraction as F

import sympy as sp

from conncomp.certify import X, Y, Z, _segment_coeffs, certify_segment, segment_poly
from conncomp.separation import certify_lower_bound, certify_separation, parity

CIRCLE = sp.Poly(X**2 + Y**2 - Z**2, X, Y, Z)
TWO = sp.Poly((X**2 + Y**2 - Z**2) * ((X - 3 * Z) ** 2 + Y**2 - Z**2), X, Y, Z)
SQUARE = [[2, 2, 1], [-2, 2, 1], [-2, -2, 1], [2, -2, 1]]


def test_segment_fast_path_agrees_with_sturm():
    H = sp.Poly((X**2 + Y**2 - Z**2) * (X**2 + 4 * Y**2 - 9 * Z**2) * (X + Y - 3 * Z) * (X - Z), X, Y, Z)
    rng = random.Random(0)
    for _ in range(200):
        P = [rng.randint(-5, 5) for _ in range(3)]
        Q = [F(rng.randint(-50, 50), rng.randint(1, 9)) for _ in range(3)]
        if sp.Matrix([P, Q]).rank() < 2:
            continue
        p = segment_poly(H, P, Q)
        expected = -1 if p.is_zero else (int(p.count_roots(0, 1)) if p.degree() > 0 else 0)
        assert certify_segment(H, P, Q).n_roots == expected
        a = _segment_coeffs(H, P, Q)  # positive multiple of p(t) with P, Q rescaled positively
        assert all(v == 0 for v in a) == p.is_zero


def test_parity_in_chart():
    ell = (0, 0, 1)
    assert parity(SQUARE, ell, [0, 0, 1]) == 1
    assert parity(SQUARE, ell, [5, 0, 1]) == 0
    assert parity(SQUARE, ell, [1, 0, 0]) == 0  # on the line ell = 0
    assert parity(SQUARE, ell, [0, 0, -7]) == 1  # another representative of the centre
    # a self-intersecting polygon (bow tie): even-odd rule
    bow = [[2, 2, 1], [-2, -2, 1], [-2, 2, 1], [2, -2, 1]]
    assert parity(bow, ell, [1, 0, 1]) == 1 and parity(bow, ell, [0, 1, 1]) == 0


def test_separation_requires_opposite_sign():
    # two witnesses inside the same circle, and a small loop inside the circle around one of them:
    # the loop avoids the curve but has the sign of the witnesses, so it does not separate them
    wit = [[0, 0, 4], [1, 1, 4]]
    loop = [[F(1, 10), 0, 1], [0, F(1, 10), 1], [F(-1, 10), 0, 1], [0, F(-1, 10), 1]]
    loop = [[int(v * 10) for v in q] for q in loop]
    cert = certify_separation(CIRCLE, wit, [(loop, (0, 0, 1))])
    assert cert.paths[0]["ok"] and cert.pairs["0,1"] is None and cert.lower_bound == 0


def test_two_circles_lower_bound():
    res = certify_lower_bound(TWO, [[0, 0, 1], [3, 0, 1], [10, 10, 1]])
    cert = res.certificate
    assert cert.complete and cert.lower_bound == 2 and not res.failed
    assert cert.pairs["0,2"] == "sign" and cert.pairs["0,1"] is not None
    # a polygon crossing the curve is rejected
    bad = certify_separation(TWO, [[0, 0, 1], [3, 0, 1]], [(SQUARE, (0, 0, 1))])
    assert not bad.paths[0]["ok"] and bad.lower_bound == 0
