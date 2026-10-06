r"""Exact (symbolic) certification of the topology of real plane curves H = 0 in RP^2.

Everything here uses exact integer/rational arithmetic in sympy. The critical point counts
are cross-checked with FLINT (python-flint): an exact resultant, plus certified real root
isolation from Arb.

Segments
--------
`certify_segment(H, P, Q)` proves that the geodesic segment of RP^2 between P and Q does not meet
the curve. The segment is {[(1 - t) P + t Q] : t in [0, 1]}; flipping the sign of Q gives the
complementary arc of the same projective line. In the chart z = 1, with P = (x1, y1, 1) and
Q = (x2, y2, 1), it is the ordinary straight segment.

Upper bound by tangents from a point (pencil of lines)
------------------------------------------------------
Let C = {H = 0} be smooth of even degree, so every component is an oval. Fix a base point O not on
C and consider the pencil of lines through O, i.e. the projection pi: RP^2 \ {O} -> RP^1. If O lies
outside an oval C_i (O is not in the disk bounded by C_i), then pi restricted to C_i lifts to R,
so it has at least two critical points (a maximum and a minimum). These are the points of C_i
whose tangent line passes through O. Hence, if O lies outside all ovals,

    #ovals <= N_tan / 2,    N_tan = #real points p of C with O on the tangent line at p,

i.e. the real points of C intersected with the polar curve sum_i O_i dH/dX_i = 0. (A bitangent through O
counts twice, once for each tangency point.)

O lies outside all ovals iff it lies in the non-orientable component of RP^2 \ C. This is certified
by a non-contractible loop through O that avoids C, given as a chain of certified segments whose
lift to S^2 runs from O to -O (`verify_outside_loop`). All other complementary components lie in
disks, where every loop is contractible. A line through O that misses C is the special case
O -> Q -> -O.

N_tan is computed exactly in coordinates with O = [0:1:0] (columns of M: P, O, R; the line Z = 0
is the line OP):
  * affine tangency points: solutions of g = g_y = 0 with g(x, y) = H'(x, y, 1). R(x) = Res_y(g, g_y)
    has their x-coordinates as roots. If the first principal subresultant coefficient s_1(x) is
    nonzero at a real root x0 of R, then the fibre over x0 has exactly one tangency point
    y0 = -s_0(x0)/s_1(x0), which is real. Fibres where s_1 also vanishes are resolved by an
    exact gcd when x0 is rational.
  * tangency points on the line Z = 0: common real roots of H'(x, y, 0) and H'_Y(x, y, 0).
The same data certifies that C has no real singular points.
"""

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import flint
import numpy as np
import sympy as sp
from sympy.polys.matrices import DomainMatrix

from conncomp.polynomials import monomials

X, Y, Z = sp.symbols("X Y Z")
x, y, t = sp.symbols("x y t")


def form_poly(coefs, deg):
    """sympy Poly in X, Y, Z of the form with (integer / rational) coefficients listed by monomials(deg)."""
    expr = sum(sp.Rational(c) * X**i * Y**j * Z**k for c, (i, j, k) in zip(coefs, monomials(deg)))
    return sp.Poly(expr, X, Y, Z)


def _rat(v):
    return sp.Rational(Fraction(v)) if not isinstance(v, sp.Basic) else v


def transform(H, M):
    """The form H o M, i.e. H'(X, Y, Z) = H(M (X, Y, Z)^T), for a 3x3 rational matrix M."""
    M = [[_rat(v) for v in row] for row in M]
    sub = {v: M[r][0] * X + M[r][1] * Y + M[r][2] * Z for r, v in enumerate((X, Y, Z))}
    return sp.Poly(H.as_expr().subs(sub, simultaneous=True), X, Y, Z)


def n_real_roots(p, multiplicity=False):
    """Number of real roots of a univariate sympy Poly: distinct, or counted with multiplicity."""
    if p.is_zero:
        raise ValueError("zero polynomial")
    if p.degree() <= 0:
        return 0
    if not multiplicity:
        return int(p.sqf_part().count_roots())
    _, factors = p.sqf_list()
    return int(sum(k * f.count_roots() for f, k in factors))


# ---------------------------------------------------------------------------
# Segments and lines
# ---------------------------------------------------------------------------


@dataclass
class SegmentCertificate:
    ok: bool  # True iff the segment does not meet the curve
    sign: int  # sign of H on the segment (with the given representatives P, Q) if ok, else 0
    n_roots: int  # number of distinct zeros of H on the segment (-1 if H vanishes identically on it)


def segment_poly(H, P, Q):
    """The univariate Poly p(t) = H((1 - t) P + t Q) over QQ."""
    P, Q = [_rat(v) for v in P], [_rat(v) for v in Q]
    sub = {v: (1 - t) * a + t * b for v, a, b in zip((X, Y, Z), P, Q)}
    return sp.Poly(H.as_expr().subs(sub, simultaneous=True), t, domain=sp.QQ)


@lru_cache(maxsize=64)
def _integer_terms(H):
    """Terms ((i, j, k), c) of a positive integer multiple of H."""
    Hz = H.clear_denoms(convert=True)[1] if H.domain != sp.ZZ else H
    return tuple((e, int(c)) for e, c in Hz.as_dict().items())


def _primitive_integer_point(P):
    """A positive multiple of the rational vector P with integer entries (same point, same ray)."""
    P = [Fraction(v) if not isinstance(v, sp.Basic) else Fraction(int(v.p), int(v.q)) for v in P]
    den = math.lcm(*(v.denominator for v in P))
    return [int(v * den) for v in P]


def _polymul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return out


def _segment_coeffs(H, P, Q):
    """Integer coefficients (ascending) of a positive multiple of p(t) = H((1 - t) P' + t Q').

    P', Q' are positive multiples of P, Q; they trace the same arc of RP^2 and give the same signs.
    """
    P, Q = _primitive_integer_point(P), _primitive_integer_point(Q)
    terms = _integer_terms(H)
    n = max(sum(e) for e, _ in terms)
    lin = [[a, b - a] for a, b in zip(P, Q)]  # (1 - t) a + t b
    powers = []
    for L in lin:
        pw = [[1]]
        for _ in range(n):
            pw.append(_polymul(pw[-1], L))
        powers.append(pw)
    out = [0] * (n + 1)
    for (i, j, k), c in terms:
        prod = _polymul(_polymul(powers[0][i], powers[1][j]), powers[2][k])
        for d, v in enumerate(prod):
            out[d] += c * v
    return out


def _descartes_no_root_in_01(a):
    """True if p(t) = sum a_k t^k provably has no root in (0, 1) (Descartes' rule after t = 1/(1+s))."""
    n = len(a) - 1
    q = [0] * (n + 1)  # q(s) = (1 + s)^n p(1 / (1 + s)) = sum_k a_k (1 + s)^(n - k)
    for k, ak in enumerate(a):
        if ak:
            m = n - k
            for i in range(m + 1):
                q[i] += ak * math.comb(m, i)
    signs = [v > 0 for v in q if v]
    return all(sg == signs[0] for sg in signs)


def certify_segment(H, P, Q):
    """Exactly decide whether the segment {[(1 - t) P + t Q] : t in [0, 1]} of RP^2 avoids H = 0.

    H: sympy Poly in X, Y, Z (see `form_poly`); P, Q: homogeneous coordinates (int, Fraction or
    sympy rationals), not proportional. A fast exact test (Descartes' rule of signs on [0, 1]) is
    tried first; if it is inconclusive, the roots are counted with Sturm sequences (sympy count_roots).
    """
    a = _segment_coeffs(H, P, Q)
    if not any(a):
        return SegmentCertificate(False, 0, -1)
    p0, p1 = a[0], sum(a)
    if p0 and p1 and _descartes_no_root_in_01(a):
        return SegmentCertificate(True, 1 if p0 > 0 else -1, 0)
    p = sp.Poly(list(reversed(a)), t, domain=sp.ZZ)
    n = int(p.count_roots(0, 1)) if p.degree() > 0 else 0
    if n:
        return SegmentCertificate(False, 0, n)
    return SegmentCertificate(True, 1 if p0 > 0 else -1, 0)


def certify_path(H, points, closed=False):
    """Certify that the polygonal path through `points` (closed if `closed`) avoids H = 0.

    Consecutive points are joined by the segments of `certify_segment`. Returns the list of
    per-segment certificates; the path avoids the curve iff all of them are ok.
    """
    pts = list(points) + ([points[0]] if closed else [])
    return [certify_segment(H, P, Q) for P, Q in zip(pts, pts[1:])]


def certify_line(H, P, Q):
    """Exactly decide whether the whole projective line through P and Q avoids H = 0."""
    first = certify_segment(H, P, Q)
    if not first.ok:
        return first
    second = certify_segment(H, Q, [-v for v in P])
    if not second.ok:
        return second
    return SegmentCertificate(True, first.sign, 0)


# ---------------------------------------------------------------------------
# Subresultants
# ---------------------------------------------------------------------------


def subresultant_coeff(a, b, j, k, dom):
    """Coefficient of y^k in the j-th subresultant S_j(f, g) (determinantal definition).

    a, b: coefficients of f, g in y (a[i] = coeff of y^i, elements of `dom`), deg f = m >= deg g = n > j.
    s_{0,0} is the resultant Res_y(f, g); s_{j,j} is the j-th principal subresultant coefficient.
    """
    m, n = len(a) - 1, len(b) - 1
    N = m + n - j
    rows = []
    for coeffs, shifts in ((a, n - j), (b, m - j)):
        for i in range(shifts - 1, -1, -1):
            row = [dom.zero] * N
            for p, c in enumerate(coeffs):
                row[N - 1 - (p + i)] = c
            rows.append(row)
    cols = list(range(m + n - 2 * j - 1)) + [N - 1 - k]
    mat = DomainMatrix([[row[c] for c in cols] for row in rows], (len(rows), len(cols)), dom)
    return mat.det()


def _y_coeffs(expr, dom):
    """Coefficients (ascending in y) of a polynomial in x, y, as elements of dom = ZZ[x]."""
    p = sp.Poly(expr, y)
    return [dom.from_sympy(c) for c in reversed(p.all_coeffs())]


# ---------------------------------------------------------------------------
# Base point outside all ovals
# ---------------------------------------------------------------------------


def _negatively_proportional(u, v):
    """True iff v = -c u with c > 0 (then the segment from u to v passes through 0)."""
    u, v = [_rat(a) for a in u], [_rat(a) for a in v]
    cross = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    return all(c == 0 for c in cross) and sum(a * b for a, b in zip(u, v)) < 0


def verify_outside_loop(H, loop):
    """Exactly verify that `loop` certifies that loop[0] = O lies outside all ovals of H = 0.

    Requirements: loop[-1] is a negative multiple of O (so the lift to S^2 runs from O to -O and the
    loop is non-contractible in RP^2), no segment passes through 0, and every segment avoids the curve.
    """
    if len(loop) < 3 or not _negatively_proportional(loop[0], loop[-1]):
        return False
    if any(_negatively_proportional(u, v) for u, v in zip(loop, loop[1:])):
        return False
    return all(c.ok for c in certify_path(H, loop))


def _shortcut(H, pts):
    """Greedily replace runs of the path by single certified segments; returns None if a step fails."""
    out = [pts[0]]
    i, end = 0, len(pts) - 1
    while i < end:
        steps = sorted({end - i} | {2**k for k in range(12) if 2**k < end - i}, reverse=True)
        for step in steps:
            j = i + step
            if not _negatively_proportional(pts[i], pts[j]) and certify_segment(H, pts[i], pts[j]).ok:
                break
        else:
            return None
        out.append(pts[j])
        i = j
    return out


def find_outside_loops(H, count=1, base_point=None, N=50, max_attempts=50, seed=0):
    """Find base points O outside all ovals, each with a certified loop (see `verify_outside_loop`).

    Loops are found on the cube lattice of conncomp.sphere (a lattice path from O to -O through
    points where H has the sign of H(O)), shortened greedily, and certified segment by segment.
    If `base_point` is given, it is joined to the nearest lattice point. Otherwise candidates are
    lattice points where |H| is large, in random order among the top ones; a lattice component
    that does not reach its antipode (the inside of an oval) is excluded entirely.
    Returns a list of (O, loop), possibly empty.
    """
    from conncomp.sphere import CubeLattice, path_to_antipode

    lat = CubeLattice(N)
    vals = lat.values(H.as_dict().items())
    if base_point is not None:
        O = [_rat(v) for v in base_point]
        path, _ = path_to_antipode(lat, vals, lat.nearest(base_point))
        if path is None:
            return []
        loop = _shortcut(H, [O] + [lat.point(i) for i in path] + [[-v for v in O]])
        return [(loop[0], loop)] if loop is not None and verify_outside_loop(H, loop) else []

    rng = random.Random(seed)
    order = [int(i) for i in np.argsort(-np.abs(vals))]
    top = order[: max(10 * count, len(order) // 10)]
    rng.shuffle(top)
    excluded, found = set(), []
    for i in top + order:
        if len(found) >= count or max_attempts <= 0:
            break
        if i in excluded:
            continue
        max_attempts -= 1
        path, visited = path_to_antipode(lat, vals, i)
        if path is None:
            excluded |= visited
            continue
        excluded |= {i, lat.antipode(i)}
        loop = _shortcut(H, [lat.point(j) for j in path])
        if loop is not None and verify_outside_loop(H, loop):
            found.append((loop[0], loop))
    return found


# ---------------------------------------------------------------------------
# Tangents in the pencil through the base point
# ---------------------------------------------------------------------------


@dataclass
class PencilCertificate:
    base_point: list  # O
    M: list  # change of coordinates, columns P, O, R (O -> [0:1:0], line OP -> Z = 0)
    valid: bool  # all real tangency points are separated by the pencil, or resolved exactly
    smooth: bool  # no real singular points
    n_affine: int  # real tangency points off the line OP
    n_line: int  # real tangency points on the line OP
    n_tangent_points: int  # N_tan = n_affine + n_line
    upper_bound: int  # floor(N_tan / 2): bound on the number of ovals not containing O
    loop: list = field(default_factory=list)  # certificate that O lies outside all ovals (if any)
    outside: bool = False  # loop verified
    certified: bool = False  # valid, smooth and outside: #ovals <= upper_bound is proven
    n_res_real: int = -1  # distinct real roots of R (for the cross-check)
    resultant: list = field(default_factory=list)  # integer coefficients of R(x), ascending
    reason: str = ""


def _integer_form(H):
    """H scaled to have integer coefficients."""
    return H.clear_denoms(convert=True)[1] if H.domain != sp.ZZ else H


def _plain(v):
    v = _rat(v)
    return int(v) if v.is_integer else str(v)


def pencil_tangents(H, O, P, R):
    """Certified count of the tangency points of the pencil of lines through O (see module doc).

    P, R: points completing O to a projective frame; the line OP becomes the line Z = 0.
    Does not check that O lies outside the ovals (see `find_outside_loops`).
    """
    n = H.total_degree()
    M = [[P[r], O[r], R[r]] for r in range(3)]
    Mi = [[_plain(v) for v in row] for row in M]
    Oi = [_plain(v) for v in O]
    fail = dict(base_point=Oi, M=Mi, valid=False, smooth=False, n_affine=-1, n_line=-1, n_tangent_points=-1,
                upper_bound=-1)
    if sp.Matrix(M).det() == 0:
        return PencilCertificate(**fail, reason="P, O, R are not a projective frame")
    Hp = _integer_form(transform(H, M))
    c = Hp.coeff_monomial(Y**n)
    if c == 0:
        return PencilCertificate(**fail, reason="the base point lies on the curve")

    dom = sp.ZZ[x]
    g = Hp.as_expr().subs(Z, 1).subs({X: x, Y: y})
    gy, gx = sp.diff(g, y), sp.diff(g, x)
    a, b = _y_coeffs(g, dom), _y_coeffs(gy, dom)
    Rx = sp.Poly(dom.to_sympy(subresultant_coeff(a, b, 0, 0, dom)), x)
    if Rx.is_zero:
        return PencilCertificate(**fail, reason="Res_y(g, g_y) = 0: H is not squarefree")
    s1 = sp.Poly(dom.to_sympy(subresultant_coeff(a, b, 1, 1, dom)), x)
    s0 = sp.Poly(dom.to_sympy(subresultant_coeff(a, b, 1, 0, dom)), x)
    Rs = Rx.sqf_part()
    resultant = [int(v) for v in reversed(Rx.all_coeffs())]
    n_res_real = int(Rs.count_roots())

    # Real roots x0 of R with s1(x0) = 0: the fibre may hold several tangency points. Rational x0
    # are handled exactly fibre by fibre; irrational ones make this frame unusable.
    G1 = sp.gcd(Rs, s1)
    special = []
    if G1.degree() > 0:
        for fac, _ in G1.factor_list()[1]:
            if fac.count_roots() == 0:
                continue
            if fac.degree() > 1:
                return PencilCertificate(
                    **fail, resultant=resultant, reason="tangency points share an irrational pencil line"
                )
            special.append(-fac.nth(0) / fac.nth(1))
    n_affine = n_res_real - len(special)
    smooth_affine = True
    for x0 in special:
        common = sp.gcd(sp.Poly(g.subs(x, x0), y), sp.Poly(gy.subs(x, x0), y))
        n_affine += n_real_roots(common)
        sing = sp.gcd(common, sp.Poly(gx.subs(x, x0), y))
        smooth_affine = smooth_affine and (sing.degree() <= 0 or sing.count_roots() == 0)

    # Affine singular points over the other real roots x0 of R: g_x(x0, -s0(x0)/s1(x0)) = 0
    gx_c = [sp.Poly(dom.to_sympy(v), x) for v in _y_coeffs(gx, dom)]
    D = len(gx_c) - 1
    Nx = sum((ck * (-s0) ** k * s1 ** (D - k) for k, ck in enumerate(gx_c)), sp.Poly(0, x))
    G2 = sp.gcd(Rs, Nx)
    for x0 in special:  # Rs is squarefree, so each special root divides G2 at most once
        lin = sp.Poly(x - x0, x)
        if G2.rem(lin).is_zero:
            G2 = G2.quo(lin)
    smooth_affine = smooth_affine and (G2.degree() <= 0 or G2.count_roots() == 0)

    # The line OP (Z = 0): points [x0 : 1 : 0] and [1 : 0 : 0]. Tangency: H' = H'_Y = 0 there.
    partials = [sp.diff(Hp.as_expr(), v).subs(Z, 0) for v in (X, Y, Z)]
    b1 = sp.Poly(Hp.as_expr().subs({Z: 0, X: x, Y: 1}), x)
    p1 = [sp.Poly(d.subs({X: x, Y: 1}), x) for d in partials]
    T = sp.gcd(b1, p1[1])
    n_line = n_real_roots(T) if not T.is_zero and T.degree() > 0 else 0
    on_curve_10 = b1.degree() < n  # [1:0:0] lies on the curve
    if on_curve_10 and partials[1].subs({X: 1, Y: 0}) == 0:
        n_line += 1
    G3 = b1
    for d in p1:
        G3 = sp.gcd(G3, d)
    smooth_line = G3.degree() <= 0 or G3.count_roots() == 0
    if on_curve_10:
        smooth_line = smooth_line and any(d.subs({X: 1, Y: 0}) != 0 for d in partials)

    smooth = smooth_affine and smooth_line
    reason = ""
    if not smooth:
        reason = "real singular point " + ("off the line OP" if not smooth_affine else "on the line OP")
    n_tan = n_affine + n_line
    return PencilCertificate(
        base_point=Oi,
        M=Mi,
        valid=True,
        smooth=smooth,
        n_affine=n_affine,
        n_line=n_line,
        n_tangent_points=n_tan,
        upper_bound=n_tan // 2,
        n_res_real=n_res_real,
        resultant=resultant,
        reason=reason,
    )


def _random_point(rng, bound):
    return [rng.randint(-bound, bound) for _ in range(3)]


def certify_upper_bound(H, target=None, base_point=None, n_base=4, n_frames=3, N=50, seed=0, verbose=False):
    """Smallest certified upper bound on the number of ovals, over several base points and frames.

    For each base point O (given, or found by `find_outside_loops`), certifies that O lies outside
    all ovals and counts the tangency points of the pencil through O for `n_frames` random choices
    of the line OP. Stops early once the bound is <= target. Returns the best certified
    PencilCertificate, else the last attempt (or None if no base point could be certified).
    """
    if H.total_degree() % 2:
        raise ValueError("the oval bound needs a curve of even degree")
    rng = random.Random(seed)
    best, last = None, None
    bases = find_outside_loops(H, count=n_base, base_point=base_point, N=N, seed=seed)
    for O, loop in bases:
        for _ in range(n_frames):
            P, R = _random_point(rng, 5), _random_point(rng, 5)
            cert = pencil_tangents(H, O, P, R)
            cert.loop = [[_plain(v) for v in p] for p in loop]
            cert.outside = True
            cert.certified = cert.valid and cert.smooth
            last = cert
            if verbose:
                print(f"  O={cert.base_point} P={P}: valid={cert.valid} smooth={cert.smooth} "
                      f"tangents={cert.n_affine}+{cert.n_line} bound={cert.upper_bound} {cert.reason}")
            if cert.certified and (best is None or cert.upper_bound < best.upper_bound):
                best = cert
            if best is not None and target is not None and best.upper_bound <= target:
                return best
    return best or last


# ---------------------------------------------------------------------------
# Cross-check with FLINT / Arb
# ---------------------------------------------------------------------------


def _flint_real_roots(p, multiplicity=False):
    _, factors = p.factor()
    return sum((k if multiplicity else 1) * sum(1 for r, _ in f.complex_roots() if r.imag.is_zero())
               for f, k in factors)


def flint_real_root_count(coeffs):
    """Number of distinct real roots of an integer polynomial (ascending coefficients), via Arb.

    Factors into irreducibles over ZZ; the roots of each factor are isolated by
    arb_fmpz_poly_complex_roots, which returns real roots with an exactly zero imaginary part.
    """
    return _flint_real_roots(flint.fmpz_poly([int(v) for v in coeffs]))


def crosscheck_flint(H, cert):
    """Recompute R = Res_y(g, g_y), its real roots and the tangency points on the line OP with FLINT."""
    Hp = _integer_form(transform(H, cert.M))
    ctx = flint.fmpz_mpoly_ctx.get(("x", "y"), "lex")
    g = ctx.from_dict({e: int(cf) for e, cf in sp.Poly(Hp.as_expr().subs(Z, 1), X, Y).as_dict().items()})
    R = g.resultant(g.derivative("y"), "y")
    Rd = R.to_dict()
    deg = max(e[0] for e in Rd) if Rd else 0
    R_flint = [int(Rd.get((i, 0), 0)) for i in range(deg + 1)]
    same = R_flint == cert.resultant or R_flint == [-v for v in cert.resultant]
    n_res_real = flint_real_root_count(R_flint)

    def fpoly(expr):
        return flint.fmpz_poly([int(v) for v in reversed(sp.Poly(expr, x).all_coeffs())])

    n = H.total_degree()
    B = Hp.as_expr().subs(Z, 0)
    BY = sp.diff(Hp.as_expr(), Y).subs(Z, 0)
    T = fpoly(B.subs({X: x, Y: 1})).gcd(fpoly(BY.subs({X: x, Y: 1})))
    n_line = _flint_real_roots(T) if T.degree() > 0 else 0
    if sp.Poly(B.subs({X: x, Y: 1}), x).degree() < n and BY.subs({X: 1, Y: 0}) == 0:
        n_line += 1
    return {
        "resultant_equal": same,
        "n_res_real": n_res_real,
        "n_line": n_line,
        "agree": same and n_res_real == cert.n_res_real and n_line == cert.n_line,
    }


# ---------------------------------------------------------------------------
# Certificate files
# ---------------------------------------------------------------------------


def add_affine_info(cert, H=None):
    """Record whether the line at infinity z = 0 misses the curve (then all ovals are compact in R^2).

    The affine plane is canonical here: an affine change of coordinates of f only rescales H(f).
    """
    H = H if H is not None else form_poly(cert["hessian"], cert["hessian_deg"])
    avoids = certify_line(H, [1, 0, 0], [0, 1, 0]).ok
    cert["affine"] = {
        "line_at_infinity_avoids_curve": avoids,
        "compact_ovals_in_R2": cert.get("proven_ovals") if avoids else None,
    }
    return cert


def certify_file(path, base_point=None, n_base=4, n_frames=3, N=50, seed=0, verbose=False):
    """Certify the oval count of a certificate candidate JSON; writes the results back into it.

    Upper bound: tangents of a pencil (`certify_upper_bound`), cross-checked with FLINT; this also
    certifies smoothness. Lower bound: separating polygons around the ovals of the witnesses
    (conncomp.separation), valid for smooth curves of even degree.
    """
    from conncomp.separation import certify_lower_bound

    cert = json.loads(Path(path).read_text())
    H = form_poly(cert["hessian"], cert["hessian_deg"])
    grid = max(cert["ovals_grid"].values())
    ub = certify_upper_bound(H, target=grid, base_point=base_point, n_base=n_base, n_frames=n_frames, N=N,
                             seed=seed, verbose=verbose)
    if ub is None:
        upper = {"certified": False, "reason": "no base point outside all ovals could be certified"}
    else:
        upper = asdict(ub)
        upper.pop("resultant")
        upper["flint"] = crosscheck_flint(H, ub) if ub.valid else None
    cert["upper"] = upper
    cert.pop("symbolic", None)

    lb = certify_lower_bound(H, [w["point"] for w in cert["witnesses"]])
    lower = asdict(lb.certificate)
    lower["failed"] = lb.failed
    lower["valid"] = bool(upper.get("smooth")) and cert["hessian_deg"] % 2 == 0
    cert["lower"] = lower

    flint_ok = bool(upper.get("flint") and upper["flint"]["agree"])
    exact = upper.get("certified") and flint_ok and lower["valid"] and lower["lower_bound"] == upper["upper_bound"]
    cert["proven_ovals"] = upper["upper_bound"] if exact else None
    add_affine_info(cert, H)
    Path(path).write_text(json.dumps(cert, indent=1))
    return cert


def main(argv=None):
    p = argparse.ArgumentParser(description="Certify oval counts of certificate candidates (exact upper/lower bounds)")
    p.add_argument("files", type=Path, nargs="+", help="certificate candidate JSON files (from conncomp-rationalize)")
    p.add_argument("--base-point", type=int, nargs=3, default=None, help="base point O (default: search)")
    p.add_argument("--n-base", type=int, default=4, help="base points to try (default: 4)")
    p.add_argument("--n-frames", type=int, default=3, help="frames (lines OP) per base point (default: 3)")
    p.add_argument("--lattice", type=int, default=50, help="cube lattice size N for the loop search (default: 50)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    for f in args.files:
        cert = certify_file(f, args.base_point, args.n_base, args.n_frames, args.lattice, args.seed, args.verbose)
        u, lo = cert["upper"], cert["lower"]
        grid = max(cert["ovals_grid"].values())
        if not u.get("valid"):
            print(f"{f.name}: grid {grid} ovals; upper bound not certified: {u.get('reason')}")
            continue
        flint_ok = u["flint"] and u["flint"]["agree"]
        verdict = f"PROVEN: exactly {cert['proven_ovals']} ovals" if cert["proven_ovals"] is not None else "not proven"
        print(
            f"{f.name}: grid {grid}; smooth={u['smooth']}; upper {u['upper_bound']} "
            f"({u['n_affine']}+{u['n_line']} tangents from O={u['base_point']}, "
            f"FLINT {'agrees' if flint_ok else 'DISAGREES'}); "
            f"lower {lo['lower_bound']} ({len(lo['paths'])} polygons, {sum(len(q['polygon']) for q in lo['paths'])} "
            f"segments) => {verdict}"
            + (" (all compact in R^2)" if cert["affine"]["compact_ovals_in_R2"] else "")
        )


if __name__ == "__main__":
    main()
