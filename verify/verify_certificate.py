#!/usr/bin/env python3
"""Standalone verifier for conncomp oval-count certificates.

Depends only on python-flint (FLINT / Arb) and the Python standard library; it does not import
conncomp. The only data taken from the certificate on trust are the integer coefficients of f.
Everything else (H, the frame, the base point loop, the witnesses and the polygons) is a hint that
is checked or recomputed.

Claim verified: the Hessian curve C = {H = 0}, H = f_xx f_yy - f_xy^2 (f homogenized with z), has
exactly k ovals in RP^2. The mathematical facts used:

  (a) Pencil bound. If C is smooth of even degree and O lies outside all ovals, every oval has at
      least 2 points whose tangent passes through O, so #ovals <= N_tan / 2.
  (b) O lies outside all ovals if a non-contractible loop through O avoids C (its lift to S^2 runs
      from O to -O). All complementary components other than the non-orientable one lie in disks.
  (c) Separation. A closed polygon in an affine chart that avoids C with constant sign s separates
      two witnesses of sign -s with different (even-odd) parity into different components of
      RP^2 minus C. For smooth C of even degree, #components = #ovals + 1.

Algorithms (chosen to differ from the main conncomp code):
  * segment checks: Vincent-Collins-Akritas (Descartes + bisection) on the squarefree part, and
    independently Arb real root isolation;
  * resultant from FLINT; subresultant coefficients by evaluation at integers, integer determinants
    (fmpz_mat) and exact interpolation;
  * real root counts via FLINT factorization and Arb root isolation;
  * parity by an independent integer/rational even-odd routine.

Usage:  python verify/verify_certificate.py data/certificates/*.json
"""

import argparse
import json
import math
import random
import sys
import time
from fractions import Fraction
from itertools import combinations

import flint

CTX3 = flint.fmpz_mpoly_ctx.get(("X", "Y", "Z"), "lex")
CTX2 = flint.fmpz_mpoly_ctx.get(("x", "y"), "lex")
T1 = flint.fmpz_poly([1, 1])  # t + 1


def monomials(deg):
    """Exponents (i, j, k) of x^i y^j z^k in the coefficient order used by the certificates."""
    return [(i, j, deg - i - j) for i in range(deg + 1) for j in range(deg - i + 1)]


class Log:
    def __init__(self, quiet=False):
        self.failures = 0
        self.quiet = quiet

    def check(self, name, ok, detail=""):
        if not ok:
            self.failures += 1
        if not self.quiet or not ok:
            print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
        return ok

    def info(self, msg):
        if not self.quiet:
            print(f"        {msg}")


def sign(v):
    return (v > 0) - (v < 0)


def int_point(P):
    """Positive integer multiple of a rational point given as ints or 'a/b' strings."""
    P = [Fraction(v) for v in P]
    den = math.lcm(*(v.denominator for v in P))
    return [int(v * den) for v in P]


# ---------------------------------------------------------------------------
# Univariate polynomials
# ---------------------------------------------------------------------------


def squarefree(p):
    return p // p.gcd(p.derivative())


def real_roots(p):
    """Number of distinct real roots of a nonzero integer polynomial (FLINT factorization + Arb)."""
    if p.degree() <= 0:
        return 0
    _, factors = p.factor()
    return sum(sum(1 for r, _ in f.complex_roots() if r.imag.is_zero()) for f, _ in factors)


def variations(coeffs):
    s = [c > 0 for c in coeffs if c != 0]
    return sum(a != b for a, b in zip(s, s[1:]))


def vca_roots_01(p):
    """Number of roots in (0, 1) of a squarefree integer polynomial with p(0) != 0 != p(1)."""
    n = p.degree()
    if n <= 0:
        return 0
    c = [int(v) for v in p.coeffs()]
    v = variations(flint.fmpz_poly(c[::-1])(T1).coeffs())  # (1 + s)^n p(1 / (1 + s))
    if v <= 1:
        return v
    left = flint.fmpz_poly([ck * 2 ** (n - k) for k, ck in enumerate(c)])  # 2^n p(t / 2)
    right = left(T1)  # 2^n p((t + 1) / 2)
    mid = 0
    if left(1) == 0:  # p(1/2) = 0
        mid = 1
        left = left // flint.fmpz_poly([-1, 1])
        right = right // flint.fmpz_poly([0, 1])
    return vca_roots_01(left) + vca_roots_01(right) + mid


def arb_roots_01(p):
    """Number of real roots in (0, 1) via Arb isolation, or None if undecided."""
    if p.degree() <= 0:
        return 0
    count = 0
    _, factors = p.factor()
    for f, _ in factors:
        for r, _ in f.complex_roots():
            if not r.imag.is_zero():
                continue
            x = r.real
            if x < 0 or x > 1:
                continue
            if x > 0 and x < 1:
                count += 1
            else:
                return None
    return count


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------


def int_terms(H):
    return [(e, int(c)) for e, c in H.to_dict().items()]


def segment_poly(terms, P, Q):
    """p(t) = H((1 - t) P + t Q) as an integer polynomial."""
    deg = max(sum(e) for e, _ in terms)
    lin = [flint.fmpz_poly([a, b - a]) for a, b in zip(P, Q)]
    powers = []
    for L in lin:
        pw = [flint.fmpz_poly([1])]
        for _ in range(deg):
            pw.append(pw[-1] * L)
        powers.append(pw)
    p = flint.fmpz_poly([0])
    for (i, j, k), c in terms:
        p += c * powers[0][i] * powers[1][j] * powers[2][k]
    return p


def check_segment(terms, P, Q):
    """(ok, sign, detail): whether the arc {[(1 - t) P + t Q] : t in [0, 1]} avoids H = 0."""
    P, Q = int_point(P), int_point(Q)
    p = segment_poly(terms, P, Q)
    if p.is_zero():
        return False, 0, "H vanishes on the segment"
    p0, p1 = int(p(0)), int(p(1))
    if p0 == 0 or p1 == 0:
        return False, 0, "endpoint on the curve"
    if sign(p0) != sign(p1):
        return False, 0, "sign change"
    sq = squarefree(p)
    n_vca, n_arb = vca_roots_01(sq), arb_roots_01(sq)
    if n_vca != 0 or n_arb != 0:
        return False, 0, f"roots in (0,1): VCA {n_vca}, Arb {n_arb}"
    return True, sign(p0), ""


def cross(u, v):
    return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])


def negatively_proportional(u, v):
    return cross(u, v) == (0, 0, 0) and sum(a * b for a, b in zip(u, v)) < 0


# ---------------------------------------------------------------------------
# Upper bound: tangency points of the pencil through O
# ---------------------------------------------------------------------------


def det3(M):
    return int(flint.fmpz_mat(M).det())


def transform(H, M):
    X, Y, Z = CTX3.gens()
    lx, ly, lz = (M[r][0] * X + M[r][1] * Y + M[r][2] * Z for r in range(3))
    return H.compose(lx, ly, lz)


def y_coeff_polys(G):
    """Coefficients of a polynomial in x, y (dict {(i, j): c}) as polynomials in x, ascending in y."""
    dy = max((j for _, j in G), default=0)
    cols = [[0] * (1 + max((i for i, _ in G), default=0)) for _ in range(dy + 1)]
    for (i, j), c in G.items():
        cols[j][i] += int(c)
    return [flint.fmpz_poly(col) for col in cols]


def subres_int(a, b, j, k):
    """Coefficient of y^k in the j-th subresultant of integer polynomials a, b (ascending lists)."""
    m, n = len(a) - 1, len(b) - 1
    N = m + n - j
    rows = []
    for coeffs, shifts in ((a, n - j), (b, m - j)):
        for i in range(shifts - 1, -1, -1):
            row = [0] * N
            for p, c in enumerate(coeffs):
                row[N - 1 - (p + i)] = c
            rows.append(row)
    cols = list(range(m + n - 2 * j - 1)) + [N - 1 - k]
    return int(flint.fmpz_mat([[row[c] for c in cols] for row in rows]).det())


def interpolate(values):
    """Integer polynomial through (x, values[x]) for x = 0, ..., D (Newton forward differences)."""
    D = len(values) - 1
    diffs, cur = [], list(values)
    for _ in range(D + 1):
        diffs.append(cur[0])
        cur = [b - a for a, b in zip(cur, cur[1:])]
    fact = math.factorial(D)
    total = flint.fmpz_poly([0])
    falling = flint.fmpz_poly([1])
    for k, dk in enumerate(diffs):
        total += dk * (fact // math.factorial(k)) * falling
        falling *= flint.fmpz_poly([-k, 1])
    coeffs = [int(c) for c in total.coeffs()]
    if any(c % fact for c in coeffs):
        raise ArithmeticError("interpolation is not integral")
    return flint.fmpz_poly([c // fact for c in coeffs])


def subres_poly(A, B, j, k):
    """Subresultant coefficient s_{j,k}(x) for polynomials A, B in y with coefficients in Z[x]."""
    m, n = len(A) - 1, len(B) - 1
    rowdeg = [max(max(c.degree(), 0) for c in A)] * (n - j) + [max(max(c.degree(), 0) for c in B)] * (m - j)
    D = sum(rowdeg)
    vals = [subres_int([int(c(x0)) for c in A], [int(c(x0)) for c in B], j, k) for x0 in range(D + 1)]
    return interpolate(vals)


def fibre(polys, x0):
    """Polynomial in y: sum polys[i](x0) y^i for rational x0, scaled to integer coefficients."""
    a, b = x0.numerator, x0.denominator
    D = max(max(p.degree(), 0) for p in polys)
    return flint.fmpz_poly(
        [sum(int(c) * a**e * b ** (D - e) for e, c in enumerate(p.coeffs())) for p in polys]
    )


def to_univariate(G3, n):
    """Restrict a form in X, Y (Z already substituted) to Y = 1 as a polynomial in X."""
    c = [0] * (n + 1)
    for (i, j, _), v in G3.to_dict().items():
        c[i] += int(v)
    return flint.fmpz_poly(c)


def pencil_count(H, M):
    """Tangency points of the pencil through O = M e2. Returns a dict (valid, smooth, n_affine, n_line)."""
    n = H.total_degree()
    Hp = transform(H, M)
    out = {"valid": False, "smooth": False, "n_affine": -1, "n_line": -1, "reason": ""}
    if int(Hp.to_dict().get((0, n, 0), 0)) == 0:
        out["reason"] = "base point on the curve"
        return out
    g = {(i, j): int(c) for (i, j, k), c in Hp.subs({"Z": 1}).to_dict().items()}
    gd = CTX2.from_dict(g)
    gy, gx = gd.derivative("y"), gd.derivative("x")
    A = y_coeff_polys(g)
    B = y_coeff_polys({e: int(c) for e, c in gy.to_dict().items()})
    Cx = y_coeff_polys({e: int(c) for e, c in gx.to_dict().items()})
    Rf = gd.resultant(gy, "y")
    R = y_coeff_polys({e: int(c) for e, c in Rf.to_dict().items()})[0]
    if R.is_zero():
        out["reason"] = "resultant vanishes"
        return out
    R_int = subres_poly(A, B, 0, 0)
    if R_int != R and R_int != -R:
        out["reason"] = "FLINT resultant and interpolated Sylvester determinant differ"
        return out
    s1, s0 = subres_poly(A, B, 1, 1), subres_poly(A, B, 1, 0)
    Rs = squarefree(R)

    special = []
    G1 = Rs.gcd(s1)
    if G1.degree() > 0:
        for fac, _ in G1.factor()[1]:
            if real_roots(fac) == 0:
                continue
            if fac.degree() > 1:
                out["reason"] = "tangency points share an irrational pencil line"
                return out
            c0, c1 = (int(v) for v in fac.coeffs())
            special.append(Fraction(-c0, c1))
    n_affine = real_roots(Rs) - len(special)
    smooth = True
    for x0 in special:
        common = fibre(A, x0).gcd(fibre(B, x0))
        n_affine += real_roots(common)
        smooth &= real_roots(common.gcd(fibre(Cx, x0))) == 0
    D = len(Cx) - 1
    Nx = flint.fmpz_poly([0])
    for k, ck in enumerate(Cx):
        Nx += ck * (-s0) ** k * s1 ** (D - k)
    G2 = Rs.gcd(Nx)
    for x0 in special:
        lin = flint.fmpz_poly([-x0.numerator, x0.denominator])
        G2 = G2 // G2.gcd(lin)
    smooth &= real_roots(G2) == 0

    # the line Z = 0 through O
    B0 = Hp.subs({"Z": 0})
    parts = [Hp.derivative(v).subs({"Z": 0}) for v in ("X", "Y", "Z")]
    b1 = to_univariate(B0, n)
    p1 = [to_univariate(q, n) for q in parts]
    at10 = [int(q.to_dict().get((n - 1, 0, 0), 0)) for q in parts]  # partials at [1:0:0] (degree n - 1)
    T = b1.gcd(p1[1])
    n_line = real_roots(T)
    on10 = b1.degree() < n
    if on10 and at10[1] == 0:
        n_line += 1
    G3 = b1
    for q in p1:
        G3 = G3.gcd(q)
    smooth &= real_roots(G3) == 0
    if on10:
        smooth &= any(at10)
    out.update(valid=True, smooth=bool(smooth), n_affine=n_affine, n_line=n_line)
    return out


# ---------------------------------------------------------------------------
# Lower bound: parity and separation
# ---------------------------------------------------------------------------


def parity(polygon, ell, w):
    """1 if w is inside the polygon (even-odd rule in the chart ell > 0), 0 otherwise."""
    lw = sum(a * b for a, b in zip(ell, w))
    if lw == 0:
        return 0
    k = max(range(3), key=lambda i: abs(ell[i]))  # coordinates: the other two axes
    a, b = [i for i in range(3) if i != k]

    def chart(p):
        lp = sum(u * v for u, v in zip(ell, p))
        return Fraction(p[a], lp), Fraction(p[b], lp)

    pts = [chart(v) for v in polygon]
    px, py = chart(w)
    inside = 0
    for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]):
        if (ay > py) != (by > py):
            xi = ax + (py - ay) * (bx - ax) / (by - ay)
            if px < xi:
                inside ^= 1
    return inside


def max_clique(m, adj):
    best = []

    def grow(cl, cand):
        nonlocal best
        if len(cl) + len(cand) <= len(best):
            return
        if not cand:
            best = cl
            return
        v, rest = cand[0], cand[1:]
        grow(cl + [v], [u for u in rest if adj(v, u)])
        grow(cl, rest)

    grow([], list(range(m)))
    return best


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------


def hessian(f):
    fxx = f.derivative("X").derivative("X")
    fyy = f.derivative("Y").derivative("Y")
    fxy = f.derivative("X").derivative("Y")
    return fxx * fyy - fxy * fxy


def verify(cert, log, extra_frames=2, seed=1):
    """Verify a certificate dict. Returns the proven number of ovals, or None."""
    d = cert["deg"]
    f = CTX3.from_dict({m: int(c) for m, c in zip(monomials(d), cert["f"]) if int(c)})
    Hf = hessian(f)
    n = Hf.total_degree()
    H = CTX3.from_dict({m: int(c) for m, c in zip(monomials(cert["hessian_deg"]), cert["hessian"]) if int(c)})
    # H must be a positive multiple of Hess(f): Hess(f) = (a/b) H with a/b > 0
    e, c = next(iter(Hf.to_dict().items()))
    h = int(H.to_dict().get(e, 0))
    ratio = Fraction(int(c), h) if h else None
    prop = ratio is not None and ratio > 0 and Hf * ratio.denominator == H * ratio.numerator
    if not log.check("stored H is a positive multiple of f_xx f_yy - f_xy^2", prop):
        return None
    if not log.check("degree of H is even", n % 2 == 0, f"degree {n}"):
        return None
    terms = int_terms(H)

    # --- base point outside all ovals
    up = cert.get("upper", {})
    loop = [int_point(p) for p in up.get("loop", [])]
    M = [[int(Fraction(v)) for v in row] for row in up.get("M", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])]
    O = [M[r][1] for r in range(3)]
    ok = len(loop) >= 3 and negatively_proportional(loop[0], loop[-1])
    ok = ok and cross(loop[0], O) == (0, 0, 0)  # the loop starts at the base point of the frame
    ok &= not any(negatively_proportional(u, v) for u, v in zip(loop, loop[1:]))
    seg_ok = ok and all(check_segment(terms, u, v)[0] for u, v in zip(loop, loop[1:]))
    if not log.check("base point O lies outside all ovals (non-contractible loop avoiding C)", seg_ok,
                     f"O = {O}, {len(loop) - 1} segments"):
        return None

    # --- pencil count, in the certificate's frame and in random extra frames
    t0 = time.time()
    if not log.check("frame M is invertible", det3(M) != 0):
        return None
    res = pencil_count(H, M)
    if not log.check("tangency points in the certificate's frame are separated or resolved", res["valid"],
                     res["reason"]):
        return None
    if not log.check("C is smooth (no real singular points)", res["smooth"]):
        return None
    n_tan = res["n_affine"] + res["n_line"]
    log.info(f"tangency points: {res['n_affine']} off the line OP + {res['n_line']} on it = {n_tan} "
             f"({time.time() - t0:.1f}s)")
    rng = random.Random(seed)
    counts, tries = [], 0
    while len(counts) < extra_frames and tries < 20:
        tries += 1
        P, R = [rng.randint(-7, 7) for _ in range(3)], [rng.randint(-7, 7) for _ in range(3)]
        M2 = [[P[r], O[r], R[r]] for r in range(3)]
        if det3(M2) == 0:
            continue
        r2 = pencil_count(H, M2)
        if r2["valid"]:
            counts.append(r2["n_affine"] + r2["n_line"])
    log.check("tangency count is independent of the frame", counts and all(c == n_tan for c in counts),
              f"extra frames: {counts}")
    upper = n_tan // 2
    log.info(f"upper bound: #ovals <= {upper}")

    # --- lower bound
    witnesses = [int_point(w["point"]) for w in cert["witnesses"]]
    wsign = [sign(int(H(*w))) for w in witnesses]
    paths = []
    t0 = time.time()
    n_seg = 0
    for k, p in enumerate(cert.get("lower", {}).get("paths", [])):
        poly = [int_point(v) for v in p["polygon"]]
        ell = [int(v) for v in p["ell"]]
        ok = len(poly) >= 3 and all(sum(a * b for a, b in zip(ell, v)) > 0 for v in poly)
        signs = set()
        for u, v in zip(poly, poly[1:] + poly[:1]):
            if not ok:
                break
            s_ok, s, _ = check_segment(terms, u, v)
            ok &= s_ok
            signs.add(s)
            n_seg += 1
        ok &= len(signs) == 1
        log.check(f"polygon {k} lies in a chart and avoids C with constant sign", ok, f"{len(poly)} vertices")
        if ok:
            paths.append((poly, ell, signs.pop(), [parity(poly, ell, w) for w in witnesses]))
    log.info(f"{n_seg} polygon segments checked ({time.time() - t0:.1f}s)")

    def separated(i, j):
        if wsign[i] == 0 or wsign[j] == 0:
            return False
        if wsign[i] != wsign[j]:
            return True
        return any(s == -wsign[i] and par[i] != par[j] for _, _, s, par in paths)

    clique = max_clique(len(witnesses), separated)
    lower = len(clique) - 1
    unsep = [(i, j) for i, j in combinations(range(len(witnesses)), 2) if not separated(i, j)]
    log.check("witnesses pairwise separated", not unsep, f"unseparated pairs: {unsep}" if unsep else
              f"{len(witnesses)} witnesses")
    log.info(f"lower bound: #ovals >= {lower}")

    log.check("lower bound equals upper bound", lower == upper, f"{lower} vs {upper}")

    # extra statement: all ovals are compact in the affine plane z = 1 iff the line z = 0 misses C
    compact = all(check_segment(terms, u, v)[0] for u, v in (([1, 0, 0], [0, 1, 0]), ([0, 1, 0], [-1, 0, 0])))
    log.info(f"line at infinity z = 0 {'misses' if compact else 'meets'} the curve"
             + (": all ovals are compact in R^2" if compact else ""))
    cert["_compact"] = compact
    return upper if log.failures == 0 else None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("-q", "--quiet", action="store_true", help="only print failures and verdicts")
    args = p.parse_args(argv)
    print(f"python-flint {flint.__version__}")
    status = 0
    for fn in args.files:
        with open(fn) as fh:
            cert = json.load(fh)
        print(f"{fn}:")
        log = Log(args.quiet)
        t0 = time.time()
        proven = verify(cert, log)
        if proven is None:
            status = 1
            print(f"  => NOT VERIFIED ({log.failures} failed checks)")
        else:
            where = "RP^2, all compact in R^2" if cert.get("_compact") else "RP^2"
            dt = time.time() - t0
            print(f"  => VERIFIED: the Hessian curve of f has exactly {proven} ovals in {where} ({dt:.1f}s)")
    return status


if __name__ == "__main__":
    sys.exit(main())
