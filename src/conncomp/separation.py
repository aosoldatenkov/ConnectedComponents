"""Exact lower bounds for the number of ovals, via separating polygons (homogeneous coordinates).

Separation criterion
--------------------
Let gamma be a closed polygon in RP^2 whose vertices (as representatives in R^3) all satisfy
ell(v) > 0 for a linear form ell. Then gamma lies in the affine chart {ell > 0}, its lift to S^2 is
closed, and gamma is contractible in RP^2. Hence for points off gamma the parity "inside / outside"
(even-odd rule in the chart; points on the line ell = 0 are outside) is well defined, and any path
between points of different parity crosses gamma. Self-intersecting polygons are allowed.

Suppose gamma avoids the curve C = {H = 0} and H has constant sign s on gamma. If two witnesses
w_i, w_j with sign H(w) = -s have different parity, they lie in different components of RP^2 \\ C:
a path within one component (where H has sign -s) cannot meet gamma. Witnesses of different signs
lie in different components trivially.

If m witnesses are pairwise separated, RP^2 \\ C has at least m components. A smooth curve of even
degree with k ovals has exactly k + 1 complementary components, so k >= m - 1.

Polygons
--------
`oval_polygon` builds, for a witness w inside an oval, a polygon close to the level curve
H = -eps sign H(w) around the oval. It works in the gnomonic chart centred at w and uses contourpy
for the level curve. The vertices are rounded to integer homogeneous coordinates, and each segment
is certified exactly (conncomp.certify.certify_segment), subdividing along the level curve where
a segment fails.
"""

from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations

import contourpy
import numpy as np
import sympy as sp

from conncomp.certify import certify_path, certify_segment

# ---------------------------------------------------------------------------
# Exact parity of a point with respect to a polygon
# ---------------------------------------------------------------------------


def _sign(H, w):
    """Exact sign of the form H (sympy Poly) at the rational point w."""
    val = sp.Rational(H(*w))
    return 1 if val > 0 else -1 if val < 0 else 0


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def chart_basis(ell):
    """Integer vectors e1, e2 completing ell to a basis of R^3 (coordinates on the plane ell = 1)."""
    k = min(range(3), key=lambda i: abs(ell[i]))
    axis = tuple(int(i == k) for i in range(3))
    e1 = _cross(ell, axis)
    return e1, _cross(ell, e1)


def _chart_point(ell, e1, e2, p):
    lp = _dot(ell, p)
    return Fraction(_dot(e1, p), lp), Fraction(_dot(e2, p), lp)


def parity(polygon, ell, w):
    """1 if w lies inside the closed polygon (even-odd rule in the chart ell > 0), else 0. Exact.

    Assumes ell > 0 at all vertices (checked by `check_path`) and that w is not on the polygon.
    """
    if _dot(ell, w) == 0:
        return 0
    e1, e2 = chart_basis(ell)
    pts = [_chart_point(ell, e1, e2, v) for v in polygon]
    px, py = _chart_point(ell, e1, e2, w)
    inside = False
    for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]):
        if (ay > py) != (by > py) and px < ax + (py - ay) * (bx - ax) / (by - ay):
            inside = not inside
    return int(inside)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@dataclass
class PathCheck:
    ok: bool
    sign: int = 0  # sign of H along the polygon
    reason: str = ""


def check_path(H, polygon, ell):
    """Exactly check that the closed polygon lies in the chart ell > 0 and avoids H = 0 with constant sign."""
    if len(polygon) < 3:
        return PathCheck(False, reason="fewer than 3 vertices")
    if any(_dot(ell, v) <= 0 for v in polygon):
        return PathCheck(False, reason="not contained in the chart ell > 0")
    certs = certify_path(H, polygon, closed=True)
    if not all(c.ok for c in certs):
        return PathCheck(False, reason=f"{sum(not c.ok for c in certs)} segments meet the curve")
    signs = {c.sign for c in certs}
    if len(signs) != 1:
        return PathCheck(False, reason="sign changes along the polygon")
    return PathCheck(True, signs.pop())


def _max_clique(m, adjacent):
    """Largest set of indices that are pairwise adjacent (brute force; m is small)."""
    best = []

    def grow(clique, candidates):
        nonlocal best
        if len(clique) + len(candidates) <= len(best):
            return
        if not candidates:
            best = list(clique)
            return
        v, rest = candidates[0], candidates[1:]
        grow(clique + [v], [u for u in rest if adjacent(v, u)])
        grow(clique, rest)

    grow([], list(range(m)))
    return best


@dataclass
class SeparationCertificate:
    witnesses: list
    signs: list
    paths: list  # [{"polygon": [...], "ell": [...], "ok": bool, "sign": int, "reason": str}]
    pairs: dict  # "i,j" -> "sign" | index of the separating path | None
    separated: list  # indices of a largest set of pairwise separated witnesses
    lower_bound: int  # len(separated) - 1 (number of ovals, if the curve is smooth of even degree)
    complete: bool  # all pairs separated


def certify_separation(H, witnesses, paths):
    """Exactly verify which pairs of witnesses lie in different components of RP^2 minus {H = 0}.

    witnesses: integer homogeneous points; paths: list of (polygon, ell) with ell a linear form
    positive at all vertices. See the module docstring for the criterion.
    """
    witnesses = [[int(v) for v in w] for w in witnesses]
    signs = [_sign(H, w) for w in witnesses]
    checks = [check_path(H, poly, ell) for poly, ell in paths]
    par = [[parity(poly, ell, w) for w in witnesses] if chk.ok else None for (poly, ell), chk in zip(paths, checks)]

    pairs = {}
    for i, j in combinations(range(len(witnesses)), 2):
        if signs[i] == 0 or signs[j] == 0:
            how = None
        elif signs[i] != signs[j]:
            how = "sign"
        else:
            how = next(
                (k for k, chk in enumerate(checks) if chk.ok and chk.sign == -signs[i] and par[k][i] != par[k][j]),
                None,
            )
        pairs[f"{i},{j}"] = how

    sep = _max_clique(len(witnesses), lambda a, b: pairs[f"{min(a, b)},{max(a, b)}"] is not None)
    return SeparationCertificate(
        witnesses=witnesses,
        signs=signs,
        paths=[
            {"polygon": [list(map(int, v)) for v in poly], "ell": list(map(int, ell)), "ok": chk.ok,
             "sign": chk.sign, "reason": chk.reason}
            for (poly, ell), chk in zip(paths, checks)
        ],
        pairs=pairs,
        separated=sep,
        lower_bound=len(sep) - 1,
        complete=all(v is not None for v in pairs.values()),
    )


# ---------------------------------------------------------------------------
# Construction of polygons around ovals
# ---------------------------------------------------------------------------


def _form_evaluator(H):
    terms = [(e, float(c)) for e, c in H.as_dict().items()]

    def f(P):
        P = P / np.linalg.norm(P, axis=-1, keepdims=True)
        out = np.zeros(P.shape[:-1])
        for (i, j, k), c in terms:
            out += c * P[..., 0] ** i * P[..., 1] ** j * P[..., 2] ** k
        return out

    return f


def _orthonormal_frame(w):
    wf = np.asarray(w, dtype=float)
    wf /= np.linalg.norm(wf)
    a = np.eye(3)[np.argmin(np.abs(wf))]
    e1 = np.cross(wf, a)
    e1 /= np.linalg.norm(e1)
    return wf, e1, np.cross(wf, e1)


def _flood(mask, start):
    """Boolean mask of the 4-connected component of `start` in `mask`."""
    comp = np.zeros_like(mask)
    comp[start] = True
    queue = deque([start])
    n0, n1 = mask.shape
    while queue:
        i, j = queue.popleft()
        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= a < n0 and 0 <= b < n1 and mask[a, b] and not comp[a, b]:
                comp[a, b] = True
                queue.append((a, b))
    return comp


def _inside_float(poly, p):
    x, y = p
    inside = False
    for (ax, ay), (bx, by) in zip(poly, np.roll(poly, -1, axis=0)):
        if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
            inside = not inside
    return inside


def _area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _certify_contour(H, verts, sign, coarse=16):
    """Certified sub-polygon of the closed vertex cycle `verts`, refining where segments fail."""
    m = len(verts)
    cuts = sorted({int(round(i * m / coarse)) % m for i in range(coarse)})

    def run(a, b):  # indices a < b on the cycle (b may equal m, meaning vertex 0)
        c = certify_segment(H, verts[a], verts[b % m])
        if c.ok and c.sign == sign:
            return [a]
        if b - a <= 1:
            return None
        mid = (a + b) // 2
        left = run(a, mid)
        right = run(mid, b) if left is not None else None
        return None if right is None else left + right

    out = []
    for a, b in zip(cuts, cuts[1:] + [cuts[0] + m]):
        part = run(a, b)
        if part is None:
            return None
        out += part
    return [verts[i] for i in out]


def oval_polygon(H, w, others=(), grids=(200, 400, 800), eps_steps=14, scale=2**24, max_radius=8.0):
    """A certified polygon around the oval containing the witness w, or None.

    The polygon follows the level curve H = -eps sign H(w) (just outside the oval) in the gnomonic
    chart centred at w. It must contain w and none of the `others` (other witnesses of the same
    sign), which is checked exactly. Returns (polygon, ell) with ell = w, or None.
    """
    f = _form_evaluator(H)
    s_in = _sign(H, w)
    wf, e1, e2 = _orthonormal_frame(w)

    def field_on(u, v):
        U, V = np.meshgrid(u, v)  # U[j, i] = u[i], V[j, i] = v[j]
        P = wf + U[..., None] * e1 + V[..., None] * e2
        return s_in * f(P)

    def chart(p):
        p = np.asarray(p, dtype=float)
        d = p @ wf
        return None if d <= 0 else (p @ e1 / d, p @ e2 / d)

    others_2d = [c for c in (chart(o) for o in others) if c is not None]

    # 1. a window containing the component of w
    r, n0 = 0.05, 201
    while True:
        u = np.linspace(-r, r, n0)
        F = field_on(u, u)
        comp = _flood(F > 0, (n0 // 2, n0 // 2))
        if comp[0, :].any() or comp[-1, :].any() or comp[:, 0].any() or comp[:, -1].any():
            r *= 2
            if r > max_radius:
                return None
            continue
        break
    rows, cols = np.nonzero(comp)
    (v0, v1), (u0, u1) = (u[rows.min()], u[rows.max()]), (u[cols.min()], u[cols.max()])
    cu, cv, half = (u0 + u1) / 2, (v0 + v1) / 2, 0.75 * max(u1 - u0, v1 - v0) + 2 * (2 * r / n0)

    # 2. level curves just outside the oval, at increasing resolution and decreasing eps
    fw = float(s_in * f(wf[None, :])[0])
    for n in grids:
        u = np.linspace(cu - half, cu + half, n)
        v = np.linspace(cv - half, cv + half, n)
        F = field_on(u, v)
        gen = contourpy.contour_generator(u, v, F)
        for k in range(1, eps_steps + 1):
            eps = fw * 2.0 ** (-k)
            cands = []
            for line in gen.lines(-eps):
                if len(line) < 4 or not np.allclose(line[0], line[-1]):
                    continue
                if _inside_float(line[:-1], (0.0, 0.0)) and not any(_inside_float(line[:-1], o) for o in others_2d):
                    cands.append(line[:-1])
            if not cands:
                continue
            line = min(cands, key=_area)
            verts = []
            for a, b in line:
                q = tuple(int(c) for c in np.rint(scale * (wf + a * e1 + b * e2)))
                if not verts or q != verts[-1]:
                    verts.append(q)
            if len(verts) > 1 and verts[0] == verts[-1]:
                verts.pop()
            poly = _certify_contour(H, verts, -s_in)
            ell = tuple(int(c) for c in w)
            if (
                poly is not None
                and len(poly) >= 3
                and all(_dot(ell, q) > 0 for q in poly)
                and parity(poly, ell, w) == 1
                and not any(parity(poly, ell, o) for o in others)
            ):
                return poly, ell
    return None


@dataclass
class LowerBoundResult:
    certificate: SeparationCertificate
    failed: list = field(default_factory=list)  # witnesses for which no polygon was found


def certify_lower_bound(H, witnesses, **kwargs):
    """Build a polygon around the oval of each witness that shares its sign with another witness,
    then verify the separation exactly (see `certify_separation`)."""
    witnesses = [[int(v) for v in w] for w in witnesses]
    signs = [_sign(H, w) for w in witnesses]
    paths, failed = [], []
    for i, w in enumerate(witnesses):
        same = [o for j, o in enumerate(witnesses) if j != i and signs[j] == signs[i]]
        if not same:
            continue
        res = oval_polygon(H, w, same, **kwargs)
        if res is None:
            failed.append(i)
        else:
            paths.append(res)
    return LowerBoundResult(certify_separation(H, witnesses, paths), failed)
