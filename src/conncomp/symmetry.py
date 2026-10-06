"""Polynomials with prescribed symmetry, and integer bases of the corresponding search spaces.

For a linear map g of the plane, the Hessian satisfies H(f o g) = det(g)^2 H(f) o g. Hence if
f o g = chi(g) f for a sign character chi: G -> {+1, -1} of a finite group G (so det g = +-1), then
H(f) o g = H(f): the Hessian curve of a semi-invariant f is G-invariant. Both invariant (chi = 1)
and anti-invariant f are useful.

A symmetry is given by generators of G (2x2 matrices acting on (x, y)) and the sign chi(g) of each
generator. G acts on each block of forms x^i y^j z^k with fixed k separately, and the blocks with
i + j <= 1 do not affect the Hessian; they are left out. The space of semi-invariant forms of
degree d is described by an integer matrix B whose columns are coefficient vectors (in the order of
conncomp.polynomials.monomials(d)); the search samples f = B a.

Presets (spec strings, optionally followed by ':' and comma-separated signs, one per generator):
    x       x -> -x                         y      y -> -y
    xy      x -> -x, y -> -y (Klein group)  diag   x <-> y
    Cn      rotation by 2 pi / n            Dn     rotation by 2 pi / n, and y -> -y
e.g. "x", "x:-1" (anti-invariant under x -> -x), "D3", "D3:1,-1", "C4:-1".
For Cn and Dn the integer basis is built from z^a zbar^b (zeta = x + i y), since the rotation
matrices have irrational entries for n != 1, 2, 4.
"""

import math
from dataclasses import dataclass

import numpy as np
import sympy as sp

from conncomp.polynomials import monomials

x, y = sp.symbols("x y")


@dataclass
class Symmetry:
    name: str
    generators: list  # 2x2 sympy matrices acting on (x, y)
    character: list  # chi(g) = +-1 for each generator
    basis: np.ndarray  # integer matrix (D, k): columns span the semi-invariant forms of degree `deg`
    deg: int

    @property
    def dim(self):
        return self.basis.shape[1]

    def _xy_degrees(self):
        mons = monomials(self.deg)
        return {mons[r][0] + mons[r][1] for r in range(len(mons)) if self.basis[r].any()}

    @property
    def effective_degree(self):
        """Largest x, y-degree occurring. If < deg, then f has lower degree and its degree-(2 deg - 4)
        Hessian form contains the line at infinity z = 0 as a multiple component."""
        return max(self._xy_degrees())

    @property
    def singular_at_origin(self):
        """True if no quadratic terms occur: f vanishes to order 3 at the origin, so H is singular there."""
        return 2 not in self._xy_degrees()

    def warnings(self):
        out = []
        if self.effective_degree < self.deg:
            out.append(f"effective degree {self.effective_degree} < {self.deg}: H contains z = 0 as a multiple line")
        if self.singular_at_origin:
            out.append("no quadratic terms: H is singular at the origin")
        return out

    def describe(self):
        signs = ", ".join(f"{'+' if c > 0 else '-'}" for c in self.character)
        text = f"{self.name} (character {signs}): {self.dim}-dimensional space of degree {self.deg} forms"
        return text + "".join(f"; WARNING: {w}" for w in self.warnings())

    def polynomials(self, affine=True):
        """The basis as sympy polynomials in x, y (affine, z = 1)."""
        mons = monomials(self.deg)
        return [sp.expand(sum(int(c) * x**i * y**j for c, (i, j, _) in zip(col, mons))) for col in self.basis.T]


def _coeff_vector(poly, deg):
    """Coefficient vector (over monomials(deg)) of a polynomial in x, y of degree <= deg, homogenized."""
    d = sp.Poly(sp.expand(poly), x, y).as_dict()
    return [int(d.get((i, j), 0)) for i, j, _ in monomials(deg)]


def _primitive(v):
    g = math.gcd(*v)
    if g == 0:
        return v
    s = -1 if next(c for c in v if c) < 0 else 1
    return [s * c // g for c in v]


def _rotation_family(n, deg, with_reflection, chi_rot, chi_ref, min_deg=2):
    """Integer basis for C_n (or D_n with the reflection y -> -y) semi-invariants (zeta = x + i y)."""
    zeta, zbar = x + sp.I * y, x - sp.I * y
    cols = []
    for m in range(min_deg, deg + 1):
        for b in range(m // 2 + 1):
            a = m - b
            q = a - b
            if chi_rot == 1 and q % n:
                continue
            if chi_rot == -1 and (n % 2 or (q - n // 2) % n):
                continue
            w = sp.Poly(sp.expand(zeta**a * zbar**b), x, y).as_dict()  # Gaussian integer coefficients
            re = sum(sp.re(c) * x**i * y**j for (i, j), c in w.items())
            im = sum(sp.im(c) * x**i * y**j for (i, j), c in w.items())
            parts = []
            if not with_reflection or chi_ref == 1:
                parts.append(re)
            if (not with_reflection or chi_ref == -1) and q != 0:
                parts.append(im)
            for p in parts:
                v = _coeff_vector(sp.expand(p), deg)
                if any(v):
                    cols.append(_primitive(v))
    return cols


def _action_matrix(g, deg):
    """Matrix of f -> f o g on coefficient vectors of degree-`deg` forms (g acts on x, y; z fixed)."""
    mons = monomials(deg)
    xs = g[0, 0] * x + g[0, 1] * y
    ys = g[1, 0] * x + g[1, 1] * y
    cols = []
    for i, j, _ in mons:
        img = sp.Poly(sp.expand(xs**i * ys**j), x, y).as_dict()
        cols.append([img.get((a, b), 0) for a, b, _ in mons])
    return sp.Matrix(cols).T


def _linear_algebra_basis(generators, character, deg, drop_low=True):
    """Integer basis of {f : f o g = chi(g) f} (relevant blocks only) by exact linear algebra over QQ."""
    mons = monomials(deg)
    rows = []
    for g, chi in zip(generators, character):
        A = _action_matrix(g, deg) - chi * sp.eye(len(mons))
        rows.append(A)
    # drop the blocks of x, y-degree <= 1 (they do not affect the Hessian)
    for idx, (i, j, _) in enumerate(mons):
        if drop_low and i + j <= 1:
            e = sp.zeros(1, len(mons))
            e[idx] = 1
            rows.append(e)
    null = sp.Matrix.vstack(*rows).nullspace()
    cols = []
    for v in null:
        if not all(c.is_rational for c in v):
            raise ValueError("the semi-invariant space is not defined over QQ for these generators")
        den = math.lcm(*(int(sp.Rational(c).q) for c in v))
        cols.append(_primitive([int(c * den) for c in v]))
    return cols


def _rotation(n):
    c, s = sp.cos(2 * sp.pi / n), sp.sin(2 * sp.pi / n)
    return sp.Matrix([[c, -s], [s, c]])


PRESETS = {
    "x": [sp.Matrix([[-1, 0], [0, 1]])],
    "y": [sp.Matrix([[1, 0], [0, -1]])],
    "xy": [sp.Matrix([[-1, 0], [0, 1]]), sp.Matrix([[1, 0], [0, -1]])],
    "diag": [sp.Matrix([[0, 1], [1, 0]])],
}


def symmetry(spec, deg, for_hessian=True):
    """The Symmetry for a spec string such as "x", "x:-1", "D3", "D4:1,-1", "C3" (see module doc).

    With for_hessian=False the terms of x, y-degree <= 1 are kept (they matter for the curve f = 0 itself).
    """
    name, _, signs = spec.partition(":")
    name = name.strip()
    if name in PRESETS:
        gens = PRESETS[name]
    elif name[:1] in "CD" and name[1:].isdigit():
        n = int(name[1:])
        gens = [_rotation(n)] + ([sp.Matrix([[1, 0], [0, -1]])] if name[0] == "D" else [])
    else:
        raise ValueError(f"unknown symmetry {spec!r}")
    character = [int(s) for s in signs.split(",")] if signs else [1] * len(gens)
    if len(character) != len(gens) or any(c not in (1, -1) for c in character):
        raise ValueError(f"{spec!r}: give one sign +-1 per generator ({len(gens)})")
    if name in PRESETS:
        cols = _linear_algebra_basis(gens, character, deg, drop_low=for_hessian)
    else:
        n = int(name[1:])
        cols = _rotation_family(n, deg, name[0] == "D", character[0], character[1] if name[0] == "D" else 1,
                                min_deg=2 if for_hessian else 0)
    if not cols:
        raise ValueError(f"{spec!r}: no semi-invariant forms of degree {deg} affect the Hessian")
    B = np.array(cols, dtype=np.int64).T
    return Symmetry(name, gens, character, B, deg)


def is_semi_invariant(coefs, deg, sym, tol=1e-9):
    """Numerical check that the form with coefficient vector `coefs` satisfies f o g = chi(g) f."""
    c = np.asarray(coefs, dtype=float)
    for g, chi in zip(sym.generators, sym.character):
        A = np.array(_action_matrix(g, deg).evalf(), dtype=float)
        if np.abs(A @ c - chi * c).max() > tol * max(1.0, np.abs(c).max()):
            return False
    return True
