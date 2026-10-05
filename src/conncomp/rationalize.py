"""Rational approximation of candidate forms, as the first step of certifying oval counts.

Given a floating-point form f of degree d whose Hessian curve has k ovals on the grid, we
look for a form with small integer coefficients whose exact Hessian H has the same grid count.
Then we pick a witness point in each sign region of H and check its sign with exact
integer arithmetic. The result is saved as a JSON "certificate candidate" for the
symbolic verification step.

The approximation is applied to f, not to H, so that the result is still a Hessian.

    uv run conncomp-rationalize data/cache_H_deg5_20261004-141957.txt --deg 5
"""

import argparse
import json
import math
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
import sympy as sp
import torch

from conncomp import DEVICE, DTYPE
from conncomp.components import component_labels
from conncomp.hessian import hessian_map
from conncomp.io import load_coefs
from conncomp.polynomials import monomials, to_sympy
from conncomp.scan import Experiment

# ---------------------------------------------------------------------------
# Continued fractions and candidate generation
# ---------------------------------------------------------------------------


def convergents(x):
    """Continued fraction convergents p/q of the exact rational value of `x` (float, int or Fraction)."""
    x = Fraction(x)
    p0, q0, p1, q1 = 0, 1, 1, 0
    while True:
        a = math.floor(x)
        p0, q0, p1, q1 = p1, q1, a * p1 + p0, a * q1 + q0
        yield Fraction(p1, q1)
        if x == a:
            return
        x = 1 / (x - a)


def cf_approx(x, tol):
    """The first continued fraction convergent of `x` within `tol` of it."""
    for r in convergents(x):
        if abs(r - Fraction(x)) <= tol:
            return r


def integer_vector(rationals):
    """Primitive integer vector proportional to a vector of Fractions."""
    den = math.lcm(*(r.denominator for r in rationals))
    ints = [int(r * den) for r in rationals]
    g = math.gcd(*ints)
    return [v // g for v in ints] if g > 1 else ints


def candidates(c, mode="cf", max_scale=2000, tols=None):
    """Integer approximations of the coefficient vector `c`, ordered by increasing height.

    mode "cf": each coefficient of c/max|c| is replaced by its first continued fraction
        convergent within tol, for tol = 10^-1, 10^-1.25, ..., 10^-12; then the
        denominators are cleared.
    mode "round": round(s * c/max|c|) for s = 1, ..., max_scale (simultaneous approximation
        with a common denominator; usually gives smaller integers).
    Yields (integer vector, parameter) with the parameter being tol or s.
    """
    c = np.asarray(c, dtype=np.float64)
    c = c / np.abs(c).max()
    seen = set()
    if mode == "cf":
        for tol in tols or [10 ** (-k / 4) for k in range(4, 49)]:
            ints = tuple(integer_vector([cf_approx(float(x), tol) for x in c]))
            if any(ints) and ints not in seen:
                seen.add(ints)
                yield list(ints), tol
    elif mode == "round":
        for s in range(1, max_scale + 1):
            ints = tuple(int(v) for v in np.rint(s * c))
            g = math.gcd(*ints)
            if g == 0:
                continue
            ints = tuple(v // g for v in ints)
            if ints not in seen:
                seen.add(ints)
                yield list(ints), s
    else:
        raise ValueError(f"unknown mode {mode!r}")


# ---------------------------------------------------------------------------
# Exact Hessians and grid counts of exact curves
# ---------------------------------------------------------------------------


def exact_hessian(deg, coefs):
    """Primitive integer coefficient vector of H(f) = f_xx f_yy - f_xy^2 for an integer form f.

    Coefficients are listed in the order of monomials(2*deg - 4).
    """
    hmap = hessian_map(deg)
    out = []
    for m in monomials(2 * deg - 4):
        total = 0
        for n, factor in hmap.get(m, {}).items():
            term = int(factor)
            for ci, e in zip(coefs, n):
                if e:
                    term *= ci**e
            total += term
        out.append(total)
    g = math.gcd(*out)
    return [v // g for v in out] if g > 1 else out


def eval_form(coefs, deg, point):
    """Exact value of a form with integer coefficients at an integer point (X, Y, Z)."""
    X, Y, Z = point
    return sum(c * X**i * Y**j * Z**k for c, (i, j, k) in zip(coefs, monomials(deg)) if c)


def _to_float_columns(int_vectors):
    """Batch of integer vectors -> (D, N) float tensor, each column scaled to max |entry| = 1."""
    cols = []
    for v in int_vectors:
        M = max(abs(x) for x in v)
        cols.append([float(Fraction(x, M)) for x in v])
    return torch.tensor(cols, dtype=DTYPE, device=DEVICE).T


@lru_cache(maxsize=None)
def _experiment(deg, width, use_hessian):
    return Experiment(deg, width, use_hessian)


def oval_counts(curve_coefs, curve_deg, width, min_size=3):
    """Grid oval counts (components - 1) of curves given by exact integer coefficient vectors."""
    exp = _experiment(curve_deg, width, False)
    return [n - 1 for n in exp.count(_to_float_columns(curve_coefs), min_size)]


def float_oval_counts(c, deg, width, min_size=3):
    """Grid oval counts of the Hessian curves of floating-point forms (columns of `c`)."""
    cc = torch.as_tensor(np.asarray(c, dtype=np.float64), device=DEVICE).reshape(len(c), -1)
    return [n - 1 for n in _experiment(deg, width, True).count(cc, min_size)]


# ---------------------------------------------------------------------------
# Approximation search
# ---------------------------------------------------------------------------


def rationalize(c, deg, widths=(200, 400, 800), mode="cf", min_size=3, max_scale=2000, chunk=256):
    """Lowest-height integer approximation of f whose Hessian has the same grid counts as f.

    The coefficients of z^d, x z^(d-1), y z^(d-1) are set to zero first (they do not affect H).

    The counts must agree at every width in `widths`. Candidates are screened in batches at
    widths[0] and the survivors are checked at the other widths. Returns a dict with keys
    f, hessian, ovals, param, height, or None if no candidate matches.
    """
    # Terms of degree <= 1 in x, y do not affect H(f); drop them to lower the height
    c = np.asarray(c, dtype=np.float64) * np.array([i + j >= 2 for i, j, _ in monomials(deg)])
    target = {w: float_oval_counts(c[:, None], deg, w, min_size)[0] for w in widths}
    hdeg = 2 * deg - 4
    gen = candidates(c, mode, max_scale)
    while True:
        batch = [x for _, x in zip(range(chunk), gen)]
        if not batch:
            return None
        hs = [exact_hessian(deg, f) for f, _ in batch]
        counts0 = oval_counts(hs, hdeg, widths[0], min_size)
        for (f, param), h, n0 in zip(batch, hs, counts0):
            if n0 != target[widths[0]]:
                continue
            if all(oval_counts([h], hdeg, w, min_size)[0] == target[w] for w in widths[1:]):
                return {
                    "f": f,
                    "hessian": h,
                    "ovals": {str(w): target[w] for w in widths},
                    "param": param,
                    "height": max(abs(x) for x in f),
                }


# ---------------------------------------------------------------------------
# Witness points
# ---------------------------------------------------------------------------


def grid_point(i, j, width):
    """Primitive integer homogeneous coordinates of the point of RP^2 at grid pixel (i, j)."""
    n = width - 1
    a, b = 2 * i - n, 2 * j - n  # chart coordinates (a/n, b/n)
    X, Y, Z = 2 * a * n, 2 * b * n, a * a + b * b - n * n
    g = math.gcd(X, Y, Z)
    return (X // g, Y // g, Z // g)


def witnesses(curve_coefs, curve_deg, width, min_size=3):
    """One witness point per sign region of the curve (regions of at least `min_size` pixels).

    For each region, takes the pixel where |H| is largest. Its exact sign is computed with integer
    arithmetic and compared to the region's sign on the grid. Returns a list of dicts with keys
    point, sign, exact_sign, pixels.
    """
    exp = _experiment(curve_deg, width, False)
    v = exp.values(_to_float_columns([curve_coefs]))[0].cpu().numpy()
    labels, sizes = component_labels(v, exp.pat)
    out = []
    absv = np.abs(v)
    for k, size in enumerate(sizes):
        if size < min_size:
            continue
        masked = np.where(labels == k, absv, -1.0)
        i, j = np.unravel_index(np.argmax(masked), masked.shape)
        p = grid_point(int(i), int(j), width)
        val = eval_form(curve_coefs, curve_deg, p)
        out.append(
            {
                "point": list(p),
                "sign": 1 if v[i, j] >= 0 else -1,
                "exact_sign": (val > 0) - (val < 0),
                "pixels": int(size),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Certificate candidates
# ---------------------------------------------------------------------------


def certificate_candidate(c, deg, widths=(200, 400, 800), mode="cf", min_size=3, max_scale=2000):
    """Run the approximation search and the witness checks. Returns a JSON-serializable dict or None."""
    res = rationalize(c, deg, widths, mode, min_size, max_scale)
    if res is None:
        return None
    hdeg = 2 * deg - 4
    wit = witnesses(res["hessian"], hdeg, max(widths), min_size)
    x, y = sp.symbols("x y")
    return {
        "deg": deg,
        "hessian_deg": hdeg,
        "ovals_grid": res["ovals"],
        "approximation": {"mode": mode, "param": res["param"], "height": res["height"]},
        "f": res["f"],
        "hessian": res["hessian"],
        "f_affine": str(sp.expand(to_sympy(res["f"], deg, x, y, 1))),
        "witnesses": wit,
        "witnesses_ok": all(w["exact_sign"] == w["sign"] for w in wit),
        "float_coefs": [float(v) for v in c],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", type=Path, help="coefficient file (one form per line)")
    p.add_argument("--deg", type=int, required=True, help="degree d of the forms f in the file")
    p.add_argument("--index", type=int, nargs="*", help="process these line indices (default: best forms)")
    p.add_argument("--limit", type=int, default=5, help="max number of best forms to process (default: 5)")
    p.add_argument("--widths", type=int, nargs="+", default=[200, 400, 800], help="grid widths that must agree")
    p.add_argument("--mode", choices=["cf", "round"], default="cf", help="approximation method (default: cf)")
    p.add_argument("--max-scale", type=int, default=2000, help="largest scale in round mode (default: 2000)")
    p.add_argument("--min-size", type=int, default=3, help="minimal component size in pixels (default: 3)")
    p.add_argument("--out-dir", type=Path, default=Path("data/certificates"), help="output directory")
    args = p.parse_args(argv)

    forms = load_coefs(args.file)
    if args.index:
        chosen = args.index
    else:
        cc = np.array(forms).T
        per_width = [float_oval_counts(cc, args.deg, w, args.min_size) for w in args.widths]
        stable = [counts[0] if len(set(counts)) == 1 else -1 for counts in zip(*per_width)]
        best = max(stable)
        chosen = [i for i, s in enumerate(stable) if s == best][: args.limit]
        print(f"{len(forms)} forms; best grid count stable across widths: {best} ovals, at lines {chosen}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i in chosen:
        cert = certificate_candidate(forms[i], args.deg, tuple(args.widths), args.mode, args.min_size, args.max_scale)
        if cert is None:
            print(f"line {i}: no approximation reproduces the grid counts")
            continue
        cert["source"] = {"file": str(args.file), "line": i}
        out = args.out_dir / f"{args.file.stem}_{i}_{args.mode}.json"
        out.write_text(json.dumps(cert, indent=1))
        a = cert["approximation"]
        print(
            f"line {i}: ovals {cert['ovals_grid']}, height {a['height']} ({args.mode} param {a['param']:.3g}), "
            f"{len(cert['witnesses'])} witnesses, exact signs {'OK' if cert['witnesses_ok'] else 'MISMATCH'} -> {out}"
        )


if __name__ == "__main__":
    main()
