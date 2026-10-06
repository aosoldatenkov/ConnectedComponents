"""Plots of sign regions on the grid, and of certified oval configurations."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from conncomp import DTYPE, DEVICE
from conncomp.polynomials import monomials
from conncomp.scan import Experiment


def plot_negative_regions(deg, width, coefs, use_hessian=True):
    """One scatter plot of the region {F < 0} per coefficient vector in `coefs`."""
    exp = Experiment(deg, width, use_hessian)
    cc = torch.as_tensor(coefs, dtype=DTYPE, device=DEVICE).reshape(len(coefs), -1).T
    v = exp.values(cc).cpu().numpy()
    figs = []
    for i in range(v.shape[0]):
        a, b = (v[i] < 0).nonzero()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(a, b, c="tab:blue", s=1)
        figs.append(fig)
    return figs


# ---------------------------------------------------------------------------
# Certificates
# ---------------------------------------------------------------------------


def _form_values(coefs, deg, P):
    """Values of the form at the unit vectors P / |P| (P of shape (..., 3))."""
    P = P / np.linalg.norm(P, axis=-1, keepdims=True)
    out = np.zeros(P.shape[:-1])
    for c, (i, j, k) in zip(coefs, monomials(deg)):
        if c:
            out += float(c) * P[..., 0] ** i * P[..., 1] ** j * P[..., 2] ** k
    return out


def _from_chart(u, v):
    """Points of S^2 for stereographic chart coordinates (u, v) (the chart of conncomp.grid)."""
    r2 = u**2 + v**2
    return np.stack([2 * u, 2 * v, r2 - 1], axis=-1) / (r2 + 1)[..., None]


def _to_chart(P, cutoff=10.0):
    """Stereographic chart coordinates of the points P (lifted to S^2 as given); NaN near the pole."""
    P = P / np.linalg.norm(P, axis=-1, keepdims=True)
    d = 1 - P[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = P[:, :2] / d[:, None]
    uv[(np.abs(uv) > cutoff).any(axis=1)] = np.nan
    return uv


def _densify(vertices, closed, samples):
    """Points along the geodesic segments (convex combinations) between consecutive vertices."""
    V = np.asarray([[float(c) for c in v] for v in vertices])
    V /= np.linalg.norm(V, axis=1, keepdims=True)  # same rays, comparable scales
    pairs = list(zip(V, np.roll(V, -1, axis=0))) if closed else list(zip(V[:-1], V[1:]))
    t = np.linspace(0, 1, samples, endpoint=False)[:, None]
    pts = [a * (1 - t) + b * t for a, b in pairs]
    pts.append((V[0] if closed else V[-1])[None, :])
    return np.concatenate(pts)


def _plot_lifts(ax, P, **kw):
    """Plot both lifts p and -p of a path in RP^2, breaking it where it jumps in the chart."""
    for sgn in (1, -1):
        uv = _to_chart(sgn * P)
        jump = np.linalg.norm(np.diff(uv, axis=0), axis=1) > 0.5
        uv = np.insert(uv, np.nonzero(jump)[0] + 1, np.nan, axis=0)
        ax.plot(uv[:, 0], uv[:, 1], **kw)
        kw.pop("label", None)


def plot_certificate(cert, width=800, extent=(-1.0, 1.0, -1.0, 1.0), ax=None, show_loop=True, samples=24):
    """Plot a certificate: sign regions of H, the curve, the separating polygons, witnesses and base point.

    Uses the stereographic chart of conncomp.grid: the unit circle is the line z = 0 and antipodal
    points of it are identified; points outside the unit disk duplicate points inside it. Polygons,
    the base point loop and the witnesses are drawn with both lifts. `extent` = (u0, u1, v0, v1)
    zooms into a part of the chart.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    u0, u1, v0, v1 = extent
    U, V = np.meshgrid(np.linspace(u0, u1, width), np.linspace(v0, v1, width), indexing="xy")
    Hv = _form_values(cert["hessian"], cert["hessian_deg"], _from_chart(U, V))
    ax.contourf(U, V, np.sign(Hv), levels=[-2, 0, 2], colors=["#c6dbef", "white"])
    ax.contour(U, V, Hv, levels=[0], colors="black", linewidths=0.8)
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color="gray", lw=0.6, ls=":")

    colors = plt.cm.tab10.colors
    for k, path in enumerate(cert.get("lower", {}).get("paths", [])):
        if path.get("ok", True):
            P = _densify(path["polygon"], closed=True, samples=samples)
            _plot_lifts(ax, P, color=colors[k % 10], lw=1.2)
    for k, w in enumerate(cert.get("witnesses", [])):
        W = np.asarray([w["point"]], dtype=float)
        for sgn in (1, -1):
            uv = _to_chart(sgn * W)[0]
            if np.all(np.isfinite(uv)):
                ax.plot(*uv, "o", ms=3, color="red" if w["sign"] > 0 else "navy")
                ax.annotate(str(k), uv, xytext=(3, 3), textcoords="offset points", fontsize=7)
    up = cert.get("upper", {})
    if show_loop and up.get("loop"):
        P = _densify([[float(v) for v in p] for p in up["loop"]], closed=False, samples=samples)
        _plot_lifts(ax, P, color="black", lw=1.0, ls="--", label="base point loop")
        O = np.asarray([[float(v) for v in up["base_point"]]])
        for sgn in (1, -1):
            uv = _to_chart(sgn * O)[0]
            if np.all(np.isfinite(uv)):
                ax.plot(*uv, "*", ms=12, color="gold", mec="black", label="base point O" if sgn > 0 else None)

    ax.set_xlim(u0, u1)
    ax.set_ylim(v0, v1)
    ax.set_aspect("equal")
    proven = cert.get("proven_ovals")
    title = f"Hessian of a degree {cert['deg']} form, height {cert['approximation']['height']}: "
    grid = max(cert["ovals_grid"].values())
    title += f"exactly {proven} ovals (certified)" if proven is not None else f"{grid} ovals (grid count)"
    ax.set_title(title, fontsize=10)
    if show_loop and up.get("loop"):
        ax.legend(loc="lower left", fontsize=8)
    return ax


def main(argv=None):
    p = argparse.ArgumentParser(description="Plot a certificate: ovals, separating polygons, witnesses, base point")
    p.add_argument("file", type=Path, help="certificate JSON (after conncomp-certify)")
    p.add_argument("-o", "--out", type=Path, default=None, help="output image (default: show the plot)")
    p.add_argument("--width", type=int, default=800, help="grid resolution (default: 800)")
    p.add_argument("--zoom", type=float, nargs=4, default=None, metavar=("U0", "U1", "V0", "V1"),
                   help="chart window to show (default: -1 1 -1 1)")
    p.add_argument("--no-loop", action="store_true", help="do not draw the base point loop")
    args = p.parse_args(argv)
    cert = json.loads(args.file.read_text())
    extent = tuple(args.zoom) if args.zoom else (-1.0, 1.0, -1.0, 1.0)
    ax = plot_certificate(cert, args.width, extent, show_loop=not args.no_loop)
    if args.out:
        ax.figure.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"saved {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
