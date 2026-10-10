#!/usr/bin/env python3
"""Plots of the certified integer quartics of each nesting type (scripts/quartic_types.py).

One image per form (sign regions, the curve, separating polygons, witnesses, the base point O with its loop) and one
overview per mode, in figures/quartic_types/. The full certificates are read from pools/quartic_<mode>/cert_*.json;
without them only the curve is drawn (from data/quartic_types/<mode>.json).

The stereographic chart of conncomp.grid: the unit disk is a copy of RP^2 (antipodal points of the circle identified).

    uv run python scripts/plot_quartic_types.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from conncomp.plotting import plot_certificate  # noqa: E402

OUT = Path("figures/quartic_types")
TYPES = ["0", "1", "2", "1<1>", "3", "4"]


def tag(t):
    return t.replace("<", "(").replace(">", ")")


def plot_cert_dict(mode, rep, cert):
    """A dict in the format of conncomp.plotting.plot_certificate."""
    curve = rep["hessian"] if mode == "hessian" else rep["f"]
    d = {"hessian": curve, "hessian_deg": 4, "deg": 4, "approximation": {"height": rep["height"]},
         "ovals_grid": {"0": 0}, "proven_ovals": cert.get("upper_bound") if cert else None}
    if cert:
        lo = cert["lower"]
        d["lower"] = {"paths": lo["paths"]}
        d["witnesses"] = [{"point": w, "sign": s} for w, s in zip(lo["witnesses"], lo["signs"])]
        d["upper"] = {"loop": cert["upper"]["loop"], "base_point": cert["upper"]["base_point"]}
    return d


def title(mode, rep, short=False):
    what = "H(f)" if mode == "hessian" else "f"
    s = f"type {rep['type']}, height {rep['height']}"
    kind = "Hessian of a quartic" if mode == "hessian" else "Plane quartic"
    return s if short else f"{kind} {what} = 0: {s} (certified)"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ext = (-1.05, 1.05, -1.05, 1.05)
    for mode in ("plain", "hessian"):
        reps = json.loads(Path(f"data/quartic_types/{mode}.json").read_text())["representatives"]
        fig_all, axes = plt.subplots(len(TYPES), 3, figsize=(12, 4 * len(TYPES)))
        for r, t in enumerate(TYPES):
            for c in range(3):
                ax = axes[r, c]
                if c >= len(reps.get(t, [])):
                    ax.axis("off")
                    continue
                rep = reps[t][c]
                p = Path(f"pools/quartic_{mode}/cert_{tag(t)}_{c + 1}.json")
                cert = json.loads(p.read_text())["certificate"] if p.exists() else None
                d = plot_cert_dict(mode, rep, cert)
                fig, ax1 = plt.subplots(figsize=(8, 8))
                plot_certificate(d, width=800, extent=ext, ax=ax1)
                ax1.set_title(title(mode, rep), fontsize=10)
                expr = rep["hessian_affine"] if mode == "hessian" else rep["f_affine"]
                lab = f"f = {rep['f_affine']}" + (f"\nH = {expr}" if mode == "hessian" else "")
                fig.text(0.5, 0.01, lab.replace("**", "^").replace("*", ""), ha="center", fontsize=7, wrap=True)
                fig.savefig(OUT / f"{mode}_{tag(t)}_{c + 1}.png", dpi=120, bbox_inches="tight")
                plt.close(fig)
                plot_certificate(d, width=400, extent=ext, ax=ax, show_loop=False)
                ax.set_title(title(mode, rep, short=True), fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
        fig_all.suptitle(f"{'Hessians of quartic polynomials' if mode == 'hessian' else 'Plane quartics'}: "
                         "certified integer forms of each nesting type (stereographic chart; unit disk = RP²)",
                         fontsize=12)
        fig_all.tight_layout(rect=(0, 0, 1, 0.98))
        fig_all.savefig(OUT / f"{mode}_overview.png", dpi=100)
        plt.close(fig_all)
        print(f"{mode}: done")


if __name__ == "__main__":
    main()
