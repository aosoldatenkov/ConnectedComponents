#!/usr/bin/env python3
"""Plots of the certified integer sextics of each nesting type (scripts/sextic_types.py).

One image per form (sign regions, the curve, separating polygons, witnesses, the base point O with its loop) and an
overview with the first form of each type; the overview goes to figures/sextic_types/ (tracked), the single images to
pools/figures/sextic_types/ (untracked). Full certificates are read from
pools/sextic_plain/cert_*.json.

    uv run python scripts/plot_sextic_types.py
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from conncomp.plotting import plot_certificate  # noqa: E402

OUT = Path("figures/sextic_types")  # overviews (tracked)
FORMS = Path("pools/figures/sextic_types")  # one image per form (untracked)
POOLS = Path("pools/sextic_plain")
EXT = (-1.05, 1.05, -1.05, 1.05)


def tag(t):
    return t.replace(" u ", "u").replace("<", "(").replace(">", ")")


def cert_dict(rep, cert):
    d = {"hessian": rep["f"], "hessian_deg": 6, "deg": 6, "approximation": {"height": rep["height"]},
         "ovals_grid": {"0": 0}, "proven_ovals": rep["ovals"]}
    if cert:
        lo = cert["lower"]
        d["lower"] = {"paths": lo["paths"]}
        d["witnesses"] = [{"point": w, "sign": s} for w, s in zip(lo["witnesses"], lo["signs"])]
        d["upper"] = {"loop": cert["upper"]["loop"], "base_point": cert["upper"]["base_point"]}
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FORMS.mkdir(parents=True, exist_ok=True)
    reps = json.loads(Path("data/sextic_types/plain.json").read_text())["representatives"]
    types = sorted((t for t in reps if reps[t]), key=lambda t: (reps[t][0]["ovals"], t))
    for t in types:
        for i, rep in enumerate(reps[t]):
            p = POOLS / f"cert_{tag(t)}_{i + 1}.json"
            cert = json.loads(p.read_text())["certificate"] if p.exists() else None
            fig, ax = plt.subplots(figsize=(8, 8))
            plot_certificate(cert_dict(rep, cert), width=900, extent=EXT, ax=ax)
            ax.set_title(f"Plane sextic f = 0: type {t} ({rep['ovals']} ovals), height {rep['height']} (certified)",
                         fontsize=10)
            fig.text(0.5, 0.01, "f = " + rep["f_affine"].replace("**", "^").replace("*", ""), ha="center",
                     fontsize=6, wrap=True)
            fig.savefig(FORMS / f"plain_{tag(t)}_{i + 1}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
    cols = 5
    rows = math.ceil(len(types) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows), squeeze=False)
    for n, ax in enumerate(axes.flat):
        if n >= len(types):
            ax.axis("off")
            continue
        rep = reps[types[n]][0]
        plot_certificate(cert_dict(rep, None), width=500, extent=EXT, ax=ax, show_loop=False)
        ax.set_title(f"{types[n]} (height {rep['height']})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Plane sextics: certified integer forms of each nesting type found "
                 "(stereographic chart; unit disk = RP²)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT / "plain_overview.png", dpi=100)
    plt.close(fig)
    print(f"{sum(len(reps[t]) for t in types)} forms, {len(types)} types")


if __name__ == "__main__":
    main()
