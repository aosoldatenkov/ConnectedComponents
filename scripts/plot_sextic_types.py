#!/usr/bin/env python3
"""Plots of the certified integer sextics of each nesting type (scripts/sextic_types.py).

One image per form (sign regions, the curve, separating polygons, witnesses, the base point O with its loop) and an
overview with the first form of each type, for the phase A/B scan (data/sextic_types/plain.json) and the births search
(births.json). The overview goes to figures/sextic_types/ (tracked), the single images to pools/figures/sextic_types/
(untracked). Full certificates are read from pools/sextic_plain/ and pools/sextic_births/ (cert_*.json).

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
EXT = (-1.05, 1.05, -1.05, 1.05)


def tag(t):
    return t.replace(" u ", "u").replace("<", "(").replace(">", ")")


def cert_dict(rep, cert):
    d = {"hessian": rep.get("hessian", rep["f"]), "hessian_deg": 6, "deg": 6,
         "approximation": {"height": rep["height"]}, "ovals_grid": {"0": 0}, "proven_ovals": rep["ovals"]}
    if cert:
        lo = cert["lower"]
        d["lower"] = {"paths": lo["paths"]}
        d["witnesses"] = [{"point": w, "sign": s} for w, s in zip(lo["witnesses"], lo["signs"])]
        d["upper"] = {"loop": cert["upper"]["loop"], "base_point": cert["upper"]["base_point"]}
    return d


SOURCES = [  # (representatives file, certificates directory, file prefix)
    ("data/sextic_types/plain.json", Path("pools/sextic_plain"), "plain"),
    ("data/sextic_types/births.json", Path("pools/sextic_births"), "births"),
    ("data/sextic_types/overnight.json", Path("pools/sextic_overnight/A"), "overnight"),
]


HESSIAN_SOURCES = [("data/hessian_types/overnight.json", Path("pools/hessian_overnight/A"), "overnight")]


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["plain", "hessian"], default="plain")
    kind = ap.parse_args().kind
    global OUT, FORMS, SOURCES
    if kind == "hessian":
        OUT, FORMS, SOURCES = Path("figures/hessian_types"), Path("pools/figures/hessian_types"), HESSIAN_SOURCES
    what = "Plane sextic f = 0" if kind == "plain" else "Hessian curve H(f) = 0 of a quintic f"
    OUT.mkdir(parents=True, exist_ok=True)
    FORMS.mkdir(parents=True, exist_ok=True)
    first = {}  # type -> (representative, source) for the overview
    for rep_file, pools, prefix in SOURCES:
        if not Path(rep_file).exists():
            continue
        reps = json.loads(Path(rep_file).read_text())["representatives"]
        for t in sorted((t for t in reps if reps[t]), key=lambda t: (reps[t][0]["ovals"], t)):
            first.setdefault(t, (reps[t][0], prefix))
            for i, rep in enumerate(reps[t]):
                p = pools / f"cert_{tag(t)}_{i + 1}.json"
                cert = json.loads(p.read_text())["certificate"] if p.exists() else None
                fig, ax = plt.subplots(figsize=(8, 8))
                plot_certificate(cert_dict(rep, cert), width=900, extent=EXT, ax=ax)
                ax.set_title(f"{what}: type {t} ({rep['ovals']} ovals), height {rep['height']} "
                             "(certified)", fontsize=10)
                lab = "f = " + rep["f_affine"] + (f"\nH = {rep['hessian_affine']}" if "hessian_affine" in rep else "")
                fig.text(0.5, 0.01, lab.replace("**", "^").replace("*", ""), ha="center", fontsize=6, wrap=True)
                fig.savefig(FORMS / f"{prefix}_{tag(t)}_{i + 1}.png", dpi=120, bbox_inches="tight")
                plt.close(fig)
    types = sorted(first, key=lambda t: (first[t][0]["ovals"], t))
    cols = 5
    rows = math.ceil(len(types) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows), squeeze=False)
    for n, ax in enumerate(axes.flat):
        if n >= len(types):
            ax.axis("off")
            continue
        rep, prefix = first[types[n]]
        plot_certificate(cert_dict(rep, None), width=500, extent=EXT, ax=ax, show_loop=False)
        mark = " *" if prefix in ("births", "overnight") else ""
        ax.set_title(f"{types[n]} (height {rep['height']}){mark}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    title = "Plane sextics" if kind == "plain" else "Hessian curves of quintics"
    fig.suptitle(f"{title}: certified integer forms of the {len(types)} nesting types found "
                 "(* = births search; stereographic chart, unit disk = RP²)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT / ("plain_overview.png" if kind == "plain" else "overview.png"), dpi=100)
    plt.close(fig)
    print(f"{len(types)} types")


if __name__ == "__main__":
    main()
