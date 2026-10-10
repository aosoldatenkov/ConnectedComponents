#!/usr/bin/env python3
"""Export the certificates of the nesting-type scans (pools/*/cert_*.json, written by scripts/quartic_types.py and
scripts/sextic_types.py) in the format of the standalone verifier (verify/verify_certificate.py).

    uv run python scripts/export_type_certificates.py
    uv run python verify/verify_certificate.py -q data/*_types/certificates/*.json
"""

import json
from pathlib import Path

RUNS = [  # (pools directory, degree of f, curve kind, output directory, file prefix)
    ("pools/quartic_plain", 4, "plain", "data/quartic_types/certificates", "plain"),
    ("pools/quartic_hessian", 4, "hessian", "data/quartic_types/certificates", "hessian"),
    ("pools/sextic_plain", 6, "plain", "data/sextic_types/certificates", "plain"),
]


def export(src, deg, curve):
    d = json.loads(Path(src).read_text())
    rep, cert = d["rep"], d["certificate"]
    lo, up = cert["lower"], cert["upper"]
    out = {"deg": deg, "curve": curve, "f": rep["f"], "type": rep["type"], "proven_ovals": cert["upper_bound"]}
    if curve == "hessian":
        out["hessian_deg"] = 2 * deg - 4
        out["hessian"] = rep["hessian"]
    out["upper"] = {"base_point": up["base_point"], "M": up["M"], "loop": up["loop"]}
    if "upper_interior" in cert:
        ui = cert["upper_interior"]
        out["upper_interior"] = {"M": ui["pencil"]["M"], "line_point": ui["line_point"]}
    out["witnesses"] = [{"point": w} for w in lo["witnesses"]]
    out["lower"] = {"paths": [{"polygon": p["polygon"], "ell": p["ell"]} for p in lo["paths"] if p["ok"]]}
    return out


def main():
    for src_dir, deg, curve, out_dir, prefix in RUNS:
        files = sorted(Path(src_dir).glob("cert_*.json"))
        if not files:
            continue
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob(f"{prefix}_*.json"):
            old.unlink()
        for f in files:
            (out / f"{prefix}_{f.stem[len('cert_'):]}.json").write_text(json.dumps(export(f, deg, curve)))
        print(f"{src_dir}: {len(files)} certificates -> {out_dir}")


if __name__ == "__main__":
    main()
