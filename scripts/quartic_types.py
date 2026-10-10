#!/usr/bin/env python3
"""Randomized scan of degree-4 forms: frequencies (volumes) of the six nesting types of smooth quartics in RP^2
(0, 1, 2, 1<1>, 3, 4), robust pools per type, and exactly certified integer representatives.

Modes: `plain` (the quartic curves {f = 0}) and `hessian` (the Hessian curves of quartic polynomials f(x, y), also
quartics). Forms f are drawn from two measures, alternating batches: uniform on the unit sphere of coefficients and
Kostlan (independent Gaussians with variance 4! / (i! j! k!), invariant under O(3)).

Stage `scan` (time-limited, checkpointed, resumable):
  * types on the CUDA sphere mesh FaceMesh(N) (default N = 30), counted per measure;
  * per (measure, coarse type) a uniform reservoir of forms (bottom-k random keys), used to estimate the grid error;
  * per type a pool of the most robust forms: robustness = min |critical value| of the normalized curve form on
    RP^2 (0 exactly on the discriminant), computed with conncomp.critical for buffered candidates.
Stage `finish`:
  * reservoirs reclassified at N = 200 (reference): confusion matrices, corrected volumes (parametric bootstrap);
  * per type, diverse robust pool forms rounded to small integer coefficients (type checked at N = 100 and 200),
    then certified exactly: oval count by pencil tangents (upper) and separating polygons (lower), cross-checked
    with FLINT. Nesting (2 versus 1<1>): the root region N contains the certified base point O, and the k + 1
    pairwise separated witnesses lie one in each region; with 2 ovals, N's sign is shared by two regions iff the
    ovals are nested. (Quartics with 0, 1, 3 or 4 ovals have no nests, by Bezout.)

Large outputs go to pools/<run>/ (untracked); integer representatives to data/quartic_types/<mode>.json.

    uv run python scripts/quartic_types.py --mode plain --minutes 27
    uv run python scripts/quartic_types.py --mode hessian --minutes 27
"""

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import sympy as sp
import torch

from conncomp import DEVICE, DTYPE
from conncomp.certify import certify_upper_bound, crosscheck_flint, form_poly
from conncomp.critical import FaceCharts, critical_points
from conncomp.gpu_trees import FaceMesh
from conncomp.hessian import hessian
from conncomp.polynomials import monomials, normalize, sample, to_sympy
from conncomp.rationalize import eval_form, exact_hessian, witnesses

TYPES = ["0", "1", "2", "1<1>", "3", "4", "other"]
NT = len(TYPES)
MEASURES = ["uniform", "kostlan"]
DEG = 4
D = len(monomials(DEG))
KOSTLAN = torch.tensor([math.sqrt(math.factorial(DEG) / (math.factorial(i) * math.factorial(j) * math.factorial(k)))
                        for i, j, k in monomials(DEG)], dtype=DTYPE, device=DEVICE)
LOW = torch.tensor([i + j < 2 for i, j, _ in monomials(DEG)], device=DEVICE)  # terms not affecting H(f)


def draw(measure, B):
    if measure == "uniform":
        return sample(DEG, B)
    return normalize(KOSTLAN[:, None] * torch.randn((D, B), dtype=DTYPE, device=DEVICE))


def curve(mode, f):
    return f if mode == "plain" else hessian(DEG, f)


def type_codes(t, B):
    """Index into TYPES per form, from GPUTrees (depth 1: oval interiors, depth 2: nested)."""
    d = torch.where(t.alive, t.depth, torch.zeros_like(t.depth))
    maxd = torch.zeros(B, dtype=torch.long, device=d.device).scatter_reduce_(0, t.form, d, "amax")
    n = t.n_ovals
    code = torch.full((B,), NT - 1, dtype=torch.long, device=d.device)
    for k, (cnt, dep) in enumerate(((0, 0), (1, 1), (2, 1), (2, 2), (3, 1), (4, 1))):
        code[(n == cnt) & (maxd == dep)] = k
    code[~t.ok] = NT - 1
    return code


def classify(fm, c, batch=None):
    """Type codes of the curve forms c (D, B) on the mesh fm, in batches bounded by GPU memory."""
    batch = batch or max(1, int(4e8 // fm.n_loc))
    return torch.cat([type_codes(fm.trees(c[:, s : s + batch], DEG), c[:, s : s + batch].shape[1])
                      for s in range(0, c.shape[1], batch)])


def robustness(c, charts):
    """min |critical value| of the normalized forms c (D, B) on RP^2; 0 if the Morse check fails."""
    B = c.shape[1]
    crit, morse = critical_points(c, DEG, charts=charts)
    m = torch.full((B,), float("inf"), dtype=DTYPE, device=c.device)
    m = m.scatter_reduce_(0, crit.form, crit.value.abs(), "amin")
    return torch.where(morse == 1, m, torch.zeros_like(m))


class State:
    """Counters, reservoirs and pools (all on the CPU, saved as one .npz)."""

    def __init__(self, res_k, pool_k):
        self.res_k, self.pool_k = res_k, pool_k
        self.counts = np.zeros((len(MEASURES), NT), dtype=np.int64)
        self.elapsed = 0.0
        self.res = {(m, k): (np.zeros(0), np.zeros((0, D))) for m in range(len(MEASURES)) for k in range(NT)}
        self.pool = {k: (np.zeros(0), np.zeros((0, D))) for k in range(NT)}

    def save(self, path):
        arrs = {"counts": self.counts, "elapsed": np.array(self.elapsed), "res_k": np.array(self.res_k),
                "pool_k": np.array(self.pool_k)}
        for (m, k), (u, f) in self.res.items():
            arrs[f"res_{m}_{k}_key"], arrs[f"res_{m}_{k}_f"] = u, f
        for k, (s, f) in self.pool.items():
            arrs[f"pool_{k}_score"], arrs[f"pool_{k}_f"] = s, f
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **arrs)
        tmp.replace(path)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        st = cls(int(z["res_k"]), int(z["pool_k"]))
        st.counts, st.elapsed = z["counts"], float(z["elapsed"])
        for m, k in st.res:
            st.res[(m, k)] = (z[f"res_{m}_{k}_key"], z[f"res_{m}_{k}_f"])
        for k in st.pool:
            st.pool[k] = (z[f"pool_{k}_score"], z[f"pool_{k}_f"])
        return st


def scan(args, run):
    ckpt = run / "checkpoint.npz"
    st = State.load(ckpt) if ckpt.exists() else State(args.reservoir, args.pool)
    fm, charts = FaceMesh(args.N), FaceCharts(DEG, N=40)
    buf = {k: [] for k in range(NT)}
    nbuf = np.zeros(NT, dtype=np.int64)

    def flush(k):
        if not buf[k]:
            return
        f = torch.cat([torch.as_tensor(st.pool[k][1].T, device=DEVICE)] + buf[k], dim=1)
        s = robustness(curve(args.mode, f), charts).cpu().numpy()
        keep = np.argsort(-s)[: st.pool_k]
        st.pool[k] = (s[keep], f.T.cpu().numpy()[keep])
        buf[k], nbuf[k] = [], 0

    t_start, t_ckpt, it = time.time(), time.time(), 0
    budget = args.minutes * 60 - st.elapsed
    while time.time() - t_start < budget:
        m = it % len(MEASURES)
        f = draw(MEASURES[m], args.batch)
        code = classify(fm, curve(args.mode, f), args.batch)
        st.counts[m] += torch.bincount(code, minlength=NT).cpu().numpy()
        key = torch.rand(args.batch, dtype=DTYPE, device=DEVICE)
        for k in range(NT):
            idx = (code == k).nonzero()[:, 0]
            if idx.numel() == 0:
                continue
            # reservoir: keep the res_k smallest keys
            u0, f0 = st.res[(m, k)]
            thr = u0.max() if len(u0) >= st.res_k else 2.0
            sel = idx[key[idx] < thr]
            if sel.numel():
                u = np.concatenate([u0, key[sel].cpu().numpy()])
                ff = np.concatenate([f0, f[:, sel].T.cpu().numpy()])
                o = np.argsort(u)[: st.res_k]
                st.res[(m, k)] = (u[o], ff[o])
            # pool candidates: a random subset per batch
            if k != NT - 1:
                take = idx[torch.randperm(idx.numel(), device=DEVICE)[: args.cand]]
                buf[k].append(f[:, take])
                nbuf[k] += take.numel()
                if nbuf[k] >= args.flush:
                    flush(k)
        it += 1
        if time.time() - t_ckpt > args.checkpoint * 60:
            for k in range(NT):
                flush(k)
            st.elapsed += time.time() - t_start
            t_start, budget = time.time(), args.minutes * 60 - st.elapsed
            st.save(ckpt)
            t_ckpt = time.time()
            tot = st.counts.sum()
            print(f"[{st.elapsed / 60:5.1f} min] {tot:,} forms ({tot / st.elapsed:,.0f}/s); uniform: "
                  + ", ".join(f"{TYPES[k]}: {st.counts[0, k]:,}" for k in range(NT)), flush=True)
    for k in range(NT):
        flush(k)
    st.elapsed += time.time() - t_start
    st.save(ckpt)
    tot = st.counts.sum()
    print(f"scan done: {tot:,} forms in {st.elapsed / 60:.1f} min ({tot / st.elapsed:,.0f}/s)", flush=True)
    return st


# ---------------------------------------------------------------------------
# Finish: grid error, volumes, integer representatives
# ---------------------------------------------------------------------------


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _dirichlet_observed(rng, row, i):
    """Posterior draw of a confusion row, supported on the observed transitions (and the diagonal)."""
    support = (row > 0) | (np.arange(NT) == i)
    out = np.zeros(NT)
    out[support] = rng.dirichlet(row[support] + 0.5)
    return out


def volumes(counts, conf, rng, n_boot=2000):
    """Raw and corrected type fractions; conf[i, j] = #(coarse i, reference j) from the reservoirs. The corrected
    intervals bootstrap the coarse counts and the confusion rows (over observed transitions only)."""
    n = counts.sum()
    raw = counts / n
    cond = (conf + 1e-12) / (conf.sum(1, keepdims=True) + 1e-12 * NT)
    corr = raw @ cond
    boots = []
    for _ in range(n_boot):
        p = rng.dirichlet(counts + 0.5)
        q = np.stack([_dirichlet_observed(rng, conf[i], i) for i in range(NT)])
        boots.append(p @ q)
    boots = np.array(boots)
    return {TYPES[j]: {"count": int(counts[j]), "raw": float(raw[j]), "raw_ci95": wilson(int(counts[j]), int(n)),
                       "corrected": float(corr[j]),
                       "corrected_ci95": (float(np.quantile(boots[:, j], 0.025)),
                                          float(np.quantile(boots[:, j], 0.975)))}
            for j in range(NT)}


def to_float_columns(vs):
    a = np.array(vs, dtype=np.float64)
    a = a / np.abs(a).max(axis=1, keepdims=True)
    return torch.as_tensor(a.T, dtype=DTYPE, device=DEVICE)


def integer_candidates(mode, f, scales):
    """Integer vectors round(s f / max|f|) (generating form) and their exact curve forms, smallest scales first."""
    f = np.where(LOW.cpu().numpy(), 0.0, f) if mode == "hessian" else f
    f = f / np.abs(f).max()
    out, seen = [], set()
    for s in scales:
        v = tuple(int(x) for x in np.rint(s * f))
        g = math.gcd(*v)
        v = tuple(x // g for x in v) if g > 1 else v
        if v in seen or not any(v):
            continue
        seen.add(v)
        c = list(v) if mode == "plain" else exact_hessian(DEG, list(v))
        if any(c):
            out.append((list(v), c, s))
    return out


def certify_type(c, k, seed=0):
    """Exact oval count and nesting type of the integer quartic c (expected k ovals). Returns a dict."""
    from conncomp.separation import certify_lower_bound

    H = form_poly(c, DEG)
    ub = certify_upper_bound(H, target=k, seed=seed)
    if ub is None or not ub.certified:
        return {"certified": False, "reason": "upper bound" if ub is None else (ub.reason or "upper bound")}
    flint = crosscheck_flint(H, ub)
    wit = [w["point"] for w in witnesses(c, DEG, 400)]
    lb = certify_lower_bound(H, wit)
    sep = lb.certificate
    proven = ub.upper_bound if (flint["agree"] and sep.lower_bound == ub.upper_bound) else None
    out = {"certified": proven is not None, "upper_bound": ub.upper_bound, "lower_bound": sep.lower_bound,
           "flint_agree": bool(flint["agree"]), "base_point": ub.base_point}
    if proven is None:
        return out
    if proven == 2:
        sO = eval_form(c, DEG, [int(v) for v in ub.base_point])
        sO = (sO > 0) - (sO < 0)
        same = sum(1 for i in sep.separated if sep.signs[i] == sO)
        out["type"] = "1<1>" if same == 2 else "2"
        out["nesting"] = {"sign_N": sO, "separated_signs": [sep.signs[i] for i in sep.separated]}
    else:
        out["type"] = str(proven)
    out["upper"] = {kk: v for kk, v in asdict(ub).items() if kk != "resultant"}
    out["lower"] = asdict(sep)
    return out


def diverse(F, n):
    """Indices of up to n rows of F (sorted by preference), pairwise |cos| < 0.95."""
    U = F / np.linalg.norm(F, axis=1, keepdims=True)
    keep = []
    for i in range(len(F)):
        if all(abs(U[i] @ U[j]) < 0.95 for j in keep):
            keep.append(i)
        if len(keep) == n:
            break
    return keep


def finish(args, run, st):
    rng = np.random.default_rng(0)
    ref, mid = FaceMesh(200), FaceMesh(100)
    summary = {"mode": args.mode, "N": args.N, "forms": int(st.counts.sum()), "minutes": st.elapsed / 60,
               "rate": float(st.counts.sum() / st.elapsed), "measures": {}}
    for m, name in enumerate(MEASURES):
        conf = np.zeros((NT, NT), dtype=np.int64)
        conf_mid = np.zeros((NT, NT), dtype=np.int64)
        for k in range(NT):
            f = st.res[(m, k)][1]
            if len(f) == 0:
                continue
            c = curve(args.mode, torch.as_tensor(f.T, device=DEVICE))
            conf[k] = torch.bincount(classify(ref, c), minlength=NT).cpu().numpy()
            conf_mid[k] = torch.bincount(classify(mid, c), minlength=NT).cpu().numpy()
        summary["measures"][name] = {
            "forms": int(st.counts[m].sum()),
            "volumes": volumes(st.counts[m], conf, rng),
            "confusion_N200": conf.tolist(), "confusion_N100": conf_mid.tolist(),
        }
        print(f"{name}: " + ", ".join(f"{t}: {v['corrected']:.3g}" for t, v in summary["measures"][name]
                                      ["volumes"].items()), flush=True)

    scales = [3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400, 512]
    reps, pool_stats = {}, {}
    for k in range(NT - 1):
        s, F = st.pool[k]
        pool_stats[TYPES[k]] = {"size": int(len(s)), "score_max": float(s.max()) if len(s) else None,
                                "score_median": float(np.median(s)) if len(s) else None}
        reps[TYPES[k]] = []
        if len(s) == 0:
            continue
        c = curve(args.mode, torch.as_tensor(F.T, device=DEVICE))
        ok = (classify(ref, c) == k).cpu().numpy()  # pool forms confirmed at the reference resolution
        order = [i for i in np.argsort(-s) if ok[i]]
        for i in [order[j] for j in diverse(F[order], 12)]:
            if len(reps[TYPES[k]]) >= args.reps:
                break
            cands = integer_candidates(args.mode, F[i], scales)
            cf = to_float_columns([cc for _, cc, _ in cands])
            good = ((classify(mid, cf) == k) & (classify(ref, cf) == k)).cpu().numpy()
            tries = 0
            for (v, cc, sc), g in zip(cands, good):
                if not g or tries >= 3:
                    continue
                tries += 1
                t0 = time.time()
                cert = certify_type(cc, int(TYPES[k][0]) if k != 3 else 2, seed=tries)
                print(f"  type {TYPES[k]}: pool #{i} (score {s[i]:.2e}) scale {sc} height {max(map(abs, v))}: "
                      f"{cert.get('type', 'not certified')} ({time.time() - t0:.1f} s)", flush=True)
                if cert.get("type") == TYPES[k]:
                    x, y = sp.symbols("x y")
                    rep = {"type": TYPES[k], "f": v, "height": max(map(abs, v)),
                           "f_affine": str(sp.expand(to_sympy(v, DEG, x, y, 1))),
                           "robustness": float(s[i])}
                    if args.mode == "hessian":
                        rep["hessian"] = cc
                        rep["hessian_affine"] = str(sp.expand(to_sympy(cc, DEG, x, y, 1)))
                    rep["certificate"] = {kk: cert[kk] for kk in ("upper_bound", "lower_bound", "flint_agree",
                                                                  "base_point") if kk in cert}
                    if "nesting" in cert:
                        rep["certificate"]["nesting"] = cert["nesting"]
                    reps[TYPES[k]].append(rep)
                    full = run / f"cert_{TYPES[k].replace('<', '(').replace('>', ')')}_{len(reps[TYPES[k]])}.json"
                    full.write_text(json.dumps({"rep": rep, "certificate": cert}, indent=1, default=str))
                    break
    summary["pools"] = pool_stats
    summary["representatives"] = {t: len(r) for t, r in reps.items()}
    (run / "summary.json").write_text(json.dumps(summary, indent=1))
    out = Path("data/quartic_types")
    out.mkdir(parents=True, exist_ok=True)
    desc = ("Integer quartic forms (coefficients in the order of conncomp.polynomials.monomials(4)) of each nesting "
            "type; oval count certified exactly (pencil tangents + separating polygons, FLINT cross-check), "
            "nesting by the sign of the region containing the certified base point.")
    (out / f"{args.mode}.json").write_text(json.dumps({"mode": args.mode, "description": desc,
                                                       "representatives": reps}, indent=1))
    write_markdown(run / "summary.md", summary)
    print((run / "summary.md").read_text())


def write_markdown(path, s):
    lines = [f"# Quartic nesting types: {s['mode']}", "",
             f"{s['forms']:,} forms in {s['minutes']:.1f} min ({s['rate']:,.0f}/s) on FaceMesh({s['N']}); "
             "corrected = coarse types mapped through the confusion matrix against N = 200.", ""]
    for name, m in s["measures"].items():
        lines += [f"## {name} ({m['forms']:,} forms)", "", "| type | count | raw fraction | corrected | 95% CI |",
                  "|---|---|---|---|---|"]
        for t, v in m["volumes"].items():
            lo, hi = v["corrected_ci95"]
            lines.append(f"| {t} | {v['count']:,} | {v['raw']:.4g} | {v['corrected']:.4g} | {lo:.3g} – {hi:.3g} |")
        lines.append("")
    lines += ["## Pools and representatives", "", "| type | pool | best robustness | certified integer forms |",
              "|---|---|---|---|"]
    for t, p in s["pools"].items():
        best = f"{p['score_max']:.3g}" if p["score_max"] is not None else "-"
        lines.append(f"| {t} | {p['size']} | {best} | {s['representatives'][t]} |")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["plain", "hessian"], required=True)
    ap.add_argument("--minutes", type=float, default=27)
    ap.add_argument("--stage", choices=["all", "scan", "finish"], default="all")
    ap.add_argument("--N", type=int, default=30, help="FaceMesh resolution of the scan")
    ap.add_argument("--batch", type=int, default=50000)
    ap.add_argument("--reservoir", type=int, default=2000, help="uniform sample per (measure, coarse type)")
    ap.add_argument("--pool", type=int, default=200, help="most robust forms kept per type")
    ap.add_argument("--cand", type=int, default=64, help="pool candidates per type and batch")
    ap.add_argument("--flush", type=int, default=4000, help="score buffered candidates when this many")
    ap.add_argument("--checkpoint", type=float, default=3, help="minutes between checkpoints")
    ap.add_argument("--reps", type=int, default=3, help="integer representatives per type")
    ap.add_argument("--out", type=Path, default=None, help="run directory (default pools/quartic_<mode>)")
    args = ap.parse_args()
    run = args.out or Path("pools") / f"quartic_{args.mode}"
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(time.time()))
    if args.stage in ("all", "scan"):
        st = scan(args, run)
    else:
        st = State.load(run / "checkpoint.npz")
    if args.stage in ("all", "finish"):
        finish(args, run, st)


if __name__ == "__main__":
    main()
