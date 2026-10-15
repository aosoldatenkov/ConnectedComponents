"""Interactive (curses) adaptive search for forms with many sign components.

Each iteration evaluates a batch of forms, drawn either uniformly at random or as
small perturbations of good forms found earlier. The good forms are kept in a diverse
cache (conncomp.pool.DiverseCache: members pairwise at least --min-angle apart, parents
drawn uniformly over lineages), so the search does not collapse onto the descendants of
one form. Press `q` to stop; the cached forms are then written to the output file, best first.

With --collect MIN the search also builds a pool: every exactly counted form with at least MIN ovals is appended to
<out stem>_ovals<n>.txt (written every --dump-every seconds and on exit), thinned to curves at least --collect-angle
apart within each oval count; the Euler screen then passes forms estimated near MIN as well (at most --screen-cap
per batch). Output goes to output/ by default.

With --seed FILE... the cache starts from the given forms (counted at --width; projected onto --symmetry), e.g. to
explore the x-symmetric family around xy-symmetric 16-oval Hessians:

    uv run conncomp-search --deg 6 --symmetry x --seed pools/hessian6_test/xy_genuine16.txt --uniform-share 0

    uv run conncomp-search --deg 6 --width 100
"""

import argparse
import curses as cr
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from conncomp import DEVICE
from conncomp.io import load_coefs, save_coefs
from conncomp.pool import Collector, DiverseCache, descriptors
from conncomp.polynomials import perturb, sample
from conncomp.components import CountPipeline
from conncomp.euler import EulerScreen, select_candidates
from conncomp.scan import Experiment
from conncomp.symmetry import symmetry

CACHE_LEN = 1000
PERTURB_RADIUS = 1e-1


def loop(stdscr, deg, width, batch_size, filtr, use_hessian, out_path, min_comp=8, sym=None, screen=None,
         slack=2, cap=50000, min_angle=10.0, uniform_share=1 / 3, collector=None, seeds=None):
    """Run the search until `q` is pressed; returns (forms, oval counts) of the cache, best first.

    Cache policy (conncomp.pool.DiverseCache): forms with at least max - 2 ovals (Hessian mode), or at least
    min(min_comp, max) ovals (plain mode), are offered to a cache whose members are pairwise at least `min_angle`
    degrees apart (between the normalized curves); parents are drawn uniformly over lineages (the uniform sample a
    form descends from). A share `uniform_share` of the batches are fresh uniform samples.

    With `screen` (conncomp.euler.EulerScreen), every batch is screened on the GPU and only forms
    whose estimated oval count is at least (maximum found - slack), at most `cap` of them, are
    counted exactly.
    """
    cr.use_default_colors()
    stdscr.nodelay(True)
    iteration = 0
    max_comp_found = 0
    cache = DiverseCache(CACHE_LEN, min_angle)
    if seeds is not None:  # (forms (D, B) on the GPU, oval counts (B,)): each seed starts its own lineage
        sc, sn = seeds
        cache.add(sc, descriptors(sc, deg, use_hessian), sn, [-1 - i for i in range(sc.shape[1])])
        max_comp_found = int(max(sn))
    comp_counts = defaultdict(int)

    def screen_floor(mx):  # collect mode: also count forms near the collection minimum
        return mx - slack if collector is None else min(mx, collector.min_ovals) - slack
    exp = Experiment(deg, width, use_hessian)
    basis = sym.basis if sym is not None else None
    # The CPU counts batch k on a worker thread while the GPU prepares batch k + 1
    pipe = CountPipeline(exp.pat, filtr)
    t_start, rate, generated, counted, n_uniform = time.perf_counter(), 0.0, 0, 0, 0

    def process(counts, coefs, lineage):
        nonlocal max_comp_found
        l = np.asarray(counts) - 1  # numbers of ovals
        if collector is not None:
            collector.add(coefs, l)
        for x in l.tolist():
            comp_counts[x] += 1
        max_comp_found = max(max_comp_found, int(l.max()) if len(l) else 0)
        thr = max_comp_found - 2 if use_hessian else min(min_comp, max_comp_found)  # plain: also the current max
        sel = np.nonzero(l >= thr)[0]
        if len(sel):
            c = coefs[:, torch.as_tensor(sel, device=coefs.device)]
            lin = lineage if np.isscalar(lineage) else [lineage[i] for i in sel]
            cache.add(c, descriptors(c, deg, use_hessian), l[sel], lin)

    while True:
        stdscr.erase()
        sym_txt = f"; symmetry {sym.describe()}" if sym is not None else ""
        stdscr.addstr(0, 0, f"Using {DEVICE} device; {'Hessian of ' if use_hessian else ''}degree {deg}, width {width}"
                      + sym_txt)
        scr_txt = f";\t counted exactly: {counted / max(generated, 1):.2%}" if screen is not None else ""
        stdscr.addstr(1, 0, f"Iteration: {iteration};\t Total samples: {generated:,};\t {rate:,.0f} samples/s"
                      + scr_txt)
        stdscr.addstr(2, 0, "Total counts: " + " ".join(f"{i}: {comp_counts[i]};" for i in sorted(comp_counts)))
        stdscr.addstr(3, 0, f"Cache: {len(cache)} forms (pairwise >= {min_angle:g} deg apart) in {cache.lineages()} "
                      f"lineages; ovals: " + ", ".join(f"{k}: {v}" for k, v in cache.histogram().items()))
        coll = f";\t collected (>= {collector.min_ovals}; kept/seen): {collector.summary()}" if collector else ""
        stdscr.addstr(4, 0, f"Maximal number of components found: {max_comp_found}" + coll)
        stdscr.addstr(5, 0, f"Press q to stop and save to {out_path}")
        stdscr.refresh()

        if len(cache) and random.random() >= uniform_share:
            parent, lineage = cache.parent()
            coefs = perturb(parent, batch_size, PERTURB_RADIUS, basis)
        else:
            coefs = sample(deg, batch_size, basis=basis)
            lineage = None  # set per form below

        generated += coefs.shape[1]
        idx = None
        if screen is not None:
            est = screen.estimate(coefs)
            idx = select_candidates(est, screen_floor(max_comp_found), cap)
            coefs = coefs[:, idx]
        if lineage is None:  # unrelated uniform forms: one new lineage each
            m = coefs.shape[1]
            lineage = list(range(n_uniform, n_uniform + m))
            n_uniform += m
        if coefs.shape[1]:
            pipe.submit(exp.signs(coefs), (coefs, lineage))
        for counts, (c, lin) in pipe.results():
            process(counts, c, lin)
            iteration += 1
            counted += len(counts)
        rate = generated / (time.perf_counter() - t_start)

        if stdscr.getch(0, 0) in (ord("q"), ord("Q")):
            for counts, (c, lin) in pipe.results(0):
                process(counts, c, lin)
            pipe.close()
            if collector is not None:
                collector.dump()
            return cache.best()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--deg", type=int, default=6, help="degree d of the form f (default: 6)")
    p.add_argument("--width", type=int, default=100, help="grid width in pixels (default: 100)")
    p.add_argument("--batch", type=int, default=None,
                   help="forms per iteration (default: 1000000 with the screen, 50000 without)")
    p.add_argument("--filter", type=int, default=3, help="minimal component size in pixels (default: 3)")
    p.add_argument("--no-hessian", action="store_true", help="count components of f itself instead of H(f)")
    p.add_argument("--symmetry", default=None,
                   help="restrict to semi-invariant forms: x, x:-1, xy, diag, C3, D3, D4:1,-1, ... (conncomp.symmetry)")
    p.add_argument("--no-screen", action="store_true", help="count every form exactly (no Euler pre-screen)")
    p.add_argument("--screen-lattice", type=int, default=40, help="cube lattice size N of the screen (default: 40)")
    p.add_argument("--screen-slack", type=int, default=2,
                   help="count forms whose estimate is >= maximum found - slack (default: 2)")
    p.add_argument("--screen-cap", type=int, default=50000, help="max forms counted exactly per batch (default: 50000)")
    p.add_argument("--min-angle", type=float, default=10.0,
                   help="minimal angle in degrees between cached forms (normalized curves; default: 10)")
    p.add_argument("--seed", type=Path, nargs="+", default=None,
                   help="coefficient files whose forms initialize the cache (projected onto --symmetry if given)")
    p.add_argument("--uniform-share", type=float, default=1 / 3, help="share of fresh uniform batches (default: 1/3)")
    p.add_argument("--collect", type=int, default=None, metavar="MIN",
                   help="also save every counted form with at least MIN ovals to <out stem>_ovals<n>.txt (appended "
                        "periodically); the screen then counts forms estimated near MIN as well")
    p.add_argument("--dump-every", type=float, default=60.0, help="seconds between dumps of collected forms")
    p.add_argument("--collect-angle", type=float, default=5.0,
                   help="thin the collected forms: keep a form only if its curve is at least this many degrees from "
                        "all kept forms with the same oval count (default: 5; 0: keep all)")
    p.add_argument("--out", type=Path, default=None,
                   help="output file (default: output/cache_<mode>_deg<d>_<time>.txt)")
    args = p.parse_args(argv)

    use_hessian = not args.no_hessian
    sym = symmetry(args.symmetry, args.deg) if args.symmetry else None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"_sym-{args.symmetry.replace(':', '_').replace(',', '_')}" if sym else ""
    out = args.out or Path("output") / f"cache_{'H' if use_hessian else 'f'}_deg{args.deg}{tag}_{stamp}.txt"
    random.seed()
    screen = None if args.no_screen else EulerScreen(args.deg, N=args.screen_lattice, use_hessian=use_hessian)
    batch = args.batch or (50000 if screen is None else 1_000_000)
    if screen is not None:
        screen.estimate(sample(args.deg, 10))  # compile before entering curses
    collector = (Collector(out, args.collect, args.dump_every, args.collect_angle, args.deg, use_hessian)
                 if args.collect is not None else None)
    seeds = None
    if args.seed:
        sc = torch.as_tensor(np.asarray([c for fn in args.seed for c in load_coefs(fn)], dtype=np.float64).T,
                             device=DEVICE)
        if sym is not None:  # project onto the symmetric family (as perturb does for the children)
            B = torch.as_tensor(np.asarray(sym.basis, dtype=np.float64), device=DEVICE)
            sc = B @ (torch.linalg.pinv(B) @ sc)
        sc = sc / sc.norm(dim=0, keepdim=True)
        sn = np.asarray(Experiment(args.deg, args.width, use_hessian).count(sc, args.filter)) - 1
        seeds = (sc, sn)
        print(f"seeds: {sc.shape[1]} forms; oval counts at width {args.width}: "
              f"{ {k: int((sn == k).sum()) for k in sorted(set(sn.tolist()), reverse=True)} }")
    best, counts = cr.wrapper(loop, args.deg, args.width, batch, args.filter, use_hessian, out, sym=sym,
                              screen=screen, slack=args.screen_slack, cap=args.screen_cap, min_angle=args.min_angle,
                              uniform_share=args.uniform_share, collector=collector, seeds=seeds)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_coefs(out, best)
    hist = {k: counts.count(k) for k in sorted(set(counts), reverse=True)}
    print(f"Saved {len(best)} coefficient vectors to {out} (best first; ovals: {hist})")
    if collector is not None:
        print(f"Collected forms with >= {args.collect} ovals ({collector.summary()}) in "
              f"{collector.file('<n>').name.replace('<n>', 'N')} files next to it")


if __name__ == "__main__":
    main()
