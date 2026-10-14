"""Interactive (curses) search along subspaces spanned by k-tuples of forms from a pool.

Each iteration picks k forms at random from the pool (k = 2 by default: a pencil) and evaluates a batch of forms
drawn by conncomp.polynomials.perturb_subspace: uniformly from the unit sphere of their span, plus Gaussian noise of
radius r in the transversal directions (r = 0: exactly on the span). As in conncomp.search, batches are screened by
the Euler characteristic estimate on the GPU, the candidates are counted exactly on the grid, and the forms reaching
the current maximum are kept in a diverse cache (conncomp.pool.DiverseCache). The loaded pool is thinned to forms
pairwise at least --pool-min-angle apart. With --grow, the cached forms at the maximum also join the pool.
With --collect MIN, every counted form with at least MIN ovals is also appended to <out stem>_ovals<n>.txt
(periodically, see --dump-every; thinned, see --collect-angle). Output goes to output/ by default. Press `q` to
stop; the best forms found are then written to the output file.

    uv run conncomp-subspace pools/hessian6_births/confirmed_15.txt --deg 6 --k 2
"""

import argparse
import curses as cr
import random
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch

from conncomp import DEVICE
from conncomp.components import CountPipeline
from conncomp.euler import EulerScreen, select_candidates
from conncomp.io import load_coefs, save_coefs
from conncomp.polynomials import deg_to_dim, perturb_subspace, sample
from conncomp.pool import Collector, DiverseCache, descriptors, spread
from conncomp.scan import Experiment

CACHE_LEN = 1000


def loop(stdscr, pool, deg, k, radius, width, batch_size, filtr, use_hessian, out_path, screen=None, slack=2,
         cap=50000, grow=False, min_angle=10.0, collector=None):
    """Run the subspace search until `q` is pressed; returns (forms, oval counts) of the cache, best first.

    pool: list of coefficient vectors (numpy, length D). Forms with at least max - 2 ovals are offered to a diverse
    cache (conncomp.pool.DiverseCache, members pairwise at least min_angle degrees apart; lineage = the pool tuple).
    With `grow`, the cached forms at the maximum are added to the pool of tuples.
    """
    cr.use_default_colors()
    stdscr.nodelay(True)
    iteration, max_comp_found = 0, 0
    cache = DiverseCache(CACHE_LEN, min_angle)
    comp_counts = defaultdict(int)

    def screen_floor(mx):  # collect mode: also count forms near the collection minimum
        return mx - slack if collector is None else min(mx, collector.min_ovals) - slack
    best_tuples = deque([], 5)  # the tuples (pool indices) that produced the last new maxima
    exp = Experiment(deg, width, use_hessian)
    pipe = CountPipeline(exp.pat, filtr)
    base = list(pool)
    t_start, rate, generated, counted = time.perf_counter(), 0.0, 0, 0

    def process(counts, coefs, tup):
        nonlocal max_comp_found, pool
        l = np.asarray(counts) - 1  # numbers of ovals
        if collector is not None:
            collector.add(coefs, l)
        for x in l.tolist():
            comp_counts[x] += 1
        if len(l) and int(l.max()) > max_comp_found:
            max_comp_found = int(l.max())
            best_tuples.append((max_comp_found, tup))
        sel = np.nonzero(l >= max_comp_found - 2)[0]
        if len(sel):
            c = coefs[:, torch.as_tensor(sel, device=coefs.device)]
            cache.add(c, descriptors(c, deg, use_hessian), l[sel], tup)
            if grow:
                forms, scores = cache.best()
                pool = base + [f for f, sc in zip(forms, scores) if sc == max_comp_found]

    while True:
        stdscr.erase()
        stdscr.addstr(0, 0, f"Using {DEVICE} device; {'Hessian of ' if use_hessian else ''}degree {deg}, width {width};"
                      f" subspaces of dimension {k} from a pool of {len(pool)} forms ({len(base)} loaded); "
                      f"transversal radius {radius:g}")
        scr_txt = f";\t counted exactly: {counted / max(generated, 1):.2%}" if screen is not None else ""
        stdscr.addstr(1, 0, f"Iteration: {iteration};\t Total samples: {generated:,};\t {rate:,.0f} samples/s"
                      + scr_txt)
        stdscr.addstr(2, 0, "Total counts: " + " ".join(f"{i}: {comp_counts[i]};" for i in sorted(comp_counts)))
        stdscr.addstr(3, 0, f"Cache: {len(cache)} forms (pairwise >= {min_angle:g} deg apart) from {cache.lineages()} "
                      f"tuples; ovals: " + ", ".join(f"{a}: {b}" for a, b in cache.histogram().items()))
        coll = f";\t collected (>= {collector.min_ovals}; kept/seen): {collector.summary()}" if collector else ""
        stdscr.addstr(4, 0, f"Maximal number of components found: {max_comp_found}" + coll)
        stdscr.addstr(5, 0, "Last new maxima from tuples: " + "; ".join(f"{l} <- {t}" for l, t in best_tuples))
        stdscr.addstr(6, 0, f"Press q to stop and save to {out_path}")
        stdscr.refresh()

        tup = tuple(random.sample(range(len(pool)), k))
        basis = np.stack([pool[i] for i in tup], axis=1)
        coefs = perturb_subspace(basis, batch_size, radius)
        generated += coefs.shape[1]
        if screen is not None:
            est = screen.estimate(coefs)
            coefs = coefs[:, select_candidates(est, screen_floor(max_comp_found), cap)]
        if coefs.shape[1]:
            pipe.submit(exp.signs(coefs), (coefs, tup))
        for counts, (c, t) in pipe.results():
            process(counts, c, t)
            iteration += 1
            counted += len(counts)
        rate = generated / (time.perf_counter() - t_start)

        if stdscr.getch(0, 0) in (ord("q"), ord("Q")):
            for counts, (c, t) in pipe.results(0):
                process(counts, c, t)
            pipe.close()
            if collector is not None:
                collector.dump()
            return cache.best()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pool", type=Path, nargs="+", help="coefficient files (one form f per line) forming the pool")
    p.add_argument("--deg", type=int, default=6, help="degree d of the form f (default: 6)")
    p.add_argument("--k", type=int, default=2, help="dimension of the subspaces (forms per tuple; default: 2)")
    p.add_argument("--radius", type=float, default=0.0,
                   help="transversal Gaussian radius (default: 0, exactly on the span)")
    p.add_argument("--grow", action="store_true", help="add the cached forms at the maximum to the pool")
    p.add_argument("--min-angle", type=float, default=10.0,
                   help="minimal angle in degrees between cached forms (normalized curves; default: 10)")
    p.add_argument("--pool-min-angle", type=float, default=5.0,
                   help="thin the loaded pool to forms pairwise at least this far apart (degrees; 0: keep all)")
    p.add_argument("--width", type=int, default=100, help="grid width in pixels (default: 100)")
    p.add_argument("--batch", type=int, default=None,
                   help="forms per iteration (default: 1000000 with the screen, 50000 without)")
    p.add_argument("--filter", type=int, default=3, help="minimal component size in pixels (default: 3)")
    p.add_argument("--no-hessian", action="store_true", help="count components of f itself instead of H(f)")
    p.add_argument("--no-screen", action="store_true", help="count every form exactly (no Euler pre-screen)")
    p.add_argument("--screen-lattice", type=int, default=40, help="cube lattice size N of the screen (default: 40)")
    p.add_argument("--screen-slack", type=int, default=2,
                   help="count forms whose estimate is >= maximum found - slack (default: 2)")
    p.add_argument("--screen-cap", type=int, default=50000, help="max forms counted exactly per batch (default: 50000)")
    p.add_argument("--collect", type=int, default=None, metavar="MIN",
                   help="also save every counted form with at least MIN ovals to <out stem>_ovals<n>.txt (appended "
                        "periodically); the screen then counts forms estimated near MIN as well")
    p.add_argument("--dump-every", type=float, default=60.0, help="seconds between dumps of collected forms")
    p.add_argument("--collect-angle", type=float, default=5.0,
                   help="thin the collected forms: keep a form only if its curve is at least this many degrees from "
                        "all kept forms with the same oval count (default: 5; 0: keep all)")
    p.add_argument("--out", type=Path, default=None,
                   help="output file (default: output/cache_<mode>_deg<d>_subspace<k>_<time>.txt)")
    args = p.parse_args(argv)

    pool = [np.asarray(c, dtype=np.float64) for fn in args.pool for c in load_coefs(fn)]
    D = deg_to_dim(args.deg)
    if any(len(c) != D for c in pool):
        raise SystemExit(f"pool forms must have {D} coefficients (degree {args.deg})")
    use_hessian = not args.no_hessian
    if args.pool_min_angle > 0:
        n0 = len(pool)
        pool = spread(pool, args.deg, use_hessian, args.pool_min_angle)
        print(f"pool: {n0} forms -> {len(pool)} pairwise >= {args.pool_min_angle:g} deg apart")
    if len(pool) < args.k:
        raise SystemExit(f"the pool has {len(pool)} forms, fewer than k = {args.k}")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = args.out or Path("output") / f"cache_{'H' if use_hessian else 'f'}_deg{args.deg}_subspace{args.k}_{stamp}.txt"
    random.seed()
    screen = None if args.no_screen else EulerScreen(args.deg, N=args.screen_lattice, use_hessian=use_hessian)
    batch = args.batch or (50000 if screen is None else 1_000_000)
    if screen is not None:
        screen.estimate(sample(args.deg, 10))  # compile before entering curses
    collector = (Collector(out, args.collect, args.dump_every, args.collect_angle, args.deg, use_hessian)
                 if args.collect is not None else None)
    best, counts = cr.wrapper(loop, pool, args.deg, args.k, args.radius, args.width, batch, args.filter, use_hessian,
                              out, screen=screen, slack=args.screen_slack, cap=args.screen_cap, grow=args.grow,
                              min_angle=args.min_angle, collector=collector)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_coefs(out, best)
    hist = {a: counts.count(a) for a in sorted(set(counts), reverse=True)}
    print(f"Saved {len(best)} coefficient vectors to {out} (best first; ovals: {hist})")
    if collector is not None:
        print(f"Collected forms with >= {args.collect} ovals ({collector.summary()}) in "
              f"{collector.file('<n>').name.replace('<n>', 'N')} files next to it")


if __name__ == "__main__":
    main()
