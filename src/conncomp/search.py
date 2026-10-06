"""Interactive (curses) adaptive search for forms with many sign components.

Each iteration evaluates a batch of forms, drawn either uniformly at random or as
small perturbations of good forms found earlier (kept in two caches). Press `q`
to stop; the best forms found are then written to the output file.

    uv run conncomp-search --deg 6 --width 100
"""

import argparse
import curses as cr
import random
import time
from collections import defaultdict, deque
from pathlib import Path

from conncomp import DEVICE
from conncomp.io import save_coefs
from conncomp.polynomials import perturb, sample
from conncomp.components import CountPipeline
from conncomp.euler import EulerScreen, select_candidates
from conncomp.scan import Experiment
from conncomp.symmetry import symmetry

CACHE_LEN = 1000
PERTURB_RADIUS = 1e-1


def loop(stdscr, deg, width, batch_size, filtr, use_hessian, out_path, min_comp=8, sym=None, screen=None,
         slack=2, cap=50000):
    """Run the search until `q` is pressed; returns the list of best coefficient vectors.

    Cache policy: cache1 always receives forms reaching the current maximum. cache0 receives
    forms with at least `min_comp` ovals (plain mode) or within 2 of the maximum (Hessian mode).

    With `screen` (conncomp.euler.EulerScreen), every batch is screened on the GPU and only forms
    whose estimated oval count is at least (maximum found - slack), at most `cap` of them, are
    counted exactly.
    """
    cr.use_default_colors()
    stdscr.nodelay(True)
    iteration = 0
    max_comp_found = 0
    cache0 = deque([], CACHE_LEN)
    cache1 = deque([], CACHE_LEN)
    comp_counts = defaultdict(int)
    exp = Experiment(deg, width, use_hessian)
    basis = sym.basis if sym is not None else None
    # The CPU counts batch k on a worker thread while the GPU prepares batch k + 1
    pipe = CountPipeline(exp.pat, filtr)
    t_start, rate, generated, counted = time.perf_counter(), 0.0, 0, 0

    def process(counts, coefs):
        nonlocal max_comp_found
        cpu_coefs = coefs.cpu().numpy()
        for j, c in enumerate(counts):
            l = c - 1  # number of ovals
            comp_counts[l] += 1
            if l > max_comp_found:
                max_comp_found = l
            if use_hessian:
                if l == max_comp_found:
                    cache1.append(cpu_coefs[:, j])
                elif l >= max_comp_found - 2 and len(cache0) < CACHE_LEN:
                    cache0.append(cpu_coefs[:, j])
            else:
                if l >= min_comp and len(cache0) < CACHE_LEN:
                    cache0.append(cpu_coefs[:, j])
                if l == max_comp_found:
                    cache1.append(cpu_coefs[:, j])

    while True:
        stdscr.erase()
        sym_txt = f"; symmetry {sym.describe()}" if sym is not None else ""
        stdscr.addstr(0, 0, f"Using {DEVICE} device; {'Hessian of ' if use_hessian else ''}degree {deg}, width {width}"
                      + sym_txt)
        scr_txt = f";\t counted exactly: {counted / max(generated, 1):.2%}" if screen is not None else ""
        stdscr.addstr(1, 0, f"Iteration: {iteration};\t Total samples: {generated:,};\t {rate:,.0f} samples/s"
                      + scr_txt)
        stdscr.addstr(2, 0, "Total counts: " + " ".join(f"{i}: {comp_counts[i]};" for i in sorted(comp_counts)))
        stdscr.addstr(3, 0, f"Cache 0: {len(cache0)};\t Cache 1: {len(cache1)}")
        stdscr.addstr(4, 0, f"Maximal number of components found: {max_comp_found}")
        stdscr.addstr(5, 0, f"Press q to stop and save to {out_path}")
        stdscr.refresh()

        dice = random.randrange(3)
        if dice == 0 and cache0:
            coefs = perturb(cache0.popleft(), batch_size, PERTURB_RADIUS, basis)
        elif dice == 1 and cache1:
            coefs = perturb(cache1[random.randrange(len(cache1))], batch_size, PERTURB_RADIUS, basis)
        else:
            coefs = sample(deg, batch_size, basis=basis)

        generated += coefs.shape[1]
        if screen is not None:
            est = screen.estimate(coefs)
            coefs = coefs[:, select_candidates(est, max_comp_found - slack, cap)]
        if coefs.shape[1]:
            pipe.submit(exp.signs(coefs), coefs)
        for counts, c in pipe.results():
            process(counts, c)
            iteration += 1
            counted += len(counts)
        rate = generated / (time.perf_counter() - t_start)

        if stdscr.getch(0, 0) in (ord("q"), ord("Q")):
            for counts, c in pipe.results(0):
                process(counts, c)
            pipe.close()
            return list(reversed(cache1))


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
    p.add_argument("--out", type=Path, default=None, help="output file (default: data/cache_<mode>_deg<d>_<time>.txt)")
    args = p.parse_args(argv)

    use_hessian = not args.no_hessian
    sym = symmetry(args.symmetry, args.deg) if args.symmetry else None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"_sym-{args.symmetry.replace(':', '_').replace(',', '_')}" if sym else ""
    out = args.out or Path("data") / f"cache_{'H' if use_hessian else 'f'}_deg{args.deg}{tag}_{stamp}.txt"
    random.seed()
    screen = None if args.no_screen else EulerScreen(args.deg, N=args.screen_lattice, use_hessian=use_hessian)
    batch = args.batch or (50000 if screen is None else 1_000_000)
    if screen is not None:
        screen.estimate(sample(args.deg, 10))  # compile before entering curses
    best = cr.wrapper(loop, args.deg, args.width, batch, args.filter, use_hessian, out, sym=sym, screen=screen,
                      slack=args.screen_slack, cap=args.screen_cap)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_coefs(out, best)
    print(f"Saved {len(best)} coefficient vectors to {out}")


if __name__ == "__main__":
    main()
