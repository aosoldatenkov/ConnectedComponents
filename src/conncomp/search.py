"""Interactive (curses) adaptive search for forms with many sign components.

Each iteration evaluates a batch of forms, drawn either uniformly at random or as
small perturbations of good forms found earlier (kept in two caches). Press `q`
to stop; the best forms found are then written to the output file.

    uv run conncomp-search --deg 6 --width 100 --batch 10000
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
from conncomp.scan import Experiment

CACHE_LEN = 1000
PERTURB_RADIUS = 1e-1


def loop(stdscr, deg, width, batch_size, filtr, use_hessian, out_path, min_comp=8):
    """Run the search until `q` is pressed; returns the list of best coefficient vectors.

    Cache policy: cache1 always receives forms reaching the current maximum. cache0 receives
    forms with at least `min_comp` ovals (plain mode) or within 2 of the maximum (Hessian mode).
    """
    cr.use_default_colors()
    stdscr.nodelay(True)
    total_samples = 0
    iteration = 0
    max_comp_found = 0
    cache0 = deque([], CACHE_LEN)
    cache1 = deque([], CACHE_LEN)
    comp_counts = defaultdict(int)
    exp = Experiment(deg, width, use_hessian)

    while True:
        stdscr.erase()
        stdscr.addstr(0, 0, f"Using {DEVICE} device; {'Hessian of ' if use_hessian else ''}degree {deg}, width {width}")
        stdscr.addstr(1, 0, f"Iteration: {iteration};\t Total samples: {total_samples}")
        stdscr.addstr(2, 0, "Total counts: " + " ".join(f"{i}: {comp_counts[i]};" for i in sorted(comp_counts)))
        stdscr.addstr(3, 0, f"Cache 0: {len(cache0)};\t Cache 1: {len(cache1)}")
        stdscr.addstr(4, 0, f"Maximal number of components found: {max_comp_found}")
        stdscr.addstr(5, 0, f"Press q to stop and save to {out_path}")
        stdscr.refresh()

        dice = random.randrange(3)
        if dice == 0 and cache0:
            coefs = perturb(cache0.popleft(), batch_size, PERTURB_RADIUS)
        elif dice == 1 and cache1:
            coefs = perturb(cache1[random.randrange(len(cache1))], batch_size, PERTURB_RADIUS)
        else:
            coefs = sample(deg, batch_size)

        counts = exp.count(coefs, filtr)
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
        iteration += 1
        total_samples += batch_size

        if stdscr.getch(0, 0) in (ord("q"), ord("Q")):
            return list(reversed(cache1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--deg", type=int, default=6, help="degree d of the form f (default: 6)")
    p.add_argument("--width", type=int, default=100, help="grid width in pixels (default: 100)")
    p.add_argument("--batch", type=int, default=10000, help="forms per iteration (default: 10000)")
    p.add_argument("--filter", type=int, default=3, help="minimal component size in pixels (default: 3)")
    p.add_argument("--no-hessian", action="store_true", help="count components of f itself instead of H(f)")
    p.add_argument("--out", type=Path, default=None, help="output file (default: data/cache_<mode>_deg<d>_<time>.txt)")
    args = p.parse_args(argv)

    use_hessian = not args.no_hessian
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = args.out or Path("data") / f"cache_{'H' if use_hessian else 'f'}_deg{args.deg}_{stamp}.txt"
    random.seed()
    best = cr.wrapper(loop, args.deg, args.width, args.batch, args.filter, use_hessian, out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_coefs(out, best)
    print(f"Saved {len(best)} coefficient vectors to {out}")


if __name__ == "__main__":
    main()
