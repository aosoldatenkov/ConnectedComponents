#!/usr/bin/env python3
"""End-to-end throughput of a search round: float64 copy vs uint8 signs vs pipelined signs.

    uv run python benchmarks/bench_search.py --deg 5 --width 100 --batch 50000 --rounds 8
"""

import argparse
import time

import torch

from conncomp import _connected
from conncomp.components import CountPipeline, count_components
from conncomp.polynomials import sample
from conncomp.scan import Experiment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deg", type=int, default=5)
    p.add_argument("--width", type=int, default=100)
    p.add_argument("--batch", type=int, default=50000)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--skip-float", action="store_true", help="skip the float64 variant (large batches)")
    p.add_argument("--screen-batch", type=int, default=1_000_000, help="batch size for the screened variant")
    args = p.parse_args()
    exp = Experiment(args.deg, args.width, True)
    N, R = args.batch, args.rounds

    def timed(name, fn):
        fn(1)  # warm-up
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(R)
        dt = time.perf_counter() - t0
        print(f"{name:34s} {dt / R:.2f} s/round  {N * R / dt:>10,.0f} samples/s")

    def float64_copy(rounds):
        for _ in range(rounds):
            v = exp.values(sample(args.deg, N)).cpu().numpy()
            [sum(1 for s in z if s >= 3) for z in _connected.components_batch(v, exp.pat)]

    def signs_sequential(rounds):
        for _ in range(rounds):
            count_components(exp.signs(sample(args.deg, N)).cpu().numpy(), exp.pat, 3)

    def signs_pipelined(rounds):
        pipe = CountPipeline(exp.pat, 3)
        for _ in range(rounds):
            c = sample(args.deg, N)
            pipe.submit(exp.signs(c), c)
            for _ in pipe.results():
                pass
        for _ in pipe.results(0):
            pass
        pipe.close()

    if not args.skip_float:
        timed("float64 copy, sequential", float64_copy)
    timed("uint8 signs, sequential", signs_sequential)
    timed("uint8 signs, pipelined", signs_pipelined)
    screened(args.deg, args.width, args.screen_batch)



def screened(deg=5, width=100, batch=1_000_000, rounds=6, N=40, cap=50000, slack=2):
    """Throughput of screened rounds (Euler pre-screen on the GPU, exact counts of the candidates)."""
    from conncomp.euler import EulerScreen, select_candidates

    exp = Experiment(deg, width, True)
    scr = EulerScreen(deg, N=N)
    scr.estimate(sample(deg, 10))
    pipe = CountPipeline(exp.pat, 3)
    best, counted = 0, 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rounds):
        c = sample(deg, batch)
        c = c[:, select_candidates(scr.estimate(c), best - slack, cap)]
        pipe.submit(exp.signs(c), c)
        for counts, _ in pipe.results():
            best = max(best, max(counts) - 1)
            counted += len(counts)
    for counts, _ in pipe.results(0):
        counted += len(counts)
    pipe.close()
    dt = time.perf_counter() - t0
    print(f"{'Euler screen + pipelined counts':34s} {dt / rounds:.2f} s/round  {batch * rounds / dt:>10,.0f} samples/s"
          f"  (counted exactly: {counted / (batch * rounds):.2%}, best {best})")


if __name__ == "__main__":
    main()
