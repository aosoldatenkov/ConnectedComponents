import numpy as np
import pytest
import sympy as sp
import torch

from conncomp.euler import EulerScreen, euler_characteristics, face_grid_points, select_candidates
from conncomp.polynomials import from_sympy, sample
from conncomp.scan import Experiment, adaptive_scan
from conncomp.sphere import CubeLattice

x, y = sp.symbols("x y")


def test_matches_explicit_mesh():
    N = 5
    lat = CubeLattice(N)
    F, E = lat.faces(), lat.edges()
    assert len(lat) - len(E) + len(F) == 2
    fg = face_grid_points(N)
    idx = np.vectorize(lambda a, b, c: lat.index[(a, b, c)])(fg[..., 0], fg[..., 1], fg[..., 2])

    def chi(sv):
        return (sv.sum() - (sv[E[:, 0]] & sv[E[:, 1]]).sum()
                + (sv[F[:, 0]] & sv[F[:, 1]] & sv[F[:, 2]] & sv[F[:, 3]]).sum())

    rng = np.random.default_rng(0)
    batch = [rng.random(len(lat)) < rng.random() for _ in range(40)]
    pos, neg = euler_characteristics(torch.as_tensor(np.stack([sv[idx] for sv in batch])))
    assert pos.tolist() == [int(chi(sv)) for sv in batch]
    assert neg.tolist() == [int(chi(~sv)) for sv in batch]


def coefs(expr, deg):
    return torch.tensor([float(v) for v in from_sympy(expr, deg)], dtype=torch.float64).reshape(-1, 1)


@pytest.mark.parametrize("expr,deg,expected", [
    (x**2 + y**2 + 1, 2, 1),  # empty curve: chi = (1, 0), indistinguishable from one oval (1, 0)
    (x**2 + y**2 - 1, 2, 1),
    ((x**2 + y**2 - sp.Rational(1, 4)) * ((x - 3) ** 2 + y**2 - sp.Rational(1, 4)), 4, 2),
])
def test_estimates_of_simple_curves(expr, deg, expected):
    scr = EulerScreen(deg, N=30, use_hessian=False, compile=False)
    c = coefs(expr, deg).to(scr.basis.device)
    assert scr.estimate(c).item() == expected


def test_nested_ovals_are_underestimated():
    # documented limitation: two nested circles (2 ovals) give a smaller estimate
    scr = EulerScreen(4, N=30, use_hessian=False, compile=False)
    c = coefs((x**2 + y**2 - sp.Rational(1, 4)) * (x**2 + y**2 - 4), 4).to(scr.basis.device)
    assert scr.estimate(c).item() < 2


def test_compiled_matches_eager_and_exact_counts():
    c = sample(5, 3000)
    eager = EulerScreen(5, N=30, compile=False).estimate(c)
    compiled = EulerScreen(5, N=30, chunk=1024).estimate(c)
    assert torch.equal(eager, compiled)
    exact = torch.tensor(Experiment(5, 100, True).count(c, 3), device=eager.device) - 1
    assert (eager == exact).float().mean() > 0.8  # a filter, not exact
    assert torch.all(eager[exact >= 6] >= 5)


def test_select_candidates():
    est = torch.tensor([3, 7, 5, 9, 1, 7])
    assert sorted(select_candidates(est, 5, 10).tolist()) == [1, 2, 3, 5]
    assert sorted(est[select_candidates(est, 5, 2)].tolist()) == [7, 9]


def test_screened_adaptive_scan_runs():
    scr = EulerScreen(5, N=30, chunk=2048)
    best, hist = adaptive_scan(5, 60, 20000, 3, screen=scr, cap=500, keep=10)
    assert best and sum(hist.values()) <= 3 * 500
