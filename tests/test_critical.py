"""Seeds of critical points (conncomp.critical) against exact critical points from sympy."""

import importlib.util
import random
from pathlib import Path

import numpy as np
import torch

from conncomp import DEVICE, DTYPE
from conncomp.critical import SADDLE, FaceCharts
from conncomp.polynomials import monomials

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_critical_seeds", ROOT / "scripts" / "check_critical_seeds.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def test_face_charts_cover_rp2():
    charts = FaceCharts(6, N=10)
    P = charts.points.reshape(-1, 3)
    assert torch.allclose(P.norm(dim=1), torch.ones(len(P), dtype=DTYPE, device=P.device))
    # every unit vector is within one grid spacing of a chart point, up to sign
    q = torch.randn(500, 3, dtype=DTYPE, device=P.device)
    q = q / q.norm(dim=1, keepdim=True)
    d = torch.arccos((q @ P.T).abs().max(dim=1).values.clamp(max=1))
    assert d.max() < charts.spacing


def test_seeds_find_all_exact_critical_points():
    rng = random.Random(1)
    forms = [[rng.randint(-50, 50) for _ in monomials(6)] for _ in range(4)]
    truth = [C.exact_critical_points(c, 6) for c in forms]
    assert all(sum(1 if k != SADDLE else -1 for _, k in t) == 1 for t in truth)  # Morse relation on RP^2
    charts = FaceCharts(6, N=40)
    seeds = charts.seeds(torch.tensor(np.array(forms, dtype=float).T, dtype=DTYPE, device=DEVICE))
    tol = 2.5 * charts.spacing
    for b, crit in enumerate(truth):
        sel = (seeds.form == b).nonzero().squeeze(1)
        P, K = seeds.point[sel].cpu().numpy(), seeds.kind[sel].cpu().numpy()
        for p, kind in crit:
            near = C.angular_distance(P, p) < tol
            assert (K[near] == kind).any()


def test_chart_derivatives_match_autograd():
    from conncomp.critical import chart_derivatives, face_point_batch
    from conncomp.polynomials import sample

    deg, K = 6, 64
    c = sample(deg, K).T.contiguous()
    face = torch.randint(0, 3, (K,), device=c.device)
    u = (torch.rand(K, dtype=DTYPE, device=c.device) * 2.2 - 1.1).requires_grad_()
    v = (torch.rand(K, dtype=DTYPE, device=c.device) * 2.2 - 1.1).requires_grad_()
    mons = torch.tensor(monomials(deg), dtype=DTYPE, device=c.device)
    P = face_point_batch(face, u, v)
    P = P / P.norm(dim=1, keepdim=True)
    h0 = (c * torch.prod(P[:, None, :] ** mons[None], dim=2)).sum(1)
    gu, gv = torch.autograd.grad(h0.sum(), (u, v), create_graph=True)
    huu, huv = torch.autograd.grad(gu.sum(), (u, v), retain_graph=True)
    _, hvv = torch.autograd.grad(gv.sum(), (u, v))
    h, g, Hm = chart_derivatives(c, face, u.detach(), v.detach(), deg)
    for a, b in ((h, h0), (g[:, 0], gu), (g[:, 1], gv), (Hm[:, 0, 0], huu), (Hm[:, 0, 1], huv), (Hm[:, 1, 1], hvv)):
        assert torch.allclose(a, b, atol=1e-12)


def test_newton_step_closed_form():
    from conncomp.critical import _newton_step

    A = torch.randn(2000, 2, 2, dtype=DTYPE, device=DEVICE)
    A = A + A.transpose(1, 2)
    special = [[[2.0, 0.0], [0.0, -3.0]], [[1.0, 0.0], [0.0, 1.0]], [[-3.0, 0.0], [0.0, 2.0]], [[0.0, 1.0], [1.0, 0.0]]]
    for k, m in enumerate(special):
        A[k] = torch.tensor(m, dtype=DTYPE)
    g = torch.randn(2000, 2, dtype=DTYPE, device=DEVICE)
    ref = -torch.linalg.solve(A, g[:, :, None])[:, :, 0]
    assert ((_newton_step(g, A, 1e-300) - ref).norm(dim=1) / ref.norm(dim=1)).max() < 1e-9


def test_refined_critical_points_match_exact():
    from conncomp.critical import critical_points

    rng = random.Random(2)
    forms = [[rng.randint(-50, 50) for _ in monomials(6)] for _ in range(4)]
    truth = [C.exact_critical_points(c, 6) for c in forms]
    crit, morse = critical_points(torch.tensor(np.array(forms, dtype=float).T, dtype=DTYPE, device=DEVICE), 6)
    assert morse.tolist() == [1, 1, 1, 1]
    for b, tr in enumerate(truth):
        sel = (crit.form == b).nonzero().squeeze(1)
        P, K = crit.point[sel].cpu().numpy(), crit.kind[sel].cpu().numpy()
        assert len(P) == len(tr)  # no missing and no extra points
        for p, kind in tr:
            d = C.angular_distance(P, p)
            j = int(np.argmin(d))
            assert d[j] < 1e-7 and K[j] == kind


def test_morse_check_on_random_batch():
    from conncomp.critical import critical_points
    from conncomp.polynomials import sample

    torch.manual_seed(0)
    _, morse = critical_points(sample(6, 3000), 6)
    assert bool((morse == 1).all())


def test_bilinear_hessian():
    from conncomp.critical import _hessian_bilinear
    from conncomp.hessian import hessian
    from conncomp.polynomials import sample

    f = sample(5, 9)
    H = torch.einsum("kij,ib,jb->kb", _hessian_bilinear(5, f.device), f, f) / 2
    assert torch.allclose(H, hessian(5, f), atol=1e-12)


def test_value_gradients_finite_differences():
    from conncomp.critical import critical_points, hessian_value_gradients, track, value_gradients
    from conncomp.hessian import hessian
    from conncomp.polynomials import sample

    torch.manual_seed(3)
    eps = 1e-6
    c = sample(6, 200)
    crit, _ = critical_points(c, 6)
    d = torch.randn_like(c)
    new = track(crit, c + eps * d, 6)
    pred = eps * (value_gradients(crit, c, 6) * d.T[crit.form]).sum(1)
    assert torch.median(((new.value - crit.value) - pred).abs() / pred.abs()) < 1e-4
    f = sample(5, 200)
    crit, _ = critical_points(hessian(5, f), 6)
    d = torch.randn_like(f)
    new = track(crit, hessian(5, f + eps * d), 6)
    pred = eps * (hessian_value_gradients(crit, f, 5) * d.T[crit.form]).sum(1)
    assert torch.median(((new.value - crit.value) - pred).abs() / pred.abs()) < 1e-4


def test_cross_walls_flips_values():
    from conncomp.critical import critical_points, cross_walls, hessian_value_gradients
    from conncomp.hessian import hessian
    from conncomp.polynomials import sample

    torch.manual_seed(4)
    f = sample(5, 3000)
    crit, _ = critical_points(hessian(5, f), 6)
    dist = crit.value.abs() / hessian_value_gradients(crit, f, 5).norm(dim=1)
    target = crit.select(torch.argsort(dist)[:40])
    _, after, crossed, _ = cross_walls(target, f, 5)
    assert crossed.float().mean() > 0.3
    assert bool((torch.sign(after.value[crossed]) == -torch.sign(target.value[crossed])).all())
    assert abs(torch.median(after.value[crossed] / -target.value[crossed]).item() - 1) < 1e-2
