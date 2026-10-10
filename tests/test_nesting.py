import numpy as np
import pytest
import sympy as sp
import torch

from conncomp import DEVICE, DTYPE
from conncomp.hessian import hessian
from conncomp.nesting import SphereMesh
from conncomp.polynomials import from_sympy, sample
from conncomp.scan import Experiment

x, y = sp.symbols("x y")
R = sp.Rational
CASES = [
    ("empty", x**2 + y**2 + 1, 2, "0"),
    ("circle", x**2 + y**2 - 1, 2, "1"),
    ("two disjoint", (x**2 + y**2 - R(1, 4)) * ((x - 2) ** 2 + y**2 - R(1, 4)), 4, "2"),
    ("nested pair", (x**2 + y**2 - R(1, 4)) * (x**2 + y**2 - 1), 4, "1<1>"),
    ("nest + separate", (x**2 + y**2 - R(1, 4)) * (x**2 + y**2 - 1) * ((x - 3) ** 2 + y**2 - R(1, 4)), 6, "1 u 1<1>"),
    ("three nested", (x**2 + y**2 - R(1, 16)) * (x**2 + y**2 - R(1, 4)) * (x**2 + y**2 - 1), 6, "1<1<1>>"),
    ("big around two",
     (x**2 + y**2 - 4) * ((x - R(1, 2)) ** 2 + y**2 - R(1, 9)) * ((x + R(1, 2)) ** 2 + y**2 - R(1, 9)), 6, "1<2>"),
]


@pytest.fixture(scope="module")
def mesh():
    return SphereMesh(60)


def coefs(expr, deg):
    return torch.tensor([float(v) for v in from_sympy(sp.expand(expr), deg)], dtype=DTYPE, device=DEVICE)[:, None]


@pytest.mark.parametrize("name,expr,deg,expected", CASES, ids=[c[0] for c in CASES])
def test_known_types(mesh, name, expr, deg, expected):
    c = coefs(expr, deg)
    t_grid = Experiment(deg, 300, False).nesting(c)[0]
    t_mesh = mesh.trees(c, deg)[0]
    assert t_grid.ok and t_mesh.ok
    assert t_grid.type == expected and t_mesh.type == expected
    assert t_mesh.n_ovals == len(t_mesh.size) - 1
    assert t_mesh.outside_sign == (1 if name != "empty" else 1)


def test_mesh_tables():
    m = SphereMesh(8)
    assert m.V == 24 * 64 + 2
    assert np.all(m.antipode[m.antipode] == np.arange(m.V))
    deg4 = (m.nbr4 >= 0).sum(1)
    assert set(deg4.tolist()) <= {3, 4}  # cube corners have valence 3


def test_mesh_trees_consistent_on_random_hessians(mesh):
    torch.manual_seed(0)
    f = sample(5, 2000)
    trees = mesh.trees(hessian(5, f), 6)
    assert all(t.ok for t in trees)


def test_gpu_trees_match_cpp(mesh):
    from conncomp.gpu_trees import to_nesting_trees

    torch.manual_seed(1)
    for coefs in (hessian(5, sample(5, 1500)), sample(6, 1500)):
        g = mesh.trees_gpu(coefs, 6)
        cpp = mesh.trees(coefs, 6)
        gt = to_nesting_trees(g, coefs.shape[1])
        assert g.n_ovals.tolist() == [t.n_ovals for t in cpp]
        assert [t.type for t in gt] == [t.type for t in cpp]
        assert bool(g.ok.all())


@pytest.mark.parametrize("name,expr,deg,expected", CASES[1:4], ids=[c[0] for c in CASES[1:4]])
def test_gpu_known_types(mesh, name, expr, deg, expected):
    from conncomp.gpu_trees import to_nesting_trees

    t = to_nesting_trees(mesh.trees_gpu(coefs(expr, deg), deg), 1)[0]
    assert t.ok and t.type == expected


def _has_cuda_module():
    try:
        from conncomp import _connected_cuda  # noqa: F401

        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.skipif(not _has_cuda_module(), reason="CUDA labelling module not built")
def test_cuda_face_mesh_matches_cpp():
    from conncomp.gpu_trees import FaceMesh, to_nesting_trees

    fm, sm = FaceMesh(40), SphereMesh(40)
    assert fm.n_loc - fm.seams.shape[0] == sm.V == int(fm.weight.sum())
    torch.manual_seed(2)
    for coefs in (hessian(5, sample(5, 2000)), sample(6, 2000)):
        g = to_nesting_trees(fm.trees(coefs, 6), coefs.shape[1])
        c = sm.trees(coefs, 6)
        assert [t.type for t in g] == [t.type for t in c]
        assert [sorted(t.size.tolist()) for t in g] == [sorted(t.size.tolist()) for t in c]
        assert all(t.ok for t in g)


@pytest.mark.skipif(not _has_cuda_module(), reason="CUDA labelling module not built")
@pytest.mark.parametrize("name,expr,deg,expected", CASES, ids=[c[0] for c in CASES])
def test_cuda_known_types(name, expr, deg, expected):
    from conncomp.gpu_trees import FaceMesh, to_nesting_trees

    t = to_nesting_trees(FaceMesh(60).trees(coefs(expr, deg), deg), 1)[0]
    assert t.ok and t.type == expected


def _key(t):
    return sorted(zip(t.form.tolist(), t.size.tolist(), t.sign.tolist(), t.alive.tolist(), t.is_root.tolist(),
                      t.depth.tolist()))


@pytest.mark.skipif(not _has_cuda_module(), reason="CUDA labelling module not built")
def test_cuda_tree_kernel_matches_torch():
    """Per-form CUDA tree kernel = PyTorch post-processing, incl. noisy forms (pruning) and the overflow fallback."""
    from conncomp.gpu_trees import FaceMesh, _trees_from_labels

    fm = FaceMesh(20)
    torch.manual_seed(4)
    s = fm.signs(sample(6, 300), 6)
    for p in (0.0, 0.003, 0.02, 0.3):  # p = 0.3: more components than the kernel's capacity
        flip = torch.rand(s.shape, device=s.device) < p
        sn = (s ^ (flip | flip[:, fm.anti]).to(torch.uint8)).contiguous()
        sn[:, fm.seams[:, 0].long()] = sn[:, fm.seams[:, 1].long()]
        for eight in (0, 1):
            L = fm.labels(sn, eight)
            a = fm._trees_cuda(sn, L, 3, eight)
            b = _trees_from_labels(sn, L, fm.nbr4, fm.anti, fm.weight, 3, sign_pairs=fm.sign_pairs)
            assert torch.equal(a.n_ovals, b.n_ovals) and torch.equal(a.ok, b.ok)
            assert _key(a) == _key(b)
    odd = fm.signs(sample(5, 100), 5)  # not antipodally symmetric: no root region
    assert not fm._trees_cuda(odd, fm.labels(odd), 3, 1).ok.any()
