import torch

from conncomp import DEVICE, DTYPE
from conncomp.pool import DiverseCache, descriptors, spread
from conncomp.polynomials import sample


def _keys(c):
    return (c / c.norm(dim=0, keepdim=True)).T


def test_niche_keeps_better_and_far_members():
    cache = DiverseCache(maxlen=10, min_angle_deg=10)
    a = torch.tensor([[1.0], [0.0], [0.0]], dtype=DTYPE, device=DEVICE)
    near = torch.tensor([[1.0], [0.05], [0.0]], dtype=DTYPE, device=DEVICE)  # 2.9 degrees from a
    far = torch.tensor([[0.0], [1.0], [0.0]], dtype=DTYPE, device=DEVICE)
    cache.add(a, _keys(a), [5], 0)
    cache.add(near, _keys(near), [5], 1)  # same niche, not better: rejected
    assert len(cache) == 1 and cache.lineage == [0]
    cache.add(near, _keys(near), [6], 1)  # same niche, better: replaces
    assert len(cache) == 1 and cache.score == [6] and cache.lineage == [1]
    cache.add(far, _keys(far), [4], 2)  # new niche
    assert len(cache) == 2 and cache.lineages() == 2


def test_full_cache_evicts_lowest_and_parents_balance_lineages():
    cache = DiverseCache(maxlen=3, min_angle_deg=1)
    c = torch.eye(4, dtype=DTYPE, device=DEVICE)
    cache.add(c[:, :3], _keys(c[:, :3]), [3, 7, 7], [0, 1, 1])
    cache.add(c[:, 3:], _keys(c[:, 3:]), [5], [2])  # evicts the member with 3 ovals
    assert sorted(cache.score) == [5, 7, 7]
    lins = [cache.parent()[1] for _ in range(2000)]
    assert abs(lins.count(2) / 2000 - 0.5) < 0.05  # two lineages, chosen uniformly


def test_descriptors_ignore_low_terms_of_f_and_spread():
    f = sample(5, 3)
    g = f.clone()
    g[0] += 0.3  # z^5 (monomials(5)[0]): degree 0 in x, y, does not change H(f)
    g[1] -= 0.2  # y z^4: degree 1 in x, y
    assert torch.allclose(descriptors(f, 5, True), descriptors(g, 5, True))
    forms = [x for x in f.T.cpu().numpy()] + [f[:, 0].cpu().numpy() * 2]
    assert len(spread(forms, 5, True, 5.0)) == 3  # the rescaled copy is dropped


def test_collector_thinning(tmp_path):
    import math

    from conncomp.io import load_coefs
    from conncomp.pool import Collector

    col = Collector(tmp_path / "c.txt", 3, every=1e9, min_angle_deg=10.0, deg=5, use_hessian=True)
    f = sample(5, 50)
    g = f + 1e-4 * sample(5, 50)  # near copies of f
    col.add(torch.cat([f, g], dim=1), [3] * 100)
    col.add(f.clone(), [3] * 50)  # exact copies arrive later: all rejected
    col.dump()
    kept = torch.as_tensor(load_coefs(tmp_path / "c_ovals3.txt"), dtype=DTYPE, device=DEVICE).T
    K = descriptors(kept, 5, True)
    G = (K @ K.T).abs() - torch.eye(K.shape[0], dtype=DTYPE, device=DEVICE)
    assert K.shape[0] <= 50 and float(G.max()) < math.cos(math.radians(10.0)) + 1e-6
    assert col.seen[3] == 150 and col.total[3] == K.shape[0]
