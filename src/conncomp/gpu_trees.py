"""Region trees (oval counts and nesting types) on the GPU: Triton union-find labelling on a mesh of S^2, followed
by tensor operations in PyTorch. Same definitions as the C++ baseline (conncomp._connected.region_trees_mesh,
conncomp.nesting): components on S^2 with 8/4 connectivity (edges for both sign classes, diagonals only for the
class `eight`); a region of RP^2 is a component with its antipodal image; the root N is the component equal to its
antipodal image; small leaf regions are pruned.

Labelling (three kernels over the flattened batch, node index b * V + p):
  init      parent[i] = i
  merge     for each same-sign neighbour j > i: union(i, j); roots found with path halving, linked with
            atomic_cas (larger root index under the smaller), retried until the roots agree
  compress  parent[i] = root(i), with read-only root finding (path-halving writes here would race with the
            final stores of other threads)
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

BLOCK = 1024  # init / compress
MERGE_BLOCK, MERGE_WARPS = 32, 1  # merge: small blocks, since the loops run until the slowest lane of a block is done


@triton.jit
def _init_kernel(parent_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(parent_ptr + i, i, mask=i < n)


@triton.jit
def _find(parent_ptr, x, active):
    """Roots of the nodes x (lanes with `active`), with path halving."""
    moving = active
    while tl.max(moving.to(tl.int32), axis=0) > 0:
        p = tl.load(parent_ptr + x, mask=moving, other=0, volatile=True)
        moving = moving & (p != x)
        gp = tl.load(parent_ptr + p, mask=moving, other=0, volatile=True)
        tl.store(parent_ptr + x, gp, mask=moving & (gp != p))  # path halving: x -> grandparent (an ancestor)
        x = tl.where(moving, gp, x)
    return x


@triton.jit
def _find_readonly(parent_ptr, x, active):
    """Roots of the nodes x without writes (used after merging: halving writes would race with final stores)."""
    moving = active
    while tl.max(moving.to(tl.int32), axis=0) > 0:
        p = tl.load(parent_ptr + x, mask=moving, other=0, volatile=True)
        moving = moving & (p != x)
        x = tl.where(moving, p, x)
    return x


@triton.jit
def _union(parent_ptr, scratch, i, j, active):
    """Lock-free union of the sets of i and j for the active lanes."""
    todo = active
    while tl.max(todo.to(tl.int32), axis=0) > 0:
        ri = _find(parent_ptr, i, todo)
        rj = _find(parent_ptr, j, todo)
        todo = todo & (ri != rj)
        hi = tl.maximum(ri, rj)
        lo = tl.minimum(ri, rj)
        # atomic_cas has no mask: inactive lanes target a scratch slot with an impossible comparison value
        ptr = tl.where(todo, parent_ptr + hi, parent_ptr + scratch)
        cmp = tl.where(todo, hi, -1)
        old = tl.atomic_cas(ptr, cmp, lo)
        todo = todo & (old != hi)  # lost a race: retry with the new roots


@triton.jit
def _merge_kernel(signs_ptr, parent_ptr, nbr4_ptr, nbr8_ptr, V, n, scratch, eight,
                  K4: tl.constexpr, K8: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    b = i // V
    p = i - b * V
    s = tl.load(signs_ptr + i, mask=m, other=0)
    for k in tl.static_range(K4):
        q = tl.load(nbr4_ptr + p * K4 + k, mask=m, other=-1)
        ok = m & (q > p)
        sq = tl.load(signs_ptr + b * V + q, mask=ok, other=255)
        ok = ok & (sq == s)
        _union(parent_ptr, scratch, i, b * V + q, ok)
    for k in tl.static_range(K8):
        q = tl.load(nbr8_ptr + p * K8 + k, mask=m, other=-1)
        ok = m & (q > p) & (s == eight)
        sq = tl.load(signs_ptr + b * V + q, mask=ok, other=255)
        ok = ok & (sq == s)
        _union(parent_ptr, scratch, i, b * V + q, ok)


@triton.jit
def _compress_kernel(parent_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    r = _find_readonly(parent_ptr, i, m)
    tl.store(parent_ptr + i, r, mask=m)


def label_components(signs, nbr4, nbr8, eight=1):
    """Component labels (root node index, global over the batch) for sign vectors (B, V) uint8 on the GPU."""
    B, V = signs.shape
    n = B * V
    if n >= 2**31 - 1:
        raise ValueError("batch too large for int32 node indices; split it")
    parent = torch.empty(n + 1, dtype=torch.int32, device=signs.device)  # last slot: scratch for atomic_cas
    grid = (triton.cdiv(n + 1, BLOCK),)
    _init_kernel[grid](parent, n + 1, BLOCK=BLOCK)
    _merge_kernel[(triton.cdiv(n, MERGE_BLOCK),)](signs.reshape(-1), parent, nbr4, nbr8, V, n, n, eight,
                                                 K4=nbr4.shape[1], K8=nbr8.shape[1], BLOCK=MERGE_BLOCK,
                                                 num_warps=MERGE_WARPS)
    _compress_kernel[grid](parent, n, BLOCK=BLOCK)
    return parent[:n].view(B, V)


@dataclass
class GPUTrees:
    """Region trees of a batch, as flat tensors over all regions (global region ids 0..R-1)."""

    n_ovals: torch.Tensor  # (B,) number of ovals (regions - 1 after pruning)
    ok: torch.Tensor  # (B,) exactly one non-orientable region and a tree
    form: torch.Tensor  # (R,) form of each region
    size: torch.Tensor  # (R,) vertices of S^2 in the region (both lifts)
    sign: torch.Tensor  # (R,) +1 / -1
    alive: torch.Tensor  # (R,) not pruned
    is_root: torch.Tensor  # (R,)
    parent: torch.Tensor  # (R,) parent region (global id), -1 for roots and pruned regions
    depth: torch.Tensor  # (R,) depth in the tree, -1 if unreached or pruned


def _region_trees_chunk(signs, mesh_tables, min_size, eight):
    nbr4, nbr8, anti = mesh_tables
    labels = label_components(signs, nbr4, nbr8, eight)
    return _trees_from_labels(signs, labels, nbr4, anti, None, min_size)


def _trees_from_labels(signs, labels, nbr4, anti, weight, min_size, sign_pairs=None):
    """Region trees from component labels (root node ids, global over the batch) of sign vectors (B, V).

    nbr4 (V, K): neighbours used for region adjacency; anti (V,): antipodal node; weight (V,) or None: weight of
    each node in the region sizes (0 for duplicated nodes, e.g. cube-edge points on several face grids).
    sign_pairs: optional function (signs) -> (p, q) global node indices of the neighbouring pairs with different signs
    (only those can join different regions); default: all nbr4 pairs are checked.
    """
    B, V = signs.shape
    n = B * V
    dev = signs.device
    labels = labels.reshape(-1)
    nodes = torch.arange(n, dtype=torch.int32, device=dev)
    rootmask = labels == nodes
    rnodes = nodes[rootmask].long()  # one representative (the root) per component on S^2
    C = rnodes.numel()
    comp_id = torch.empty(n, dtype=torch.int32, device=dev)
    comp_id[rnodes] = torch.arange(C, dtype=torch.int32, device=dev)
    comp = comp_id[labels.long()]  # (n,) component of each node
    # antipodal image of each component, region of RP^2 = {component, antipodal image}
    rb, rp = rnodes // V, rnodes % V
    comp_anti = comp[rb * V + anti.long()[rp]].long()
    cid = torch.arange(C, device=dev)
    region_key = torch.minimum(cid, comp_anti)
    keys, reg_of_comp = torch.unique(region_key, return_inverse=True)
    R = keys.numel()
    form = (rb[keys]).long()
    if weight is None:
        csize = torch.bincount(comp.long(), minlength=C)
    else:
        csize = torch.zeros(C, dtype=torch.long, device=dev).index_add_(0, comp.long(), weight.long().repeat(B))
    size = torch.zeros(R, dtype=torch.long, device=dev).index_add_(0, reg_of_comp, csize)
    sign = signs.reshape(-1)[rnodes[keys]].long() * 2 - 1
    is_root = torch.zeros(R, dtype=torch.bool, device=dev)
    is_root[reg_of_comp[comp_anti == cid]] = True
    n_roots = torch.bincount(form[is_root], minlength=B)
    # edges between adjacent regions (only neighbouring nodes of different signs can be in different regions)
    if sign_pairs is not None:
        pp, qq = sign_pairs(signs)
        rp_, rq_ = reg_of_comp[comp[pp].long()], reg_of_comp[comp[qq].long()]
        e = torch.unique(torch.minimum(rp_, rq_) * R + torch.maximum(rp_, rq_))
    else:
        reg_node = reg_of_comp[comp.long()].view(B, V)
        pairs = []
        for k in range(nbr4.shape[1]):
            q = nbr4[:, k].long()
            valid = q >= 0
            rp_ = reg_node[:, valid]
            rq_ = reg_node[:, q[valid]]
            d = rp_ != rq_
            if d.any():
                a, c = torch.minimum(rp_, rq_)[d], torch.maximum(rp_, rq_)[d]
                pairs.append(torch.unique(a * R + c))
        e = torch.unique(torch.cat(pairs)) if pairs else torch.zeros(0, dtype=torch.long, device=dev)
    ea, eb = e // R, e % R
    # prune small leaves (not roots), all current leaves at once, until stable
    alive = torch.ones(R, dtype=torch.bool, device=dev)
    while True:
        ealive = alive[ea] & alive[eb]
        deg = torch.bincount(ea[ealive], minlength=R) + torch.bincount(eb[ealive], minlength=R)
        kill = alive & ~is_root & (size < min_size) & (deg <= 1)
        if not bool(kill.any()):
            break
        alive &= ~kill
    ealive = alive[ea] & alive[eb]
    ea, eb = ea[ealive], eb[ealive]
    # depths and parents by frontier propagation from the roots
    depth = torch.full((R,), -1, dtype=torch.long, device=dev)
    parent = torch.full((R,), -1, dtype=torch.long, device=dev)
    depth[is_root & alive] = 0
    d = 0
    while True:
        fa = (depth[ea] == d) & (depth[eb] < 0)
        fb = (depth[eb] == d) & (depth[ea] < 0)
        if not bool(fa.any() or fb.any()):
            break
        parent[eb[fa]] = ea[fa]
        parent[ea[fb]] = eb[fb]
        depth[eb[fa]] = d + 1
        depth[ea[fb]] = d + 1
        d += 1
    n_alive = torch.bincount(form[alive], minlength=B)
    n_edges = torch.bincount(form[ea], minlength=B)
    reached = torch.bincount(form[alive & (depth >= 0)], minlength=B)
    ok = (n_roots == 1) & (n_edges == n_alive - 1) & (reached == n_alive)
    return GPUTrees(n_alive - 1, ok, form, size, sign, alive, is_root & alive, parent, depth)


def region_trees(signs, mesh_tables, min_size=3, eight=1, max_nodes=4e7):
    """Region trees for sign vectors (B, V) uint8 on the GPU; mesh_tables = (nbr4, nbr8, antipode) GPU tensors.

    The batch is processed in chunks of at most `max_nodes` mesh nodes; results are concatenated (global ids).
    """
    B, V = signs.shape
    chunk = max(1, int(max_nodes // V))
    parts, f_off, r_off = [], 0, 0
    for s0 in range(0, B, chunk):
        t = _region_trees_chunk(signs[s0 : s0 + chunk], mesh_tables, min_size, eight)
        t.form = t.form + f_off
        t.parent = torch.where(t.parent >= 0, t.parent + r_off, t.parent)
        parts.append(t)
        f_off += signs[s0 : s0 + chunk].shape[0]
        r_off += t.form.numel()
    if len(parts) == 1:
        return parts[0]
    return GPUTrees(*[torch.cat([getattr(p, f) for p in parts]) for f in GPUTrees.__dataclass_fields__])


def to_nesting_trees(t, B):
    """Convert GPUTrees to a list of conncomp.nesting.NestingTree (for type strings)."""
    import numpy as np

    from conncomp.nesting import NestingTree

    form = t.form.cpu().numpy()
    alive = t.alive.cpu().numpy()
    size, sign, parent, depth = (x.cpu().numpy() for x in (t.size, t.sign, t.parent, t.depth))
    is_root, ok = t.is_root.cpu().numpy(), t.ok.cpu().numpy()
    out = []
    order = np.argsort(form, kind="stable")
    bounds = np.searchsorted(form[order], np.arange(B + 1))
    for b in range(B):
        ids = order[bounds[b]:bounds[b + 1]]
        ids = ids[alive[ids]]
        local = {g: k for k, g in enumerate(ids)}
        par = np.array([local.get(parent[g], -1) for g in ids])
        roots = [local[g] for g in ids if is_root[g]]
        out.append(NestingTree(size[ids], sign[ids], par, depth[ids], roots[0] if roots else -1, bool(ok[b])))
    return out


# ---------------------------------------------------------------------------
# CUDA backend: block-based union-find on the six cube-face grids (conncomp._connected_cuda)
# ---------------------------------------------------------------------------


class FaceMesh:
    """The surface of the cube [-N, N]^3 as six (2N+1) x (2N+1) face grids (points on cube edges and corners are
    duplicated on 2 or 3 faces). Same vertices, edges and diagonals as conncomp.nesting.SphereMesh(N)."""

    def __init__(self, N=30, device=None):
        import numpy as np

        from conncomp import DEVICE, DTYPE
        from conncomp.euler import face_grid_points

        self.device = device or DEVICE
        self.N, self.M = N, 2 * N + 1
        M = self.M
        P = face_grid_points(N).reshape(-1, 3)  # (6 M^2, 3), face f = 2 * axis + side
        self.n_loc = P.shape[0]
        first, canon = {}, np.empty(self.n_loc, dtype=np.int64)
        for k, p in enumerate(map(tuple, P.tolist())):
            canon[k] = first.setdefault(p, k)
        dup = np.nonzero(canon != np.arange(self.n_loc))[0]
        self.seams = torch.as_tensor(np.stack([dup, canon[dup]], 1).astype(np.int32), device=self.device)
        self.weight = torch.as_tensor((canon == np.arange(self.n_loc)).astype(np.int32), device=self.device)
        f, i, j = np.unravel_index(np.arange(self.n_loc), (6, M, M))
        self.anti = torch.as_tensor(np.ravel_multi_index((f ^ 1, M - 1 - i, M - 1 - j), (6, M, M)), device=self.device)
        nbr = -np.ones((self.n_loc, 4), dtype=np.int64)
        for k, (di, dj) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
            ok = (i + di >= 0) & (i + di < M) & (j + dj >= 0) & (j + dj < M)
            nbr[ok, k] = np.ravel_multi_index((f[ok], i[ok] + di, j[ok] + dj), (6, M, M))
        self.nbr4 = torch.as_tensor(nbr, device=self.device)
        U = torch.as_tensor(P / np.linalg.norm(P, axis=1, keepdims=True), dtype=DTYPE, device=self.device)
        self.points = U
        self._basis = {}

    def signs(self, coefs, deg, chunk=None):
        """uint8 sign classes at the face-grid points for forms (D, B); shape (B, 6 M^2)."""
        from conncomp.polynomials import monomial_basis, monomials

        if deg not in self._basis:
            self._basis[deg] = monomial_basis(monomials(deg), self.points.T)
        basis = self._basis[deg]
        chunk = chunk or max(16, int(8e7 // self.n_loc))
        out = torch.empty((coefs.shape[1], self.n_loc), dtype=torch.uint8, device=coefs.device)
        for s0 in range(0, coefs.shape[1], chunk):
            out[s0 : s0 + chunk] = (coefs[:, s0 : s0 + chunk].T @ basis) >= 0
        return out

    def sign_pairs(self, signs):
        """Global node indices (p, q) of horizontally / vertically neighbouring face-grid nodes of different signs."""
        B, M = signs.shape[0], self.M
        g = signs.view(B, 6, M, M)
        h = (g[..., :, 1:] != g[..., :, :-1]).nonzero()  # (b, f, i, j): nodes (i, j) and (i, j + 1)
        v = (g[..., 1:, :] != g[..., :-1, :]).nonzero()  # nodes (i, j) and (i + 1, j)
        def node(t, di, dj):
            return ((t[:, 0] * 6 + t[:, 1]) * M + t[:, 2] + di) * M + t[:, 3] + dj
        return torch.cat([node(h, 0, 0), node(v, 0, 0)]), torch.cat([node(h, 0, 1), node(v, 1, 0)])

    def labels(self, signs, eight=1):
        """Component labels (root node index, global over the batch) with the CUDA kernels; signs (B, 6 M^2)."""
        from conncomp import _connected_cuda

        signs = signs.contiguous()
        B = signs.shape[0]
        parent = torch.empty((B, self.n_loc), dtype=torch.int32, device=signs.device)
        _connected_cuda.label_faces(signs.data_ptr(), parent.data_ptr(), self.seams.data_ptr(), self.seams.shape[0],
                                    B, self.M, eight, torch.cuda.current_stream().cuda_stream)
        return parent

    def trees(self, coefs, deg, min_size=3, eight=1, max_nodes=4e7):
        """Region trees (GPUTrees) of the curves {form = 0} for forms (D, B), labelled with the CUDA kernels."""
        signs = self.signs(coefs, deg)
        B = signs.shape[0]
        chunk = max(1, int(max_nodes // self.n_loc))
        parts, f_off, r_off = [], 0, 0
        for s0 in range(0, B, chunk):
            sc = signs[s0 : s0 + chunk]
            t = _trees_from_labels(sc, self.labels(sc, eight), self.nbr4, self.anti, self.weight, min_size,
                                   sign_pairs=self.sign_pairs)
            t.form = t.form + f_off
            t.parent = torch.where(t.parent >= 0, t.parent + r_off, t.parent)
            parts.append(t)
            f_off += sc.shape[0]
            r_off += t.form.numel()
        if len(parts) == 1:
            return parts[0]
        return GPUTrees(*[torch.cat([getattr(p, f) for p in parts]) for f in GPUTrees.__dataclass_fields__])
