"""Nesting types of oval configurations of even-degree curves in RP^2 (region trees).

The regions of RP^2 minus the curve are the vertices of a tree whose edges are the ovals (each oval separates
RP^2). The root is the non-orientable region N (outside all ovals); the children of a region R are the regions
just inside the ovals that bound R from inside. The type is written recursively, as in the literature:
type(R) = sum over the ovals bounding R from inside of 1<type(inner region)>, with 1<empty> written as 1, e.g.
"9 u 1<1>" (Harnack), "1 u 1<9>" (Hilbert), "5 u 1<5>" (Gudkov), "10" (ten empty ovals), "0" (empty curve).

The trees are computed in C++ (conncomp._connected.region_trees_batch) on the stereographic grid of
conncomp.grid, via the double cover S^2 -> RP^2: N is the unique region whose lift to S^2 is connected. Regions
smaller than `min_size` pixels that are leaves of the tree are pruned (spurious ovals).
"""

from dataclasses import dataclass

import numpy as np
import torch

from conncomp import _connected


@dataclass
class NestingTree:
    size: np.ndarray  # pixels per region
    sign: np.ndarray  # +1 / -1 per region
    parent: np.ndarray  # parent region (-1 for the root)
    depth: np.ndarray  # depth in the tree (root = 0)
    root: int
    ok: bool  # exactly one non-orientable region and the region graph is a tree

    @property
    def n_ovals(self):
        return len(self.size) - 1

    @property
    def max_depth(self):
        return int(self.depth.max()) if len(self.depth) else 0

    @property
    def outside_sign(self):
        return int(self.sign[self.root]) if self.root >= 0 else 0

    def children(self, r):
        return np.nonzero(self.parent == r)[0]

    def type_of(self, r):
        kids = self.children(r)
        empty = sum(1 for c in kids if len(self.children(c)) == 0)
        nested = sorted(f"1<{self.type_of(c)}>" for c in kids if len(self.children(c)) > 0)
        parts = ([str(empty)] if empty else []) + nested
        return " u ".join(parts) if parts else "0"

    @property
    def type(self):
        return self.type_of(self.root) if self.ok else "?"


def nesting_trees(vals, pat, min_size=3, num_threads=-1):
    """Region trees for a batch of grids `vals` (values or uint8 signs, shape (N, W, W)); list of NestingTree."""
    if isinstance(vals, torch.Tensor):
        vals = (vals >= 0).to(torch.uint8).cpu().numpy() if vals.dtype != torch.uint8 else vals.cpu().numpy()
    elif vals.dtype != np.uint8:
        vals = np.ascontiguousarray(vals >= 0, dtype=np.uint8)
    res = _connected.region_trees_batch(vals, pat, min_size, num_threads)
    off, size, sign, parent, depth, root, nroots, tree, _ = res
    out = []
    for i in range(len(root)):
        a, b = off[i], off[i + 1]
        out.append(NestingTree(size[a:b], sign[a:b], parent[a:b], depth[a:b], int(root[i]),
                               bool(nroots[i] == 1 and tree[i] and root[i] >= 0)))
    return out


# ---------------------------------------------------------------------------
# Region trees on the cube-surface mesh of S^2 (exact antipodal symmetry, no chart seams)
# ---------------------------------------------------------------------------


class SphereMesh:
    """The cube-surface lattice of conncomp.sphere as a mesh of S^2: 4-neighbours (edges), diagonal neighbours
    (used by the 8-connected class only) and the antipodal map. Forms are evaluated at the unit lattice points."""

    def __init__(self, N=100, device=None):
        from conncomp import DEVICE, DTYPE
        from conncomp.sphere import CubeLattice

        lat = CubeLattice(N)
        V = len(lat)
        self.N, self.V = N, V
        F = lat.faces()
        E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 3]], F[:, [3, 0]]]), axis=1), axis=0)
        D = np.unique(np.sort(np.concatenate([F[:, [0, 2]], F[:, [1, 3]]]), axis=1), axis=0)
        self.nbr4 = self._table(E, V, 4)
        self.nbr8 = self._table(D, V, 4)
        pts = lat.points
        self.antipode = np.array([lat.index[tuple(-v for v in p)] for p in pts.tolist()], dtype=np.int32)
        self.device = device or DEVICE
        P = torch.as_tensor(pts / np.linalg.norm(pts, axis=1, keepdims=True), dtype=DTYPE, device=self.device)
        self.points = P
        self._basis = {}

    @staticmethod
    def _table(pairs, V, K):
        out = -np.ones((V, K), dtype=np.int32)
        fill = np.zeros(V, dtype=np.int64)
        for a, b in pairs:
            out[a, fill[a]] = b
            fill[a] += 1
            out[b, fill[b]] = a
            fill[b] += 1
        return out

    def signs(self, coefs, deg, chunk=None):
        """uint8 sign classes (1 where the form is >= 0) at the mesh points for forms (D, B); shape (B, V)."""
        from conncomp.polynomials import monomial_basis, monomials

        if deg not in self._basis:
            self._basis[deg] = monomial_basis(monomials(deg), self.points.T)  # (D, V)
        basis = self._basis[deg]
        chunk = chunk or max(16, int(8e7 // self.V))
        out = torch.empty((coefs.shape[1], self.V), dtype=torch.uint8, device=coefs.device)
        for s in range(0, coefs.shape[1], chunk):
            out[s : s + chunk] = (coefs[:, s : s + chunk].T @ basis) >= 0
        return out

    def trees(self, coefs, deg, min_size=3, eight=1, num_threads=-1):
        """Region trees (NestingTree) of the curves {form = 0} for forms (D, B) of degree deg."""
        s = self.signs(coefs, deg).cpu().numpy()
        off, size, sign, parent, depth, root, nroots, tree, _ = _connected.region_trees_mesh(
            s, self.nbr4, self.nbr8, self.antipode, eight, min_size, num_threads)
        return [NestingTree(size[off[i]:off[i + 1]], sign[off[i]:off[i + 1]], parent[off[i]:off[i + 1]],
                            depth[off[i]:off[i + 1]], int(root[i]), bool(nroots[i] == 1 and tree[i] and root[i] >= 0))
                for i in range(len(root))]
