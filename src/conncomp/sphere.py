"""An exact lattice model of S^2 / RP^2 in homogeneous coordinates, for studying a single curve.

The lattice consists of the integer points p in Z^3 with max |p_i| = N (the surface of the cube
[-N, N]^3). Every lattice point is an exact homogeneous coordinate vector, rays through the
lattice cover S^2 evenly, and the lattice is symmetric under the antipodal map p -> -p. Two points
are neighbours if they differ by at most 1 in each coordinate. The segment (1 - t) p + t q between
neighbours never passes through 0, so it is a short geodesic arc that can be certified exactly
(see conncomp.certify.certify_segment).

(The batched search instead uses the stereographic pixel grid of conncomp.grid, which is
optimized for evaluating many forms at once in C++.)
"""

from collections import deque
from itertools import product

import numpy as np

DIRECTIONS = [d for d in product((-1, 0, 1), repeat=3) if d != (0, 0, 0)]


class CubeLattice:
    """The integer points of the surface of the cube [-N, N]^3."""

    def __init__(self, N):
        self.N = N
        r = np.arange(-N, N + 1)
        u, v = (a.ravel() for a in np.meshgrid(r, r, indexing="ij"))
        faces = []
        for axis in range(3):
            for s in (-N, N):
                p = np.empty((u.size, 3), dtype=np.int64)
                others = [b for b in range(3) if b != axis]
                p[:, axis] = s
                p[:, others[0]] = u
                p[:, others[1]] = v
                # each edge/corner point is kept on the face of its smallest axis with |p_axis| = N
                keep = np.all(np.abs(p[:, :axis]) < N, axis=1) if axis else np.ones(u.size, dtype=bool)
                faces.append(p[keep])
        self.points = np.concatenate(faces)
        self.index = {p: i for i, p in enumerate(map(tuple, self.points.tolist()))}

    def __len__(self):
        return len(self.points)

    def point(self, i):
        return [int(v) for v in self.points[i]]

    def antipode(self, i):
        return self.index[tuple(-v for v in self.points[i].tolist())]

    def neighbours(self, i):
        p = self.points[i].tolist()
        for d in DIRECTIONS:
            q = (p[0] + d[0], p[1] + d[1], p[2] + d[2])
            j = self.index.get(q)
            if j is not None:
                yield j

    def nearest(self, P):
        """Index of the lattice point on the ray through P (rounded)."""
        P = np.asarray([float(v) for v in P])
        q = np.rint(P * self.N / np.abs(P).max()).astype(np.int64)
        return self.index[tuple(q.tolist())]

    def faces(self):
        """Unit squares of the cube surface as an (F, 4) index array (corners in cyclic order).

        Together with `edges` this is a quad mesh of S^2 (V - E + F = 2), symmetric under p -> -p.
        """
        N, out = self.N, []
        r = range(-N, N)
        for axis in range(3):
            o0, o1 = [b for b in range(3) if b != axis]
            for s in (-N, N):
                for u in r:
                    for v in r:
                        quad = []
                        for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                            p = [0, 0, 0]
                            p[axis], p[o0], p[o1] = s, u + du, v + dv
                            quad.append(self.index[tuple(p)])
                        out.append(quad)
        return np.array(out, dtype=np.int64)

    def edges(self):
        """Edges of the quad mesh (see `faces`) as an (E, 2) index array."""
        F = self.faces()
        e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 3]], F[:, [3, 0]]])
        return np.unique(np.sort(e, axis=1), axis=0)

    def values(self, terms):
        """Values of a form at the unit vectors p/|p|; `terms` is a list of ((i, j, k), coefficient)."""
        u = self.points / np.linalg.norm(self.points, axis=1, keepdims=True)
        out = np.zeros(len(u))
        for (i, j, k), c in terms:
            out += float(c) * u[:, 0] ** i * u[:, 1] ** j * u[:, 2] ** k
        return out


def path_to_antipode(lattice, vals, start):
    """Shortest lattice path from `start` to its antipode through points where `vals` has the sign of vals[start].

    Returns (path, visited): the path as a list of indices, or None if the antipode is not reachable
    (e.g. `start` lies inside an oval); `visited` is the set of explored points (the whole lattice
    component of `start` when the path is None).
    """
    s = np.sign(vals[start])
    if s == 0:
        return None, {start}
    goal = lattice.antipode(start)
    parent = {start: None}
    queue = deque([start])
    while queue:
        i = queue.popleft()
        if i == goal:
            path = []
            while i is not None:
                path.append(i)
                i = parent[i]
            return path[::-1], set(parent)
        for j in lattice.neighbours(i):
            if j not in parent and np.sign(vals[j]) == s:
                parent[j] = i
                queue.append(j)
    return None, set(parent)
