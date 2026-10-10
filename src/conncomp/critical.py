"""Critical points of forms on RP^2, batched on the GPU.

For a form H of even degree n, the critical points of h = H restricted to the unit sphere (modulo p -> -p)
are the points where grad H(p) is parallel to p (Lagrange; by Euler's identity the multiplier is n H(p)).
A generic form has n^2 - n + 1 complex critical points on P^2.

Charts: every point of RP^2 has a representative with its largest coordinate positive, so the three cube
faces x = 1, y = 1, z = 1 cover RP^2. Face k is parametrized by (u, v) in [-(1 + overlap), 1 + overlap]^2
(the overlap keeps critical points near the seams away from the chart border). Since H is homogeneous, h is
H evaluated at the normalized points, and the critical points of h(u, v) in a chart are those on the sphere.

Seeding (this module, step 1): on each face grid, discrete local maxima and minima (comparison with the 8
neighbours, via max_pool2d) and discrete saddles (at least 4 sign changes of h(neighbour) - h(centre) around
the 8-ring). Border pixels are skipped; the overlap covers them. Seeds are later refined by Newton's method.
"""

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F

from conncomp import DEVICE, DTYPE
from conncomp.hessian import hessian
from conncomp.polynomials import monomial_basis, monomials

MIN, SADDLE, MAX = 0, 1, 2

# the 8-neighbour ring in cyclic order
RING = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def face_point(face, u, v):
    """Points of face k (k = 0, 1, 2: coordinate k equals 1) for chart coordinates u, v (any shape)."""
    one = torch.ones_like(u)
    if face == 0:
        return torch.stack([one, u, v], dim=-1)
    if face == 1:
        return torch.stack([u, one, v], dim=-1)
    return torch.stack([u, v, one], dim=-1)


@dataclass
class Seeds:
    """Flat list of seeds: form index, face, grid indices, chart coordinates, unit point, value, kind."""

    form: torch.Tensor
    face: torch.Tensor
    i: torch.Tensor
    j: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    point: torch.Tensor  # (K, 3) unit vectors
    value: torch.Tensor  # h at the seed (with the form normalized to unit coefficient norm)
    kind: torch.Tensor  # MIN, SADDLE or MAX

    def __len__(self):
        return self.form.numel()


class FaceCharts:
    """The three face charts of RP^2 on an (M x M) grid each, M = 2N + 1, for forms of degree `deg`."""

    def __init__(self, deg, N=40, overlap=0.1, device=DEVICE):
        self.deg, self.N, self.M = deg, N, 2 * N + 1
        self.t = torch.linspace(-(1 + overlap), 1 + overlap, self.M, dtype=DTYPE, device=device)
        uu, vv = torch.meshgrid(self.t, self.t, indexing="ij")
        P = torch.stack([face_point(k, uu, vv) for k in range(3)])  # (3, M, M, 3)
        self.points = P / P.norm(dim=-1, keepdim=True)
        self.basis = monomial_basis(monomials(deg), self.points.permute(3, 0, 1, 2))  # (D, 3, M, M)
        self.spacing = float(self.t[1] - self.t[0])

    def values(self, coefs):
        """h on the face grids for a batch of forms (D, B), normalized to unit norm; shape (B, 3, M, M)."""
        c = coefs / coefs.norm(dim=0, keepdim=True)
        return torch.tensordot(c.T, self.basis, dims=1)

    def seeds(self, coefs, chunk=None):
        """Discrete critical points of h for a batch of forms (D, B); see the module docstring.

        Forms are processed in chunks of about 8e7 grid values (default), to bound GPU memory.
        """
        chunk = chunk or max(64, int(8e7 // (3 * self.M * self.M)))
        parts = []
        for s0 in range(0, coefs.shape[1], chunk):
            h = self.values(coefs[:, s0 : s0 + chunk])
            parts.append(_seed_chunk(h, s0, self.t, self.points))
        return Seeds(*[torch.cat([p[k] for p in parts]) for k in range(9)])


def _seed_chunk(h, offset, t, points):
    B, _, M, _ = h.shape
    flat = h.reshape(B * 3, 1, M, M)
    is_max = flat == F.max_pool2d(flat, 3, stride=1, padding=1)
    is_min = -flat == F.max_pool2d(-flat, 3, stride=1, padding=1)
    # sign changes of h(neighbour) - h(centre) around the 8-ring (interior pixels only)
    c = h[:, :, 1:-1, 1:-1]
    signs = [h[:, :, 1 + di : M - 1 + di, 1 + dj : M - 1 + dj] > c for di, dj in RING]
    changes = sum((signs[k] != signs[(k + 1) % 8]).to(torch.int8) for k in range(8))
    kind = torch.full((B, 3, M - 2, M - 2), -1, dtype=torch.int8, device=h.device)
    kind[changes >= 4] = SADDLE
    kind[is_min.reshape(B, 3, M, M)[:, :, 1:-1, 1:-1]] = MIN
    kind[is_max.reshape(B, 3, M, M)[:, :, 1:-1, 1:-1]] = MAX
    b, f, i, j = (kind >= 0).nonzero(as_tuple=True)
    i, j = i + 1, j + 1
    return (b + offset, f, i, j, t[i], t[j], points[f, i, j], h[b, f, i, j], kind[b, f, i - 1, j - 1].long())


# ---------------------------------------------------------------------------
# Step 3: batched Newton refinement in the face charts
# ---------------------------------------------------------------------------


def _face_exponents(deg, device=DEVICE):
    """Exponents (a, b) of u and v in each monomial, for the three faces: tensors (3, D)."""
    mons = torch.tensor(monomials(deg), device=device)  # (D, 3) exponents of X, Y, Z
    free = [(1, 2), (0, 2), (0, 1)]  # face k: the coordinates set to u and v
    A = torch.stack([mons[:, a] for a, _ in free])
    B = torch.stack([mons[:, b] for _, b in free])
    return A, B


def chart_derivatives(c, face, u, v, deg):
    """h, grad h, Hessian of h(u, v) = H(P(u, v)) / |P|^deg in the face charts, for K points at once.

    c: (K, D) coefficients of the form of each point; face, u, v: (K,). Returns h (K,), g (K, 2), Hm (K, 2, 2).
    """
    A, B = _face_exponents(deg, c.device)
    a, b = A[face].to(c.dtype), B[face].to(c.dtype)  # (K, D)
    ai, bi = A[face], B[face]
    powers = torch.arange(deg + 1, device=c.device, dtype=c.dtype)
    Up, Vp = u[:, None] ** powers, v[:, None] ** powers  # (K, deg + 1)

    def P(table, e):  # table[k, max(e, 0)]
        return torch.gather(table, 1, e.clamp(min=0))

    ua, ua1, ua2 = P(Up, ai), P(Up, ai - 1), P(Up, ai - 2)
    vb, vb1, vb2 = P(Vp, bi), P(Vp, bi - 1), P(Vp, bi - 2)
    g = (c * ua * vb).sum(1)
    gu = (c * a * ua1 * vb).sum(1)
    gv = (c * b * ua * vb1).sum(1)
    guu = (c * a * (a - 1) * ua2 * vb).sum(1)
    guv = (c * a * b * ua1 * vb1).sum(1)
    gvv = (c * b * (b - 1) * ua * vb2).sum(1)
    n = deg
    s = 1 + u * u + v * v
    w = s ** (-n / 2)
    wu, wv = -n * u * w / s, -n * v * w / s
    wuu = w * (n * (n + 2) * u * u / s**2 - n / s)
    wuv = w * n * (n + 2) * u * v / s**2
    wvv = w * (n * (n + 2) * v * v / s**2 - n / s)
    h = g * w
    grad = torch.stack([gu * w + g * wu, gv * w + g * wv], dim=1)
    huu = guu * w + 2 * gu * wu + g * wuu
    huv = guv * w + gu * wv + gv * wu + g * wuv
    hvv = gvv * w + 2 * gv * wv + g * wvv
    hess = torch.stack([torch.stack([huu, huv], 1), torch.stack([huv, hvv], 1)], 1)
    return h, grad, hess


@dataclass
class Critical:
    """Refined critical points (flat list): form, face, chart coordinates, unit point, value, kind, diagnostics."""

    form: torch.Tensor
    face: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    point: torch.Tensor
    value: torch.Tensor
    kind: torch.Tensor
    grad_norm: torch.Tensor
    det: torch.Tensor
    converged: torch.Tensor

    def __len__(self):
        return self.form.numel()

    def select(self, mask):
        return Critical(*[getattr(self, f)[mask] for f in self.__dataclass_fields__])


def _newton_step(g, Hm, eig_floor):
    """-Hm^-1 g for symmetric 2 x 2 matrices, with eigenvalues floored in magnitude (sign kept). Closed form."""
    a, b, c = Hm[:, 0, 0], Hm[:, 0, 1], Hm[:, 1, 1]
    m, r = (a + c) / 2, torch.sqrt(((a - c) / 2) ** 2 + b * b)
    lam1, lam2 = m - r, m + r
    # eigenvector of lam1: (b, lam1 - a) or (lam1 - c, b), whichever is longer; (1, 0) for scalar matrices
    q1 = torch.stack([b, lam1 - a], dim=1)
    q2 = torch.stack([lam1 - c, b], dim=1)
    q = torch.where((q1.norm(dim=1) >= q2.norm(dim=1))[:, None], q1, q2)
    qn = q.norm(dim=1, keepdim=True)
    e1 = torch.tensor([1.0, 0.0], dtype=g.dtype, device=g.device).expand_as(q)
    q = torch.where(qn > 0, q / qn.clamp(min=1e-300), e1)
    p = torch.stack([-q[:, 1], q[:, 0]], dim=1)  # eigenvector of lam2

    def floor(lam):
        return torch.where(lam.abs() < eig_floor, torch.where(lam < 0, -eig_floor, eig_floor), lam)

    return -(q * ((q * g).sum(1) / floor(lam1))[:, None] + p * ((p * g).sum(1) / floor(lam2))[:, None])


def refine(seeds, coefs, deg, iters=8, max_step=None, tol=1e-12, eig_floor=1e-10, bound=1.25):
    """Newton refinement of seeds (see `FaceCharts.seeds`) in their face charts.

    The step is -Hm^-1 grad, with the eigenvalues of the 2 x 2 Hessian floored in magnitude (sign kept, so
    saddles converge as well as extrema) and the step length capped at max_step. A point is converged if
    |grad h| < tol (forms normalized to unit coefficient norm) and it stays in the extended chart |u|, |v| < bound.
    """
    c = coefs / coefs.norm(dim=0, keepdim=True)
    C = c.T[seeds.form]  # (K, D)
    u, v, face = seeds.u.clone(), seeds.v.clone(), seeds.face
    max_step = max_step if max_step is not None else 0.05
    for _ in range(iters):
        h, g, Hm = chart_derivatives(C, face, u, v, deg)
        step = _newton_step(g, Hm, eig_floor)
        norm = step.norm(dim=1, keepdim=True)
        step = step * torch.clamp(max_step / norm.clamp(min=1e-300), max=1.0)
        u, v = u + step[:, 0], v + step[:, 1]
    h, g, Hm = chart_derivatives(C, face, u, v, deg)
    det = Hm[:, 0, 0] * Hm[:, 1, 1] - Hm[:, 0, 1] ** 2
    tr = Hm[:, 0, 0] + Hm[:, 1, 1]
    kind = torch.where(det < 0, SADDLE, torch.where(tr < 0, MAX, MIN))
    gn = g.norm(dim=1)
    converged = (gn < tol) & (u.abs() < bound) & (v.abs() < bound)
    P = face_point_batch(face, u, v)
    P = P / P.norm(dim=1, keepdim=True)
    return Critical(seeds.form, face, u, v, P, h, kind, gn, det, converged)


def face_point_batch(face, u, v):
    """Points (K, 3) of the faces `face` (K,) for chart coordinates u, v (K,)."""
    one = torch.ones_like(u)
    P = torch.stack([one, u, v], dim=1)
    P = torch.where((face == 1)[:, None], torch.stack([u, one, v], dim=1), P)
    return torch.where((face == 2)[:, None], torch.stack([u, v, one], dim=1), P)


# ---------------------------------------------------------------------------
# Step 4 (partial): canonical representatives, deduplication per form, Morse check
# ---------------------------------------------------------------------------


def canonical(P):
    """Representatives of points of RP^2 with the largest coordinate (in absolute value) positive."""
    k = P.abs().argmax(dim=1)
    s = torch.sign(P.gather(1, k[:, None]))
    return P * s


def deduplicate(crit, n_forms, tol=1e-6):
    """Keep one point per cluster (angular distance < tol, up to sign) within each form; converged points only."""
    crit = crit.select(crit.converged)
    if len(crit) == 0:
        return crit
    order = torch.argsort(crit.form, stable=True)
    crit = crit.select(order)
    counts = torch.bincount(crit.form, minlength=n_forms)
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(len(crit), device=crit.form.device) - start[crit.form]
    kmax = int(counts.max())
    Ppad = torch.full((n_forms, kmax, 3), float("nan"), dtype=crit.point.dtype, device=crit.point.device)
    Ppad[crit.form, pos] = crit.point
    dot = torch.einsum("bik,bjk->bij", Ppad, Ppad).abs()  # |cos| (sign-invariant)
    close = torch.nan_to_num(dot, nan=0.0) > torch.cos(torch.tensor(tol, dtype=dot.dtype))
    earlier = torch.tril(torch.ones(kmax, kmax, dtype=torch.bool, device=dot.device), diagonal=-1)
    dup = (close & earlier).any(dim=2)  # an earlier point in the same cluster exists
    keep = ~dup[crit.form, pos]
    out = crit.select(keep)
    out.point = canonical(out.point)
    return out


def morse_check(crit, n_forms):
    """Per form: #max - #saddle + #min (equals chi(RP^2) = 1 for a Morse function on RP^2)."""
    sign = torch.where(crit.kind == SADDLE, -1, 1)
    return torch.zeros(n_forms, dtype=torch.long, device=sign.device).index_add_(0, crit.form, sign)


def _concat(parts):
    return Critical(*[torch.cat([getattr(p, f) for p in parts]) for f in Critical.__dataclass_fields__])


def critical_points(coefs, deg, N=40, iters=8, charts=None, max_N=320):
    """All critical points for a batch of forms (D, B): seeds, Newton refinement, deduplication.

    Forms failing the Morse check (#max - #saddle + #min != 1, typically two critical points closer than the grid
    spacing) are recomputed on grids with N doubled, up to max_N. Returns (critical points, Morse sums per form).
    """
    B = coefs.shape[1]
    charts = charts or FaceCharts(deg, N=N)
    crit = deduplicate(refine(charts.seeds(coefs), coefs, deg, iters=iters, max_step=2 * charts.spacing), B)
    morse = morse_check(crit, B)
    n = charts.N
    while n < max_N:
        bad = (morse != 1).nonzero().squeeze(1)
        if bad.numel() == 0:
            break
        n *= 2
        fine = FaceCharts(deg, N=n)
        sub = coefs[:, bad]
        c2 = deduplicate(refine(fine.seeds(sub), sub, deg, iters=iters, max_step=2 * fine.spacing), bad.numel())
        c2.form = bad[c2.form]
        keep = ~torch.isin(crit.form, bad)
        crit = _concat([crit.select(keep), c2])
        morse = morse_check(crit, B)
    order = torch.argsort(crit.form, stable=True)
    return crit.select(order), morse


# ---------------------------------------------------------------------------
# Step 5: tracking and gradients of critical values with respect to parameters
# ---------------------------------------------------------------------------


def track(crit, coefs, deg, iters=8, max_step=0.05):
    """Re-locate critical points after the forms changed: Newton started at the previous points.

    crit.form indexes the columns of the new `coefs` (D, B). Returns a Critical (not deduplicated).
    """
    seeds = Seeds(crit.form, crit.face, None, None, crit.u, crit.v, crit.point, crit.value, crit.kind)
    return refine(seeds, coefs, deg, iters=iters, max_step=max_step)


def monomial_values(points, deg):
    """Monomials of degree `deg` at the points (K, 3); shape (K, D)."""
    mons = torch.tensor(monomials(deg), dtype=points.dtype, device=points.device)
    return torch.prod(points[:, None, :] ** mons[None], dim=2)


def value_gradients(crit, coefs, deg):
    """Gradients of the critical values with respect to the coefficients of the forms (curve = form).

    v_p = H^.m(p) with H^ = H / |H|; with p fixed (envelope theorem) dv_p = (m - v_p H^).dH / |H|.
    Returns (K, D).
    """
    norm = coefs.norm(dim=0)
    Hhat = (coefs / norm).T[crit.form]
    m = monomial_values(crit.point, deg)
    return (m - crit.value[:, None] * Hhat) / norm[crit.form, None]


@lru_cache(maxsize=None)
def _hessian_bilinear(deg_f, device=DEVICE):
    """S = T + T^t with H_k(f) = f^t T_k f; tensor (D_H, D_f, D_f)."""
    from conncomp.hessian import hessian_map

    hmons, fmons = monomials(2 * deg_f - 4), monomials(deg_f)
    T = torch.zeros((len(hmons), len(fmons), len(fmons)), dtype=DTYPE, device=device)
    for m, terms in hessian_map(deg_f).items():
        k = hmons.index(m)
        for e, factor in terms.items():
            idx = [i for i, ei in enumerate(e) for _ in range(ei)]  # the two coefficients (possibly equal)
            T[k, idx[0], idx[1]] += float(factor)
    return T + T.transpose(1, 2)


def hessian_value_gradients(crit, f, deg_f, basis=None):
    """Gradients of the critical values of the Hessian curves H(f) with respect to the coefficients of f.

    crit: critical points of H(f) (curve degree 2 deg_f - 4), crit.form indexing the columns of f (D_f, B).
    dv_p = (m - v_p H^).dH / |H| with dH = J df, J_k = S_k f (H is quadratic in f). With `basis` (D_f, k),
    f = B a and the gradients are with respect to a. Returns (K, D_f) or (K, k).
    """
    S = _hessian_bilinear(deg_f, f.device)
    H = torch.einsum("kij,ib,jb->kb", S, f, f) / 2  # H_k = f^t T_k f
    norm = H.norm(dim=0)
    Hhat = (H / norm).T[crit.form]
    w = (monomial_values(crit.point, 2 * deg_f - 4) - crit.value[:, None] * Hhat) / norm[crit.form, None]
    SF = torch.einsum("kij,jb->bki", S, f)  # (B, D_H, D_f): J for each form
    grad = torch.einsum("Kk,Kki->Ki", w, SF[crit.form])
    if basis is not None:
        grad = grad @ torch.as_tensor(basis, dtype=grad.dtype, device=grad.device)
    return grad


def cross_walls(target, f, deg_f, overshoot=1.0, iters=6, basis=None):
    """Push the critical value of each target point (one per form, f[:, target.form]) through zero.

    Newton iterations on v(theta) = -overshoot * v0 along the gradient of v (Hessian family, parameters = f or
    basis coordinates), re-locating the critical point after each step. Returns (f_new (D_f, K), tracked
    critical points, crossed mask, relative parameter change).
    """
    Bt = None if basis is None else torch.as_tensor(basis, dtype=f.dtype, device=f.device).to(f.dtype)
    cols = f[:, target.form].clone()
    v0 = target.value.clone()
    goal = -overshoot * v0
    cur = Critical(*[getattr(target, k).clone() for k in Critical.__dataclass_fields__])
    cur.form = torch.arange(len(target), device=f.device)
    for _ in range(iters):
        g = hessian_value_gradients(cur, cols, deg_f, basis=basis)
        step = ((goal - cur.value) / (g * g).sum(1).clamp(min=1e-300))[:, None] * g
        cols = cols + (step.T if Bt is None else Bt @ step.T)
        cur = track(cur, hessian(deg_f, cols), 2 * deg_f - 4)
    crossed = (torch.sign(cur.value) == -torch.sign(v0)) & cur.converged
    rel = (cols - f[:, target.form]).norm(dim=0) / f[:, target.form].norm(dim=0)
    return cols, cur, crossed, rel


# ---------------------------------------------------------------------------
# Distances to walls, degeneracy scores, simultaneous crossing of several walls
# ---------------------------------------------------------------------------


def good_wall(kind, value):
    """Walls whose crossing is predicted to add an oval: min with v > 0, max with v < 0, saddle with v > 0.

    (Extrema on the 'wrong' side give a birth; for saddles the rule is empirical, about 75% for Hessians.)
    """
    return ((kind == MIN) & (value > 0)) | ((kind == MAX) & (value < 0)) | ((kind == SADDLE) & (value > 0))


def wall_distances(crit, f, deg_f, basis=None):
    """Estimated relative parameter distances d_p = |v_p| / |grad v_p| to the walls, with |f| = 1 (Hessian family).

    Returns (d (K,), gradients (K, D_f))."""
    fn = f / f.norm(dim=0, keepdim=True)
    g = hessian_value_gradients(crit, fn, deg_f, basis=basis)
    return crit.value.abs() / g.norm(dim=1).clamp(min=1e-300), g


def degeneracy(crit, d, n_forms, tau=1e-2):
    """Per form: number of walls within distance tau, number of good walls within tau, soft score sum exp(-d/tau)."""
    close = (d < tau).long()
    good = close * good_wall(crit.kind, crit.value).long()
    z = torch.zeros(n_forms, dtype=torch.long, device=d.device)
    soft = torch.zeros(n_forms, dtype=d.dtype, device=d.device).index_add_(0, crit.form, torch.exp(-d / tau))
    return z.index_add(0, crit.form, close), z.index_add(0, crit.form, good), soft


def cross_walls_multi(crit, f, deg_f, select, overshoot=1.0, iters=6, damping=1e-9):
    """Cross several walls of each form at once: Gauss-Newton on v_i(f + df) = -overshoot v_i0 for the selected
    critical points (minimal-norm steps), re-tracking all selected points after each step.

    crit: critical points of H(f) with crit.form indexing the columns of f (D_f, B); select: bool mask (K,).
    The normal equations are damped relative to their diagonal (nearly parallel wall gradients).
    Forms without selected points are unchanged. Returns (f_new (D_f, B), crossed fraction per form (B,)).
    """
    B = f.shape[1]
    sel = crit.select(select)
    if len(sel) == 0:
        return f.clone(), torch.zeros(B, dtype=f.dtype, device=f.device)
    order = torch.argsort(sel.form, stable=True)
    sel = sel.select(order)
    counts = torch.bincount(sel.form, minlength=B)
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(len(sel), device=f.device) - start[sel.form]
    m = int(counts.max())
    v0 = sel.value.clone()
    goal = -overshoot * v0
    cols = f / f.norm(dim=0, keepdim=True)
    cur = sel
    for _ in range(iters):
        g = hessian_value_gradients(cur, cols, deg_f)  # (K, D_f)
        G = torch.zeros((B, m, g.shape[1]), dtype=f.dtype, device=f.device)
        r = torch.zeros((B, m), dtype=f.dtype, device=f.device)
        G[cur.form, pos] = g
        r[cur.form, pos] = goal - cur.value
        A = G @ G.transpose(1, 2)
        scale = torch.diagonal(A, dim1=1, dim2=2).mean(1).clamp(min=1e-300)  # relative damping per form
        A = A + damping * scale[:, None, None] * torch.eye(m, dtype=f.dtype, device=f.device)
        A = A + torch.diag_embed((counts[:, None] <= torch.arange(m, device=f.device)[None]).to(f.dtype))
        lam = torch.linalg.solve(A, r[:, :, None])
        cols = cols + (G.transpose(1, 2) @ lam)[:, :, 0].T
        cur = track(cur, hessian(deg_f, cols), 2 * deg_f - 4)
    ok = ((torch.sign(cur.value) == -torch.sign(v0)) & cur.converged).to(f.dtype)
    frac = torch.zeros(B, dtype=f.dtype, device=f.device).index_add_(0, cur.form, ok) / counts.clamp(min=1)
    return cols, frac
