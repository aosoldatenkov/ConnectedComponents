"""Fast GPU pre-screening of oval counts via the Euler characteristic.

For a curve whose ovals are not nested, one sign class of RP^2 minus the curve is a union of k disks
(Euler characteristic k) and the other is RP^2 minus k disks (Euler characteristic 1 - k). The Euler
characteristic of a sign class is local: on a quad mesh of S^2 it is

    chi = #vertices in the class - #edges with both ends in it + #squares with all corners in it,

computed with a few gathers of sign bits and no connectivity. The mesh is the cube-surface lattice
of conncomp.sphere, which is symmetric under p -> -p, so the lift of a set in RP^2 has twice its
Euler characteristic. The estimate max(chi_+, chi_-) / 2 equals k for k >= 1 non-nested ovals
resolved by the mesh (an empty curve also gives 1). Nested ovals reduce it, and pixel noise changes
it, so it is only a filter. Candidates are re-counted with the exact component count
(conncomp.components).
"""

import numpy as np
import torch

from conncomp import DEVICE, DTYPE
from conncomp.hessian import hessian
from conncomp.polynomials import monomial_basis, monomials


def face_grid_points(N):
    """Points of the 6 faces of the cube [-N, N]^3 as an array (6, 2N + 1, 2N + 1, 3).

    Face f = (axis, side) has p[axis] = side * N; the other two coordinates run over -N..N, so the
    points of the cube edges and corners occur on 2 and 3 faces, respectively.
    """
    r = np.arange(-N, N + 1)
    U, V = np.meshgrid(r, r, indexing="ij")
    out = np.empty((6, 2 * N + 1, 2 * N + 1, 3), dtype=np.int64)
    f = 0
    for axis in range(3):
        o0, o1 = [b for b in range(3) if b != axis]
        for side in (-1, 1):
            out[f, :, :, axis] = side * N
            out[f, :, :, o0] = U
            out[f, :, :, o1] = V
            f += 1
    return out


def _window_lut():
    """4 * (vertex/4 - edge/2 + square) weights of the 16 patterns of a 2x2 window, packed for both classes.

    Pattern bits (a, b, c, d) are the window corners in cyclic order. Entry = lut(pattern) + 2^32 *
    lut(complement), so one gather and one int64 sum give both classes.
    """
    lut = []
    for idx in range(16):
        a, b, c, d = (idx >> 0) & 1, (idx >> 1) & 1, (idx >> 2) & 1, (idx >> 3) & 1
        lut.append(a + b + c + d - 2 * (a * b + b * c + c * d + d * a) + 4 * a * b * c * d)
    return [lut[i] + (lut[15 - i] << 32) for i in range(16)]


_LUT = {}


def _unpack(total):
    """Split int64 sums of packed (low + 2^32 high) values with |low| < 2^31 into (low, high)."""
    low = ((total + 2**31) % 2**32) - 2**31
    return low, (total - low) >> 32


def euler_characteristics(s):
    """Euler characteristics (chi of {s}, chi of {not s}) on the cube-surface quad mesh, for a batch.

    s: boolean tensor (n, 6, M, M) on the points of `face_grid_points`. Every vertex gets weight 1/4 in
    each 2x2 window containing it and every edge 1/2; summed over the 6 faces, this counts every vertex,
    edge and square of the mesh exactly once except the 8 cube corners (3/4 instead of 1). Hence
    4 chi = sum over windows of LUT(pattern) + number of corners in the class.
    """
    key = s.device
    if key not in _LUT:
        _LUT[key] = torch.tensor(_window_lut(), dtype=torch.int64, device=s.device)
    u = s.to(torch.uint8)
    idx = u[:, :, :-1, :-1] + 2 * u[:, :, :-1, 1:] + 4 * u[:, :, 1:, 1:] + 8 * u[:, :, 1:, :-1]
    pos4, neg4 = _unpack(_LUT[key][idx.long()].sum((1, 2, 3)))
    corners = (u[:, :, 0, 0].long() + u[:, :, 0, -1] + u[:, :, -1, 0] + u[:, :, -1, -1]).sum(1) // 3
    return (pos4 + corners) // 4, (neg4 + (8 - corners)) // 4


def euler_characteristic(s):
    """Euler characteristic of the class {s} (see `euler_characteristics`)."""
    return euler_characteristics(s)[0]


class EulerScreen:
    """Estimated oval counts of the Hessian curves (or the curves f = 0) of batches of forms.

    The per-chunk computation (evaluation, signs, Euler characteristics) is fused with torch.compile
    (about 20x faster than eager mode), with a fixed chunk size; set compile=False to disable.
    """

    def __init__(self, deg, N=40, use_hessian=True, device=DEVICE, dtype=torch.float32, chunk=8192, compile=True):
        self.deg, self.N, self.use_hessian = deg, N, use_hessian
        self.curve_deg = 2 * deg - 4 if use_hessian else deg
        self.M, self.chunk, self.dtype = 2 * N + 1, chunk, dtype
        P = face_grid_points(N).reshape(-1, 3).astype(np.float64)
        pts = torch.as_tensor(P / np.linalg.norm(P, axis=1, keepdims=True), dtype=DTYPE, device=device)
        self.basis = monomial_basis(monomials(self.curve_deg), pts.T).to(dtype)  # (D, 6 M^2)
        self._step = self._chunk_estimate
        if compile:
            try:
                self._step = torch.compile(self._chunk_estimate, dynamic=False)
            except Exception:  # no compiler backend available
                pass

    def _chunk_estimate(self, cc):
        vals = (cc.T @ self.basis).reshape(-1, 6, self.M, self.M)
        chi_pos, chi_neg = euler_characteristics(vals >= 0)
        return torch.maximum(chi_pos, chi_neg) // 2

    def estimate(self, coefs):
        """Estimated number of ovals for each form in the batch `coefs` (D, n); int tensor (n,)."""
        cc = hessian(self.deg, coefs) if self.use_hessian else coefs
        cc = (cc / cc.norm(dim=0, keepdim=True)).to(self.dtype)
        n = cc.shape[1]
        out = torch.empty(n, dtype=torch.int64, device=cc.device)
        for s0 in range(0, n, self.chunk):
            part = cc[:, s0 : s0 + self.chunk]
            k = part.shape[1]
            if k < self.chunk:  # pad to the compiled shape
                part = torch.cat([part, part[:, :1].expand(-1, self.chunk - k)], dim=1)
            out[s0 : s0 + k] = self._step(part.contiguous())[:k]
        return out


def select_candidates(estimates, threshold, cap):
    """Indices of the forms with estimate >= threshold; at most `cap` of them, highest estimates first."""
    idx = (estimates >= threshold).nonzero().squeeze(1)
    if idx.numel() > cap:
        idx = idx[torch.topk(estimates[idx], cap).indices]
    return idx
