"""A pixel grid model of the real projective plane RP^2.

The W x W grid covers the square [-1, 1]^2 of a stereographic chart of the unit
sphere S^2 (the closed unit disk is the lower hemisphere). Every point of RP^2
has a representative in the disk, so the grid covers RP^2; the identification
of antipodal points p ~ -p/|p|^2 is encoded in the neighbour pattern.
"""

import numpy as np
import torch

from conncomp import DEVICE, DTYPE

NUM_NEIGHBOURS = 12


def sphere_points(width, device=DEVICE):
    """Points of S^2 corresponding to the grid pixels; shape (3, width, width)."""
    t = torch.linspace(-1, 1, width, dtype=DTYPE, device=device)
    xx, yy = torch.meshgrid(t, t, indexing="ij")
    nn = xx**2 + yy**2
    return torch.stack([2.0 * xx, 2.0 * yy, nn - 1.0]) / (nn + 1.0)


def neighbour_pattern(width):
    """Neighbour indices for the component labelling; int32 array of shape (12, 2, width, width).

    pat[k, :, i, j] is the k-th neighbour of pixel (i, j): k = 0..7 are the 8 adjacent
    pixels, k = 8..11 are the 4 pixels surrounding the antipodal point. Missing
    neighbours (grid boundary, antipode outside the square) point to (i, j) itself.
    """
    pat = np.empty((NUM_NEIGHBOURS, 2, width, width), dtype=np.int32)
    dirs = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    mid = (width - 1) / 2
    for i in range(width):
        for j in range(width):
            pat[:, 0, i, j] = i
            pat[:, 1, i, j] = j
            for k, (di, dj) in enumerate(dirs):
                if 0 <= i + di < width and 0 <= j + dj < width:
                    pat[k, 0, i, j] = i + di
                    pat[k, 1, i, j] = j + dj
            rr = (i / mid - 1) ** 2 + (j / mid - 1) ** 2
            if rr == 0:
                continue
            ai = mid * (1 + (1 - i / mid) / rr)
            aj = mid * (1 + (1 - j / mid) / rr)
            k0, k1 = int(np.floor(ai)), int(np.ceil(ai))
            l0, l1 = int(np.floor(aj)), int(np.ceil(aj))
            for k, (a, b) in enumerate([(k0, l0), (k1, l0), (k0, l1), (k1, l1)]):
                if 0 <= a < width and 0 <= b < width:
                    pat[k + 8, 0, i, j] = a
                    pat[k + 8, 1, i, j] = b
    return pat
