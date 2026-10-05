"""Connected components of the sign regions {F >= 0} and {F < 0} on the grid."""

import numpy as np
import torch

from conncomp import _connected


def _as_numpy(vals):
    if isinstance(vals, torch.Tensor):
        vals = vals.detach().cpu().numpy()
    return np.ascontiguousarray(vals, dtype=np.float64)


def component_sizes(vals, pat, num_threads=-1):
    """Pixel counts of the sign components for each grid in the batch `vals` (shape (N, W, W))."""
    return _connected.components_batch(_as_numpy(vals), pat, num_threads)


def component_labels(vals, pat):
    """Labels of the sign components of a single grid `vals` (shape (W, W)).

    Returns (labels, sizes): labels[i, j] is the index of the component of pixel (i, j) in `sizes`.
    """
    return _connected.component_labels(_as_numpy(vals), pat)


def count_components(vals, pat, min_size=1, num_threads=-1):
    """Number of sign components with at least `min_size` pixels, for each grid in the batch.

    Small components are discarded as discretization noise. For a curve of even
    degree in RP^2, the complement of k ovals has k + 1 components.
    """
    return [sum(1 for s in sizes if s >= min_size) for sizes in component_sizes(vals, pat, num_threads)]
