"""Counting connected components of real algebraic varieties.

The current focus is on real plane curves in RP^2, in particular the Hessian
curves f_xx f_yy - f_xy^2 = 0 of ternary forms f (Arnold / Ortiz-Rodriguez
conjectures on the number of components of parabolic curves).
"""

import torch

DTYPE = torch.float64
DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

from conncomp.polynomials import deg_to_dim, monomials, monomial_basis, evaluate, normalize, sample  # noqa: E402
from conncomp.hessian import hessian, hessian_map  # noqa: E402
from conncomp.grid import sphere_points, neighbour_pattern  # noqa: E402
from conncomp.components import component_sizes, count_components  # noqa: E402

__all__ = [
    "DTYPE",
    "DEVICE",
    "deg_to_dim",
    "monomials",
    "monomial_basis",
    "evaluate",
    "normalize",
    "sample",
    "hessian",
    "hessian_map",
    "sphere_points",
    "neighbour_pattern",
    "component_sizes",
    "count_components",
]
