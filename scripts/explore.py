#!/usr/bin/env python3
"""Exploratory scans (run cell by cell in Spyder / VS Code / IPython via the #%% markers).

Run from the repository root so that the data/ paths resolve.
"""

# %%
import sympy as sp

from conncomp import DEVICE
from conncomp.io import load_coefs
from conncomp.plotting import plot_negative_regions
from conncomp.polynomials import to_sympy
from conncomp.scan import batch_scan, center_scan

print(f"Using {DEVICE} device")

# %% Uniform scan of Hessians of quintics
save_coefs = batch_scan(5, 100, 50000, 100, 8)

# %%
save_coefs = batch_scan(5, 200, 10000, 1000, 9)

# %% Refine around a good candidate
save_coefs2 = center_scan(5, 200, save_coefs[9][0], 1e-3, 10000, 1, 11)

# %%
save_coefs3 = center_scan(5, 200, save_coefs2[10][2], 1e-2, 1000, 1, 10, filtr=3)

# %%
save_coefs4 = center_scan(5, 500, save_coefs3[10][-19], 1e-2, 100, 1, 10, filtr=3)

# %% Load the best sextics found by the interactive search and plot their Hessian curves
cache = load_coefs("data/save_cache_H.txt")
plot_negative_regions(6, 200, cache[:3])

# %% Print a form and its Hessian symbolically
x, y, z = sp.symbols("x y z")
deg = 6
p = sp.Poly(to_sympy(cache[0], deg, x, y, z), x, y, z)
print(p.as_expr())
h = p.diff((0, 2)) * p.diff((1, 2)) - p.diff((0, 1), (1, 1)) ** 2
print(h.as_expr())
