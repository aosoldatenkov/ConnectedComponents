"""Plots of sign regions on the grid."""

import matplotlib.pyplot as plt
import torch

from conncomp import DTYPE, DEVICE
from conncomp.scan import Experiment


def plot_negative_regions(deg, width, coefs, use_hessian=True):
    """One scatter plot of the region {F < 0} per coefficient vector in `coefs`."""
    exp = Experiment(deg, width, use_hessian)
    cc = torch.as_tensor(coefs, dtype=DTYPE, device=DEVICE).reshape(len(coefs), -1).T
    v = exp.values(cc).cpu().numpy()
    figs = []
    for i in range(v.shape[0]):
        a, b = (v[i] < 0).nonzero()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(a, b, c="tab:blue", s=1)
        figs.append(fig)
    return figs
