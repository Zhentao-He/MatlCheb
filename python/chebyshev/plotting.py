"""Spectral-coefficient plot.

Port of ``spec_plot.m``.
"""

from __future__ import annotations

import numpy as np

from .transform import real_to_cheb


def spec_plot(f, ax=None, show: bool = False):
    """Plot ``log10 |a_n|`` for the Chebyshev coefficients of ``f``.

    Parameters
    ----------
    f : array_like
        Samples on a Chebyshev Gauss–Lobatto grid. See
        :func:`chebyshev.transform.real_to_cheb`.
    ax :
        Matplotlib axes. A new figure is created when omitted.
    show :
        If true, call ``pyplot.show()`` before returning.

    Returns
    -------
    matplotlib.axes.Axes

    A row or column of coefficients is drawn as one marker series. A matrix
    draws one series per column, as MATLAB ``plot`` does.
    """
    import matplotlib.pyplot as plt

    coeffs = np.asarray(real_to_cheb(f), dtype=np.float64)
    with np.errstate(divide="ignore"):
        y = np.log10(np.abs(coeffs))
    if y.ndim == 2 and 1 in y.shape:
        y = y.reshape(-1)

    if ax is None:
        _fig, ax = plt.subplots()
    ax.plot(y, linestyle="None", marker=".", markersize=10)
    ax.set_xlabel("mode n")
    ax.set_ylabel(r"$\log_{10}|a_n|$")
    ax.set_title("Chebyshev coefficient decay")
    if show:
        plt.show()
    return ax
