"""Maps between grid samples and Chebyshev coefficients.

Ports of ``real_to_cheb.m`` and ``cheb_to_real.m``.
"""

from __future__ import annotations

import numpy as np

from ._array import orient_input, restore_orientation
from .fct import myFCT1


def real_to_cheb(f):
    """Chebyshev coefficients of samples on the Gauss–Lobatto grid.

    Parameters
    ----------
    f : array_like
        Values ``f(x_j)`` at ``x_j = cos(pi j / N)``, ``j = 0..N``.
        The length of each column must be odd (``N`` even). A matrix is
        transformed column by column. A 1-D array is a single column;
        a shape-(1, n) row stays a row.

    Returns
    -------
    ndarray
        Coefficients ``a_0, ..., a_N`` in the basis ``T_n(x) = cos(n arccos x)``,
        same orientation as ``f``, so that

            f(x_j) = sum_{n=0}^{N} a_n T_n(x_j).

    The standard discrete formula is, with weights ``w_0 = w_N = 1/2`` and
    ``w_j = 1`` otherwise,

        a_n = (2 / (N c_n)) sum_{j=0}^{N} w_j f_j cos(pi j n / N),

    where ``c_0 = c_N = 2`` and ``c_n = 1`` otherwise. This routine evaluates
    that sum through :func:`myFCT1`, as the MATLAB code does.
    """
    cols, kind = orient_input(f)
    N = cols.shape[0] - 1
    if N % 2 != 0:
        raise ValueError("The length of f is required to be odd !")
    # myFCT1(f) * 2/N, then halve the endpoints:
    #   a_0 = F_0 / N,  a_N = F_N / N,  a_n = 2 F_n / N otherwise.
    a = (2.0 / N) * myFCT1(cols)
    a[0, :] *= 0.5
    a[-1, :] *= 0.5
    return restore_orientation(a, kind)


def cheb_to_real(a):
    """Samples of a Chebyshev series on the Gauss–Lobatto grid.

    Parameters
    ----------
    a : array_like
        Coefficients ``a_0, ..., a_N``. Each column is one series.
        When ``N`` is even the inverse of :func:`real_to_cheb` is applied
        with :func:`myFCT1` (the original ``O(N log N)`` path). When ``N``
        is odd the series is summed with the cosine matrix, matching
        ``cheb_to_real.m``.

    Returns
    -------
    ndarray
        Grid values, same orientation as ``a``.
    """
    cols, kind = orient_input(a, name="a")
    N = cols.shape[0] - 1
    if N < 0:
        raise ValueError("a must contain at least one coefficient")
    if N % 2 == 0:
        if N == 0:
            # T_0 = 1; the FCT path divides by N. One coefficient is the constant.
            return restore_orientation(cols.copy(), kind)
        # Undo real_to_cheb, then use FCT(FCT(f)) = (N/2) f.
        F = np.empty_like(cols)
        F[0, :] = N * cols[0, :]
        F[-1, :] = N * cols[-1, :]
        if N > 1:
            F[1:-1, :] = (N / 2.0) * cols[1:-1, :]
        values = (2.0 / N) * myFCT1(F)
    else:
        modes = np.arange(N + 1, dtype=np.float64)
        # cos(pi * i * n / N), rows are nodes, columns are modes.
        basis = np.cos(np.pi * modes[:, None] * modes[None, :] / N)
        values = basis @ cols
    return restore_orientation(values, kind)
