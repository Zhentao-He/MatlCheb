"""Chebyshev differentiation matrix on Gauss–Lobatto nodes.

Port of ``cheb.m`` (Trefethen, *Spectral Methods in MATLAB*, p. 54).
"""

from __future__ import annotations

import numpy as np


def cheb(N: int, x1: float | None = None, x2: float | None = None):
    """Build the Chebyshev differentiation matrix.

    Parameters
    ----------
    N :
        Polynomial degree. The grid has ``N + 1`` Gauss–Lobatto nodes
        ``x_i = cos(pi * i / N)``, ordered from ``+1`` down to ``-1``.
    x1, x2 :
        Optional endpoints of an affine image of ``[-1, 1]``. Both must be
        given. Unlike ``cheb_interpolate``, the endpoints are not swapped
        when ``x1 > x2``.

    Returns
    -------
    D : ndarray, shape (N + 1, N + 1)
        Differentiation matrix. For ``N == 0`` this is the scalar ``0``.
    x : ndarray, shape (N + 1,)
        Grid points. MATLAB returns a column; the values are identical.
        For ``N == 0`` this is the scalar ``1`` (the ``N == 0`` branch
        returns before any affine rescaling).

    Notes
    -----
    Off-diagonal entries are

        D_ij = (c_i / c_j) * (-1)^(i+j) / (x_i - x_j),

    with ``c_0 = c_N = 2`` and ``c_i = 1`` otherwise. The diagonal is set so
    that each row sums to zero (the derivative of a constant vanishes).
    On an interval of length ``L = x2 - x1`` the matrix is divided by ``L/2``.
    """
    N = _as_degree(N)
    if N == 0:
        return np.float64(0.0), np.float64(1.0)

    x = np.cos(np.pi * np.arange(N + 1, dtype=np.float64) / N)
    c = np.ones(N + 1, dtype=np.float64)
    c[0] = 2.0
    c[-1] = 2.0
    c *= (-1.0) ** np.arange(N + 1, dtype=np.float64)

    X = np.repeat(x[:, None], N + 1, axis=1)
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1, dtype=np.float64))
    D -= np.diag(np.sum(D, axis=1))

    if x1 is not None or x2 is not None:
        if x1 is None or x2 is None:
            raise ValueError("cheb expects both endpoints x1 and x2, or neither")
        half_length = 0.5 * (float(x2) - float(x1))
        if half_length == 0.0:
            raise ValueError("x1 and x2 must be distinct")
        x = half_length * (x + 1.0) + float(x1)
        D = D / half_length
    return D, x


def _as_degree(N) -> int:
    try:
        N_float = float(N)
    except (TypeError, ValueError) as exc:
        raise ValueError("N must be a non-negative integer") from exc
    if not np.isfinite(N_float) or int(N_float) != N_float or N_float < 0:
        raise ValueError("N must be a non-negative integer")
    return int(N_float)
