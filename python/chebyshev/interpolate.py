"""Evaluate a Chebyshev series at arbitrary points.

Port of ``cheb_interpolate.m``.
"""

from __future__ import annotations

import numpy as np

from ._array import orient_input


def cheb_interpolate(a, x1, x2, x):
    """Evaluate ``sum_n a_n T_n(z(x))`` on an interval.

    Parameters
    ----------
    a : array_like
        Chebyshev coefficients, mode 0 first. A 1-D array or a column is
        one series. A matrix gives one series per column (a single row is
        one series, as in MATLAB).
    x1, x2 :
        Endpoints of the interval the series was built on. If ``x1 > x2``
        they are swapped. They must not be equal.
    x : array_like
        Query locations. A scalar returns a scalar for one series and a
        column of length ``m`` for ``m`` series. A column ``x`` returns one
        row per query point. A row ``x`` returns one column per query point.
        A 1-D ``x`` is treated as a column and, for a single series, the
        result is 1-D.

    Returns
    -------
    ndarray or numpy scalar

    The physical coordinate is mapped by ``z = 2 (x - x1) / (x2 - x1) - 1``.
    Values of ``z`` that fall a hair outside ``[-1, 1]`` are clipped before
    ``arccos``, matching the MATLAB routine. The series is then
    ``sum_n a_n cos(n arccos z)``.
    """
    x1 = float(x1)
    x2 = float(x2)
    if x1 == x2:
        raise ValueError("x1 and x2 must be distinct")
    if x1 > x2:
        x1, x2 = x2, x1
    length = x2 - x1

    coeffs, coeff_kind = orient_input(a, name="a")
    # A row of coefficients is a single series, already transposed to a column.
    del coeff_kind  # orientation of `a` does not affect the layout of `vals`

    x_arr = np.asarray(x)
    if np.iscomplexobj(x_arr):
        raise ValueError("x must be real")
    x_arr = np.asarray(x_arr, dtype=np.float64)

    N = coeffs.shape[0] - 1
    modes = np.arange(N + 1, dtype=np.float64).reshape(-1, 1)

    # 0-D values and 1-by-1 arrays match MATLAB isscalar. A length-1 1-D
    # array stays a vector so the result keeps an array shape.
    scalar_x = x_arr.ndim == 0 or x_arr.shape == (1, 1)
    if scalar_x:
        z = _map_to_chebyshev(float(np.reshape(x_arr, ())), x1, length)
        theta = float(np.arccos(z))
        # sum over modes -> shape (m,), then a column if there are several series
        vals = np.sum(coeffs * np.cos(modes * theta), axis=0)
        if vals.size == 1:
            return vals.reshape(())
        return vals.reshape(-1, 1)

    x_was_1d = x_arr.ndim == 1
    if x_arr.ndim == 1:
        flagrow = False
        theta = np.arccos(_map_to_chebyshev(x_arr, x1, length)).reshape(1, -1)
    elif x_arr.ndim == 2 and x_arr.shape[1] == 1:
        flagrow = False
        theta = np.arccos(_map_to_chebyshev(x_arr.reshape(-1), x1, length)).reshape(1, -1)
    elif x_arr.ndim == 2 and x_arr.shape[0] == 1:
        flagrow = True
        theta = np.arccos(_map_to_chebyshev(x_arr.reshape(-1), x1, length)).reshape(1, -1)
    else:
        raise ValueError("x must be a scalar or a 1-D vector")

    # (m, N+1) @ (N+1, n_query) -> (m, n_query)
    vals = coeffs.T @ np.cos(modes @ theta)
    if flagrow:
        out = vals
    else:
        out = vals.T
    if x_was_1d and out.ndim == 2 and out.shape[1] == 1:
        return out.reshape(-1)
    if x_was_1d:
        return out
    return out


def _map_to_chebyshev(x, x1: float, length: float):
    z = 2.0 * (np.asarray(x, dtype=np.float64) - x1) / length - 1.0
    return np.clip(z, -1.0, 1.0)
