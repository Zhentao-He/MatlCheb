"""Round-trip check for :func:`chebyshev.fct.myFCT1`.

Port of ``cheb_test.m``.
"""

from __future__ import annotations

import warnings

import numpy as np

from ._array import orient_input
from .fct import myFCT1


def cheb_test(f):
    """Compare ``f`` with the FCT round trip ``2 * myFCT1(myFCT1(f)) / N``.

    Parameters
    ----------
    f : array_like
        Real samples whose length (per column) is ``N + 1`` with ``N`` even
        and positive. See :func:`myFCT1` for orientation rules.

    Returns
    -------
    err : ndarray
        Absolute error ``|f - p|``, same shape as ``f``.
    p : ndarray
        Round-trip values, same shape as ``f``.

    A ``RuntimeWarning`` is issued when the error is above ``1e-13``. For a
    matrix, MATLAB's ``if`` warns only when every column mean exceeds the
    tolerance; that rule is preserved.
    """
    arr = np.asarray(f)
    cols, _kind = orient_input(arr)
    N = cols.shape[0] - 1
    recovered = 2.0 * myFCT1(myFCT1(arr)) / N
    # Broadcast against the original object so a 1-D input stays 1-D.
    err = np.abs(np.asarray(arr, dtype=np.float64) - recovered)
    tol = 1e-13
    if err.ndim == 2 and err.shape[0] > 1 and err.shape[1] > 1:
        too_big = bool(np.all(np.mean(err, axis=0) > tol))
    else:
        too_big = bool(np.mean(err) > tol)
    if too_big:
        warnings.warn(
            f"The error is not small enough! Tol = {tol}",
            RuntimeWarning,
            stacklevel=2,
        )
    return err, recovered
