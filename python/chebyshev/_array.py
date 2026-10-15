"""Shape helpers that mirror MATLAB row, column, and matrix orientation."""

from __future__ import annotations

import numpy as np

# How an input was oriented before it was turned into columns.
# "1d"    : NumPy 1-D array, treated as a MATLAB column vector
# "row"   : shape (1, n), n > 1
# "2d"    : any other matrix, including a single column of shape (n, 1)
# "scalar": 0-D or 1-by-1
Kind = str


def as_real_array(a, name: str = "f") -> np.ndarray:
    """Convert ``a`` to a real float64 array, or raise."""
    arr = np.asarray(a)
    if arr.ndim > 2:
        raise ValueError(f"{name} is required to be a matrix in myFCT1(f)!")
    if np.iscomplexobj(arr):
        raise ValueError(f"{name} must be real")
    if arr.dtype != np.float64:
        arr = np.asarray(arr, dtype=np.float64)
    return arr


def orient_input(a, name: str = "f") -> tuple[np.ndarray, Kind]:
    """Return ``(columns, kind)``.

    Each column of the returned array is one series. One-dimensional NumPy
    arrays are treated as MATLAB column vectors. A shape-(1, n) array with
    ``n > 1`` is a row vector and is transposed.
    """
    arr = as_real_array(a, name=name)
    if arr.ndim == 0:
        return arr.reshape(1, 1), "scalar"
    if arr.ndim == 1:
        return arr.reshape(-1, 1), "1d"
    if arr.shape == (1, 1):
        return arr.reshape(1, 1), "scalar"
    if arr.shape[0] == 1 and arr.shape[1] > 1:
        return np.ascontiguousarray(arr.T), "row"
    return np.ascontiguousarray(arr), "2d"


def restore_orientation(cols: np.ndarray, kind: Kind) -> np.ndarray:
    """Undo :func:`orient_input`."""
    if kind == "row":
        return np.ascontiguousarray(cols.T)
    if kind == "1d":
        return cols.reshape(-1)
    if kind == "scalar":
        return cols.reshape(())
    return cols
