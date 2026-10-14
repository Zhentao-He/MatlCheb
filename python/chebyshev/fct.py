"""Fast cosine transform used by the Chebyshev coefficient routines.

Port of ``myFCT1.m``. For real samples ``f_0, ..., f_N`` with ``N`` even and
positive, ``myFCT1`` is the unnormalized DCT-I realized through an FFT of an
auxiliary length-``N`` vector, matching the MATLAB implementation (including
the even-``N`` restriction).
"""

from __future__ import annotations

import numpy as np

from ._array import orient_input, restore_orientation


def myFCT1(f):
    """First-kind cosine transform of real data.

    Parameters
    ----------
    f : array_like, shape (N + 1,) or (N + 1, m) or (1, N + 1)
        Real samples. ``N = length - 1`` must be even and positive.
        A 1-D array is treated as a column. A matrix is transformed
        column by column. A row vector stays a row vector.

    Returns
    -------
    ndarray
        Transform coefficients, same shape as ``f`` (1-D in, 1-D out).
        Mode ``0`` is the first entry and mode ``N`` is the last.

    Notes
    -----
    Applying the transform twice recovers the samples up to a scale:

        myFCT1(myFCT1(f)) * 2 / N == f

    within roundoff. That identity is what ``cheb_test`` checks. The
    Chebyshev coefficients themselves are the scaled, endpoint-halved
    coefficients returned by :func:`chebyshev.transform.real_to_cheb`.
    """
    cols, kind = orient_input(f)
    transformed = _fct_columns(cols)
    return restore_orientation(transformed, kind)


def _fct_columns(f: np.ndarray) -> np.ndarray:
    """Column-wise FCT. ``f`` has shape ``(N + 1, m)``."""
    n_rows = f.shape[0]
    N = n_rows - 1
    if N % 2 != 0:
        raise ValueError("N has to be even.")
    if N <= 0:
        raise ValueError("N has to be positive and even.")

    F = np.zeros_like(f)
    # y_j = 1/2 (f_j + f_{N-j}) - sin(j pi / N) (f_j - f_{N-j}), j = 0..N-1
    f_head = f[:-1, :]
    f_tail = np.flip(f[1:, :], axis=0)
    jj = np.arange(N, dtype=np.float64)[:, None]
    y = 0.5 * (f_head + f_tail) - np.sin(jj * np.pi / N) * (f_head - f_tail)

    u = np.fft.fft(y, axis=0)
    R = np.real(u)
    I = -np.imag(u)

    # Even modes: F_{2k} = Re(fft(y))_k for k = 0..N/2.
    kk = np.arange(N // 2 + 1)
    F[2 * kk, :] = R[kk, :]

    # F_1 from the explicit cosine sum (MATLAB index 2).
    j = np.arange(1, N)
    F[1, :] = 0.5 * (f[0, :] - f[-1, :]) + np.sum(
        f[j, :] * np.cos(j * np.pi / N)[:, None], axis=0
    )

    # Odd modes above 1: F_{2k+1} = F_{2k-1} + I_k, k = 1..N/2-1,
    # i.e. F_{2m+1} = F_1 + sum_{k=1}^{m} I_k.
    if N // 2 > 1:
        ks = np.arange(1, N // 2)
        partial = np.cumsum(I[ks, :], axis=0)
        F[2 * ks + 1, :] = F[1, :] + partial
    return F
