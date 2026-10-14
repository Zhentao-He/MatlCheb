"""Arbitrary-precision Chebyshev differentiation matrix.

Port of ``cheb_mp.m``. The MATLAB file calls ``mp`` (Advanpix Multiprecision
Computing Toolbox), which is not part of base MATLAB or GNU Octave. This
module uses :mod:`mpmath` instead.
"""

from __future__ import annotations

import mpmath as mp

from .cheb import _as_degree


def cheb_mp(N: int, dps: int = 50):
    """Multiprecision version of :func:`chebyshev.cheb.cheb` on ``[-1, 1]``.

    Parameters
    ----------
    N :
        Polynomial degree. ``N == 0`` returns ``(mpf(0), mpf(1))``.
    dps :
        Decimal places of working precision (mpmath ``workdps``).

    Returns
    -------
    D : mpmath.matrix, shape (N + 1, N + 1)
    x : mpmath.matrix, shape (N + 1, 1)
        Nodes ordered from ``+1`` to ``-1``.

    The same recurrence as ``cheb.m`` is used, so float64(D) agrees with
    :func:`cheb` to roundoff once ``dps`` is large enough. Further arithmetic
    on the returned values should be done under ``mp.workdps(dps)``.
    """
    if int(dps) < 1:
        raise ValueError("dps must be a positive integer")
    N = _as_degree(N)
    with mp.workdps(int(dps)):
        if N == 0:
            return mp.mpf(0), mp.mpf(1)

        pi = mp.pi
        x_list = [mp.cos(pi * i / N) for i in range(N + 1)]
        c = []
        for i in range(N + 1):
            scale = mp.mpf(2) if i == 0 or i == N else mp.mpf(1)
            c.append(scale * ((-1) ** i))

        D = mp.matrix(N + 1)
        for i in range(N + 1):
            for j in range(N + 1):
                if i == j:
                    # Diagonal of (dX + I) is 1, and c_i/c_i = 1.
                    D[i, j] = mp.mpf(1)
                else:
                    D[i, j] = (c[i] / c[j]) / (x_list[i] - x_list[j])
        for i in range(N + 1):
            row_sum = mp.fsum(D[i, j] for j in range(N + 1))
            D[i, i] = D[i, i] - row_sum

        x = mp.matrix(N + 1, 1)
        for i in range(N + 1):
            x[i, 0] = x_list[i]
        return D, x
