"""Analytic mesh refinement of a Chebyshev grid.

Port of ``AnMR.m``.
"""

from __future__ import annotations

import numpy as np

from .cheb import cheb


def AnMR(N: int, xB: float, kappa: float):
    """Differentiate on a Chebyshev grid clustered toward one endpoint.

    Parameters
    ----------
    N :
        Polynomial degree passed to :func:`cheb`.
    xB :
        Endpoint that should be refined, ``+1`` or ``-1``.
    kappa :
        Clustering strength. Larger values pack nodes more tightly near
        ``xB``. ``kappa = 0`` is a removable singularity of this formula
        (the map tends to the identity) but the expression as written in
        ``AnMR.m`` divides by ``sinh(2 kappa)`` and is not evaluated here
        in that limit.

    Returns
    -------
    Dxx, Dx : ndarray, shape (N + 1, N + 1)
        Second- and first-derivative matrices in the mapped coordinate ``x``.
    x : ndarray, shape (N + 1,)
        Mapped nodes. MATLAB returns a column; the values match.
        ``y`` is the reference Chebyshev grid and

            x(y) = xB * (1 - 2 sinh(kappa (1 - xB y)) / sinh(2 kappa)).

    Chain rule, with ``x' = dx/dy`` and ``x'' = d²x/dy²``:

        d/dx = (1 / x') d/dy
        d²/dx² = (1 / x'²) d²/dy² - (x'' / x'³) d/dy

    so ``Dx = diag(1/x') Dy`` and ``Dxx`` is the corresponding combination
    of ``Dy`` and ``Dy @ Dy``.
    """
    Dy, y = cheb(N)
    if np.ndim(Dy) == 0:
        raise ValueError("AnMR requires N >= 1")
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    xB = float(xB)
    kappa = float(kappa)
    if kappa == 0.0:
        raise ValueError(
            "kappa = 0 makes AnMR singular (sinh(2 kappa) = 0); "
            "use cheb(N) for the unclustered grid"
        )

    sinh_2k = np.sinh(2.0 * kappa)
    phase = kappa * (1.0 - xB * y)
    xy = kappa * (xB ** 2) * 2.0 * np.cosh(phase) / sinh_2k
    xyy = -2.0 * (kappa ** 2) * (xB ** 3) / sinh_2k * np.sinh(phase)
    x = xB * (1.0 - 2.0 * np.sinh(phase) / sinh_2k)

    Dx = Dy / xy
    Dxx = (-Dy * xyy) / (xy ** 3) + (Dy @ Dy) / (xy ** 2)
    return Dxx, Dx, x.reshape(-1)
