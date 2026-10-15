"""Run a short numerical demo of every ported routine.

    cd python
    python -m chebyshev
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import numpy as np

from chebyshev import (
    AnMR,
    cheb,
    cheb_interpolate,
    cheb_mp,
    cheb_test,
    cheb_to_real,
    myFCT1,
    real_to_cheb,
    spec_plot,
)


def main() -> None:
    N = 16
    D, x = cheb(N)
    # Exact for polynomials of degree <= N.
    values = x ** 4
    deriv_err = np.max(np.abs(D @ values - 4.0 * x ** 3))
    print(f"cheb: max |D x^4 - 4 x^3| = {deriv_err:.3e}")

    samples = np.cos(3.0 * np.arccos(x))  # T_3
    coeffs = real_to_cheb(samples)
    roundtrip = np.max(np.abs(cheb_to_real(coeffs) - samples))
    mode3 = coeffs[3]
    print(f"real_to_cheb / cheb_to_real: T_3 coeff = {mode3:.6f}, round-trip {roundtrip:.3e}")

    err, _recovered = cheb_test(samples)
    print(f"cheb_test / myFCT1: mean |error| = {np.mean(err):.3e}")
    _ = myFCT1(samples)

    query = np.array([-0.4, 0.2, 0.8])
    predicted = cheb_interpolate(coeffs, -1.0, 1.0, query)
    exact = np.cos(3.0 * np.arccos(query))
    print(f"cheb_interpolate: max |error| = {np.max(np.abs(predicted - exact)):.3e}")

    Dxx, Dx, x_mapped = AnMR(N, 1.0, 0.6)
    y = np.cos(np.pi * np.arange(N + 1) / N)
    # u(y) = y^2 is differentiated exactly; convert with the chain rule.
    u = y ** 2
    phase = 0.6 * (1.0 - y)
    xy = 0.6 * 2.0 * np.cosh(phase) / np.sinh(1.2)
    chain = (Dx @ u) - (2.0 * y / xy)
    print(f"AnMR: max chain-rule error = {np.max(np.abs(chain)):.3e}")
    print(f"AnMR: x in [{x_mapped.min():.3f}, {x_mapped.max():.3f}], Dxx shape {Dxx.shape}")

    D_mp, _x_mp = cheb_mp(8, dps=40)
    D8, _x8 = cheb(8)
    mp_err = max(abs(float(D_mp[i, j]) - D8[i, j]) for i in range(9) for j in range(9))
    print(f"cheb_mp: max |mp - float64| = {mp_err:.3e}")

    import matplotlib.pyplot as plt

    ax = spec_plot(np.exp(x))
    out = Path(__file__).resolve().parents[1] / "examples"
    out.mkdir(parents=True, exist_ok=True)
    figure_path = out / "spectrum_exp.png"
    ax.figure.savefig(figure_path, dpi=120, bbox_inches="tight")
    plt.close(ax.figure)
    print(f"spec_plot: wrote {figure_path}")


if __name__ == "__main__":
    main()
