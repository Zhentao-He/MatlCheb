"""Analytic checks that do not need Octave."""

import warnings

import numpy as np
import pytest
import scipy.fft
from numpy.testing import assert_allclose

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


def _direct_coeffs(f):
    """Discrete Chebyshev coefficients from the cosine sum."""
    f = np.asarray(f, dtype=float).reshape(-1)
    N = f.size - 1
    j = np.arange(N + 1)
    w = np.ones(N + 1)
    w[0] = w[-1] = 0.5
    # angles[j, n] = pi * j * n / N; sum over nodes j.
    angles = np.pi * j[:, None] * j[None, :] / N
    raw = (w * f) @ np.cos(angles)
    scale = np.full(N + 1, 2.0 / N)
    scale[0] = scale[-1] = 1.0 / N
    return scale * raw


def _scipy_dct_coeffs(f):
    f = np.asarray(f, dtype=float).reshape(-1)
    N = f.size - 1
    transformed = scipy.fft.dct(f, type=1)
    coeffs = transformed / N
    coeffs[0] *= 0.5
    coeffs[-1] *= 0.5
    return coeffs


def _closed_form_D(N):
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1)
    c[0] = c[-1] = 2.0
    D = np.zeros((N + 1, N + 1))
    idx = np.arange(N + 1)
    for i in idx:
        for j in idx:
            if i != j:
                D[i, j] = (c[i] / c[j]) * ((-1.0) ** (i + j)) / (x[i] - x[j])
    D[0, 0] = (2.0 * N ** 2 + 1.0) / 6.0
    D[-1, -1] = -D[0, 0]
    for i in range(1, N):
        D[i, i] = -x[i] / (2.0 * (1.0 - x[i] ** 2))
    return D, x


def test_cheb_N0_and_N1():
    D0, x0 = cheb(0)
    assert D0 == 0.0
    assert x0 == 1.0
    D, x = cheb(1)
    assert_allclose(x, [1.0, -1.0])
    assert_allclose(D, [[0.5, -0.5], [0.5, -0.5]])
    assert_allclose(D @ x, np.ones(2))


def test_cheb_row_sums_and_polynomials():
    for N in (4, 8, 17, 32):
        D, x = cheb(N)
        assert_allclose(D.sum(axis=1), 0.0, atol=1e-12)
        assert_allclose(D @ np.ones(N + 1), 0.0, atol=1e-12)
        assert_allclose(D @ x, np.ones(N + 1), atol=1e-9, rtol=1e-12)
        # Exact on polynomials of degree <= N. x^5 needs N >= 5.
        degree = min(5, N)
        values = x ** degree
        expected = degree * x ** (degree - 1)
        assert_allclose(D @ values, expected, atol=1e-8, rtol=1e-9)


def test_cheb_matches_closed_form_off_diagonal():
    N = 20
    D, x = cheb(N)
    closed, x_closed = _closed_form_D(N)
    assert_allclose(x, x_closed)
    mask = ~np.eye(N + 1, dtype=bool)
    assert_allclose(D[mask], closed[mask], rtol=1e-12, atol=1e-12)
    # The diagonal formula is algebraically equivalent but less stable.
    assert_allclose(np.diag(D), np.diag(closed), rtol=1e-8, atol=1e-8)


def test_cheb_affine_map():
    D, x = cheb(12, -2.0, 5.0)
    assert_allclose(x[0], 5.0)
    assert_allclose(x[-1], -2.0)
    assert_allclose(D @ np.ones_like(x), 0.0, atol=1e-11)
    assert_allclose(D @ x, np.ones_like(x), atol=1e-9)
    # Reversed endpoints flip the chain-rule sign, as in cheb.m.
    D_rev, x_rev = cheb(8, 3.0, -1.0)
    assert_allclose(x_rev[0], -1.0)
    assert_allclose(x_rev[-1], 3.0)
    assert_allclose(D_rev @ x_rev, np.ones_like(x_rev), atol=1e-9)


def test_cheb_rejects_bad_degree():
    with pytest.raises(ValueError):
        cheb(-1)
    with pytest.raises(ValueError):
        cheb(2.5)
    with pytest.raises(ValueError):
        cheb(4, 0.0)
    with pytest.raises(ValueError):
        cheb(4, 1.0, 1.0)


def test_myFCT1_known_N2_and_roundtrip():
    samples = np.array([1.0, 2.0, 3.0])
    assert_allclose(myFCT1(samples), [4.0, -1.0, 0.0])
    assert_allclose(real_to_cheb(samples), [2.0, -1.0, 0.0])
    # f = 2 - x on {1, 0, -1}.
    assert_allclose(cheb_to_real(real_to_cheb(samples)), samples, atol=1e-14)
    query = np.array([0.3, -0.2])
    assert_allclose(cheb_interpolate(real_to_cheb(samples), -1.0, 1.0, query), 2.0 - query)


def test_myFCT1_matches_coeff_definitions():
    rng = np.random.default_rng(1)
    for N in (2, 8, 16, 32):
        samples = rng.normal(size=N + 1)
        coeffs = real_to_cheb(samples)
        assert_allclose(coeffs, _direct_coeffs(samples), rtol=1e-11, atol=1e-11)
        assert_allclose(coeffs, _scipy_dct_coeffs(samples), rtol=1e-11, atol=1e-11)
        assert_allclose(cheb_to_real(coeffs), samples, rtol=1e-11, atol=1e-11)
        err, recovered = cheb_test(samples)
        assert_allclose(recovered, samples, rtol=1e-11, atol=1e-11)
        assert np.mean(err) < 1e-13


def test_orientation_and_columns():
    rng = np.random.default_rng(2)
    column = rng.normal(size=(17, 1))
    row = column.T.copy()
    matrix = rng.normal(size=(17, 3))
    assert myFCT1(column).shape == (17, 1)
    assert myFCT1(row).shape == (1, 17)
    assert myFCT1(matrix).shape == (17, 3)
    assert_allclose(myFCT1(row).ravel(), myFCT1(column).ravel())
    assert_allclose(real_to_cheb(matrix), np.column_stack([real_to_cheb(matrix[:, k]) for k in range(3)]))
    assert real_to_cheb(row).shape == (1, 17)
    assert cheb_to_real(real_to_cheb(matrix)).shape == (17, 3)
    assert_allclose(cheb_to_real(real_to_cheb(matrix)), matrix, atol=1e-11)


def test_chebyshev_polynomials_are_unit_coefficients():
    N = 16
    _D, x = cheb(N)
    theta = np.arccos(x)
    for mode in (0, 1, 3, 7, 16):
        samples = np.cos(mode * theta)
        coeffs = real_to_cheb(samples)
        expected = np.zeros(N + 1)
        expected[mode] = 1.0
        assert_allclose(coeffs, expected, atol=1e-12)
        assert_allclose(cheb_to_real(coeffs), samples, atol=1e-12)


def test_cheb_to_real_odd_degree_uses_cosine_matrix():
    # N = 3 (odd): values of T_1 and T_2 on four nodes.
    nodes = np.cos(np.pi * np.arange(4) / 3.0)
    coeffs = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    values = cheb_to_real(coeffs)
    expected = np.column_stack([nodes, 2.0 * nodes ** 2 - 1.0])
    assert values.shape == (4, 2)
    assert_allclose(values, expected, atol=1e-14)


def test_interpolate_scalar_row_column_and_domain():
    coeffs = np.array([0.0, 0.0, 1.0])  # T_2
    assert cheb_interpolate(coeffs, -1.0, 1.0, 0.3).shape == ()
    assert_allclose(cheb_interpolate(coeffs, -1.0, 1.0, 0.3), 2.0 * 0.3 ** 2 - 1.0)

    column_x = np.array([[-0.5], [0.25], [0.9]])
    out = cheb_interpolate(coeffs.reshape(1, -1), -1.0, 1.0, column_x)
    assert out.shape == (3, 1)
    z = column_x.ravel()
    assert_allclose(out.ravel(), 2.0 * z ** 2 - 1.0)

    row_x = np.array([[-0.2, 0.4]])
    row_out = cheb_interpolate(coeffs, -1.0, 1.0, row_x)
    assert row_out.shape == (1, 2)

    # On [0, 4], T_1(z) = z = x/2 - 1.
    linear = np.array([0.0, 1.0])
    physical = np.array([0.0, 2.0, 4.0])
    # cos(arccos(z)) is the same evaluation MATLAB uses; it is not bit-exact at z = 0.
    assert_allclose(
        cheb_interpolate(linear, 0.0, 4.0, physical),
        physical / 2.0 - 1.0,
        atol=1e-15,
        rtol=0.0,
    )
    assert_allclose(
        cheb_interpolate(linear, 4.0, 0.0, physical),
        cheb_interpolate(linear, 0.0, 4.0, physical),
    )
    # Clipped back onto the endpoint z = 1.
    assert_allclose(cheb_interpolate(coeffs, -1.0, 1.0, 1.0000001), 1.0)

    multi = np.column_stack([np.array([0.0, 1.0, 0.0]), coeffs])
    scalar_multi = cheb_interpolate(multi, -1.0, 1.0, 0.3)
    assert scalar_multi.shape == (2, 1)
    assert_allclose(scalar_multi.ravel(), [0.3, 2.0 * 0.3 ** 2 - 1.0])
    grid = cheb_interpolate(multi, -1.0, 1.0, np.array([0.3, -0.5]))
    assert grid.shape == (2, 2)


def test_interpolate_rejects_bad_interval_and_shape():
    with pytest.raises(ValueError):
        cheb_interpolate([1.0, 0.0], 1.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        cheb_interpolate([1.0, 0.0], -1.0, 1.0, np.ones((2, 2)))
    with pytest.raises(ValueError):
        myFCT1(np.ones((2, 2, 2)))
    with pytest.raises(ValueError):
        real_to_cheb(np.ones(4))  # length 4 => N = 3 odd
    with pytest.raises(ValueError):
        myFCT1(np.ones(4))


def test_anmr_chain_rule_is_exact_for_polynomials_in_y():
    N = 16
    kappa = 0.8
    xB = 1.0
    Dxx, Dx, x = AnMR(N, xB, kappa)
    y = np.cos(np.pi * np.arange(N + 1) / N)
    phase = kappa * (1.0 - xB * y)
    xy = kappa * (xB ** 2) * 2.0 * np.cosh(phase) / np.sinh(2.0 * kappa)
    xyy = -2.0 * (kappa ** 2) * (xB ** 3) / np.sinh(2.0 * kappa) * np.sinh(phase)
    assert_allclose(x[0], 1.0, atol=1e-12)
    assert_allclose(x[-1], -1.0, atol=1e-12)
    assert np.all(np.diff(x) < 0.0)

    u = y ** 3 + 0.5 * y
    up = 3.0 * y ** 2 + 0.5
    upp = 6.0 * y
    assert_allclose(Dx @ u, up / xy, atol=1e-9, rtol=1e-10)
    second = upp / xy ** 2 - up * xyy / xy ** 3
    assert_allclose(Dxx @ u, second, atol=1e-8, rtol=1e-9)

    # Clustering: the first interior gap is smaller near xB = +1 than near -1.
    assert (x[0] - x[1]) < (x[-2] - x[-1])


def test_anmr_other_endpoint_and_zero_kappa():
    _Dxx, _Dx, x = AnMR(10, -1.0, 1.3)
    assert_allclose(x[0], 1.0, atol=1e-12)
    assert_allclose(x[-1], -1.0, atol=1e-12)


def test_anmr_kappa_zero_is_cheb():
    """kappa = 0 is the removable limit, i.e. the unclustered Chebyshev grid."""
    for N, xB in ((4, 1.0), (8, -1.0), (15, 1.0)):
        Dxx, Dx, x = AnMR(N, xB, 0.0)
        Dy, y = cheb(N)
        assert_allclose(x, y)
        assert_allclose(Dx, Dy)
        assert_allclose(Dxx, Dy @ Dy)
        # A small positive kappa, still on the original formula, sits near that limit.
        Dxx_s, Dx_s, x_s = AnMR(N, xB, 1e-4)
        assert_allclose(x_s, y, atol=1e-7, rtol=0.0)
        assert_allclose(Dx_s, Dy, atol=1e-5, rtol=1e-8)
        assert_allclose(Dxx_s, Dy @ Dy, atol=1e-4, rtol=1e-8)


def test_cheb_mp_matches_float_and_closed_form():
    import mpmath as mp

    N = 10
    D_ref, x_ref = cheb(N)
    D_mp, x_mp = cheb_mp(N, dps=40)
    float_D = np.array([[float(D_mp[i, j]) for j in range(N + 1)] for i in range(N + 1)])
    float_x = np.array([float(x_mp[i, 0]) for i in range(N + 1)])
    assert_allclose(float_D, D_ref, rtol=1e-12, atol=1e-12)
    assert_allclose(float_x, x_ref, rtol=1e-14, atol=1e-14)

    with mp.workdps(40):
        closed, _x = _mp_closed_form(N)
        for i in range(N + 1):
            for j in range(N + 1):
                assert abs(D_mp[i, j] - closed[i, j]) < mp.mpf("1e-30")
        row_sum = [mp.fsum(D_mp[i, j] for j in range(N + 1)) for i in range(N + 1)]
        assert max(abs(value) for value in row_sum) < mp.mpf("1e-30")

    zero, one = cheb_mp(0, dps=20)
    assert zero == 0
    assert one == 1


def _mp_closed_form(N):
    import mpmath as mp

    x = [mp.cos(mp.pi * i / N) for i in range(N + 1)]
    c = [mp.mpf(2 if i in (0, N) else 1) for i in range(N + 1)]
    D = mp.matrix(N + 1)
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i, j] = (c[i] / c[j]) * ((-1) ** (i + j)) / (x[i] - x[j])
    D[0, 0] = (2 * N ** 2 + 1) / mp.mpf(6)
    D[N, N] = -D[0, 0]
    for i in range(1, N):
        D[i, i] = -x[i] / (2 * (1 - x[i] ** 2))
    return D, x


def test_spec_plot_draws_log10_coefficients():
    _D, x = cheb(16)
    samples = np.exp(x)
    ax = spec_plot(samples)
    ydata = ax.lines[0].get_ydata()
    with np.errstate(divide="ignore"):
        expected = np.log10(np.abs(real_to_cheb(samples)))
    assert_allclose(ydata, expected)
    ax.figure.clf()


def test_cheb_test_does_not_warn_on_smooth_data():
    _D, x = cheb(16)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        cheb_test(np.cos(x))
