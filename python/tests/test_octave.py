"""Compare the Python port with GNU Octave on the original MATLAB files."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.io import loadmat, savemat

from chebyshev import (
    AnMR,
    cheb,
    cheb_interpolate,
    cheb_test,
    cheb_to_real,
    myFCT1,
    real_to_cheb,
)

REPO = Path(__file__).resolve().parents[2]
MFILE = Path(__file__).resolve().parent / "octave_reference.m"

# Octave and this port follow the same formulas in float64. Differentiation
# matrices grow like N^2, so the absolute tolerance scales up for N = 64.
# Transforms and interpolation stay near 1e-11 absolute.
ATOL_D = 1e-9
RTOL_D = 1e-11
ATOL_T = 1e-11
RTOL_T = 1e-11
ATOL_D64 = 1e-8
RTOL_D64 = 1e-10


def _octave_available() -> bool:
    try:
        subprocess.run(["octave", "--version"], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


pytestmark = pytest.mark.skipif(not _octave_available(), reason="GNU Octave is not installed")


def _align(matlab, python):
    """Compare values, allowing a MATLAB column to match a 1-D NumPy vector."""
    matlab = np.asarray(matlab, dtype=float)
    python = np.asarray(python, dtype=float)
    if python.shape == () and matlab.size == 1:
        return np.squeeze(matlab), python
    if matlab.ndim == 2 and 1 in matlab.shape and python.ndim == 1 and matlab.size == python.size:
        matlab = np.squeeze(matlab)
    return matlab, python


def _check(name, matlab, python, rtol, atol):
    matlab, python = _align(matlab, python)
    assert matlab.shape == python.shape, f"{name}: shape {matlab.shape} vs {python.shape}"
    if not np.allclose(matlab, python, rtol=rtol, atol=atol, equal_nan=True):
        finite = np.isfinite(matlab) & np.isfinite(python)
        if finite.any():
            err = np.max(np.abs(matlab[finite] - python[finite]))
        else:
            err = np.nan
        raise AssertionError(f"{name}: max abs diff {err:.3e} (rtol={rtol}, atol={atol})")


@pytest.fixture(scope="module")
def octave_data(tmp_path_factory):
    rng = np.random.default_rng(20260314)
    f16 = rng.normal(size=(17, 1))
    f8 = rng.normal(size=(9, 1))
    f64 = rng.normal(size=(65, 1))
    ac = np.zeros((17, 1))
    ac[0, 0] = 0.4
    ac[1, 0] = -0.2
    ac[2, 0] = 0.7
    ac[3, 0] = 0.1
    ac[-1, 0] = -0.05
    work = tmp_path_factory.mktemp("octave")
    inputs = work / "inputs.mat"
    outputs = work / "outputs.mat"
    savemat(
        inputs,
        {
            "f16": f16,
            "f16_row": np.ascontiguousarray(f16.T),
            "f16_mat": np.column_stack([f16.ravel(), 0.3 * f16.ravel(), -f16.ravel()]),
            "f8": f8,
            "f64": f64,
            "a_odd": np.array([[0.2], [1.0], [-0.5], [0.25]]),
            "ac": ac,
            "ar": np.array([[0.0, 0.0, 1.0, 0.0, 0.0]]),
            "Amat": np.array([[0.0, 1.0], [1.0, 0.0], [0.0, -0.5], [0.0, 0.0], [0.0, 0.0]]),
            "xs": np.array([[0.3]]),
            "xcol": np.array([[-0.7], [0.2], [0.9]]),
            "xrow": np.array([[-0.7, 0.2, 0.9]]),
            "x_domain": np.array([[0.0], [1.5], [4.0]]),
        },
        format="5",
    )
    completed = subprocess.run(
        [
            "octave",
            "--no-gui",
            "--quiet",
            str(MFILE),
            str(inputs),
            str(outputs),
            str(REPO),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Octave reference failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    loaded = loadmat(outputs)
    loaded["_stderr"] = completed.stderr
    loaded["_inputs"] = loadmat(inputs)
    return loaded


def test_cheb_matches_octave(octave_data):
    cases = [
        (16, "D16", "x16", ATOL_D, RTOL_D, None),
        (32, "D32", "x32", ATOL_D, RTOL_D, None),
        (64, "D64", "x64", ATOL_D64, RTOL_D64, None),
        (1, "D1", "x1", ATOL_D, RTOL_D, None),
    ]
    for N, d_name, x_name, atol, rtol, _endpoints in cases:
        D, x = cheb(N)
        _check(d_name, octave_data[d_name], D, rtol, atol)
        _check(x_name, octave_data[x_name], x, rtol, atol)
    _check("D0", octave_data["D0"], cheb(0)[0], RTOL_D, ATOL_D)
    _check("x0", octave_data["x0"], cheb(0)[1], RTOL_D, ATOL_D)
    D, x = cheb(16, -2.0, 5.0)
    _check("Dmap", octave_data["Dmap"], D, RTOL_D, ATOL_D)
    _check("xmap", octave_data["xmap"], x, RTOL_D, ATOL_D)
    D, x = cheb(8, 3.0, -1.0)
    _check("Drev", octave_data["Drev"], D, RTOL_D, ATOL_D)
    _check("xrev", octave_data["xrev"], x, RTOL_D, ATOL_D)


def test_transforms_match_octave(octave_data):
    inputs = octave_data["_inputs"]
    pairs = [
        ("F16", myFCT1(inputs["f16"]), ATOL_T),
        ("F16_row", myFCT1(inputs["f16_row"]), ATOL_T),
        ("F16_mat", myFCT1(inputs["f16_mat"]), ATOL_T),
        ("F8", myFCT1(inputs["f8"]), ATOL_T),
        ("F64", myFCT1(inputs["f64"]), 5e-10),
        ("a16", real_to_cheb(inputs["f16"]), ATOL_T),
        ("a16_row", real_to_cheb(inputs["f16_row"]), ATOL_T),
        ("a16_mat", real_to_cheb(inputs["f16_mat"]), ATOL_T),
        ("a8", real_to_cheb(inputs["f8"]), ATOL_T),
        ("a64", real_to_cheb(inputs["f64"]), 5e-10),
        ("p16", cheb_to_real(real_to_cheb(inputs["f16"])), ATOL_T),
        ("p16_row", cheb_to_real(real_to_cheb(inputs["f16_row"])), ATOL_T),
        ("p16_mat", cheb_to_real(real_to_cheb(inputs["f16_mat"])), ATOL_T),
        ("p_from_user", cheb_to_real(inputs["ac"]), ATOL_T),
        ("p_odd", cheb_to_real(inputs["a_odd"]), ATOL_T),
    ]
    for name, python, atol in pairs:
        # Keep MATLAB orientation (do not squeeze matrices).
        matlab = np.asarray(octave_data[name], dtype=float)
        python = np.asarray(python, dtype=float)
        assert matlab.shape == python.shape, f"{name}: {matlab.shape} vs {python.shape}"
        assert_allclose(matlab, python, rtol=RTOL_T, atol=atol, err_msg=name)

    for err_name, p_name, samples in (
        ("err16", "ptest16", inputs["f16"]),
        ("err_row", "ptest_row", inputs["f16_row"]),
        ("err_mat", "ptest_mat", inputs["f16_mat"]),
    ):
        err, recovered = cheb_test(samples)
        assert_allclose(octave_data[err_name], err, rtol=RTOL_T, atol=ATOL_T, err_msg=err_name)
        assert_allclose(octave_data[p_name], recovered, rtol=RTOL_T, atol=ATOL_T, err_msg=p_name)


def test_interpolate_and_mesh_match_octave(octave_data):
    inputs = octave_data["_inputs"]
    ac = inputs["ac"]
    checks = {
        "v_s": cheb_interpolate(ac, -1.0, 1.0, inputs["xs"]),
        "v_col": cheb_interpolate(ac, -1.0, 1.0, inputs["xcol"]),
        "v_row": cheb_interpolate(ac, -1.0, 1.0, inputs["xrow"]),
        "v_domain": cheb_interpolate(ac, 0.0, 4.0, inputs["x_domain"]),
        "v_swap": cheb_interpolate(ac, 4.0, 0.0, inputs["x_domain"]),
        "v_mat_s": cheb_interpolate(inputs["Amat"], -1.0, 1.0, inputs["xs"]),
        "v_mat_col": cheb_interpolate(inputs["Amat"], -1.0, 1.0, inputs["xcol"]),
        "v_mat_row": cheb_interpolate(inputs["Amat"], -1.0, 1.0, inputs["xrow"]),
        "v_row_a": cheb_interpolate(inputs["ar"], -1.0, 1.0, inputs["xcol"]),
        "v_clip": cheb_interpolate(ac, -1.0, 1.0, 1.0000001),
    }
    shape_sensitive = {
        "v_col",
        "v_row",
        "v_domain",
        "v_swap",
        "v_mat_col",
        "v_mat_row",
        "v_mat_s",
        "v_row_a",
    }
    for name, python in checks.items():
        matlab = np.asarray(octave_data[name], dtype=float)
        python = np.asarray(python, dtype=float)
        if name in shape_sensitive:
            assert matlab.shape == python.shape, f"{name}: {matlab.shape} vs {python.shape}"
            assert_allclose(matlab, python, rtol=RTOL_T, atol=ATOL_T, err_msg=name)
            continue
        matlab_c = np.squeeze(matlab)
        python_c = np.squeeze(np.atleast_1d(python))
        assert matlab_c.shape == python_c.shape, f"{name}: {matlab.shape} vs {python.shape}"
        assert_allclose(matlab_c, python_c, rtol=RTOL_T, atol=ATOL_T, err_msg=name)

    for N, xB, kappa, names in (
        (16, 1.0, 0.8, ("Dxx", "Dx", "xam")),
        (12, -1.0, 1.5, ("Dxx2", "Dx2", "xam2")),
        (4, 1.0, 0.7, ("Dxx4", "Dx4", "xam4")),
    ):
        Dxx, Dx, x = AnMR(N, xB, kappa)
        _check(names[0], octave_data[names[0]], Dxx, 1e-10, 1e-8)
        _check(names[1], octave_data[names[1]], Dx, 1e-10, 1e-9)
        _check(names[2], octave_data[names[2]], x, RTOL_T, ATOL_T)


def test_spec_plot_series_matches_octave(octave_data):
    samples = octave_data["_inputs"]["f16"]
    series = np.log10(np.abs(real_to_cheb(samples)))
    _check("spec", octave_data["spec"], series, RTOL_T, ATOL_T)
    # spec_plot.m is executed inside the Octave driver. A failure there is
    # reported; the series above is what the plot would draw.
    ok = int(np.squeeze(octave_data["spec_plot_ok"]))
    if ok != 1:
        message = octave_data["spec_plot_msg"]
        pytest.fail(f"Octave could not execute spec_plot.m: {message}")
