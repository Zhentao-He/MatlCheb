"""Chebyshev pseudospectral routines ported from the MATLAB sources.

Each public name matches the MATLAB file it came from. See ``python/README.md``
for the file map and for how to run the checks.
"""

from .cheb import cheb
from .cheb_mp import cheb_mp
from .cheb_test import cheb_test
from .fct import myFCT1
from .interpolate import cheb_interpolate
from .mesh import AnMR
from .plotting import spec_plot
from .transform import cheb_to_real, real_to_cheb

__all__ = [
    "AnMR",
    "cheb",
    "cheb_interpolate",
    "cheb_mp",
    "cheb_test",
    "cheb_to_real",
    "myFCT1",
    "real_to_cheb",
    "spec_plot",
]
