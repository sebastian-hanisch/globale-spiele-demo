"""Normalverteilung ohne scipy: Handwerte, Umkehrung, Symmetrie, Bisektion."""

import math

import numpy as np
import pytest

from glo_core import SQ2PI, bisect, ncdf, npdf, nppf


def test_ncdf_hand_values():
    assert ncdf(0.0) == 0.5
    assert ncdf(1.0) == pytest.approx(0.8413447460685429, abs=1e-14)       # Tafelwert 0,8413
    assert ncdf(-1.959963984540054) == pytest.approx(0.025, abs=1e-12)
    assert ncdf(-40.0) < 1e-300 and ncdf(40.0) == 1.0


def test_ncdf_is_accurate_in_the_lower_tail():
    """Φ(−10) = 7,6199·10^-24: 1 − Φ(10) liefe auf 0 hinaus, erfc nicht (so liegen die Randwurzeln des Modells L)."""
    assert ncdf(-10.0) == pytest.approx(7.619853024160527e-24, rel=1e-12)


def test_ncdf_symmetry_and_array_equals_scalar():
    z = np.linspace(-6, 6, 41)
    assert np.allclose(ncdf(z) + ncdf(-z), 1.0, atol=1e-15)
    assert np.array_equal(ncdf(z), np.array([ncdf(float(v)) for v in z]))
    assert isinstance(ncdf(0.3), float) and ncdf(z).shape == z.shape


def test_npdf_hand_values():
    assert npdf(0.0) == pytest.approx(1 / SQ2PI, abs=1e-15) and npdf(0.0) == pytest.approx(0.3989422804014327, abs=1e-15)
    assert npdf(1.0) == pytest.approx(0.24197072451914337, abs=1e-14)
    assert np.allclose(npdf(np.array([-1.0, 1.0])), npdf(1.0))


def test_nppf_inverts_ncdf():
    for p in (1e-6, 0.01, 0.25, 0.5, 0.6, 0.975, 0.999999):
        assert ncdf(nppf(p)) == pytest.approx(p, rel=1e-10)
    assert nppf(0.5) == pytest.approx(0.0, abs=1e-12) and nppf(0.975) == pytest.approx(1.959963984540054, abs=1e-9)


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.5])
def test_nppf_rejects_values_outside_the_open_interval(p):
    with pytest.raises(ValueError):
        nppf(p)


def test_bisect_finds_roots_of_both_directions_and_geometric_mode():
    assert bisect(lambda t: t * t - 2.0, 0.0, 2.0) == pytest.approx(math.sqrt(2.0), abs=1e-14)
    assert bisect(lambda t: 2.0 - t * t, 0.0, 2.0) == pytest.approx(math.sqrt(2.0), abs=1e-14)
    assert bisect(lambda t: t - 3e-9, 1e-300, 1.0, geometric=True) == pytest.approx(3e-9, rel=1e-9)
