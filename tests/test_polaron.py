from typing import Callable
import numpy as np
import pytest

from meqpy.utils import polaron_spectrum, Jrect


class TestJrect:
    def test_reorg(self):
        x = np.arange(0, 50e-3, 1e-3)
        reorg_energy = 150e-3
        J = Jrect(x, reorg_energy, 10e-3, 20e-3)
        assert np.isclose(reorg_energy, np.sum(J * x))

    def test_step_bounds(self):
        x = np.arange(0, 50e-3, 1e-3)
        J = Jrect(x, 150e-3, 10e-3, 20e-3)
        assert J[9] == J[21] == 0.0
        assert J[10] == J[20]

    def test_validate_x_array(self):
        # wrong type
        x = 1.0
        with pytest.raises(TypeError) as e_info:
            Jrect(x, 150e-3, 10e-3, 20e-3)
        assert str(e_info.value) == "x must be ndarray, but got float"

        # not 1D array
        x = np.zeros((4, 4))
        with pytest.raises(ValueError) as e_info:
            Jrect(x, 150e-3, 10e-3, 20e-3)
        assert "x must be 1-dimensional, but got " in str(e_info.value)

        # not starting with 0
        x = np.arange(10e-3, 50e-3, 1e-3)
        with pytest.raises(ValueError) as e_info:
            Jrect(x, 150e-3, 10e-3, 20e-3)
        assert "x must start at 0, but first element is at" in str(e_info.value)

        # not increasing
        x = np.arange(0, 50e-3, 1e-3) * -1
        with pytest.raises(ValueError) as e_info:
            Jrect(x, 150e-3, 10e-3, 20e-3)
        assert str(e_info.value) == "x must increase with constant step size."

        # not constant step size
        x = np.array([0, 1, 2, 4])
        with pytest.raises(ValueError) as e_info:
            Jrect(x, 150e-3, 10e-3, 20e-3)
        assert str(e_info.value) == "x must increase with constant step size."

    def test_xmin_xmax_negative(self):
        x = np.arange(0, 50e-3, 1e-3)

        with pytest.raises(ValueError) as e_info:
            Jrect(x, 150e-3, -10e-3, 20e-3)
        assert "x_min must be >= 0" in str(e_info.value)

        with pytest.raises(ValueError) as e_info:
            Jrect(x, 150e-3, 10e-3, -20e-3)
        assert "x_max must be >= 0" in str(e_info.value)

    def test_xmax_larger_xmin(self):
        x = np.arange(0, 50e-3, 1e-3)

        with pytest.raises(ValueError) as e_info:
            Jrect(x, 150e-3, 20e-3, 10e-3)
        assert str(e_info.value) == "x_max must be larger than x_min"


class TestSpectrum:
    def test_returns_callable(self):
        dx = 1e-3
        x = np.arange(0, 50e-3, dx)
        J = Jrect(x, 150e-3, 10e-3, 20e-3)
        lineshape = polaron_spectrum(J, dx)
        assert isinstance(lineshape, Callable)
        assert isinstance(lineshape(x), np.ndarray)

    def test_mean_energy_matches_reorg_energy(self):
        # test underlying physical assumption
        dx = 1e-4
        x = np.arange(0, 50e-3, dx)
        reorg_energy = 150e-3
        J = Jrect(x, reorg_energy, 10e-3, 20e-3)

        lineshape = polaron_spectrum(
            J, dx, energy_range=1.0, lorentzian=1e-3, gaussian=1e-3, integrate=False
        )
        xx = np.arange(-1.0, 1.0, dx)
        yy = lineshape(xx)

        area = np.trapezoid(yy, xx)
        mean_energy = np.sum(yy * xx) / area

        assert np.isclose(area, 1.0, atol=1e-3)
        assert np.isclose(mean_energy, reorg_energy, atol=1e-3)

    def test_boundary_integrated(self):
        dx = 1e-3
        x = np.arange(0, 50e-3, dx)
        J = Jrect(x, 150e-3, 10e-3, 20e-3)
        lineshape = polaron_spectrum(J, dx)
        assert lineshape(-1.5) == 0.0
        assert lineshape(1.5) == 1.0

    def test_boundary_not_integrated(self):
        dx = 1e-3
        x = np.arange(0, 50e-3, dx)
        J = Jrect(x, 150e-3, 10e-3, 20e-3)
        lineshape = polaron_spectrum(J, dx, integrate=False)
        int_lineshape = np.sum(lineshape(np.arange(-1, 1, 1e-3))) * 1e-3
        assert lineshape(-1.5) == 0.0
        assert lineshape(1.5) == 0.0
        assert np.isclose(int_lineshape, 1.0, atol=1e-2)

    def test_lorentzian_hwhm(self):
        hwhm = 50e-3
        lineshape = polaron_spectrum(
            0, 1e-4, lorentzian=hwhm, gaussian=0, integrate=False
        )
        assert np.isclose(lineshape(hwhm) / lineshape(0), 0.5, atol=1e-5)
        assert np.isclose(lineshape(-hwhm) / lineshape(0), 0.5, atol=1e-5)

    def test_gaussian_hwhm(self):
        hwhm = 50e-3
        lineshape = polaron_spectrum(
            0, 1e-4, lorentzian=0, gaussian=hwhm, integrate=False
        )
        assert np.isclose(lineshape(hwhm) / lineshape(0), 0.5, atol=1e-5)
        assert np.isclose(lineshape(-hwhm) / lineshape(0), 0.5, atol=1e-5)

    def test_J_type_validation(self):
        with pytest.raises(TypeError):
            polaron_spectrum("not_an_array", 1e-3)

    def test_integrate_type_validation(self):
        dx = 1e-3
        x = np.arange(0, 50e-3, dx)
        J = Jrect(x, 150e-3, 10e-3, 20e-3)
        with pytest.raises(TypeError):
            polaron_spectrum(J, dx, integrate="yes")

    def test_J_domain_must_fit_inside_energy_range(self):
        dx = 1e-3
        x_long = np.arange(0, 2.0, dx)
        J_long = Jrect(x_long, 0.15, 10e-3, 20e-3)

        with pytest.raises(ValueError):
            polaron_spectrum(J_long, dx, energy_range=0.05)

    def test_negative_J_rejected(self):
        dx = 1e-3
        x = np.arange(0, 50e-3, dx)
        J = -Jrect(x, 0.15, 10e-3, 20e-3)
        with pytest.raises(ValueError) as e_info:
            polaron_spectrum(J, dx, integrate=False)
        assert str(e_info.value) == "J must not contain negative values."
