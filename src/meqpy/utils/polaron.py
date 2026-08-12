from typing import Callable

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from ..utils.constants import HBAR_EV
from ..utils.types import (
    validate_nonnegative_float,
    require_type,
    validate_real_or_1darray,
    validate_positive_float,
)


def polaron_spectrum(
    J: np.ndarray,
    dx: float,
    energy_range: float = 1.0,
    lorentzian: float = 1e-3,
    gaussian: float = 1e-3,
    integrate=True,
) -> Callable[[float], float]:
    """Create callable Franck-Condon spectrum for given phonon density `J`, using polaron model.

    Parameters
    ----------
    J : np.ndarray
        1d-array of phonon density. Corresponding x values are assumed to start at zero and have constant spacing dx.
    dx : float
        Energy spacing in eV
    energy_range : float
        Energy range for Fourier Transformation in eV, default is 1.0
    lorentzian : float, optional
        Half-width-half-maxium of lorentzian broadening, default 1e-3
    gaussian : float, optional
        Half-width-half-maxium of gaussian broadening, default is 1e-3
    integrate : bool, optional
        Integrate spectrum to obtain normalized lineshape, default is True

    Returns
    -------
    Callable[[float], float]
        (Integrated) polaron lineshape.
    """

    validate_positive_float(dx, "dx")
    validate_positive_float(energy_range, "energy_range")
    validate_nonnegative_float(lorentzian, "lorentzian")
    validate_nonnegative_float(gaussian, "gaussian")
    require_type(integrate, bool, "integrate")

    J = _validate_J(J, dx, energy_range)

    x = np.arange(-energy_range, energy_range, dx)
    npnts = len(x)

    # x values in time-domain
    ft = np.fft.fftfreq(npnts, d=dx / (2 * np.pi * HBAR_EV))

    fy = np.fft.ifft(J * dx, n=npnts, norm="forward")
    corrfunc = np.exp(fy)  # "convolution" in time-domain

    # apply broadenings
    if lorentzian:
        lorentzian_shape_fft = np.fft.ifft(
            _lorentzian(x, lorentzian) * dx, n=npnts, norm="forward"
        )
        corrfunc *= lorentzian_shape_fft * np.exp(1j * x[0] / HBAR_EV * ft)

    if gaussian:
        gaussian_shape_fft = np.fft.ifft(
            _gaussian(x, gaussian) * dx, n=npnts, norm="forward"
        )
        corrfunc *= gaussian_shape_fft * np.exp(1j * x[0] / HBAR_EV * ft)

    # transfrom back into energy-domain
    Sy = np.fft.fft(corrfunc / dx, norm="forward")
    Sy *= 1 / np.exp(np.sum(J * dx))  # renorm spectrum, such that np.sum(Sy)*dE = 1
    Sy = np.fft.fftshift(Sy).real

    # x values in energy-domain
    Sx = np.fft.fftfreq(npnts) * dx * npnts
    Sx = np.fft.fftshift(Sx)

    if integrate:
        Sy = cumulative_trapezoid(Sy, Sx, initial=0)
        Sy /= np.max(Sy)
        fill_value = (0.0, 1.0)
    else:
        fill_value = (0.0, 0.0)

    # make callable interpolation function
    lineshape = interp1d(
        Sx, Sy, kind="cubic", bounds_error=False, fill_value=fill_value
    )

    return lineshape


# ------------------------


def Jrect(x: np.ndarray, reorg_energy: float, x_min: float, x_max: float) -> np.ndarray:
    """Rectangular phonon dispersion spectrum.

    Parameters
    ----------
    x : np.ndarray
        1d-array of energy values in eV, starting at 0 and constant positive gradient.
    reorg_energy : float
        Total reorganization energy in eV.
    x_min : float
        Lower bound of constant phonon dispersion in eV.
    x_max : float
        Upper bound of constant phonon dispersion in eV.

    Returns
    -------
    J : np.ndarray
        Phonon dispersion spectrum.
    """

    _validate_x(x)
    validate_nonnegative_float(reorg_energy, "reorg_energy")
    validate_nonnegative_float(x_min, "x_min")
    validate_nonnegative_float(x_max, "x_max")
    if x_max <= x_min:
        raise ValueError("x_max must be larger than x_min")

    rect = np.heaviside(x - x_min, 1.0) * np.heaviside(x_max - x, 1.0)
    return rect * reorg_energy / np.sum(rect * x)


def _lorentzian(x: np.ndarray, hwhm: float) -> np.ndarray:
    """Lorentzian lineshape"""
    return hwhm / np.pi / (hwhm**2 + x**2)


def _gaussian(x, hwhm):
    """Gaussian lineshape"""
    return np.sqrt(np.log(2) / np.pi) / hwhm * np.exp(-(np.log(2) * x**2 / hwhm**2))


def _validate_x(x: np.ndarray):
    """Validate x is 1-dimensional array starting at 0 and constant positive gradient."""

    require_type(x, np.ndarray, "x")
    if x.ndim != 1:
        raise ValueError(f"x must be 1-dimensional, but got {x.ndim}D array.")

    if x[0] != 0.0:
        raise ValueError(f"x must start at 0, but first element is at {x[0]}")

    dx = np.gradient(x)
    if not np.all(np.isclose(dx, dx[0])) or dx[0] <= 0:
        raise ValueError("x must increase with constant step size.")


def _validate_J(J: np.ndarray, dx: float, energy_range: float):
    """Validate J is array of non-negative values and within energy range"""

    J = validate_real_or_1darray(J, "J")

    if len(J) * dx > energy_range:
        raise ValueError(
            f"Energy range of J ({len(J) * dx}eV) is exceeding energy_range of {energy_range}eV."
        )

    if any(J < 0):
        raise ValueError("J must not contain negative values.")

    return J
