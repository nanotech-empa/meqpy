from numbers import Real
import numpy as np
from scipy.special import erf
from .types import ValidatedEnum, is_nonnegative_float


class LineShape(str, ValidatedEnum):
    """Line shape for transition rate derivative"""

    GAUSS = "gaussian"
    LOR = "lorentzian"
    DIRAC = "dirac"


def lineshape_integral(lineshape: LineShape | str, x: float | np.ndarray, hwhm: Real):
    """Calculate integral over lineshape.

    Parameters
    ----------
    lineshape: LineShape | str
        Which lineshape to use, options are "dirac", "gaussian" or "lorentzian".
    x : (M,) np.ndarray | float
        Energy variable.
    hwhm : float
        Half width at half maximum of lineshape. If hwhm == 0, lineshape defaults to "dirac".

    Returns
    -------
    integral : (M,) np.ndarray | float
        Integral over lineshape.
    """

    if isinstance(lineshape, str):
        lineshape = LineShape(lineshape)

    if not isinstance(lineshape, LineShape):
        raise TypeError(f"lineshape must be type LineShape, but got {type(lineshape)}")

    if not isinstance(x, (Real, np.ndarray)):
        raise TypeError(f"x must be Real or np.ndarray, but got {type(x)}")

    is_nonnegative_float(hwhm, "hwhm")

    if hwhm == 0:
        lineshape = LineShape.DIRAC

    match lineshape:
        case LineShape.DIRAC:
            return dirac_lineshape_integral(x)

        case LineShape.GAUSS:
            return gaussian_lineshape_integral(x, hwhm)

        case LineShape.LOR:
            return lorentzian_lineshape_integral(x, hwhm)


def dirac_lineshape_integral(x: np.ndarray | float) -> np.ndarray | float:
    """Calculate integral over dirac peak (Heaviside function).

    Parameters
    ----------
    x : (M,) np.ndarray | float
        Energy variable.

    Returns
    -------
    integral : (M,) np.ndarray | float
        Integral over dirac peak (Heaviside function).
    """

    integral = 0.5 * (np.sign(x) + 1)
    return integral


def gaussian_lineshape_integral(
    x: np.ndarray | float,
    hwhm: float,
) -> np.ndarray | float:
    """Calculate integral over Gaussian lineshape.

    Parameters
    ----------
    x : (M,) np.ndarray | float
        Energy variable.

    Returns
    -------
    integral : (M,) np.ndarray | float
        Integral over Gaussian lineshape.
    """

    sigma = hwhm / np.sqrt(2 * np.log(2))
    integral = 0.5 * (erf(x / (np.sqrt(2) * sigma)) + 1)
    return integral


def lorentzian_lineshape_integral(
    x: np.ndarray | float,
    hwhm: float,
) -> np.ndarray | float:
    """Calculate integral over Lorentzian lineshape.

    Parameters
    ----------
    x : (M,) np.ndarray | float
        Energy variable.

    Returns
    -------
    integral : (M,) np.ndarray | float
        Integral over Lorentzian lineshape.
    """

    gamma = hwhm
    integral = 0.5 + (1 / np.pi) * np.arctan(x / gamma)
    return integral
