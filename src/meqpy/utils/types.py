from enum import Enum
from numbers import Real
import numpy as np
from scipy.special import erf
from scipy import constants as const

# Fundamental constants
ELECTRON_MASS = const.electron_mass  # kg
ELEMENTARY_CHARGE = const.elementary_charge  # C
HBAR = const.hbar  # J·s


class ValidatedEnum(Enum):
    """Enum class that raises a ValueError with allowed values when an invalid value is provided"""

    @classmethod
    def _missing_(cls, value):
        allowed = [m.value for m in cls]
        raise ValueError(
            f"Invalid {cls.__name__} value: '{value}'. Allowed values: {allowed}"
        )


class KappaMode(str, ValidatedEnum):
    """Mode for calculating the kappa factor for tunneling decay rates"""

    FAC10 = "10"
    CONSTANT = "constant"
    FULL = "full"

    def kappa(
        self,
        bias: float | np.ndarray,
        delta: float | np.ndarray,
        workfunction: float,
    ) -> np.ndarray:
        """Return decay constant for given barrier height and dependent on mode of kappa.

        Parameters
        ----------
        bias : float | (M,) np.ndarray
            Bias voltage or 1d array of bias voltages, in V. Only used in case of kappa_mode='full'.
        delta : float | np.ndarray
            Only used for kappa_mode='full'. `delta` corresponds to the change in tunneling barrier height an electron experiences,
            when tunneling to/from the system, charging it from initial state `i` to final state `f`.
        workfunction : float
            Workfunction of System in eV

        Returns
        -------
        kappa : (M, delta.shape) np.ndarray
            Decay constant kappa for each transition, depending on kappa_mode.

        Notes
        -----
        The decay constant kappa is calculated based on the selected kappa_mode:
            - '10': kappa = log(10)/2.0
            - 'constant': kappa = sqrt(2 * ELECTRON_MASS * ELEMENTARY_CHARGE * (workfunction) / HBAR^2)*1e-10
            - 'full': kappa = sqrt(2 * ELECTRON_MASS * ELEMENTARY_CHARGE * (workfunction - delta + bias/2) / HBAR^2)*1e-10
        """
        bias = is_real_or_1darray(bias, "bias")

        if not isinstance(delta, (Real, np.ndarray)):
            raise TypeError(f"delta must be Real or np.ndarray, but got {type(delta)}")

        delta = np.asarray(delta)

        is_nonnegative_float(workfunction, "workfunction")

        barrier_height = np.ones(bias.shape + delta.shape) * workfunction

        match self.value:
            case "10":
                return self.__kappa10(barrier_height)
            case "constant":
                return self.__kappa(barrier_height)
            case "full":
                barrier_height += np.subtract.outer(bias / 2, delta)
                return self.__kappa(barrier_height)

    @staticmethod
    def __kappa(barrier_height):
        """Return decay constant for given barrier height"""

        if not (barrier_height > 0).all():
            raise ValueError("barrier_height must be positive.")

        kappa = np.sqrt(
            2 * ELECTRON_MASS * ELEMENTARY_CHARGE / HBAR**2 * barrier_height
        )
        kappa *= 1e-10
        return kappa

    @staticmethod
    def __kappa10(barrier_height):
        """Return decay constant (of shape barrier height) such that exp(2*kappa) = 10"""
        return np.ones_like(barrier_height) * np.log(10) / 2.0


class LineShape(str, ValidatedEnum):
    """Line shape for transition rate derivative"""

    GAUSS = "gaussian"
    LOR = "lorentzian"
    DIRAC = "dirac"

    def lineshape_integral(self, x: float | np.ndarray, hwhm: Real):
        """Calculate integral over lineshape.

        Parameters
        ----------
        x : (M,) np.ndarray | float
            Energy variable.
        hwhm : float
            Half width at half maximum of lineshape. If hwhm == 0, lineshape defaults to "dirac".

        Returns
        -------
        integral : (M,) np.ndarray | float
            Integral over lineshape.
        """

        if not isinstance(x, (Real, np.ndarray)):
            raise TypeError(f"x must be Real or np.ndarray, but got {type(x)}")

        is_nonnegative_float(hwhm, "hwhm")

        if hwhm == 0:
            lineshape = "dirac"
        else:
            lineshape = self.value

        match lineshape:
            case "dirac":
                return self.dirac_lineshape_integral(x)

            case "gaussian":
                return self.gaussian_lineshape_integral(x, hwhm)

            case "lorentzian":
                return self.lorentzian_lineshape_integral(x, hwhm)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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


def is_nonnegative_float(input: float, label: str) -> float:
    """Verify input is real number and not negative."""
    if not isinstance(input, Real):
        raise TypeError(f"{label} has to be non-negative float but got {type(input)}")
    if input < 0:
        raise ValueError(f"{label} has to be non-negative float but got {input}")
    return float(input)


def is_real_or_1darray(value, name: str):
    """Check if value is real or np.ndarray of dimension 1, and convert to np.ndarray if it is real.

    Parameters
    ----------
    value : Real | (M,) np.ndarray
        Value to check and convert if necessary.
    name : str
        Name of parameter to be used in error messages.

    Returns
    -------
    (M,) np.ndarray
        The value converted to a 1D numpy array.

    Raises
    ------
    TypeError
        If the value is not a real number or a numpy array.
    ValueError
        If the value is a numpy array but not 1D.
    """
    if isinstance(value, Real):
        return np.asarray([value])

    if not isinstance(value, np.ndarray):
        raise TypeError(
            f"{name} must be float or 1D np.ndarray, but got {type(value)}."
        )

    if value.ndim == 0:
        return np.asarray([value])

    if value.ndim > 1:
        raise ValueError(
            f"{name} must be float or 1D np.ndarray, "
            f"but got {value.ndim}D array with shape {value.shape}."
        )

    return value


def is_stack_of_square_matrices(array: np.ndarray, name: str, dims: int = None):
    """Raise error if array is not np.ndarray with last two dimesions of equal length.

    Parameters
    ----------
    array : (...,N,N) np.ndarray
        Input array to check
    name : str
        Name of parameter to be used in error messages.
    dims : int, optional
        Check for given number of dimensions, by default None

    Raises
    ------
    TypeError
        If array is not np.ndarray
    ValueError
        If last two dimensions do not have same length or array does not have specified number of dimensions.
    """

    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be np.ndarray, but got {type(array)}.")

    elif type(dims) is not type(None) and array.ndim != dims:
        raise ValueError(
            f"{name} must be a {dims}D np.ndarray, but got array with shape {array.shape}."
        )

    elif array.shape[-2] != array.shape[-1]:
        raise ValueError(
            f"Last two dimensions of {name} must be a square matrix, but got array with shape {array.shape}."
        )
