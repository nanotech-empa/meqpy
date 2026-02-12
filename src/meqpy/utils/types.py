from enum import Enum
from numbers import Real
import numpy as np


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


class LineShape(str, ValidatedEnum):
    """Line shape for transition rate derivative"""

    GAUSS = "gaussian"
    LOR = "lorentzian"
    DIRAC = "dirac"


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
        value = np.asarray(value)

    elif not isinstance(value, np.ndarray):
        raise TypeError(
            f"{name} must be float or 1D np.ndarray, but got {type(value)}."
        )

    elif not value.ndim == 1:
        raise ValueError(
            f"{name} must be float or 1D np.ndarray, but got array with shape {value.shape}."
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
