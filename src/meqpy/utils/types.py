from enum import Enum
from numbers import Real
from typing import Sequence
import numpy as np


class ValidatedEnum(Enum):
    """Enum class that raises a ValueError with allowed values when an invalid value is provided"""

    @classmethod
    def _missing_(cls, value):
        allowed = [m.value for m in cls]
        raise ValueError(
            f"Invalid {cls.__name__} value: '{value}'. Allowed values: {allowed}"
        )


def is_nonnegative_float(input: float, label: str) -> float:
    """Verify input is real number and not negative."""
    if not isinstance(input, Real):
        raise TypeError(f"{label} has to be non-negative float but got {type(input)}")
    if input < 0:
        raise ValueError(f"{label} has to be non-negative float but got {input}")
    return float(input)


def is_real_or_1darray(value, name: str) -> np.ndarray:
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


def is_stack_of_square_matrices(array: np.ndarray, name: str, dims: int = None) -> bool:
    """Raise error if array is not np.ndarray with last two dimensions of equal length.
    Optional: check if array is of dimension dims.

    Parameters
    ----------
    array : (...,N,N) np.ndarray
        Input array to check
    name : str
        Name of parameter to be used in error messages.
    dims : int, optional
        Check for given number of dimensions, by default None

    Returns
    -------
    bool
        True if array is np.ndarray with last two dimensions of equal length and of dimension dims if given.


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

    return True


def is_sequence_of_pairs(sequence: Sequence, val_type: type, name: str) -> bool:
    """Check if input is a Sequence of pairs, each containing two elements of type val_type.

    Parameters
    ----------
    sequence : Sequence
        Input sequence to check.
    val_type : type
        Type of elements within pairs.
    name : str
        Name of sequence object, used for raising errors.

    Returns
    -------
    bool
        True if sequence is a Sequence of pairs, each containing two elements of type val_type.

    Raises
    ------
    TypeError
        If sequence is not a Sequence
    """
    if not isinstance(sequence, Sequence):
        raise TypeError(f"{name} must be Sequence, but got {type(sequence)}.")

    for pair in sequence:
        is_pair(pair, val_type, name=f"pair in {name}")

    return True


def is_pair(pair: Sequence, val_type: type, name: str) -> bool:
    """Check if pair is a Sequence of length 2, with each element being of type val_type

    Parameters
    ----------
    pair : Sequence
        Input sequence to check.
    val_type : type
        Type of elements within pairs.
    name : str
        Name of sequence object, used for raising errors.


    Returns
    -------
    bool
        True if pair is a Sequence of length 2, with each element being of type val_type

    Raises
    ------
    TypeError
        Input sequence is not of type Sequence.
    IndexError
        Input sequence is not of length 2.
    TypeError
        Elements in sequence are not of type val_type.
    """
    if not isinstance(pair, Sequence):
        raise TypeError(f"{name} must be Sequence, but got {type(pair)}.")

    if len(pair) != 2:
        raise IndexError(
            f"{name} must be Sequence of length 2, "
            f"but got object of length {len(pair)}."
        )

    for value in pair:
        if not isinstance(value, val_type):
            raise TypeError(
                f"Elements in {name} must be {val_type}, but got {type(value)}"
            )

    return True
