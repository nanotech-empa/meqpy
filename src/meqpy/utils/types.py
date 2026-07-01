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


def _make_bounded_number_validator(caster, type_check, *, minimum=None):
    def validator(value, name="value"):
        if not isinstance(value, type_check):
            raise TypeError(
                f"{name} must be {type_check.__name__}, got {type(value).__name__}"
            )
        value = caster(value)
        if minimum is not None and value < minimum:
            raise ValueError(f"{name} must be >= {minimum}, got {value}")
        return value

    return validator


validate_nonnegative_float = _make_bounded_number_validator(float, Real, minimum=0)
validate_nonnegative_int = _make_bounded_number_validator(int, int, minimum=0)


def require_type(value: object, expected: type, name="value"):
    if not isinstance(value, expected):
        types = expected if isinstance(expected, tuple) else (expected,)
        want = " or ".join(t.__name__ for t in types)
        raise TypeError(f"{name} must be {want}, but got {type(value).__name__}")
    return value


def validate_real_or_1darray(value, name: str) -> np.ndarray:
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

    msg = f"{name} must be float or 1D np.ndarray"
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{msg}, but got {type(value).__name__}.")

    if value.ndim == 0:
        return np.asarray([value])

    if value.ndim > 1:
        raise ValueError(
            f"{msg}, but got {value.ndim}D array with shape {value.shape}."
        )

    return value


def validate_stack_of_square_matrices(array: np.ndarray, name: str) -> bool:
    """Raise error if array is not np.ndarray with last two dimensions of equal length.

    Parameters
    ----------
    array : (...,N,N) np.ndarray
        Input array to check
    name : str
        Name of parameter to be used in error messages.

    Returns
    -------
    bool
        True if array is np.ndarray with last two dimensions of equal length.


    Raises
    ------
    TypeError
        If array is not np.ndarray
    """

    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be np.ndarray, but got {type(array).__name__}.")

    elif array.shape[-2] != array.shape[-1]:
        raise ValueError(
            f"Last two dimensions of {name} must be a square matrix, but got array with shape {array.shape}."
        )

    return True


def validate_non_negative_offdiagonal(array: np.ndarray, name: str) -> bool:
    """Raise error if off-diagonal elements are negative.
    Will also call validate_stack_of_square_matrices to ensure
    a diagonal is well defined.

    Parameters
    ----------
    array : (...,N,N) np.ndarray
        Input array to check
    name : str
        Name of parameter to be used in error messages.

    Returns
    -------
    bool
        True if array is np.ndarray with no off-diagonal entries being negative,
        last two dimensions are of equal length.

    Raises
    ------
    TypeError
        If array is not np.ndarray
    ValueError
        Any Off-diagonal elements are negative.
        If last two dimensions do not have same length.
    """

    validate_stack_of_square_matrices(array, name)

    antidiag = np.ones_like(array) - np.eye(array.shape[-1])

    if np.any(antidiag * array < 0):
        raise ValueError(f"Some off-diagonal elements in {name} are negative.")

    return True


def validate_sequence_of_pairs(sequence: Sequence, val_type: type, name: str) -> bool:
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
        raise TypeError(f"{name} must be Sequence, but got {type(sequence).__name__}.")

    for pair in sequence:
        validate_pair(pair, val_type, name=f"pair in {name}")

    return True


def validate_pair(pair: Sequence, val_type: type, name: str) -> bool:
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
        raise TypeError(f"{name} must be Sequence, but got {type(pair).__name__}.")

    if len(pair) != 2:
        raise IndexError(
            f"{name} must be Sequence of length 2, "
            f"but got object of length {len(pair)}."
        )

    for value in pair:
        if not isinstance(value, val_type):
            raise TypeError(
                f"Elements in {name} must be {val_type}, but got {type(value).__name__}"
            )

    return True
