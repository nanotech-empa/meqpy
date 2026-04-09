import numpy as np
from numbers import Real
from .types import is_real_or_1darray


def pad_lin_extrapolate(array: np.ndarray, pad: int) -> np.ndarray:
    array = is_real_or_1darray(array, "array")

    if not isinstance(pad, int):
        raise ValueError(f"pad must be a non-negative integer, got {type(pad)}.")

    if pad < 0:
        raise ValueError(f"pad must be a non-negative integer, got {pad}.")

    return np.pad(array, pad, _pad_lin_extrapolate_func)


def _pad_lin_extrapolate_func(vector: np.ndarray, pad_width: tuple, iaxis: int, kwargs):
    """Function for `np.pad()` to extrapolate linearly (see numpy docs).
    EXAMPLE: `x_padded = np.pad(x, pad_width, pad_lin_extrapolate)`
    """
    dd = vector[pad_width[0] + 1] - vector[pad_width[0]]
    vector[: pad_width[0]] = (
        vector[pad_width[0]] - (pad_width[0] - np.arange(pad_width[0])) * dd
    )
    vector[-pad_width[1] :] = (
        vector[-pad_width[1] - 1] + (np.arange(pad_width[1]) + 1) * dd
    )


def value_to_index(value: float, array: np.ndarray) -> int:
    """Check if value is in range of array and return index closest to value.

    Parameters
    ----------
    value : Real
        Input value to check.
    array : np.ndarray
        Array to compare value with.

    Returns
    -------
    index : int
        Index of point in array closest to value, if in range of array.

    Raises
    ------
    ValueError
        Value is not in within range of array.
    TypeError
        Value is not type Real, or array is not 1 dimensional np.ndarray.
    """

    if not isinstance(value, Real):
        TypeError(f"value must be of type Real, but got {type(value)}.")

    if not isinstance(array, np.ndarray):
        TypeError(f"array must be np.ndarray, but got {type(array)}")

    if array.ndim != 1:
        TypeError(
            f"array must be 1 dimensional np.ndarray, "
            f"but got array of dimension {array.ndim}."
        )

    if value > np.max(array) or value < np.min(array):
        raise ValueError(f"{value} is not within array.")

    return int(np.argmin(np.abs(array - value)))
