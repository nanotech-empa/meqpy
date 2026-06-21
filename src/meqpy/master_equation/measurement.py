import numpy as np
from ..utils.types import validate_stack_of_square_matrices, require_type


def measurement(M: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Simulate measurement of system in given occupational state(s) P.

    Parameters
    ----------
    M : (..., N, N) np.ndarray
        Measurement operator.
    P : (..., N) np.ndarray
        Population vector(s) of system.

    Returns
    -------
    (...,) np.ndarray
        Measurement result.

    Raises
    ------
    TypeError
        If P is not np.ndarray.
    ValueError
        If last dimension of P does not match last dimensions of M.
    """
    validate_stack_of_square_matrices(M, "M")
    require_type(P, np.ndarray, "P")

    if P.shape[-1] != M.shape[-1]:
        raise ValueError(
            f"Last dimension of M must match length of P, but got M with shape {M.shape} and P with shape {P.shape}."
        )

    return np.einsum("...ij,...j->...", M, P)
