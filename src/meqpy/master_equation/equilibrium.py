import numpy as np
from scipy.linalg import null_space
import warnings


def solve_equilibrium(W, tol=1e-12):
    """
    Solve the master equation at equilibrium: W @ Peq = 0.

    Parameters
    ----------
    W : (N, N) np.ndarray
        Master equation matrix.
    tol : float, optional
        Numerical tolerance for checking positivity of 'Peq' and property of 'W'.
        - Non-negative.
        - Default is 1e-12.

    Returns
    -------
    Peq : (N,) np.ndarray
        Equilibrium occupation probability vector.
        - Sum normalized to 1.
        - All entries ≥ 0 within 'tol'.

    Notes
    -----
    If W @ Peq = 0 has multiple linearly independent solutions,
        the first one is used and a warning is issued.

    Diagonal of W is filled such that each column sums to zero before solving.
    """
    # check tol
    if not isinstance(tol, (int, float)) or tol < 0:
        raise ValueError(f"tol must be a non-negative number, but got {tol}.")

    # check W
    if not isinstance(W, np.ndarray) or W.ndim != 2:
        raise ValueError(
            f"W must be a 2D NumPy array, but got {type(W)} with"
            f" shape {getattr(W, 'shape', None)}."
        )
    if W.shape[0] != W.shape[1]:
        raise ValueError(f"Matrix W must be square, but has shape {W.shape}.")

    # fill diagonal
    W = fill_diagonal(W)

    # solve W @ Peq = 0, with scipy.null_space
    Peq = null_space(W)

    # check solution
    nsol = Peq.shape[1]  # nr of solutions
    if nsol == 0:
        raise ValueError("No solutions found.")
    if nsol > 1:
        warnings.warn(f"Multiple solutions found ({nsol}). Proceeding with the first.")

    # use first solution
    Peq = Peq[:, 0]

    # normalize
    Peq /= np.sum(Peq)

    # check positivity
    if np.any(Peq < -tol):
        raise ValueError(
            f"Some entries of Peq are negative beyond numerical tolerance {tol}."
        )

    return Peq


def solve_equilibrium_nd(W: np.ndarray, tol=1e-12) -> np.ndarray:
    """
    Solve the master equation at equilibrium: W @ Peq = 0 for N-dimensional arrays
    where the last two dimensions are the square matrix.

    Parameters
    ----------
    W : (..., N, N) np.ndarray
        Array with last two dimensions containing Master equation matrices.
    tol : float, optional
        Numerical tolerance for checking positivity of 'Peq' and property of 'W'.
        - Non-negative.
        - Default is 1e-12.

    Returns
    -------
    Peq : (..., N) np.ndarray
        Equilibrium occupation probability vector.
        - Sum normalized to 1.
        - All entries ≥ 0 within 'tol'.

    Notes
    -----
    If W @ Peq = 0 has multiple linearly independent solutions,
        the first one is used and a warning is issued.
    Diagonal of last two axes of W is filled,
        such that each column sums to zero before solving.
    """
    # check W
    if not isinstance(W, np.ndarray) or W.ndim < 2:
        raise ValueError(
            f"W must be at least a 2D NumPy array, but got {type(W)} with"
            f" shape {getattr(W, 'shape', None)}."
        )
    if W.shape[-2] != W.shape[-1]:
        raise ValueError(
            f"Matrix W must be square in the last two dimensions, but has shape {W.shape}."
        )

    # prepare output
    Peq_shape = W.shape[:-1]
    Peq = np.zeros(Peq_shape)

    # iterate over all indices except the last two
    it = np.nditer(Peq[..., 0], flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        # get current index
        idx = it.multi_index

        # extract current W matrix
        Wi = W[idx]

        # solve equilibrium for current W
        Peq[idx] = solve_equilibrium(Wi, tol=tol)

        it.iternext()

    return Peq


def fill_diagonal(W: np.ndarray) -> np.ndarray:
    """
    Fill the diagonal of the master equation matrix W such that each column sums to zero.
    Supports N-dimensional arrays where the last two dimensions are the square matrix.

    Parameters
    ----------
    W : (..., N, N) np.ndarray
        Master equation matrix with off-diagonal elements representing transition rates.

    Returns
    -------
    W : (..., N, N) np.ndarray
        Master equation matrix with filled diagonal.
    """
    sum_over_cols = np.sum(W, axis=-2)
    diag_indices = np.arange(W.shape[-1])
    W[..., diag_indices, diag_indices] -= sum_over_cols
    return W
