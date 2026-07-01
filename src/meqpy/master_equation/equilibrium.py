import numpy as np
from scipy.linalg import null_space
import warnings
from ..utils.types import (
    validate_nonnegative_float,
    validate_non_negative_offdiagonal,
    validate_nonnegative_int,
)


def solve_equilibrium(
    W: np.ndarray, tol=1e-12, anchor: int = 0, fuzz: float = 0.0, use_2d: bool = False
) -> np.ndarray:
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
    anchor: int, optional
        Reference state with intensity before normalizing occupation vector Peq, default 0.
    fuzz : float, optional
        Add a rate to all transitions, relative to maximum rate in W, default 0.

    Returns
    -------
    Peq : (..., N) np.ndarray
        Equilibrium occupation probability vector.
        - Sum normalized to 1.
        - All entries ≥ 0 within 'tol'.

    Notes
    -----
    Diagonal of last two axes of W is filled,
        such that each column sums to zero before solving.
    """

    if use_2d:
        return solve_equilibrium_2d(W, tol=tol)
    else:
        return solve_equilibrium_nd(W, tol=tol, anchor=anchor, fuzz=fuzz)


def solve_equilibrium_2d(W, tol=1e-12):
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
    validate_nonnegative_float(tol, "tol")

    # fill diagonal
    W = fill_diagonal(W)

    # check W
    if W.ndim != 2:
        raise ValueError(
            f"W must be a 2D np.ndarray, but got array with shape {W.shape}."
        )

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


def solve_equilibrium_nd(
    W: np.ndarray, tol=1e-12, anchor: int = 0, fuzz: float = 0.0
) -> np.ndarray:
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
    anchor: int, optional
        Reference state with intensity before normalizing occupation vector Peq, default 0.
    fuzz : float, optional
        Add a rate to all transitions, relative to maximum rate in W, default 0.

    Returns
    -------
    Peq : (..., N) np.ndarray
        Equilibrium occupation probability vector.
        - Sum normalized to 1.
        - All entries ≥ 0 within 'tol'.

    Notes
    -----
    Diagonal of last two axes of W is filled,
        such that each column sums to zero before solving.
    """
    # check tol
    validate_nonnegative_float(tol, "tol")
    validate_nonnegative_int(anchor, "anchor")
    validate_nonnegative_float(fuzz, "fuzz")
    validate_non_negative_offdiagonal(W, "W")

    if anchor >= W.shape[-1]:
        raise ValueError(
            f"anchor must be smaller than last dimension of W, but got {anchor}."
        )

    if fuzz > 0:
        W = W + fuzz * np.max(W)

    # fill diagonal
    W = fill_diagonal(W)

    # set occupance of anchor state to 1,
    # then solve the rest accordingly
    subW = np.delete(W, anchor, axis=-1)
    subW = np.delete(subW, anchor, axis=-2)

    b = -np.delete(W[..., anchor], anchor, axis=-1)

    try:
        Peq = np.linalg.solve(subW, b[..., None])[..., 0]
    except np.linalg.LinAlgError as err:
        if "Singular matrix" in str(err):
            raise ValueError(
                "Rate Matrix W is ill defined, no single ground-state found. "
                "Try using a different ``anchor`` or ``fuzz``."
            )
        else:
            raise

    # insert first state again and normalize to 1
    Peq = np.insert(Peq, anchor, 1.0, axis=-1)
    Peq /= np.sum(Peq, axis=-1)[..., None]

    # check positivity
    if np.any(Peq < -tol):
        raise ValueError(
            f"Some entries of Peq are negative beyond numerical tolerance {tol}."
        )

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

    # check W
    validate_non_negative_offdiagonal(W, "W")

    sum_over_cols = np.sum(W, axis=-2)
    diag_indices = np.arange(W.shape[-1])
    W[..., diag_indices, diag_indices] -= sum_over_cols
    return W
