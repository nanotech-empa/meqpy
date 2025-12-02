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
        Numerical tolerance for checking positivity of 'Peq'.
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
    """
    # check W
    if not isinstance(W, np.ndarray) or W.ndim != 2:
        raise ValueError(
            f"W must be a 2D NumPy array, but got {type(W)} with"
            f" shape {getattr(W, 'shape', None)}."
        )
    if W.shape[0] != W.shape[1]:
        raise ValueError(f"Matrix W must be square, but has shape {W.shape}.")

    # check tol
    if not isinstance(tol, (int, float)) or tol < 0:
        raise ValueError(f"tol must be a non-negative number, but got {tol}.")

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
