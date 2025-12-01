import numpy as np
from scipy.linalg import null_space
import warnings


class MasterEquation:
    """
    Constructs and solves the Master Equation:
        dP/dt = W P
    """

    def __init__(self, system):
        self.system = system
        self.n = len(system.states)
        self.W = np.zeros((self.n, self.n))

    def solve_eq(self):
        """
        Solves the Master Equation at equilibrium, dP/dt = 0
        Solves W Peq = 0 for Peq, returning a normalized vector Peq
        """

        # check that W is square matrix
        if not isinstance(self.W, np.ndarray) or len(self.W.shape) != 2:
            raise ValueError(
                f"W must be a 2D NumPy array, but got {type(self.W)} with"
                f" shape {getattr(self.W, 'shape', None)}."
            )
        if self.W.shape[0] != self.W.shape[1]:
            raise ValueError(f"Matrix W must be square, but has shape {self.W.shape}.")

        # solve W Peq = 0, with scipy.null_space
        Peq = null_space(self.W)

        # check solutions
        nsol = Peq.shape[1]  # nr of solutions
        if nsol == 0:
            raise ValueError("No null-space vector found.")
        if nsol > 1:
            warnings.warn(
                f"Multiple null-space vectors found ({nsol}).Proceeding with the first."
            )

        Peq = Peq[:, 0]

        # normalize
        Peq /= np.sum(Peq)

        # check positivity
        tol = 1e-12
        if np.any(Peq < -tol):
            raise ValueError(
                f"Some entries of the null-space vector are negative"
                f" beyond numerical tolerance {tol}."
            )

        return Peq
