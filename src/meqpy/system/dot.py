from .system import System
from ..utils.physical_constants import m_e, q_e, hbar
from numbers import Real
import numpy as np


class QuantumDot(System):
    """A class representing a quantum dot system, inheriting from System."""

    def __init__(self, eta_sample: float = 1.0, eta_tip: float = 1.0, **kwargs):
        super().__init__(**kwargs)

        self.eta_sample = eta_sample
        self.eta_tip = eta_tip

    @property
    def eta_sample(self) -> float:
        """Constant calibration factor of transition rate by tunneling to/from sample."""
        return self._eta_sample

    @eta_sample.setter
    def eta_sample(self, new_eta: float):
        self._eta_sample = super()._verify_input_nonnegative_float(
            new_eta, "eta_sample"
        )

    @property
    def eta_tip(self) -> float:
        """Constant calibration factor of transition rate by tunneling to/from tip."""
        return self._eta_tip

    @eta_tip.setter
    def eta_tip(self, new_eta: float):
        self._eta_tip = super()._verify_input_nonnegative_float(new_eta, "eta_tip")

    def tunneling_probability(
        self,
        z: float | np.ndarray,
        V: float | np.ndarray = 0.0,
        dE: float | np.ndarray = None,
    ):
        if isinstance(z, Real):
            z = np.asarray(z)
        elif not isinstance(z, np.ndarray):
            raise ValueError("z must be a float or np.ndarray, but got {type(z)}.")
        elif not z.ndim == 1:
            raise ValueError(
                f"z must be a float or 1D np.ndarray, but got array with shape {z.shape}."
            )

        if dE is None:
            dE = self.dE
        elif isinstance(dE, Real):
            dE = self.ones * float(dE)

        kappa_mat = self.kappa(V, dE)

        tunneling_prob = np.exp(-2 * kappa_mat[None, ...] * z[..., None, None, None])

        return np.squeeze(tunneling_prob)

    def kappa(self, V: float | np.ndarray, dE: float | np.ndarray) -> np.ndarray:
        """Calculate decay constant kappa for given energy difference(s) and bias voltage(s).

        Parameters
        ----------
        V : float | (M,) np.ndarray
            Bias voltage or 1d array of bias voltages, in eV
        dE : float | (N,N) np.ndarray
            Energy difference between one pair of states or 2d matrix of energy differences between all states, in eV

        Returns
        -------
        (M,N,N) np.ndarray
            Array containing kappa for each voltage and each pair of states, in 1/Angstrom
            If E or V are float, they are converted to arrays of length 1 for broadcasting.
            The returned array is squeezed to remove any dimensions of size 1.

        Notes
        -----
        The decay constant kappa is calculated based on the selected kappa_mode:
            - '10': kappa = log(10)/2.0
            - 'constant': kappa = sqrt(2*m_e*q_e*(workfunction)/hbar^2)*1e-10
            - 'full': kappa = sqrt(2*m_e*q_e*(workfunction - E + V/2)/hbar^2)*1e-10
        """
        if isinstance(V, Real):
            V = np.asarray(V)
        elif not isinstance(V, np.ndarray):
            raise ValueError(f"V must be a float or np.ndarray, but got {type(V)}.")
        elif not V.ndim == 1:
            raise ValueError(
                f"V must be a float or 1D np.ndarray, but got array with shape {V.shape}."
            )

        if isinstance(dE, Real):
            dE = np.asarray(dE)
        elif not isinstance(dE, np.ndarray):
            raise ValueError(f"E must be a float or np.ndarray, but got {type(dE)}.")
        elif not dE.ndim == 2:
            raise ValueError(
                f"E must be a float or 2D np.ndarray, but got array with shape {dE.shape}."
            )
        elif dE.shape[0] != dE.shape[1]:
            raise ValueError(
                f"E must be a square matrix, but got array with shape {dE.shape}."
            )

        kappa = np.zeros(V.shape + dE.shape)
        if self.kappa_mode == "10":
            kappa.fill(np.log(10) / 2.0)
        elif self.kappa_mode == "constant":
            kappa.fill(np.sqrt(2 * m_e * q_e * (self.workfunction) / hbar**2) * 1e-10)
        elif self.kappa_mode == "full":
            kappa = (
                np.sqrt(
                    2
                    * m_e
                    * q_e
                    * (self.workfunction - dE[None, ...] + V[..., None, None] / 2)
                    / hbar**2
                )
                * 1e-10
            )

        return np.squeeze(kappa)
