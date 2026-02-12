from .system import System
from ..utils.types import (
    KappaMode,
    is_real_or_1darray,
    is_stack_of_square_matrices,
)
from ..utils.physical_constants import m_e, q_e, hbar
from numbers import Real
import numpy as np


class QuantumDot(System):
    """A class representing a quantum dot system, inheriting from System."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def charging_rates(
        self,
        z: float | np.ndarray,
        V: float | np.ndarray = 0.0,
        dE: float | np.ndarray = None,
    ) -> np.ndarray:
        tunneling_prob = self.coupling_strength(z, V, dE)
        tunneling_prob *= self.normalized_charging_transitions(V)
        tunneling_prob *= self.clebsch_gordan_factors

        return np.squeeze(tunneling_prob)

    def coupling_strength(
        self,
        z: float | np.ndarray,
        V: float | np.ndarray = 0.0,
        dE: float | np.ndarray = None,
    ) -> np.ndarray:
        z = is_real_or_1darray(z, "z")
        kappa_mat = self.kappa(V, dE)
        return np.exp(-2 * kappa_mat[None, ...] * z[..., None, None, None])

    def kappa(
        self,
        V: float | np.ndarray,
        dE: float | np.ndarray = None,
        kappa_mode: str = None,
    ) -> np.ndarray:
        """Calculate decay constant kappa for given energy difference(s) and bias voltage(s).

        Parameters
        ----------
        V : float | (M,) np.ndarray
            Bias voltage or 1d array of bias voltages, in eV. Only used in case of kappa_mode='full'.
        dE : float | (N,N) np.ndarray
            Energy difference between one pair of states or 2d matrix of energy differences between all states, in eV. Only used in case of kappa_mode='full'.
            If a float is given, it is used for all pairs of states.
            If None (default) is given,  the 2d matrix of energy differences is calculated from the state energies and reorganization shift:
                dE[f,i] = (energy[f] - energy[i] + reorg_shift) * (charge[i] - charge[f])

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

        if kappa_mode is not None:
            kappa_mode = KappaMode(kappa_mode).value
        else:
            kappa_mode = self.kappa_mode

        V = is_real_or_1darray(V, "V")

        if type(dE) is type(None):
            dE = -(self.dE + self.reorg_shift) * self.dQ
        elif isinstance(dE, Real):
            dE = np.asarray(dE)
        else:
            is_stack_of_square_matrices(dE, "dE", dims=2)  # check if dE is valid input

        kappa = np.zeros(V.shape + dE.shape)
        if kappa_mode == "10":
            kappa.fill(np.log(10) / 2.0)

        elif kappa_mode == "constant":
            kappa.fill(np.sqrt(2 * m_e * q_e * (self.workfunction) / hbar**2) * 1e-10)

        elif kappa_mode == "full":
            barrier_height = self.workfunction - dE[None, ...] + V[..., None, None] / 2
            kappa = np.sqrt(2 * m_e * q_e / hbar**2 * barrier_height)
            kappa *= 1e-10  # convert from 1/m to 1/Angstrom

        return np.squeeze(kappa)
