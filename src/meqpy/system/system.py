from .state import State
from ..utils.types import (
    LineShape,
    KappaMode,
    is_real_or_1darray,
    is_nonnegative_float,
    is_stack_of_square_matrices,
)
from ..utils.physical_constants import m_e, q_e, hbar
from typing import Optional, Sequence
from numbers import Real
import numpy as np
from scipy.special import erf


class System:
    """
    Class to define a physical system for creating a transition matrix
    """

    def __init__(
        self,
        name: Optional[str] = None,
        states: Optional[Sequence] = None,
        hwhm: float = 0.0,
        lineshape: LineShape = LineShape.GAUSS,
        workfunction: float = 5.0,
        reorg_shift: float = 0.0,
        kappa_mode: KappaMode = KappaMode.FULL,
        **kwargs,
    ):
        """Initialize System.

        Parameters
        ----------
        name : str, optional
            Name of System, by default "GenericSystem"
        states : _type_, optional
            List of States, by default None
        hwhm : float, optional
            Half width at half maximum of energy dependent transition rate in eV, by default 0.0
        lineshape : str, optional
            Lineshape of transition rate derivative: 'gaussian', 'lorentzian' or 'dirac'., by default "gaussian"
        workfunction : float, optional
            Workfunction of System in eV, by default 5.0
        reorg_shift : float, optional
            Shift of ion resonances, due to reorganization energy, in eV, by default 0.0
        kappa_mode : str, optional
            Flag to handle decay of wavefunction into vacuum:"
            - '10' : decay by a factor of 0.1 per Angstrom
            - 'constant' : decay through rectangular potential barrier with height given by workfunction
            - 'full' (default) : decay through rectangular potential barrier depending on workfunction, bias voltage and energy of states
        """
        self.name = name if name is not None else self.__class__.__name__

        self.states = list(states) if states is not None else []

        self.hwhm = hwhm
        self.lineshape = lineshape
        self.workfunction = workfunction
        self.reorg_shift = reorg_shift
        self.kappa_mode = kappa_mode

    @property
    def states(self) -> list[State]:
        """List containing states in system."""
        return self._states

    @states.setter
    def states(self, new_states: list[State]):
        if not isinstance(new_states, Sequence):
            new_states = [new_states]
        else:
            new_states = list(new_states)

        self._states = []
        for state in new_states:
            self.add_state(state)

    @property
    def hwhm(self) -> float:
        """Half width at half maximum of energy dependent transition rate in eV."""
        return self._hwhm

    @hwhm.setter
    def hwhm(self, hwhm: float):
        self._hwhm = is_nonnegative_float(hwhm, "hwhm")

    @property
    def lineshape(self) -> str:
        """Lineshape of transition rate derivative: 'gaussian', 'lorentzian' or 'dirac'."""
        return self._lineshape.value

    @lineshape.setter
    def lineshape(self, lineshape: str):
        self._lineshape = LineShape(lineshape)

    @property
    def workfunction(self) -> float:
        """Workfunction of System in eV."""
        return self._workfunction

    @workfunction.setter
    def workfunction(self, workfunction: float):
        self._workfunction = is_nonnegative_float(workfunction, "workfunction")

    @property
    def reorg_shift(self) -> float:
        """Shift of ion resonances, due to reorganization energy, in eV."""
        return self._reorg_shift

    @reorg_shift.setter
    def reorg_shift(self, reorg_shift: float):
        self._reorg_shift = is_nonnegative_float(reorg_shift, "reorg_shift")

    @property
    def kappa_mode(self) -> str:
        """Flag to handle decay of wavefunction into vacuum:
        - '10' : decay by a factor of 0.1 per Angstrom
        - 'constant' : decay through rectangular potential barrier with height given by workfunction
        - 'full' : decay through rectangular potential barrier depending on workfunction, bias voltage and energy of states
        """
        return self._kappa_mode.value

    @kappa_mode.setter
    def kappa_mode(self, kappa_mode: str):
        self._kappa_mode = KappaMode(kappa_mode)

    def add_state(self, state: State):
        """Add state to system.

        Parameters
        ----------
        state : State
            New state to be added to system.

        Raises
        ------
        TypeError
            If state is not instance of State class.
        """
        if isinstance(state, State):
            self._states.append(state)
        else:
            raise TypeError(f"state has to be State class, but got {type(state)}")

    def get_state(self, label: str | int) -> State:
        """Get State object for given label or index.

        Parameters
        ----------
        label : str | int
            Label or or index of state in System.

        Returns
        -------
        State
            State object.

        Raises
        ------
        ValueError
            If given label cannot be be found in system.
        """
        if isinstance(label, int):
            return self.states[label]
        for state in self.states:
            if state.label == label:
                return state
        raise ValueError(f"State with label {label} not found in the system.")

    def get_index(self, label: str) -> int:
        """Get index of state for given label.

        Parameters
        ----------
        label : str
            Label of state in system.

        Returns
        -------
        int
            Index of state in system.

        Raises
        ------
        ValueError
            If given label cannot be be found in system.
        """
        for i, state in enumerate(self.states):
            if state.label == label:
                return i
        raise ValueError(f"State with label {label} not found in the system.")

    @property
    def num_states(self) -> int:
        """Number of states in system"""
        return len(self.states)

    @property
    def energies(self) -> np.ndarray:
        """Energies of all states in system

        Returns
        -------
        energies : (N,) np.ndarray
            Vector containing energies of all states in system.
        """
        return np.array([state.energy for state in self.states])

    @property
    def charges(self) -> np.ndarray:
        """Charges of all states in system

        Returns
        -------
        charges : (N,) np.ndarray
            Vector containing charges of all states in system.
        """
        return np.array([state.charge for state in self.states])

    @property
    def multiplicities(self) -> np.ndarray:
        """Multiplicities of all states in system

        Returns
        -------
        spins : (N,) np.ndarray
            Vector containing multiplicities of all states in system.
        """
        return np.array([state.multiplicity for state in self.states])

    @property
    def dE(self) -> np.ndarray:
        """Energy difference matrix of states in system.

        Returns
        -------
        dE : (N,N) np.ndarray
            2d Matrix containing energy differences of all states
            with dE[f,i] = E_f - E_i
        """
        return self.energies[:, None] - self.energies[None, :]

    @property
    def dQ(self) -> np.ndarray:
        """Charge state difference matrix of states in system.

        Returns
        -------
        dQ : (N,N) np.ndarray
            2d Matrix containing charge differences of all states
            with dQ[f,i] = Q_f - Q_i
        """
        return self.charges[:, None] - self.charges[None, :]

    @property
    def zeros(self) -> np.ndarray:
        """Return square numpy array, filled with zeros.

        Returns
        -------
        (N,N) np.ndarray
            Square array of with shape (n,n)
            and n being number of states in system.
        """
        return np.zeros((self.num_states, self.num_states))

    @property
    def ones(self) -> np.ndarray:
        """Return square numpy array, filled with ones.

        Returns
        -------
        (N,N) np.ndarray
            Square array of with shape (n,n)
            and n being number of states in system.
        """
        return np.ones((self.num_states, self.num_states))

    def matrix_by_states(
        self, initial: int | str, final: int | str, symmetric: bool = False
    ) -> np.ndarray:
        """Return array for normalized transition from initial to final state.

        Parameters
        ----------
        initial : int | str
            Label or index of initial state.
        final : int | str
            Label or index of final state.
        symmetric : bool, optional
            Make array symmetric, by default False

        Returns
        -------
        matrix : (N,N) np.ndarray
            Matrix of shape (n,n) with
            - n being number of states in system
            - all entries are zero
            - matrix[final,inital] = 1
            - matrix[initial,final] = 1, if symmetric is True

        Raises
        ------
        IndexError
            If `initial` or `final` parameter are out of range.
        """
        if not isinstance(initial, int):
            initial = self.get_index(initial)
        elif initial < 0 or initial >= self.num_states:
            raise IndexError(f"Index {initial} for initial state out of range.")

        if not isinstance(final, int):
            final = self.get_index(final)
        elif final < 0 or final >= self.num_states:
            raise IndexError(f"Index {final} for final state out of range.")

        mat = self.zeros
        mat[final, initial] = 1.0
        if symmetric:
            mat[initial, final] = 1.0
        return mat

    def dirac_lineshape_integral(self, x: np.ndarray | float) -> np.ndarray | float:
        """Calculate integral over dirac peak (Heaviside function).

        Parameters
        ----------
        x : (M,) np.ndarray | float
            Energy variable.

        Returns
        -------
        integral : (M,) np.ndarray | float
            Integral over dirac peak (Heaviside function).
        """

        integral = 0.5 * (np.sign(x) + 1)
        return integral

    def gaussian_lineshape_integral(self, x: np.ndarray | float) -> np.ndarray | float:
        """Calculate integral over Gaussian lineshape.

        Parameters
        ----------
        x : (M,) np.ndarray | float
            Energy variable.

        Returns
        -------
        integral : (M,) np.ndarray | float
            Integral over Gaussian lineshape.
        """

        sigma = self.hwhm / np.sqrt(2 * np.log(2))
        integral = 0.5 * (erf(x / (np.sqrt(2) * sigma)) + 1)
        return integral

    def lorentzian_lineshape_integral(
        self, x: np.ndarray | float
    ) -> np.ndarray | float:
        """Calculate integral over Lorentzian lineshape.

        Parameters
        ----------
        x : (M,) np.ndarray | float
            Energy variable.

        Returns
        -------
        integral : (M,) np.ndarray | float
            Integral over Lorentzian lineshape.
        """

        gamma = self.hwhm
        integral = 0.5 + (1 / np.pi) * np.arctan(x / gamma)
        return integral

    def normalized_charging_transitions(
        self,
        V: float | np.ndarray,
        dE: float | np.ndarray = None,
        dQ: int | np.ndarray = None,
    ) -> np.ndarray:
        """Calculate normalized charging transition rates between all states for given bias voltage(s).

        Parameters
        ----------
        V : float | (M,) np.ndarray
            Bias voltage(s) in V.

        Returns
        -------
        W_charging : (N,N) np.ndarray | (M,N,N) np.ndarray
            Normalized charging transition rates matrix/matrices.
        """

        if type(dE) is type(None) and type(dQ) is type(None):
            dE = self.dE + self.reorg_shift
            dQ = self.dQ
        elif isinstance(dE, Real) and isinstance(dQ, int):
            dE = np.asarray(dE)
            dQ = np.asarray(dQ)
        elif (
            isinstance(dE, np.ndarray)
            and isinstance(dQ, np.ndarray)
            and dE.shape == dQ.shape
        ):
            is_stack_of_square_matrices(dE, "dE", dims=2)  # check if dE is valid input
            is_stack_of_square_matrices(dQ, "dQ", dims=2)  # check if dQ is valid input
        else:
            raise TypeError(
                "dE and dQ are not compatible with code. Need better error message."
            )

        # select lineshape of transition rate derivative
        if self.lineshape == "dirac" or self.hwhm == 0.0:
            lineshape_integral = self.dirac_lineshape_integral
        elif self.lineshape == "gaussian":
            lineshape_integral = self.gaussian_lineshape_integral
        elif self.lineshape == "lorentzian":
            lineshape_integral = self.lorentzian_lineshape_integral

        V = is_real_or_1darray(V, "V")  # make sure V is array of shape (M,)

        # offset voltages by energies of ion resonances --> shape (M,N,N)
        energy_arg = -dE[None, ...] - dQ[None, ...] * V[..., None, None]

        # voltage dependend transition probability for charging
        W_charging = lineshape_integral(energy_arg)

        # assert only charging transitions with dQ == +1 or -1 are non-zero
        W_charging *= np.abs(self.dQ) == 1

        return np.squeeze(W_charging)

    @property
    def clebsch_gordan_factors(self) -> np.ndarray:
        """Return matrix with Clebsch-Gordan factors for all state transitions in system.

        Returns
        -------
        cg_factors : (N,N) np.ndarray
            Clebsch-Gordan factors matrix.

        Notes
        -----
        The Clebsch-Gordan factor for a transition from state i to state f is calculated as:
            cg_factor(f,i) = max(mf,mi)/mf
        with mi and mf being the multiplicities of the initial and final states, respectively.
        """

        multiplicities = self.multiplicities.astype(float)

        mf = multiplicities[:, None]
        mi = multiplicities[None, :]

        return np.maximum(mf, mi) / mf

    def charging_rates(
        self,
        z: float | np.ndarray,
        V: float | np.ndarray = 0.0,
        squeeze: bool = True,
    ) -> np.ndarray:
        """Get transition rates by charging of system:
        transition rate = coupling strength * normalized charging transition * Clebsch-Gordan factors

        Parameters
        ----------
        z : float | (K,) np.ndarray
            Distance between System and lead in Angstrom.
        V : float | (M,) np.ndarray, optional
            Bias voltage(s) between system and lead in V, by default 0.0. Only used in case of kappa_mode='full'.
        squeeze : bool, optional
                The returned array is squeezed to remove any dimensions of size 1, default is `True`.

        Returns
        -------
        charging_rates : (K,M,N,N) np.ndarray
            Array containing transition rates by charging, with N being number of states in system.
            The returned array is squeezed to remove any dimensions of size 1 if `squeeze` is `True`.

        Notes
        -----
        This method is a simple wrapper for combining:
            * `self.coupling_strength`
            * `self.normalized_charging_transition`
            * `self.clebsch_gordan_factors`
        """

        charging_rates = self.coupling_strength(z, V, squeeze=False)
        charging_rates *= self.normalized_charging_transitions(V)
        charging_rates *= self.clebsch_gordan_factors

        if squeeze:
            charging_rates = np.squeeze(charging_rates)

        return charging_rates

    def coupling_strength(
        self,
        z: float | np.ndarray,
        V: float | np.ndarray = 0.0,
        dE: float | np.ndarray = None,
        squeeze: bool = True,
    ) -> np.ndarray:
        """Calculate coupling strength between System and lead (e.g. sample or tip), assuming planar wave approximation:
        - coupling strength = exp[ -2 * kappa * z].

        Parameters
        ----------
        z : float | (K,) np.ndarray
            Distance between System and lead in Angstrom.
        V : float | (M,) np.ndarray, optional
            Bias voltage(s) between system and lead in V, by default 0.0. Only used in case of kappa_mode='full'.
        dE : float | (N,N) np.ndarray, optional
            Energy difference between one pair of states or 2d matrix of energy differences between all states, in eV and by default None. Only used in case of kappa_mode='full'.
            See `self.kappa` for more information.
        squeeze : bool, optional
                The returned array is squeezed to remove any dimensions of size 1, default is `True`.

        Returns
        -------
        coupling_strength : (K,M,N,N) np.ndarray
            Array containing coupling strength.
            The returned array is squeezed to remove any dimensions of size 1 if `squeeze` is `True`.

        Notes
        -----
        The decay constant kappa is calculated based on the selected kappa_mode:
            - '10': kappa = log(10)/2.0
            - 'constant': kappa = sqrt(2*m_e*q_e*(workfunction)/hbar^2)*1e-10
            - 'full': kappa = sqrt(2*m_e*q_e*(workfunction - E + V/2)/hbar^2)*1e-10
        """

        z = is_real_or_1darray(z, "z")
        kappa_mat = self.kappa(V, dE, squeeze=False)
        coupling_strength = np.exp(-2 * kappa_mat[None, ...] * z[..., None, None, None])

        if squeeze:
            coupling_strength = np.squeeze(coupling_strength)

        return coupling_strength

    def kappa(
        self,
        V: float | np.ndarray,
        dE: float | np.ndarray = None,
        kappa_mode: str = None,
        squeeze: bool = True,
    ) -> np.ndarray:
        """Calculate decay constant kappa for given energy difference(s) and bias voltage(s).

        Parameters
        ----------
        V : float | (M,) np.ndarray
            Bias voltage or 1d array of bias voltages, in V. Only used in case of kappa_mode='full'.
        dE : float | (N,N) np.ndarray
            Energy difference between one pair of states or 2d matrix of energy differences between all states, in eV. Only used in case of kappa_mode='full'.
            If a float is given, it is used for all pairs of states. By default None.
            If None (default) is given,  the 2d matrix of energy differences is calculated from the state energies and reorganization shift:
                dE[f,i] = (energy[f] - energy[i] + reorg_shift) * (charge[i] - charge[f])
        kappa_mode: str
            Optional parameter to temporarily overwrite kappa_mode. If None (default), `self.kappa_mode` will be used.
        squeeze: bool
            The returned array is squeezed to remove any dimensions of size 1, default is `True`.

        Returns
        -------
        kappa : (M,N,N) np.ndarray
            Array containing kappa for each voltage and each pair of states, in 1/Angstrom
            If E or V are float, they are converted to arrays of length 1 for broadcasting.
            The returned array is squeezed to remove any dimensions of size 1 if `squeeze` is `True`.

        Notes
        -----
        The decay constant kappa is calculated based on the selected kappa_mode:
            - '10': kappa = log(10)/2.0
            - 'constant': kappa = sqrt(2 * m_e * q_e * (workfunction)/hbar^2)*1e-10
            - 'full': kappa = sqrt(2 * m_e * q_e * (workfunction - E + V/2)/hbar^2)*1e-10
        """

        if kappa_mode is not None:
            kappa_mode = KappaMode(kappa_mode).value
        else:
            kappa_mode = self.kappa_mode

        V = is_real_or_1darray(V, "V")

        if type(dE) is type(None):
            dE = -(self.dE + self.reorg_shift) * self.dQ
        elif isinstance(dE, Real):
            dE = np.asarray([[dE]])
        else:
            is_stack_of_square_matrices(dE, "dE", dims=2)  # check if dE is valid input

        if kappa_mode == "10":
            kappa = np.zeros(V.shape + dE.shape)
            kappa.fill(np.log(10) / 2.0)

        elif kappa_mode == "constant":
            kappa = np.zeros(V.shape + dE.shape)
            kappa.fill(np.sqrt(2 * m_e * q_e * (self.workfunction) / hbar**2) * 1e-10)

        elif kappa_mode == "full":
            barrier_height = self.workfunction - dE[None, ...] + V[..., None, None] / 2
            kappa = np.sqrt(2 * m_e * q_e / hbar**2 * barrier_height)
            kappa *= 1e-10  # convert from 1/m to 1/Angstrom

        if squeeze:
            kappa = np.squeeze(kappa)

        return kappa

    def __repr__(self):
        return f"System(name={self.name}, states={self.states})"
