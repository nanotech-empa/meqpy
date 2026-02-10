from .state import State
from numbers import Real
from collections.abc import Iterable
import numpy as np
from scipy.special import erf


class System:
    """
    Class to define a physical system for creating a hopping matrix
    """

    def __init__(
        self,
        name="GenericSystem",
        states=None,
        hwhm: float = 0.0,
        lineshape: str = "gaussian",
        workfunction: float = 5.0,
        reorg_shift: float = 0.0,
        eta: float = 1.0,
        kappa_mode: str = "full",
        **kwargs,
    ):
        self.name = name

        self._states = []
        self.states = states if states is not None else []

        self.hwhm = hwhm
        self.lineshape = lineshape
        self.workfunction = workfunction
        self.reorg_shift = reorg_shift
        self.eta = eta
        self.kappa_mode = kappa_mode

    @property
    def states(self) -> list[State]:
        """List containing states in system."""
        return self._states

    @states.setter
    def states(self, new_states: list[State]):
        if not isinstance(new_states, Iterable):
            new_states = [new_states]

        self._states = []
        for state in new_states:
            self.add_state(state)

    @property
    def hwhm(self) -> float:
        """Half width at half maximum of energy dependent transition rate in eV."""
        return self._hwhm

    @hwhm.setter
    def hwhm(self, new_hwhm: float):
        self._hwhm = self._verify_input_nonnegative_float(new_hwhm, "hwhm")

    @property
    def lineshape(self) -> str:
        """Lineshape of transition rate derivative: 'gaussian', 'lorentzian' or 'stepwise'."""
        return self._lineshape

    @lineshape.setter
    def lineshape(self, new_lineshape: str):
        allowed_lineshapes = ["gaussian", "lorentzian", "stepwise"]
        self._lineshape = self._verify_input_allowed_str(
            new_lineshape, allowed_lineshapes, "lineshape"
        )

    @property
    def workfunction(self) -> float:
        """Workfunction of System in eV."""
        return self._workfunction

    @workfunction.setter
    def workfunction(self, new_workfunction: float):
        self._workfunction = self._verify_input_nonnegative_float(
            new_workfunction, "workfunction"
        )

    @property
    def reorg_shift(self) -> float:
        """Shift of ion resonances, due to reorganization energy, in eV."""
        return self._reorg_shift

    @reorg_shift.setter
    def reorg_shift(self, new_reorg_shift: float):
        self._reorg_shift = self._verify_input_nonnegative_float(
            new_reorg_shift, "reorg_shift"
        )

    @property
    def eta(self) -> float:
        """Constant calibration factor of tunneling current."""
        return self._eta

    @eta.setter
    def eta(self, new_eta: float):
        self._eta = self._verify_input_nonnegative_float(new_eta, "eta")

    @property
    def kappa_mode(self) -> str:
        """Flag to handle decay of wavefunction into vacuum:
        - '10' : decay by a factor of 0.1 per Angstrom
        - 'constant' : decay through rectangular potential barrier with height given by workfunction
        - 'full' : decay through rectangular potential barrier depending on workfunction, bias voltage and energy of states
        """
        return self._kappa_mode

    @kappa_mode.setter
    def kappa_mode(self, new_kappa_mode: str):
        allowed_kappa_modes = ["10", "constant", "full"]
        self._kappa_mode = self._verify_input_allowed_str(
            new_kappa_mode, allowed_kappa_modes, "kappa_mode"
        )

    @staticmethod
    def _verify_input_nonnegative_float(input: float, label: str) -> float:
        """Verify input is real number and not negative."""
        if not isinstance(input, Real):
            raise TypeError(f"{label} has to be non-negative float but got {input}")
        if input < 0:
            raise ValueError(f"{label} has to be non-negative float but got {input}")
        return float(input)

    @staticmethod
    def _verify_input_allowed_str(
        input: str, allowed_str: list[str], label: str
    ) -> str:
        """Verifies input is a string in allowed_str list."""
        if not isinstance(input, str):
            raise TypeError(f"{label} has to be string but got {input}")

        input = input.lower()
        if input in allowed_str:
            return input
        raise ValueError(
            f"{label} has to be one of the following strings: "
            f"{', '.join(allowed_str)}; "
            f"but got {input}"
        )

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
            raise TypeError(f"state has to be State class, but got {state}")

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
    def n(self) -> int:
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
        return np.zeros((self.n, self.n))

    @property
    def ones(self) -> np.ndarray:
        """Return square numpy array, filled with ones.

        Returns
        -------
        (N,N) np.ndarray
            Square array of with shape (n,n)
            and n being number of states in system.
        """
        return np.ones((self.n, self.n))

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
        elif initial < 0 or initial >= self.n:
            raise IndexError(f"Index {initial} for initial state out of range.")

        if not isinstance(final, int):
            final = self.get_index(final)
        elif final < 0 or final >= self.n:
            raise IndexError(f"Index {final} for final state out of range.")

        mat = np.zeros((self.n, self.n))
        mat[final, initial] = 1.0
        if symmetric:
            mat[initial, final] = 1.0
        return mat

    def stepwise_lineshape_integral(self, x: np.ndarray | float) -> np.ndarray | float:
        """Calculate integral over stepwise lineshape.

        Parameters
        ----------
        x : (N,N) np.ndarray | float
            Energy variable.

        Returns
        -------
        integral : (N,N) np.ndarray | float
            Integral over stepwise lineshape.
        """

        integral = 0.5 * (np.sign(x) + 1)
        return integral

    def gaussian_lineshape_integral(self, x: np.ndarray | float) -> np.ndarray | float:
        """Calculate integral over Gaussian lineshape.

        Parameters
        ----------
        x : (N,N) np.ndarray | float
            Energy variable.

        Returns
        -------
        integral : (N,N) np.ndarray | float
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
        x : (N,N) np.ndarray | float
            Energy variable.

        Returns
        -------
        integral : (N,N) np.ndarray | float
            Integral over Lorentzian lineshape.
        """

        gamma = self.hwhm
        integral = 0.5 + (1 / np.pi) * np.arctan(x / gamma)
        return integral

    def charging_transitions_normalized(self, V: float | np.ndarray) -> np.ndarray:
        """Calculate normalized charging transition rates matrix.

        Parameters
        ----------
        V : float | (M,) np.ndarray
            Bias voltage(s) in eV.

        Returns
        -------
        W_charging : (N,N) np.ndarray | (M,N,N) np.ndarray
            Normalized charging transition rates matrix/matrices.
        """

        dE = self.dE + self.reorg_shift
        dQ = self.dQ

        if self.lineshape == "stepwise" or self.hwhm == 0.0:
            lineshape_integral = self.stepwise_lineshape_integral
        elif self.lineshape == "gaussian":
            lineshape_integral = self.gaussian_lineshape_integral
        elif self.lineshape == "lorentzian":
            lineshape_integral = self.lorentzian_lineshape_integral

        if isinstance(V, np.ndarray):
            W_charging = np.zeros((V.size, self.n, self.n))
            for i, V_i in enumerate(V):
                energy_arg = -dE - dQ * V_i
                W_charging[i] = lineshape_integral(energy_arg)
        else:
            energy_arg = -dE - dQ * V
            W_charging = lineshape_integral(energy_arg)

        W_charging *= np.abs(self.dQ) == 1

        return W_charging

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

    def __repr__(self):
        return f"System(name={self.name}, states={self.states})"
