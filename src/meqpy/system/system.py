from .state import State
from numbers import Real
from collections.abc import Iterable
import numpy as np


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
        return self._states

    @states.setter
    def states(self, new_states):
        if not isinstance(new_states, Iterable):
            new_states = [new_states]

        self._states = []
        for state in new_states:
            self.add_state(state)

    @property
    def hwhm(self) -> float:
        return self._hwhm

    @hwhm.setter
    def hwhm(self, new_hwhm: float):
        self._hwhm = self._verify_input_nonnegative_float(new_hwhm, "hwhm")

    @property
    def lineshape(self) -> str:
        return self._lineshape

    @lineshape.setter
    def lineshape(self, new_lineshape: str):
        allowed_lineshapes = ["gaussian", "lorentzian"]
        self._lineshape = self._verify_input_allowed_str(
            new_lineshape, allowed_lineshapes, "lineshape"
        )

    @property
    def workfunction(self) -> float:
        return self._workfunction

    @workfunction.setter
    def workfunction(self, new_workfunction: float):
        self._workfunction = self._verify_input_nonnegative_float(
            new_workfunction, "workfunction"
        )

    @property
    def reorg_shift(self) -> float:
        return self._reorg_shift

    @reorg_shift.setter
    def reorg_shift(self, new_reorg_shift: float):
        self._reorg_shift = self._verify_input_nonnegative_float(
            new_reorg_shift, "reorg_shift"
        )

    @property
    def eta(self) -> float:
        return self._eta

    @eta.setter
    def eta(self, new_eta: float):
        self._eta = self._verify_input_nonnegative_float(new_eta, "eta")

    @property
    def kappa_mode(self) -> str:
        return self._kappa_mode

    @kappa_mode.setter
    def kappa_mode(self, new_kappa_mode: str):
        allowed_kappa_modes = ["10", "constant", "full"]
        self._kappa_mode = self._verify_input_allowed_str(
            new_kappa_mode, allowed_kappa_modes, "kappa_mode"
        )

    @staticmethod
    def _verify_input_nonnegative_float(input: float, label: str) -> float:
        if not isinstance(input, Real):
            raise TypeError(f"{label} has to be non-negative float but got {input}")
        if input < 0:
            raise ValueError(f"{label} has to be non-negative float but got {input}")
        return float(input)

    @staticmethod
    def _verify_input_allowed_str(
        input: str, allowed_str: list[str], label: str
    ) -> str:
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
        if isinstance(state, State):
            self._states.append(state)
        else:
            raise TypeError(f"state has to be State class, but got {state}")

    def get_state(self, label: str | int) -> State:
        if isinstance(label, int):
            return self.states[label]
        for state in self.states:
            if state.label == label:
                return state
        raise ValueError(f"State with label {label} not found in the system.")

    def get_index(self, label: str) -> int:
        for i, state in enumerate(self.states):
            if state.label == label:
                return i
        raise ValueError(f"State with label {label} not found in the system.")

    @property
    def n(self) -> int:
        return len(self.states)

    @property
    def energies(self) -> np.ndarray:
        return np.array([state.energy for state in self.states])

    @property
    def charges(self) -> np.ndarray:
        return np.array([state.charge for state in self.states])

    @property
    def dE(self) -> np.ndarray:
        return self.energies[:, None] - self.energies[None, :]

    @property
    def dQ(self) -> np.ndarray:
        return self.charges[:, None] - self.charges[None, :]

    @property
    def zeros(self) -> np.ndarray:
        return np.zeros((self.n, self.n))

    @property
    def ones(self) -> np.ndarray:
        return np.ones((self.n, self.n))

    def matrix_by_states(
        self, initial: int | str, final: int | str, symm: bool = True
    ) -> np.ndarray:
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
        if symm:
            mat[initial, final] = 1.0
        return mat

    def __repr__(self):
        return f"System(name={self.name}, states={self.states})"
