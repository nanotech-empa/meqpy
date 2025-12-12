from numbers import Real


class State:
    """
    Represents a many-body state.

    Attributes
    ----------
    label : str
        Label for the state (unique).
    energy : Real
        Energy of the state, in eV.
    charge : int
        Charge of the state.
    multiplicity : int, optional
        Multiplicity of the state.
        - Integer larger zero.
        - Default is 1.
    """

    def __init__(
        self, label: str, energy: Real, charge: int, multiplicity: int = 1, **kwargs
    ):
        self.label = label
        self.energy = energy
        self.charge = charge
        self.multiplicity = multiplicity

    @property
    def label(self) -> str:
        """Label for the state (unique)."""
        return self._label

    @label.setter
    def label(self, new_label: str):
        if not isinstance(new_label, str):
            raise TypeError(f"label must be a string, but got {type(new_label)}.")
        self._label = new_label

    @property
    def energy(self) -> float:
        """Energy of the state, in eV."""
        return self._energy

    @energy.setter
    def energy(self, new_energy):
        if not isinstance(new_energy, Real):
            raise TypeError(
                f"energy must be a real number, but got {type(new_energy)}."
            )
        self._energy = float(new_energy)

    @property
    def charge(self) -> int:
        """Charge of the state."""
        return self._charge

    @charge.setter
    def charge(self, new_charge):
        if not isinstance(new_charge, int):
            raise TypeError(f"charge must be an integer, but got {type(new_charge)}.")
        self._charge = new_charge

    @property
    def multiplicity(self) -> int:
        """Multiplicity of the state."""
        return self._multiplicity

    @multiplicity.setter
    def multiplicity(self, new_multiplicity):
        if not isinstance(new_multiplicity, int):
            raise TypeError(
                f"multiplicity must be an integer, but got {type(new_multiplicity)}."
            )
        if new_multiplicity <= 0:
            raise ValueError(
                f"multiplicity must be larger than zero, but got {new_multiplicity}."
            )
        self._multiplicity = new_multiplicity

    def __repr__(self):
        attrs = f"label={self.label}, energy={self.energy}, charge={self.charge}, multiplicity={self.multiplicity}"
        return f"State({attrs})"
