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
    spin : Real or None, optional
        Spin of the state.
        - Non-negative integer or half-integer (0, 0.5, 1, 1.5, ...), or None if unspecified.
        - Default is None.
    """

    def __init__(
        self, label: str, energy: Real, charge: int, spin: Real | None = None, **kwargs
    ):
        self._validate_label(label)
        self._validate_energy(energy)
        self._validate_charge(charge)
        self._validate_spin(spin)

        self.label = label
        self.energy = float(energy)
        self.charge = charge
        self.spin = float(spin) if spin is not None else None

    @staticmethod
    def _validate_label(label):
        if not isinstance(label, str):
            raise TypeError(f"label must be a string, but got {type(label)}.")

    @staticmethod
    def _validate_energy(energy):
        if not isinstance(energy, Real):
            raise TypeError(f"energy must be a real number, but got {type(energy)}.")

    @staticmethod
    def _validate_charge(charge):
        if not isinstance(charge, int):
            raise TypeError(f"charge must be an integer, but got {type(charge)}.")

    @staticmethod
    def _validate_spin(spin):
        if spin is not None:
            if not isinstance(spin, Real):
                raise TypeError(f"spin must be a real number, but got {type(spin)}.")
            if spin < 0 or (2 * spin) % 1 != 0:
                raise ValueError(
                    f"spin must be a non-negative integer or half-integer, but got {spin}."
                )

    def __repr__(self):
        attrs = f"label={self.label}, energy={self.energy}, charge={self.charge}"
        if self.spin is not None:
            attrs += f", spin={self.spin}"
        return f"State({attrs})"
