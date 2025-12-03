class State:
    """
    Represents a many-body state.

    Attributes
    ----------
    label : str
        Label for the state (unique).
    energy : float
        Energy of the state, in eV.
    charge : int
        Charge of the state.
    spin : float, optional
        Spin of the state.
        - Non-negative integer or half-integer (0, 0.5, 1, 1.5, ...).
        - Default is None.
    """

    def __init__(self, label, energy, charge, spin=None, **kwargs):
        # check label
        if not isinstance(label, str):
            raise ValueError(f"label must be a string, but got {type(label)}.")

        # check energy
        if not isinstance(energy, (int, float)):
            raise ValueError(f"energy must be a real number, but got {type(energy)}.")

        # check charge
        if not isinstance(charge, int):
            raise ValueError(f"charge must be an integer, but got {type(charge)}.")

        # check spin
        if spin is not None:
            if not isinstance(spin, (int, float)):
                raise ValueError(f"spin must be a real number, but got {type(spin)}.")
            if spin < 0 or (2 * spin) % 1 != 0:
                raise ValueError(
                    f"spin must be non-negative integer or half-integer, but got {spin}."
                )

        self.label = label
        self.energy = energy
        self.charge = charge
        self.spin = spin

    def __repr__(self):
        attrs = f"label={self.label}, energy={self.energy}, charge={self.charge}"
        if self.spin is not None:
            attrs += f", spin={self.spin}"
        return f"State({attrs})"
