class State:
    """
    Class to represent a many-body state in the system
    """

    def __init__(self, label, energy, charge=None, **kwargs):
        self.label = label
        self.energy = energy
        self.charge = charge
    
    def __repr__(self):
        return f"State(label={self.label}, energy={self.energy}, charge={self.charge})"