import numpy as np


class MasterEquation:
    """
    Constructs and solves the Master Equation:
        dP/dt = M P
    """

    def __init__(self, system):
        self.system = system
        self.n = len(system.states)
        self.M = np.zeros((self.n, self.n))
