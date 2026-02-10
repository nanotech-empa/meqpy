from enum import Enum


class KappaMode(str, Enum):
    """Mode for calculating the kappa factor for tunneling decay rates"""

    FAC10 = "10"
    CONSTANT = "constant"
    FULL = "full"


class LineShape(str, Enum):
    """Line shape for transition rate derivative"""

    GAUSS = "gaussian"
    LOR = "lorentzian"
    DIRAC = "dirac"
