from enum import Enum


class ValidatedEnum(Enum):
    """Enum class that raises a ValueError with allowed values when an invalid value is provided"""

    @classmethod
    def _missing_(cls, value):
        allowed = [m.value for m in cls]
        raise ValueError(
            f"Invalid {cls.__name__} value: '{value}'. Allowed values: {allowed}"
        )


class KappaMode(str, ValidatedEnum):
    """Mode for calculating the kappa factor for tunneling decay rates"""

    FAC10 = "10"
    CONSTANT = "constant"
    FULL = "full"


class LineShape(str, ValidatedEnum):
    """Line shape for transition rate derivative"""

    GAUSS = "gaussian"
    LOR = "lorentzian"
    DIRAC = "dirac"
