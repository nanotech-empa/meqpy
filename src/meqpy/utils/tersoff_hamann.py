import numpy as np
from numbers import Real

from .types import validate_nonnegative_float, require_type

import scipy.constants as const

BOHR2ANG = const.physical_constants["Bohr radius"][0] * 1e10  # Angstrom
ELEMENTARY_CHARGE = const.elementary_charge  # Ampere seconds


def ldos_to_rate(tip_radius: float, kappa: np.ndarray) -> np.ndarray:
    """Renormalization factor approximation to get hopping rate from LDOS for s-wave tip,
    according to Tersoff-Hamann equation (10) in https://doi.org/10.1103/PhysRevB.31.805

    Parameters
    ----------
    tip_radius: float | np.ndarray
        radius of s-wave tip in Angstrom.
    kappa : float | np.ndarray
        decay constant in 1/Angstrom.

    Returns
    -------
    float | np.ndarray
        Renomalization factor for coupling strength.
    """

    validate_nonnegative_float(tip_radius, "tip_radius")

    require_type(kappa, (Real, np.ndarray), "kappa")
    kappa = np.asarray(kappa)

    return (
        0.1
        * (tip_radius / BOHR2ANG) ** 2
        * np.exp(2 * kappa * tip_radius)
        / ELEMENTARY_CHARGE
    )
