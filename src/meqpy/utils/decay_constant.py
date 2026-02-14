from numbers import Real
import numpy as np
from scipy import constants as const
from .types import ValidatedEnum, is_real_or_1darray, is_nonnegative_float


# Fundamental constants
ELECTRON_MASS = const.electron_mass  # kg
ELEMENTARY_CHARGE = const.elementary_charge  # C
HBAR = const.hbar  # J·s


class KappaMode(str, ValidatedEnum):
    """Mode for calculating the kappa factor for tunneling decay rates"""

    FAC10 = "10"
    CONSTANT = "constant"
    FULL = "full"


def decay_constant(
    kappa_mode: KappaMode | str,
    bias: float | np.ndarray,
    delta: float | np.ndarray,
    workfunction: float,
) -> np.ndarray:
    """Return decay constant for given barrier height and dependent on selected mode of kappa.

    Parameters
    ----------
    kappa_mode: KappaMode | str
        KappaMode class to select which level of approximation to use (see notes)
    bias : float | (M,) np.ndarray
        Bias voltage or 1d array of bias voltages, in V. Only used in case of kappa_mode='full'.
    delta : float | np.ndarray
        Only used for kappa_mode='full'. `delta` corresponds to the change in tunneling barrier height an electron experiences,
        when tunneling to/from the system, charging it from initial state `i` to final state `f`.
    workfunction : float
        Workfunction of System in eV

    Returns
    -------
    kappa : (M, delta.shape) np.ndarray
        Decay constant kappa for each transition, depending on kappa_mode.

    Notes
    -----
    The decay constant kappa is calculated based on the selected kappa_mode:
        - '10': kappa = log(10)/2.0
        - 'constant': kappa = sqrt(2 * ELECTRON_MASS * ELEMENTARY_CHARGE * (workfunction) / HBAR^2)*1e-10
        - 'full': kappa = sqrt(2 * ELECTRON_MASS * ELEMENTARY_CHARGE * (workfunction - delta + bias/2) / HBAR^2)*1e-10
    """

    if isinstance(kappa_mode, str):
        kappa_mode = KappaMode(kappa_mode)

    if not isinstance(kappa_mode, KappaMode):
        raise TypeError(
            f"kappa_mode must be type KappaMode, but got {type(kappa_mode)}"
        )

    bias = is_real_or_1darray(bias, "bias")

    if not isinstance(delta, (Real, np.ndarray)):
        raise TypeError(f"delta must be Real or np.ndarray, but got {type(delta)}")

    delta = np.asarray(delta)

    is_nonnegative_float(workfunction, "workfunction")

    barrier_height = np.ones(bias.shape + delta.shape) * workfunction

    match kappa_mode:
        case KappaMode.FAC10:
            return kappa10(barrier_height)
        case KappaMode.CONSTANT:
            return kappa_const_full(barrier_height)
        case KappaMode.FULL:
            barrier_height += np.subtract.outer(bias / 2, delta)
            return kappa_const_full(barrier_height)


def kappa_const_full(barrier_height):
    """Return decay constant for given barrier height"""

    if not (barrier_height > 0).all():
        raise ValueError("barrier_height must be positive.")

    kappa = np.sqrt(2 * ELECTRON_MASS * ELEMENTARY_CHARGE / HBAR**2 * barrier_height)
    kappa *= 1e-10
    return kappa


def kappa10(barrier_height):
    """Return decay constant (of shape barrier height) such that exp(2*kappa) = 10"""
    return np.ones_like(barrier_height) * np.log(10) / 2.0
