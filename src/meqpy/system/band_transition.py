import numpy as np
from scipy.ndimage import gaussian_filter1d
from numbers import Real

from ..utils import validate_real_or_1darray, validate_nonnegative_float

import scipy.constants as const

ELEMENTARY_CHARGE = const.elementary_charge  # C
ELECTRON_MASS = const.electron_mass  # kg
HBAR = const.hbar  # J·s

EV_TO_K = 2 * ELECTRON_MASS / HBAR**2 * ELEMENTARY_CHARGE * 1e-20  # 1/V/Ang**2


class BandTransition:
    """Charging transition through a parabolic band of electronic states.

    Models a continuum of tip-sample tunneling channels arising from a
    parabolic band. The density of states (DOS) is treated as uniform between
    the band vertex and ``bandwidth``.

    Parameters
    ----------
    kpar_offset : float, optional
        In-plane momentum k_∥ of the band maximum/minimum in 1/Å.
        Defaults to ``0.0``.
    effective_mass : float, optional
        Effective mass of the band in units of the electron mass.
        Defaults to ``0.0``.
    bandwidth : float, optional
        Full bandwidth of the band in eV. Defaults to ``1.0``.
    hwhm : float, optional
        Half-width at half-maximum (HWHM) of the Gaussian broadening applied
        to the transition rates, in eV. ``0`` disables broadening (default).
    dx : float, optional
        Energy grid spacing used for rate integration, in eV.
        Defaults to ``1e-3``.
    """

    # Number of HWHM widths to extend the internal energy grid beyond the
    # band edges, to avoid boundary artefacts from Gaussian broadening.
    _BROADENING_ORDER = 10

    def __init__(
        self,
        kpar_offset: float = 0.0,
        effective_mass: float = 0.0,
        bandwidth: float = 1.0,
        hwhm: float = 0,
        dx: float = 1e-3,
    ):
        """Initialize BandTransition."""
        self.kpar_offset = kpar_offset
        self.effective_mass = effective_mass
        self.bandwidth = bandwidth
        self.hwhm = hwhm
        self.dx = dx

        self._delta_Q: int | None = None
        self._delta_E: float | None = None

        # Cache for the internal energy grid; invalidated when grid parameters change.
        self._energy_internal_cache: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Properties with validation
    # ------------------------------------------------------------------

    @property
    def kpar_offset(self) -> float:
        """In-plane momentum k_∥ of the band maximum/minimum in 1/Å."""
        return self._kpar_offset

    @kpar_offset.setter
    def kpar_offset(self, value: float):
        validate_real_or_1darray(value, "kpar_offset")
        self._kpar_offset = value

    @property
    def effective_mass(self) -> float:
        """Effective mass of the band in units of the electron mass."""
        return self._effective_mass

    @effective_mass.setter
    def effective_mass(self, value: float):
        validate_nonnegative_float(value, "effective_mass")
        self._effective_mass = value

    @property
    def bandwidth(self) -> float:
        """Full bandwidth of the band in eV."""
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, value: float):
        validate_nonnegative_float(value, "bandwidth")
        self._bandwidth = value
        self._energy_internal_cache = None  # invalidate cache

    @property
    def hwhm(self) -> float:
        """HWHM of the Gaussian broadening applied to transition rates, in eV."""
        return self._hwhm

    @hwhm.setter
    def hwhm(self, value: float):
        validate_nonnegative_float(value, "hwhm")
        self._hwhm = value
        self._energy_internal_cache = None  # invalidate cache

    @property
    def dx(self) -> float:
        """Energy grid spacing for rate integration, in eV."""
        return self._dx

    @dx.setter
    def dx(self, value: float):
        if not isinstance(value, Real):
            raise TypeError(f"dx must be real, but got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"dx must be positive, but got {value}.")
        self._dx = value
        self._energy_internal_cache = None  # invalidate cache

    @property
    def delta_Q(self) -> int | None:
        """Charge change of the transition in elementary charge units (±1)."""
        return self._delta_Q

    @delta_Q.setter
    def delta_Q(self, value: int):
        if not isinstance(value, Real):
            raise TypeError(f"delta_Q must be integer, but got {type(value).__name__}")
        if abs(value) != 1:
            raise ValueError(f"delta_Q must be +1 or -1, but got {value}.")
        self._delta_Q = value

    @property
    def delta_E(self) -> float | None:
        """Energy difference between initial and final state in eV."""
        return self._delta_E

    @delta_E.setter
    def delta_E(self, value: float):
        if not isinstance(value, Real):
            raise TypeError(f"delta_E must be real, but got {type(value).__name__}")
        self._delta_E = value

    # ------------------------------------------------------------------
    # Charge/energy setup
    # ------------------------------------------------------------------

    def set_charge_energy(self, delta_Q: int, delta_E: float) -> None:
        """Set the charge and energy change for this transition.

        Must be called before accessing :attr:`energy` or computing rates.

        Parameters
        ----------
        delta_Q : int
            Change of charge state in elementary charge units. Must be ±1.
        delta_E : float
            Energy difference between initial and final state in eV.

        Raises
        ------
        TypeError
            If ``delta_Q`` is not a real number, or ``delta_E`` is not real.
        ValueError
            If ``delta_Q`` is not ±1.
        """
        self.delta_Q = delta_Q
        self.delta_E = delta_E

    # ------------------------------------------------------------------
    # Energy grid
    # ------------------------------------------------------------------

    @property
    def _energy_internal(self) -> np.ndarray:
        """Internal energy grid centred on the band, in eV.

        Extends from ``-dx - order * hwhm`` to ``bandwidth + order * hwhm``
        with step ``dx``. The result is cached and recomputed only when ``dx``,
        ``bandwidth``, or ``hwhm`` change.
        """
        if self._energy_internal_cache is None:
            margin = self._BROADENING_ORDER * self.hwhm
            x_min = -self.dx - margin
            x_max = self.bandwidth + margin
            self._energy_internal_cache = np.arange(x_min, x_max, self.dx)
        return self._energy_internal_cache

    @property
    def energy(self) -> np.ndarray:
        """Energy grid for rate calculation.

        Returns
        -------
        energy : (E,) np.ndarray
            Energy grid in eV.

        Raises
        ------
        TypeError
            If :attr:`delta_E` or :attr:`delta_Q` have not yet been set via
            :meth:`set_charge_energy`.
        """
        if self._delta_E is None:
            raise TypeError("delta_E has not been set; call set_charge_energy first.")
        if self._delta_Q is None:
            raise TypeError("delta_Q has not been set; call set_charge_energy first.")
        return (self._energy_internal + self.delta_E) * np.sign(-self.delta_Q)

    # ------------------------------------------------------------------
    # Band properties
    # ------------------------------------------------------------------

    @property
    def kpar(self) -> np.ndarray:
        """In-plane momentum k_∥ over the internal energy grid, in 1/Å.

        Returns
        -------
        kpar : (E,) np.ndarray
            In-plane momentum array.
        """
        eps = self._energy_internal
        k = np.sqrt(np.abs(EV_TO_K * self.effective_mass * eps)) * (eps >= 0)
        return k - self.kpar_offset

    @property
    def dos(self) -> np.ndarray:
        """Uniform DoS mask: 1 inside the band, 0 outside.

        Returns
        -------
        dos : (E,) np.ndarray
            DoS mask array.
        """
        eps = self._energy_internal
        return ((eps >= 0) & (eps <= self.bandwidth)).astype(float)

    # ------------------------------------------------------------------
    # Broadening
    # ------------------------------------------------------------------

    def gaussian_broadening(self, rates: np.ndarray) -> np.ndarray:
        """Apply Gaussian broadening to a rate array along the energy axis,
        if ``self.hwhm`` is non-zero.

        Parameters
        ----------
        rates : np.ndarray
            Rate array whose last axis corresponds to the energy grid.

        Returns
        -------
        rates : np.ndarray
            Broadened rate array, same shape as ``rates``.
        """
        if self.hwhm == 0:
            return rates

        # Convert HWHM (eV) → sigma (eV) → sigma (grid samples)
        sigma_eV = self.hwhm / (2 * np.sqrt(2 * np.log(2)))
        sigma_samples = sigma_eV / self.dx
        return gaussian_filter1d(rates, sigma_samples, axis=-1)
