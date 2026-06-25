import numpy as np
from numbers import Real

from ..utils import validate_real_or_1darray, validate_nonnegative_float, require_type
from ..utils.constants import ev_to_k2


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
        Defaults to ``1.0``.
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
        effective_mass: float = 1.0,
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

        # Cache for the internal energy grid; invalidated when grid parameters change.
        self._energy_cache: np.ndarray | None = None

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
        self._energy_cache = None  # invalidate cache

    @property
    def hwhm(self) -> float:
        """HWHM of the Gaussian broadening applied to transition rates, in eV."""
        return self._hwhm

    @hwhm.setter
    def hwhm(self, value: float):
        validate_nonnegative_float(value, "hwhm")
        self._hwhm = value
        self._energy_cache = None  # invalidate cache

    @property
    def dx(self) -> float:
        """Energy grid spacing for rate integration, in eV."""
        return self._dx

    @dx.setter
    def dx(self, value: float):
        require_type(value, Real, "dx")
        if value <= 0:
            raise ValueError(f"dx must be positive, but got {value}.")
        self._dx = value
        self._energy_cache = None  # invalidate cache

    # ------------------------------------------------------------------
    # Energy grid
    # ------------------------------------------------------------------

    @property
    def energy(self) -> np.ndarray:
        """Internal energy grid with band vertex at 0 and opening to positive, in eV.

        Extends from ``-dx - order * hwhm`` to ``bandwidth + order * hwhm``
        with step ``dx``. The result is cached and recomputed only when ``dx``,
        ``bandwidth``, or ``hwhm`` change.
        """
        if self._energy_cache is None:
            margin = self.dx + self._BROADENING_ORDER * self.hwhm
            x_min = -margin
            x_max = self.bandwidth + margin
            self._energy_cache = np.arange(x_min, x_max, self.dx)
        return self._energy_cache

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
        eps = self.energy
        k = np.sqrt(np.abs(ev_to_k2 * self.effective_mass * eps)) * (eps >= 0)
        sign = np.sign(self.kpar_offset)
        return k if sign == 0 else self.kpar_offset - k * sign

    @property
    def dos(self) -> np.ndarray:
        """Uniform DoS mask: 1 inside the band, 0 outside.

        Returns
        -------
        dos : (E,) np.ndarray
            DoS mask array.
        """
        eps = self.energy
        return ((eps >= 0) & (eps <= self.bandwidth)).astype(float)
