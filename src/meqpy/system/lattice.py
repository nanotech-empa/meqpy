import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter1d

from .system import System
from .band_transition import BandTransition
from ..utils import validate_real_or_1darray, KappaMode, decay_constant

import scipy.constants as const

ELEMENTARY_CHARGE = const.elementary_charge  # C
ELECTRON_MASS = const.electron_mass  # kg
HBAR = const.hbar  # J·s

EV_TO_K = 2 * ELECTRON_MASS / HBAR**2 * ELEMENTARY_CHARGE * 1e-20  # 1/V/Ang**2
G0 = ELEMENTARY_CHARGE / (
    2 * np.pi * HBAR
)  # Conductance quantum for each spin channel in 1/s


class Lattice(System):
    """System for periodic samples with band structures.

    Extends :class:`System` with support for band-resolved charging transitions,
    where tunneling rates are computed by integrating over a continuous band of
    electronic states rather than discrete levels.

    Parameters
    ----------
    **kwargs
        Passed through to :class:`System`.
    """

    def __init__(self, **kwargs):
        """Initialize Lattice."""
        super().__init__(**kwargs)
        self._band_transition_dict = {}

    # ------------------------------------------------------------------
    # Band transition dict
    # ------------------------------------------------------------------

    def add_band_transition(self, a: str | int, b: str | int, band: BandTransition):
        """Register a :class:`BandTransition` for the charging transition a → b.

        Parameters
        ----------
        a : str | int
            Label or index of the initial state.
        b : str | int
            Label or index of the final state.
        band : BandTransition
            Band transition object.

        Raises
        ------
        TypeError
            If ``band`` is not an instance of :class:`BandTransition`.
        ValueError
            If the states ``a`` and ``b`` do not satisfy the charging
            (and spin) selection rules.
        """
        if not isinstance(band, BandTransition):
            raise TypeError(
                f"band must be type BandTransition, but got type {type(band)}"
            )

        self._valid_charging_pair(a, b)

        key = self._state_tuple(a, b, sorted=False)
        self._band_transition_dict[key] = band

    @property
    def band_transition_dict(self) -> dict[tuple[str, str], BandTransition]:
        """Dictionary of :class:`BandTransition` objects keyed by ``(label_a, label_b)``.

        Returns
        -------
        dict[tuple[str, str], BandTransition]
            Mapping from state-label pairs to their associated band transition.
        """
        return self._band_transition_dict

    @band_transition_dict.setter
    def band_transition_dict(self, bands: dict[tuple[str, str], BandTransition]):
        """Set multiple band transitions at once from a dictionary.

        Each key must be a ``(label_a, label_b)`` tuple. Validation is
        delegated to :meth:`add_band_transition`.

        Parameters
        ----------
        bands : dict[tuple[str, str], BandTransition]
            Dictionary mapping state-label pairs to :class:`BandTransition`
            objects.

        Raises
        ------
        TypeError
            If ``bands`` is not a :class:`dict`.
        """
        if not isinstance(bands, dict):
            raise TypeError(f"bands must be a dict, but got {type(bands)}")
        for (a, b), band in bands.items():
            self.add_band_transition(a, b, band)

    def band_energy(self, key: tuple) -> np.ndarray:
        """Energy of band transition.

        Returns
        -------
        energy : (E,) np.ndarray
            Energy grid in eV.
        """

        i, f = self.get_index(key[0]), self.get_index(key[1])
        band = self.band_transition_dict[key]
        return (band.energy + self.dE[f, i]) * np.sign(-self.dQ[f, i])

    # ------------------------------------------------------------------
    # Rate computation
    # ------------------------------------------------------------------

    def band_kappa(
        self,
        key: tuple,
        bias: float | np.ndarray,
        kappa_mode: str | None = None,
        squeeze: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the decay constant κ over the band energy grid, with
        ``κ = sqrt(2m/hbar² * barrier_height + k_∥²)``.

        Parameters
        ----------
        key : tuple
            Key for band transition whose energy grid and in-plane momentum are used.
        bias : float | (M,) np.ndarray
            Bias voltage(s) in volts. Scalar or 1-D array of length ``nv``.
        kappa_mode : str | None, optional
            Kappa mode to use. ``None`` falls back to the instance default.
        squeeze : bool, optional
            If ``True`` (default), remove length-1 dimensions from the output.

        Returns
        -------
        energy : np.ndarray
            Energy grid of the band, shape ``(E,)``.
        kappa_mat : np.ndarray
            Decay constant array. Shape ``(nv,E)`` if kappa_mode is "full",
            and ``(E,)`` otherwise (before squeezing).
        """
        bias = validate_real_or_1darray(bias, "bias")
        kappa_mode = self._resolve_kappa_mode(kappa_mode)

        band = self.band_transition_dict[key]
        energy = self.band_energy(key)

        kappa_mat = decay_constant(kappa_mode, bias, energy, self.workfunction)
        if kappa_mode is not KappaMode.FULL:
            kappa_mat = kappa_mat[0]

        kappa_mat = np.sqrt(kappa_mat + band.kpar**2)

        if squeeze:
            kappa_mat = np.squeeze(kappa_mat)

        return energy, kappa_mat

    def get_band_charging_rate(
        self,
        key: tuple,
        z: float | np.ndarray,
        bias: float | np.ndarray,
        kappa_mode: str | None = None,
        squeeze: bool = True,
    ) -> np.ndarray:
        """Compute the charging rate for a single :class:`BandTransition`.

        The rate is obtained by:
        1. Computing the ldos at tip position
        2. Compute tunneling rates by integrating over energy,
           assuming quantum of conductance G0 at z = 0.
        3. Returning tunneling rate at given bias by interpolation.

        Parameters
        ----------
        key : tuple
            Key for band transition to evaluate.
        z : float | np.ndarray
            Tip-sample distance(s) in Å. Scalar or 1-D array of length ``nz``.
        bias : float | np.ndarray
            Bias voltage(s) in volts. Scalar or 1-D array of length ``nv``.
        kappa_mode : str | None, optional
            Kappa mode to use. ``None`` falls back to the instance default.
        squeeze : bool, optional
            If ``True`` (default), remove length-1 dimensions from the result.

        Returns
        -------
        np.ndarray
            Charging rates of shape ``(nz, nv)`` (before squeezing).
        """
        z = validate_real_or_1darray(z, "z")
        bias = validate_real_or_1darray(bias, "bias")
        nz, nv = len(z), len(bias)
        band = self.band_transition_dict[key]
        bias_inc = np.flip(bias) if bias[0] > bias[-1] else bias

        # ----------------------------------------
        # 1. Compute the differential rates (LDOS)
        # ----------------------------------------

        # kappa_mat can be of shape (1, M) or (nv, M)
        # depending on kappa_mode and shape of bias
        energy, kappa_mat = self.band_kappa(
            key, bias_inc, kappa_mode=kappa_mode, squeeze=False
        )

        # ldos can be of shape (1, M), (nz, M) or (nz, nv, M)
        # depending on kappa_mode and shape of z and bias
        ldos = np.exp(-2 * np.multiply.outer(z, kappa_mat)) * band.dos

        # broaden with gaussian lineshape if band.hwhm > 0:
        if band.hwhm > 0:
            sigma_eV = band.hwhm / (2 * np.sqrt(2 * np.log(2)))
            sigma_samples = sigma_eV / band.dx
            ldos = gaussian_filter1d(ldos, sigma_samples, axis=-1)

        # ----------------------------------------
        # 2. Compute tunneling rates
        #    by integrating LDOS over energy
        # ----------------------------------------

        raw_rates = np.cumsum(ldos, axis=-1) * band.dx * G0

        # ----------------------------------------
        # 3. Interpolate the integrated rate
        #    at the actual bias voltages
        # ----------------------------------------

        # make sure energy is in ascending order for interpolation
        if energy[0] > energy[-1]:
            energy = np.flip(energy)
            raw_rates = np.flip(raw_rates, axis=-1)

        # raw_rates: (..., M); depending on kappa_mode and shape of z
        # it can be (1, M), (nz, M) or (nz, nv, M).
        # in the end we want charging_rates: (nz, nv)
        M = len(energy)
        leading_shape = raw_rates.shape[:-1]  # e.g. () or (nz,) or (nz, nv)
        n_rows = int(np.prod(leading_shape)) if leading_shape else 1
        rr_flat = raw_rates.reshape(n_rows, M)  # (n_rows, M)
        if n_rows == 1:
            rr_flat = np.tile(rr_flat[0], (2, 1))
            row_indices = np.arange(2)
        else:
            row_indices = np.arange(n_rows)

        interp = RectBivariateSpline(row_indices, energy, rr_flat, kx=1, ky=3)

        if n_rows == 1:
            # raw_rates shape (M,): skip filler-dimension and evaluate for bias (nz,) times
            charging_rates = interp(0, bias_inc, grid=True)[None, 0, :].repeat(
                nz, axis=0
            )
        elif raw_rates.ndim == 2:
            # raw_rates shape (nz, M): evaluate on full (nz × nv) grid
            charging_rates = interp(row_indices, bias_inc, grid=True)  # (nz, nv)
        else:
            # raw_rates shape (nz, nv, M): paired evaluation
            flat_bias = np.tile(bias_inc, nz) if raw_rates.ndim == 3 else bias
            charging_rates = interp(row_indices, flat_bias, grid=False).reshape(nz, nv)

        if bias[0] > bias[-1]:
            charging_rates = np.flip(charging_rates, axis=-1)

        if squeeze:
            charging_rates = np.squeeze(charging_rates)

        return charging_rates

    def charging_rates(
        self,
        z: float | np.ndarray,
        bias: float | np.ndarray = 0.0,
        kappa_mode: str | None = None,
        squeeze: bool = True,
    ) -> np.ndarray:
        """Compute charging rate matrix, including all band transitions.

        Parameters
        ----------
        z : float | np.ndarray
            Tip-sample distance(s) in Å. Scalar or 1-D array of length ``nz``.
        bias : float | np.ndarray, optional
            Bias voltage(s) in volts. Scalar or 1-D array of length ``nv``.
            Defaults to ``0.0``.
        kappa_mode : str | None, optional
            Kappa mode to use. ``None`` falls back to the instance default.
        squeeze : bool, optional
            If ``True`` (default), remove length-1 dimensions from the result.

        Returns
        -------
        np.ndarray
            Rate matrix of shape ``(..., n_states, n_states)`` where the
            leading dimensions correspond to ``z`` and ``bias``.
        """
        z = validate_real_or_1darray(z, "z")
        bias = validate_real_or_1darray(bias, "bias")
        kappa_mode = self._resolve_kappa_mode(kappa_mode)

        charging_rates = super().charging_rates(z, bias, kappa_mode, squeeze=False)

        # self.update_band_transitions()
        for key in self.band_transition_dict.keys():
            i, f = self.get_index(key[0]), self.get_index(key[1])
            band_rate = self.get_band_charging_rate(
                key, z, bias, kappa_mode, squeeze=False
            )
            band_rate *= self.clebsch_gordan_factors[f, i]
            charging_rates[..., f, i] = band_rate

        if squeeze:
            charging_rates = np.squeeze(charging_rates)

        return charging_rates

    # ------------------------------------------------------------------
    # kappa / kpar utilities
    # ------------------------------------------------------------------

    @property
    def kpar_offset(self) -> np.ndarray:
        """In-plane momentum matrix ``k_∥`` for all transitions.

        Entries are zero for non-band transitions, and ``band.kpar_offset``
        otherwise.

        Returns
        -------
        np.ndarray
            Symmetric matrix of shape ``(M, M)``.
        """
        kpar_mat = self.zeros
        for key, band in self.band_transition_dict.items():
            i, f = self.get_index(key[0]), self.get_index(key[1])
            kpar_mat[f, i] = band.kpar_offset
            kpar_mat[i, f] = band.kpar_offset
        return kpar_mat

    def kappa(
        self,
        bias: float | np.ndarray,
        kappa_mode: str | None = None,
        squeeze: bool = True,
    ) -> np.ndarray:
        """Calculate decay constant kappa for given energy difference(s) and bias voltage(s).

        Expands :meth:`System.kappa` by inclusion of the in-plane momentum.
        This method is to be used for tunneling out of bands, as it does not
        consider energy-dependent dispersion.

        Parameters
        ----------
        bias : float | (N,) np.ndarray
            Tip bias voltage(s) in volts.
        kappa_mode : str | None, optional
            Kappa mode to use. ``None`` falls back to the instance default.
        squeeze : bool, optional
            If ``True`` (default), remove length-1 dimensions from the result.

        Returns
        -------
        kappa : (N, M, M) np.ndarray
            Effective decay constant matrix, see :meth:`System.kappa`.
        """
        kappa_mat = super().kappa(bias, kappa_mode, squeeze)
        return np.sqrt(kappa_mat**2 + self.kpar_offset**2)
