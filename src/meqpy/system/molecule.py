from .system import System
from .dyson import Dyson
from ..utils.types import (
    is_real_or_1darray,
    is_nonnegative_float,
    is_sequence_of_pairs,
    is_pair,
)
from ..utils.decay_constant import KappaMode
from ..utils.coordinates import pad_lin_extrapolate, value_to_index
import numpy as np
from numbers import Real
from typing import Sequence
from warnings import warn
import scipy.constants as const

BOHR2ANG = const.physical_constants["Bohr radius"][0] * 1e10  # Angstrom
ELEMENTARY_CHARGE = const.elementary_charge  #


class Molecule(System):
    """System using Dyson orbitals for charging transitions."""

    def __init__(self, tip_radius: float = 2.0, **kwargs):
        """Initialize Molecule.

        Parameters
        ----------
        tip_radius : float, optional
            Radius of s-wave tip in Angstrom, by default 2.0
        **kwargs
            Keyword arguments for System class."""
        super().__init__(**kwargs)

        self._dysons = {}
        self._tip_radius = tip_radius
        """Radius of s-wave tip in Angstrom"""

    @property
    def tip_radius(self) -> float:
        """Radius of s-wave tip in Angstrom."""
        return self._tip_radius

    @tip_radius.setter
    def tip_radius(self, tip_radius):
        if is_nonnegative_float(tip_radius, "tip_radius"):
            self._tip_radius = tip_radius

    def add_dyson(self, a: str | int, b: str | int, dyson: Dyson):
        """Add dyson transition to molecule.

        Parameters
        ----------
        a : str | int
            Label or index of first state.
        b : str | int
            Label or index of second state.
        dyson : Dyson
            Dyson object for this transition.

        Raises
        ------
        TypeError
            If dyson is not type Dyson.
        ValueError
            If given states are not valid for charging transition.
        """
        if not isinstance(dyson, Dyson):
            raise TypeError(f"dyson must be type Dyson, but got type {type(dyson)}")

        if not self._valid_charging_pair(a, b):
            raise ValueError("States must differ in charge by 1.")

        key = self._state_tuple(a, b)
        self._dysons[key] = dyson

    def _state_tuple(self, a: str | int, b: str | int) -> tuple[str, str]:
        """Create normed tuple for two states, to be used as key in `Molecule.dysons`.

        Parameters
        ----------
        a : str | int
            Label or index of first state.
        b : str | int
            Label or index of second state.

        Returns
        -------
        tuple[str,str]
            Alphabetically sorted tuple containing label of two states.
        """
        if isinstance(a, (int, np.int_)):
            a = self.states[int(a)].label
        else:
            self.get_index(a)  # check if a is existing label

        if isinstance(b, (int, np.int_)):
            b = self.states[int(b)].label
        else:
            self.get_index(b)  # check if b is existing label

        return (min(a, b), max(a, b))

    def _valid_charging_pair(self, a: str | int, b: str | int) -> bool:
        """Check if two given states differ in charge by exactly 1.

        Parameters
        ----------
        a : str | int
            Label or index of first state.
        b : str | int
            Label or index of second state.

        Returns
        -------
        bool
            True if the two states differ in charge by exactly 1.
        """
        dQ = self.get_state(a).charge - self.get_state(b).charge
        return abs(dQ) == 1

    @property
    def dysons(self) -> dict[tuple[str, str], Dyson]:
        """Dictionary containing Dyson objects for charging transitions."""
        # should we use copy here? otherwise it is easy to accitentally change it
        return self._dysons.copy()

    @dysons.setter
    def dysons(self, dysons: dict[tuple[str, str], Dyson]):
        if not isinstance(dysons, dict):
            raise TypeError(f"dysons must be dictionary but got {type(Dyson)}")

        for key, dyson in dysons.items():
            if not isinstance(key, tuple):
                raise TypeError(f"key must be tuple with length 2, but got {type(key)}")

            if len(key) != 2:
                raise TypeError(
                    f"key must have length 2, but got Iterator of length {len(key)}"
                )

            a, b = key

            self.add_dyson(a, b, dyson)

    @property
    def dyson_shape(self) -> tuple[int, int]:
        """Shape of Dyson orbitals."""
        if not len(self._dysons):
            return ()

        # pick random dyson orbital as reference
        keys = list(self._dysons.keys())
        key = keys.pop(0)
        shape = self._dysons[key].shape

        # check all other dyson orbitals have same shape
        for key in keys:
            if shape != self._dysons[key].shape:
                raise ValueError("Dysons do not have same shape.")

        return shape

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Shape of System: (nx, ny, num_states, num_states)"""
        return self.dyson_shape + super().shape

    @property
    def dyson_amplitudes(self) -> np.ndarray:
        """Amplitudes of charging transitions, given my Dyson orbitals.

        Returns
        -------
        amplitudes : (N,N) np.ndarray
            2d matrix containing amplitudes for each transition.
        """
        amps = np.zeros_like(self.dE)
        for key, dyson in self._dysons.items():
            amps += self.matrix_by_states(*key, symmetric=True) * dyson.amplitude
        return amps

    @property
    def missing_dysons(self) -> set[tuple[str, str]]:
        """Return set of all possible charging transitions without assigned Dyson object."""

        # keys for all possible charging transitions
        possible_transitions = np.argwhere(abs(self.dQ) == 1).astype(int)
        possible_keys = set({})
        for transition in possible_transitions:
            f, i = transition
            possible_keys.add(self._state_tuple(f, i))

        # remove all keys found in self.dysons and return missing rest
        for key in self.dysons.keys():
            possible_keys.remove(key)

        return possible_keys

    def dyson_key_to_indices(self, key: tuple[str, str]) -> tuple[int, int]:
        """Return indices of states for given pair of State labels

        Parameters
        ----------
        key : tuple[str, str]
            Tuple containing exactly two labels for existing states in Molecule.

        Returns
        -------
        indices : tuple[int,int]
            Tuple containing indices for given states.

        Raises
        ------
        TypeError
            If key is not a tuple
        ValueError
            if key is not of length 2.
        """
        if not isinstance(key, tuple):
            raise TypeError(f"key must be tuple, but got {type(key)}.")

        if len(key) != 2:
            raise ValueError("key must be of length 2.")

        a, b = key
        return self.get_index(a), self.get_index(b)

    @property
    def x(self) -> np.ndarray:
        """Array containing values of x-axis of Dyson orbitals"""
        return self._axis("x")

    @property
    def y(self) -> np.ndarray:
        """Array containing values of y-axis of Dyson orbitals"""
        return self._axis("y")

    def _axis(self, axis: str) -> np.ndarray:
        """Function to get values along arbitrary axis of Dyson orbitals

        Parameters
        ----------
        axis : str
            Axis to get values for, must be "x" or "y".

        Returns
        -------
        np.ndarray
            1d array containing values along given axis of Dyson orbitals.

        Raises
        ------
        TypeError
            If axis is not a string.
        ValueError
            If axis is not "x" or "y".
        ValueError
            If axes of Dyson orbitals are not consistent.
        """
        if not len(self._dysons.keys()):
            return None

        if not isinstance(axis, str):
            raise TypeError(f"axis must be str, but got {type(axis)}")

        if axis not in ["x", "y"]:
            raise ValueError(f"axis must be 'x' or 'y' but got {axis}")

        # pick random dyson orbital as reference
        keys = list(self._dysons.keys())
        key = keys.pop(0)
        ax = getattr(self._dysons[key], axis)

        # check all remaining dysons have the same axis values
        for key in keys:
            axi = getattr(self._dysons[key], axis)
            if len(ax) != len(axi) or (ax != axi).any():
                raise ValueError(f"{axis} is not the same for all dysons.")

        return ax

    def x_padded(self, pad: int = 0) -> np.ndarray:
        """Array containing values of x-axis, extrapolated on both ends.

        Parameters
        ----------
        pad : int, optional
            Number of points to be added on both ends, by default 0

        Returns
        -------
        np.ndarray
            1d array containing values along x-axis of Dyson orbitals, linearily extrapolated on both ends by `pad` points.
        """
        return pad_lin_extrapolate(self.x, pad)

    def y_padded(self, pad: int = 0) -> np.ndarray:
        """Array containing values of y-axis, extrapolated on both ends.

        Parameters
        ----------
        pad : int, optional
            Number of points to be added on both ends, by default 0

        Returns
        -------
        np.ndarray
            1d array containing values along y-axis of Dyson orbitals, linearily extrapolated on both ends by `pad` points.
        """
        return pad_lin_extrapolate(self.y, pad)

    def get_indices(self, xy_pairs: Sequence) -> list[tuple[int, int]]:
        """Translate coordinates of points in xy-plane to indices for given Dyson orbitals.

        Parameters
        ----------
        xy_pairs : Sequence
            Sequence of (x,y) pairs.

        Returns
        -------
        Indices: list[tuple[int, int]]
            List of tuples with indices for Dyson orbitals.
        """
        try:
            if is_pair(xy_pairs, Real, "xy_pairs"):
                xy_pairs = [xy_pairs]
                single_pair = True
        except (TypeError, IndexError):
            is_sequence_of_pairs(xy_pairs, Real, "xy_pairs")
            single_pair = False

        indices = []
        for xy in xy_pairs:
            x, y = xy

            ix = value_to_index(x, self.x)
            iy = value_to_index(y, self.y)

            indices.append((ix, iy))

        # keep output in same form as input
        if single_pair:
            indices = indices[0]

        return indices

    def charging_rates(
        self,
        z: float | np.ndarray,
        bias: float | np.ndarray = 0.0,
        kappa_mode: str = None,
        squeeze: bool = True,
        scale_by_dyson: bool = True,
    ) -> np.ndarray:
        """Get transition rates by charging of system, including amplitude of Dyson orbitals:
        transition rate = coupling strength * normalized charging transition * Clebsch-Gordan factors * dyson_amplitudes
        This method does not consider xy-dependence of the coupling, use `Molecule.charging_rate_dyson()` instead.

        Parameters
        ----------
        z : float | (K,) np.ndarray
            Distance between Molecule and lead in Angstrom.
        bias : float | (M,) np.ndarray, optional
            Bias voltage(s) between system and lead in V, by default 0.0. Only used in case of kappa_mode='full'.
        kappa_mode : str
            Optional parameter to temporarily overwrite kappa_mode. If None (default), `self.kappa_mode` will be used.
        squeeze : bool, optional
            The returned array is squeezed to remove any dimensions of size 1, default is `True`.
        scale_by_dyson : bool, optional
            If False, return charging_rates without scaling by Dyson amplitudes.

        Returns
        -------
        charging_rates : (K,M,N,N) np.ndarray
            Array containing transition rates by charging, with N being number of states in system.
            The returned array is squeezed to remove any dimensions of size 1 if `squeeze` is `True`.

        Raises
        ------
        TypeError
            If `scale_by_dyson` is not type bool.

        Notes
        -----
        This method is a simple wrapper for combining:
            * `self.coupling_strength`
            * `self.normalized_charging_transition`
            * `self.clebsch_gordan_factors`
            * `self.dyson_amplitudes`
        """
        if not isinstance(scale_by_dyson, bool):
            raise TypeError(
                f"scale_by_dyon must be bool, but got {type(scale_by_dyson)}"
            )

        charging_rates = super().charging_rates(z, bias, kappa_mode, squeeze=False)
        if scale_by_dyson:
            charging_rates *= self.dyson_amplitudes

        if squeeze:
            charging_rates = np.squeeze(charging_rates)

        return charging_rates

    def charging_rates_dyson(
        self,
        z: float | np.ndarray,
        bias: float | np.ndarray = 0.0,
        kappa_mode: str = None,
        pad: int = 0,
        center_mass: bool = True,
        squeeze: bool = True,
        suppress_warning: bool = False,
    ) -> np.ndarray:
        """Get transition rates by charging of system, including coupling via Dyson orbitals:
        transition rate = coupling strength_dyson(x,y) * normalized charging transition * Clebsch-Gordan factors

        Parameters
        ----------
        z : float | (K,) np.ndarray
            Height of tip. The center of tip is at `z + self.tip_radius`.
            If `center_mass` is True, z is relative to center of mass of molecule.
        bias : float | (M,) np.ndarray, optional
            Bias voltage(s) between system and lead in V, by default 0.0. Only used in case of kappa_mode='full'.
        kappa_mode : str
            Optional parameter to temporarily overwrite kappa_mode. If None (default), `self.kappa_mode` will be used.
        pad : int, optional
            pad wavefunction slice with `pad` number of points (zeros) on all sides, by default 0.
        center_mass : bool, optional
            If `True` (default): Consider `z` to be relative to center of mass of molecule.
            If `False`: `z` is in absolute coordinates.
        squeeze : bool, optional
            The returned array is squeezed to remove any dimensions of size 1, default is `True`.
        suppress_warning : bool, optional
            If False (default): Warn if array is expected to exceed 4GB in memeroy.

        Returns
        -------
        charging_rates : (nx, ny, K, M, N, N) np.ndarray
            Array containing all transition rates by charging, with (nx,ny) being shape of Dyson orbitals
            and N being number of states in system.
            The returned array is squeezed to remove any dimensions of size 1 if `squeeze` is `True`.

        Notes
        -----
        This method is a simple wrapper for combining:
            * `self.coupling_strength_dyson`
            * `self.normalized_charging_transition`
            * `self.clebsch_gordan_factors`

        All transitions are calculated at once, which can lead to large memory requirement.
        If only a few point in the xy-plane are needed, consider using `Molecule.charging_rates_pointspec` instead.
        """
        z = is_real_or_1darray(z, "z")
        bias = is_real_or_1darray(bias, "bias")

        if not suppress_warning:
            out_shape = z.shape + bias.shape + self.shape
            size_GB = np.prod(out_shape) * 8 * 1e-9  # 8 bytes per value
            if size_GB > 4:
                warn_message = (
                    f"Array expected to require more than {size_GB:.1f}GB of memory."
                )
                if len(z) > 1 or len(bias) > 1:
                    warn_message += (
                        " Consider using `Molecule.charging_rates_pointspec()`."
                    )
                warn(warn_message)

        charging_rates = self.coupling_strength_dyson(
            z,
            bias,
            kappa_mode=kappa_mode,
            pad=pad,
            center_mass=center_mass,
            squeeze=False,
        )
        charging_rates *= self.normalized_charging_transitions(bias, squeeze=False)
        charging_rates *= self.clebsch_gordan_factors

        if squeeze:
            charging_rates = np.squeeze(charging_rates)

        return charging_rates

    def charging_rates_pointspec(
        self,
        points: Sequence,
        z: float | np.ndarray,
        bias: float | np.ndarray = 0.0,
        kappa_mode: str = None,
        pad: int = 0,
        center_mass: bool = True,
        squeeze: bool = True,
    ) -> np.ndarray:
        """Get transition rates by charging of system, including coupling via Dyson orbitals at certain points in the xy-plane.
        This method is a for-loop wrapper for `Molecule.charging_rates_dyson()`,
        allowing to calculate charging rates for certain tip positions only, in order to reduce the memory needed.

        Parameters
        ----------
        points : (P,) Sequence
            Sequence of integer pairs (px,py), for which to calculate the charging rates.
        z : float | (K,) np.ndarray
            Height of tip. The center of tip is at `z + self.tip_radius`.
            If `center_mass` is True, z is relative to center of mass of molecule.
        bias : float | (M,) np.ndarray, optional
            Bias voltage(s) between system and lead in V, by default 0.0. Only used in case of kappa_mode='full'.
        kappa_mode : str
            Optional parameter to temporarily overwrite kappa_mode. If None (default), `self.kappa_mode` will be used.
        pad : int, optional
            pad wavefunction slice with `pad` number of points (zeros) on all sides, by default 0.
        center_mass : bool, optional
            If `True` (default): Consider `z` to be relative to center of mass of molecule.
            If `False`: `z` is in absolute coordinates.
        squeeze : bool, optional
            The returned array is squeezed to remove any dimensions of size 1, default is `True`.

        Returns
        -------
        charging_rates : (P, K, M, N, N) np.ndarray
            Array containing all transition rates by charging, with N being number of states in system.
            The returned array is squeezed to remove any dimensions of size 1 if `squeeze` is `True`.
        """
        try:
            if is_pair(points, int, "points"):
                points = [points]
        except (TypeError, IndexError):
            is_sequence_of_pairs(points, int, "points")

        z = is_real_or_1darray(z, "z")
        bias = is_real_or_1darray(bias, "bias")

        norm_rates = self.normalized_charging_transitions(bias, squeeze=False)
        norm_rates *= self.clebsch_gordan_factors

        coupling_mat = np.zeros((len(points),) + z.shape + norm_rates.shape)

        # calculate coupling strength sequentially
        # for each combination of z and bias
        # and store only values for given points
        for i, iz in enumerate(z):
            for j, jbias in enumerate(bias):
                tmp_mat = self.coupling_strength_dyson(
                    iz,
                    jbias,
                    kappa_mode=kappa_mode,
                    pad=pad,
                    center_mass=center_mass,
                    squeeze=False,
                )

                for k, point in enumerate(points):
                    px, py = point
                    coupling_mat[k, i, j] = tmp_mat[px, py]

        charging_rates = coupling_mat * norm_rates

        if squeeze:
            charging_rates = np.squeeze(charging_rates)

        return charging_rates

    def coupling_strength_dyson(
        self,
        z: float | np.ndarray,
        bias: float | np.ndarray = 0.0,
        kappa_mode: str = None,
        pad: int = 0,
        center_mass: bool = True,
        squeeze: bool = True,
        warn_missing_dysons: bool = True,
    ) -> np.ndarray:
        """Calculate coupling strength between Molecule and s-wave contact (i.e. tip), using Dyson orbitals.

        Parameters
        ----------
        z : float | (K,) np.ndarray
            Height of tip. The center of tip is at `z + self.tip_radius`.
            If `center_mass` is True, z is relative to center of mass of molecule.
        bias : float | (M,) np.ndarray, optional
            Bias voltage(s) between molecule and tip in V, by default 0.0. Only used in case of kappa_mode='full'.
        kappa_mode : str
            Optional parameter to temporarily overwrite kappa_mode. If None (default), `self.kappa_mode` will be used.
        pad : int, optional
            pad wavefunction slice with `pad` number of points (zeros) on all sides, by default 0.
        center_mass : bool, optional
            If `True` (default): Consider `z` to be relative to center of mass of molecule.
            If `False`: `z` is in absolute coordinates.
        squeeze : bool, optional
            The returned array is squeezed to remove any dimensions of size 1, default is `True`.
        warn_missing_dysons: bool, optional
            Raise a warning, in case some charging transitions are missing a Dyson instance, default True.
            In case of missing Dyson instance, the transition will be set to zero.

        Returns
        -------
        coupling_strength : (nx, ny, K, M, N, N) np.ndarray
            Array containing all coupling strengths, with (nx,ny) being shape of Dyson orbitals
            and N being number of states in system.
            The returned array is squeezed to remove any dimensions of size 1 if `squeeze` is `True`.

        Raises
        ------
        TypeError
            If pad is not int.
        ValueError
            If pad is negative.

        Notes
        -----
        - coupling strength = C * Psi(r)**2, with Psi being Dyson orbital wavefunctino and r the position of center of tip.
        - C ≈ 0.1 * tip_radius**2 * exp(2 * tip_radius * kappa),
        according to equation (10) in Phys. Rev. B 31, 805 (1985),
        https://doi.org/10.1103/PhysRevB.31.805
        This is only an approximation, to obtain values in the correct order of magnitude.
        """
        if isinstance(kappa_mode, type(None)):
            kappa_mode = self.kappa_mode
        kappa_mode = KappaMode(kappa_mode)

        if not isinstance(pad, int):
            raise TypeError(f"pad must be int but got {type(pad)}.")

        if pad < 0:
            raise ValueError("pad must not be negative.")

        if warn_missing_dysons and len(self.missing_dysons):
            warn(
                "Some charging transition have not been assigned a Dyson instance. Will scale transition to zero."
            )

        z = is_real_or_1darray(z, "z") + self.tip_radius

        kappa_mat = self.kappa(bias, kappa_mode=kappa_mode, squeeze=False)

        # create output array with correct shape
        dyson_shape = (self.dyson_shape[0] + 2 * pad, self.dyson_shape[1] + 2 * pad)
        out_shape = dyson_shape + z.shape + kappa_mat.shape
        coupling_strength_mat = np.zeros(out_shape)

        # fill array:
        for key, dyson in self._dysons.items():
            a, b = self.dyson_key_to_indices(key)

            kappa_ab = kappa_mat[..., a, b]
            coupling_strength_mat[..., a, b] = dyson.coupling_strength(
                z, kappa_ab, pad, center_mass, squeeze=False
            ) * self._renorm_factor_dysons(kappa_ab)

            if kappa_mode == KappaMode.FULL:
                kappa_ba = kappa_mat[..., b, a]
                coupling_strength_mat[..., b, a] = dyson.coupling_strength(
                    z, kappa_ba, pad, center_mass, squeeze=False
                ) * self._renorm_factor_dysons(kappa_ba)
            else:  # kappa is constant and the matrix symmetric
                coupling_strength_mat[..., b, a] = coupling_strength_mat[..., a, b]

        if squeeze:
            coupling_strength_mat = np.squeeze(coupling_strength_mat)

        return coupling_strength_mat

    def _renorm_factor_dysons(self, kappa: np.ndarray) -> np.ndarray:
        """Renormalization factor approximation, according to Tersoff-Hamann
        equation (10) in https://doi.org/10.1103/PhysRevB.31.805

        Parameters
        ----------
        kappa : float | np.ndarray
            decay constant

        Returns
        -------
        float | np.ndarray
            Renomalization factor for coupling strength.
        """

        if isinstance(kappa, Real):
            kappa = np.array([kappa])

        if not isinstance(kappa, np.ndarray):
            raise TypeError(
                f"kappa must be a real number or np.ndarray but got {type(kappa)}."
            )

        renorm_factor = 0.1
        renorm_factor *= (self.tip_radius / BOHR2ANG) ** 2
        renorm_factor *= np.exp(2 * kappa * self.tip_radius)
        renorm_factor *= 1 / ELEMENTARY_CHARGE

        return renorm_factor
