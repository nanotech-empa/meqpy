import os
from numbers import Real
import numpy as np
from .transition import Transition
from ..io.cube import Cube
from ..utils.types import (
    validate_real_or_1darray,
    validate_nonnegative_int,
    require_type,
)
from ..utils.constants import bohr


class Dyson(Transition):
    def __init__(
        self,
        cube: Cube | str | os.PathLike = None,
        slice_height: int = 1.5,
        center_mass: bool = True,
    ):
        """Initialize Dyson instance for charged transitions within Molecule.
        A slice of the cube data at certain height is used to extrapolate the wavefunction
        exponentionally into the vacuum.

        Parameters
        ----------
        cube : Cube | str | os.PathLike, optional
            Cube object, or path to cube file, from which to create Dyson instance.
        slice_height : int, optional
            Height in Angstrom at which to extract a slice of the cube, by default 1.5 Angstrom.
            See also `center_mass` parameter.
        center_mass : bool, optional
            Shift origin of coordinate system to molecule's center of mass, default is True.

        Raises
        ------
        TypeError | FileNotFoundError
            If `cube` is neither instance of Cube class, nor path to a cube file.
        """

        self.amplitude = None
        """Amplitude (magnitude squared) of Dyson orbital."""

        self.data = None
        """Slice of cube used for extrapolation into vacuum."""

        self.slice_height = slice_height
        """Height in Angstrom at which to extract a slice of the cube, by default 1.5 Angstrom."""

        cube = super().file_to_cube(cube)
        self.load_cube(cube, slice_height, center_mass)

    def load_cube(
        self,
        cube: Cube,
        slice_height: int = 1.5,
        center_mass: bool = True,
    ):
        """Load cube data and extract slice for extrapolation into vacuum.

        Parameters
        ----------
        cube : Cube
            Cube object from which to create Dyson instance.
        slice_height : int, optional
            Height in Angstrom at which to extract a slice of the cube, by default 1.5 Angstrom.
            See also `center_mass` parameter.
        center_mass : bool, optional
            Shift origin of coordinate system to molecule's center of mass, default is True.
        """

        require_type(slice_height, Real, "slice_height")

        super().parse_cube_dimensions(cube=cube, center_mass=center_mass)

        self.spacing = self.spacing[:2]

        self.origin = self.origin[:2]

        self.amplitude = cube.magsqr

        # switch to cube box coordinate system
        if center_mass:
            distance = slice_height + cube.center_of_mass[2]
        else:
            distance = slice_height - cube.origin[2]

        self.data = cube.get_slice_data(distance, axis=2) / bohr**1.5

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of sliced data."""
        return self.data.shape

    @property
    def x(self):
        """x values of cube"""
        return super().get_cart_axis(0)

    @property
    def y(self):
        """y values of cube grid"""
        return super().get_cart_axis(1)

    def extrapolate_wavefunction(
        self,
        height: float,
        kappa: float | np.ndarray,
        pad: int = 0,
    ) -> np.ndarray:
        """Extrapolate wavefunction exponentially from `slice_height` till `height`.

        Parameters
        ----------
        height : (K,) float
            Height in Angstrom to which to extrapolate wavefunction.
            Depends on `center_mass`
        kappa : float | (any) np.ndarray
            Decay constant(s) in 1/Angstrom.
        pad : int, optional
            Padding in pixels around the wavefunction slice, by default 0

        Returns
        -------
        wf_extrapolated : (nx, ny, K, any) np.ndarray
            Extrapolated wavefunction, with first two dimensions corresponding to x and y positions,
            the third dimension to different `height` values, and the remaining are given by the shape of `kappa`.

        Raises
        ------
        TypeError
            If any of the input parameters are of incorrect type.
        ValueError
            If `kappa` is not positive, or if `pad` is negative.
        """
        if isinstance(kappa, Real):
            kappa = np.array([kappa])

        require_type(kappa, np.ndarray, "kappa")

        if (kappa <= 0).any():
            raise ValueError("kappa must be positive.")

        if kappa.ndim == 0:
            kappa = np.array([kappa])

        validate_nonnegative_int(pad, "pad")

        padding = ((pad, pad), (pad, pad))
        data = np.pad(self.data, padding)
        n1, n2 = data.shape

        height = validate_real_or_1darray(height, "height")
        dz = abs(height - self.slice_height)

        # =====================================================================
        # The following code snipped was adapted from:
        # https://github.com/nanotech-empa/cp2k-spm-tools/blob/main/cp2k_spm_tools/cp2k_grid_orbitals.py
        #
        fourier = np.fft.rfft2(data)

        # Build the in-plane reciprocal basis
        b1, b2 = 2 * np.pi * np.linalg.inv(self.spacing[:, :2]).T

        i = np.fft.fftfreq(n1)
        j = np.fft.rfftfreq(n2)

        # Cartesian wavevector at each FFT bin
        kx = i[:, None] * b1[0] + j[None, :] * b2[0]
        ky = i[:, None] * b1[1] + j[None, :] * b2[1]

        kappa_xy = np.sqrt(kx**2 + ky**2 + kappa[..., None, None] ** 2)
        decay = np.exp(-np.multiply.outer(dz, kappa_xy))

        wf_extrapolated = np.fft.irfft2(fourier * decay, data.shape)
        #
        # =====================================================================

        # sort x,y axis to the front
        return np.moveaxis(wf_extrapolated, [-2, -1], [0, 1])

    def coupling_strength(
        self,
        height: float,
        kappa: float | np.ndarray,
        pad: int = 0,
        squeeze: bool = True,
    ) -> np.ndarray:
        """Get coupling strength between molecule wavefunction and tip,
        by extrapolating the wavefunction to the tip position and squaring it.

        Parameters
        ----------
        height : (K,) float
            Height in Angstrom to which to extrapolate wavefunction.
            Depends on `center_mass`
        kappa : float | (any) np.ndarray
            Decay constant(s) in 1/Angstrom.
        pad : int, optional
            Padding in pixels around the wavefunction slice, by default 0
        squeeze : bool, optional
            If `True` (default), squeeze the output array to remove singleton dimensions.

        Returns
        -------
        coupling_strength_matrix : (K, any) np.ndarray
            Coupling strength map, with first two dimensions corresponding to x and y positions,
            the third dimension to different `height` values, and the remaining are given by the shape of `kappa`.

        Notes
        -----
        This function is a wrapper and returns the square of `Dyson.extrapolate_wavefunction()`.
        """
        wf_map2 = (
            self.extrapolate_wavefunction(height=height, kappa=kappa, pad=pad) ** 2
        )

        if squeeze:
            wf_map2 = np.squeeze(wf_map2)

        return wf_map2
