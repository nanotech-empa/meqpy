import os
from numbers import Real
import numpy as np
from .transition import Transition
from ..io.cube import Cube
from ..utils.types import is_real_or_1darray
from ase.units import Bohr


class Dyson(Transition):
    def __init__(
        self,
        cube: Cube | str | os.PathLike = None,
        slice_height: int = 1.5,
        center_mass: bool = True,
        axis: int = 2,
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
        axis : int, optional
            Axis to be considered z-axis, default 2.

        Raises
        ------
        TypeError | FileNotFoundError
            If `cube` is neither instance of Cube class, nor path to a cube file.
        """

        self.grid = None
        """Axis grid of cube."""

        self.spacing = None
        """Spacing of cube data points in Angstrom."""

        self.steps = None
        """Stepsize between each points in Angstrom"""

        self.amplitude = None
        """Amplitude (magnitude squared) of Dyson orbital."""

        self.x = None
        """x values of cube data points."""

        self.y = None
        """y values of cube data points."""

        self.wf_slice = None
        """Slice of cube used for extrapolation into vacuum."""

        self.slice_height = slice_height
        """Height in Angstrom at which to extract a slice of the cube, by default 1.5 Angstrom."""

        cube = super().file_to_cube(cube)
        self.load_cube(cube, slice_height, center_mass, axis)

    def load_cube(
        self,
        cube: Cube,
        slice_height: int = 1.5,
        center_mass: bool = True,
        axis: int = 2,
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
        axis : int, optional
            Axis to be considered z-axis, default 2.
        """

        if not isinstance(slice_height, Real):
            raise TypeError(
                f"slice_height must be of type float, but got {type(slice_height)}"
            )

        if not isinstance(axis, int) or axis not in [0, 1, 2]:
            raise TypeError(
                f"axis must be of type int and one of [0, 1, 2], but got {axis}"
            )

        super().parse_cube_dimensions(cube=cube, center_mass=center_mass, axis=axis)

        self.grid = self.grid[:2]
        self.spacing = self.spacing[:2]
        self.steps = self.steps[:2]

        self.x, self.y = self.grid

        self.amplitude = cube.magsqr

        # switch to cube box coordinate system
        if center_mass:
            distance = slice_height + cube.center_of_mass[axis]
        else:
            distance = slice_height - cube.origin[axis]

        self.wf_slice = cube.get_slice_data(distance, axis=axis) / Bohr**1.5

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of wf_slice."""
        return self.wf_slice.shape

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

        if not isinstance(kappa, np.ndarray):
            raise TypeError(f"kappa must be Real or np.ndarray, but got {type(kappa)}")

        if (kappa <= 0).any():
            raise ValueError("kappa must be positive.")

        if kappa.ndim == 0:
            kappa = np.array([kappa])

        if not isinstance(pad, int):
            raise TypeError(
                f"pad must be of type int and non-negative, but got {type(pad)}"
            )

        if pad < 0:
            raise ValueError(f"pad must be of type int and non-negative, but got {pad}")

        padding = ((pad, pad), (pad, pad))
        wf_slice = np.pad(self.wf_slice, padding)

        height = is_real_or_1darray(height, "height")
        dz = abs(height - self.slice_height)

        # =====================================================================
        # The following code snipped was adapted from:
        # https://github.com/nanotech-empa/cp2k-spm-tools/blob/main/cp2k_spm_tools/cp2k_grid_orbitals.py
        #
        fourier = np.fft.rfft2(wf_slice)
        kx_arr = 2 * np.pi * np.fft.fftfreq(wf_slice.shape[0], self.steps[0])
        ky_arr = 2 * np.pi * np.fft.rfftfreq(wf_slice.shape[1], self.steps[1])

        kx2 = kx_arr[:, None] ** 2
        ky2 = ky_arr[None, :] ** 2

        kappa_xy = np.sqrt(kx2 + ky2 + kappa[..., None, None] ** 2)
        decay = np.exp(-np.multiply.outer(dz, kappa_xy))

        wf_extrapolated = np.fft.irfft2(fourier * decay, wf_slice.shape)
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
