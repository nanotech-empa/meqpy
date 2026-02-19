import os
from numbers import Real
import numpy as np
from ..io.cube import Cube
from ..utils.types import is_real_or_1darray


class Dyson:
    def __init__(
        self,
        cube: Cube | str | os.PathLike = None,
        amplitude: float = None,
        cut_off_height: int = 1.5,
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
        amplitude : float, optional
            Amplitude of charged transition. If None (default) the it is given by the integral of the cube data squared.
        cut_off_height : int, optional
            Height in Angstrom at which to extra a slice of the cube, by default 1.5 Angstrom above center of molecule. See also `center_mass` parameter.
        center_mass : bool, optional
            If `True´ (default): Consider `cut_off_height` to be relative to center of mass of molecule.
            If `False`: `cut_off_height` is in absolute coordinates.
        axis : int, optional
            Select orientation of slice, perpendicular to `axis`, by default 2.

        Raises
        ------
        TypeError | FileNotFoundError
            If `cube` is neither instance of Cube class, nor path to a cube file.
        """

        self.filename = None
        """Filename of cube."""

        self.WF_slice = None
        """Slice of cube used for extrapolation into vacuum."""

        self.cut_off_height_rel = None
        """z value of slice, relative to center of mass of molecule, in Angstrom"""
        self.cut_off_height_abs = None
        """Absolute z value of slice, in Angstrom"""

        self.x = None
        """x values of cube data points."""
        self.y = None
        """y values of cube data points."""

        self.dx = None
        """x spacing of cube data points."""
        self.dy = None
        """y spacing of cube data points."""

        if isinstance(cube, Cube):
            self.load_cube(cube, amplitude, cut_off_height, center_mass, axis)
            return None

        try:
            if os.path.isfile(cube):
                cube = Cube(cube)
                self.load_cube(cube, amplitude, cut_off_height, center_mass, axis)
            else:
                raise FileNotFoundError(
                    f"cube must be of type Cube or path to cube file, but got {cube}"
                )
        except TypeError:  # will not be triggered anyway
            raise TypeError(
                f"cube must be of type Cube or path to cube file, but got {type(cube)}"
            )

    def load_cube(
        self,
        cube: Cube,
        amplitude: float = None,
        cut_off_height: int = 1.5,
        center_mass: bool = True,
        axis: int = 2,
    ):
        """Load cube data and extract slice for extrapolation into vacuum.

        Parameters
        ----------
        cube : Cube
            Cube object from which to create Dyson instance.
        amplitude : float, optional
            Amplitude of charged transition. If None (default) the it is given by the integral of the cube data squared.
        cut_off_height : int, optional
            Height in Angstrom at which to extra a slice of the cube, by default 1.5 Angstrom above center of molecule. See also `center_mass` parameter.
        center_mass : bool, optional
            If `True´ (default): Consider `cut_off_height` to be relative to center of mass of molecule.
            If `False`: `cut_off_height` is in absolute coordinates.
        axis : int, optional
            Select orientation of slice, perpendicular to `axis`, by default 2.
        """

        if not isinstance(cube, Cube):
            raise TypeError(f"cube must be of type Cube, but got {type(cube)}")

        if not isinstance(amplitude, (type(None), Real)):
            raise TypeError(
                f"amplitude must be of type float or None, but got {type(amplitude)}"
            )

        if not isinstance(cut_off_height, Real):
            raise TypeError(
                f"cut_off_height must be of type float, but got {type(cut_off_height)}"
            )

        if not isinstance(center_mass, bool):
            raise TypeError(
                f"center_mass must be of type bool, but got {type(center_mass)}"
            )

        if not isinstance(axis, int) or axis not in [0, 1, 2]:
            raise TypeError(
                f"axis must be of type int and one of [0, 1, 2], but got {axis}"
            )

        if isinstance(cube.filename, str):
            self.filename = cube.filename

        if center_mass:
            self.cut_off_height_abs = cut_off_height + cube.center_of_mass[axis]
            self.cut_off_height_rel = cut_off_height
        else:
            self.cut_off_height_abs = cut_off_height
            self.cut_off_height_rel = cut_off_height - cube.center_of_mass[axis]

        if type(amplitude) is type(None):
            self.amplitude = cube.magsqr
        else:
            self.amplitude = amplitude

        self.axis = axis
        self.wf_slice = cube.integration_plane(self.cut_off_height_abs, axis=axis)

        axes = [0, 1, 2]
        axes.pop(axis)
        self.x = cube._cartesian_axis(axes[0])
        self.y = cube._cartesian_axis(axes[1])

        self.dx = np.linalg.norm(cube.vecs[axes[0]])
        self.dy = np.linalg.norm(cube.vecs[axes[1]])

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of wf_slice."""
        return self.wf_slice.shape

    def extrapolate_wavefunction(
        self,
        z: float,
        kappa: float | np.ndarray,
        pad: int = 0,
        center_mass: bool = True,
    ) -> np.ndarray:
        """Extrapolate wavefunction exponentially from cut_off_height till z.

        Parameters
        ----------
        z : (K,) float
            Height in Angstrom at which to extrapolate wavefunction.
        kappa : float | (any) np.ndarray
            Decay constant(s) in 1/Angstrom.
        pad : int, optional
            Padding in pixels around the wavefunction slice, by default 0
        center_mass : bool, optional
            If `True` (default): Consider `z` to be relative to center of mass of molecule.
            If `False`: `z` is in absolute coordinates.

        Returns
        -------
        wf_extrapolated : (nx, ny, K, any) np.ndarray
            Extrapolated wavefunction, with first two dimensions corresponding to x and y positions,
            the third dimension to different z values, and the remaining are given by the shape of `kappa`.

        Raises
        ------
        TypeError
            If any of the input parameters are of incorrect type.
        ValueError
            If `kappa` is not positive, or if `pad` is negative.
        """
        padding = ((pad, pad), (pad, pad))
        wf_slice = np.pad(self.wf_slice, padding)

        z = is_real_or_1darray(z, "z")

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

        if not isinstance(center_mass, bool):
            raise TypeError(
                f"center_mass must be of type bool, but got {type(center_mass)}"
            )

        if center_mass:
            dz = abs(z - self.cut_off_height_rel)
        else:
            dz = abs(z - self.cut_off_height_abs)

        # =====================================================================
        # The following code snipped was adapted from:
        # https://github.com/nanotech-empa/cp2k-spm-tools/blob/main/cp2k_spm_tools/cp2k_grid_orbitals.py
        #
        fourier = np.fft.rfft2(wf_slice)
        kx_arr = 2 * np.pi * np.fft.fftfreq(wf_slice.shape[0], self.dx)
        ky_arr = 2 * np.pi * np.fft.rfftfreq(wf_slice.shape[1], self.dy)
        kx_grid, ky_grid = np.meshgrid(kx_arr, ky_arr, indexing="ij")

        kappa_xy = np.sqrt(kx_grid**2 + ky_grid**2 + kappa[..., None, None] ** 2)
        decay = np.exp(-np.multiply.outer(dz, kappa_xy))

        wf_extrapolated = np.fft.irfft2(fourier * decay, wf_slice.shape)
        #
        # =====================================================================

        # sort x,y axis to the front
        return np.moveaxis(wf_extrapolated, [-2, -1], [0, 1])

    def coupling_strength(
        self,
        z: float,
        kappa: float | np.ndarray,
        pad: int = 0,
        center_mass: bool = True,
        squeeze: bool = True,
    ) -> np.ndarray:
        """Get coupling strength between molecule wavefunction and tip, by extrapolating the wavefunction to the tip position and squaring it.

        Parameters
        ----------
        z : (K,) float
            Height in Angstrom at which to extrapolate wavefunction.
        kappa : float | (any) np.ndarray
            Decay constant(s) in 1/Angstrom.
        pad : int, optional
            Padding in pixels around the wavefunction slice, by default 0
        center_mass : bool, optional
             If `True` (default): Consider `z` to be relative to center of mass of molecule.
             If `False`: `z` is in absolute coordinates.
        squeeze : bool, optional
            If `True` (default), squeeze the output array to remove singleton dimensions.

        Returns
        -------
        coupling_strength_matrix : (K, any) np.ndarray
            Coupling strength map, with first two dimensions corresponding to x and y positions,
            the third dimension to different z values, and the remaining are given by the shape of `kappa`.

        Notes
        -----
        This function is a wrapper and returns the square of `Dyson.extrapolate_wavefunction()`.
        """
        wf_map2 = (
            self.extrapolate_wavefunction(
                z=z, kappa=kappa, pad=pad, center_mass=center_mass
            )
            ** 2
        )

        if squeeze:
            wf_map2 = np.squeeze(wf_map2)

        return wf_map2
