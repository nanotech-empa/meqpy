from ase.io.cube import read_cube
import numpy as np
import os
from ase.units import Bohr, _hbar, _me, _e


def _pad_lin_extrapolate(vector: np.ndarray, pad_width: tuple, iaxis: int, kwargs):
    """Function for `np.pad()` to extrapolate linearly (see numpy docs).
    EXAMPLE: `x_padded = np.pad(x, pad_width, pad_lin_extrapolate)`
    """
    dd = vector[pad_width[0] + 1] - vector[pad_width[0]]
    vector[: pad_width[0]] = (
        vector[pad_width[0]] - (pad_width[0] - np.arange(pad_width[0])) * dd
    )
    vector[-pad_width[1] :] = (
        vector[-pad_width[1] - 1] + (np.arange(pad_width[1]) + 1) * dd
    )


class Cube:
    """
    Class to represent a gaussian cube file.
    """

    def __init__(self, filename: os.PathLike):
        cube_data = read_cube(file_obj=open(filename, "r"))
        self.data = cube_data["data"]
        self.original_atoms = cube_data["atoms"]
        self.origin = cube_data["origin"]
        self.spacing = cube_data["spacing"]

        # Shift the atomic positions to be relative to the origin of the cube
        shifted_atoms = self.original_atoms.copy()
        shifted_atoms.positions -= self.origin
        self.atoms = shifted_atoms

    def __repr__(self):
        return f"Cube(atoms={len(self.atoms)}, grid={self.data.shape}, "

    @property
    def cart_coords(self):
        """Returns the cartesian coordinates inside the cell."""
        return self.atoms.positions

    @property
    def elements(self):
        """Returns a list of the elements in the cube data."""
        return self.atoms.get_chemical_symbols()

    @property
    def masses(self):
        """Returns a list of the masses of the elements in the cube data."""
        return self.atoms.get_masses()

    @property
    def atoms(self):
        """Returns the ASE Atoms object inside the cell."""
        return self._atoms

    @atoms.setter
    def atoms(self, value):
        """The setter: allows assignment to self.atoms."""
        # You can even add validation here if you want
        self._atoms = value

    @property
    def center_of_mass(self):
        """Returns the center of mass of the Structure:"""
        return self.atoms.get_center_of_mass()

    def get_axis_grid(self, axis=0):
        """Get the grid for a particular axis.

        Args:
            ind (int): Axis index.
        """
        if axis not in [0, 1, 2]:
            raise ValueError("Axis must be 0, 1, or 2.")
        ng = self.data.shape
        num_pts = ng[axis]
        lengths = self.atoms.cell.cellpar()[:3]
        return [i / num_pts * lengths[axis] for i in range(num_pts)]

    @property
    def magsqr(self):
        """Returns the magnitude squared of the cube data."""
        spacings = np.linalg.norm(self.spacing, axis=1)
        voxel_size = np.prod(spacings) / Bohr**3
        return np.sum(self.data**2) * voxel_size

    def get_slice_data(self, distance: float, axis: int = 2) -> np.ndarray:
        """
        Return a 2D slice of the volumetric cube data along a given axis.

        Parameters
        ----------
        distance : float Position along the axis.
        axis : int Axis normal to the plane of the slice (0 for x, 1 for y, 2 for z).
        """
        if axis not in [0, 1, 2]:
            raise ValueError("Axis must be 0, 1, or 2.")
        length = self.atoms.cell.cellpar()[:3][axis]

        if not (0 <= distance <= length):
            raise ValueError(
                f"Distance must be between 0 and {length} along the specified axis."
            )

        # Find the closest index in the grid to the specified distance
        grid = np.array(self.get_axis_grid(axis))
        plane_index = np.argmin(np.abs(grid - distance))

        return np.take(self.data, plane_index, axis=axis)

    def extrapolate_wavefunction(
        self,
        tip_height: float = 7.0,
        cut_off_height: float = 1.5,
        workfunction: float = 5.4,
        center_mass=True,
        MO: int = None,
        pad: int = 0,
        axis: int = 2,
    ) -> np.ndarray:
        extrapolation_needed = True  # abs(tip_height) >= abs(cut_off_height)

        if center_mass:
            tip_height += self.center_of_mass[axis]
            cut_off_height += self.center_of_mass[axis]

        if extrapolation_needed:
            return self.extrapolate_WF(
                tip_height, cut_off_height, workfunction, MO=MO, pad=pad, axis=axis
            )
        else:
            return self.integration_plane(cut_off_height, MO=MO, pad=pad, axis=axis)

    def extrapolate_WF(
        self,
        z: float = 7.0,
        cut_off_height: float = 1.5,
        workfunction: float = 5.4,
        MO: int = None,
        pad: int = 0,
        axis: int = 2,
    ) -> np.ndarray:
        """Cut off wave function at height `cut_off` and extrapolate into vacuum to height `z`.

        :param z: Height in Angstrom to which wave function will be extrapolated, defaults to 7.
        :type z: float, optional
        :param cut_off_height: Height in Angstrom at which the wavefunction will be cut off, defaults to 1.5
        :type cut_off_height: float, optional
        :param workfunction: Vacuum barrier height in eV, defaults to 5.4
        :type workfunction: float, optional
        :param MO: Index of molecular orbital, in case of multi-orbital cube, defaults to None
        :type MO: int, optional
        :param pad: Pad map with with number of zeros in both axis, defaults to 0
        :type pad: int, optional
        :return: _description_
        :rtype: np.ndarray
        """
        morb_plane = self.integration_plane(cut_off_height, MO=MO, pad=pad, axis=axis)

        axes = [0, 1, 2]
        axes.pop(axis)

        # The following code snipped was adapted from:
        # https://github.com/nanotech-empa/cp2k-spm-tools/blob/main/cp2k_spm_tools/cp2k_grid_orbitals.py
        #
        fourier = np.fft.rfft2(morb_plane)
        kx_arr = (
            2
            * np.pi
            * np.fft.fftfreq(morb_plane.shape[0], np.linalg.norm(self.vecs[axes[0]]))
        )
        ky_arr = (
            2
            * np.pi
            * np.fft.rfftfreq(morb_plane.shape[1], np.linalg.norm(self.vecs[axes[1]]))
        )
        kx_grid, ky_grid = np.meshgrid(kx_arr, ky_arr, indexing="ij")

        fac = 2 * _me * _e / _hbar**2 * 1e-20
        kappa = np.sqrt(kx_grid**2 + ky_grid**2 + fac * workfunction)

        dz = abs(z - cut_off_height)
        return np.fft.irfft2(fourier * np.exp(-kappa * dz), morb_plane.shape)

    def integration_plane(
        self,
        z: float = 1.5,
        MO: int = None,
        pad: int = 0,
        axis: int = 2,
    ) -> np.ndarray:
        """Return xy slice of cube at height plane.

        :param z: Height of plane in Angstrom, defaults to 1.5
        :type z: float, optional
        :param MO: Index of molecular orbital, in case of multi-orbital cube, defaults to None
        :type MO: int, optional
        :param pad: Pad map with with number of zeros in both axis, defaults to 0
        :type pad: int, optional
        :return: 2d array
        :rtype: np.ndarray
        """
        assert self._ax_is_perp(axis), "z is not perpendicular to xy plane."
        plane_idx = np.searchsorted(
            self._cartesian_axis(axis), z
        )  # make sure this works also for reversed z-axes

        assert self._cartesian_axis(axis)[plane_idx] - z <= np.linalg.norm(
            self.vecs[axis]
        ), "Height of integration plane not within z-range of cube data."

        if self.data.ndim == 3 and MO == 0:
            MO = None
        padding = ((pad, pad), (pad, pad))

        morb = np.moveaxis(self.data, axis, 0)[plane_idx : plane_idx + 1, :, :, MO]
        return np.pad(np.squeeze(morb), padding)
