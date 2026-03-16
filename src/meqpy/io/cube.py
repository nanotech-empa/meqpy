from ase.io.cube import read_cube
from ase.atoms import Atoms
import numpy as np
import os
import scipy.constants as const

BOHR2ANG = const.physical_constants["Bohr radius"][0] * 1e10  # Angstrom


class Cube:
    """
    Class to represent a gaussian cube file.
    """

    def __init__(self, filename: os.PathLike):
        cube_data = read_cube(file_obj=open(filename, "r"))
        self.data = cube_data["data"]
        self.atoms = cube_data["atoms"]
        self.origin = cube_data["origin"]
        self.spacing = cube_data["spacing"]

    def __repr__(self):
        return f"Cube(atoms={len(self.atoms)}, grid={self.data.shape}, "

    @property
    def cart_coords(self) -> np.ndarray:
        """Returns the cartesian coordinates of the atoms in cube data."""
        return self.atoms.positions

    @property
    def elements(self) -> list:
        """Returns a list of the elements of the atoms in cube data."""
        return self.atoms.get_chemical_symbols()

    @property
    def masses(self) -> np.ndarray:
        """Returns the masses of the atoms in cube data."""
        return self.atoms.get_masses()

    @property
    def atoms(self) -> Atoms:
        """Returns the ASE Atoms object."""
        return self._atoms

    @atoms.setter
    def atoms(self, value):
        """Sets the atoms property of the cube data."""
        if isinstance(value, Atoms):
            self._atoms = value
        else:
            raise TypeError(f"Value must be ASE Atoms object, but got {type(value)}")

    @property
    def center_of_mass(self) -> np.ndarray:
        """Returns the center of mass of the Structure:"""
        return self.atoms.get_center_of_mass()

    def get_axis_grid(self, axis) -> np.ndarray:
        """Get the grid for a particular axis.

        Args:
            ind (int): Axis index.
        """
        if axis not in [0, 1, 2]:
            raise ValueError("Axis must be 0, 1, or 2.")
        spacing = self.spacing[axis, axis]
        lengths = self.atoms.cell.cellpar()[axis]
        return np.arange(0, lengths, spacing) + self.origin[axis]

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

        grid = self.get_axis_grid(axis)
        if not (grid[0] <= distance <= grid[-1]):
            raise ValueError(
                f"Distance must be between {grid[0]} and {grid[-1]} along axis {axis}."
            )

        # Find the closest index in the grid to the specified distance
        plane_index = np.argmin(np.abs(grid - distance))

        return np.take(self.data, plane_index, axis=axis)

    @property
    def magsqr(self) -> float:
        voxel_size = np.prod(np.linalg.norm(self.spacing, axis=0))
        voxel_size /= BOHR2ANG**3
        return np.sum(self.data**2) * voxel_size
