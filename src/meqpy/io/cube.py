from ase.io.cube import read_cube
import numpy as np
import os
from ..utils.constants import bohr


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
        voxel_size = np.prod(spacings) / bohr**3
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
