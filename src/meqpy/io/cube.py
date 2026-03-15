from ase.io.cube import read_cube
import numpy as np
import os


class Cube:
    """
    Class to represent a gaussian cube file. Prefer Cube.from_file() for normal use
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
    def cart_coords(self):
        """Returns the cartesian coordinates of the cube data."""
        return self.atoms.positions - self.origin

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
        """Returns the ASE Atoms object."""
        return self._atoms

    @atoms.setter
    def atoms(self, value):
        """Sets the atoms property of the cube data."""
        self._atoms = value

    @property
    def center_of_mass(self):
        """Returns the center of mass of the Structure:"""
        return self.atoms.get_center_of_mass() - self.origin

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

    def get_slide_data(self, axis: int = 2, distance: float = 0.5):
        """
        Return a 2D slice of the volumetric cube data along a given axis.

        Parameters
        ----------
        axis : int Axis normal to the plane of the slice (0 for x, 1 for y, 2 for z).
        distance : float Position along the axis.
        """
        if axis not in [0, 1, 2]:
            raise ValueError("Axis must be 0, 1, or 2.")
        length = self.atoms.cell.cellpar()[:3][axis]

        if not (0 <= distance <= length):
            raise ValueError(
                f"Distance must be between 0 and {length} along the specified axis."
            )

        # Find the closest index in the grid to the specified distance
        grid = self.get_axis_grid(axis)
        plane_index = np.searchsorted(grid, distance)

        return np.take(self.data, plane_index, axis=axis)

    def dim(self):
        """Returns the dimensions of the cube data."""
        return self.data.shape
