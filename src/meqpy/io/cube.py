from pymatgen.io.common import VolumetricData
from pymatgen.core.structure import Molecule
import os


class Cube:
    """
    Class to represent a gaussian cube file. Prefer Cube.from_file() for normal use
    """

    def __init__(self, cube_data: VolumetricData):
        self.cube_data = cube_data
        self.data = self.cube_data.data["total"]
        # Molecule Object from PyMatgen, which can be used to calculate the center of mass and other properties of the structure.
        self.molecule = Molecule(
            self.cube_data.structure.species, self.cube_data.structure.cart_coords
        )

    def __repr__(self):
        return f"Cube(atoms={len(self.molecule)}, grid={self.data.shape}, "

    @classmethod
    def from_file(cls, filename: os.PathLike) -> "Cube":
        """
        Read a Gaussian cube file and return a Cube instance.

        Parameters
        ----------
        filename : path-like
            Path to the .cube file.
        """
        cube_data = VolumetricData.from_cube(filename)
        return cls(cube_data)

    @property
    def cart_coords(self):
        """Returns the cartesian coordinates of the cube data."""
        return self.cube_data.structure.cart_coords

    @property
    def elements(self):
        """Returns a list of the elements in the cube data."""
        return [i.symbol for i in self.cube_data.structure.species]

    @property
    def masses(self):
        """Returns a list of the masses of the elements in the cube data."""
        return [i.atomic_mass for i in self.cube_data.structure.species]

    @property
    def structure(self):
        """Returns the Pymatgen object Structure of the cube data."""
        return self.cube_data.structure

    @property
    def molecule(self):
        """Returns the Pymatgen object Molecule of the cube data.
        Note: Molecule object has no periodicity. Useful for calculating molecule properties like center of mass, it will contain the coordinates and other properties of the atoms in the cube data."""
        return self._molecule

    @molecule.setter
    def molecule(self, value):
        """Sets the molecule property of the cube data."""
        self._molecule = value

    @property
    def center_of_mass(self):
        """Returns the center of mass of the Structure:"""
        return self.molecule.center_of_mass

    def linear_slice(self, p1, p2, n=100):
        """
        Returns a linear slice of the cube data between two points p1 and p2. The number of points in the slice can be specified with n.
        p1 and p2 list of fractional coordinates.
        """
        return self.cube_data.get_linear_slice(p1, p2, n)

    def get_average_along_axis(self, axis=0):
        """
        Returns the average of the cube data along a specified axis. The default is 0, which corresponds to the x-axis.
        """
        return self.cube_data.get_average_along_axis(axis)

    def get_axis_grid(self, axis=0):
        """
        Returns the grid points along a specified axis. The default is 0, which corresponds to the x-axis.
        """
        return self.cube_data.get_axis_grid(axis)

    def dim(self):
        """Returns the dimensions of the cube data."""
        return self.cube_data.dim
