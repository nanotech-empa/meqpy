from pymatgen.io.common import VolumetricData
from pymatgen.core.structure import Molecule
import os


class Cube:
    """
    Class to represent a gaussian cube file. It requires the filename of the cube to be read.
    """

    def __init__(self, filename: os.PathLike):
        self.cube_data = VolumetricData.from_cube(filename)
        self.data = self.cube_data["total"]
        # Molecule Object from PyMatgen, which can be used to calculate the center of mass and other properties of the structure.
        self.molecule = Molecule(
            self.cube_data.structure.species, self.cube_data.structure.cart_coords
        )

    @property
    def cart_coords(self):
        """Returns the cartesian coordinates of the cube data."""
        return self.cube_data.structure.cart_coords

    @property
    def elements(self):
        """Returns a list of the elements in the cube data."""
        return [i.symbol for i in self.cube_data.structure.elements]

    @property
    def masses(self):
        """Returns a list of the masses of the elements in the cube data."""
        return [i.atomic_mass for i in self.cube_data.structure.elements]

    @property
    def structure(self):
        """Returns the Pymatgen object Structure of the cube data."""
        return self.cube_data.structure

    @property
    def molecule(self):
        """Returns the Pymatgen object Molecule of the cube data.
        Note: Molecule object has no periodicity. Useful for calculating molecule properties like center of mass, it will contain the coordinates and other properties of the atoms in the cube data."""
        return self.molecule

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
