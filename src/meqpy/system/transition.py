import os
import numpy as np
from ..io.cube import Cube


class Transition:
    def __init__(
        self, cube: Cube | str | os.PathLike, center_mass: bool = True, axis: int = 2
    ):
        """Initialize Transition instance for transitions within Molecule.

        Parameters
        ----------
        cube : Cube | str | os.PathLike, optional
            Cube object, or path to cube file, from which to create Transition instance.
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

        cube = self.file_to_cube(cube)
        self.parse_cube_dimensions(cube, center_mass, axis)

    @staticmethod
    def file_to_cube(cube: Cube | str | os.PathLike) -> Cube:
        """Check if input is Cube instance or path to cube file and return Cube instance.

        Parameters
        ----------
        cube : Cube | str | os.PathLike
            Cube object, or path to cube file, from which to create Cube object.

        Returns
        -------
        Cube
            Cube object instance.

        Raises
        ------
        FileNotFoundError
            If `cube` is neither instance of Cube class, nor path to a cube file.
        """
        if not isinstance(cube, Cube):
            if os.path.isfile(cube):
                cube = Cube(cube)
            else:
                raise FileNotFoundError(
                    f"cube must be of type Cube or path to cube file, but got {cube}"
                )

        return cube

    def parse_cube_dimensions(
        self,
        cube: Cube,
        center_mass: bool = True,
        axis: int = 2,
    ):
        """Get axis grid and point spacing from Cube instance.

        Parameters
        ----------
        cube : Cube
            Cube object from which to create Transition instance.
        center_mass : bool, optional
            Shift origin of coordinate system to molecule's center of mass, default is True.
        axis : int, optional
            Axis is to be considered as z-axis, default is 2.
        """

        if not isinstance(cube, Cube):
            raise TypeError(f"cube must be of type Cube, but got {type(cube)}")

        if not isinstance(axis, int) or axis not in [0, 1, 2]:
            raise TypeError(
                f"axis must be of type int and one of [0, 1, 2], but got {axis}"
            )

        axes = [0, 1, 2]
        axes.pop(axis)
        self.axes = axes + [axis]

        self.grid = []
        for axis in self.axes:
            grid = np.array(cube.get_axis_grid(axis))

            # switch from cube box coordinate system
            if center_mass:
                grid -= cube.center_of_mass[axis]
            else:
                grid += cube.origin[axis]

            self.grid.append(grid)

        self.spacing = cube.spacing[axes]

    @property
    def steps(self):
        """Stepsize between each points in Angstrom"""
        return np.linalg.norm(self.spacing, axis=1)
