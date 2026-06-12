import os
import numpy as np
from ..io.cube import Cube
from ..utils.types import validate_nonnegative_int


class Transition:
    def __init__(self, cube: Cube | str | os.PathLike, center_mass: bool = True):
        """Initialize Transition instance for transitions within Molecule.

        Parameters
        ----------
        cube : Cube | str | os.PathLike, optional
            Cube object, or path to cube file, from which to create Transition instance.
        center_mass : bool, optional
            Shift origin of coordinate system to molecule's center of mass, default is True.

        Raises
        ------
        TypeError | FileNotFoundError
            If `cube` is neither instance of Cube class, nor path to a cube file.
        """

        self.origin = None
        """Origin of volumetric data."""

        self.spacing = None
        """Spacing of cube data points in Angstrom."""

        cube = self.file_to_cube(cube)
        self.parse_cube_dimensions(cube, center_mass)

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
    ):
        """Get axis grid and point spacing from Cube instance.

        Parameters
        ----------
        cube : Cube
            Cube object from which to create Transition instance.
        center_mass : bool, optional
            Shift origin of coordinate system to molecule's center of mass, default is True.
        """

        if not isinstance(cube, Cube):
            raise TypeError(f"cube must be of type Cube, but got {type(cube)}")

        self.spacing = cube.spacing

        self.is_z_perp_and_cartesian()

        if center_mass:
            self.origin = -cube.center_of_mass
        else:
            self.origin = cube.origin

    def is_z_perp_and_cartesian(self) -> bool:
        """Returns True if last axis of cube grid is perpendicular other axes
        and aligns with z-axis. Raises ValueError otherwise."""

        z = self.spacing[-1] / np.linalg.norm(self.spacing[-1])

        if np.abs(self.spacing[-1, -1]) != self.steps[-1]:
            raise ValueError("Last axis is not aligned with z-axis")

        for ax in self.spacing[:-1]:
            proj = z.dot(ax) / np.linalg.norm(ax)
            if abs(proj) > 1e-8:
                raise ValueError("z axis is not orthogonal to other axes.")

        return True

    @property
    def steps(self):
        """Stepsize between each points in Angstrom"""
        return np.linalg.norm(self.spacing, axis=1)

    def grid(self, pad: int = 0):
        """Axis grid of data in internal coordinates.

        Parameters
        ----------
        pad : int, optional
            Pad the grid before and after, by default 0

        Returns
        -------
        grid: list of np.ndarray
            The arrays being of shape (Ni+2pad,) for dimension i.
        """

        validate_nonnegative_int(pad, "pad")

        if not hasattr(self, "data"):
            raise AttributeError("No volumetric data in object.")

        axes_grids = []
        for i in range(self.data.ndim):
            step = self.steps[i]
            num_pts = self.data.shape[i]
            ax_grid = np.arange(-pad, num_pts + pad) * step
            axes_grids.append(ax_grid)
        return axes_grids

    def get_cart_axis(self, axis: int) -> np.ndarray:
        """Get grid in cartesian coordinates

        Parameters
        ----------
        axis : int
            Index of axis, must be 0, 1 or 2.

        Returns
        -------
        axis_grid: np.ndarray
            Cartesian coordinates for axis.

        Raises
        ------
        ValueError
            - if axis is not 0, 1 or 2.
            - if cube axes are not orthogonal
            - if cube axis is not aligned with cartesian coordinate system.
        """

        ndims = list(range(len(self.spacing)))
        if axis not in ndims:
            raise ValueError(f"axis must be in {ndims}, but got {axis}")

        # check if axis is perpendicular to rest
        ndims.pop(axis)
        for i in ndims:
            projection = np.dot(self.spacing[i], self.spacing[axis])
            projection /= self.steps[i] * self.steps[axis]
            if abs(projection) > 1e-8:
                raise ValueError(
                    f"Cube axes {i} and {axis} are not orthogonal. "
                    "Consider using methods grid() or mesh_cartesian() instead."
                )

        # check if axis is aligned with cartesian axis
        if np.abs(self.spacing[axis, axis]) != self.steps[axis]:
            raise ValueError(
                f"Cube axis {i} is not aligned with cartesian coordinate system. "
                "Consider using methods grid() or mesh_cartesian() instead."
            )

        return self.grid()[axis] + self.origin[axis]

    def mesh_cartesian(self, pad: int = 0):
        """Create meshgrid of data points in cartesian coordinates.

        Parameters
        ----------
        pad : int, optional
            Pad the grid before and after, by default 0.

        Returns
        -------
        mesh: tuple of np.ndarrays
            M dimensional array, with M being dimensionality of cube data
            and each dimension i of length (Ni + 2*pad).

        Raises
        ------
        Type Error
            If pad is not int
        ValueError
            If pad is negative
        AttributeError
            If no volumetric data is present.
        """

        validate_nonnegative_int(pad, "pad")

        if not hasattr(self, "data"):
            raise AttributeError("No volumetric data in object.")

        num_pts = [range(-pad, n + pad) for n in self.data.shape]
        mesh_indices = np.meshgrid(*num_pts, indexing="ij")

        ndim = self.data.ndim
        index_to_cartesian_matrix = self.spacing[:, :ndim].T

        mesh = np.einsum("ij,j...->...i", index_to_cartesian_matrix, mesh_indices)
        mesh += self.origin

        return np.moveaxis(mesh, -1, 0)  # make X, Y(, Z) first dimension
