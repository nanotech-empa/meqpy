import os
from numbers import Real

import numpy as np
from scipy.signal import fftconvolve
import scipy.constants as const

from .transition import Transition
from ..io.cube import Cube
from ..utils.types import validate_nonnegative_int, require_type
from ..utils.constants import BOHR, ELEMENTARY_CHARGE

EPSILON_0 = const.epsilon_0 * 1e-10  # As/Vm * m/Å = As/VÅ


class OpticalTransition(Transition):
    """
    Class for calculating coupling strength to tip plasmon from transition density cube.
    """

    def __init__(
        self,
        cube: Cube | str | os.PathLike,
        center_mass: bool = True,
    ):
        """The volumetric data of the corresponding transition density orbital is used to
        calculate tip position dependent coupling to the junction plasmon using
        a simple two-point-charge model.

        Parameters
        ----------
        cube : Cube | str | os.PathLike
            Cube object, or path to cube file containing transition density orbital.
        center_mass : bool, optional
            Shift origin of coordinate system to molecule's center of mass, default is True.

        Raises
        ------
        TypeError | FileNotFoundError
            If `cube` is neither instance of Cube class, nor path to a cube file.
        """

        cube = super().file_to_cube(cube)
        super().parse_cube_dimensions(cube=cube, center_mass=center_mass)
        self.data = cube.data / BOHR**1.5

        self.voxel_size = cube.voxel_size
        """Volume of a voxel in Å³"""

    def _kernel_mesh(self, pad: int = 0):
        """Create Meshgrid of shape ``(2*nx'+1, 2*ny'+1, nz)``
        with x,y axis being symmetric around 0, with ni' = ni + 2*pad."""

        nz = self.shape[2]
        num_pts = [range(-n - pad, n + pad + 1) for n in self.shape[:2]] + [range(nz)]
        mesh_indices = np.meshgrid(*num_pts, indexing="ij")
        index_to_cartesian_matrix = self.spacing.T

        mesh = np.einsum("ij,j...->...i", index_to_cartesian_matrix, mesh_indices)
        mesh[..., 2] += self.origin[2]

        return np.moveaxis(mesh, -1, 0)  # make X, Y(, Z) first dimension

    @staticmethod
    def _coulomb_potential(
        mesh: np.ndarray,
        z_pointcharge: float,
    ):
        """Fill meshgrid with Coulomb potential of elementary charge located at position (0, 0, z_pointcharge)."""

        distance = np.sqrt(mesh[0] ** 2 + mesh[1] ** 2 + (mesh[2] - z_pointcharge) ** 2)

        too_close = distance < 1.0
        safe_distance = np.where(too_close, 1.0, distance)
        COULOMB_FAC = ELEMENTARY_CHARGE / 4 / np.pi / EPSILON_0  # V/Å
        return np.where(too_close, 0.0, COULOMB_FAC / safe_distance)

    @classmethod
    def _two_point_charges(
        cls,
        mesh: np.ndarray,
        mirror_plane: float,
        z_pointcharge: float,
    ):
        """Fill meshgrid with Coulomb potential of two elementary charge located
        at position (0, 0, z) and (0, 0, 2 * mirror_plane - z)."""

        z_up = z_pointcharge
        z_down = 2 * mirror_plane - z_pointcharge
        potential = cls._coulomb_potential(mesh, z_up)
        potential -= cls._coulomb_potential(mesh, z_down)

        return potential

    def plasmon_coupling(self, mirror_plane: float, z_pointcharge: float, pad: int = 0):
        """Calculate coupling between transition density and junction plasmon,
        using two point charges to model the latter.

        Parameters
        ----------
        mirror_plane : float
            z height of point charge mirror plane, i.e. substrate.
        z_pointcharge : float
            z height of point charge, i.e. center of tip.
        pad : int, optional
            Padding the volumetric data in the xy-plane, by default 0.

        Returns
        -------
        plasmon_coupling : np.ndarray
            2D array containing plasmon coupling strength to transition density for each point in grid.
        """

        require_type(mirror_plane, Real, "mirror_plane")
        require_type(z_pointcharge, Real, "z_pointcharge")
        validate_nonnegative_int(pad, "pad")

        kernel_mesh = self._kernel_mesh(pad)
        potential = self._two_point_charges(kernel_mesh, mirror_plane, z_pointcharge)

        padding = ((pad, pad), (pad, pad), (0, 0))
        data = np.pad(self.data, padding)
        plasmon_coupling = np.sum(
            fftconvolve(data, potential, mode="same", axes=(0, 1)), axis=2
        )
        plasmon_coupling *= self.voxel_size

        return plasmon_coupling

    def emission_strength(
        self, mirror_plane, z_pointcharge, pad: int = 0, normalize: bool = True
    ):
        """Calculation the plasmon induced emission strength, using two-point-charge model.

        Parameters
        ----------
        mirror_plane : float
            z height of point charge mirror plane, i.e. substrate.
        z_pointcharge : float
            z height of point charge, i.e. center of tip.
        pad : int, optional
            Padding the volumetric data in the xy-plane, by default 0.
        normalize : bool, optional
            If True normalize output to 1.0, by default True

        Returns
        -------
        emission_strength : np.ndarray
            2D array containing plasmon induced emission strength for each point in grid.
        """

        require_type(normalize, bool, "normalize")

        plasmon_coupling = self.plasmon_coupling(mirror_plane, z_pointcharge, pad)
        emission_strength = plasmon_coupling**2

        if normalize and emission_strength.max() > 0:
            emission_strength /= emission_strength.max()

        return emission_strength
