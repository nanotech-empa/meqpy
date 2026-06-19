import numpy as np
from ase.atoms import Atoms
from ase.units import Bohr
from .cube import Cube
from ..utils.types import validate_real_or_1darray, validate_nonnegative_float
from numbers import Real


# cube from TB vectors
class Cube_2pz(Cube):
    """Class to create Cube object from Tight-Binding like calculations considering 2pz orbitals."""

    def __init__(
        self,
        positions: np.ndarray,
        eigenvector: np.ndarray,
        boundary: float = 5.0,
        spacing: float = 0.333333 * Bohr,
        elements: np.ndarray = None,
        fill_cube: bool = False,
    ):
        """Create Cube instance from Tight-Binding based calculations for 2pz orbitals.

        Parameters
        ----------
        positions : (N,3) np.ndarray
            Positions of hopping sites.
        eigenvector : (N,) np.ndarray
            Wavefunction in basis of hopping sites.
        boundary : float, optional
            Boundary in Angstrom around atoms to determine cube size, by default 5.0
        spacing : float, optional
            Step size of cube grid, by default 0.333333*Bohr
        elements : np.ndarray, optional
            List of elements of hopping sites, defaults to carbon.
        fill_cube : bool, optional
            Calculate wavefunction in cube for all points, by default False
        """

        self.set_positions_and_eigenvector(positions, eigenvector)
        self.set_volume(positions, boundary, spacing)
        self.set_atoms(positions, elements)

        if fill_cube:
            self.fill_cube()

    def set_positions_and_eigenvector(self, positions, eigenvector):
        """Add positions and eigenvector to the instance."""

        self.positions = np.array(positions)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"Positions must be np.ndarray of shape (N,3), "
                f"but got object with shape {positions.shape}"
            )

        self.eigenvector = validate_real_or_1darray(eigenvector, "eigenvector")
        if len(eigenvector) != len(positions):
            raise IndexError(
                f"Positions and eigenvector must be of same length, "
                f"but got shapes {positions.shape} and {eigenvector.shape}."
            )

    def set_volume(self, positions, boundary, spacing):
        """Create empty cube volume and add to attributes.

        Parameters
        ----------
        positions : np.ndarray
            Positions of hopping sites.
        boundary : float
            Boundary in Angstrom around atoms to determine cube size.
        spacing : float
            Step size of cube grid.
        """

        if not isinstance(boundary, Real):
            raise TypeError(
                f"Boundary must be real but got type {type(boundary).__name__}"
            )

        self.boundary = boundary

        self.spacing = np.eye(3) * validate_nonnegative_float(spacing, "spacing")

        # find origin of volume
        self.origin = np.min(positions, axis=0) - boundary

        # calculate number of points for all axes
        maxs = np.max(positions, axis=0) + boundary
        size = maxs - self.origin
        num_pnts = size / np.diag(self.spacing)
        shape = [int(round(n)) for n in num_pnts]

        # make empty volume
        self.data = np.zeros(shape)

    def set_atoms(self, positions: np.ndarray, elements: np.ndarray):
        """Create ASE Atoms object for cube and absolute coordinate system.

        Parameters
        ----------
        positions : np.ndarray
            Positions of hopping sites.
        elements : np.ndarray
            List of elements of hopping sites, defaults to carbon.
        """
        if elements is None:
            elements = np.zeros(positions.shape[0])
            elements.fill(6)

        # save atoms in original coordinates
        cell = np.diag(self.data.shape) * self.spacing
        self.original_atoms = Atoms(symbols=elements, positions=positions, cell=cell)

        # Shift the atomic positions to be relative to the origin of the cube
        shifted_atoms = self.original_atoms.copy()
        shifted_atoms.positions -= self.origin
        self.atoms = shifted_atoms

    @property
    def magsqr(self) -> float:
        """Returns the magnitude squared of the cube data."""
        return np.sum(self.eigenvector**2)

    def fill_cube(self):
        """Fill cube data with wavefunction values at each grid point."""
        XX, YY, ZZ = np.meshgrid(
            self.get_axis_grid(0),
            self.get_axis_grid(1),
            self.get_axis_grid(2),
            indexing="ij",
        )

        self.data.fill(0.0)
        for atom, coeff in zip(self.atoms, self.eigenvector):
            self.data += self.orbital_2pz(XX, YY, ZZ, atom) * coeff

    def orbital_2pz(self, x: float, y: float, z: float, atom: Atoms) -> np.ndarray:
        """Calculate the 2pz orbital contribution of an atom at given coordinates.

        Parameters
        ----------
        x : float or np.ndarray
            x coordinate grid
        y : float or np.ndarray
            y coordinate grid
        z : float or np.ndarray
            z coordinate grid
        atom : Atoms
            Atom for which to calculate the orbital contribution.

        Returns
        -------
        np.ndarray
            Orbital contribution at given coordinates.
        """

        x0, y0, z0 = atom.position
        xdiff = x - x0
        ydiff = y - y0
        zdiff = z - z0
        rdiff_abs = np.sqrt(xdiff**2 + ydiff**2 + zdiff**2)

        # effective nuclear charge
        Zeff = atom.number
        Zeff -= 2 * 0.85
        Zeff -= (atom.number - 3) * 0.35

        # decay constant
        alpha = Zeff / 2.0 / Bohr

        prefactor = np.sqrt(alpha**5 / np.pi) * Bohr**1.5

        return zdiff * np.exp(-alpha * rdiff_abs) * prefactor

    def get_slice_data(
        self,
        distance: float,
        **kwargs,
    ) -> np.ndarray:
        """
        Return a 2D slice of the volumetric cube data along a given axis.

        Parameters
        ----------
        distance : float
            Position along the z-axis at which to take the slice.

        Returns
        -------
        np.ndarray
            2D slice of the wavefunction at the specified distance.
        """
        XX, YY = np.meshgrid(
            self.get_axis_grid(0), self.get_axis_grid(1), indexing="ij"
        )
        ZZ = np.ones_like(XX) * distance

        wf_slice = np.zeros_like(XX)
        for atom, coeff in zip(self.atoms, self.eigenvector):
            wf_slice += self.orbital_2pz(XX, YY, ZZ, atom) * coeff

        return wf_slice
