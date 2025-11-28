import numpy as np
import os


from .utils import bohr2ang, hbar, m_e, q_e  # some physical constants

# --------------------------------------------------------------------------------------------------------------


class Cube:
    """Class to load Gaussian formatted cubes and simulate constant height STS maps."""

    def __init__(self, input: str | os.PathLike):
        """Initialize instance of `Cube` class

        :param input: Either path to existing cube file, or string containing header of Gaussian cube.
        :type input: str | os.PathLike
        :raises ValueError: Inputstr is neither valid file nor valid header.
        """

        self.filename = None
        """Filename of cube."""
        self.basename = None
        """Basename of `self.filename`"""
        self.data = None
        """Array containing data of cube"""
        self.natoms = None
        """Number of atoms in molecule"""
        self.atoms = None
        """List of dictionaries containing information for all atoms"""

        self.origin = None
        """Origin of cube volume"""
        self.NVal = None
        """Number of values per voxel"""
        self.n1 = None
        """Number of points in first axis"""
        self.n2 = None
        """Number of points in second axis"""
        self.n3 = None
        """Number of points in third axis"""
        self.vec1 = None
        """Length and direction of first axis"""
        self.vec2 = None
        """Length and direction of first axis"""
        self.vec3 = None
        """Length and direction of first axis"""

        self.nMO = 1
        """Number of Orbitals in cube file"""
        self.isMO = False
        """Does cube contain Molecular orbitals?"""
        self.vecMO = None
        """List of molecular orbitals"""
        self.header = None
        """Header lines of cube file"""

        try:  # assume input is path to cube file
            self.filename = input
            self.basename = self._get_basename(self.filename)
            header, pointer_position = self.load_cube_header(input)
            self.parse_cube_header(header)
            self.data = self.load_cube_data(input, pointer_position)

        except AssertionError:  # assume input is string containing a cube header
            raise ValueError("Inputstr is not a valid cube file.")

    # -----------------------------------------------------------------------------------------------------

    @property
    def content(self) -> str:
        """Load full content of *.cub file with which instance was initialized.

        :return: Content of cubefile.
        :rtype: str
        """
        with open(self.filename, "r") as cubefile:
            content = cubefile.read()
        return content

    @staticmethod
    def _get_basename(filename: str) -> str:
        return os.path.splitext(os.path.basename(filename))[0]

    @staticmethod
    def load_cube_header(filename: str | os.PathLike) -> tuple[str, int]:
        """Load header from cube file.

        :param filename: Path to cube file.
        :type filename: str | os.PathLike
        :return: Tuple containing header as string and file position where header ends.
        :rtype: tuple[str, int]
        """
        assert os.path.exists(filename), "File does not exists."

        with open(filename, "br") as f:
            header = ""

            # read first 5 lines of header: 2 comments, 1 origin, 3 cube size
            for i in range(6):
                header += f.readline().decode("utf-8")

            nAtoms = int(header.split("\n")[2].split()[0])  # number of atoms

            # read list of atoms
            for i in range(abs(nAtoms)):
                header += f.readline().decode("utf-8")

            if (
                nAtoms < 0
            ):  # MOs are stored in cubes with additional line of information
                header += f.readline().decode("utf-8")

            return header, f.tell()

    def parse_cube_header(self, header: str):
        """Parse header of cube to class attributes.

        :param header: String containing header of cube file.
        :type header: str
        """

        def parse_cubeheader_line(line: str, factor=1.0):
            """Helper function to parse single lines in cube file"""
            line = line.split()
            try:
                return (
                    abs(int(line[0])),
                    np.array([float(x) for x in line[1:4]]) * factor,
                    int(line[4]),
                )
            except IndexError:
                return (
                    abs(int(line[0])),
                    np.array([float(x) for x in line[1:4]]) * factor,
                    1,
                )

        lines = header.removesuffix("\n").split("\n")

        self.header = [
            lines[0].removesuffix("\n"),
            lines[1].removesuffix("\n"),
        ]  # first two lines of cube are comments

        # cube dimensions
        self.natoms, self.origin, self.NVal = parse_cubeheader_line(
            lines[2], factor=bohr2ang
        )
        self.n1, self.vec1, _ = parse_cubeheader_line(lines[3], factor=bohr2ang)
        self.n2, self.vec2, _ = parse_cubeheader_line(lines[4], factor=bohr2ang)
        self.n3, self.vec3, _ = parse_cubeheader_line(lines[5], factor=bohr2ang)

        # atoms in molecule
        self.atoms = []
        for i in range(self.natoms):
            line = lines[6 + i].split()
            atom = {}
            element, charge, coords = (
                int(line[0]),
                float(line[1]),
                np.array([float(x) for x in line[2:]]) * bohr2ang,
            )
            atom["element"] = element
            atom["nuclearcharge"] = charge
            atom["coords"] = coords
            atom["mass"] = default_masses[element]
            self.atoms.append(atom)

        # if cube contains molecular orbital(s), parse the additional line stating how many and which MOs are in the cube
        if len(lines) == self.natoms + 7:
            line = lines[self.natoms + 6].split()
            self.isMO = True
            self.nMO = abs(int(line[0]))
            self.vecMO = np.array([float(x) for x in line[1:]])

    def load_cube_data(
        self, filename: str | os.PathLike, pointer_position: int
    ) -> np.ndarray:
        """Load data from cube file into array.

        :param filename: Path to cube file
        :type filename: str | os.PathLike
        :param pointer_position: File position where data starts. See also `cube_utils.load_cube_header()`.
        :type pointer_position: int
        :return: Cube data.
        :rtype: np.ndarray
        """

        data = []
        with open(filename, "br") as f:
            f.seek(pointer_position)
            for line in f:
                line = line.split()
                data += [float(val) for val in line]

        return self.reshape_data(np.array(data))

    def reshape_data(self, data: np.ndarray) -> np.ndarray:
        """Reshape 1d data to n-dimensional array, depending on cubesize, number of orbitals and number of values per voxel.

        :param data: 1d array of cube data.
        :type data: np.ndarray
        :return: nd array of cube data.
        :rtype: np.ndarray
        """
        if self.NVal == 1:
            if self.nMO > 1:
                return data.reshape(self.shape + (self.nMO,))
            else:
                return data.reshape(self.shape)
        else:
            if self.nMO > 1:
                return data.reshape(self.shape + (self.nMO, self.NVal))
            else:
                return data.reshape(self.shape + (self.NVal,))

    # -----------------------------------------------------------------------------------------------------

    @property
    def coords(self) -> np.array:
        """Cartesiaon coordinates of atoms in Angstrom.

        :return: Array of shape (nAtoms,3).
        :rtype: np.array
        """
        return np.array([atom["coords"] for atom in self.atoms])

    @coords.setter
    def coords(self, new_coords):
        """Set coordinates of atoms.

        :param new_coords: 2d array of dimension (number_of_atoms,3).
        :type new_coords: np.ndarray
        """
        for i, icoords in enumerate(new_coords):
            self.atoms[i]["coords"] = icoords

    @property
    def element(self) -> np.ndarray:
        """Element number of atoms."""
        return np.array([atom["element"] for atom in self.atoms])

    @property
    def masses(self) -> np.ndarray:
        """Masses of atoms."""
        return default_masses[self.element]

    @property
    def center_of_mass(self) -> np.ndarray:
        """Center of mass of molecule."""
        return np.sum(self.coords * self.masses[:, None], axis=0) / np.sum(self.masses)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of cube dataset"""
        return (self.n1, self.n2, self.n3)

    @property
    def vecs(self) -> np.ndarray:
        """2d array containing vec1, vec2, vec3."""
        return np.stack((self.vec1, self.vec2, self.vec3))

    # -----------------------------------------------------------------------------------------------------

    def _cartesian_axis(self, dimension: int) -> np.ndarray:
        """Linspace of cartesian axis, if cube dimension aligns with it."""
        axis_name = ["x", "y", "z"][dimension]
        assert abs(self.vecs[dimension, dimension]) == np.linalg.norm(
            self.vecs[dimension, :]
        ), f"Cube axis {dimension} does not align with {axis_name} axis."
        return np.linspace(
            self.origin[dimension],
            self.origin[dimension]
            + (self.shape[dimension] - 1) * self.vecs[dimension, dimension],
            self.shape[dimension],
        )

    @property
    def x(self) -> np.ndarray:
        """Array containing values of x-axis."""
        return self._cartesian_axis(0)

    @property
    def y(self) -> np.ndarray:
        """Array containing values of y-axis."""
        return self._cartesian_axis(1)

    @property
    def z(self) -> np.ndarray:
        """Array containing values of z-axis."""
        return self._cartesian_axis(2)

    def meshgrid(self, pad: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create meshgrid of cube

        :param pad: Pad cube with `pad` points in all axes, defaults to 0
        :type pad: int, optional
        :return: Three 3d arrays.
        :rtype: tuple[np.ndarray,np.ndarray,np.ndarray]
        """
        if pad == 0:
            return np.meshgrid(self.x, self.y, self.z, indexing="ij")
        else:
            x = np.pad(self.x, pad, pad_lin_extrapolate)
            y = np.pad(self.y, pad, pad_lin_extrapolate)
            z = np.pad(self.z, pad, pad_lin_extrapolate)
            return np.meshgrid(x, y, z, indexing="ij")

    def meshgrid_xy(self, pad: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Create meshgrid of cube considering only x and y axes.

        :param pad: Pad cube with `pad` points in all axes, defaults to 0
        :type pad: int, optional
        :return: Two 2d arrays.
        :rtype: tuple[np.ndarray,np.ndarray]
        """
        if pad == 0:
            return np.meshgrid(self.x, self.y, indexing="ij")
        else:
            x = np.pad(self.x, pad, pad_lin_extrapolate)
            y = np.pad(self.y, pad, pad_lin_extrapolate)
            return np.meshgrid(x, y, indexing="ij")

    # -----------------------------------------------------------------------------------------------------

    def _internal_axis(self, dimension: int) -> np.ndarray:
        """Linspace of internal axis."""
        origin = np.dot(
            self.origin - self.center_of_mass, self.vecs[dimension, :]
        ) / np.linalg.norm(self.vecs[dimension, :])
        return origin + np.linalg.norm(self.vecs[dimension, :]) * np.arange(
            self.shape[dimension]
        )

    @property
    def x_int(self) -> np.ndarray:
        """Array containing values of first internal axis."""
        return self._internal_axis(0)

    @property
    def y_int(self) -> np.ndarray:
        """Array containing values of second internal axis."""
        return self._internal_axis(1)

    @property
    def z_int(self) -> np.ndarray:
        """Array containing values of third internal axis."""
        return self._internal_axis(2)

    def internal_meshgrid(
        self, pad: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create meshgrid of cube using internal coordinates.

        :param pad: Pad cube with `pad` points in all axes, defaults to 0
        :type pad: int, optional
        :return: Three 3d arrays.
        :rtype: tuple[np.ndarray,np.ndarray,np.ndarray]
        """
        if pad == 0:
            return np.meshgrid(self.x_int, self.y_int, self.z_int, indexing="ij")
        else:
            x = np.pad(self.x_int, pad, pad_lin_extrapolate)
            y = np.pad(self.y_int, pad, pad_lin_extrapolate)
            z = np.pad(self.z_int, pad, pad_lin_extrapolate)
            return np.meshgrid(x, y, z, indexing="ij")

    def _ax_is_perp(self, axis: int) -> bool:
        """Is `axis` perpendicular to the other two axes?

        :param axis: Axis number 0, 1, or 2.
        :type axis: int
        :return: True if axis is perpendicular.
        :rtype: bool
        """
        t = [0, 1, 2]
        t.remove(axis)
        return not np.linalg.norm(
            np.cross(np.cross(self.vecs[t[0]], self.vecs[t[1]]), self.vecs[axis])
        )

    # -----------------------------------------------------------------------------------------------------

    def extrapolate_wavefunction(
        self,
        tip_height: float = 7.0,
        cut_off_height: float = 1.5,
        workfunction: float = 5.4,
        center_mass=True,
        MO: int = None,
        pad: int = 0,
    ) -> np.ndarray:
        """Extrapolate wavefunction to `tip_height`, using Bardeen or Tersoff-Hamann (TH) formalism, if `tip_height > plane`.

        :param tip_height: Height of tip center in Angstrom, defaults to 7.
        :type tip_height: float, optional
        :param cut_off_height: Plane height in Angstrom for wave function cut off, defaults to 1.5
        :type cut_off_height: float, optional
        :param workfunction: Vacuum barrier height in eV, defaults to 5.4
        :type workfunction: float, optional
        :param center_mass: `tip_height` and `plane` are relative to molecules center of mass (default), otherwise they are absolute values.
        :type center_mass: bool, optional
        :param MO: Index of molecular orbital, in case of multi-orbital cube, defaults to None
        :type MO: int, optional
        :param pad: Pad map with with number of zeros in both axis, defaults to 0
        :type pad: int, optional
        :return: 2d array with wavefunction extrapolated to `tip_height`.
        :rtype: np.ndarray
        """

        if center_mass:
            tip_height += self.center_of_mass[2]
            cut_off_height += self.center_of_mass[2]

        if abs(tip_height) <= abs(cut_off_height):
            return self.integration_plane(cut_off_height, MO=MO, pad=pad)
        else:
            return self.extrapolate_WF(
                tip_height, cut_off_height, workfunction, MO=MO, pad=pad
            )

    def extrapolate_WF(
        self,
        z: float = 7.0,
        cut_off_height: float = 1.5,
        workfunction: float = 5.4,
        MO: int = None,
        pad: int = 0,
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
        morb_plane = self.integration_plane(cut_off_height, MO=MO, pad=pad)

        # =====================================================================
        # The following code snipped was adapted from:
        # https://github.com/nanotech-empa/cp2k-spm-tools/blob/main/cp2k_spm_tools/cp2k_grid_orbitals.py
        #
        fourier = np.fft.rfft2(morb_plane)
        kx_arr = (
            2 * np.pi * np.fft.fftfreq(morb_plane.shape[0], np.linalg.norm(self.vec1))
        )
        ky_arr = (
            2 * np.pi * np.fft.rfftfreq(morb_plane.shape[1], np.linalg.norm(self.vec2))
        )
        kx_grid, ky_grid = np.meshgrid(kx_arr, ky_arr, indexing="ij")

        fac = 2 * m_e * q_e / hbar**2 * 1e-20
        kappa = np.sqrt(kx_grid**2 + ky_grid**2 + fac * workfunction)

        dz = abs(z - cut_off_height)
        return np.fft.irfft2(fourier * np.exp(-kappa * dz), morb_plane.shape)
        #
        # =====================================================================

    def integration_plane(
        self, z: float = 1.5, MO: int = None, pad: int = 0
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
        assert self._ax_is_perp(2), "z is not perpendicular to xy plane."
        plane_idx = np.searchsorted(
            self.z, z
        )  # make sure this works also for reversed z-axes

        assert self.z[plane_idx] - z <= np.linalg.norm(self.vec3), (
            "Height of integration plane not within z-range of cube data."
        )

        if self.data.ndim == 3 and MO == 0:
            MO = None
        padding = ((pad, pad), (pad, pad))

        return np.pad(
            np.squeeze(self.data[:, :, plane_idx : plane_idx + 1, MO]), padding
        )


def pad_lin_extrapolate(vector: np.ndarray, pad_width: tuple, iaxis: int, kwargs):
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


# array to be used for default masses of elements
default_masses = np.array(
    [
        np.nan,
        1.008,
        4.0026,
        6.94,
        9.0122,
        10.81,
        12.011,
        14.007,
        15.999,
        18.998,
        20.18,
        22.99,
        24.305,
        26.982,
        28.085,
        30.974,
        32.06,
        35.45,
        39.95,
        39.098,
        40.078,
        44.956,
        47.867,
        50.942,
        51.996,
        54.938,
        55.845,
        58.933,
        58.693,
        63.546,
        65.38,
        69.723,
        72.63,
        74.922,
        78.971,
        79.904,
        83.798,
        85.468,
        87.62,
        88.906,
        91.222,
        92.906,
        95.95,
        np.nan,
        101.07,
        102.91,
        106.42,
        107.87,
        112.41,
        114.82,
        118.71,
        121.76,
        127.6,
        126.9,
        131.29,
        132.91,
        137.33,
        138.91,
        140.12,
        140.91,
        144.24,
        np.nan,
        150.36,
        151.96,
        157.25,
        158.93,
        162.5,
        164.93,
        167.26,
        168.93,
        173.05,
        174.97,
        178.49,
        180.95,
        183.84,
        186.21,
        190.23,
        192.22,
        195.08,
        196.97,
        200.59,
        204.38,
        207.2,
        208.98,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        232.04,
        231.04,
        238.03,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
)
