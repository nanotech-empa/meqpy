import numpy as np
import pytest
import scipy.constants as const
from meqpy.io import Cube
from meqpy.system import OpticalTransition
from meqpy.utils.constants import BOHR


def coulomb_potential_ref(mesh_x, mesh_y, mesh_z, x_q, y_q, z_q):
    """Independent reference implementation of k*q/r in V, r in Å, computed
    directly from scipy constants (not reusing the module's EPSILON_0)."""
    k_ang = 1 / (4 * np.pi * const.epsilon_0) * 1e10  # V*Å/C
    r = np.sqrt((x_q - mesh_x) ** 2 + (y_q - mesh_y) ** 2 + (z_q - mesh_z) ** 2)
    return k_ang * const.elementary_charge / r


class TestOpticalTransitionInit:
    def test_from_file(self, cube_path):
        cube = Cube(cube_path("cartesian"))
        optical = OpticalTransition(cube_path("cartesian"))

        assert optical.shape == (20, 20, 20)
        assert np.allclose(optical.spacing, np.eye(3) * 0.5, atol=1e-6)
        assert np.allclose(optical.data, cube.data / BOHR**1.5, atol=1e-10)

    def test_from_cube(self, cube_path):
        cube = Cube(cube_path("cartesian"))
        optical = OpticalTransition(cube)

        assert np.allclose(optical.data, cube.data / BOHR**1.5, atol=1e-10)

    def test_from_file_no_center_mass(self, cube_path):
        optical = OpticalTransition(cube_path("cartesian"), center_mass=False)
        assert np.allclose(optical.origin, [0.0, 0.0, 0.0], atol=1e-3)

    def test_missing_cube_raises(self):
        with pytest.raises(FileNotFoundError):
            OpticalTransition("not_a_real_path.cube")


class TestKernelMesh:
    def test_shape_no_pad(self, cube_path):
        optical = OpticalTransition(cube_path("cartesian"))
        kernel_mesh = optical._kernel_mesh(pad=0)
        assert kernel_mesh.shape == (3, 41, 41, 20)

    def test_shape_with_pad(self, cube_path):
        optical = OpticalTransition(cube_path("cartesian"))
        kernel_mesh = optical._kernel_mesh(pad=3)
        assert kernel_mesh.shape == (3, 47, 47, 20)

    def test_xy_symmetric_around_zero(self, cube_path):
        optical = OpticalTransition(cube_path("cartesian"))
        kernel_mesh = optical._kernel_mesh(pad=0)

        assert np.isclose(kernel_mesh[0][20, 20, 0], 0.0, atol=1e-6)
        assert np.allclose(
            kernel_mesh[0][0, :, 0], -kernel_mesh[0][-1, :, 0], atol=1e-6
        )
        assert np.allclose(
            kernel_mesh[1][:, 0, 0], -kernel_mesh[1][:, -1, 0], atol=1e-6
        )

    def test_z_matches_cartesian_z(self, cube_path):
        optical = OpticalTransition(cube_path("cartesian"))
        kernel_mesh = optical._kernel_mesh(pad=0)

        assert np.allclose(kernel_mesh[2][0, 0, :], optical.z, atol=1e-6)


class TestCoulombPotential:
    def test_potential_is_zero_if_too_close(self):
        mesh_x = np.array([0.5])
        mesh_y = np.array([0.0])
        mesh_z = np.array([0.0])
        mesh = [mesh_x, mesh_y, mesh_z]

        potential = OpticalTransition._coulomb_potential(
            mesh,
            z_pointcharge=0.0,
        )

        assert potential[0] == 0.0

    def test_decays_with_distance(self):
        mesh_x = np.array([1.0, 2.0, 4.0])
        mesh_y = np.zeros(3)
        mesh_z = np.zeros(3)
        mesh = [mesh_x, mesh_y, mesh_z]

        potential = OpticalTransition._coulomb_potential(
            mesh,
            z_pointcharge=0.0,
        )

        # Coulomb potential must fall off with increasing distance.
        assert potential[0] > potential[1] > potential[2] > 0


class TestTwoPointCharges:
    def test_zero_when_charge_on_mirror_plane(self, meshgrid):
        potential = OpticalTransition._two_point_charges(
            meshgrid(), mirror_plane=2.0, z_pointcharge=2.0
        )

        assert np.allclose(potential, 0.0, atol=1e-8)

    def test_zero_on_mirror_plane_itself(self, meshgrid):
        mirror_plane = 2.0
        potential = OpticalTransition._two_point_charges(
            meshgrid(z_value=mirror_plane), mirror_plane=mirror_plane, z_pointcharge=7.0
        )

        assert np.allclose(potential, 0.0, atol=1e-8)

    def test_antisymmetric_about_mirror_plane(self, meshgrid):
        above = OpticalTransition._two_point_charges(
            meshgrid(z_value=2.0), mirror_plane=0.0, z_pointcharge=5.0
        )
        below = OpticalTransition._two_point_charges(
            meshgrid(z_value=-2.0), mirror_plane=0.0, z_pointcharge=5.0
        )

        assert np.allclose(above, -below, atol=1e-8)


class TestPlasmonCoupling:
    def test_shape_no_pad(self, cube_path):
        optical = OpticalTransition(cube_path("cartesian"))
        coupling = optical.plasmon_coupling(mirror_plane=-8.0, z_pointcharge=3.0, pad=0)
        assert coupling.shape == (20, 20)

    def test_shape_with_pad(self, cube_path):
        optical = OpticalTransition(cube_path("cartesian"))
        coupling = optical.plasmon_coupling(mirror_plane=-8.0, z_pointcharge=3.0, pad=3)
        assert coupling.shape == (26, 26)

    def test_pad_validators(self, cube_path):
        optical = OpticalTransition(cube_path("cartesian"))

        with pytest.raises(ValueError) as e_info:
            optical.plasmon_coupling(-8.0, 3.0, pad=-1)
        assert str(e_info.value) == "pad must be >= 0, got -1"

        with pytest.raises(TypeError) as e_info:
            optical.plasmon_coupling(-8.0, 3.0, pad=0.5)
        assert str(e_info.value) == "pad must be int, got float"


class TestEmissionStrength:
    def test_matches_squared_coupling_unnormalized(self, cube_path):
        optical = OpticalTransition(cube_path("cartesian"))
        coupling = optical.plasmon_coupling(mirror_plane=-8.0, z_pointcharge=3.0, pad=0)
        emission = optical.emission_strength(
            mirror_plane=-8.0, z_pointcharge=3.0, pad=0, normalize=False
        )
        assert np.allclose(emission, coupling**2, atol=1e-12)

    def test_normalized_output_bounded_by_one(self, cube_path):
        optical = OpticalTransition(cube_path("cartesian"))
        emission = optical.emission_strength(
            mirror_plane=-8.0, z_pointcharge=3.0, pad=0
        )

        assert emission.max() == pytest.approx(1.0)
        assert emission.min() >= 0.0

    def test_all_zero_data_does_not_raise_or_produce_nan(self, cube_path):
        optical = OpticalTransition(cube_path("cartesian"))
        optical.data = np.zeros_like(optical.data)

        emission = optical.emission_strength(mirror_plane=-8.0, z_pointcharge=3.0)

        assert np.allclose(emission, 0.0)
        assert not np.any(np.isnan(emission))
