import numpy as np
import pytest
from meqpy.io import Cube
from meqpy.system import Transition


def test_from_file(sample_cube):
    transition = Transition(sample_cube)
    assert isinstance(transition, Transition)

    spacing = np.array(
        [
            [0.500000, 0.000000, 0.000000],
            [0.000000, 0.500000, 0.000000],
            [0.000000, 0.000000, 0.500000],
        ]
    )
    assert np.allclose(transition.spacing, spacing, atol=1e-6)
    assert np.allclose(transition.origin, [-5.00137, -5.00000, -5.00000], atol=1e-6)
    assert np.allclose(transition.steps, [0.5, 0.5, 0.5], atol=1e-6)


def test_from_file_no_center_mass(sample_cube):
    transition = Transition(sample_cube, center_mass=False)
    assert isinstance(transition, Transition)
    assert np.allclose(transition.origin, [0.0, 0.0, 0.0], atol=1e-3)


def test_from_cube(sample_cube):
    cube = Cube(sample_cube)
    test_from_file(cube)


def test_rotated_cube(sample_cube_rotated):
    with pytest.raises(ValueError) as e_info:
        Transition(sample_cube_rotated)
    assert str(e_info.value) == "Last axis is not aligned with z-axis"


def test_non_ortho_cube(sample_cube_non_ortho_z):
    with pytest.raises(ValueError) as e_info:
        Transition(sample_cube_non_ortho_z)
    assert str(e_info.value) == "z axis is not orthogonal to other axes."


def test_rhombic_cube(sample_cube_rhombic):
    transition = Transition(sample_cube_rhombic, center_mass=False)
    assert isinstance(transition, Transition)

    spacing = np.array(
        [
            [0.500000, 0.000000, 0.000000],
            [-0.250000, 0.433012, 0.000000],
            [0.000000, 0.000000, 0.500000],
        ]
    )
    assert np.allclose(transition.spacing, spacing, atol=1e-6)
    assert np.allclose(transition.steps, [0.5, 0.5, 0.5], atol=1e-6)
