import numpy as np
import pytest
from meqpy.io import Cube
from meqpy.system import Transition


def nottest(obj):
    obj.__test__ = False
    return obj


@nottest
def test_transition_cartesian(transition):
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


def test_from_file(cube_path):
    transition = Transition(cube_path("cartesian"))
    test_transition_cartesian(transition)


def test_from_cube(cube_path):
    cube = Cube(cube_path("cartesian"))
    transition = Transition(cube)
    test_transition_cartesian(transition)


def test_from_file_no_center_mass(cube_path):
    transition = Transition(cube_path("cartesian"), center_mass=False)
    assert isinstance(transition, Transition)
    assert np.allclose(transition.origin, [0.0, 0.0, 0.0], atol=1e-3)


def test_rotated_cube(cube_path):
    with pytest.raises(ValueError) as e_info:
        Transition(cube_path("rotated"))
    assert str(e_info.value) == "Last axis is not aligned with z-axis"


def test_non_ortho_cube(cube_path):
    with pytest.raises(ValueError) as e_info:
        Transition(cube_path("non_ortho_z"))
    assert str(e_info.value) == "z axis is not orthogonal to other axes."


def test_rhombic_cube(cube_path):
    transition = Transition(cube_path("rhombic"), center_mass=False)
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
