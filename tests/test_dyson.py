import numpy as np
import pytest
from meqpy.system import Dyson


def test_from_file(sample_cube):
    dyson = Dyson(sample_cube)
    assert isinstance(dyson, Dyson)

    assert dyson.shape == (20, 20)

    spacing = np.array(
        [
            [0.500000, 0.000000, 0.000000],
            [0.000000, 0.500000, 0.000000],
        ]
    )
    assert np.allclose(dyson.spacing, spacing, atol=1e-6)
    assert np.allclose(dyson.origin, [-5.00137, -5.00000], atol=1e-6)
    assert np.allclose(dyson.steps, [0.5, 0.5], atol=1e-6)

    x = np.arange(20) * 0.5 - 5.00137
    y = np.arange(20) * 0.5 - 5.00000

    assert np.allclose(x, dyson.x, atol=1e-5)
    assert np.allclose(y, dyson.y, atol=1e-5)

    with pytest.raises(ValueError) as e_info:
        dyson.get_cart_axis(2)
    assert str(e_info.value) == "axis must be in [0, 1], but got 2"


def test_rhombic_cube(sample_cube_rhombic):
    dyson = Dyson(sample_cube_rhombic, center_mass=False)
    assert isinstance(dyson, Dyson)

    spacing = np.array(
        [
            [0.500000, 0.000000, 0.000000],
            [-0.250000, 0.433012, 0.000000],
        ]
    )
    assert np.allclose(dyson.spacing, spacing, atol=1e-6)
    assert np.allclose(dyson.steps, [0.5, 0.5], atol=1e-6)

    with pytest.raises(ValueError) as e_info:
        dyson.get_cart_axis(0)
    assert (
        str(e_info.value)
        == "Cube axes 1 and 0 are not orthogonal. Consider using methods grid() or mesh_cartesian() instead."
    )

    grid = np.arange(-5, 16) * 0.5
    assert np.allclose(dyson.grid(pad=5)[0], grid, atol=1e-6)
    assert np.allclose(dyson.grid(pad=5)[1], grid, atol=1e-6)


def test_cartesian_mesh(sample_cube_rhombic):
    dyson = Dyson(sample_cube_rhombic, center_mass=False)

    assert dyson.mesh_cartesian().shape == (2, 11, 11)

    point = dyson.mesh_cartesian(pad=5)[:, 7, 8]
    assert np.allclose(point, np.array([-1.41336, -0.65322]), atol=1e-5)


def test_dyson_coupling_strength(sample_cube):
    dyson = Dyson(sample_cube)

    sts_map = dyson.coupling_strength(
        height=5.0,
        kappa=np.ones((4, 1, 5)),
        pad=5,
    )

    assert sts_map.shape == (30, 30, 4, 5)
