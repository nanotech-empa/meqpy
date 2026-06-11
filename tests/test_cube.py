import numpy as np
from meqpy.io import Cube


def test_from_file(sample_cube):
    cube = Cube(sample_cube)
    assert isinstance(cube, Cube)
    assert cube.data.shape == (20, 20, 20)
    assert len(cube.atoms) == 4
    assert cube.cart_coords.shape == (4, 3)
    coords = np.array(
        [
            [5.0018, 5.0000, 5.0000],
            [6.0118, 5.0000, 5.0000],
            [4.4932, 4.1274, 5.0000],
            [4.4932, 5.8726, 5.0000],
        ]
    )
    assert np.allclose(cube.cart_coords, coords)
    spacing = np.array(
        [
            [0.500000, 0.000000, 0.000000],
            [0.000000, 0.500000, 0.000000],
            [0.000000, 0.000000, 0.500000],
        ]
    )
    assert np.allclose(cube.spacing, spacing, atol=1e-7)
    assert cube.elements == ["N", "H", "H", "H"]
    assert np.allclose(cube.masses, [14.007, 1.008, 1.008, 1.008], atol=1e-3)
    assert np.allclose(cube.center_of_mass, [5.00137, 5.00000, 5.00000], atol=1e-3)


def test_magsqr(sample_orbital):
    cube = Cube(sample_orbital)
    # Check that the values are non-negative
    print(cube.magsqr)
    assert np.isclose(cube.magsqr, 1.0, atol=1e-3)


def test_get_slice(sample_cube):
    cube = Cube(sample_cube)
    # Test getting a slice along the x-axis at a distance of 5.0
    slice_x = cube.get_slice_data(distance=5.0, axis=0)
    assert slice_x.shape == (20, 20)  # The slice should be 2D with shape (20, 20)

    # Test distance exceeding the lattice parameter
    try:
        cube.get_slice_data(
            distance=15.0,
            axis=0,
        )
    except ValueError as e:
        length = cube.atoms.cell.cellpar()[0]
        assert (
            str(e)
            == f"Distance must be between 0 and {length} along the specified axis."
        )
