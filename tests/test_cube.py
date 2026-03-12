import numpy as np
from meqpy.io import Cube
from pymatgen.core.structure import Molecule


def test_cube_initialization(sample_volumetricdata):
    cube = Cube(sample_volumetricdata)
    assert cube.cube_data == sample_volumetricdata
    assert cube.data.shape == (10, 10, 10)
    assert len(cube.molecule) == 4


def test_cube_properties(sample_volumetricdata):
    cube = Cube(sample_volumetricdata)
    assert cube.elements == ["N", "H", "H", "H"]
    assert np.allclose(cube.masses, [14.007, 1.008, 1.008, 1.008], atol=1e-3)
    assert np.allclose(cube.center_of_mass, [5.00137, 5.00000, 5.00000], atol=1e-3)
    assert isinstance(cube.molecule, Molecule)


def test_cube_cart_coords(sample_volumetricdata):
    cube = Cube(sample_volumetricdata)
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


def test_from_file(sample_cube):
    cube = Cube.from_file(sample_cube)
    assert isinstance(cube, Cube)
    assert cube.data.shape == (
        20,
        20,
        20,
    )  # We use a dummy cube file with a 20x20x20 grid for testing
    assert len(cube.molecule) == 4


def test_get_slice(sample_volumetricdata):
    cube = Cube(sample_volumetricdata)
    # Test getting a slice along the x-axis at a distance of 5.0
    slice_x = cube.get_slide_data(axis=0, distance=5.0)
    assert slice_x.shape == (10, 10)  # The slice should be 2D with shape (10, 10)

    # Test distance exceeding the lattice parameter
    try:
        cube.get_slide_data(axis=0, distance=15.0)
    except ValueError as e:
        length = cube.structure.lattice.abc[0]
        assert (
            str(e)
            == f"Distance must be between 0 and {length} along the specified axis."
        )
