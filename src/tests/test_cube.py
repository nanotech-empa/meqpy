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
