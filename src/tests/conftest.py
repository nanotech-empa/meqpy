import pytest
import numpy as np
from pathlib import Path
from pymatgen.io.common import VolumetricData
from pymatgen.core.structure import Structure, Lattice

# Dir file to store test data files, if needed in the future
DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def sample_volumetricdata():
    lattice = Lattice.cubic(10)
    structure = Structure(
        lattice,
        ["N", "H", "H", "H"],
        [
            [0.50018, 0.50000, 0.50000],
            [0.60118, 0.50000, 0.50000],
            [0.44932, 0.41274, 0.50000],
            [0.44932, 0.58726, 0.50000],
        ],
    )
    data = {"total": np.random.rand(10, 10, 10)}
    return VolumetricData(structure=structure, data=data)


@pytest.fixture
def sample_cube():
    return DATA_DIR / "sample.cube"
