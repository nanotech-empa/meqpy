import pytest
from pathlib import Path

# Dir file to store test data files, if needed in the future
DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def sample_cube():
    return DATA_DIR / "sample.cube"


@pytest.fixture
def sample_orbital():
    return DATA_DIR / "sample_orbital.cube"


@pytest.fixture
def sample_cube_rotated():
    return DATA_DIR / "sample_rotated.cube"


@pytest.fixture
def sample_cube_non_ortho_z():
    return DATA_DIR / "sample_non_ortho_z.cube"


@pytest.fixture
def sample_cube_rhombic():
    return DATA_DIR / "sample_rhombic.cube"
