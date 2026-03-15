import pytest
from pathlib import Path

# Dir file to store test data files, if needed in the future
DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def sample_cube():
    return DATA_DIR / "sample.cube"
