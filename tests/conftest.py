import pytest
from pathlib import Path

# Dir file to store test data files, if needed in the future
DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def cube_path():
    """Return a resolver: cube_path("rhombic") -> DATA_DIR/'sample_rhombic.cube'."""

    def _resolve(name):
        path = DATA_DIR / f"sample_{name}.cube"
        if not path.exists():
            pytest.fail(f"Missing test data file: {path}")
        return path

    return _resolve
