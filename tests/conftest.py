import pytest
from pathlib import Path
from meqpy.system import Dyson, Molecule, State


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


@pytest.fixture
def make_molecule():
    """Build a small Molecule with GS, PIR, NIR states (no Dyson orbitals attached)."""

    def _resolve(**kwargs):
        states = [
            State("GS", 0.0, 0, multiplicity=1),
            State("PIR", 0.5, 1, multiplicity=2),
            State("NIR", 0.3, -1, multiplicity=2),
        ]
        defaults = dict(hwhm=0.0, lineshape="dirac", workfunction=5.0, kappa_mode="10")
        defaults.update(kwargs)
        return Molecule(states=states, **defaults)

    return _resolve


@pytest.fixture
def make_molecule_with_dyson(make_molecule, cube_path):
    """Build a small Molecule with GS, PIR, NIR states, including dyson for GS -> PIR transition."""

    def _resolve(cube_name, **kwargs):
        molecule = make_molecule(**kwargs)
        dyson = Dyson(cube_path(cube_name))
        molecule.add_dyson("GS", "PIR", dyson)
        return molecule, dyson

    return _resolve
