import pytest
from pathlib import Path
from meqpy.system import System, Dyson, Molecule, State, BandSystem, BandTransition


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
def make_system():
    """Return resolver to create 3-state system. If quartet = True: add state Q"""

    def _resolve(quartet: bool = False, **kwargs):
        states = [
            State("GS", 0.0, 0, multiplicity=1),
            State("PIR", 0.5, 1, multiplicity=2),
            State("NIR", 0.3, -1, multiplicity=2),
        ]
        defaults = dict(hwhm=0.0, lineshape="dirac", workfunction=5.0, kappa_mode="10")
        defaults.update(kwargs)
        if quartet:
            states.append(State("Q", 1.5, 1, multiplicity=4))
        return System(states=states, **defaults)

    return _resolve


@pytest.fixture
def make_molecule():
    """Return resolver to create Molecule, without dyson."""

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
    """Return resolver to create Molecule, with dyson for GS -> PIR transition."""

    def _resolve(cube_name, **kwargs):
        molecule = make_molecule(**kwargs)
        dyson = Dyson(cube_path(cube_name))
        molecule.add_dyson("GS", "PIR", dyson)
        return molecule, dyson

    return _resolve


@pytest.fixture
def make_bandtransition():
    """Return resolver to create BandTransition."""

    def _resolve(**kwargs):
        band_defaults = dict(bandwidth=1.0, effective_mass=1.0, dx=0.01, hwhm=0.0)
        band_defaults.update(kwargs)
        band = BandTransition(**band_defaults)
        return band

    return _resolve


@pytest.fixture
def make_bandsystem():
    """Return resolver to create 3-state BandSystem, without band transitions."""

    def _resolve(**kwargs):
        states = [
            State("GS", 0.0, 0, multiplicity=1),
            State("VB", 0.5, 1, multiplicity=2),
            State("CB", 0.3, -1, multiplicity=2),
        ]
        defaults = dict(hwhm=0.0, lineshape="dirac", workfunction=5.0, kappa_mode="10")
        defaults.update(kwargs)
        return BandSystem(states=states, **defaults)

    return _resolve


@pytest.fixture
def make_bandsystem_with_band(make_bandsystem):
    """Return resolver to create BandSystem, with a BandTransition for GS -> VB."""

    def _resolve(band_kwargs=None, **kwargs):
        bandsystem = make_bandsystem(**kwargs)
        band_kwargs = band_kwargs or {}
        band_defaults = dict(bandwidth=1.0, effective_mass=1.0, dx=0.01)
        band_defaults.update(band_kwargs)
        band = BandTransition(**band_defaults)
        key = ("GS", "VB")
        bandsystem.add_band_transition(*key, band)
        return bandsystem, key

    return _resolve
