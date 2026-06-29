import numpy as np
import pytest
from meqpy.system import BandTransition


def test_bandwidth_negative_raises():
    with pytest.raises(ValueError):
        BandTransition(bandwidth=-1.0)


def test_dx_not_positive_raises():
    with pytest.raises(ValueError) as e_info:
        BandTransition(dx=0)
    assert str(e_info.value) == "dx must be positive, but got 0."


def test_dx_wrong_type():
    with pytest.raises(TypeError):
        BandTransition(dx="0.1")


def test_e_offset_negative_raises():
    with pytest.raises(ValueError):
        BandTransition(e_offset=-1.0)


def test_dos_is_unit_step_within_band(make_bandtransition):
    band = make_bandtransition()
    dos = band.dos
    energy = band.energy

    assert np.all(dos[(energy > 0) & (energy < band.bandwidth)] == 1.0)
    assert np.all(dos[(energy < 0) | (energy > band.bandwidth)] == 0.0)
    assert np.all(dos[(energy == 0) | (energy == band.bandwidth)] == 0.5)


def test_kpar_zero_below_band_bottom(make_bandtransition):
    band = make_bandtransition()
    energy = band.energy
    kpar = band.kpar
    assert np.all(kpar[energy < 0] == -band.kpar_offset)


def test_kpar_offset_shifts_kpar(make_bandtransition):
    band_no_offset = make_bandtransition()
    band_offset = make_bandtransition(kpar_offset=0.3)

    assert np.allclose(band_offset.kpar, 0.3 - band_no_offset.kpar, atol=1e-9)


def test_kpar_increases_with_effective_mass(make_bandtransition):
    band_light = make_bandtransition(effective_mass=0.5)
    band_heavy = make_bandtransition(effective_mass=2.0)
    energy = band_light.energy
    inside_band = energy > 0
    assert np.all(band_heavy.kpar[inside_band] > band_light.kpar[inside_band])


def test_energy_internal_cache_invalidated_by_bandwidth_change(make_bandtransition):
    band = make_bandtransition()
    first = band.energy
    band.bandwidth = 2.0
    second = band.energy
    assert first.max() != second.max()
    assert np.isclose(second.max(), 2.0, atol=band.dx)


def test_energy_internal_cache_invalidated_by_hwhm_change(make_bandtransition):
    band = make_bandtransition()
    first = band.energy
    band.hwhm = 0.1
    second = band.energy
    assert second.min() < first.min()
    assert second.max() > first.max()


def test_energy_internal_cache_invalidated_by_dx_change(make_bandtransition):
    band = make_bandtransition()
    n_points_before = len(band.energy)
    band.dx = 0.001
    n_points_after = len(band.energy)
    assert n_points_after > n_points_before
