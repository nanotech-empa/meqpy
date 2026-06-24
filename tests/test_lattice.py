import numpy as np
import pytest


def test_add_and_get_band_transition(make_lattice, make_bandtransition):
    lattice = make_lattice()
    band = make_bandtransition()

    lattice.add_band_transition("GS", "VB", band)
    assert lattice.band_transition_dict == {("GS", "VB"): band}

    lattice.add_band_transition("VB", "GS", band)
    assert lattice.band_transition_dict == {("GS", "VB"): band, ("VB", "GS"): band}


def test_add_band_transition_wrong_type(make_lattice):
    lattice = make_lattice()
    with pytest.raises(TypeError) as e_info:
        lattice.add_band_transition("GS", "VB", "not a band")
    assert "band must be type BandTransition" in str(e_info.value)


def test_add_band_transition_invalid_charging_pair(make_lattice, make_bandtransition):
    lattice = make_lattice()
    band = make_bandtransition()
    with pytest.raises(ValueError):
        lattice.add_band_transition("VB", "CB", band)


def test_band_transition_dict_setter(make_lattice, make_bandtransition):
    lattice = make_lattice()
    band = make_bandtransition()
    lattice.band_transition_dict = {("GS", "VB"): band}
    assert lattice.band_transition_dict == {("GS", "VB"): band}


def test_band_transition_dict_setter_wrong_type(make_lattice):
    lattice = make_lattice()
    with pytest.raises(TypeError) as e_info:
        lattice.band_transition_dict = "not a dict"
    assert "bands must be a dict" in str(e_info.value)


def test_band_energy(make_lattice_with_band):
    lattice, key = make_lattice_with_band()

    energy_plus = lattice.band_energy(key)
    assert np.isclose(energy_plus.max(), -0.49, atol=1e-9)
    assert np.isclose(energy_plus.min(), -1.50, atol=1e-9)

    band = lattice.band_transition_dict[key]
    lattice.add_band_transition("GS", "CB", band)
    energy_minus = lattice.band_energy(("GS", "CB"))
    assert np.isclose(energy_minus.min(), 0.29, atol=1e-9)
    assert np.isclose(energy_minus.max(), 1.30, atol=1e-9)


def test_band_kappa_shape_default_squeeze(make_lattice_with_band):
    lattice, key = make_lattice_with_band()

    energy, kappa = lattice.band_kappa(key, bias=0.0)
    assert energy.shape == kappa.shape


def test_band_kappa_increases_with_kpar(make_lattice_with_band):
    lattice, key = make_lattice_with_band()

    # band.kpar_offset = 0.
    _, kappa_low = lattice.band_kappa(key, bias=0.0)

    lattice.band_transition_dict[key].kpar_offset = 1.0
    _, kappa_high = lattice.band_kappa(key, bias=0.0)
    assert kappa_high[0] > kappa_low[0]


def test_band_kappa_full_mode_increases_with_bias(make_lattice_with_band):
    lattice, key = make_lattice_with_band(kappa_mode="full")

    _, kappa_low = lattice.band_kappa(key, bias=0.0)
    _, kappa_high = lattice.band_kappa(key, bias=1.0)
    assert np.all(kappa_high > kappa_low)


def test_band_kappa_multi_bias_shape(make_lattice_with_band):
    lattice, key = make_lattice_with_band(kappa_mode="full")

    bias = np.array([0.0, 0.5, 1.0])
    energy, kappa = lattice.band_kappa(key, bias=bias, kappa_mode="full", squeeze=False)
    assert kappa.shape == (3, len(energy))


def test_get_band_charging_rate_shape(make_lattice_with_band):
    lattice, key = make_lattice_with_band()

    rate = lattice.get_band_charging_rate(key, z=5.0, bias=-1.0)
    assert np.isscalar(rate) or rate.shape == ()
    assert rate > 0


def test_get_band_charging_rate_increases_with_bias_magnitude(
    make_lattice_with_band,
):
    # Within the band's energy domain, integrating further into the band
    # (more negative bias here) should monotonically increase the rate.
    lattice, key = make_lattice_with_band()

    rate_shallow = lattice.get_band_charging_rate(key, z=5.0, bias=-0.6)
    rate_deep = lattice.get_band_charging_rate(key, z=5.0, bias=-1.3)
    assert rate_deep > rate_shallow


def test_get_band_charging_rate_shape_multi_z_bias(make_lattice_with_band):
    lattice, key = make_lattice_with_band()

    z = np.array([5.0, 6.0])
    bias = np.array([0.0, 0.1, 0.2])
    rate = lattice.get_band_charging_rate(key, z=z, bias=bias, squeeze=False)
    assert rate.shape == (2, 3)


def test_charging_rates_shape(make_lattice_with_band):
    lattice, _ = make_lattice_with_band()
    rates = lattice.charging_rates(z=5.0, bias=0.0)
    assert rates.shape == (3, 3)


def test_charging_rates_multi_z_bias_shape(make_lattice_with_band):
    lattice, _ = make_lattice_with_band()
    rates = lattice.charging_rates(
        z=np.array([5.0, 6.0]), bias=np.array([-0.1, 0.0, 0.1, 0.2])
    )
    assert rates.shape == (2, 4, 3, 3)


def test_kpar_zero_without_band_transitions(make_lattice):
    lattice = make_lattice()
    assert np.array_equal(lattice.kpar_offset, np.zeros((3, 3)))


def test_kpar_symmetric_and_matches_band_offset(make_lattice_with_band):
    lattice, key = make_lattice_with_band({"kpar_offset": 0.3})

    i, f = lattice.get_index(key[0]), lattice.get_index(key[1])
    kpar = lattice.kpar_offset
    assert kpar[i, f] == 0.3
    assert kpar[f, i] == 0.3
    assert np.array_equal(kpar, kpar.T)


def test_kappa_includes_kpar_contribution(make_lattice_with_band):
    lattice_low, key = make_lattice_with_band({"kpar_offset": 0.0})
    lattice_high, _ = make_lattice_with_band({"kpar_offset": 0.5})

    kappa_low = lattice_low.kappa(0.0)
    kappa_high = lattice_high.kappa(0.0)

    i, f = lattice_low.get_index(key[0]), lattice_low.get_index(key[1])
    assert kappa_high[f, i] > kappa_low[f, i]
    assert np.isclose(kappa_high[f, i] ** 2 - kappa_low[f, i] ** 2, 0.5**2, atol=1e-6)
