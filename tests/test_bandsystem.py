import numpy as np
import pytest


def test_add_and_get_band_transition(make_bandsystem, make_bandtransition):
    bandsystem = make_bandsystem()
    band = make_bandtransition()

    bandsystem.add_band_transition("GS", "VB", band)
    assert bandsystem.band_transition_dict == {("GS", "VB"): band}

    bandsystem.add_band_transition("VB", "GS", band)
    assert bandsystem.band_transition_dict == {("GS", "VB"): band, ("VB", "GS"): band}


def test_add_band_transition_wrong_type(make_bandsystem):
    bandsystem = make_bandsystem()
    with pytest.raises(TypeError) as e_info:
        bandsystem.add_band_transition("GS", "VB", "not a band")
    assert "band must be BandTransition" in str(e_info.value)


def test_add_band_transition_invalid_charging_pair(
    make_bandsystem, make_bandtransition
):
    bandsystem = make_bandsystem()
    band = make_bandtransition()
    with pytest.raises(ValueError):
        bandsystem.add_band_transition("VB", "CB", band)


def test_band_transition_dict_setter(make_bandsystem, make_bandtransition):
    bandsystem = make_bandsystem()
    band = make_bandtransition()
    bandsystem.add_band_transition("GS", "CB", band)
    bandsystem.band_transition_dict = {("GS", "VB"): band}
    assert bandsystem.band_transition_dict == {("GS", "VB"): band}


def test_band_transition_dict_setter_wrong_type(make_bandsystem):
    bandsystem = make_bandsystem()
    with pytest.raises(TypeError) as e_info:
        bandsystem.band_transition_dict = "not a dict"
    assert "bands must be dict" in str(e_info.value)


def test_band_energy(make_bandsystem_with_band):
    bandsystem, key = make_bandsystem_with_band()

    energy_plus = bandsystem.band_energy(key)
    assert np.isclose(energy_plus.max(), -0.49, atol=1e-9)
    assert np.isclose(energy_plus.min(), -1.50, atol=1e-9)

    band = bandsystem.band_transition_dict[key]
    bandsystem.add_band_transition("GS", "CB", band)
    energy_minus = bandsystem.band_energy(("GS", "CB"))
    assert np.isclose(energy_minus.min(), 0.29, atol=1e-9)
    assert np.isclose(energy_minus.max(), 1.30, atol=1e-9)


def test_band_kappa_shape_default_squeeze(make_bandsystem_with_band):
    bandsystem, key = make_bandsystem_with_band()

    energy, kappa = bandsystem.band_kappa(key, bias=0.0)
    assert energy.shape == kappa.shape


def test_band_kappa_increases_with_kpar(make_bandsystem_with_band):
    bandsystem, key = make_bandsystem_with_band()

    # band.kpar_offset = 0.
    _, kappa_low = bandsystem.band_kappa(key, bias=0.0)

    bandsystem.band_transition_dict[key].kpar_offset = 1.0
    _, kappa_high = bandsystem.band_kappa(key, bias=0.0)
    assert kappa_high[0] > kappa_low[0]


def test_band_kappa_full_mode_increases_with_bias(make_bandsystem_with_band):
    bandsystem, key = make_bandsystem_with_band(kappa_mode="full")

    _, kappa_low = bandsystem.band_kappa(key, bias=0.0)
    _, kappa_high = bandsystem.band_kappa(key, bias=1.0)
    assert np.all(kappa_high > kappa_low)


def test_band_kappa_multi_bias_shape(make_bandsystem_with_band):
    bandsystem, key = make_bandsystem_with_band(kappa_mode="full")

    bias = np.array([0.0, 0.5, 1.0])
    energy, kappa = bandsystem.band_kappa(
        key, bias=bias, kappa_mode="full", squeeze=False
    )
    assert kappa.shape == (3, len(energy))


def test_get_band_charging_rate_shape(make_bandsystem_with_band):
    bandsystem, key = make_bandsystem_with_band()

    rate = bandsystem.get_band_charging_rate(key, z=5.0, bias=-1.0)
    assert np.isscalar(rate) or rate.shape == ()
    assert rate > 0


def test_get_band_charging_rate_increases_with_bias_magnitude(
    make_bandsystem_with_band,
):
    # Within the band's energy domain, integrating further into the band
    # (more negative bias here) should monotonically increase the rate.
    bandsystem, key = make_bandsystem_with_band()

    rate_shallow = bandsystem.get_band_charging_rate(key, z=5.0, bias=-0.6)
    rate_deep = bandsystem.get_band_charging_rate(key, z=5.0, bias=-1.3)
    assert rate_deep > rate_shallow


def test_get_band_charging_rate_shape_multi_z_bias(make_bandsystem_with_band):
    bandsystem, key = make_bandsystem_with_band()

    z = np.array([5.0, 6.0])
    bias = np.array([0.0, 0.1, 0.2])
    rate = bandsystem.get_band_charging_rate(key, z=z, bias=bias, squeeze=False)
    assert rate.shape == (2, 3)


def test_charging_rates_shape(make_bandsystem_with_band):
    bandsystem, _ = make_bandsystem_with_band()
    rates = bandsystem.charging_rates(z=5.0, bias=0.0)
    assert rates.shape == (3, 3)


def test_charging_rates_multi_z_bias_shape(make_bandsystem_with_band):
    bandsystem, _ = make_bandsystem_with_band()
    rates = bandsystem.charging_rates(
        z=np.array([5.0, 6.0]), bias=np.array([-0.1, 0.0, 0.1, 0.2])
    )
    assert rates.shape == (2, 4, 3, 3)


def test_kpar_zero_without_band_transitions(make_bandsystem):
    bandsystem = make_bandsystem()
    assert np.array_equal(bandsystem.kpar_offsets, np.zeros((3, 3)))


def test_kpar_symmetric_and_matches_band_offset(make_bandsystem_with_band):
    bandsystem, key = make_bandsystem_with_band({"kpar_offset": 0.3})

    i, f = bandsystem.get_index(key[0]), bandsystem.get_index(key[1])
    kpar = bandsystem.kpar_offsets
    assert kpar[i, f] == 0.3
    assert kpar[f, i] == 0.3
    assert np.array_equal(kpar, kpar.T)


def test_kappa_includes_kpar_contribution(make_bandsystem_with_band):
    bandsystem_low, key = make_bandsystem_with_band({"kpar_offset": 0.0})
    bandsystem_high, _ = make_bandsystem_with_band({"kpar_offset": 0.5})

    kappa_low = bandsystem_low.kappa(0.0)
    kappa_high = bandsystem_high.kappa(0.0)

    i, f = bandsystem_low.get_index(key[0]), bandsystem_low.get_index(key[1])
    assert kappa_high[f, i] > kappa_low[f, i]
    assert np.isclose(kappa_high[f, i] ** 2 - kappa_low[f, i] ** 2, 0.5**2, atol=1e-6)
