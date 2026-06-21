import numpy as np
import pytest
from meqpy.system import Dyson


def test_init_with_kwargs(make_molecule):
    molecule = make_molecule(tip_radius=3.0, padding=2)
    assert molecule.tip_radius == 3.0
    assert molecule.padding == 2


def test_add_and_get_dyson(make_molecule_with_dyson):
    molecule, dyson = make_molecule_with_dyson("cartesian")
    assert molecule.dyson_dict == {("GS", "PIR"): dyson}
    assert molecule.missing_dysons == {("GS", "NIR")}


def test_add_dyson_wrong_type(make_molecule):
    molecule = make_molecule()
    with pytest.raises(TypeError) as e_info:
        molecule.add_dyson("GS", "PIR", "not a dyson")
    assert str(e_info.value) == "dyson must be Dyson, but got str"


def test_add_dyson_invalid_charging_pair(make_molecule, cube_path):
    molecule = make_molecule()
    dyson = Dyson(cube_path("cartesian"))

    # PIR -> NIR differs in charge by 2, not a valid charging transition
    with pytest.raises(ValueError) as e_info:
        molecule.add_dyson("PIR", "NIR", dyson)
    assert str(e_info.value) == "States must differ in charge and multiplicity by 1."


def test_dyson_dict_setter(make_molecule, cube_path):
    molecule = make_molecule()
    dyson = Dyson(cube_path("cartesian"))

    molecule.dyson_dict = {("GS", "PIR"): dyson}
    assert molecule.dyson_dict == {("GS", "PIR"): dyson}


def test_dyson_dict_setter_wrong_type(make_molecule):
    molecule = make_molecule()
    with pytest.raises(TypeError) as e_info:
        molecule.dyson_dict = "not a dict"
    assert str(e_info.value) == "dysons must be dict, but got str"


def test_dyson_dict_setter_wrong_key_type(make_molecule, cube_path):
    molecule = make_molecule()
    dyson = Dyson(cube_path("cartesian"))
    with pytest.raises(TypeError) as e_info:
        molecule.dyson_dict = {"GS": dyson}
    assert str(e_info.value) == "key must be tuple, but got str"


def test_dyson_dict_setter_wrong_key_length(make_molecule, cube_path):
    molecule = make_molecule()
    dyson = Dyson(cube_path("cartesian"))
    with pytest.raises(TypeError) as e_info:
        molecule.dyson_dict = {("GS",): dyson}
    assert str(e_info.value) == "key must have length 2, but got Iterator of length 1"


def test_dyson_key_to_indices(make_molecule_with_dyson):
    molecule, _ = make_molecule_with_dyson("cartesian")
    assert molecule.dyson_key_to_indices(("GS", "PIR")) == (0, 1)


def test_dyson_key_to_indices_wrong_length(make_molecule):
    molecule = make_molecule()
    with pytest.raises(ValueError) as e_info:
        molecule.dyson_key_to_indices(("GS", "PIR", "NIR"))
    assert str(e_info.value) == "key must be of length 2."


def test_dyson_shape_and_molecule_shape(make_molecule_with_dyson):
    molecule, dyson = make_molecule_with_dyson("cartesian", padding=2)

    nx, ny = dyson.shape
    assert molecule.dyson_shape == (nx + 4, ny + 4)
    assert molecule.shape == (nx + 4, ny + 4, 3, 3)


def test_dyson_shape_mismatch_raises(make_molecule_with_dyson, cube_path):
    molecule, dyson = make_molecule_with_dyson("cartesian")

    # patch in a second dyson with a different shape to trigger the mismatch
    other = Dyson(cube_path("cartesian"))
    other.data = other.data[:-1, :]
    molecule.add_dyson("GS", "NIR", other)

    with pytest.raises(ValueError) as e_info:
        molecule.dyson_shape
    assert str(e_info.value) == "Dysons do not have same shape."


def test_dyson_amplitudes(make_molecule, cube_path):
    molecule = make_molecule()
    dyson_gs_pir = Dyson(cube_path("cartesian"))
    dyson_gs_pir.amplitude = 2.0
    molecule.add_dyson("GS", "PIR", dyson_gs_pir)

    amps = molecule.dyson_amplitudes
    expected = np.zeros((3, 3))
    expected[0, 1] = 2.0
    expected[1, 0] = 2.0
    assert np.allclose(amps, expected, atol=1e-6)


def test_charging_rates_scaled_by_dyson_amplitude(make_molecule, cube_path):
    molecule = make_molecule()
    dyson = Dyson(cube_path("cartesian"))
    dyson.amplitude = 2.0
    molecule.add_dyson("GS", "PIR", dyson)

    rates_scaled = molecule.charging_rates(z=5.0, bias=0.0, scale_by_dyson=True)
    rates_unscaled = molecule.charging_rates(z=5.0, bias=0.0, scale_by_dyson=False)

    assert rates_scaled.shape == (3, 3)
    # scaled rates should be exactly `amplitude` times the unscaled ones,
    # wherever a Dyson orbital was assigned
    assert np.allclose(rates_scaled[0, 1], 2.0 * rates_unscaled[0, 1], atol=1e-9)


def test_charging_rates_dyson(make_molecule_with_dyson):
    molecule, _ = make_molecule_with_dyson("cartesian", padding=5)

    bias = np.linspace(-1, 1, 5)
    with pytest.warns() as w_info:
        rates = molecule.charging_rates_dyson(z=5.0, bias=bias)

    assert rates.shape == (30, 30, 5, 3, 3)

    warn_msg = (
        "Some charging transition have not been assigned a Dyson instance."
        " Will scale corresponding transitions to zero."
    )
    assert str(w_info[0].message) == warn_msg


def test_get_xy_points(make_molecule_with_dyson):
    molecule, _ = make_molecule_with_dyson("rhombic", padding=5)
    assert molecule.get_xy_indices((-1.41336, -0.65322)) == (7, 8)


def test_charging_rates_pointspec(make_molecule_with_dyson):
    molecule, _ = make_molecule_with_dyson("cartesian", padding=5)

    xy_rates = molecule.charging_rates_dyson(z=5.0, bias=0.0)
    points = [(7, 8), (14, 3)]
    point_rates = molecule.charging_rates_pointspec(
        points=points,
        z=5.0,
        bias=0.0,
    )

    assert len(point_rates) == len(points)
    assert np.allclose(point_rates[0], xy_rates[7, 8, ...], atol=1e-9)
    assert np.allclose(point_rates[1], xy_rates[14, 3, ...], atol=1e-9)
