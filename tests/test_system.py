import numpy as np
import pytest
from meqpy.system import State, System


def make_system(**kwargs):
    """Build a small 3-state system: GS, PIR, NIR."""
    states = [
        State("GS", 0.0, 0, multiplicity=1),
        State("PIR", 0.5, 1, multiplicity=2),
        State("NIR", 0.3, -1, multiplicity=2),
    ]
    defaults = dict(hwhm=0.0, lineshape="dirac", workfunction=5.0, kappa_mode="10")
    defaults.update(kwargs)
    return System(states=states, **defaults)


def make_system_with_high_spin_state(**kwargs):
    """Like make_system, but with an added quartet state (M=4)."""

    system = make_system(**kwargs)
    system.add_state(State("Q", 1.5, 1, multiplicity=4))
    return system


def test_add_and_get_state():
    system = System()
    state = State("a", 1.0, 0)
    system.add_state(state)

    assert system.num_states == 1
    assert system.get_state("a") is state
    assert system.get_state(0) is state
    assert system.get_index("a") == 0


def test_add_state_overwrites_existing_label():
    system = System(states=[State("a", 0.0, 0)])
    system.add_state(State("a", 5.0, 0))

    assert system.num_states == 1
    assert system.get_state("a").energy == 5.0


def test_add_state_wrong_type():
    system = System()
    with pytest.raises(TypeError) as e_info:
        system.add_state("not a state")
    assert str(e_info.value) == "state must be State, but got str"


def test_get_index_missing_label():
    system = make_system()
    with pytest.raises(ValueError) as e_info:
        system.get_index("missing")
    assert str(e_info.value) == "State with label missing not found in the system."


def test_spin_selection_rule_wrong_type():
    with pytest.raises(TypeError) as e_info:
        System(spin_selection_rule="yes")
    assert str(e_info.value).startswith("spin_selection_rule must be bool")


def test_energies_charges_multiplicities():
    system = make_system()
    assert np.allclose(system.energies, [0.0, 0.5, 0.3])
    assert np.array_equal(system.charges, [0, 1, -1])
    assert np.array_equal(system.multiplicities, [1, 2, 2])


def test_shape_zeros_ones():
    system = make_system()
    assert system.shape == (3, 3)
    assert np.array_equal(system.zeros, np.zeros((3, 3)))
    assert np.array_equal(system.ones, np.ones((3, 3)))


def test_dE_dQ_dM():
    system = make_system()

    dE = np.array(
        [
            [0.0, -0.5, -0.3],
            [0.5, 0.0, 0.2],
            [0.3, -0.2, 0.0],
        ]
    )
    dQ = np.array(
        [
            [0, -1, 1],
            [1, 0, 2],
            [-1, -2, 0],
        ]
    )
    dM = np.array(
        [
            [0, -1, -1],
            [1, 0, 0],
            [1, 0, 0],
        ]
    )

    assert np.allclose(system.dE, dE, atol=1e-6)
    assert np.array_equal(system.dQ, dQ)
    assert np.array_equal(system.dM, dM)


def test_matrix_by_states():
    system = make_system()

    mat = system.matrix_by_states("GS", "PIR")
    expected = np.zeros((3, 3))
    expected[1, 0] = 1.0
    assert np.array_equal(mat, expected)

    mat_sym = system.matrix_by_states("GS", "PIR", symmetric=True)
    expected[0, 1] = 1.0
    assert np.array_equal(mat_sym, expected)

    # works with indices too
    mat_idx = system.matrix_by_states(0, 1)
    expected_idx = np.zeros((3, 3))
    expected_idx[1, 0] = 1.0
    assert np.array_equal(mat_idx, expected_idx)


def test_rescale_by_states():
    system = make_system()

    mat = system.rescale_by_states("GS", "PIR", 2.0)
    expected = np.ones((3, 3))
    expected[1, 0] = 2.0
    assert np.array_equal(mat, expected)

    mat_sym = system.rescale_by_states("GS", "PIR", 2.0, symmetric=True)
    expected[0, 1] = 2.0
    assert np.array_equal(mat_sym, expected)


def test_rescale_by_states_wrong_value_type():
    system = make_system()
    with pytest.raises(TypeError) as e_info:
        system.rescale_by_states("GS", "NIR", "two")
    assert "value must be Real, but got str" in str(e_info.value)


def test_clebsch_gordan_factors():
    system = make_system()

    cg = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
        ]
    )
    assert np.allclose(system.clebsch_gordan_factors, cg, atol=1e-6)


def test_normalized_charging_transitions_with_spin_rule():
    system = make_system_with_high_spin_state()

    # only transitions with |dQ|==1 AND |dM|==1 survive;
    # Q -> GS (|dQ|=1, |dM|=3) is excluded here
    w = system.normalized_charging_transitions(0.0)
    expected = np.zeros((4, 4))
    expected[0, 1] = 1.0
    expected[0, 2] = 1.0
    assert np.allclose(w, expected, atol=1e-6)


def test_normalized_charging_transitions_without_spin_rule():
    system = make_system_with_high_spin_state(spin_selection_rule=False)

    # only |dQ|==1 is required now; Q -> GS is now allowed too
    w = system.normalized_charging_transitions(0.0)
    expected = np.zeros((4, 4))
    expected[0, 1] = 1.0
    expected[0, 2] = 1.0
    expected[0, 3] = 1.0
    assert np.allclose(w, expected, atol=1e-6)


def test_kappa_modes():
    system = make_system(kappa_mode="10")
    assert np.allclose(system.kappa(0.0), np.log(10) / 2.0, atol=1e-9)

    system_const = make_system(kappa_mode="constant")
    kappa_const = system_const.kappa(0.0)
    assert np.all(kappa_const > 0)
    assert np.allclose(kappa_const, kappa_const[0, 0], atol=1e-9)

    system_full = make_system(kappa_mode="full")
    kappa_full = system_full.kappa(0.0)
    assert np.all(kappa_full > 0)
    # check if PIR tunneling is harder than NIR
    assert kappa_full[1, 0] > kappa_full[2, 0]

    kappa_biased = system_full.kappa(1.0)
    # check that pos bias increases kappa
    assert kappa_biased[1, 0] > kappa_full[1, 0]


def test_coupling_strength_shape_and_positivity():
    system = make_system()
    coupling = system.coupling_strength(z=5.0)
    assert coupling.shape == (3, 3)
    assert np.all(coupling > 0)

    # larger distance => smaller coupling strength
    coupling_far = system.coupling_strength(z=10.0)
    assert np.all(coupling_far < coupling)


def test_charging_rates_shape():
    system = make_system()
    rates = system.charging_rates(z=5.0, bias=0.0)
    assert rates.shape == (3, 3)

    rates_multi = system.charging_rates(
        z=np.array([5.0, 6.0]), bias=np.array([0.0, 0.1])
    )
    assert rates_multi.shape == (2, 2, 3, 3)


def test_states_setter_accepts_single_state():
    system = System()
    system.states = State("a", 0.0, 0)
    assert system.num_states == 1
    assert system.get_state(0).label == "a"
