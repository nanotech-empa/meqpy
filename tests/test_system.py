import numpy as np
import pytest
from meqpy.system import State, System


class TestSystemStates:
    def test_add_and_get_state(self):
        system = System()
        state = State("a", 1.0, 0)
        system.add_state(state)

        assert system.num_states == 1
        assert system.get_state("a") is state
        assert system.get_state(0) is state
        assert system.get_index("a") == 0

    def test_states_setter_accepts_single_state(self):
        system = System()
        system.states = State("a", 0.0, 0)
        assert system.num_states == 1
        assert system.get_state(0).label == "a"

    def test_add_state_overwrites_existing_label(self):
        system = System(states=[State("a", 0.0, 0)])
        system.add_state(State("a", 5.0, 0))

        assert system.num_states == 1
        assert system.get_state("a").energy == 5.0

    def test_add_state_wrong_type(self):
        system = System()
        with pytest.raises(TypeError) as e_info:
            system.add_state("not a state")
        assert str(e_info.value) == "state must be State, but got str"

    def test_get_index_missing_label(self, make_system):
        system = make_system()
        with pytest.raises(ValueError) as e_info:
            system.get_index("missing")
        assert str(e_info.value) == "State with label missing not found in the system."


class TestSystemHelperFuncs:
    def test_energies_charges_multiplicities(self, make_system):
        system = make_system()
        assert np.allclose(system.energies, [0.0, 0.5, 0.3])
        assert np.array_equal(system.charges, [0, 1, -1])
        assert np.array_equal(system.multiplicities, [1, 2, 2])

    def test_shape_zeros_ones(self, make_system):
        system = make_system()
        assert system.shape == (3, 3)
        assert np.array_equal(system.zeros, np.zeros((3, 3)))
        assert np.array_equal(system.ones, np.ones((3, 3)))

    def test_dE_dQ_dM(self, make_system):
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

    def test_matrix_by_states(self, make_system):
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

    def test_rescale_by_states(self, make_system):
        system = make_system()

        mat = system.rescale_by_states("GS", "PIR", 2.0)
        expected = np.ones((3, 3))
        expected[1, 0] = 2.0
        assert np.array_equal(mat, expected)

        mat_sym = system.rescale_by_states("GS", "PIR", 2.0, symmetric=True)
        expected[0, 1] = 2.0
        assert np.array_equal(mat_sym, expected)

    def test_rescale_by_states_wrong_value_type(self, make_system):
        system = make_system()
        with pytest.raises(TypeError) as e_info:
            system.rescale_by_states("GS", "NIR", "two")
        assert "value must be Real, but got str" in str(e_info.value)

    def test_clebsch_gordan_factors(self, make_system):
        system = make_system()

        cg = np.array(
            [
                [1.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
            ]
        )
        assert np.allclose(system.clebsch_gordan_factors, cg, atol=1e-6)


class TestSystemKappa:
    def test_kappa_modes(self, make_system):
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


class TestSystemChargingRates:
    def test_coupling_strength_shape_and_positivity(self, make_system):
        system = make_system()
        coupling = system.coupling_strength(z=5.0)
        assert coupling.shape == (3, 3)
        assert np.all(coupling > 0)

        # larger distance => smaller coupling strength
        coupling_far = system.coupling_strength(z=10.0)
        assert np.all(coupling_far < coupling)

    def test_charging_rates_shape(self, make_system):
        system = make_system()
        rates = system.charging_rates(z=5.0, bias=0.0)
        assert rates.shape == (3, 3)

        rates_multi = system.charging_rates(
            z=np.array([5.0, 6.0]), bias=np.array([0.0, 0.1])
        )
        assert rates_multi.shape == (2, 2, 3, 3)


class TestSystemSpinRules:
    def test_spin_selection_rule_wrong_type(self):
        with pytest.raises(TypeError) as e_info:
            System(spin_selection_rule="yes")
        assert str(e_info.value).startswith("spin_selection_rule must be bool")

    def test_normalized_charging_transitions_with_spin_rule(self, make_system):
        system = make_system(quartet=True)

        # only transitions with |dQ|==1 AND |dM|==1 survive;
        # Q -> GS (|dQ|=1, |dM|=3) is excluded here
        w = system.normalized_charging_transitions(0.0)
        expected = np.zeros((4, 4))
        expected[0, 1] = 1.0
        expected[0, 2] = 1.0
        assert np.allclose(w, expected, atol=1e-6)

    def test_normalized_charging_transitions_without_spin_rule(self, make_system):
        system = make_system(quartet=True, spin_selection_rule=False)

        # only |dQ|==1 is required now; Q -> GS is now allowed too
        w = system.normalized_charging_transitions(0.0)
        expected = np.zeros((4, 4))
        expected[0, 1] = 1.0
        expected[0, 2] = 1.0
        expected[0, 3] = 1.0
        assert np.allclose(w, expected, atol=1e-6)


class TestSystemLineshapes:
    @pytest.mark.parametrize("value", ["gaussian", "lorentzian", "dirac"])
    def test_accepts_builtin_strings(self, make_system, value):
        system = make_system()
        system.lineshape = value
        assert system.lineshape == value

    def test_accepts_custom_callable(self, make_system):
        system = make_system()

        def func(x):
            return np.clip(0.5 + 0.05 * x, 0.0, 1.0)

        system.lineshape = func
        assert system.lineshape is func

    def test_invalid_string_raises_at_assignment(self, make_system):
        system = make_system()
        with pytest.raises(ValueError):
            system.lineshape = "not_a_lineshape"

    def test_reorg_shift_is_applied_for_builtin(self, make_system):
        system = make_system(lineshape="gaussian", hwhm=0.1)

        system.reorg_shift = 0.1
        w_with_shift = system.normalized_charging_transitions(bias=0.0)

        system.reorg_shift = 0.0
        w_without_shift = system.normalized_charging_transitions(bias=0.0)

        assert not np.allclose(w_with_shift, w_without_shift)

    def test_reorg_shift_is_ignored_for_callable(self, make_system):
        system = make_system()

        def func(x):
            return 1 / (1 + np.exp(-x / 0.1))

        system.lineshape = func

        system.reorg_shift = 0.1
        w_with_shift = system.normalized_charging_transitions(bias=0.0)

        system.reorg_shift = 0.0
        w_without_shift = system.normalized_charging_transitions(bias=0.0)

        assert np.allclose(w_with_shift, w_without_shift)

    def test_hwhm_is_ignored_for_callable(self, make_system):
        system = make_system()

        def func(x):
            return 1 / (1 + np.exp(-x / 0.1))

        system.lineshape = func

        system.hwhm = 0.1
        w1 = system.normalized_charging_transitions(bias=0.0)
        system.hwhm = 5.0
        w2 = system.normalized_charging_transitions(bias=0.0)

        assert np.allclose(w1, w2)

    def test_selection_rules_with_custom_output(self, make_system):
        # Even if the custom lineshape returns a nonzero value everywhere,
        # the dQ/dM masks in normalized_charging_transitions must still zero
        # out disallowed transitions.
        system = make_system(hwhm=0.1, reorg_shift=0.05)

        def func(x):
            return np.full_like(x, 0.9)

        system.lineshape = func
        W = system.normalized_charging_transitions(bias=0.0, squeeze=False)

        allowed = np.abs(system.dQ) == 1
        assert np.all(W[..., ~allowed] == 0.0)
        assert np.all(W[..., allowed] == pytest.approx(0.9))

    def test_custom_lineshape_matches_builtin(self, make_system):
        system = make_system(lineshape="gaussian", hwhm=0.1)
        w_builtin = system.normalized_charging_transitions(bias=0.3)

        hwhm = system.hwhm
        shift = system.reorg_shift
        sigma = hwhm / np.sqrt(2 * np.log(2))
        from scipy.special import erf

        def manual_gaussian(x):
            return 0.5 * (erf((x - shift) / (np.sqrt(2) * sigma)) + 1)

        system.lineshape = manual_gaussian
        w_custom = system.normalized_charging_transitions(bias=0.3)

        assert np.allclose(w_builtin, w_custom)
