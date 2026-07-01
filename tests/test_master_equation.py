import numpy as np
import pytest

from meqpy.master_equation.equilibrium import (
    fill_diagonal,
    solve_equilibrium,
)


# ---------------------------------------------------------------------------
# fill_diagonal
# ---------------------------------------------------------------------------


class TestFillDiagonal:
    def test_not_ndarray_raises(self):
        with pytest.raises(TypeError) as e_info:
            fill_diagonal([[0, 1], [1, 0]])
        assert "W must be np.ndarray, but got list" in str(e_info.value)

    def test_non_square_raises(self):
        with pytest.raises(ValueError) as e_info:
            fill_diagonal(np.ones((3, 4)))
        assert "square matrix" in str(e_info.value)

    def test_negative_offdiagonal_raises(self):
        W = np.ones((3, 3)) * -1
        with pytest.raises(ValueError) as e_info:
            fill_diagonal(W)
        assert "off-diagonal elements" in str(e_info.value)

    def test_all_columns_sum_to_zero(self):
        W = np.arange(9).reshape(3, 3)
        W = fill_diagonal(W)
        assert np.allclose(W.sum(axis=0), 0.0)

    def test_off_diagonal_values_unchanged(self):
        W = np.arange(9).reshape(3, 3)
        Wf = fill_diagonal(W.copy())
        rows, cols = np.indices(W.shape)
        off_diag = rows != cols
        assert np.array_equal(Wf[off_diag], W[off_diag])

    def test_output_shape_preserved(self):
        W = np.arange(16).reshape(4, 4)
        W = fill_diagonal(W)
        assert W.shape == (4, 4)

    def test_nd_stack_all_columns_sum_to_zero(self):
        W = np.arange(36).reshape(4, 3, 3)
        W = fill_diagonal(W)
        assert np.allclose(W.sum(axis=-2), 0.0)


# ---------------------------------------------------------------------------
# solve_equilibrium
# ---------------------------------------------------------------------------


class TestSolveEquilibrium:
    # --- validation ---------------------------------------------------------

    def test_W_not_ndarray_raises(self):
        with pytest.raises(TypeError) as e_info:
            solve_equilibrium([[0, 1], [1, 0]])
        assert "W must be np.ndarray, but got list" in str(e_info.value)

    def test_W_not_square_raises(self):
        with pytest.raises(ValueError) as e_info:
            solve_equilibrium(np.ones((2, 3, 4)))
        assert "square matrix" in str(e_info.value)

    def test_negative_offdiagonal_raises(self):
        W = np.ones((3, 3, 3)) * -1
        with pytest.raises(ValueError) as e_info:
            fill_diagonal(W)
        assert "off-diagonal elements" in str(e_info.value)

    def test_tol_negative_raises(self):
        W = np.ones((2, 3, 3))
        with pytest.raises(ValueError) as e_info:
            solve_equilibrium(W, tol=-1.0)
        assert "tol must be >= 0" in str(e_info.value)

    def test_tol_wrong_type_raises(self):
        W = np.ones((2, 3, 3))
        with pytest.raises(TypeError) as e_info:
            solve_equilibrium(W, tol="small")
        assert "tol must be Real" in str(e_info.value)

    def test_anchor_raises(self):
        W = np.ones((2, 3, 3))
        with pytest.raises(ValueError) as e_info:
            solve_equilibrium(W, anchor=-1)
        assert "anchor must be >= 0" in str(e_info.value)
        with pytest.raises(ValueError) as e_info:
            solve_equilibrium(W, anchor=4)
        assert "anchor must be smaller" in str(e_info.value)

    def test_fuzz_negative_raises(self):
        W = np.ones((2, 3, 3))
        with pytest.raises(ValueError) as e_info:
            solve_equilibrium(W, fuzz=-1e-20)
        assert "fuzz must be >= 0" in str(e_info.value)

    def test_fuzz_wrong_type_raises(self):
        W = np.ones((2, 3, 3))
        with pytest.raises(TypeError) as e_info:
            solve_equilibrium(W, fuzz="small")
        assert "fuzz must be Real" in str(e_info.value)

    # --- output properties --------------------------------------------------

    def test_output_shape(self):
        W = np.arange(36).reshape(4, 3, 3)
        Peq = solve_equilibrium(W)
        assert Peq.shape == (4, 3)

    def test_peq_sums_to_one(self):
        W = np.arange(36).reshape(4, 3, 3)
        Peq = solve_equilibrium(W)
        assert np.allclose(Peq.sum(axis=-1), 1.0)

    def test_output_all_nonnegative(self):
        W = np.arange(36).reshape(4, 3, 3)
        Peq = solve_equilibrium(W, tol=1e-12)
        assert np.all(Peq >= -1e-12)

    def test_residual_near_zero(self):
        W = np.arange(36).reshape(4, 3, 3)
        W = fill_diagonal(W)
        Peq = solve_equilibrium(W)
        residuals = np.einsum("...ij,...j->...i", W, Peq)
        assert np.allclose(residuals, 0.0, atol=1e-10)

    # --- known analytical solutions -----------------------------------------

    def test_balanced_two_state_system(self):
        # Equal forward and backward rates -> uniform distribution.
        W = np.ones((2, 2)) - np.eye(2)
        Peq = solve_equilibrium(W)
        assert np.allclose(Peq, [0.5, 0.5])

    def test_asymmetric_two_state_system(self):
        W = np.array([[0, 3], [1, 0]])
        Peq = solve_equilibrium(W)
        assert np.allclose(Peq, [0.75, 0.25])

    def test_pure_decay_chain_all_weight_at_end(self):
        # Irreversible 2 -> 1 -> 0 chain
        W = np.diag([1.0, 1.0], k=1)
        Peq = solve_equilibrium(W)
        assert np.isclose(Peq[0], 1.0)
        assert np.isclose(Peq[1], 0.0)
        assert np.isclose(Peq[2], 0.0)

    # --- edge / warning cases -----------------------------------------------

    def test_anchor(self):
        # Irreversible 2 -> 1 -> 0 chain, choose different anchor
        W = np.diag([1.0, 1.0], k=-1)
        with pytest.raises(ValueError) as e_info:
            solve_equilibrium(W)
        assert "Rate Matrix W is ill defined" in str(e_info.value)
        Peq = solve_equilibrium(W, anchor=2)
        assert np.isclose(Peq[0], 0.0)
        assert np.isclose(Peq[1], 0.0)
        assert np.isclose(Peq[2], 1.0)

    def test_fuzz(self):
        # Irreversible 2 -> 1 -> 0 chain, use fuzzing to solve
        W = np.diag([1.0, 1.0], k=-1)
        with pytest.raises(ValueError) as e_info:
            solve_equilibrium(W)
        assert "Rate Matrix W is ill defined" in str(e_info.value)
        Peq = solve_equilibrium(W, fuzz=1e-20)
        assert np.isclose(Peq[0], 0.0)
        assert np.isclose(Peq[1], 0.0)
        assert np.isclose(Peq[2], 1.0)
