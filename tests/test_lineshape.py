import numpy as np
import pytest

from meqpy.utils.lineshape import (
    LineShape,
    lineshape_integral,
    evaluate_lineshape,
    dirac_lineshape_integral,
    gaussian_lineshape_integral,
    lorentzian_lineshape_integral,
)


class TestBuiltinLineshapeMath:
    """Check the closed-form integrals directly, independent of System."""

    pytestmark = pytest.mark.parametrize(
        "integral_fn",
        [gaussian_lineshape_integral, lorentzian_lineshape_integral],
    )

    def test_symmetry_around_zero(self, integral_fn):
        # F(-x) == 1 - F(x) for a symmetric, normalized lineshape.
        x = np.linspace(0.01, 5.0, 25)
        hwhm = 0.7
        assert np.allclose(integral_fn(-x, hwhm), 1 - integral_fn(x, hwhm))

    def test_bounded_in_unit_interval(self, integral_fn):
        x = np.linspace(-10, 10, 201)
        out = integral_fn(x, hwhm=1.0)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_monotonically_nondecreasing(self, integral_fn):
        x = np.linspace(-10, 10, 200)
        out = integral_fn(x, hwhm=1.0)
        assert np.all(np.diff(out) >= 0)

    def test_narrower_hwhm_approaches_step(self, integral_fn):
        # Smaller hwhm -> steeper transition -> closer to Heaviside at fixed x != 0.
        x = 0.5
        narrow_val = integral_fn(x, 0.1)
        wide_val = integral_fn(x, 0.2)
        assert narrow_val > wide_val


class TestLineshapeIntegral:
    @pytest.mark.parametrize("name", ["gaussian", "lorentzian", "dirac"])
    def test_accepts_string_and_enum(self, name):
        val_from_str = lineshape_integral(name, 0.3, hwhm=1.0)
        val_from_enum = lineshape_integral(LineShape(name), 0.3, hwhm=1.0)
        assert val_from_str == val_from_enum

    def test_zero_hwhm_forces_dirac(self):
        # hwhm == 0 short-circuits to LineShape.DIRAC even if "gaussian" was requested.
        x = np.linspace(-1.0, 1.0, 200)
        result = lineshape_integral("gaussian", x, hwhm=0.0)
        assert np.allclose(result, dirac_lineshape_integral(x))

    def test_invalid_lineshape_string_raises(self):
        with pytest.raises(ValueError):
            lineshape_integral("not_a_real_lineshape", 0.0, hwhm=1.0)

    def test_negative_hwhm_raises(self):
        with pytest.raises(ValueError):
            lineshape_integral("gaussian", 0.0, hwhm=-1.0)

    def test_lineshape_integral_does_not_accept_callable(self):
        with pytest.raises(TypeError):
            lineshape_integral(lambda x: x, 0.0, hwhm=1.0)


class TestCustomLineshapeValidation:
    def test_valid_callable_passes_through(self):
        def func(x):
            return np.clip(0.5 + x, 0.0, 1.0)

        x = np.linspace(-10, 10, 250).reshape(-1, 5, 5)
        out = evaluate_lineshape(func, x)
        assert out.shape == x.shape
        assert np.allclose(out, func(x))

    def test_non_callable_raises_typeerror(self):
        with pytest.raises(TypeError):
            evaluate_lineshape("not_callable", np.zeros((2, 2)))

    def test_callable_raises_valueerror(self):
        def broken(x):
            raise RuntimeError("boom")

        with pytest.raises(ValueError, match="raised an exception"):
            evaluate_lineshape(broken, np.zeros((2, 2)))

    def test_non_ndarray_output_raises_typeerror(self):
        def func(x):
            return 0.5

        with pytest.raises(TypeError):
            evaluate_lineshape(func, np.zeros((2, 2)))

    def test_shape_mismatch_raises_valueerror(self):
        def func(x):
            return np.zeros(x.shape + (1,))  # wrong shape

        with pytest.raises(ValueError, match="same shape"):
            evaluate_lineshape(func, np.zeros((3, 3)))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16, float])
    def test_accepts_all_float_dtypes(self, dtype):
        def func(x):
            return x.astype(dtype) * 0 + 0.5

        out = evaluate_lineshape(func, np.zeros((2, 2)))
        assert out.dtype == dtype

    def test_integer_dtype_output_raises_typeerror(self):
        def func(x):
            return np.zeros(x.shape, dtype=int)

        with pytest.raises(TypeError):
            evaluate_lineshape(func, np.zeros((2, 2)))
