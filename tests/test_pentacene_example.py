"""Calculation-level checks for the evolving pentacene STM example."""

import importlib.util
import json
from pathlib import Path

import meqpy
import numpy as np
import pytest


EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "docs/source/tutorials/dyson_from_2pz_vector/pentacene_calculation.py"
)
spec = importlib.util.spec_from_file_location("pentacene_calculation", EXAMPLE)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


@pytest.fixture(scope="module")
def calculation():
    return example.run_calculation()


def test_conservation_and_zero_bias(calculation):
    assert calculation["report"]["passed"]
    a = calculation["arrays"]
    np.testing.assert_allclose(a["zero_populations"], [1.0, 0.0, 0.0])
    assert abs(a["zero_current_a"]) <= example.TOLERANCES["current_atol_a"]


@pytest.mark.parametrize("name", ["iv", "map"])
def test_analytic_three_state_populations(calculation, name):
    a = calculation["arrays"]
    rates = a[name + "_tip_rates_per_s"] + a["substrate_rates_per_s"]
    ratios = rates[..., 1:, 0] / rates[..., 0, 1:]
    p0 = 1.0 / (1.0 + ratios.sum(axis=-1))
    analytic = np.concatenate([p0[..., None], ratios * p0[..., None]], axis=-1)
    np.testing.assert_allclose(
        a[name + "_populations"], analytic, rtol=1e-12, atol=1e-14
    )


def test_original_notebook_snapshot(calculation):
    reference = json.loads(
        (Path(__file__).parent / "data/pentacene_reference.json").read_text()
    )
    a = calculation["arrays"]
    assert list(a["map_current_a"].shape) == reference["map_shape"]
    for sample in reference["iv_samples"]:
        index = sample["index"]
        assert a["bias_v"][index] == pytest.approx(sample["bias_v"])
        np.testing.assert_allclose(
            a["iv_current_a"][index] * 1e9, sample["current_na"], rtol=5e-8, atol=1e-12
        )
        np.testing.assert_allclose(
            a["iv_populations"][index], sample["populations"], rtol=5e-8, atol=1e-14
        )
    for sample in reference["map_samples"]:
        x, y = sample["index_xy"]
        np.testing.assert_allclose(
            a["map_current_a"][x, y] * 1e9,
            sample["current_na"],
            rtol=5e-8,
            atol=1e-12,
        )


def test_checks_detect_nonstationary_normalized_populations(calculation):
    a = calculation["arrays"]
    populations = a["iv_populations"].copy()
    populations[-1] = [0.0, 1.0, 0.0]
    report = example.stationary_checks(
        populations,
        a["iv_tip_rates_per_s"] + a["substrate_rates_per_s"],
        a["iv_current_a"],
        a["iv_tip_current_a"],
    )
    assert report["max_population_sum_error"] <= 1e-12
    assert report["max_relative_stationary_residual"] > 1e-3
    assert not report["passed"]


def test_saved_results_and_figures(calculation, tmp_path):
    pytest.importorskip("matplotlib", reason="Figure export needs the examples extra")
    example.write_results(calculation, tmp_path)
    with np.load(tmp_path / "results.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(
            arrays["map_current_a"], calculation["arrays"]["map_current_a"]
        )
    metadata = json.loads((tmp_path / "parameters.json").read_text())
    assert metadata["parameters"] == calculation["parameters"]
    assert metadata["versions"]["meqpy"] == meqpy.__version__
    assert all(len(digest) == 64 for digest in metadata["input_sha256"].values())
    assert json.loads((tmp_path / "checks.json").read_text())["passed"]
    for name in ("iv", "current_map", "populations"):
        assert (tmp_path / f"{name}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert (tmp_path / f"{name}.pdf").read_bytes().startswith(b"%PDF-")


def test_run_from_another_working_directory(calculation, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repeated = example.run_calculation()
    np.testing.assert_array_equal(
        repeated["arrays"]["iv_current_a"], calculation["arrays"]["iv_current_a"]
    )
    assert not list(tmp_path.iterdir())
