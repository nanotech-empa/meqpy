"""Reproducible version of the accompanying pentacene STM example.

The model and defaults follow pentacene_example.ipynb at ebc6965. This is a
model-orbital regression example, not a multireference or experimental benchmark.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform

import ase
from ase.units import Bohr
import meqpy
import numpy as np
import scipy


DATA_DIR = Path(__file__).resolve().parent
INPUT_FILES = ("pentacene.xyz", "pentacene_HOMO.dat", "pentacene_LUMO.dat")
TOLERANCES = dict(
    population_atol=1e-12,
    stationary_rtol=1e-12,
    current_atol_a=1e-20,
    current_rtol=1e-8,
)


def example_parameters():
    """Return independent, editable copies of the original example's inputs."""
    return dict(
        states=[
            dict(label="GS", energy=0.0, charge=0, multiplicity=1),
            dict(label="D0-", energy=0.8, charge=-1, multiplicity=2),
            dict(label="D0+", energy=0.4, charge=1, multiplicity=2),
        ],
        lineshape="gaussian",
        hwhm_ev=0.02,
        kappa_mode="10",
        workfunction_ev=5.0,
        reorg_shift_ev=0.0,
        spin_selection_rule=True,
        padding=0,
        cube_boundary_angstrom=5.0,
        cube_spacing_angstrom=0.333333 * Bohr,
        source_slice_z_angstrom=9.0,
        center_mass=False,
        tip_radius_angstrom=3.0,
        tip_z_angstrom=12.5,
        sample_distance_angstrom=3.0,
        point_xy_angstrom=[10.0, 11.0],
        bias_min_v=-1.2,
        bias_max_v=1.2,
        bias_points=200,
        map_bias_v=-0.6,
        # Preserve the rounded value in the original notebook for this baseline.
        elementary_charge_c=1.6022e-19,
    )


def read_carbon_sites():
    """Read sites in the same order as the original notebook and coefficient files."""
    sites = []
    for line in (DATA_DIR / "pentacene.xyz").read_text().splitlines()[2:]:
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "C":
            sites.append(list(map(float, parts[1:4])))
    return np.asarray(sites)


def stationary_checks(populations, rates, sample_current, tip_current):
    """Check stationarity and both lead currents; currents share tip-to-sample sign."""
    generator = meqpy.fill_diagonal(rates.copy())
    residual = np.einsum("...ij,...j->...i", generator, populations)
    scale = np.max(np.sum(np.abs(generator), axis=-1), axis=-1)
    relative_residual = np.max(np.abs(residual), axis=-1) / np.maximum(
        scale, np.finfo(float).tiny
    )
    probability_error = float(np.max(np.abs(populations.sum(axis=-1) - 1)))
    current_error = np.abs(sample_current - tip_current)
    current_limit = TOLERANCES["current_atol_a"] + TOLERANCES[
        "current_rtol"
    ] * np.maximum(np.abs(sample_current), np.abs(tip_current))
    finite = all(
        np.isfinite(array).all()
        for array in (populations, rates, sample_current, tip_current)
    )
    passed = (
        finite
        and probability_error <= TOLERANCES["population_atol"]
        and populations.min() >= -TOLERANCES["population_atol"]
        and np.max(relative_residual) <= TOLERANCES["stationary_rtol"]
        and np.all(current_error <= current_limit)
    )
    return dict(
        passed=bool(passed),
        finite=bool(finite),
        max_population_sum_error=probability_error,
        min_population=float(populations.min()),
        max_relative_stationary_residual=float(np.max(relative_residual)),
        max_lead_current_difference_a=float(np.max(current_error)),
    )


def run_calculation(parameters=None):
    """Run the original I(V) and current-map calculation, with extra diagnostics."""
    p = deepcopy(example_parameters() if parameters is None else parameters)
    sites = read_carbon_sites()
    molecule = meqpy.Molecule(
        states=[meqpy.State(**state) for state in p["states"]],
        lineshape=p["lineshape"],
        hwhm=p["hwhm_ev"],
        kappa_mode=p["kappa_mode"],
        workfunction=p["workfunction_ev"],
        reorg_shift=p["reorg_shift_ev"],
        spin_selection_rule=p["spin_selection_rule"],
        padding=p["padding"],
        tip_radius=p["tip_radius_angstrom"],
    )
    for label, filename in [
        ("D0+", "pentacene_HOMO.dat"),
        ("D0-", "pentacene_LUMO.dat"),
    ]:
        cube = meqpy.Cube_2pz(
            sites,
            np.loadtxt(DATA_DIR / filename),
            boundary=p["cube_boundary_angstrom"],
            spacing=p["cube_spacing_angstrom"],
        )
        molecule.add_dyson(
            "GS",
            label,
            meqpy.Dyson(
                cube,
                slice_height=p["source_slice_z_angstrom"],
                center_mass=p["center_mass"],
            ),
        )

    sample_rates = molecule.charging_rates(p["sample_distance_angstrom"], bias=0.0)
    current_operator = molecule.dQ * sample_rates * p["elementary_charge_c"]
    bias = np.linspace(p["bias_min_v"], p["bias_max_v"], p["bias_points"])
    indices = molecule.get_xy_indices(tuple(p["point_xy_angstrom"]))
    point_rates = molecule.charging_rates_pointspec(indices, p["tip_z_angstrom"], bias)
    map_rates = molecule.charging_rates_dyson(p["tip_z_angstrom"], p["map_bias_v"])
    zero_rates = molecule.charging_rates_pointspec(indices, p["tip_z_angstrom"], 0.0)

    arrays = dict(
        bias_v=bias,
        x_angstrom=molecule.x,
        y_angstrom=molecule.y,
        carbon_sites_angstrom=sites,
        point_indices=np.asarray(indices),
        point_sampled_xy_angstrom=np.array(
            [molecule.x[indices[0]], molecule.y[indices[1]]]
        ),
        substrate_rates_per_s=sample_rates,
        charge_differences=molecule.dQ,
    )
    checks = {}
    for name, tip_rates in [
        ("iv", point_rates),
        ("map", map_rates),
        ("zero", zero_rates),
    ]:
        rates = tip_rates + sample_rates
        populations = meqpy.solve_equilibrium_nd(rates)
        sample_current = meqpy.measurement(current_operator, populations)
        tip_current = meqpy.measurement(
            -molecule.dQ * tip_rates * p["elementary_charge_c"], populations
        )
        arrays.update(
            {
                name + "_populations": populations,
                name + "_current_a": sample_current,
                name + "_tip_current_a": tip_current,
                name + "_tip_rates_per_s": tip_rates,
            }
        )
        checks[name] = stationary_checks(
            populations, rates, sample_current, tip_current
        )
    checks["zero_bias_current"] = dict(
        passed=bool(
            max(abs(arrays["zero_current_a"]), abs(arrays["zero_tip_current_a"]))
            <= TOLERANCES["current_atol_a"]
        ),
        sample_current_a=float(arrays["zero_current_a"]),
        tip_current_a=float(arrays["zero_tip_current_a"]),
    )
    report = dict(
        passed=all(check["passed"] for check in checks.values()),
        tolerances=TOLERANCES.copy(),
        checks=checks,
        interpretation="Regression and internal consistency; no experimental validation.",
    )
    return dict(parameters=p, arrays=arrays, report=report)


def plot_results(result):
    """Return I(V), current-map, and population figures, without changing the data."""
    import matplotlib.pyplot as plt

    p, a = result["parameters"], result["arrays"]
    height = p["tip_z_angstrom"] - a["carbon_sites_angstrom"][:, 2].mean()
    figures = {}
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.plot(a["bias_v"], a["iv_current_a"] * 1e9, label="Substrate current")
    ax.plot(a["bias_v"], a["iv_tip_current_a"] * 1e9, "--", label="Tip current")
    ax.set(
        xlabel="Bias (V)",
        ylabel="Steady-state current (nA)",
        title=f"Pentacene model · height {height:.1f} Å",
    )
    ax.legend()
    figures["iv"] = fig

    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    image = ax.imshow(
        np.abs(a["map_current_a"].T) * 1e9,
        extent=[
            a["x_angstrom"].min(),
            a["x_angstrom"].max(),
            a["y_angstrom"].min(),
            a["y_angstrom"].max(),
        ],
        origin="lower",
        cmap="inferno",
        aspect="equal",
    )
    ax.scatter(*a["carbon_sites_angstrom"][:, :2].T, color="white", s=8)
    ax.scatter(
        *a["point_sampled_xy_angstrom"], marker="x", color="cyan", label="I(V) position"
    )
    ax.set(
        xlabel="x (Å)",
        ylabel="y (Å)",
        title=f"Height {height:.1f} Å · bias {p['map_bias_v']:.1f} V",
    )
    ax.legend()
    fig.colorbar(image, ax=ax, label="|Steady-state current| (nA)", shrink=0.7)
    figures["current_map"] = fig

    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    for index, state in enumerate(p["states"]):
        ax.plot(a["bias_v"], a["iv_populations"][:, index], label=state["label"])
    ax.set(
        xlabel="Bias (V)",
        ylabel="Population",
        title="State populations at the I(V) position",
    )
    ax.legend()
    figures["populations"] = fig
    return figures


def write_results(result, output_dir):
    """Save the complete numerical result, inputs, checks, and review figures."""
    import matplotlib
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "results.npz", **result["arrays"])
    metadata = dict(
        parameters=result["parameters"],
        units=dict(energy="eV", length="Å", rate="s^-1", current="A"),
        map_axis_order=["x", "y", "state (if present)"],
        source_slice="Absolute z coordinate; no centering or automatic rotation.",
        tip_height="Tip surface z; meqpy evaluates the orbital at z + tip_radius.",
        population_convention="Spin-multiplet populations; CG factors are included in rates.",
        current_sign="Both lead currents use the tip-to-substrate sign convention.",
        model="Supplied HOMO/LUMO site vectors and model carbon 2pz orbitals, not multireference Dyson orbitals.",
        input_sha256={
            name: hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest()
            for name in INPUT_FILES
        },
        versions=dict(
            python=platform.python_version(),
            meqpy=meqpy.__version__,
            numpy=np.__version__,
            scipy=scipy.__version__,
            ase=ase.__version__,
            matplotlib=matplotlib.__version__,
        ),
    )
    for name, value in [
        ("parameters.json", metadata),
        ("checks.json", result["report"]),
    ]:
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    for name, figure in plot_results(result).items():
        for extension in ("png", "pdf"):
            figure.savefig(output / f"{name}.{extension}", dpi=150)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_calculation()
    write_results(result, args.output_dir)
    print(json.dumps(result["report"], indent=2))
    if not result["report"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
