"""Bounded periodic reference sensitivity and independent long-run refinement.

Run with ``python -m dysthe_experiments.pilot_sensitivity --run RUN_DIRECTORY``.
Only the three validation inlets from the CPU pilot are used. No neural model is
trained or selected. Results describe sampled finite-distance numerical behavior,
not continuum existence, whole-line stability, or physical model validity.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import time


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sobolev_norm(fields, length, order=1):
    """Periodic H^order norm with physical quadrature and profile axis last."""
    import numpy as np
    n = fields.shape[-1]
    k = 2 * np.pi * np.fft.fftfreq(n, d=length / n)
    coefficients = np.fft.fft(fields, axis=-1) / n
    return np.sqrt(length * np.sum((1 + k * k)**order * abs(coefficients)**2, axis=-1))


def _relative_error(approximate, reference, length, order):
    import numpy as np
    from scipy.signal import resample
    if approximate.shape[-1] != reference.shape[-1]:
        approximate = resample(approximate, reference.shape[-1], axis=-1)
    denominator = sobolev_norm(reference, length, order)
    if np.any(denominator <= np.finfo(float).tiny):
        raise RuntimeError("A zero reference norm prevents a relative comparison")
    return sobolev_norm(approximate - reference, length, order) / denominator


def _direction(u0, length):
    """Fixed complex modes 2 and 5, normalized in H1; no symmetry removal."""
    import numpy as np
    n = len(u0)
    tau = np.arange(n) * length / n - length / 2
    h = np.exp(4j * np.pi * tau / length) + (.6 + .4j) * np.exp(10j * np.pi * tau / length)
    h /= sobolev_norm(h, length)
    k = 2 * np.pi * np.fft.fftfreq(n, d=length / n)
    tangent = np.stack((1j * u0, np.fft.ifft(1j * k * np.fft.fft(u0))), axis=1)
    # Phase/translation shifts have real coefficients. Measure the distance to
    # their tangent span, but retain the same declared perturbation for all cases.
    real_matrix = np.concatenate((tangent.real, tangent.imag), axis=0)
    real_h = np.concatenate((h.real, h.imag))
    coefficients = np.linalg.lstsq(real_matrix, real_h, rcond=None)[0]
    complement = h - tangent @ coefficients
    non_symmetry_fraction = float(sobolev_norm(complement, length, 0)
                                  / sobolev_norm(h, length, 0))
    if non_symmetry_fraction < 1e-6:
        raise RuntimeError("Declared perturbation is indistinguishable from a symmetry direction")
    return h, non_symmetry_fraction


def sensitivity(config, records, data):
    import numpy as np
    from .cpu_pilot import integrate

    length = config["length"]
    positions = np.linspace(0., 4., 81)
    rows = []
    configurations = (("delta", 512, .0025, 1e-3),
                      ("fine", 512, .0025, 5e-4),
                      ("grid_coarse", 256, .0025, 5e-4),
                      ("step_coarse", 512, .005, 5e-4))
    for index, record in enumerate(records):
        if record["split"] != "validation":
            continue
        u0 = data["high"][index, 0]
        h, complement = _direction(u0, length)
        response, costs = {}, []
        for name, n, step, delta in configurations:
            positive, plus_cost = integrate(u0 + delta * h, record["physics"]["epsilon"],
                                             length, n, step, positions)
            negative, minus_cost = integrate(u0 - delta * h, record["physics"]["epsilon"],
                                              length, n, step, positions)
            response[name] = (positive - negative) / (2 * delta)
            costs.append({"configuration": name, "n": n, "step": step, "delta": delta,
                          "positive_seconds": plus_cost, "negative_seconds": minus_cost})
        fine = response["fine"]
        initial_error = float(_relative_error(fine[0], h, length, 1))
        if initial_error > 1e-8:
            raise RuntimeError("Initial centered response does not recover the declared direction")
        gain = sobolev_norm(fine, length, 1) / sobolev_norm(h, length, 1)
        errors = {key: _relative_error(response[key], fine, length, 1)
                  for key in ("delta", "grid_coarse", "step_coarse")}
        grid_error = float(np.max(errors["grid_coarse"]))
        step_error = float(np.max(errors["step_coarse"]))
        delta_error = float(np.max(errors["delta"]))
        rows.append({
            "case_id": record["case_id"], "epsilon": record["physics"]["epsilon"],
            "direction_h1_norm": float(sobolev_norm(h, length, 1)),
            "initial_centered_response_relative_h1_error": initial_error,
            "direction_l2_fraction_outside_phase_translation_tangent_span": complement,
            "max_h1_amplification": float(np.max(gain)),
            "endpoint_h1_amplification": float(gain[-1]),
            "delta_halving_max_relative_h1_response_error": delta_error,
            "grid_max_relative_h1_response_error": grid_error,
            "step_max_relative_h1_response_error": step_error,
            "numerical_refinement_gate_pass": max(grid_error, step_error) < .01,
            "delta_halving_gate_pass": delta_error < .01,
            "costs": costs,
            "stations": [{"xi": float(xi), "h1_amplification": float(gain[j]),
                          "delta_halving_relative_h1_response_error": float(errors["delta"][j]),
                          "grid_relative_h1_response_error": float(errors["grid_coarse"][j]),
                          "step_relative_h1_response_error": float(errors["step_coarse"][j])}
                         for j, xi in enumerate(positions)]})
        print("sensitivity", record["case_id"], "max H1 amplification", rows[-1]["max_h1_amplification"],
              "refinement gate", rows[-1]["numerical_refinement_gate_pass"], flush=True)
    return {
        "definition": "w_delta(xi) = [S_xi(u0+delta*h)-S_xi(u0-delta*h)]/(2*delta)",
        "direction": "h proportional to exp(2i*2*pi*tau/L)+(.6+.4i)*exp(5i*2*pi*tau/L), ||h||_H1=1",
        "delta_interpretation": "absolute H1 perturbation sizes 0.001 and 0.0005; same direction for both signs",
        "length": length, "xi_end": 4., "sample_step": .05,
        "numerical_gate": "Both independent grid and step differences <1% of the signed reference response H1 norm at every recorded station",
        "delta_gate": "Delta-halving difference <1% of the signed fine-delta response at every recorded station",
        "limitation": "Refinement differences are measured, not rigorous error bounds. No alignment, amplitude renormalization or symmetry reset is applied. Finite-distance directional sensitivity does not establish nonlinear stability or existence.",
        "cases": rows}


def long_refinement(config, records, data):
    import numpy as np
    from .cpu_pilot import integrate

    length = config["length"]
    positions = np.linspace(0., 16., 33)
    cases = []
    configurations = (("fine", 1024, .00125), ("grid_coarse", 512, .00125),
                      ("step_coarse", 1024, .0025))
    for index, record in enumerate(records):
        if record["split"] != "validation":
            continue
        u0 = data["high"][index, 0]
        fields, costs = {}, []
        for name, n, step in configurations:
            fields[name], seconds = integrate(u0, record["physics"]["epsilon"], length, n, step, positions)
            costs.append({"configuration": name, "n": n, "step": step, "seconds": seconds})
        fine = fields["fine"]
        errors = {name: {f"h{order}": _relative_error(fields[name], fine, length, order)
                         for order in range(3)} for name in ("grid_coarse", "step_coarse")}
        norms = {f"h{order}": sobolev_norm(fine, length, order) for order in range(3)}
        cases.append({
            "case_id": record["case_id"], "epsilon": record["physics"]["epsilon"], "costs": costs,
            "max_relative_errors": {name: {key: float(np.max(value)) for key, value in groups.items()}
                                    for name, groups in errors.items()},
            "fine_max_norms": {key: float(np.max(value)) for key, value in norms.items()},
            "stations": [{"xi": float(xi), "fine_norms": {key: float(value[j]) for key, value in norms.items()},
                          "relative_errors": {name: {key: float(value[j]) for key, value in groups.items()}
                                              for name, groups in errors.items()}}
                         for j, xi in enumerate(positions)]})
        print("long refinement", record["case_id"], cases[-1]["max_relative_errors"], flush=True)
    return {
        "length": length, "xi_end": 16., "sample_step": .5,
        "reference": {"n": 1024, "step": .00125},
        "norm_convention": "h0 is L2; hs^2 = L*sum_k (1+k^2)^s*|fft(u)/N|^2",
        "comparison": "Grid is refined at fixed step, and step at fixed grid; errors use raw complex fields, not aligned fields or norm differences",
        "limitation": "Two-level numerical refinement in a fixed periodic box. Agreement in sampled H2 does not establish all-order smoothness, continuum error bounds, global existence, or whole-line/physical validity.",
        "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--publish-dir", type=Path,
                        help="Default: repository results/<run-directory-name>")
    args = parser.parse_args()
    # Set before importing the numerical libraries. FFT loops remain serial;
    # BLAS/OpenMP pools, when used, are limited to the requested two CPU threads.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    import numpy as np
    import scipy
    from dysthe_core.contracts import require_water_wave_model, validate_splits
    from dysthe_core.reference.spatial import SpatialDysthe
    from .cpu_pilot import integrate, read_config

    run = args.run.resolve()
    config = read_config(run / "config.json")
    manifest = json.loads((run / "manifest.json").read_text())
    require_water_wave_model(manifest)
    validate_splits(manifest["records"])
    if _digest(run / "dataset.npz") != manifest["sha256"]:
        raise RuntimeError("Dataset checksum mismatch")
    selected = [r for r in manifest["records"] if r["split"] == "validation"]
    if len(selected) != 3 or {r["family"] for r in selected} != {"packet", "sideband", "random"}:
        raise ValueError("This bounded test requires the three declared pilot validation fields")
    with np.load(run / "dataset.npz", allow_pickle=False) as archive:
        data = {"high": archive["high"]}
    identity = {
        "model_id": config["model_id"], "physical_system": config["physical_system"],
        "dataset_sha256": manifest["sha256"], "config_sha256": _digest(run / "config.json"),
        "script_sha256": _digest(__file__),
        "integrator_source_sha256": hashlib.sha256(inspect.getsource(integrate).encode()).hexdigest(),
        "spatial_reference_sha256": _digest(inspect.getfile(SpatialDysthe))}
    publish = args.publish_dir or Path(__file__).resolve().parents[2] / "results" / run.name
    publish.mkdir(parents=True, exist_ok=True)
    environment = {"python": platform.python_version(), "numpy": np.__version__,
                   "scipy": scipy.__version__, "platform": platform.platform(), "threads": 2}
    for name, compute in (("sensitivity.json", sensitivity), ("long-refinement.json", long_refinement)):
        path = run / name
        if path.exists():
            result = json.loads(path.read_text())
            if result.get("identity") != identity:
                raise ValueError(f"Existing {name} has a different identity; preserve it before rerunning")
            print("reuse", path, flush=True)
        else:
            started = time.perf_counter()
            result = {"identity": identity, "environment": environment,
                      **compute(config, manifest["records"], data)}
            result["wall_seconds"] = time.perf_counter() - started
            path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        (publish / name).write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
        print("published", publish / name, flush=True)


if __name__ == "__main__":
    main()
