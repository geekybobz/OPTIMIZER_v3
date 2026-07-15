"""Closed-loop v3 workflow for the universal fourth-order robust system.

Why this file exists
--------------------
This is the first real downstream-style closed-loop script for OPTIMIZER v3.  It is
not a new optimizer hidden outside the library.  It is orchestration that calls the
public v3 API in the same way a notebook should:

    opt.guesses.*       create starting controls
    opt.optimizers.*    move controls using system.evaluate/gradient
    opt.utils.*         diagnose, project, and repair hard residuals
    system.with_params  change cost weights between curriculum stages

Two workflows are supported:

``--start random``
    Start from a random Fourier pulse, run short weighted-objective stages, then try
    hard/fourth-order repair.

``--start reference``
    Start from the copied previous best control and attempt energy reduction on the
    full fourth-order constraint surface by projecting the energy gradient into the
    residual nullspace, taking a trial step, and repairing.

The generated run artifacts live under ``runs/`` and are ignored by git.  The copied
previous-best control lives under ``systems/universal_robust_4th/reference``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import optimizer as opt
from optimizer.controls import Controls
from systems.universal_robust_4th import FOURTH_ORDER_BEST_CONTROLS, system


RUN_ROOT = ROOT / "runs" / "universal_robust_4th"


@dataclass(frozen=True)
class Weights:
    """System objective prefactors used for one curriculum stage."""

    infidelity_weight: float
    lambda2: float
    lambda4: float
    energy_weight: float = 0.0

    def to_params(self) -> dict[str, float]:
        return asdict(self)


def json_default(value: Any) -> Any:
    """Convert NumPy/scientific values into JSON-safe values."""

    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return [
                {"real": float(item.real), "imag": float(item.imag)}
                for item in value.reshape(-1)
            ]
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Controls):
        return {
            "name": value.name,
            "shape": list(value.shape),
            "max_abs": value.max_abs(),
            "norm": value.norm(),
        }
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def metric_subset(metrics: dict[str, Any]) -> dict[str, float]:
    """Keep the scalar metrics needed for human review and comparisons."""

    return {
        "J": float(metrics["J"]),
        "fidelity": float(metrics["fidelity"]),
        "infidelity": float(metrics["infidelity"]),
        "F_norm2": float(metrics["F_norm2"]),
        "C_sym_norm2": float(metrics["C_sym_norm2"]),
        "energy": float(metrics["energy"]),
    }


def evaluate_report(base_params: dict[str, Any], controls: Controls) -> dict[str, Any]:
    """Evaluate controls with unit diagnostic weights for comparable metrics."""

    qsys = system(
        {
            **base_params,
            "infidelity_weight": 1.0,
            "lambda2": 1.0,
            "lambda4": 1.0,
            "energy_weight": 0.0,
        }
    )
    metrics = qsys.evaluate(controls)
    out = metric_subset(metrics)
    out["u_abs_max"] = controls.max_abs()
    out["hard_norm"] = float(np.linalg.norm(qsys.residuals(controls, name="hard")))
    out["fourth_order_norm"] = float(np.linalg.norm(qsys.residuals(controls, name="fourth_order")))
    out["max_abs"] = controls.max_abs()
    return out


def save_controls(path: Path, qsys: Any, controls: Controls) -> None:
    """Save controls using the same NPZ layout as the previous-best registry."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        t=qsys.tgrid.copy(),
        ux=controls.channel("ux", copy=True),
        uy=controls.channel("uy", copy=True),
        uz=controls.channel("uz", copy=True),
        u=controls.as_matrix(copy=True),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a compact history table when rows exist."""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def controls_plus_direction(controls: Controls, direction: Controls, *, scale: float, name: str) -> Controls:
    """Return ``controls + scale * direction`` with stable metadata."""

    return Controls.from_matrix(
        controls.spec,
        controls.as_matrix(copy=False) + float(scale) * direction.as_matrix(copy=False),
        copy=True,
        name=name,
        meta={"source": controls.name, "direction": direction.name, "scale": float(scale)},
    )


def initial_controls(args: argparse.Namespace, base_params: dict[str, Any]) -> Controls:
    """Build the requested starting pulse."""

    qsys = system(base_params)
    if args.start == "reference":
        return qsys.reference_controls()

    controls = opt.guesses.random_fourier_guess(
        qsys,
        amplitude=float(args.random_amplitude),
        modes=int(args.random_modes),
        seed=int(args.seed),
        name="random_fourier_start",
    )
    if float(args.bias_ux) != 0.0:
        matrix = controls.as_matrix(copy=True)
        matrix[controls.spec.channel_index("ux")] += float(args.bias_ux)
        controls = Controls.from_matrix(
            controls.spec,
            matrix,
            copy=False,
            name="random_fourier_start_biased",
            meta={**controls.meta, "bias_ux": float(args.bias_ux)},
        )
    return controls


def run_weighted_stage(
    *,
    label: str,
    base_params: dict[str, Any],
    controls: Controls,
    weights: Weights,
    adam_iters: int,
    lbfgs_iters: int,
    step_size: float,
    max_step_norm: float,
    trace: Any,
    rows: list[dict[str, Any]],
) -> Controls:
    """Run one short curriculum stage through public optimizer calls."""

    qsys = system({**base_params, **weights.to_params()})
    before = evaluate_report(base_params, controls)
    rows.append({"stage": label, "event": "before", **before})

    current = controls
    if int(adam_iters) > 0:
        adam = opt.optimizers.adam(
            qsys,
            current,
            maxiter=int(adam_iters),
            step_size=float(step_size),
            max_step_norm=float(max_step_norm),
            trace=trace,
            create_trace=False,
        )
        current = adam.controls.copy(name=f"{label}_adam")
        rows.append({"stage": label, "event": "after_adam", **evaluate_report(base_params, current)})

    if int(lbfgs_iters) > 0:
        lbfgs = opt.optimizers.lbfgs(
            qsys,
            current,
            maxiter=int(lbfgs_iters),
            step_size=float(step_size),
            max_step_norm=float(max_step_norm),
            trace=trace,
            create_trace=False,
        )
        current = lbfgs.controls.copy(name=f"{label}_lbfgs")
        rows.append({"stage": label, "event": "after_lbfgs", **evaluate_report(base_params, current)})

    return current


def repair_stage(
    *,
    label: str,
    base_params: dict[str, Any],
    controls: Controls,
    residual_name: str,
    maxiter: int,
    tolerance: float,
    max_step_norm: float,
    rows: list[dict[str, Any]],
) -> Controls:
    """Run a residual repair stage using the v3 repair utility."""

    qsys = system(
        {
            **base_params,
            "infidelity_weight": 0.0,
            "lambda2": 0.0,
            "lambda4": 0.0,
            "energy_weight": 0.0,
        }
    )
    before_norm = float(np.linalg.norm(qsys.residuals(controls, name=residual_name)))
    repair = opt.utils.repair_newton(
        qsys,
        controls,
        residuals=residual_name,
        method="lm",
        maxiter=int(maxiter),
        tolerance=float(tolerance),
        damping=1.0e-10,
        max_step_norm=float(max_step_norm),
        line_search=True,
        max_backtracks=6,
    )
    after = evaluate_report(base_params, repair.controls)
    rows.append(
        {
            "stage": label,
            "event": f"repair_{residual_name}",
            "before_residual_norm": before_norm,
            "after_residual_norm": repair.residual_norm,
            "repair_converged": repair.converged,
            "repair_iterations": repair.iterations,
            **after,
        }
    )
    return repair.controls.copy(name=f"{label}_repaired_{residual_name}")


def energy_polish_cycle(
    *,
    base_params: dict[str, Any],
    controls: Controls,
    cycle: int,
    step_norm: float,
    tolerance: float,
    rows: list[dict[str, Any]],
) -> tuple[Controls, bool]:
    """Try one projected energy-reduction step on the full fourth-order surface."""

    q_constraints = system(
        {
            **base_params,
            "infidelity_weight": 0.0,
            "lambda2": 0.0,
            "lambda4": 0.0,
            "energy_weight": 0.0,
        }
    )
    q_energy = system(
        {
            **base_params,
            "infidelity_weight": 0.0,
            "lambda2": 0.0,
            "lambda4": 0.0,
            "energy_weight": 1.0,
        }
    )
    before = evaluate_report(base_params, controls)
    energy_gradient = q_energy.gradient(controls)
    projected_info = opt.utils.project_gradient(
        q_constraints,
        controls,
        energy_gradient,
        residuals="fourth_order",
        return_info=True,
    )
    projected_gradient = projected_info["projected_gradient"]
    projected_norm = projected_gradient.norm()
    if projected_norm <= 1.0e-14:
        rows.append(
            {
                "stage": "energy_polish",
                "event": "projected_gradient_too_small",
                "cycle": cycle,
                "projected_gradient_norm": projected_norm,
                **before,
            }
        )
        return controls, False

    descent = projected_gradient * (-1.0 / projected_norm)
    trial = controls_plus_direction(
        controls,
        descent,
        scale=float(step_norm),
        name=f"energy_polish_trial_{cycle:04d}",
    )
    repaired = opt.utils.repair_newton(
        q_constraints,
        trial,
        residuals="fourth_order",
        method="lm",
        maxiter=3,
        tolerance=float(tolerance),
        damping=1.0e-10,
        max_step_norm=max(0.25, float(step_norm)),
        line_search=True,
        max_backtracks=5,
    )
    after = evaluate_report(base_params, repaired.controls)
    residual_guard = max(before["fourth_order_norm"] * 1.05, float(tolerance) * 10.0)
    accepted = bool(
        repaired.residual_norm <= residual_guard
        and after["energy"] < before["energy"]
        and after["F_norm2"] <= max(before["F_norm2"] * 10.0, 1.0e-12)
        and after["C_sym_norm2"] <= max(before["C_sym_norm2"] * 10.0, 1.0e-12)
    )
    rows.append(
        {
            "stage": "energy_polish",
            "event": "accepted" if accepted else "rejected",
            "cycle": cycle,
            "step_norm": float(step_norm),
            "projected_gradient_norm": projected_norm,
            "linearized_hard_leak_norm": projected_info["first_order_residual_change_norm"],
            "repair_residual_norm": repaired.residual_norm,
            "residual_guard": residual_guard,
            "before_energy": before["energy"],
            **after,
        }
    )
    if accepted:
        return repaired.controls.copy(name=f"energy_polish_accepted_{cycle:04d}"), True
    return controls, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", choices=["random", "reference"], default="random")
    parser.add_argument("--N", type=int, default=101)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--random-modes", type=int, default=8)
    parser.add_argument("--random-amplitude", type=float, default=0.08)
    parser.add_argument("--bias-ux", type=float, default=0.4 * np.pi)
    parser.add_argument("--adam-iters", type=int, default=8)
    parser.add_argument("--lbfgs-iters", type=int, default=4)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--max-step-norm", type=float, default=1.0)
    parser.add_argument("--repair-iters", type=int, default=4)
    parser.add_argument("--repair-tol", type=float, default=1.0e-9)
    parser.add_argument("--polish-cycles", type=int, default=2)
    parser.add_argument("--polish-step", type=float, default=0.25)
    args = parser.parse_args()

    if args.start == "reference" and int(args.N) != 1001:
        raise ValueError("The copied previous-best control is endpoint-sampled with N=1001.")

    run_id = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_dir = RUN_ROOT / f"v3_closed_loop_{run_id}_{args.start}_N{int(args.N)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    base_params = {
        "N": int(args.N),
        "tau": float(args.tau),
        "robust_n_dirs": 250,
        "robust_seed": int(args.seed),
        "robust_parallel": False,
    }
    q_base = system(base_params)
    reference_summary = None
    if int(args.N) == 1001:
        reference = q_base.reference_controls()
        reference_summary = evaluate_report(base_params, reference)

    trace = opt.trace(f"universal_robust_4th_{run_id}")
    rows: list[dict[str, Any]] = []
    controls = initial_controls(args, base_params)
    rows.append({"stage": "initial", "event": args.start, **evaluate_report(base_params, controls)})

    if args.start == "random":
        stages = [
            ("fidelity", Weights(10.0, 0.0, 0.0, 0.0)),
            ("second_order", Weights(10.0, 25.0, 0.0, 0.0)),
            ("fourth_order", Weights(10.0, 25.0, 25.0, 0.0)),
        ]
        for label, weights in stages:
            controls = run_weighted_stage(
                label=label,
                base_params=base_params,
                controls=controls,
                weights=weights,
                adam_iters=int(args.adam_iters),
                lbfgs_iters=int(args.lbfgs_iters),
                step_size=float(args.step_size),
                max_step_norm=float(args.max_step_norm),
                trace=trace,
                rows=rows,
            )
        controls = repair_stage(
            label="hard_repair",
            base_params=base_params,
            controls=controls,
            residual_name="hard",
            maxiter=int(args.repair_iters),
            tolerance=float(args.repair_tol),
            max_step_norm=float(args.max_step_norm),
            rows=rows,
        )
        controls = repair_stage(
            label="fourth_order_repair",
            base_params=base_params,
            controls=controls,
            residual_name="fourth_order",
            maxiter=int(args.repair_iters),
            tolerance=float(args.repair_tol),
            max_step_norm=float(args.max_step_norm),
            rows=rows,
        )

    for cycle in range(1, int(args.polish_cycles) + 1):
        controls, accepted = energy_polish_cycle(
            base_params=base_params,
            controls=controls,
            cycle=cycle,
            step_norm=float(args.polish_step),
            tolerance=float(args.repair_tol),
            rows=rows,
        )
        if not accepted:
            break

    final = evaluate_report(base_params, controls)
    controls_path = run_dir / "final_controls.npz"
    history_path = run_dir / "history.csv"
    summary_path = run_dir / "summary.json"
    save_controls(controls_path, q_base, controls)
    write_csv(history_path, rows)

    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": str(Path(__file__).relative_to(ROOT)),
        "args": vars(args),
        "base_params": base_params,
        "reference_control": str(FOURTH_ORDER_BEST_CONTROLS),
        "reference_summary": reference_summary,
        "initial_summary": rows[0],
        "final_summary": final,
        "artifacts": {
            "run_dir": str(run_dir),
            "controls_npz": str(controls_path),
            "history_csv": str(history_path),
            "summary_json": str(summary_path),
        },
        "trace": trace.to_dict(),
    }
    if reference_summary is not None:
        summary["comparison_to_previous_best"] = {
            "energy_delta": float(final["energy"] - reference_summary["energy"]),
            "C_sym_norm2_ratio": float(
                final["C_sym_norm2"] / max(reference_summary["C_sym_norm2"], 1.0e-300)
            ),
            "F_norm2_ratio": float(final["F_norm2"] / max(reference_summary["F_norm2"], 1.0e-300)),
        }
    summary_path.write_text(json.dumps(summary, indent=2, default=json_default) + "\n", encoding="utf-8")

    print("final")
    print(
        f"  fidelity={final['fidelity']:.12f} infidelity={final['infidelity']:.3e} "
        f"F2={final['F_norm2']:.3e} C2={final['C_sym_norm2']:.3e} "
        f"energy={final['energy']:.6f}"
    )
    if reference_summary is not None:
        print("previous_best")
        print(
            f"  F2={reference_summary['F_norm2']:.3e} C2={reference_summary['C_sym_norm2']:.3e} "
            f"energy={reference_summary['energy']:.6f}"
        )
    print("saved")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
