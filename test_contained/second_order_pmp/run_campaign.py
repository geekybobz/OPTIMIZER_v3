"""Run a source-aligned, multi-stage tracker campaign for Method 1.

Each optimizer stage owns one closed journal.  ``campaigns/latest.json`` only
links those immutable stage journals for the local read-only UI; it is not a
second source of numerical facts and never calls the system.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parent
CONTAINED_ROOT = ROOT.parents[1]
if str(CONTAINED_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTAINED_ROOT))

import optimizer as opt
from optimizer.controls import Controls
from optimizer.tracker import JournalObserver, RunJournal, RunReceipt, SourceReference
from optimizer_log import read_terminal

from system import SecondOrderPMPParams, SecondOrderPMPSystem


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _initial_controls(system: SecondOrderPMPSystem) -> Controls:
    """A declared pi-transfer baseline plus a fixed smooth transverse texture."""
    time = np.linspace(0.0, system.params.tf, system.params.N)
    matrix = np.zeros(system.control_spec().shape, dtype=float)
    matrix[0] = np.pi
    matrix[1] = 0.15 * np.sin(2.0 * np.pi * time)
    matrix[2] = 0.10 * np.cos(2.0 * np.pi * time)
    return Controls.from_matrix(
        system.control_spec(),
        matrix,
        name="declared_pi_transfer_baseline",
        meta={"construction": "ux=pi; uy=0.15 sin(2pi t); uz=0.10 cos(2pi t)"},
    )


def _receipt(
    *,
    run_id: str,
    system: SecondOrderPMPSystem,
    method: str,
    stage: str,
    maxiter: int,
    extra: dict[str, Any],
) -> RunReceipt:
    return RunReceipt.create(
        run_id=run_id,
        objective_label="Method 1 nominal infidelity plus full three-axis second-order channel penalty",
        objective_metric="J",
        direction="minimize",
        references={
            "problem": SourceReference.create(
                "second-order-PMP-Method-1-problem",
                path="/Users/billabobz/phd/rob_unversal_article/second_order/system/system_method1.py",
            ),
            "system": SourceReference.create("contained-source-aligned-Method-1-adapter", path=str(ROOT / "system.py")),
        },
        metric_descriptors=system.tracking_spec()["metrics"],
        method_config={"method": method, "stage": stage, "maxiter": int(maxiter), **extra},
        provenance={
            "claim_boundary": "Numerical optimizer/tracker exercise, not a robustness certificate or global-optimality claim.",
            "grid": "source-aligned tau=1, N=1001 stored nodes, N-1=1000 propagator intervals, dt=1e-3",
            "gradient": "exact discrete adjoint validated by directional finite differences before the campaign",
        },
        seed=2301,
    )


def _stage_record(name: str, method: str, status: str, *, run_dir: Path | None = None, result: Any | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": name, "method": method, "status": status}
    if run_dir is not None:
        payload["run_dir"] = str(run_dir.relative_to(ROOT))
    if result is not None:
        payload["iterations"] = int(result.iterations)
        payload["metrics"] = {key: float(value) for key, value in result.metrics.items()}
        payload["stop_reason"] = result.stop_reason
    return payload


def main() -> Path:
    system = SecondOrderPMPSystem(SecondOrderPMPParams())
    initial = _initial_controls(system)
    # The campaign checks directions only; it does not spend N finite differences.
    gradient_check = opt.utils.verify_gradient(system, initial, directions=3, seed=2302, rtol=2.0e-5)
    if not gradient_check["passed"]:
        raise RuntimeError(f"Source-aligned discrete-adjoint gradient check failed: {gradient_check}")

    campaign_id = "second-order-method1-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    campaign_dir = ROOT / "campaigns" / campaign_id
    campaign_dir.mkdir(parents=True, exist_ok=False)
    latest_path = ROOT / "campaigns" / "latest.json"
    stages: list[dict[str, Any]] = []

    def publish(status: str, active_stage: str | None = None, active_run_dir: Path | None = None) -> None:
        _write_json_atomically(
            latest_path,
            {
                "campaign_id": campaign_id,
                "problem": "source-aligned Method 1 second-order robust state transfer",
                "status": status,
                "active_stage": active_stage,
                "active_run_dir": None if active_run_dir is None else str(active_run_dir.relative_to(ROOT)),
                "grid": {"tf": system.params.tf, "N_nodes": system.params.N, "intervals": system.params.N - 1, "dt": system.dt},
                "gradient_check": gradient_check,
                "stages": stages,
            },
        )

    def run_stage(
        name: str,
        method: str,
        maxiter: int,
        call: Callable[[RunJournal], Any],
        receipt_extra: dict[str, Any],
    ) -> Any:
        run_id = f"{campaign_id}-{name}"
        run_dir = campaign_dir / name
        stages.append(_stage_record(name, method, "running", run_dir=run_dir))
        publish("running", name, run_dir)
        journal = RunJournal.create(run_dir, _receipt(
            run_id=run_id, system=system, method=method, stage=name, maxiter=maxiter, extra=receipt_extra,
        ))
        try:
            result = call(journal)
        except Exception:
            stages[-1] = _stage_record(name, method, "failed", run_dir=run_dir)
            publish("failed", name, run_dir)
            raise
        stages[-1] = _stage_record(name, method, "completed", run_dir=run_dir, result=result)
        publish("running", None, None)
        return result

    _write_json_atomically(
        campaign_dir / "campaign_manifest.json",
        {
            "campaign_id": campaign_id,
            "initial_controls": initial.meta,
            "system_parameters": {**system.static_parameters(), "dt": system.dt, "propagator_intervals": system.params.N - 1},
            "gradient_check": gradient_check,
            "claim_boundary": "Tracker exercise only; finite-alpha checks remain post-run diagnostics.",
        },
    )

    adam_result = run_stage(
        "adam_exploration", "adam", 120,
        lambda journal: opt.optimizers.adam(
            system, initial, step_size=0.05, maxiter=120,
            observers=[JournalObserver(journal)], stage="adam_exploration", chunk=1, checkpoint_accepted=True,
        ),
        {"step_size": 0.05, "initial_control": initial.name},
    )
    lbfgs_result = run_stage(
        "lbfgs_warmstart", "lbfgs", 80,
        lambda journal: opt.optimizers.lbfgs(
            system, warmstart=adam_result.warmstart(target_optimizer="lbfgs"), step_size=0.25, maxiter=80,
            history_size=12, observers=[JournalObserver(journal)], stage="lbfgs_warmstart", chunk=2, checkpoint_accepted=True,
        ),
        {"step_size": 0.25, "history_size": 12, "warmstart": "Adam controls/metrics/provenance transferred; Adam moments intentionally excluded"},
    )
    line_result = run_stage(
        "line_search_comparison", "line_search", 50,
        lambda journal: opt.optimizers.line_search(
            system, initial, variant="backtracking", step_size=0.30, min_step=1.0e-8, max_backtracks=12, maxiter=50,
            observers=[JournalObserver(journal)], stage="line_search_comparison", chunk=3, checkpoint_accepted=True,
        ),
        {"variant": "backtracking", "step_size": 0.30, "initial_control": initial.name, "comparison": "same declared baseline as Adam"},
    )
    directions = np.random.default_rng(2303).normal(size=(48, 3))
    diagnostics = {
        "adam_selected": system.robust_fidelity(adam_result.controls, alpha=0.05, directions=directions),
        "lbfgs_selected": system.robust_fidelity(lbfgs_result.controls, alpha=0.05, directions=directions),
        "line_search_selected": system.robust_fidelity(line_result.controls, alpha=0.05, directions=directions),
    }
    _write_json_atomically(campaign_dir / "post_run_diagnostics.json", diagnostics)
    _write_json_atomically(
        campaign_dir / "campaign_manifest.json",
        {
            "campaign_id": campaign_id,
            "initial_controls": initial.meta,
            "system_parameters": {**system.static_parameters(), "dt": system.dt, "propagator_intervals": system.params.N - 1},
            "gradient_check": gradient_check,
            "stages": stages,
            "terminal_indices": {stage["name"]: read_terminal(ROOT / stage["run_dir"]) for stage in stages},
            "post_run_finite_alpha_diagnostics": diagnostics,
            "claim_boundary": "Tracker exercise only; finite-alpha values are post-run diagnostics, not the optimized objective.",
        },
    )
    publish("completed", None, None)
    print(json.dumps({"campaign_dir": str(campaign_dir), "stages": stages, "diagnostics": diagnostics}, indent=2, sort_keys=True))
    return campaign_dir


if __name__ == "__main__":
    main()
