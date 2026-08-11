"""A deliberately long, live-observable source-grid tracker run.

This script is for inspecting the UI and journal while the optimizer is active.
It uses the original Method 1 numerical grid and records every attempted Adam
iteration.  It is not a convergence or robustness certificate.
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

from system import SecondOrderPMPParams, SecondOrderPMPSystem


STAGE_SPECS = (
    ("adam_exploration_1200", "adam", 1200),
    ("lbfgs_warmstart_1200", "lbfgs", 1200),
    ("line_search_same_start_1200", "line_search", 1200),
    ("adam_continuation_1200", "adam", 1200),
)


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _initial_controls(system: SecondOrderPMPSystem) -> Controls:
    time = np.linspace(0.0, system.params.tf, system.params.N)
    matrix = np.zeros(system.control_spec().shape, dtype=float)
    matrix[0] = np.pi
    matrix[1] = 0.15 * np.sin(2.0 * np.pi * time)
    matrix[2] = 0.10 * np.cos(2.0 * np.pi * time)
    return Controls.from_matrix(system.control_spec(), matrix, name="declared_pi_transfer_baseline")


def main() -> Path:
    system = SecondOrderPMPSystem(SecondOrderPMPParams())
    initial = _initial_controls(system)
    gradient_check = opt.utils.verify_gradient(system, initial, directions=3, seed=2302, rtol=2.0e-5)
    if not gradient_check["passed"]:
        raise RuntimeError(f"gradient check failed: {gradient_check}")

    campaign_id = "second-order-live-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    campaign_dir = ROOT / "campaigns" / campaign_id
    campaign_dir.mkdir(parents=True, exist_ok=False)
    latest = ROOT / "campaigns" / "latest.json"
    stages = [
        {"name": name, "method": method, "status": "pending", "run_dir": str((campaign_dir / name).relative_to(ROOT))}
        for name, method, _ in STAGE_SPECS
    ]

    def publish(status: str, *, active_index: int | None = None) -> None:
        active_stage = None if active_index is None else stages[active_index]
        _write_json_atomically(latest, {
            "campaign_id": campaign_id,
            "problem": "source-aligned Method 1 live tracker observation",
            "status": status,
            "active_stage": None if active_stage is None else active_stage["name"],
            "active_run_dir": None if active_stage is None else active_stage["run_dir"],
            "grid": {"tf": 1.0, "N_nodes": 1001, "intervals": 1000, "dt": 1.0e-3},
            "gradient_check": gradient_check,
            "stages": stages,
            "claim_boundary": "Long UI/tracker observation run only; not a robustness certificate or global-optimality claim.",
        })

    def run_stage(index: int, call: Callable[[RunJournal], Any], extra: dict[str, Any]) -> Any:
        name, method, maxiter = STAGE_SPECS[index]
        run_dir = campaign_dir / name
        stages[index]["status"] = "running"
        publish("running", active_index=index)
        receipt = RunReceipt.create(
            run_id=f"{campaign_id}-{name}",
            objective_label="Method 1 nominal infidelity plus full three-axis second-order channel penalty",
            objective_metric="J",
            direction="minimize",
            references={
                "problem": SourceReference.create("second-order-PMP-Method-1-problem", path="/Users/billabobz/phd/rob_unversal_article/second_order/system/system_method1.py"),
                "system": SourceReference.create("contained-source-aligned-Method-1-adapter", path=str(ROOT / "system.py")),
            },
            metric_descriptors=system.tracking_spec()["metrics"],
            method_config={"method": method, "stage": name, "maxiter": maxiter, **extra},
            provenance={"purpose": "live tracker inspection", "grid": "tau=1, N=1001, dt=1e-3", "gradient": "exact discrete adjoint"},
            seed=2301,
        )
        result = call(RunJournal.create(run_dir, receipt))
        stages[index].update({
            "status": "completed", "iterations": int(result.iterations),
            "metrics": {key: float(value) for key, value in result.metrics.items()},
            "stop_reason": result.stop_reason,
        })
        publish("running")
        return result

    adam_one = run_stage(0, lambda journal: opt.optimizers.adam(
        system, initial, step_size=0.05, maxiter=1200, observers=[JournalObserver(journal)],
        stage="adam_exploration_1200", chunk=10, checkpoint_accepted=True,
    ), {"step_size": 0.05, "initial_control": initial.name})
    lbfgs = run_stage(1, lambda journal: opt.optimizers.lbfgs(
        system, warmstart=adam_one.warmstart(target_optimizer="lbfgs"), step_size=0.25, history_size=12,
        maxiter=1200, observers=[JournalObserver(journal)], stage="lbfgs_warmstart_1200", chunk=20, checkpoint_accepted=True,
    ), {"step_size": 0.25, "history_size": 12, "warmstart": "Adam controls transferred; incompatible Adam moments excluded"})
    line = run_stage(2, lambda journal: opt.optimizers.line_search(
        system, initial, variant="backtracking", step_size=0.30, min_step=1.0e-8, max_backtracks=12,
        maxiter=1200, observers=[JournalObserver(journal)], stage="line_search_same_start_1200", chunk=30, checkpoint_accepted=True,
    ), {"variant": "backtracking", "step_size": 0.30, "initial_control": initial.name})
    adam_two = run_stage(3, lambda journal: opt.optimizers.adam(
        system, warmstart=lbfgs.warmstart(target_optimizer="adam"), step_size=0.02, maxiter=1200,
        observers=[JournalObserver(journal)], stage="adam_continuation_1200", chunk=40, checkpoint_accepted=True,
    ), {"step_size": 0.02, "warmstart": "L-BFGS controls transferred; L-BFGS history excluded"})
    diagnostics = {
        "adam_exploration": system.robust_fidelity(adam_one.controls, alpha=0.05, directions=np.random.default_rng(2303).normal(size=(48, 3))),
        "lbfgs_warmstart": system.robust_fidelity(lbfgs.controls, alpha=0.05, directions=np.random.default_rng(2303).normal(size=(48, 3))),
        "line_search_same_start": system.robust_fidelity(line.controls, alpha=0.05, directions=np.random.default_rng(2303).normal(size=(48, 3))),
        "adam_continuation": system.robust_fidelity(adam_two.controls, alpha=0.05, directions=np.random.default_rng(2303).normal(size=(48, 3))),
    }
    _write_json_atomically(campaign_dir / "post_run_diagnostics.json", diagnostics)
    publish("completed")
    print(json.dumps({"campaign_dir": str(campaign_dir), "stages": stages, "diagnostics": diagnostics}, indent=2, sort_keys=True))
    return campaign_dir


if __name__ == "__main__":
    main()
