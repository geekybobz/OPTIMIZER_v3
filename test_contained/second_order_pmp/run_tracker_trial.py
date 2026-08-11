"""Run a small, real closed-loop tracker demonstration for second-order PMP."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
CONTAINED_ROOT = ROOT.parents[1]
if str(CONTAINED_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTAINED_ROOT))

import optimizer as opt
from optimizer.tracker import JournalObserver, RunJournal, RunReceipt, SourceReference
from optimizer_log import inspect_writer, read_terminal, read_timeline

from system import SecondOrderPMPParams, SecondOrderPMPSystem


def main() -> Path:
    # Deliberately small: direct discrete-gradient and tracker verification.
    system = SecondOrderPMPSystem(SecondOrderPMPParams(tf=1.0, N=16, control_weight=0.0))
    initial = opt.guesses.random_fourier_guess(system, amplitude=1.2, modes=3, seed=31)
    gradient_check = opt.utils.verify_gradient(system, initial, directions=3, seed=32, rtol=2.0e-5)
    if not gradient_check["passed"]:
        raise RuntimeError(f"Second-order gradient check failed: {gradient_check}")

    run_id = "second-order-adam-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "runs" / run_id
    receipt = RunReceipt.create(
        run_id=run_id,
        objective_label="second-order universal-robustness penalty",
        objective_metric="J",
        direction="minimize",
        references={
            "problem": SourceReference.create(
                "second-order-PMP-problem",
                path="/Users/billabobz/phd/rob_unversal_article/second_order/latex/pmp_secondorder.tex",
            ),
            "system": SourceReference.create("contained-second-order-PMP-adapter", path=str(ROOT / "system.py")),
        },
        metric_descriptors=system.tracking_spec()["metrics"],
        method_config={"method": "adam", "variant": "adam", "maxiter": 4, "step_size": 0.08},
        provenance={"purpose": "small closed-loop tracker demonstration; not a robustness certificate"},
        seed=31,
    )
    journal = RunJournal.create(run_dir, receipt)
    result = opt.optimizers.adam(
        system,
        initial,
        step_size=0.08,
        maxiter=4,
        observers=[JournalObserver(journal)],
        stage="tracker-demo",
        chunk=1,
        checkpoint_accepted=True,
    )
    directions = np.random.default_rng(33).normal(size=(12, 3))
    robust_postcheck = system.robust_fidelity(result.controls, alpha=0.05, directions=directions)
    timeline = read_timeline(run_dir)
    summary = {
        "run_dir": str(run_dir),
        "gradient_check": gradient_check,
        "result": result.to_dict(),
        "post_run_robust_diagnostic": robust_postcheck,
        "terminal": read_terminal(run_dir),
        "writer": inspect_writer(run_dir),
        "event_kinds": [record["kind"] for record in timeline.records],
        "events": len(timeline.records),
        "claim_boundary": "Small contained closed-loop tracker demonstration only.",
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return run_dir


if __name__ == "__main__":
    main()
