"""Run a small, real closed-loop tracker demonstration for HO purification."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
CONTAINED_ROOT = ROOT.parents[1]
if str(CONTAINED_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTAINED_ROOT))

import optimizer as opt
from optimizer.tracker import JournalObserver, RunJournal, RunReceipt, SourceReference
from optimizer_log import inspect_writer, read_terminal, read_timeline

from system import HOPurificationParams, HOPurificationTrackerSystem


def main() -> Path:
    # Deliberately small: tracker plumbing, not the definitive physics grid.
    system = HOPurificationTrackerSystem(
        HOPurificationParams(tf=1.0, N=24, n_osc=4, g=0.2, r=0.6, fock_N=1)
    )
    initial = opt.guesses.random_fourier_guess(system, amplitude=0.8, modes=3, seed=17)
    gradient_check = opt.utils.verify_gradient(system, initial, directions=4, seed=18)
    if not gradient_check["passed"]:
        raise RuntimeError(f"HO gradient check failed: {gradient_check}")

    run_id = "ho-line-search-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "runs" / run_id
    receipt = RunReceipt.create(
        run_id=run_id,
        objective_label="purification objective",
        objective_metric="J",
        direction="minimize",
        references={
            "problem": SourceReference.create(
                "HO-purification-problem",
                path="/Users/billabobz/phd/HO_purification/Non-unitary_Control_Qubit_HO.tex",
            ),
            "system": SourceReference.create("contained-HO-tracker-adapter", path=str(ROOT / "system.py")),
        },
        metric_descriptors=system.tracking_spec()["metrics"],
        method_config={"method": "line_search", "variant": "backtracking", "maxiter": 4},
        provenance={"purpose": "small closed-loop tracker demonstration; not a science result"},
        seed=17,
    )
    journal = RunJournal.create(run_dir, receipt)
    result = opt.optimizers.line_search(
        system,
        initial,
        variant="backtracking",
        step_size=0.25,
        max_backtracks=8,
        maxiter=4,
        observers=[JournalObserver(journal)],
        stage="tracker-demo",
        chunk=1,
        checkpoint_accepted=True,
    )
    timeline = read_timeline(run_dir)
    summary = {
        "run_dir": str(run_dir),
        "gradient_check": gradient_check,
        "result": result.to_dict(),
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
