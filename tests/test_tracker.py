"""Focused tests for the standalone durable tracker model and journal."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from optimizer.controls import ControlSpec, Controls
from optimizer.tracker import (
    AnalysisRecord,
    InterventionRecord,
    IterationRecord,
    RunJournal,
    RunReceipt,
    ScalarSnapshot,
    SourceReference,
    TerminalRecord,
    info as tracker_info,
    load_checkpoint,
    read_live,
    read_receipt,
    read_terminal,
    read_timeline,
)


def _receipt() -> RunReceipt:
    return RunReceipt.create(
        run_id="quadratic-demo",
        objective_label="J",
        direction="minimize",
        references={
            "problem": SourceReference.create("quadratic-problem", path="problem.py", version="v1"),
            "system": SourceReference.create("quadratic-system", path="system.py", content_hash="abc123"),
            "initial_state": SourceReference.create("random-seed-7"),
        },
        method_config={"name": "adam", "maxiter": 12},
        provenance={"optimizer": "contained"},
        seed=7,
    )


def _snapshot(j_value: float, *, objective: int, gradient: int) -> ScalarSnapshot:
    return ScalarSnapshot.create(
        metrics={"J": j_value, "gradient_norm": 0.4},
        metric_sources={"J": "cached-evaluation", "gradient_norm": "cached-gradient"},
        counters={"objective": objective, "gradient": gradient},
    )


class _CallSpy:
    """Represents a system that must never be called by passive tracking."""

    def __init__(self) -> None:
        self.objective_calls = 0
        self.gradient_calls = 0

    def evaluate(self) -> None:
        self.objective_calls += 1
        raise AssertionError("The passive tracker must not call evaluate().")

    def gradient(self) -> None:
        self.gradient_calls += 1
        raise AssertionError("The passive tracker must not call gradient().")


class TrackerTests(unittest.TestCase):
    def test_info_describes_the_passive_storage_contract_without_a_run(self) -> None:
        payload = tracker_info()

        self.assertEqual(payload["schema_version"], "optimizer.tracker-info/v1")
        self.assertEqual(payload["run_schema_version"], "optimizer.tracker/v2")
        self.assertTrue(payload["capabilities"]["passive_recording"])
        self.assertFalse(payload["capabilities"]["core_bridge"])
        self.assertEqual(payload["storage"]["timeline"], "timeline.jsonl")
        self.assertIn("info", payload["public_api"])

    def test_causal_iteration_journal_refreshes_live_without_system_calls(self) -> None:
        spy = _CallSpy()
        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt())
            initial = _snapshot(2.0, objective=1, gradient=1)
            journal.record_start(initial, method="adam")
            journal.record_iteration(
                IterationRecord.create(
                    iteration=1,
                    method="adam",
                    before=initial,
                    proposal={"learning_rate": 0.1, "update_norm": 0.2},
                    after=_snapshot(1.25, objective=2, gradient=2),
                    best_after=_snapshot(1.25, objective=2, gradient=2),
                    accepted=True,
                    reason="accepted_step",
                )
            )

            timeline = read_timeline(journal.run_dir)
            live = read_live(journal.run_dir)
            self.assertEqual(spy.objective_calls, 0)
            self.assertEqual(spy.gradient_calls, 0)
            self.assertIsNone(timeline.recovered_tail)
            self.assertEqual([event["kind"] for event in timeline.records], ["run_start", "iteration"])
            self.assertEqual(live["iteration"], 1)
            self.assertEqual(live["current"]["metrics"]["J"], 1.25)

    def test_receipt_references_authorities_instead_of_copying_the_system(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt())
            receipt = read_receipt(journal.run_dir)
            self.assertEqual(receipt["references"]["system"]["path"], "system.py")
            self.assertEqual(receipt["references"]["problem"]["identifier"], "quadratic-problem")
            self.assertNotIn("system", receipt)
            self.assertNotIn("problem", receipt)

    def test_analysis_intervention_and_terminal_preserve_causal_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt())
            initial = _snapshot(2.0, objective=1, gradient=1)
            current = _snapshot(1.25, objective=2, gradient=2)
            journal.record_start(initial, method="adam")
            journal.record_iteration(
                IterationRecord.create(
                    iteration=1,
                    method="adam",
                    before=initial,
                    after=current,
                    accepted=True,
                    reason="accepted_step",
                )
            )
            analysis = journal.record_analysis(
                AnalysisRecord.create(
                    analysis_code="progress_window",
                    start_iteration=0,
                    end_iteration=1,
                    inputs={"window": 2, "objective": "J"},
                    findings={"relative_improvement": 0.375, "classification": "continue"},
                    counters={"objective": 2, "gradient": 2},
                )
            )
            final = _snapshot(1.0, objective=3, gradient=3)
            journal.record_intervention(
                InterventionRecord.create(
                    action_code="reduce_step",
                    reason="progress_window_policy",
                    before=current,
                    after=final,
                    outcome_code="improved",
                    parameters={"from": 0.1, "to": 0.05},
                    analysis_event_id=analysis["event_id"],
                )
            )
            terminal = journal.record_terminal(
                TerminalRecord.create(
                    status="completed",
                    reason="iteration_budget",
                    final=final,
                    selected_checkpoint={"checkpoint_id": "best-0001"},
                    certification={"projected_kkt": {"status": "not_requested"}},
                )
            )

            records = read_timeline(journal.run_dir).records
            final_index = read_terminal(journal.run_dir)
            self.assertEqual([event["kind"] for event in records], ["run_start", "iteration", "analysis", "intervention", "run_end"])
            self.assertEqual(records[3]["payload"]["analysis_event_id"], analysis["event_id"])
            self.assertEqual(final_index["event_id"], terminal["event_id"])
            self.assertEqual(final_index["certification"]["projected_kkt"]["status"], "not_requested")
            self.assertEqual(read_live(journal.run_dir)["iteration"], 1)
            self.assertEqual(read_live(journal.run_dir)["method"], "adam")

    def test_counter_regression_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt())
            initial = _snapshot(2.0, objective=4, gradient=4)
            journal.record_start(initial, method="adam")
            with self.assertRaisesRegex(ValueError, "regressed"):
                journal.record_iteration(
                    IterationRecord.create(
                        iteration=1,
                        method="adam",
                        before=initial,
                        after=_snapshot(1.0, objective=3, gradient=3),
                        accepted=True,
                        reason="invalid_counter",
                    )
                )

    def test_checkpoint_is_explicit_and_reader_recovers_only_damaged_tail(self) -> None:
        spec = ControlSpec(keys=("u",), control_dim=3, dt=0.1)
        controls = Controls(spec, np.array([[0.2, 0.4, 0.6]]), name="initial")
        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt())
            self.assertFalse((journal.run_dir / "checkpoints").exists())
            reference = journal.save_checkpoint(
                checkpoint_id="initial-0001",
                label="initial",
                controls=controls,
                optimizer_state={"step": 0},
                metadata={"reason": "user-requested"},
            )
            restored = load_checkpoint(journal.run_dir, reference)
            self.assertTrue((journal.run_dir / reference.relative_path).is_file())
            np.testing.assert_allclose(restored.controls.as_matrix(), controls.as_matrix())
            with (journal.run_dir / "timeline.jsonl").open("a", encoding="utf-8") as handle:
                handle.write('{"unfinished":')

            result = read_timeline(journal.run_dir)
            self.assertEqual([event["kind"] for event in result.records], ["checkpoint"])
            self.assertEqual(result.recovered_tail["line"], 2)


if __name__ == "__main__":
    unittest.main()
