"""Focused tests for the standalone durable tracker model and journal."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from optimizer.controls import ControlSpec, Controls
from optimizer.tracker import (
    AnalysisRecord,
    CheckpointReference,
    ChunkRecord,
    InterventionRecord,
    IterationRecord,
    MetricDescriptor,
    RunJournal,
    RunReceipt,
    ScalarSnapshot,
    SourceReference,
    TerminalRecord,
    info as tracker_info,
    follow_timeline,
    inspect_writer,
    load_checkpoint,
    metric_coverage,
    read_live,
    read_receipt,
    read_terminal,
    read_timeline,
    project_timeline,
    validate_event,
    validate_receipt,
)


def _receipt() -> RunReceipt:
    return RunReceipt.create(
        run_id="quadratic-demo",
        objective_label="J",
        direction="minimize",
        objective_metric="J",
        metric_descriptors={
            "J": MetricDescriptor.create(
                "J", label="quadratic cost", role="objective", direction="minimize"
            ),
            "gradient_norm": MetricDescriptor.create("gradient_norm", role="diagnostic"),
        },
        api_context={"request_id": "api-7", "bounds": {"lower": -1.0, "upper": 1.0}},
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


class TrackerTests(unittest.TestCase):
    def test_info_describes_the_passive_storage_contract_without_a_run(self) -> None:
        payload = tracker_info()

        self.assertEqual(payload["schema_version"], "optimizer.tracker-info/v1")
        self.assertEqual(payload["run_schema_version"], "optimizer.tracker/v3")
        self.assertTrue(payload["capabilities"]["passive_recording"])
        self.assertTrue(payload["capabilities"]["core_bridge"])
        self.assertTrue(payload["capabilities"]["standalone_reader"])
        self.assertTrue(payload["capabilities"]["writer_locking"])
        self.assertTrue(payload["capabilities"]["writer_ownership_record"])
        self.assertTrue(payload["capabilities"]["incremental_follow"])
        self.assertEqual(payload["storage"]["timeline"], "timeline.jsonl")
        self.assertIn("info", payload["public_api"])

    def test_causal_iteration_journal_refreshes_live_from_supplied_facts(self) -> None:
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
            self.assertEqual(receipt["objective"]["metric_id"], "J")
            self.assertEqual(receipt["selection"]["direction"], "minimize")
            self.assertEqual(receipt["api_context"]["request_id"], "api-7")
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
            journal.record_start(_snapshot(2.0, objective=1, gradient=0), method="adam")
            self.assertFalse((journal.run_dir / "checkpoints").exists())
            reference = journal.save_checkpoint(
                checkpoint_id="initial-0001",
                label="initial",
                controls=controls,
                optimizer_state={"step": 0},
                snapshot=_snapshot(2.0, objective=1, gradient=0),
                api_context={"call": "manual_checkpoint"},
                reason="user_requested",
                metadata={"reason": "user-requested"},
            )
            restored = load_checkpoint(journal.run_dir, reference)
            self.assertTrue((journal.run_dir / reference.relative_path).is_file())
            np.testing.assert_allclose(restored.controls.as_matrix(), controls.as_matrix())
            self.assertEqual(restored.snapshot["metrics"]["J"], 2.0)
            self.assertEqual(restored.api_context["receipt"]["request_id"], "api-7")
            self.assertEqual(restored.api_context["checkpoint"]["call"], "manual_checkpoint")
            self.assertEqual(restored.metadata["reason"], "user_requested")
            self.assertIsNotNone(reference.sha256)
            with (journal.run_dir / "timeline.jsonl").open("a", encoding="utf-8") as handle:
                handle.write('{"unfinished":')

            result = read_timeline(journal.run_dir)
            self.assertEqual([event["kind"] for event in result.records], ["run_start", "checkpoint"])
            self.assertEqual(result.recovered_tail["line"], 3)

    def test_lifecycle_compact_payload_and_writer_ownership_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt())
            initial = _snapshot(2.0, objective=1, gradient=0)
            journal.record_start(initial, method="adam")
            with self.assertRaisesRegex(RuntimeError, "lifecycle"):
                journal.record_start(initial, method="adam")
            with self.assertRaisesRegex(ValueError, "sequence is too large"):
                journal.record_analysis(
                    AnalysisRecord.create(
                        analysis_code="oversized",
                        start_iteration=0,
                        end_iteration=0,
                        inputs={"values": list(range(65))},
                        counters={"objective": 1},
                    )
                )
            journal.record_terminal(
                TerminalRecord.create(
                    status="completed",
                    reason="manual_stop",
                    final=initial,
                )
            )
            self.assertEqual(inspect_writer(journal.run_dir)["state"], "closed")
            self.assertFalse((journal.run_dir / ".writer.lock").exists())
            with self.assertRaisesRegex(RuntimeError, "ended"):
                journal.record_iteration(
                    IterationRecord.create(
                        iteration=1,
                        method="adam",
                        before=initial,
                        after=initial,
                        accepted=False,
                        reason="too_late",
                    )
                )

    def test_incremental_reader_and_projection_keep_causal_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt())
            initial = _snapshot(2.0, objective=1, gradient=0)
            journal.record_start(initial, method="adam", chunk=3, system_params={"alpha": 0.5})
            first = follow_timeline(journal.run_dir)
            self.assertEqual([record["kind"] for record in first.records], ["run_start"])
            journal.record_terminal(
                TerminalRecord.create(status="completed", reason="done", final=initial)
            )
            second = follow_timeline(journal.run_dir, offset=first.next_offset)
            projected = project_timeline(
                list(first.records + second.records),
                fields={"kind": "kind", "method": "payload.method"},
            )
            self.assertEqual([row["kind"] for row in projected], ["run_start", "run_end"])
            self.assertEqual(projected[0]["method"], "adam")

    def test_full_read_offset_cannot_skip_a_concurrent_append(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt())
            initial = _snapshot(2.0, objective=1, gradient=0)
            journal.record_start(initial, method="adam")
            timeline_path = journal.run_dir / "timeline.jsonl"
            original_read_bytes = Path.read_bytes
            appended = False

            def read_then_append(path: Path) -> bytes:
                nonlocal appended
                data = original_read_bytes(path)
                if path == timeline_path and not appended:
                    appended = True
                    journal.record_terminal(
                        TerminalRecord.create(status="completed", reason="done", final=initial)
                    )
                return data

            with mock.patch.object(Path, "read_bytes", read_then_append):
                first = read_timeline(journal.run_dir)
            self.assertEqual([record["kind"] for record in first.records], ["run_start"])
            second = follow_timeline(
                journal.run_dir,
                offset=first.next_offset,
                previous_sequence=first.last_sequence,
            )
            self.assertEqual([record["kind"] for record in second.records], ["run_end"])

    def test_incremental_reader_reports_permanent_corruption_without_looping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt())
            initial = _snapshot(2.0, objective=1, gradient=0)
            journal.record_start(initial, method="adam")
            journal.record_terminal(
                TerminalRecord.create(status="completed", reason="done", final=initial)
            )
            path = journal.run_dir / "timeline.jsonl"
            lines = path.read_bytes().splitlines(keepends=True)
            first_offset = len(lines[0])
            path.write_bytes(lines[0] + b'{"broken":}\n' + lines[1])

            result = follow_timeline(
                journal.run_dir,
                offset=first_offset,
                previous_sequence=1,
            )
            self.assertTrue(result.stalled)
            self.assertEqual(result.reason, "corrupt_record")
            self.assertEqual(result.corrupt_at_offset, first_offset)
            self.assertGreater(result.next_offset, first_offset)

    def test_schema_gate_unknown_events_and_validator_failures_are_explicit(self) -> None:
        receipt = _receipt().to_dict()
        with self.assertRaisesRegex(ValueError, "missing required"):
            validate_receipt({"schema_version": "optimizer.tracker/v3"})
        bad_direction = json.loads(json.dumps(receipt))
        bad_direction["objective"]["direction"] = "sideways"
        with self.assertRaisesRegex(ValueError, "direction"):
            validate_receipt(bad_direction)
        future = json.loads(json.dumps(receipt))
        future["schema_version"] = "optimizer.tracker/v99"
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            validate_receipt(future)

        event = {
            "schema_version": "optimizer.tracker/v3",
            "run_id": receipt["run_id"],
            "event_id": "future:1",
            "sequence": 1,
            "kind": "future_kind",
            "payload": {},
        }
        self.assertFalse(validate_event(event))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            validate_event({**event, "sequence": 3}, previous_sequence=1)
        with self.assertRaisesRegex(ValueError, "payload"):
            validate_event({**event, "payload": []})
        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt())
            journal.record_start(_snapshot(2.0, objective=1, gradient=0), method="adam")
            future_event = {
                **event,
                "run_id": receipt["run_id"],
                "event_id": f"{receipt['run_id']}:000002",
                "sequence": 2,
            }
            with (journal.run_dir / "timeline.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(future_event) + "\n")
            result = read_timeline(journal.run_dir)
            self.assertEqual([record["kind"] for record in result.records], ["run_start"])
            self.assertEqual(result.unsupported_events[0]["kind"], "future_kind")
            self.assertEqual(result.last_sequence, 2)

    def test_checkpoint_paths_integrity_payload_size_and_metric_coverage_are_guarded(self) -> None:
        spec = ControlSpec(keys=("u",), control_dim=2)
        controls = Controls.constant(spec, 0.5)
        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt())
            initial = _snapshot(2.0, objective=1, gradient=0)
            journal.record_start(initial, method="adam")
            reference = journal.save_checkpoint(
                checkpoint_id="safe",
                label="safe",
                controls=controls,
            )
            missing_digest = CheckpointReference(
                checkpoint_id=reference.checkpoint_id,
                label=reference.label,
                relative_path=reference.relative_path,
                shape=reference.shape,
                sha256=None,
            )
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_checkpoint(journal.run_dir, missing_digest)
            load_checkpoint(journal.run_dir, missing_digest, verify=False)
            traversal = {
                **reference.to_dict(),
                "relative_path": "../outside.npz",
            }
            with self.assertRaisesRegex(ValueError, "checkpoints directory"):
                load_checkpoint(journal.run_dir, traversal)
            target = journal.run_dir / reference.relative_path
            target.write_bytes(target.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                load_checkpoint(journal.run_dir, reference)

            with self.assertRaisesRegex(ValueError, "inline limit"):
                journal.record_analysis(
                    AnalysisRecord.create(
                        analysis_code="aggregate-too-large",
                        start_iteration=0,
                        end_iteration=0,
                        inputs={f"k{index}": [float(value) for value in range(64)] for index in range(80)},
                        counters={"objective": 1},
                    )
                )
            coverage = metric_coverage(
                read_receipt(journal.run_dir),
                read_timeline(journal.run_dir).records,
            )
            self.assertIn("J", coverage["observed"])
            self.assertIn("gradient_norm", coverage["observed"])

    def test_elapsed_time_starts_at_run_start_and_all_post_terminal_writes_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt())
            initial = _snapshot(2.0, objective=1, gradient=0)
            journal.record_start(initial, method="adam")
            start = read_timeline(journal.run_dir).records[0]
            self.assertLess(start["elapsed_s"], 0.1)
            self.assertGreaterEqual(start["payload"]["setup_s"], 0.0)
            journal.record_terminal(
                TerminalRecord.create(status="completed", reason="done", final=initial)
            )
            operations = (
                lambda: journal.record_analysis(
                    AnalysisRecord.create(
                        analysis_code="late",
                        start_iteration=0,
                        end_iteration=0,
                    )
                ),
                lambda: journal.record_chunk(
                    ChunkRecord.create(
                        method="adam",
                        chunk=0,
                        start_iteration=0,
                        end_iteration=0,
                        start_metrics={"J": 2.0},
                        end_metrics={"J": 2.0},
                    )
                ),
                lambda: journal.save_checkpoint(
                    checkpoint_id="late",
                    label="late",
                    controls=Controls.constant(ControlSpec(("u",), 1), 0.0),
                ),
            )
            for operation in operations:
                with self.assertRaisesRegex(RuntimeError, "ended"):
                    operation()


if __name__ == "__main__":
    unittest.main()
