"""Behavioral tests for the live core-to-journal tracker bridge."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from optimizer.controls import ControlSpec, Controls
from optimizer.core.engine import StepProposal, run_chunk
from optimizer.core.observer import ObserverAdapter
from optimizer.optimizers.cma_es import cma_es
from optimizer.optimizers.line_search import line_search
from optimizer.system_olgs import basic_system_spec
from optimizer.tracker import (
    InterventionRecord,
    JournalObserver,
    RunJournal,
    RunReceipt,
    ScalarSnapshot,
    SourceReference,
    TerminalRecord,
    load_checkpoint,
    read_live,
    read_terminal,
    read_timeline,
    inspect_writer,
)


class CountingQuadraticSystem:
    """Small deterministic system that exposes whether tracking added calls."""

    def __init__(self, *, fail_above: float | None = None) -> None:
        self.spec = ControlSpec(("u",), 4)
        self.fail_above = fail_above
        self.evaluate_calls = 0
        self.gradient_calls = 0

    def control_spec(self) -> ControlSpec:
        return self.spec

    def system_spec(self) -> dict[str, object]:
        return basic_system_spec(
            "counting-quadratic",
            control_spec=self.spec,
            metrics={"energy": {"label": "Energy", "role": "resource"}},
        )

    def evaluate(self, controls: Controls) -> dict[str, float]:
        self.evaluate_calls += 1
        values = controls.as_matrix(copy=False)
        if self.fail_above is not None and np.max(np.abs(values)) > self.fail_above:
            return {"J": float("inf"), "energy": float("inf")}
        return {
            "J": float(np.sum(values * values)),
            "energy": float(np.sum(values * values)),
        }

    def gradient(self, controls: Controls) -> Controls:
        self.gradient_calls += 1
        return Controls.from_matrix(self.spec, 2.0 * controls.as_matrix(copy=False), name="gradient")

    def with_secondary(self, **updates: object) -> "CountingQuadraticSystem":
        del updates
        return self


class VectorMetricSystem(CountingQuadraticSystem):
    def system_spec(self) -> dict[str, object]:
        return basic_system_spec(
            "vector-metric-quadratic",
            control_spec=self.spec,
            metrics={
                "energy": {"label": "Energy", "role": "resource"},
                "profile": {"label": "Profile", "shape": [3]},
                "secondary_nan": {"label": "Optional nonfinite diagnostic", "allow_nonfinite": True},
            },
        )

    def evaluate(self, controls: Controls) -> dict[str, object]:
        payload = super().evaluate(controls)
        payload["profile"] = np.asarray([1.0, 2.0, 3.0])
        payload["secondary_nan"] = float("nan")
        return payload


class FirstEvaluationOnlySystem(CountingQuadraticSystem):
    def evaluate(self, controls: Controls) -> dict[str, float]:
        if self.evaluate_calls:
            self.evaluate_calls += 1
            return {"J": float("inf")}
        return super().evaluate(controls)


class InterruptingPopulationSystem(CountingQuadraticSystem):
    """Raises after CMA-ES has emitted its run-start event."""

    def evaluate(self, controls: Controls) -> dict[str, float]:
        if self.evaluate_calls:
            raise KeyboardInterrupt()
        return super().evaluate(controls)


class RaisingObserver(ObserverAdapter):
    def on_iteration(self, event) -> None:
        del event
        raise RuntimeError("observer write failed")


class FailingJournalObserver(JournalObserver):
    def on_iteration(self, event) -> None:
        del event
        raise OSError("simulated journal write failure")


def descent_step(context):
    return StepProposal(
        controls=context.state.controls - 0.25 * context.gradient,
        step_size=0.25,
        technical={"step_kind": "deterministic_descent"},
    )


def exploding_step(context):
    return StepProposal(
        controls=Controls.constant(context.state.controls.spec, 2.0, name="explode"),
        technical={"step_kind": "exploding_trial"},
    )


def compact_step(context):
    return StepProposal(
        controls=context.state.controls - 0.25 * context.gradient,
        technical={
            "candidate_vector": [0.1, 0.2, 0.3, 0.4],
            "attempts": [{"attempt": 0, "step_size": 0.25}],
        },
    )


def loud_failure_step(context):
    del context
    raise ValueError("x" * 9000)


def interrupted_step(context):
    del context
    raise KeyboardInterrupt()


def _receipt(run_id: str = "bridge-run") -> RunReceipt:
    return RunReceipt.create(
        run_id=run_id,
        objective_label="J",
        direction="minimize",
        references={
            "problem": SourceReference.create("bridge-test-problem"),
            "system": SourceReference.create("bridge-test-system"),
        },
        method_config={"name": "descent"},
    )


def _snapshot(value: float) -> ScalarSnapshot:
    return ScalarSnapshot.create(metrics={"J": value}, counters={"evaluation": 1})


class TrackerBridgeTests(unittest.TestCase):
    def test_tracker_on_off_have_identical_numerics_and_no_extra_system_calls(self) -> None:
        baseline_system = CountingQuadraticSystem()
        tracked_system = CountingQuadraticSystem()
        baseline_controls = Controls.constant(baseline_system.control_spec(), 1.0, name="initial")
        tracked_controls = Controls.constant(tracked_system.control_spec(), 1.0, name="initial")

        baseline = run_chunk(
            baseline_system,
            baseline_controls,
            step=descent_step,
            optimizer_name="descent",
            maxiter=3,
            create_trace=False,
        )

        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt())
            tracked = run_chunk(
                tracked_system,
                tracked_controls,
                step=descent_step,
                optimizer_name="descent",
                maxiter=3,
                create_trace=False,
                observers=[JournalObserver(journal)],
                stage="phase-a",
                chunk=2,
                system_params={"alpha": 0.5},
                api_context={"request_id": "run-chunk-api"},
            )

            np.testing.assert_allclose(tracked.controls.as_matrix(), baseline.controls.as_matrix())
            self.assertEqual(tracked.metrics, baseline.metrics)
            self.assertEqual(tracked.stop_reason, baseline.stop_reason)
            self.assertEqual(tracked_system.evaluate_calls, baseline_system.evaluate_calls)
            self.assertEqual(tracked_system.gradient_calls, baseline_system.gradient_calls)

            records = read_timeline(journal.run_dir).records
            iterations = [event for event in records if event["kind"] == "iteration"]
            self.assertEqual([event["kind"] for event in records][0], "run_start")
            self.assertEqual([event["kind"] for event in records][-1], "run_end")
            self.assertIn("chunk", [event["kind"] for event in records])
            self.assertEqual(len(iterations), 3)
            self.assertTrue(all(event["payload"]["trial"] is not None for event in iterations))
            self.assertTrue(all(event["payload"]["chunk"] == 2 for event in iterations))
            self.assertEqual(iterations[0]["payload"]["system_params"]["alpha"], 0.5)
            chunk_event = next(event for event in records if event["kind"] == "chunk")
            self.assertEqual(chunk_event["payload"]["phase"], "phase-a")
            self.assertEqual(records[-1]["payload"]["chunk"], 2)
            live = read_live(journal.run_dir)
            terminal = read_terminal(journal.run_dir)
            self.assertEqual(live["last_event_id"], records[-1]["event_id"])
            self.assertEqual(live["best"]["metrics"]["J"], tracked.metrics["J"])
            self.assertEqual(terminal["status"], "completed")
            self.assertEqual(terminal["selected_checkpoint"]["label"], "best_J")
            self.assertIsNone(terminal["selection_fallback"])
            checkpoint_events = [event for event in records if event["kind"] == "checkpoint"]
            self.assertEqual([event["payload"]["checkpoint"]["label"] for event in checkpoint_events], ["chunk_start", "best_J", "chunk_end"])

    def test_keyboard_interrupt_closes_journal_without_an_extra_evaluation(self) -> None:
        system = CountingQuadraticSystem()
        controls = Controls.constant(system.control_spec(), 1.0)
        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt("interrupted-run"))
            with self.assertRaises(KeyboardInterrupt):
                run_chunk(
                    system,
                    controls,
                    step=interrupted_step,
                    optimizer_name="interruptible",
                    maxiter=5,
                    create_trace=False,
                    observers=[JournalObserver(journal)],
                )
            terminal = read_terminal(journal.run_dir)
            records = read_timeline(journal.run_dir).records
            self.assertEqual(terminal["status"], "interrupted")
            self.assertEqual(terminal["reason"], "interrupted")
            self.assertEqual(records[-1]["kind"], "run_end")
            self.assertEqual(inspect_writer(journal.run_dir)["state"], "closed")
            self.assertFalse((journal.run_dir / ".writer.lock").exists())
            self.assertEqual(system.evaluate_calls, 1)

    def test_cma_es_keyboard_interrupt_closes_journal(self) -> None:
        system = InterruptingPopulationSystem()
        controls = Controls.constant(system.control_spec(), 1.0)
        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt("cma-interrupted"))
            with self.assertRaises(KeyboardInterrupt):
                cma_es(
                    system,
                    controls,
                    maxiter=2,
                    population_size=3,
                    create_trace=False,
                    observers=[JournalObserver(journal)],
                )
            self.assertEqual(read_terminal(journal.run_dir)["status"], "interrupted")
            self.assertEqual(inspect_writer(journal.run_dir)["state"], "closed")
            self.assertFalse((journal.run_dir / ".writer.lock").exists())

    def test_vector_and_nonfinite_secondary_metrics_are_omitted_without_changing_numerics(self) -> None:
        baseline_system = VectorMetricSystem()
        tracked_system = VectorMetricSystem()
        baseline_controls = Controls.constant(baseline_system.control_spec(), 1.0)
        tracked_controls = Controls.constant(tracked_system.control_spec(), 1.0)
        baseline = run_chunk(
            baseline_system,
            baseline_controls,
            step=descent_step,
            optimizer_name="descent",
            maxiter=1,
            create_trace=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt("vector-metrics"))
            tracked = run_chunk(
                tracked_system,
                tracked_controls,
                step=descent_step,
                optimizer_name="descent",
                maxiter=1,
                create_trace=False,
                observers=[JournalObserver(journal)],
            )
            np.testing.assert_allclose(tracked.controls.as_matrix(), baseline.controls.as_matrix())
            self.assertEqual(tracked.stop_reason, baseline.stop_reason)
            records = read_timeline(journal.run_dir).records
            start = records[0]["payload"]["initial"]
            self.assertEqual(start["omitted_metrics"], ["profile", "secondary_nan"])
            chunk = next(record["payload"] for record in records if record["kind"] == "chunk")
            self.assertEqual(
                chunk["omitted_metrics"]["start"],
                ["profile", "secondary_nan"],
            )
            self.assertEqual(read_terminal(journal.run_dir)["status"], "failed")

    def test_passive_observer_failure_is_quarantined_and_reported(self) -> None:
        system = CountingQuadraticSystem()
        controls = Controls.constant(system.control_spec(), 1.0)
        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt("observer-failure"))
            result = run_chunk(
                system,
                controls,
                step=descent_step,
                optimizer_name="descent",
                maxiter=2,
                create_trace=False,
                observers=[RaisingObserver(), JournalObserver(journal)],
            )
            self.assertEqual(result.iterations, 2)
            self.assertFalse(result.to_dict()["tracking_complete"])
            self.assertEqual(result.tracking_failures[0]["callback"], "on_iteration")
            kinds = [record["kind"] for record in read_timeline(journal.run_dir).records]
            self.assertIn("observer_failure", kinds)
            self.assertEqual(kinds[-1], "run_end")

    def test_journal_failure_closes_incomplete_evidence_without_aborting_numerics(self) -> None:
        baseline_system = CountingQuadraticSystem()
        tracked_system = CountingQuadraticSystem()
        baseline_controls = Controls.constant(baseline_system.control_spec(), 1.0)
        tracked_controls = Controls.constant(tracked_system.control_spec(), 1.0)
        baseline = run_chunk(
            baseline_system,
            baseline_controls,
            step=descent_step,
            optimizer_name="descent",
            maxiter=2,
            create_trace=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt("journal-failure"))
            tracked = run_chunk(
                tracked_system,
                tracked_controls,
                step=descent_step,
                optimizer_name="descent",
                maxiter=2,
                create_trace=False,
                observers=[FailingJournalObserver(journal)],
            )
            np.testing.assert_allclose(tracked.controls.as_matrix(), baseline.controls.as_matrix())
            self.assertEqual(tracked.metrics, baseline.metrics)
            self.assertEqual(tracked_system.evaluate_calls, baseline_system.evaluate_calls)
            self.assertEqual(tracked.tracking_failures[0]["callback"], "on_iteration")
            kinds = [record["kind"] for record in read_timeline(journal.run_dir).records]
            self.assertEqual(kinds[-1], "tracking_failure")
            self.assertEqual(read_live(journal.run_dir)["status"], "tracking_failed")

    def test_long_errors_and_non_json_api_context_do_not_abort_tracking(self) -> None:
        system = CountingQuadraticSystem()
        controls = Controls.constant(system.control_spec(), 1.0)
        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt("payload-safety"))
            result = run_chunk(
                system,
                controls,
                step=loud_failure_step,
                optimizer_name="loud",
                maxiter=1,
                create_trace=False,
                observers=[JournalObserver(journal)],
                api_context={"owner": object(), "request_id": "safe"},
                random_state=lambda: {"generator": object()},
            )
            self.assertEqual(result.stop_reason, "proposal_failed")
            self.assertFalse(result.tracking_failures)
            records = read_timeline(journal.run_dir).records
            start_context = records[0]["payload"]["api_context"]["caller"]
            self.assertEqual(start_context["request_id"], "safe")
            self.assertIn("owner", start_context["_tracker_omitted_fields"])
            iteration = next(record["payload"] for record in records if record["kind"] == "iteration")
            self.assertLessEqual(len(iteration["technical"]["error"]), 8192)
            failure_checkpoint = next(
                record["payload"]["checkpoint"]
                for record in records
                if record["kind"] == "checkpoint"
                and record["payload"]["checkpoint"]["label"] == "failure_proposal"
            )
            restored = load_checkpoint(journal.run_dir, failure_checkpoint)
            self.assertEqual(restored.metadata["random_state_source"], "callable")
            self.assertIn("generator", restored.metadata["random_state"]["_tracker_omitted_fields"])

    def test_nonfinite_trial_is_a_failed_outcome_not_a_nonfinite_log_metric(self) -> None:
        system = CountingQuadraticSystem(fail_above=1.0)
        controls = Controls.constant(system.control_spec(), 0.0, name="initial")

        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt("nonfinite-run"))
            result = run_chunk(
                system,
                controls,
                step=exploding_step,
                optimizer_name="explode",
                maxiter=3,
                create_trace=False,
                observers=[JournalObserver(journal)],
            )

            records = read_timeline(journal.run_dir).records
            iteration = next(event["payload"] for event in records if event["kind"] == "iteration")
            self.assertEqual(result.stop_reason, "nonfinite_trial")
            self.assertEqual(iteration["reason"], "nonfinite_trial")
            self.assertIsNone(iteration["trial"])
            self.assertEqual(iteration["after"]["metrics"]["J"], 0.0)
            self.assertEqual(iteration["after"]["counters"]["evaluation"], 1)
            self.assertEqual(iteration["after"]["counters"]["evaluation_attempt"], 2)
            self.assertIn("error", iteration["technical"])
            self.assertEqual(read_terminal(journal.run_dir)["status"], "failed")

    def test_line_search_attempt_scalars_are_kept_without_trial_controls(self) -> None:
        system = CountingQuadraticSystem()
        controls = Controls.constant(system.control_spec(), 1.0, name="initial")

        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt("line-search-run"))
            line_search(
                system,
                controls,
                variant="backtracking",
                maxiter=1,
                step_size=1.0,
                create_trace=False,
                observers=[JournalObserver(journal)],
            )

            iteration = next(
                event["payload"]
                for event in read_timeline(journal.run_dir).records
                if event["kind"] == "iteration"
            )
            attempts = iteration["technical"]["proposal"]["attempts"]
            self.assertGreaterEqual(len(attempts), 1)
            self.assertIn("trial_value", attempts[0])
            self.assertNotIn("controls", attempts[0])
            start = next(
                event["payload"]
                for event in read_timeline(journal.run_dir).records
                if event["kind"] == "run_start"
            )
            self.assertEqual(
                start["api_context"]["caller"]["method_config"]["variant"],
                "backtracking",
            )

    def test_control_sized_technical_sequences_are_omitted(self) -> None:
        system = CountingQuadraticSystem()
        controls = Controls.constant(system.control_spec(), 1.0, name="initial")

        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt("compact-run"))
            run_chunk(
                system,
                controls,
                step=compact_step,
                optimizer_name="compact",
                maxiter=1,
                create_trace=False,
                observers=[JournalObserver(journal)],
            )

            iteration = next(
                event
                for event in read_timeline(journal.run_dir).records
                if event["kind"] == "iteration"
            )
            proposal = iteration["payload"]["technical"]["proposal"]
            self.assertNotIn("candidate_vector", proposal)
            self.assertIn("candidate_vector", proposal["_tracker_omitted_fields"])
            self.assertEqual(proposal["attempts"], [{"attempt": 0, "step_size": 0.25}])

    def test_journal_refuses_direct_reopening_and_preserves_known_best(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt("manual-run"))
            initial = _snapshot(2.0)
            best = _snapshot(1.0)
            worse = _snapshot(1.5)
            journal.record_start(initial, method="manual")
            journal.record_intervention(
                InterventionRecord.create(
                    action_code="manual_change",
                    reason="test",
                    before=initial,
                    after=worse,
                    outcome_code="worse",
                )
            )
            journal.record_terminal(
                TerminalRecord.create(
                    status="completed",
                    reason="manual_stop",
                    final=worse,
                    best=best,
                )
            )

            self.assertEqual(read_live(journal.run_dir)["best"]["metrics"]["J"], 1.0)
            self.assertEqual(read_terminal(journal.run_dir)["best"]["metrics"]["J"], 1.0)
            with self.assertRaisesRegex(RuntimeError, "RunJournal.create"):
                RunJournal(journal.run_dir, _receipt("manual-run"))

    def test_initial_evaluation_failure_is_closed_in_the_journal(self) -> None:
        system = CountingQuadraticSystem(fail_above=0.5)
        controls = Controls.constant(system.control_spec(), 1.0, name="invalid-initial")
        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt("initial-failure"))
            with self.assertRaisesRegex(ValueError, "initial controls failed"):
                run_chunk(
                    system,
                    controls,
                    step=descent_step,
                    optimizer_name="descent",
                    maxiter=1,
                    create_trace=False,
                    observers=[JournalObserver(journal)],
                )
            records = read_timeline(journal.run_dir).records
            self.assertEqual([event["kind"] for event in records], ["run_start", "run_end"])
            self.assertEqual(records[-1]["payload"]["reason"], "initial_evaluation_failed")
            self.assertEqual(read_terminal(journal.run_dir)["status"], "failed")

    def test_cma_es_uses_common_observer_events_and_reproducible_checkpoints(self) -> None:
        system = CountingQuadraticSystem()
        controls = Controls.constant(system.control_spec(), 0.5, name="initial")
        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt("cma-run"))
            result = cma_es(
                system,
                controls,
                maxiter=2,
                population_size=4,
                seed=11,
                create_trace=False,
                observers=[JournalObserver(journal)],
                api_context={"request_id": "cma-api"},
            )
            records = read_timeline(journal.run_dir).records
            kinds = [event["kind"] for event in records]
            self.assertEqual(kinds[0], "run_start")
            self.assertEqual(kinds[-1], "run_end")
            self.assertEqual(kinds.count("iteration"), 2)
            self.assertEqual(kinds.count("chunk"), 1)
            checkpoint_events = [event for event in records if event["kind"] == "checkpoint"]
            self.assertGreaterEqual(len(checkpoint_events), 2)
            final_reference = checkpoint_events[-1]["payload"]["checkpoint"]
            restored = load_checkpoint(journal.run_dir, final_reference)
            np.testing.assert_allclose(
                restored.controls.as_matrix(), result.controls.as_matrix()
            )
            self.assertEqual(restored.api_context["checkpoint"]["caller"]["request_id"], "cma-api")
            self.assertIn("random_state", restored.metadata)
            self.assertIsInstance(restored.optimizer_state["mean"], np.ndarray)
            self.assertEqual(restored.optimizer_state["mean"].dtype, np.dtype(float))

    def test_cma_tracker_equivalence_and_all_failed_populations_are_degenerate(self) -> None:
        baseline_system = CountingQuadraticSystem()
        tracked_system = CountingQuadraticSystem()
        baseline_controls = Controls.constant(baseline_system.control_spec(), 0.5)
        tracked_controls = Controls.constant(tracked_system.control_spec(), 0.5)
        baseline = cma_es(
            baseline_system,
            baseline_controls,
            maxiter=2,
            population_size=4,
            seed=17,
            create_trace=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt("cma-equivalence"))
            tracked = cma_es(
                tracked_system,
                tracked_controls,
                maxiter=2,
                population_size=4,
                seed=17,
                create_trace=False,
                observers=[JournalObserver(journal)],
            )
            np.testing.assert_allclose(tracked.controls.as_matrix(), baseline.controls.as_matrix())
            self.assertEqual(tracked.metrics, baseline.metrics)
            self.assertEqual(tracked_system.evaluate_calls, baseline_system.evaluate_calls)

        failing_system = FirstEvaluationOnlySystem()
        failing_controls = Controls.constant(failing_system.control_spec(), 0.5)
        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt("cma-degenerate"))
            result = cma_es(
                failing_system,
                failing_controls,
                maxiter=2,
                population_size=4,
                seed=3,
                create_trace=False,
                use_cache=False,
                observers=[JournalObserver(journal)],
            )
            np.testing.assert_allclose(result.controls.as_matrix(), failing_controls.as_matrix())
            terminal = read_terminal(journal.run_dir)
            self.assertEqual(result.stop_reason, "no_usable_candidates")
            self.assertEqual(terminal["status"], "degenerate")
            iterations = [
                record["payload"]
                for record in read_timeline(journal.run_dir).records
                if record["kind"] == "iteration"
            ]
            self.assertTrue(all(item["technical"]["elite_count"] == 0 for item in iterations))

    def test_selection_rule_has_explicit_fallback_when_checkpoints_are_disabled(self) -> None:
        system = CountingQuadraticSystem()
        controls = Controls.constant(system.control_spec(), 1.0)
        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal.create(Path(directory) / "run", _receipt("no-checkpoints"))
            run_chunk(
                system,
                controls,
                step=descent_step,
                optimizer_name="descent",
                maxiter=1,
                create_trace=False,
                observers=[JournalObserver(journal)],
                checkpoint_start=False,
                checkpoint_latest=False,
                checkpoint_accepted=False,
                checkpoint_best=False,
                checkpoint_final=False,
            )
            terminal = read_terminal(journal.run_dir)
            self.assertIsNone(terminal["selected_checkpoint"])
            self.assertEqual(terminal["selection_fallback"], "no_durable_checkpoint_available")

    def test_standalone_reader_does_not_import_the_optimizer_or_numpy(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import sys; import optimizer_log.reader; "
                    "print('optimizer' in sys.modules, 'numpy' in sys.modules)"
                ),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        self.assertEqual(completed.stdout.strip(), "False False")


if __name__ == "__main__":
    unittest.main()
