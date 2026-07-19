import tempfile
import unittest
from pathlib import Path

import numpy as np

import optimizer as opt
from optimizer.controls import ControlSpec, Controls


class QuadraticSystem:
    def __init__(self, *, target=0.0):
        self.spec = ControlSpec(("u",), 4)
        self.target = float(target)
        self.params = {"target": self.target}
        self.evaluate_calls = 0
        self.gradient_calls = 0

    def control_spec(self):
        return self.spec

    def evaluate(self, controls):
        self.evaluate_calls += 1
        diff = controls.as_matrix(copy=False) - self.target
        return {"J": float(np.sum(diff * diff)), "energy": float(np.sum(controls.as_matrix(copy=False) ** 2))}

    def gradient(self, controls):
        self.gradient_calls += 1
        diff = controls.as_matrix(copy=False) - self.target
        return Controls.from_matrix(self.spec, 2.0 * diff, name="grad")

    def with_secondary(self, **updates):
        target = updates.get("target", self.target)
        return QuadraticSystem(target=target)


class LinearResidualSystem(QuadraticSystem):
    def with_secondary(self, **updates):
        target = updates.get("target", self.target)
        return LinearResidualSystem(target=target)

    def residuals(self, controls, name="hard"):
        del name
        return np.asarray([float(np.sum(controls.as_matrix(copy=False)))])

    def jacobian(self, controls, name="hard"):
        del controls, name
        return np.ones((1, self.spec.size), dtype=float)


class BlackBoxTests(unittest.TestCase):
    def test_writer_records_numeric_ledger_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            system = QuadraticSystem()
            controls = Controls.constant(system.control_spec(), 1.0, name="initial")
            next_controls = Controls.constant(system.control_spec(), 0.5, name="next")
            gradient = system.gradient(controls)

            box = opt.blackbox.start(
                run_dir,
                system=system,
                controls=controls,
                metrics={"J": 4.0},
                optimizer="manual",
                objective={"metric": "J", "mode": "min"},
            )
            box.record_iteration(
                optimizer="manual",
                iteration=1,
                global_iteration=1,
                metrics={"J": 1.0},
                previous_metrics={"J": 4.0},
                trial_metrics={"J": 1.0},
                controls=next_controls,
                previous_controls=controls,
                proposal_controls=next_controls,
                gradient=gradient,
                technical={"acceptance": {"current_value": 4.0, "trial_value": 1.0, "tolerance": 0.0}},
                accepted=True,
                reason="accepted",
            )
            box.close(type("Result", (), {"controls": next_controls, "metrics": {"J": 1.0}, "stop_reason": "done", "iterations": 1, "optimizer": "manual"})())

            manifest = opt.blackbox.read_manifest(run_dir)
            records = opt.blackbox.read_records(run_dir)

            self.assertEqual(manifest["status"], "completed")
            self.assertGreaterEqual(manifest["counts"]["iterations"], 1)
            self.assertTrue((run_dir / manifest["artifacts"]["initial_controls"]["path"]).exists())
            iteration = [record for record in records if record["kind"] == "iteration"][0]
            self.assertEqual(iteration["metrics"]["dJ"], -3.0)
            self.assertEqual(iteration["decision"]["delta"], -3.0)
            self.assertIn("gradient", iteration)

    def test_engine_blackbox_uses_existing_evaluations_and_gradients(self):
        controls = Controls.constant(ControlSpec(("u",), 4), 1.0, name="initial")

        baseline_system = QuadraticSystem()
        baseline = opt.adam(baseline_system, controls, maxiter=3, step_size=0.1)

        with tempfile.TemporaryDirectory() as tmp:
            logged_system = QuadraticSystem()
            logged = opt.adam(logged_system, controls, maxiter=3, step_size=0.1, blackbox=Path(tmp) / "run")
            records = opt.blackbox.read_records(Path(tmp) / "run", kind="iteration")
            manifest = opt.blackbox.read_manifest(Path(tmp) / "run")

        self.assertAlmostEqual(logged.J, baseline.J)
        self.assertEqual(logged_system.evaluate_calls, baseline_system.evaluate_calls)
        self.assertEqual(logged_system.gradient_calls, baseline_system.gradient_calls)
        self.assertEqual(len(records), 3)
        self.assertEqual(manifest["status"], "completed")
        self.assertIsNotNone(logged.blackbox_path)

    def test_window_analysis_and_diagnostics_are_structured(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            system = QuadraticSystem()
            controls = Controls.constant(system.control_spec(), 1.0)
            result = opt.adam(
                system,
                controls,
                maxiter=4,
                step_size=0.1,
                blackbox=run_dir,
                blackbox_policy={"analysis_every": 2},
            )

            analysis = opt.blackbox.analyze(run_dir, window=4)
            diagnostics = opt.diagnostics(run_dir, details="gradient", window=4)

            self.assertEqual(result.stop_reason, "maxiter")
            self.assertEqual(analysis["kind"], "window_analysis")
            self.assertEqual(analysis["cost"]["extra_evaluations"], 0)
            self.assertIn("signals", analysis)
            self.assertIn("gradient", diagnostics)

    def test_repair_records_residual_before_after_without_metric_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            system = LinearResidualSystem()
            controls = Controls.constant(system.control_spec(), 1.0)

            result = opt.repair_newton(system, controls, maxiter=2, blackbox=run_dir)
            repairs = opt.blackbox.read_records(run_dir, kind="repair")

            self.assertTrue(result.converged)
            self.assertEqual(system.evaluate_calls, 0)
            self.assertEqual(len(repairs), 1)
            self.assertLess(repairs[0]["residual"]["norm_after"], repairs[0]["residual"]["norm_before"])

    def test_reset_and_prune_manage_run_folder_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            system = QuadraticSystem()
            controls = Controls.constant(system.control_spec(), 1.0)
            opt.adam(system, controls, maxiter=1, step_size=0.1, blackbox=run_dir)

            prune_result = opt.blackbox.prune(run_dir, keep_labels=("initial", "final"))
            self.assertEqual(prune_result["kind"], "prune")

            reset_result = opt.blackbox.reset(run_dir, section="arrays")
            self.assertEqual(reset_result["section"], "arrays")
            self.assertEqual(list((run_dir / "arrays").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
