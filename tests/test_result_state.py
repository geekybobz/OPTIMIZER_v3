import unittest

import numpy as np

from optimizer.controls import ControlSpec, Controls
from optimizer.result import Evaluation, OptimizerResult
from optimizer.state import RunState, WarmStartState


def sample_controls(name="controls"):
    spec = ControlSpec(keys=("uA", "uB"), control_dim=2, dt=0.1)
    return Controls.from_matrix(spec, [[1.0, 2.0], [3.0, 4.0]], name=name)


class EvaluationTests(unittest.TestCase):
    def test_evaluation_validates_and_exports_metrics(self):
        controls = sample_controls()
        evaluation = Evaluation.from_metrics(
            controls,
            {"J": np.float64(3.5), "energy": np.array([1.0, 2.0])},
        )

        self.assertEqual(evaluation.J, 3.5)
        payload = evaluation.to_dict()
        self.assertEqual(payload["metrics"]["J"], 3.5)
        self.assertEqual(payload["metrics"]["energy"], [1.0, 2.0])
        self.assertEqual(payload["controls"]["spec"]["keys"], ["uA", "uB"])

    def test_evaluation_rejects_missing_J(self):
        with self.assertRaises(KeyError):
            Evaluation.from_metrics(sample_controls(), {"energy": 1.0})


class RunStateTests(unittest.TestCase):
    def test_initial_state_tracks_current_and_best_when_metrics_exist(self):
        controls = sample_controls()
        state = RunState.initial(
            controls,
            metrics={"J": 10.0},
            optimizer_name="adam",
            step_size=0.1,
            system_params={"lambda2": 1.0},
            trace_id="run-1",
        )

        self.assertEqual(state.iteration, 0)
        self.assertEqual(state.metrics["J"], 10.0)
        self.assertIsNotNone(state.best_controls)
        self.assertEqual(state.best_metrics["J"], 10.0)
        self.assertEqual(state.system_params["lambda2"], 1.0)

    def test_update_current_and_best_by_metric(self):
        state = RunState.initial(sample_controls(), metrics={"J": 10.0})
        state.update_current(sample_controls("new"), {"J": 8.0}, step_size=0.05)

        self.assertEqual(state.iteration, 1)
        self.assertEqual(state.global_iteration, 1)
        self.assertEqual(state.step_size, 0.05)
        self.assertTrue(state.update_best_by_metric(metric="J", mode="min"))
        self.assertEqual(state.best_metrics["J"], 8.0)

    def test_update_best_by_metric_handles_no_improvement(self):
        state = RunState.initial(sample_controls(), metrics={"J": 10.0})
        state.update_current(sample_controls("new"), {"J": 12.0})

        self.assertFalse(state.update_best_by_metric(metric="J", mode="min"))
        self.assertEqual(state.best_metrics["J"], 10.0)


class WarmStartTests(unittest.TestCase):
    def test_warmstart_transfers_optimizer_state_only_when_compatible(self):
        state = RunState.initial(
            sample_controls(),
            metrics={"J": 2.0},
            optimizer_name="adam",
            step_size=0.03,
        )
        state.optimizer_state["m"] = np.array([1.0, 2.0])

        same = WarmStartState.from_run_state(state, target_optimizer="adam")
        different = WarmStartState.from_run_state(state, target_optimizer="line_search")

        self.assertIsNotNone(same.optimizer_state)
        np.testing.assert_allclose(same.optimizer_state["m"], [1.0, 2.0])
        self.assertIsNone(different.optimizer_state)
        self.assertEqual(different.metrics["J"], 2.0)
        self.assertEqual(different.step_size, 0.03)

    def test_warmstart_from_result_without_state_uses_public_fields(self):
        result = OptimizerResult(
            controls=sample_controls(),
            metrics={"J": 1.0},
            stop_reason="done",
            iterations=3,
            optimizer="line_search",
            system_params={"energy_weight": 0.1},
            trace_id="trace-1",
            checkpoint_ids={"latest": "ckpt"},
        )

        warm = result.warmstart(target_optimizer="adam")

        self.assertEqual(warm.source_optimizer, "line_search")
        self.assertEqual(warm.target_optimizer, "adam")
        self.assertIsNone(warm.optimizer_state)
        self.assertEqual(warm.system_params["energy_weight"], 0.1)
        self.assertEqual(warm.checkpoint_ids["latest"], "ckpt")


class OptimizerResultTests(unittest.TestCase):
    def test_result_from_state_and_to_dict(self):
        state = RunState.initial(
            sample_controls(),
            metrics={"J": 4.0, "fidelity": 0.9},
            optimizer_name="momentum",
            system_params={"lambda4": 10.0},
            trace_id="trace",
        )
        state.iteration = 7
        state.stop_reason = "maxiter"
        state.checkpoint_ids["latest"] = "ckpt-7"

        result = OptimizerResult.from_state(state, stop_reason=state.stop_reason)
        payload = result.to_dict(include_state=True)

        self.assertEqual(result.J, 4.0)
        self.assertEqual(result.iterations, 7)
        self.assertEqual(result.optimizer, "momentum")
        self.assertEqual(payload["controls"]["matrix"], [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(payload["state"]["iteration"], 7)
        self.assertEqual(payload["checkpoint_ids"]["latest"], "ckpt-7")


if __name__ == "__main__":
    unittest.main()

