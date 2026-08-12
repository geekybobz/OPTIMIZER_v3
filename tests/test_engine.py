import unittest

import numpy as np

from optimizer.controls import ControlSpec, Controls
from optimizer.core.engine import StepProposal, run_chunk
from optimizer.core.evaluate import SystemEvaluator
from optimizer.core.stopping import StoppingConfig
from optimizer.utils.parallel import ParallelConfig, parallel_map
from optimizer.logs.trace import Trace
from optimizer.system_olgs import basic_system_spec


class QuadraticSystem:
    """Small analytical-gradient system used to exercise the shared engine."""

    def __init__(self, *, target=0.0, weight=1.0, fail_above=None):
        self.spec = ControlSpec(("u",), 4)
        self.target = float(target)
        self.weight = float(weight)
        self.fail_above = fail_above
        self.params = {"target": self.target, "weight": self.weight}
        self.evaluate_calls = 0
        self.gradient_calls = 0

    def control_spec(self):
        return self.spec

    def system_spec(self):
        return basic_system_spec(
            "engine-quadratic",
            control_spec=self.spec,
            metrics={"energy": {"label": "Energy", "role": "resource"}},
        )

    def evaluate(self, controls):
        self.evaluate_calls += 1
        matrix = controls.as_matrix(copy=False)
        if self.fail_above is not None and np.max(np.abs(matrix)) > self.fail_above:
            return {"J": np.inf, "energy": np.inf}
        diff = matrix - self.target
        return {
            "J": self.weight * float(np.sum(diff * diff)),
            "energy": float(np.sum(matrix * matrix)),
        }

    def gradient(self, controls):
        self.gradient_calls += 1
        diff = controls.as_matrix(copy=False) - self.target
        return Controls.from_matrix(self.spec, 2.0 * self.weight * diff, name="grad")

    def with_secondary(self, **updates):
        params = dict(self.params)
        params.update(updates)
        return QuadraticSystem(target=params["target"], weight=params["weight"])


class MaximizingQuadraticSystem(QuadraticSystem):
    """A system whose declared analytical objective must be maximized."""

    def system_spec(self):
        return basic_system_spec(
            "engine-maximizing-quadratic",
            control_spec=self.spec,
            objective_direction="maximize",
        )

    def evaluate(self, controls):
        self.evaluate_calls += 1
        diff = controls.as_matrix(copy=False) - self.target
        return {"J": -self.weight * float(np.sum(diff * diff))}

    def gradient(self, controls):
        self.gradient_calls += 1
        diff = controls.as_matrix(copy=False) - self.target
        return Controls.from_matrix(self.spec, -2.0 * self.weight * diff, name="gradient")


def descent_step(context):
    step_size = context.state.step_size or 0.2
    calls = int(context.state.optimizer_state.get("calls", 0)) + 1
    return StepProposal(
        controls=context.state.controls - step_size * context.gradient,
        step_size=step_size,
        optimizer_state={"calls": calls},
        technical={"kind": "descent"},
    )


def same_step(context):
    return StepProposal(
        controls=context.state.controls.copy(name="same"),
        step_size=context.state.step_size,
        technical={"kind": "same"},
    )


def uphill_step(context):
    step_size = context.state.step_size or 0.2
    return StepProposal(
        controls=context.state.controls + step_size * context.gradient,
        step_size=step_size,
        technical={"kind": "uphill"},
    )


def exploding_step(context):
    return StepProposal(
        controls=Controls.constant(context.state.controls.spec, 2.0, name="explode"),
        step_size=context.state.step_size,
        technical={"kind": "explode"},
    )


def square(value):
    return value * value


class EngineTests(unittest.TestCase):
    def test_run_chunk_moves_down_quadratic_and_records_trace(self):
        system = QuadraticSystem(target=0.0)
        controls = Controls.constant(system.control_spec(), 1.0, name="initial")
        trace = Trace(run_id="test-run")

        result = run_chunk(
            system,
            controls,
            step=descent_step,
            optimizer_name="gradient_descent",
            maxiter=3,
            step_size=0.25,
            trace=trace,
            create_trace=False,
            stage="phase5-test",
        )

        self.assertEqual(result.stop_reason, "maxiter")
        self.assertEqual(result.iterations, 3)
        self.assertLess(result.J, 4.0)
        self.assertIs(result.trace, trace)
        self.assertEqual(len(trace.iteration_records), 3)
        self.assertEqual(len(trace.chunk_records), 1)
        self.assertIn("chunk_start", trace.labels)
        self.assertIn("accepted", trace.labels)
        self.assertIn("best_J", trace.labels)
        self.assertEqual(result.state.optimizer_state["calls"], 3)

    def test_default_accept_rejects_worse_trial_without_moving_controls(self):
        system = QuadraticSystem(target=0.0)
        controls = Controls.constant(system.control_spec(), 1.0, name="initial")

        result = run_chunk(
            system,
            controls,
            step=uphill_step,
            optimizer_name="bad_step",
            maxiter=1,
            step_size=0.25,
        )

        np.testing.assert_allclose(result.controls.as_matrix(), controls.as_matrix())
        self.assertEqual(result.stop_reason, "maxiter")
        self.assertFalse(result.trace.iteration_records[0].accepted)

    def test_declared_maximization_uses_system_objective_for_direction_and_acceptance(self):
        system = MaximizingQuadraticSystem(target=1.0)
        controls = Controls.constant(system.control_spec(), 0.0, name="initial")

        result = run_chunk(
            system,
            controls,
            step=descent_step,
            optimizer_name="gradient_descent",
            maxiter=2,
            step_size=0.25,
        )

        self.assertGreater(result.J, -4.0)
        self.assertTrue(result.trace.iteration_records[0].accepted)

        targeted = run_chunk(
            system,
            controls,
            step=descent_step,
            optimizer_name="gradient_descent",
            maxiter=10,
            step_size=0.25,
            target_value=-1.0,
        )
        self.assertEqual(targeted.stop_reason, "target")

    def test_nonfinite_trial_stops_chunk_cleanly(self):
        system = QuadraticSystem(target=0.0, fail_above=1.0)
        controls = Controls.constant(system.control_spec(), 0.0, name="initial")

        result = run_chunk(
            system,
            controls,
            step=exploding_step,
            optimizer_name="explode",
            maxiter=5,
        )

        self.assertEqual(result.stop_reason, "nonfinite_trial")
        self.assertEqual(result.iterations, 1)
        self.assertFalse(result.trace.iteration_records[0].accepted)
        self.assertIn("ValueError", result.trace.iteration_records[0].technical["error"])

    def test_target_can_stop_before_iterations(self):
        system = QuadraticSystem(target=0.0)
        controls = Controls.constant(system.control_spec(), 0.0, name="initial")

        result = run_chunk(
            system,
            controls,
            step=descent_step,
            optimizer_name="gradient_descent",
            maxiter=10,
            target_value=0.0,
        )

        self.assertEqual(result.stop_reason, "target")
        self.assertEqual(result.iterations, 0)
        self.assertEqual(len(result.trace.iteration_records), 0)

    def test_stall_stops_when_metric_does_not_improve(self):
        system = QuadraticSystem(target=0.0)
        controls = Controls.constant(system.control_spec(), 1.0, name="initial")

        result = run_chunk(
            system,
            controls,
            step=same_step,
            optimizer_name="same",
            maxiter=10,
            stall_patience=2,
        )

        self.assertEqual(result.stop_reason, "stall")
        self.assertEqual(result.iterations, 2)
        self.assertEqual(len(result.trace.iteration_records), 2)

    def test_evaluation_cache_reuses_identical_control_content(self):
        system = QuadraticSystem(target=0.0)
        controls = Controls.constant(system.control_spec(), 1.0, name="initial")
        evaluator = SystemEvaluator(system)

        first = evaluator.evaluate(controls)
        second = evaluator.evaluate(controls.copy(name="copy"))

        self.assertEqual(first.J, second.J)
        self.assertIsNot(first, second)
        self.assertEqual(evaluator.evaluation_count, 1)
        self.assertEqual(system.evaluate_calls, 1)

        first.metrics["J"] = -123.0
        first.controls.set_channel("u", [9.0, 9.0, 9.0, 9.0])
        restored = evaluator.evaluate(controls)
        self.assertEqual(restored.J, 4.0)
        np.testing.assert_allclose(restored.controls.as_matrix(), controls.as_matrix())

    def test_explicit_stopping_config_inherits_maximize_objective_direction(self):
        system = MaximizingQuadraticSystem(target=1.0)
        controls = Controls.constant(system.control_spec(), 0.0, name="initial")

        automatic = run_chunk(
            system,
            controls,
            step=descent_step,
            optimizer_name="gradient_descent",
            maxiter=3,
            step_size=0.25,
            target_value=-1.0,
        )
        explicit = run_chunk(
            system,
            controls,
            step=descent_step,
            optimizer_name="gradient_descent",
            maxiter=3,
            step_size=0.25,
            stopping=StoppingConfig(maxiter=3, target_value=-1.0),
        )

        self.assertEqual(explicit.stop_reason, automatic.stop_reason)
        self.assertEqual(explicit.iterations, automatic.iterations)

    def test_library_cache_is_effectual_on_direct_and_bound_public_calls(self):
        system = QuadraticSystem(target=0.0)
        controls = Controls.constant(system.control_spec(), 1.0, name="initial")
        library = __import__("optimizer").OptimizerLibrary()

        for _ in range(3):
            library.evaluate(system, controls, use_cache=True)
        self.assertEqual(system.evaluate_calls, 1)
        for _ in range(3):
            library.evaluate(system, controls, use_cache=False)
        self.assertEqual(system.evaluate_calls, 4)

        context = library.context(system)
        for _ in range(2):
            context.evaluate(controls, use_cache=True)
        self.assertEqual(system.evaluate_calls, 4)

    def test_parallel_map_serial_and_thread_preserve_order(self):
        values = [1, 2, 3, 4]

        self.assertEqual(parallel_map(square, values), [1, 4, 9, 16])
        threaded = parallel_map(
            square,
            values,
            config=ParallelConfig(backend="thread", workers=2),
        )
        self.assertEqual(threaded, [1, 4, 9, 16])


if __name__ == "__main__":
    unittest.main()
