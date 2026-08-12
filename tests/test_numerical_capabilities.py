"""Integration tests for conditional OLGS numerical capabilities."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

import optimizer as opt
from optimizer.tracker import JournalObserver, RunJournal, RunReceipt, SourceReference, read_timeline


class SmallQuadratic:
    def __init__(self, target=(1.0, -2.0), scale=1.0):
        self.target = np.asarray(target, dtype=float)
        self.scale = float(scale)
        self.value_gradient_calls = 0
        self.evaluate_calls = 0
        self.gradient_calls = 0
        self.hvp_calls = 0

    def control_spec(self):
        return opt.ControlSpec(("u",), 2)

    def system_spec(self):
        return opt.basic_system_spec(
            "small-quadratic",
            control_spec=self.control_spec(),
            derivatives={"hvp": {"hook": "hvp", "source": "analytical"}},
        )

    def evaluate(self, controls):
        self.evaluate_calls += 1
        diff = controls.flatten(copy=False) - self.scale * self.target
        return {"J": 0.5 * float(np.dot(diff, diff))}

    def gradient(self, controls):
        self.gradient_calls += 1
        diff = controls.flatten(copy=False) - self.scale * self.target
        return opt.Controls.from_flat(self.control_spec(), diff, name="gradient")

    def value_and_gradient(self, controls):
        self.value_gradient_calls += 1
        return opt.ValueGradient(self.evaluate(controls), self.gradient(controls))

    def hvp(self, controls, vector):
        del controls
        self.hvp_calls += 1
        return vector.copy(name="hvp")

    def with_secondary(self, **updates):
        return SmallQuadratic(target=self.target, scale=updates.get("scale", self.scale))


class MaximizingSmallQuadratic(SmallQuadratic):
    """Exact negation of SmallQuadratic used for direction-symmetry coverage."""

    def system_spec(self):
        return opt.basic_system_spec(
            "maximizing-small-quadratic",
            control_spec=self.control_spec(),
            objective_direction="maximize",
            derivatives={"hvp": {"hook": "hvp", "source": "analytical"}},
        )

    def evaluate(self, controls):
        self.evaluate_calls += 1
        diff = controls.flatten(copy=False) - self.scale * self.target
        return {"J": -0.5 * float(np.dot(diff, diff))}

    def gradient(self, controls):
        self.gradient_calls += 1
        diff = controls.flatten(copy=False) - self.scale * self.target
        return opt.Controls.from_flat(self.control_spec(), -diff, name="gradient")

    def hvp(self, controls, vector):
        del controls
        self.hvp_calls += 1
        return -vector.copy(name="hvp")


class EqualityQuadratic:
    def control_spec(self):
        return opt.ControlSpec(("u",), 2)

    def system_spec(self):
        return opt.basic_system_spec(
            "equality-quadratic",
            control_spec=self.control_spec(),
            metrics={"sum": {"label": "Control sum", "role": "constraint"}},
            derivatives={"constraint_jacobian": {"hook": "jacobian", "source": "analytical"}},
            constraints={"hard": {"kind": "equality", "dimension": 1}},
        )

    def evaluate(self, controls):
        flat = controls.flatten(copy=False)
        return {"J": 0.5 * float(np.dot(flat, flat)), "sum": float(np.sum(flat))}

    def gradient(self, controls):
        return controls.copy(name="gradient")

    def residuals(self, controls, *, name="hard"):
        self.last_residual_name = name
        return np.array([float(np.sum(controls.flatten(copy=False)) - 1.0)])

    def jacobian(self, controls, *, name="hard"):
        del controls, name
        return np.array([[1.0, 1.0]])

    def control_bounds(self):
        spec = self.control_spec()
        return opt.ControlBounds(
            lower=opt.Controls.from_flat(spec, np.zeros(2)),
            upper=opt.Controls.from_flat(spec, np.ones(2)),
        )

    def constraint_spec(self):
        return {"sets": {"hard": {"kind": "equality", "dimension": 1, "scales": [1.0]}}}

    def with_secondary(self, **updates):
        del updates
        return EqualityQuadratic()


class BoxQuadratic(SmallQuadratic):
    def __init__(self):
        super().__init__(target=(2.0, -2.0))

    def control_bounds(self):
        spec = self.control_spec()
        return opt.ControlBounds(
            lower=opt.Controls.from_flat(spec, np.array([-1.0, -1.0])),
            upper=opt.Controls.from_flat(spec, np.array([1.0, 1.0])),
        )

    def with_secondary(self, **updates):
        del updates
        return BoxQuadratic()


class ActiveEqualityQuadratic(EqualityQuadratic):
    def control_bounds(self):
        spec = self.control_spec()
        return opt.ControlBounds(
            lower=opt.Controls.from_flat(spec, np.array([0.8, 0.0])),
            upper=opt.Controls.from_flat(spec, np.ones(2)),
        )

    def with_secondary(self, **updates):
        del updates
        return ActiveEqualityQuadratic()


class L1Composite:
    def __init__(self, weight=0.25):
        self.weight = float(weight)
        self.target = np.array([1.0, -0.1])

    def control_spec(self):
        return opt.ControlSpec(("u",), 2)

    def system_spec(self):
        return opt.basic_system_spec(
            "l1-composite",
            control_spec=self.control_spec(),
            metrics={
                "smooth_J": {"label": "Smooth objective", "role": "objective_component"},
                "nonsmooth_J": {"label": "Nonsmooth objective", "role": "objective_component"},
            },
            derivatives={
                "smooth_gradient": {"hook": "smooth_gradient", "source": "analytical"},
                "prox": {"hook": "prox", "source": "analytical"},
            },
        )

    def evaluate(self, controls):
        flat = controls.flatten(copy=False)
        smooth = 0.5 * float(np.dot(flat - self.target, flat - self.target))
        nonsmooth = self.weight * float(np.sum(np.abs(flat)))
        return {"J": smooth + nonsmooth, "smooth_J": smooth, "nonsmooth_J": nonsmooth}

    def gradient(self, controls):
        flat = controls.flatten(copy=False)
        subgradient = flat - self.target + self.weight * np.sign(flat)
        return opt.Controls.from_flat(self.control_spec(), subgradient, name="subgradient")

    def smooth_gradient(self, controls):
        flat = controls.flatten(copy=False)
        return opt.Controls.from_flat(self.control_spec(), flat - self.target, name="smooth_gradient")

    def prox(self, controls, step_size):
        flat = controls.flatten(copy=False)
        threshold = float(step_size) * self.weight
        result = np.sign(flat) * np.maximum(np.abs(flat) - threshold, 0.0)
        return opt.Controls.from_flat(self.control_spec(), result, name="prox")

    def with_secondary(self, **updates):
        return L1Composite(weight=updates.get("weight", self.weight))


class SizedQuadratic:
    def __init__(self, size=2):
        self.size = int(size)

    def control_spec(self):
        return opt.ControlSpec(("u",), self.size, dt=1.0 / self.size)

    def system_spec(self):
        return opt.basic_system_spec("sized-quadratic", control_spec=self.control_spec())

    def evaluate(self, controls):
        return {"J": 0.5 * float(np.dot(controls.flatten(copy=False), controls.flatten(copy=False)))}

    def gradient(self, controls):
        return controls.copy(name="gradient")

    def with_secondary(self, **updates):
        del updates
        return SizedQuadratic(self.size)


class NumericalCapabilityTests(unittest.TestCase):
    def test_capability_probe_and_combined_api(self):
        system = SmallQuadratic()
        controls = opt.Controls.zeros(system.control_spec())
        info = opt.system_info(system)
        self.assertTrue(info["capabilities"]["value_and_gradient"]["available"])
        self.assertTrue(info["capabilities"]["second_order"]["newton_cg_ready"])

        evaluation, gradient = opt.value_and_gradient(system, controls)
        self.assertGreater(evaluation.J, 0.0)
        self.assertEqual(gradient.spec.shape, controls.spec.shape)

        result = opt.adam(system, controls, step_size=0.05, maxiter=1)
        self.assertLess(result.J, evaluation.J)
        self.assertGreaterEqual(system.value_gradient_calls, 2)

    def test_projected_optimizer_restores_and_satisfies_kkt(self):
        system = EqualityQuadratic()
        controls = opt.Controls.from_flat(system.control_spec(), np.array([0.0, 0.0]))
        result = opt.projected_gradient(
            system,
            controls,
            step_size=0.5,
            maxiter=20,
            feasibility_tolerance=1.0e-10,
            stationarity_tolerance=1.0e-9,
        )
        np.testing.assert_allclose(result.controls.flatten(), np.array([0.5, 0.5]), atol=1.0e-8)
        self.assertLessEqual(abs(np.sum(result.controls.flatten()) - 1.0), 1.0e-10)
        self.assertIn(result.stop_reason, {"projected_kkt_converged", "maxiter"})
        self.assertLessEqual(result.state.optimizer_state["projected_kkt_norm"], 1.0e-8)

    def test_projected_optimizer_supports_pure_box_constraints(self):
        system = BoxQuadratic()
        controls = opt.Controls.zeros(system.control_spec())
        result = opt.projected_gradient(system, controls, step_size=1.0, maxiter=5)
        np.testing.assert_allclose(result.controls.flatten(), np.array([1.0, -1.0]), atol=1.0e-10)
        self.assertEqual(result.stop_reason, "projected_kkt_converged")
        self.assertEqual(result.state.optimizer_state["active_bound_count"], 2)

    def test_projected_kkt_combines_equalities_and_active_bounds(self):
        system = ActiveEqualityQuadratic()
        controls = opt.Controls.from_flat(system.control_spec(), np.array([0.8, 0.2]))
        result = opt.projected_gradient(system, controls, maxiter=3, stationarity_tolerance=1.0e-8)
        np.testing.assert_allclose(result.controls.flatten(), np.array([0.8, 0.2]), atol=1.0e-9)
        self.assertEqual(result.stop_reason, "projected_kkt_converged")
        self.assertEqual(result.state.optimizer_state["active_bound_count"], 1)

    def test_newton_cg_and_trust_region_use_hvp(self):
        system = SmallQuadratic()
        controls = opt.Controls.zeros(system.control_spec())
        newton = opt.newton_cg(system, controls, maxiter=5, step_size=1.0)
        trust = opt.trust_region(system, controls, maxiter=10, initial_radius=3.0)
        np.testing.assert_allclose(newton.controls.flatten(), system.target, atol=1.0e-8)
        np.testing.assert_allclose(trust.controls.flatten(), system.target, atol=1.0e-7)
        self.assertEqual(newton.stop_reason, "gradient_converged")
        self.assertEqual(trust.stop_reason, "gradient_converged")
        self.assertGreater(newton.trace.iteration_records[0].technical["counters"]["hvp"], 0)
        self.assertGreater(trust.trace.iteration_records[0].technical["counters"]["hvp"], 0)

    def test_newton_cg_has_minimize_maximize_hvp_symmetry(self):
        minimizing = SmallQuadratic()
        maximizing = MaximizingSmallQuadratic()
        min_result = opt.newton_cg(minimizing, opt.Controls.zeros(minimizing.control_spec()), maxiter=3)
        max_result = opt.newton_cg(maximizing, opt.Controls.zeros(maximizing.control_spec()), maxiter=3)

        np.testing.assert_allclose(min_result.controls.flatten(), max_result.controls.flatten(), atol=1.0e-12)
        self.assertFalse(max_result.trace.iteration_records[0].technical["proposal"]["negative_curvature"])

    def test_proximal_gradient_uses_mapping_stationarity(self):
        system = L1Composite(weight=0.25)
        controls = opt.Controls.zeros(system.control_spec())
        result = opt.proximal_gradient(system, controls, step_size=1.0, maxiter=5)
        expected = np.array([0.75, 0.0])
        np.testing.assert_allclose(result.controls.flatten(), expected, atol=1.0e-10)
        self.assertEqual(result.stop_reason, "proximal_mapping_converged")
        self.assertLessEqual(result.state.optimizer_state["proximal_gradient_mapping_norm"], 1.0e-8)

    def test_continuation_rebuilds_system_and_links_stages(self):
        system = SmallQuadratic(scale=0.0)
        controls = opt.Controls.zeros(system.control_spec())
        result = opt.continuation(
            system,
            controls,
            stages=[
                opt.ContinuationStage(
                    label="half_target",
                    optimizer="newton_cg",
                    secondary_updates={"scale": 0.5},
                    optimizer_options={"maxiter": 3},
                    gate=opt.StageGate(objective_target=1.0e-12),
                ),
                opt.ContinuationStage(
                    label="full_target",
                    optimizer="newton_cg",
                    secondary_updates={"scale": 1.0},
                    optimizer_options={"maxiter": 3},
                    gate=opt.StageGate(objective_target=1.0e-12),
                ),
            ],
        )
        self.assertEqual(result.stop_reason, "completed")
        self.assertEqual(len(result.stages), 2)
        self.assertTrue(all(stage.gate_passed for stage in result.stages))
        self.assertTrue(all(stage.transfer == "compatible_copy" for stage in result.stages))
        np.testing.assert_allclose(result.controls.flatten(), system.target, atol=1.0e-8)
        self.assertEqual(result.to_dict()["plan_id"], result.plan_id)

    def test_capability_gates_fail_before_iterations(self):
        system = SmallQuadratic()
        controls = opt.Controls.zeros(system.control_spec())
        with self.assertRaisesRegex(TypeError, "residuals"):
            opt.projected_gradient(system, controls, maxiter=1)
        with self.assertRaisesRegex(TypeError, "smooth_gradient"):
            opt.proximal_gradient(system, controls, maxiter=1)

    def test_advanced_tracker_is_passive_and_preserves_hvp_counters(self):
        baseline_system = SmallQuadratic()
        tracked_system = SmallQuadratic()
        baseline_controls = opt.Controls.zeros(baseline_system.control_spec())
        tracked_controls = opt.Controls.zeros(tracked_system.control_spec())
        baseline = opt.newton_cg(
            baseline_system,
            baseline_controls,
            maxiter=3,
            create_trace=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            receipt = RunReceipt.create(
                run_id="advanced-newton",
                objective_label="J",
                direction="minimize",
                references={
                    "problem": SourceReference.create("numerical-capability-test"),
                    "system": SourceReference.create("small-quadratic"),
                },
                method_config={"name": "newton_cg"},
            )
            journal = RunJournal.create(Path(directory) / "run", receipt)
            tracked = opt.newton_cg(
                tracked_system,
                tracked_controls,
                maxiter=3,
                create_trace=False,
                observers=[JournalObserver(journal)],
            )
            np.testing.assert_allclose(tracked.controls.flatten(), baseline.controls.flatten())
            self.assertEqual(tracked.metrics, baseline.metrics)
            self.assertEqual(
                (tracked_system.evaluate_calls, tracked_system.gradient_calls, tracked_system.hvp_calls),
                (baseline_system.evaluate_calls, baseline_system.gradient_calls, baseline_system.hvp_calls),
            )
            iteration = next(
                record["payload"]
                for record in read_timeline(journal.run_dir).records
                if record["kind"] == "iteration"
            )
            self.assertGreater(iteration["after"]["counters"]["hvp"], 0)
            self.assertIn("cg_iterations", iteration["technical"]["proposal"])

    def test_continuation_requires_explicit_layout_transfer(self):
        system = SizedQuadratic(2)
        controls = opt.Controls.from_flat(system.control_spec(), np.array([1.0, -1.0]))
        stage = opt.ContinuationStage(
            label="refine",
            optimizer="adam",
            system_factory=lambda previous: SizedQuadratic(previous.size + 1),
            optimizer_options={"maxiter": 0},
        )
        with self.assertRaisesRegex(ValueError, "explicit control_transfer"):
            opt.continuation(system, controls, stages=[stage])

        def transfer(previous, next_system, previous_controls):
            del previous
            old = previous_controls.flatten(copy=False)
            new_grid = np.linspace(0.0, 1.0, next_system.size)
            old_grid = np.linspace(0.0, 1.0, old.size)
            return opt.Controls.from_flat(next_system.control_spec(), np.interp(new_grid, old_grid, old))

        result = opt.continuation(system, controls, stages=[stage], control_transfer=transfer)
        self.assertEqual(result.stages[0].transfer, transfer.__qualname__)
        self.assertEqual(result.controls.spec.control_dim, 3)

    def test_public_api_transition_exposes_conditional_methods(self):
        for name in ("projected_gradient", "newton_cg", "trust_region", "proximal_gradient"):
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(opt, name)))
                self.assertTrue(callable(getattr(opt.optimizers, name)))
                self.assertEqual(opt.methods()[name].status, "implemented")
                self.assertEqual(opt.info(name)["status"], "implemented")
        self.assertTrue(callable(opt.continuation))
        self.assertTrue(callable(opt.value_and_gradient))
        context = opt.context(SmallQuadratic())
        self.assertTrue(callable(context.newton_cg))
        self.assertTrue(callable(context.continuation))


if __name__ == "__main__":
    unittest.main()
