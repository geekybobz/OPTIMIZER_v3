"""Cross-family numerical invariants for the shared core consolidation."""
from __future__ import annotations
import json
import unittest
from unittest.mock import patch
import numpy as np
import optimizer as opt
from fixtures.quadratic_system import QuadraticVectorSystem
from fixtures.shooting_systems import RectangularShootingSystem, FoldContinuationShootingSystem
from optimizer.checkpoints.codec import _Codec
from optimizer.core.numerics import residual_step, tangent_projection, jacobian_decomposition
from optimizer.core.runtime import candidate_key


class ScaledQuadratic:
    def __init__(self, scales=(1e6, 1e-3)):
        self.scales = np.asarray(scales)
        self.calls = 0

    def control_spec(self):
        return opt.ControlSpec(("large", "small"), 1, scales=self.scales)

    def system_spec(self):
        return opt.basic_system_spec("scaled-quadratic", control_spec=self.control_spec())

    def evaluate(self, controls):
        self.calls += 1
        self.check(controls)
        z = controls.flatten(copy=False) / self.scales
        return {"J": float(0.5 * np.dot(z - 1, z - 1))}

    def gradient(self, controls):
        self.check(controls)
        return controls.with_flat((controls.flatten(copy=False) / self.scales - 1) / self.scales)

    def hvp(self, controls, vector):
        self.check(controls)
        return vector.with_flat(vector.flatten(copy=False) / self.scales ** 2)

    def check(self, controls):
        if controls.spec != self.control_spec():
            raise AssertionError("Physics received normalized coordinates/layout.")

    def with_secondary(self, **updates):
        return ScaledQuadratic(self.scales)


class ScaledLine(RectangularShootingSystem):
    def __init__(self):
        super().__init__()
        self._spec = opt.ShootingSpec((opt.ShootingBlock("a", scale=1e6), opt.ShootingBlock("b", scale=1e-3)))

    def system_spec(self):
        return opt.basic_shooting_system_spec("scaled-line", shooting_spec=self._spec,
                                              residual_size=1, jacobian_source="analytical")

    def residuals(self, variables, *, options=None):
        return np.array([np.sum(variables.flatten(copy=False) / variables.scales) - 2])

    def jacobian(self, variables, *, options=None):
        return np.array([[1e-6, 1e3]])


class SharedCoreTests(unittest.TestCase):
    def test_cpu_reserve_caps_executors_and_aggregate_job_threads(self):
        from optimizer.jobs.contracts import JobsConfig
        from optimizer.utils.resources import cpu_budget
        with patch("optimizer.utils.resources.available_cpus", return_value=10):
            self.assertEqual(cpu_budget(), 7)
            self.assertEqual(opt.ParallelConfig(backend="process").resolved_workers(), 7)
            self.assertEqual(opt.ParallelConfig(backend="thread", workers=99).resolved_workers(), 7)
            JobsConfig(max_workers=3, threads_per_worker=2)
            with self.assertRaisesRegex(ValueError, "minus three"):
                JobsConfig(max_workers=4, threads_per_worker=2)
        with patch("optimizer.utils.resources.available_cpus", return_value=2):
            self.assertEqual(cpu_budget(), 1)

    def test_scaled_underdetermined_newton_and_lm_choose_dimensionless_minimum_norm(self):
        system = ScaledLine()
        initial = opt.ShootingVariables.zeros(system.shooting_spec())
        for method in ("newton", "lm", "damped"):
            result = opt.solve_shooting(system, initial, method=method,
                                        verification_options={"profile": "independent"})
            self.assertTrue(result.qualified, result.to_dict())
            np.testing.assert_allclose(result.variables.values / initial.scales, [1, 1], atol=1e-9)

    def test_scaled_gradient_methods_are_invariant_to_units_and_keep_physics_physical(self):
        physical, normalized = ScaledQuadratic(), ScaledQuadratic((1, 1))
        for name in ("adam", "momentum", "adagrad", "rmsprop", "lbfgs", "nonlinear_cg", "newton_cg", "trust_region"):
            with self.subTest(method=name):
                method = getattr(opt, name)
                options = dict(maxiter=5, gradient_tolerance=None)
                a = method(physical, opt.Controls.zeros(physical.control_spec()), **options)
                b = method(normalized, opt.Controls.zeros(normalized.control_spec()), **options)
                np.testing.assert_allclose(a.controls.flatten() / physical.scales, b.controls.flatten(), rtol=2e-12, atol=1e-12)
                self.assertEqual(a.counters, b.counters)
                self.assertEqual(a.state.controls.spec, physical.control_spec())

    def test_scaled_chunk_state_matches_uninterrupted_adam(self):
        system = ScaledQuadratic()
        initial = opt.Controls.zeros(system.control_spec())
        full = opt.adam(system, initial, maxiter=8, gradient_tolerance=None)
        first = opt.adam(system, initial, maxiter=3, gradient_tolerance=None)
        second = opt.adam(system, state=first.state, maxiter=5, gradient_tolerance=None)
        np.testing.assert_allclose(second.controls.flatten(), full.controls.flatten(), rtol=1e-14)

    def test_variables_serialization_and_legacy_checkpoint_defaults(self):
        system = ScaledQuadratic()
        controls = opt.Controls.constant(system.control_spec(), {"large": 2e6, "small": 3e-3})
        shooting = opt.ShootingVariables.zeros(ScaledLine().shooting_spec())
        for variables in (controls, shooting):
            self.assertIsInstance(variables, opt.core.Variables)
            rebuilt = type(variables).from_dict(variables.spec, variables.to_dict())
            np.testing.assert_array_equal(rebuilt.flatten(), variables.flatten())
        complex_controls = opt.Controls.from_flat(opt.ControlSpec(("u",), 2, dtype=complex), [1+2j, 3-4j])
        rebuilt = opt.Controls.from_dict(complex_controls.spec, json.loads(json.dumps(complex_controls.to_dict())))
        np.testing.assert_array_equal(rebuilt.flatten(), complex_controls.flatten())
        codec = _Codec()
        payload = codec.encode(controls)
        rebuilt = _Codec(read_blob=codec.blobs.__getitem__).decode(payload)
        self.assertEqual(rebuilt.spec.scales, (1e6, 1e-3))
        legacy = _Codec().encode(opt.ControlSpec(("u",), 2))
        legacy["fields"]["$dict"] = [(key, value) for key, value in legacy["fields"]["$dict"] if key != "scales"]
        self.assertEqual(_Codec().decode(legacy).scales, (1.0,))

    def test_custom_shooting_step_uses_common_context_and_preserves_verification(self):
        system = RectangularShootingSystem()
        initial = opt.ShootingVariables.from_flat(system.shooting_spec(), [4, 3])
        calls = []
        def update(context):
            calls.append(context.iteration)
            delta = np.linalg.lstsq(context.jacobian, -context.residuals, rcond=None)[0]
            return opt.StepProposal(context.variables.with_flat(context.variables.values + delta),
                                    optimizer_state_on_accept={"custom": True})
        result = opt.run_chunk(system, initial, step=update, optimizer_name="custom", maxiter=4)
        self.assertTrue(result.solver_converged)
        self.assertEqual(result.method, "custom")
        self.assertFalse(result.qualified)  # Working accuracy alone cannot qualify.
        independent = opt.solve_shooting(system, initial, step=update, verification_options={"profile": "independent"})
        self.assertTrue(independent.qualified)
        self.assertTrue(independent.state.optimizer_state["custom"])
        self.assertEqual(calls, [0, 0])

    def test_shooting_step_cap_is_dimensionless_and_invalid_proposal_is_named(self):
        system = ScaledLine()
        initial = opt.ShootingVariables.zeros(system.shooting_spec())
        result = opt.solve_shooting(system, initial, method="newton", maxiter=1, max_step_norm=0.25)
        self.assertAlmostEqual(np.linalg.norm(result.variables.values / initial.scales), 0.25)
        failed = opt.solve_shooting(system, initial, step=lambda context: None)
        self.assertEqual(failed.stop_reason, "proposal_failed")
        self.assertFalse(failed.qualified)

    def test_repair_captures_failed_jacobian_without_new_final_evaluation(self):
        system = QuadraticVectorSystem()
        controls = system.reference_controls(amplitude=2)
        with patch.object(system, "jacobian", side_effect=ValueError("broken derivative")):
            result = opt.repair_newton(system, controls)
        self.assertEqual(result.stop_reason, "jacobian_failed")
        self.assertEqual(result.counters["jacobian_attempt"], 1)
        self.assertEqual(result.counters.get("jacobian", 0), 0)
        self.assertEqual(result.counters["residual"], 1)

    def test_records_are_bounded_and_diagnosis_makes_no_system_calls(self):
        system = QuadraticVectorSystem()
        result = opt.adam(system, system.reference_controls(amplitude=2), maxiter=270, gradient_tolerance=None)
        self.assertEqual(result.run_record.total, 270)
        self.assertEqual(len(result.run_record.rows), 256)
        self.assertTrue(result.run_record.truncated)
        with patch.object(system, "evaluate", side_effect=AssertionError("extra propagation")):
            diagnosis = opt.diagnose_run(result)
            json.dumps(result.to_dict(), allow_nan=False)
        self.assertEqual(diagnosis["iterations"], 270)
        codec = _Codec()
        payload = codec.encode(result)
        decoded = _Codec(read_blob=codec.blobs.__getitem__).decode(payload)
        self.assertEqual(decoded.run_record.total, 270)

    def test_cma_generator_state_survives_typed_checkpoint(self):
        system = QuadraticVectorSystem()
        controls = system.reference_controls(amplitude=2)
        options = dict(seed=7, population_size=8)
        full = opt.cma_es(system, controls, maxiter=9, **options)
        first = opt.cma_es(system, controls, maxiter=4, **options)
        codec = _Codec(); payload = codec.encode(first)
        saved = _Codec(read_blob=codec.blobs.__getitem__).decode(payload)
        second = opt.cma_es(system, state=saved.state, maxiter=5, **options)
        np.testing.assert_array_equal(full.controls.flatten(), second.controls.flatten())
        np.testing.assert_array_equal(full.state.optimizer_state["mean"], second.state.optimizer_state["mean"])
        self.assertEqual(second.state.optimizer_state["generation"], 9)

    def test_cache_identity_includes_scales_and_captured_failure_counts_attempt(self):
        a = opt.Controls.zeros(opt.ControlSpec(("u",), 2, scales=(1,)))
        b = opt.Controls.zeros(opt.ControlSpec(("u",), 2, scales=(100,)))
        self.assertNotEqual(candidate_key(a), candidate_key(b))
        system = RectangularShootingSystem()
        runtime = opt.ShootingEvaluator(system)
        with patch.object(system, "residuals", side_effect=ValueError("propagation failed")):
            outcome = runtime.try_residuals(opt.ShootingVariables.zeros(system.shooting_spec()))
        self.assertFalse(outcome.ok)
        self.assertEqual(runtime.counters()["residual_attempt"], 1)
        self.assertEqual(runtime.counters()["residual"], 0)

    def test_wide_kernel_does_not_allocate_variable_space_identity(self):
        matrix = np.ones((2, 10000)); residual = np.array([1., 1.])
        with patch("optimizer.core.numerics.np.eye", wraps=np.eye) as identity:
            delta = residual_step(matrix, residual, method="lm", damping=1e-8)
        self.assertEqual(identity.call_args.args[0], 2)
        np.testing.assert_allclose(matrix @ delta, -residual, atol=1e-10)
        projected = tangent_projection(np.ones(10000), matrix)
        self.assertLess(np.linalg.norm(matrix @ projected), 1e-8)

    def test_parallel_population_preserves_serial_candidates(self):
        system = QuadraticVectorSystem()
        initial = system.reference_controls(amplitude=2)
        serial = opt.cma_es(system, initial, maxiter=3, seed=9, population_size=6)
        threaded = opt.cma_es(system, initial, maxiter=3, seed=9, population_size=6,
                              parallel=opt.ParallelConfig(backend="thread", workers=2),
                              system_factory=QuadraticVectorSystem)
        np.testing.assert_array_equal(serial.controls.flatten(), threaded.controls.flatten())
        self.assertEqual(serial.state.optimizer_state["rng_state"], threaded.state.optimizer_state["rng_state"])

    def test_continuation_attempts_preserve_rejections_and_exact_target(self):
        system = FoldContinuationShootingSystem()
        previous = opt.ShootingVariables.from_flat(system.shooting_spec(), [-1, 1])
        current = opt.ShootingVariables.from_flat(system.shooting_spec(), [-.8, .64])
        result = opt.continue_shooting(system, previous, current, steps=12, step_size=.4,
                                       target_block="x", target_value=.2,
                                       verification_options={"profile": "independent"})
        self.assertEqual(result.stop_reason, "target_reached")
        self.assertAlmostEqual(result.points[-1].variables.block("x"), .2)
        self.assertTrue(all({"index", "accepted", "reason"} <= set(row) for row in result.step_history))

    def test_rank_policy_matches_numpy_for_rectangular_rank_deficiency(self):
        matrix = np.zeros((100, 2)); matrix[0, 0] = 1; matrix[1, 1] = 1e-15
        _, _, rank, _, _ = jacobian_decomposition(matrix)
        self.assertEqual(rank, np.linalg.matrix_rank(matrix))


if __name__ == "__main__":
    unittest.main()
