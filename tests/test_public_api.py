import unittest

import numpy as np

import optimizer as opt
from fixtures.universal_robust_4th.system import TemporaryUniversalFourthOrderSystem


def public_descent_step(context):
    step_size = context.state.step_size or 1.0
    return opt.StepProposal(
        controls=context.state.controls - step_size * context.gradient,
        step_size=step_size,
        optimizer_state={"public_steps": context.iteration + 1},
        technical={"backend": "public_facade_test"},
    )


def square(value):
    return value * value


class PublicApiTests(unittest.TestCase):
    def test_import_optimizer_as_opt_exposes_direct_surface(self):
        self.assertTrue(callable(opt.run_chunk))
        self.assertTrue(callable(opt.evaluate))
        self.assertTrue(callable(opt.gradient))
        self.assertTrue(callable(opt.adam))
        self.assertTrue(callable(opt.context))
        self.assertTrue(callable(opt.bind))
        self.assertIsInstance(opt.library, opt.OptimizerLibrary)

        methods = opt.methods()
        self.assertEqual(methods["run_chunk"].status, "implemented")
        self.assertEqual(methods["context"].status, "implemented")
        self.assertEqual(methods["adam"].status, "implemented")
        self.assertEqual(methods["momentum"].status, "implemented")
        self.assertEqual(methods["line_search"].status, "implemented")

    def test_public_evaluate_gradient_and_run_chunk_with_fourth_order_fixture(self):
        system = TemporaryUniversalFourthOrderSystem(N=17, lambda2=0.25, lambda4=0.05)
        controls = opt.zeros(system.control_spec(), name="zero")

        evaluation = opt.evaluate(system, controls)
        gradient = opt.gradient(system, controls)

        self.assertGreater(evaluation.J, 0.0)
        self.assertEqual(gradient.shape, system.control_spec().shape)
        self.assertIn("C_sym_norm2", evaluation.metrics)
        self.assertTrue(np.all(np.isfinite(gradient.as_matrix())))

        trace = opt.trace("public-fourth-order")
        result = opt.run_chunk(
            system,
            controls,
            step=public_descent_step,
            optimizer_name="public_descent",
            maxiter=4,
            step_size=1.0,
            trace=trace,
            create_trace=False,
        )

        self.assertEqual(result.stop_reason, "maxiter")
        self.assertLess(result.J, evaluation.J)
        self.assertIs(result.trace, trace)
        self.assertEqual(len(trace.iteration_records), 4)
        self.assertIn("accepted", result.checkpoint_ids)
        self.assertEqual(opt.warmstart(result, target_optimizer="public_descent").target_optimizer, "public_descent")

    def test_bound_context_supports_curriculum_param_changes_without_global_system(self):
        system = TemporaryUniversalFourthOrderSystem(N=17, lambda2=0.0, lambda4=0.0)
        ctx = opt.context(system, trace="curriculum-context", step_size=1.0)
        controls = ctx.zeros(name="ctx-zero")

        self.assertIsInstance(ctx, opt.OptimizerContext)
        self.assertEqual(ctx.control_spec().keys, ("ux", "uy", "uz"))
        self.assertEqual(ctx.params["lambda2"], 0.0)

        first = ctx.run_chunk(
            controls,
            step=public_descent_step,
            optimizer_name="ctx_descent",
            maxiter=2,
        )

        ctx_second = ctx.with_params(lambda2=0.25, lambda4=0.05)
        second = ctx_second.run_chunk(
            first.controls,
            step=public_descent_step,
            optimizer_name="ctx_descent",
            maxiter=2,
            warmstart=first.warmstart(target_optimizer="ctx_descent"),
        )

        self.assertEqual(ctx.params["lambda2"], 0.0)
        self.assertEqual(ctx_second.params["lambda2"], 0.25)
        self.assertIs(ctx.trace, ctx_second.trace)
        self.assertLess(second.J, ctx_second.evaluate(first.controls).J)
        self.assertEqual(len(ctx.trace.chunk_records), 2)

    def test_bind_alias_and_context_optimizer_methods_run(self):
        ctx = opt.bind(TemporaryUniversalFourthOrderSystem(N=9))
        controls = ctx.constant(0.0, name="zero")

        result = ctx.adam(controls, maxiter=2, step_size=0.05)

        self.assertEqual(result.optimizer, "adam")
        self.assertLess(result.J, ctx.evaluate(controls).J)
        with self.assertRaisesRegex(NotImplementedError, "Phase 9"):
            ctx.fourier_guess(n_terms=3)

    def test_system_owned_params_and_vectorized_direction_simulation(self):
        system = TemporaryUniversalFourthOrderSystem(N=17, energy_weight=0.0)
        controls = system.reference_controls(amplitude=0.2)
        base = opt.evaluate(system, controls)
        heavier = opt.evaluate(system.with_params(lambda4=2.0), controls)

        self.assertGreaterEqual(heavier.J, base.J)

        directions = np.eye(3)
        robust = system.simulate(controls, alpha=0.05, directions=directions, parallel=True)
        self.assertEqual(robust["robust_backend"], "vectorized")
        self.assertFalse(robust["robust_parallel_used"])
        self.assertEqual(robust["robust_fidelity_per_direction"].shape, (3,))
        self.assertIn("terminal_overlap", robust)

    def test_future_tool_methods_still_fail_clearly_until_implemented(self):
        system = TemporaryUniversalFourthOrderSystem(N=9)

        with self.assertRaisesRegex(NotImplementedError, "Phase 9"):
            opt.fourier_guess(system.control_spec())

    def test_public_diagnostic_and_repair_methods_are_implemented(self):
        system = TemporaryUniversalFourthOrderSystem(N=9)
        controls = system.reference_controls(amplitude=0.8)

        self.assertEqual(opt.methods()["repair_newton"].status, "implemented")
        self.assertEqual(opt.methods()["geometry_probe"].status, "implemented")
        self.assertIn("metrics", opt.diagnostic_report(system, controls))
        self.assertIn("rank", opt.geometry_probe(system, controls, eps=1.0e-6))

    def test_parallel_map_is_available_from_public_facade(self):
        self.assertEqual(opt.parallel_map(square, [1, 2, 3]), [1, 4, 9])


if __name__ == "__main__":
    unittest.main()
