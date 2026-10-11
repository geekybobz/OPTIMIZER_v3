import unittest

import numpy as np

import optimizer as opt
from fixtures.shooting_systems import (
    AccuracySensitiveShootingSystem,
    CubicRootShootingSystem,
    DirectCostShootingSystem,
    FoldContinuationShootingSystem,
    MultipleShootingHarmonicSystem,
    NumericalDeclaredJacobianSystem,
    PendulumBoundaryShootingSystem,
    QubitTransferShootingSystem,
    RectangularShootingSystem,
    RedundantFoldContinuationSystem,
    SymmetryEquivalentCircleSystem,
)
from test_contained.olss_scenarios.systems import BadlyScaledSystem


INDEPENDENT = {"profile": "independent"}


class OLSSContractTests(unittest.TestCase):
    def test_variable_blocks_pack_unpack_and_family_discovery(self):
        system = FoldContinuationShootingSystem()
        variables = opt.ShootingVariables.from_blocks(
            system.shooting_spec(), {"x": -1.0, "lambda": 1.0}
        )

        self.assertEqual(variables.spec.size, 2)
        self.assertEqual(variables.block("x"), -1.0)
        self.assertEqual(
            variables.spec.blocks_with_role("continuation_parameter")[0].name,
            "lambda",
        )
        with self.assertRaises(ValueError):
            variables.values[0] = 3.0
        info = opt.system_info(system)
        self.assertEqual(info["family"], "olss")
        self.assertTrue(info["probe"]["required_ok"])
        self.assertEqual(system.system_spec()["family"], "olss")
        self.assertEqual(opt.context_info(system)["system_contract"]["family"], "olss")
        self.assertEqual(opt.shooting.solve_shooting.info()["id"], "shooting.solve_shooting")

    def test_rectangular_root_combined_evaluation_and_jacobian_check(self):
        system = RectangularShootingSystem()
        initial = opt.ShootingVariables.from_flat(system.shooting_spec(), [4.0, 3.0])

        check = opt.verify_shooting_jacobian(system, initial)
        result = opt.solve_shooting(
            system,
            initial,
            method="newton",
            tolerance=1.0e-12,
            verification_options=INDEPENDENT,
        )

        self.assertTrue(check["passed"], check)
        self.assertTrue(result.qualified, result.to_dict())
        np.testing.assert_allclose(result.variables.flatten(), [1.0, -2.0], atol=1.0e-11)
        self.assertLessEqual(system.combined_calls, result.iterations + 1)
        self.assertGreater(result.counters.get("combined", 0), 0)


class OLSSProblemRobustnessTests(unittest.TestCase):
    def test_nonlinear_pendulum_boundary_shooting(self):
        system = PendulumBoundaryShootingSystem()
        initial = opt.ShootingVariables.from_flat(system.shooting_spec(), [0.3])

        result = opt.solve_shooting(
            system,
            initial,
            method="newton",
            tolerance=1.0e-11,
            verification_tolerance=2.0e-9,
            options={"steps": 800},
            verification_options={"steps": 4000, "profile": "tight"},
        )

        self.assertTrue(result.solver_converged, result.to_dict())
        self.assertTrue(result.verified, result.to_dict())
        self.assertLess(result.verification.residual_norm, 2.0e-9)
        self.assertGreater(float(result.variables.block("initial_velocity")), 0.0)

    def test_multiple_shooting_continuity_and_terminal_boundary(self):
        system = MultipleShootingHarmonicSystem()
        initial = opt.ShootingVariables.from_flat(system.shooting_spec(), [0.2, 0.0, 0.0])

        result = opt.solve_shooting(
            system,
            initial,
            method="newton",
            tolerance=1.0e-12,
            verification_options=INDEPENDENT,
        )

        self.assertTrue(result.qualified, result.to_dict())
        expected = [1.0, np.sqrt(0.5), np.sqrt(0.5)]
        np.testing.assert_allclose(result.variables.flatten(), expected, atol=1.0e-11)
        self.assertLess(result.residual_norm, 1.0e-12)

    def test_qubit_terminal_state_transfer_with_rectangular_residual(self):
        system = QubitTransferShootingSystem()
        initial = opt.ShootingVariables.from_flat(system.shooting_spec(), [0.5])

        derivative = opt.verify_shooting_jacobian(system, initial)
        result = opt.solve_shooting(
            system,
            initial,
            method="newton",
            tolerance=1.0e-11,
            verification_tolerance=1.0e-11,
            verification_options=INDEPENDENT,
        )

        self.assertTrue(derivative["passed"], derivative)
        self.assertTrue(result.qualified, result.to_dict())
        self.assertAlmostEqual(
            float(result.variables.block("control_amplitude")), np.pi / 2.0, places=9
        )
        self.assertEqual(result.residuals.shape, (2,))

    def test_tight_verification_rejects_a_coarse_false_root(self):
        system = AccuracySensitiveShootingSystem()
        initial = opt.ShootingVariables.from_flat(system.shooting_spec(), [0.0])

        result = opt.solve_shooting(
            system,
            initial,
            method="newton",
            tolerance=1.0e-12,
            verification_tolerance=1.0e-4,
            options={"steps": 10},
            verification_options={"steps": 100000},
        )

        self.assertTrue(result.solver_converged)
        self.assertFalse(result.verified)
        self.assertFalse(result.qualified)
        self.assertEqual(result.stop_reason, "verification_failed")
        self.assertGreater(result.verification.residual_norm, 0.09)

    def test_multistart_discovers_deduplicates_and_ranks_roots(self):
        system = CubicRootShootingSystem()
        seeds = [
            opt.ShootingVariables.from_flat(system.shooting_spec(), [value])
            for value in (-2.0, -0.2, 0.2, 2.0)
        ]

        campaign = opt.solve_shooting_multistart(
            system,
            seeds,
            parallel=opt.ParallelConfig(backend="thread", workers=2),
            system_factory=CubicRootShootingSystem,
            solver_options={"method": "newton", "tolerance": 1.0e-11},
        )

        self.assertEqual(len(campaign.results), 4)
        self.assertEqual(len(campaign.unique_qualified), 3)
        self.assertEqual(campaign.selection_metric, "cost")
        self.assertAlmostEqual(float(campaign.best.variables.block("x")), 0.0, places=9)

    def test_pseudo_arclength_crosses_the_fold(self):
        system = FoldContinuationShootingSystem()
        previous = opt.ShootingVariables.from_flat(system.shooting_spec(), [-1.0, 1.0])
        current = opt.ShootingVariables.from_flat(system.shooting_spec(), [-0.8, 0.64])

        result = opt.continue_shooting(
            system,
            previous,
            current,
            steps=12,
            step_size=0.18,
            tolerance=1.0e-12,
            verification_tolerance=1.0e-10,
            verification_options=INDEPENDENT,
        )

        x_values = [float(point.variables.block("x")) for point in result.points]
        self.assertEqual(result.stop_reason, "completed", result.to_dict())
        self.assertEqual(result.completed_steps, 12)
        self.assertLess(min(abs(value) for value in x_values), 0.12)
        self.assertGreater(x_values[-1], 0.0)
        self.assertTrue(all(point.verified for point in result.points))

    def test_direct_shooting_selects_cost_minimum_on_root_manifold(self):
        system = DirectCostShootingSystem()
        initial = opt.ShootingVariables.from_flat(system.shooting_spec(), [0.0, 1.0])

        result = opt.solve_direct_shooting(
            system,
            initial,
            tolerance=1.0e-9,
            verification_tolerance=1.0e-9,
            verification_options=INDEPENDENT,
            maxiter=40,
            step_size=0.25,
        )

        self.assertTrue(result.qualified, result.to_dict())
        self.assertAlmostEqual(float(result.variables.block("x")), 2.0, places=6)
        self.assertAlmostEqual(float(result.variables.block("y")), 0.0, places=10)
        self.assertLess(float(result.metrics["cost"]), 1.0e-11)


class OLSSFixNowRegressionTests(unittest.TestCase):
    def test_multistart_uses_variable_scales_and_system_root_equivalence(self):
        scaled = BadlyScaledSystem()
        scaled_seeds = [
            opt.ShootingVariables.from_flat(scaled.shooting_spec(), [0.9e6, 0.9e-3]),
            opt.ShootingVariables.from_flat(scaled.shooting_spec(), [1.1e6, 2.2e-3]),
        ]
        scaled_campaign = opt.solve_shooting_multistart(
            scaled,
            scaled_seeds,
            solver_options={"method": "newton", "tolerance": 1.0e-12},
        )

        symmetric = SymmetryEquivalentCircleSystem()
        symmetric_seeds = [
            opt.ShootingVariables.from_flat(symmetric.shooting_spec(), [1.0, 0.0]),
            opt.ShootingVariables.from_flat(symmetric.shooting_spec(), [0.0, 1.0]),
        ]
        symmetric_campaign = opt.solve_shooting_multistart(
            symmetric,
            symmetric_seeds,
            solver_options={"method": "newton", "tolerance": 1.0e-12},
        )

        self.assertEqual(len(scaled_campaign.qualified), 2)
        self.assertEqual(len(scaled_campaign.unique_qualified), 2)
        self.assertEqual(len(symmetric_campaign.qualified), 2)
        self.assertEqual(len(symmetric_campaign.unique_qualified), 1)
        with self.assertRaisesRegex(ValueError, "dedup_tolerance"):
            opt.solve_shooting_multistart(
                scaled,
                scaled_seeds,
                dedup_tolerance=float("nan"),
            )

    def test_missing_verification_options_cannot_qualify_a_root(self):
        system = RectangularShootingSystem()
        initial = opt.ShootingVariables.from_flat(system.shooting_spec(), [4.0, 3.0])

        result = opt.solve_shooting(
            system,
            initial,
            method="newton",
            tolerance=1.0e-12,
        )

        self.assertTrue(result.solver_converged)
        self.assertFalse(result.verified)
        self.assertFalse(result.qualified)
        self.assertEqual(result.stop_reason, "verification_not_independent")
        self.assertFalse(result.verification.independent)
        self.assertEqual(
            result.verification.independence_source,
            "working_options_reused",
        )

        empty = opt.solve_shooting(
            system,
            initial,
            method="newton",
            tolerance=1.0e-12,
            verification_options={},
        )
        self.assertFalse(empty.qualified)
        self.assertFalse(empty.verification.independent)
        self.assertEqual(
            empty.verification.independence_source,
            "explicit_options_not_distinct",
        )

    def test_numerical_declared_jacobian_check_uses_independent_step(self):
        system = NumericalDeclaredJacobianSystem()
        variables = opt.ShootingVariables.from_flat(system.shooting_spec(), [0.7])

        check = opt.verify_shooting_jacobian(system, variables, eps=1.0e-6)

        self.assertTrue(check["passed"], check)
        self.assertTrue(check["independent"])
        self.assertEqual(check["declared_source"], "numerical-declared")
        self.assertEqual(check["comparison_strategy"], "separated_central_difference_step")
        self.assertAlmostEqual(check["comparison_eps"], 1.0e-5)
        self.assertGreater(check["max_abs_error"], 0.0)

    def test_pseudo_arclength_accepts_redundant_rows_when_rank_is_n_minus_one(self):
        system = RedundantFoldContinuationSystem()
        previous = opt.ShootingVariables.from_flat(system.shooting_spec(), [-1.0, 1.0])
        current = opt.ShootingVariables.from_flat(system.shooting_spec(), [-0.8, 0.64])

        result = opt.continue_shooting(
            system,
            previous,
            current,
            steps=12,
            step_size=0.18,
            tolerance=1.0e-12,
            verification_tolerance=1.0e-10,
        )

        x_values = [float(point.variables.block("x")) for point in result.points]
        self.assertEqual(result.stop_reason, "completed", result.to_dict())
        self.assertEqual(result.completed_steps, 12)
        self.assertGreater(x_values[-1], 0.0)
        self.assertTrue(all(point.verification_independent for point in result.points))


if __name__ == "__main__":
    unittest.main()
