import unittest

import numpy as np

import optimizer as opt
from fixtures.shooting_systems import CubicRootShootingSystem
from test_contained.olss_scenarios.probe import article_scenario_matrix
from test_contained.olss_scenarios.systems import (
    ArticleFourthOrderTwentySystem,
    ArticleGateTwelveSystem,
    ArticleSixthOrderPlanarSystem,
    BoundedSystem,
    CartDirectPolynomialSystem,
    InconsistentOverdeterminedSystem,
    PendulumFixedTargetSystem,
    PendulumSwingUpSystem,
    RobustTransferSecondOrderSystem,
    UnstableMultipleShootingSystem,
    WrongJacobianSystem,
)


def _variables(system, values):
    return opt.ShootingVariables.from_flat(system.shooting_spec(), values)


class OLSSArticleScenarioTests(unittest.TestCase):
    def test_article_dimension_probes_solve_known_roots(self):
        cases = (
            (ArticleGateTwelveSystem, (12, 12)),
            (ArticleFourthOrderTwentySystem, (20, 21)),
            (ArticleSixthOrderPlanarSystem, (4, 4)),
        )
        for system_type, expected_shape in cases:
            with self.subTest(system=system_type.__name__):
                system = system_type()
                initial = opt.ShootingVariables.zeros(system.shooting_spec())
                derivative = opt.verify_shooting_jacobian(system, initial)
                result = opt.solve_shooting(
                    system,
                    initial,
                    method="newton",
                    tolerance=1.0e-11,
                    verification_tolerance=1.0e-11,
                    maxiter=5,
                )

                self.assertEqual(
                    (system.shooting_spec().size, system.system_spec()["residuals"]["size"]),
                    expected_shape,
                )
                self.assertTrue(derivative["passed"], derivative)
                self.assertTrue(result.qualified, result.to_dict())
                np.testing.assert_allclose(
                    result.variables.flatten(), system.target(), atol=1.0e-11
                )

    def test_second_order_8x9_physical_transaction_and_reconstruction(self):
        system = RobustTransferSecondOrderSystem(primary={"steps": 80})
        point = _variables(system, [2.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])

        residuals = system.residuals(point)
        jacobian = system.jacobian(point)
        derivative = opt.verify_shooting_jacobian(
            system, point, eps=1.0e-5, atol=2.0e-5, rtol=2.0e-4
        )
        recovered = system.recover_controls(point, times=np.linspace(0.0, 1.0, 41))

        self.assertEqual(residuals.shape, (9,))
        self.assertEqual(jacobian.shape, (9, 8))
        self.assertTrue(np.all(np.isfinite(residuals)))
        self.assertTrue(np.all(np.isfinite(jacobian)))
        self.assertTrue(derivative["passed"], derivative)
        self.assertEqual(recovered["values"].shape, (3, 41))
        self.assertEqual(recovered["channels"], ("ux", "uy", "uz"))

    def test_method_probe_separates_availability_from_applicability(self):
        matrix = article_scenario_matrix()
        dimensions = {
            name: (
                report["dimensions"]["variables"],
                report["dimensions"]["residuals"],
            )
            for name, report in matrix.items()
        }

        self.assertEqual(dimensions["second_order_8x9"], (8, 9))
        self.assertEqual(dimensions["gate_12x12"], (12, 12))
        self.assertEqual(dimensions["fourth_order_20x21"], (20, 21))
        self.assertEqual(dimensions["sixth_order_planar_4x4"], (4, 4))
        self.assertTrue(all(report["contract_valid"] for report in matrix.values()))
        for report in matrix.values():
            self.assertTrue(report["methods"]["root"]["available"])
            self.assertTrue(report["methods"]["multistart"]["available"])
            self.assertTrue(report["methods"]["jacobian_check"]["available"])
        self.assertTrue(matrix["fold_2x1"]["methods"]["pseudo_arclength"]["applicable"])
        self.assertTrue(matrix["direct_3x2"]["methods"]["direct"]["applicable"])
        self.assertTrue(
            matrix["second_order_8x9"]["methods"]["control_recovery"]["applicable"]
        )
        self.assertFalse(matrix["gate_12x12"]["methods"]["direct"]["applicable"])

    def test_natural_parameter_steps_then_pseudo_arclength(self):
        roots = []
        guess = 0.2
        for fraction in (0.1, 0.2):
            system = PendulumFixedTargetSystem(
                primary={"steps": 300}, secondary={"target_fraction": fraction}
            )
            result = opt.solve_shooting(
                system,
                _variables(system, [guess]),
                method="lm",
                tolerance=1.0e-10,
                verification_tolerance=1.0e-8,
                maxiter=40,
                max_step_norm=0.5,
                options={"steps": 300},
                verification_options={"steps": 1200},
            )
            self.assertTrue(result.qualified, result.to_dict())
            guess = float(result.variables.block("initial_velocity"))
            roots.append((guess, fraction))

        branch_system = PendulumSwingUpSystem(primary={"steps": 300})
        branch = opt.continue_shooting(
            branch_system,
            _variables(branch_system, roots[0]),
            _variables(branch_system, roots[1]),
            steps=5,
            step_size=0.12,
            tolerance=1.0e-9,
            verification_tolerance=1.0e-7,
            options={"steps": 300},
            verification_options={"steps": 1200},
        )

        self.assertEqual(branch.stop_reason, "completed", branch.to_dict())
        self.assertEqual(branch.completed_steps, 5)
        self.assertTrue(all(point.verified for point in branch.points))
        self.assertGreater(
            float(branch.points[-1].variables.block("target_fraction")), 0.4
        )

    def test_direct_route_improves_cost_after_feasibility(self):
        system = CartDirectPolynomialSystem()
        initial = opt.ShootingVariables.zeros(system.shooting_spec())
        feasible = opt.solve_shooting(
            system, initial, method="newton", tolerance=1.0e-12
        )
        selected = opt.solve_direct_shooting(
            system,
            initial,
            tolerance=1.0e-9,
            verification_tolerance=1.0e-9,
            maxiter=20,
            step_size=1.0,
        )

        self.assertTrue(feasible.qualified, feasible.to_dict())
        self.assertTrue(selected.qualified, selected.to_dict())
        self.assertLess(float(selected.metrics["energy"]), float(feasible.metrics["energy"]))
        self.assertLess(selected.residual_norm, 1.0e-9)

    def test_multiple_shooting_stabilizes_long_unstable_propagation(self):
        system = UnstableMultipleShootingSystem(primary={"k": 10.0, "segments": 8})
        initial = _variables(system, [-10.0] + [0.0, 0.0] * 7)
        result = opt.solve_shooting(
            system,
            initial,
            method="newton",
            tolerance=1.0e-10,
            verification_tolerance=1.0e-10,
            maxiter=5,
        )

        self.assertEqual(system.shooting_spec().size, 15)
        self.assertEqual(system.system_spec()["residuals"]["size"], 15)
        self.assertTrue(result.qualified, result.to_dict())
        self.assertLess(result.residual_norm, 1.0e-10)

    def test_failure_bounds_and_wrong_derivative_are_visible(self):
        inconsistent = InconsistentOverdeterminedSystem()
        failed = opt.solve_shooting(
            inconsistent,
            _variables(inconsistent, [0.0]),
            method="newton",
            tolerance=1.0e-12,
            maxiter=10,
        )
        bounded = BoundedSystem()
        clipped = opt.solve_shooting(
            bounded,
            _variables(bounded, [2.0]),
            method="lm",
            tolerance=1.0e-10,
            maxiter=40,
        )
        wrong = WrongJacobianSystem()
        derivative = opt.verify_shooting_jacobian(wrong, _variables(wrong, [0.2]))

        self.assertFalse(failed.qualified)
        self.assertGreater(failed.residual_norm, 0.7)
        self.assertTrue(clipped.qualified, clipped.to_dict())
        self.assertGreaterEqual(float(clipped.variables.block("x")), 0.0)
        self.assertLessEqual(float(clipped.variables.block("x")), 1.0)
        self.assertFalse(derivative["passed"])

    def test_process_multistart_owns_worker_systems(self):
        system = CubicRootShootingSystem()
        seeds = [_variables(system, [value]) for value in (-2.0, -0.2, 0.2, 2.0)]

        campaign = opt.solve_shooting_multistart(
            system,
            seeds,
            parallel=opt.ParallelConfig(backend="process", workers=2),
            system_factory=CubicRootShootingSystem,
            solver_options={
                "method": "newton",
                "tolerance": 1.0e-11,
                "verification_options": {"profile": "independent"},
            },
        )

        self.assertEqual(len(campaign.results), 4)
        self.assertEqual(len(campaign.unique_qualified), 3)
        self.assertAlmostEqual(float(campaign.best.variables.block("x")), 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
