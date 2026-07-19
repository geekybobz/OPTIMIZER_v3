"""Tests for diagnostics, repair, projection, guards, schedules, and spectrum tools."""

import unittest

import numpy as np

import optimizer as opt
from fixtures.quadratic_system import QuadraticVectorSystem


def small_system():
    """Return a small vector fixture so finite differences stay cheap."""

    return QuadraticVectorSystem(
        N=9,
        residual_weight=0.25,
        energy_weight=1.0e-3,
    )


class UtilityToolTests(unittest.TestCase):
    def test_diagnostic_report_and_geometry_probe_use_named_residuals(self):
        system = small_system()
        controls = opt.zeros(system.control_spec(), name="zero")

        report = opt.diagnostic_report(system, controls)
        geometry = opt.geometry_probe(system, controls, eps=1.0e-6)

        self.assertEqual(report["kind"], "diagnostic_report")
        self.assertTrue(report["residuals"]["available"])
        self.assertGreater(report["residuals"]["norm"], 0.0)
        self.assertEqual(geometry["kind"], "geometry_probe")
        self.assertEqual(geometry["jacobian_source"], "analytical")
        self.assertGreater(geometry["rank"], 0)

    def test_verify_gradient_passes_on_fixture_analytical_gradient(self):
        system = small_system()
        controls = system.reference_controls(amplitude=0.8)

        check = opt.verify_gradient(
            system,
            controls,
            eps=1.0e-6,
            directions=4,
            rtol=1.0e-3,
            atol=1.0e-5,
        )

        self.assertEqual(check["kind"], "gradient_check")
        self.assertTrue(check["passed"], check)

    def test_finite_difference_jacobian_project_gradient_and_nullspace(self):
        system = small_system()
        controls = system.reference_controls(amplitude=0.8)
        gradient = opt.gradient(system, controls)

        jacobian = opt.finite_difference_jacobian(system, controls, eps=1.0e-6)
        projection = opt.project_gradient(
            system,
            controls,
            gradient,
            eps=1.0e-6,
            return_info=True,
        )
        basis = opt.nullspace_basis(system, controls, eps=1.0e-6)

        self.assertEqual(jacobian.shape[1], controls.spec.size)
        self.assertIsInstance(projection["projected_gradient"], opt.Controls)
        self.assertLessEqual(
            projection["first_order_residual_change_norm"],
            np.linalg.norm(jacobian @ gradient.flatten(copy=False)) + 1.0e-12,
        )
        self.assertEqual(basis.shape[0], controls.spec.size)

    def test_repair_newton_reduces_residual_norm(self):
        system = small_system()
        controls = system.reference_controls(amplitude=0.75)
        before = np.linalg.norm(system.residuals(controls))

        fixed = opt.repair_newton(
            system,
            controls,
            maxiter=4,
            tolerance=1.0e-8,
            eps=1.0e-6,
            damping=1.0e-8,
        )

        self.assertIsInstance(fixed, opt.RepairResult)
        self.assertLess(fixed.residual_norm, before)
        self.assertIn(fixed.stop_reason, {"converged", "maxiter", "line_search_failed"})

    def test_metric_guard_controls_engine_acceptance(self):
        system = small_system()
        controls = opt.zeros(system.control_spec(), name="zero")
        guard = opt.metric_guard(
            improve="J",
            mode="min",
            require={"fidelity": (">=", 0.0)},
        )

        result = opt.line_search(
            system,
            controls,
            variant="fixed",
            step_size=0.1,
            maxiter=2,
            accept=guard,
        )

        self.assertLess(result.J, opt.evaluate(system, controls).J)
        self.assertTrue(result.trace.iteration_records[0].accepted)

    def test_schedules_spectrum_and_smoothness_reports(self):
        system = small_system()
        controls = system.reference_controls(amplitude=0.8)

        constant = opt.constant_schedule(0.25)
        adaptive = opt.adaptive_step_schedule(initial_step=1.0, shrink=0.5, grow=1.2, max_step=2.0)
        spectrum = opt.control_spectrum(controls)
        smoothness = opt.smoothness_report(controls)

        self.assertEqual(constant.update(accepted=False), 0.25)
        self.assertEqual(adaptive.update(1.0, accepted=False), 0.5)
        self.assertEqual(adaptive.update(1.0, accepted=True), 1.2)
        self.assertIn("ux", spectrum["dominant_frequency"])
        self.assertEqual(smoothness["kind"], "smoothness_report")


if __name__ == "__main__":
    unittest.main()
