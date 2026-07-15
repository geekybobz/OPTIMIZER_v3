"""Tests for the real universal fourth-order system adapter.

Why this file exists
--------------------
The temporary fixture is useful for fast optimizer tests, but the library also needs
one real downstream-style ``system.py`` that follows the v3 contract.  These tests
make sure the local adapter can evaluate the copied previous-best control, expose
analytical gradients, and provide residual Jacobians for repair/projected tools.
"""

import unittest

import optimizer as opt
from systems.universal_robust_4th import FOURTH_ORDER_BEST_CONTROLS, system


class UniversalRobustFourthSystemTests(unittest.TestCase):
    def test_reference_best_control_evaluates_to_fourth_order_metrics(self):
        if not FOURTH_ORDER_BEST_CONTROLS.exists():
            self.skipTest("private fourth-order reference control is not tracked in git")

        qsys = system({"N": 1001, "tau": 1.0})
        controls = qsys.reference_controls()

        metrics = qsys.evaluate(controls)

        self.assertGreater(metrics["fidelity"], 0.999999999999)
        self.assertLess(metrics["F_norm2"], 1.0e-14)
        self.assertLess(metrics["C_sym_norm2"], 1.0e-11)
        self.assertAlmostEqual(metrics["energy"], 212.720982011202, places=9)

    def test_adapter_satisfies_v3_system_contract(self):
        qsys = system({"N": 31, "tau": 1.0, "lambda2": 0.2, "lambda4": 0.1})
        controls = opt.guesses.random_fourier_guess(qsys, amplitude=0.15, modes=3, seed=9)

        probe = opt.probe_system(qsys)
        gradient = qsys.gradient(controls)
        hard = qsys.residuals(controls, name="hard")
        full = qsys.residuals(controls, name="fourth_order")
        jac = qsys.jacobian(controls, name="hard")

        self.assertTrue(probe.required_ok)
        self.assertTrue(probe.has_residuals)
        self.assertTrue(probe.has_jacobian)
        self.assertEqual(gradient.shape, controls.shape)
        self.assertEqual(hard.shape, (8,))
        self.assertEqual(full.shape, (20,))
        self.assertEqual(jac.shape, (8, controls.spec.size))

    def test_analytical_gradient_and_hard_jacobian_match_finite_differences(self):
        qsys = system({"N": 31, "tau": 1.0, "lambda2": 0.1, "lambda4": 0.1})
        controls = opt.guesses.random_fourier_guess(qsys, amplitude=0.2, modes=3, seed=1)

        grad_check = opt.utils.verify_gradient(qsys, controls, eps=1.0e-6, directions=2, seed=3)
        jac_check = opt.utils.verify_jacobian(
            qsys,
            controls,
            residuals="hard",
            eps=1.0e-6,
            directions=1,
            seed=4,
        )

        self.assertTrue(grad_check["passed"])
        self.assertLess(jac_check["max_relative_error"], 1.0e-5)


if __name__ == "__main__":
    unittest.main()
