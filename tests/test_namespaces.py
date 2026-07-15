"""Tests for the public role namespaces.

Why this file exists
--------------------
The library has two public calling styles:

1. preferred explicit namespaces, such as ``opt.optimizers.adam(...)`` and
   ``opt.utils.verify_gradient(...)``;
2. short direct aliases, such as ``opt.adam(...)`` and ``opt.verify_gradient(...)``.

The namespace style is easier to review when the library grows because the method
name tells the reader what role the tool plays.  These tests make that style a stable
API contract instead of an accidental side effect of Python import caching.
"""

import unittest

import optimizer as opt
from fixtures.universal_robust_4th.system import TemporaryUniversalFourthOrderSystem


class NamespaceApiTests(unittest.TestCase):
    def test_role_namespaces_are_explicit_public_api(self):
        self.assertIs(opt.util, opt.utils)

        for name in ("optimizers", "utils", "util", "guesses", "schedules"):
            with self.subTest(name=name):
                self.assertIn(name, opt.__all__)

        self.assertTrue(callable(opt.optimizers.adam))
        self.assertTrue(callable(opt.optimizers.lbfgs))
        self.assertTrue(callable(opt.utils.verify_gradient))
        self.assertTrue(callable(opt.util.repair_newton))
        self.assertTrue(callable(opt.guesses.random_fourier_guess))
        self.assertTrue(callable(opt.schedules.AdaptiveStepSchedule))

    def test_namespaced_calls_run_with_the_standard_system_contract(self):
        system = TemporaryUniversalFourthOrderSystem(N=11, lambda2=0.2, lambda4=0.05)

        controls = opt.guesses.random_fourier_guess(
            system,
            amplitude=0.1,
            modes=3,
            seed=12,
        )
        initial = opt.evaluate(system, controls).J

        result = opt.optimizers.adam(system, controls, step_size=0.04, maxiter=2)
        diagnostic = opt.utils.diagnostic_report(system, result.controls)
        gradient_check = opt.util.verify_gradient(
            system,
            result.controls,
            eps=1.0e-6,
            directions=2,
            seed=5,
        )

        self.assertEqual(result.optimizer, "adam")
        self.assertLess(result.J, initial)
        self.assertIn("metrics", diagnostic)
        self.assertIn("max_relative_error", gradient_check)

    def test_direct_shortcuts_remain_available_for_compact_notebook_use(self):
        system = TemporaryUniversalFourthOrderSystem(N=9)
        controls = opt.zero_guess(system)

        direct = opt.adagrad(system, controls, step_size=0.03, maxiter=1)
        namespaced = opt.optimizers.adagrad(system, controls, step_size=0.03, maxiter=1)

        self.assertEqual(direct.optimizer, namespaced.optimizer)
        self.assertEqual(direct.controls.shape, namespaced.controls.shape)


if __name__ == "__main__":
    unittest.main()
