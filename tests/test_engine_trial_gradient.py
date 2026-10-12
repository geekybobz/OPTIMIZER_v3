"""Standard-acceptance methods evaluate trials together with their gradient."""

from __future__ import annotations

import unittest

import numpy as np

import optimizer as opt
from fixtures.quadratic_system import QuadraticVectorSystem


class CountingQuadratic(QuadraticVectorSystem):
    """Counts forward propagations; the gradient recomputes the forward pass."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.forward_passes = 0

    def evaluate(self, controls):
        self.forward_passes += 1
        return super().evaluate(controls)

    def gradient(self, controls):
        self.forward_passes += 1
        return super().gradient(controls)


class CombinedCountingQuadratic(CountingQuadratic):
    def value_and_gradient(self, controls):
        self.forward_passes += 1
        return QuadraticVectorSystem.evaluate(self, controls), QuadraticVectorSystem.gradient(self, controls)


class TrialGradientTests(unittest.TestCase):
    def test_combined_hook_halves_forward_passes_with_identical_iterates(self):
        separate = CountingQuadratic()
        combined = CombinedCountingQuadratic()
        start = separate.reference_controls(amplitude=0.7)

        plain = opt.adam(separate, start, step_size=0.05, maxiter=20)
        fused = opt.adam(combined, start, step_size=0.05, maxiter=20)

        np.testing.assert_array_equal(plain.controls.as_matrix(), fused.controls.as_matrix())
        self.assertEqual(separate.forward_passes, 41)  # start + (gradient, trial) per iteration
        self.assertEqual(combined.forward_passes, 21)  # start + one combined call per iteration

    def test_monotone_acceptance_keeps_separate_trial_evaluation(self):
        system = CombinedCountingQuadratic()
        start = system.reference_controls(amplitude=0.7)
        result = opt.line_search(system, start, maxiter=5)

        self.assertEqual(result.iterations, 5)
        technical = result.trace.iteration_records[-1].technical["counters"]
        self.assertGreaterEqual(technical["evaluation"], technical["value_gradient"])


class BoundedCacheTests(unittest.TestCase):
    def test_evaluator_caches_keep_only_recent_entries(self):
        from optimizer.core.evaluate import SystemEvaluator
        from optimizer.shooting import ShootingEvaluator
        from fixtures.shooting_systems import RectangularShootingSystem

        olgs = SystemEvaluator(QuadraticVectorSystem(), cache_size=4)
        base = olgs.system.reference_controls(amplitude=0.1)
        for index in range(10):
            olgs.evaluate(base * (1.0 + index))
        self.assertEqual(len(olgs._evaluations), 4)
        before = olgs.evaluation_count
        olgs.evaluate(base * 10.0)  # most recent point: served from the cache
        self.assertEqual(olgs.evaluation_count, before)

        system = RectangularShootingSystem()
        olss = ShootingEvaluator(system, cache_size=4)
        for index in range(10):
            olss.residuals_and_jacobian(opt.ShootingVariables.from_flat(system.shooting_spec(), [index, 1.0]))
        self.assertEqual(len(olss._residuals), 4)
        self.assertEqual(len(olss._jacobians), 4)

        unbounded = SystemEvaluator(QuadraticVectorSystem(), cache_size=None)
        for index in range(50):
            unbounded.evaluate(base * (1.0 + index))
        self.assertEqual(len(unbounded._evaluations), 50)


if __name__ == "__main__":
    unittest.main()
