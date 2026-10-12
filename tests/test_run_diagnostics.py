"""Short-run diagnosis: relative thresholds, true start value, and explicit dispatch."""

from __future__ import annotations

import unittest

import optimizer as opt
from fixtures.quadratic_system import QuadraticVectorSystem
from fixtures.shooting_systems import RectangularShootingSystem
from optimizer.utils.run_diagnostics import _classify


# Recorded from Adam on the review's qubit problem: J fell from 0.0173 to 0.0045
# while swinging up and down.
OSCILLATING = [0.0173, 0.00444, 0.00704, 0.00825, 0.00745, 0.00467, 0.00231, 0.00242, 0.00438, 0.00543, 0.0045]
STEADY = [0.01714, 0.01687, 0.01649, 0.01601, 0.01547, 0.01486, 0.01421, 0.01353, 0.01283, 0.01213]


def _label(values: list[float]) -> str:
    return _classify(values, rejection_ratio=0.0, attempts=len(values) - 1)[0]


class ClassificationTests(unittest.TestCase):
    def test_labels_do_not_depend_on_the_objective_scale(self):
        cases = {
            "oscillating": OSCILLATING,
            "converging": STEADY,
            "slow": [1.0, 0.999, 0.998, 0.997, 0.996],
            "stalled": [2.0, 2.0, 2.0, 2.0],
            "diverging": [1.0, 1.1, 1.3, 1.6, 2.0],
        }
        for expected, values in cases.items():
            for scale in (1.0e-9, 1.0, 1.0e6):
                with self.subTest(expected=expected, scale=scale):
                    self.assertEqual(_label([value * scale for value in values]), expected)

    def test_oscillation_is_reported_before_a_net_rise(self):
        self.assertEqual(_label([1.0, 0.5, 1.2, 0.4, 1.3, 0.3, 1.4]), "oscillating")

    def test_repeated_rejections_take_precedence(self):
        label = _classify([1.0, 1.0, 1.0, 1.0], rejection_ratio=1.0, attempts=3)[0]
        self.assertEqual(label, "repeating_rejections")


class DiagnoseRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.system = QuadraticVectorSystem()
        self.controls = self.system.reference_controls(amplitude=0.5)

    def test_the_trend_starts_at_the_initial_objective(self):
        result = opt.momentum(self.system, self.controls, step_size=0.05, maxiter=8)
        diagnosis = opt.diagnose_run(result)

        start = opt.evaluate(self.system, self.controls).metrics["J"]
        self.assertAlmostEqual(diagnosis["objective_trend"]["start"], start)
        self.assertEqual(diagnosis["objective_trend"]["count"], diagnosis["iterations"] + 1)
        self.assertIn("thresholds", diagnosis)

    def test_a_run_without_samples_says_so(self):
        result = opt.adam(self.system, self.controls, maxiter=3, create_trace=False)
        diagnosis = opt.diagnose_run(result)

        self.assertEqual(diagnosis["family"], "olgs")
        self.assertEqual(diagnosis["classification"], "no_trace")

    def test_shooting_results_and_other_objects_dispatch_by_type(self):
        system = RectangularShootingSystem()
        root = opt.solve_shooting(
            system,
            opt.ShootingVariables.from_flat(system.shooting_spec(), [4.0, 3.0]),
            verification_options={"profile": "independent"},
        )
        diagnosis = opt.diagnose_run(root)

        self.assertEqual(diagnosis["family"], "olss")
        self.assertEqual(diagnosis["classification"], "converging")
        with self.assertRaises(TypeError):
            opt.diagnose_run({"history": []})


if __name__ == "__main__":
    unittest.main()
