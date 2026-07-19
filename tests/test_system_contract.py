import unittest
from dataclasses import dataclass, replace

import numpy as np

from optimizer.controls import ControlSpec, Controls
from optimizer.system_olgs import (
    evaluate_system,
    gradient_system,
    optional_jacobian,
    optional_residuals,
    probe_system,
    require_system,
    validate_controls_for_system,
)


@dataclass(frozen=True)
class DummyParams:
    control_dim: int = 4
    weight: float = 1.0


class DummySystem:
    def __init__(self, params=None):
        self.params = params or DummyParams()

    def control_spec(self):
        return ControlSpec(keys=("u",), control_dim=self.params.control_dim, dt=0.1)

    def evaluate(self, controls):
        u = controls.channel("u")
        return {"J": self.params.weight * float(np.sum(u * u)), "energy": float(np.sum(u * u))}

    def gradient(self, controls):
        u = controls.channel("u")
        return Controls.from_dict(self.control_spec(), {"u": 2.0 * self.params.weight * u})

    def with_secondary(self, **updates):
        return DummySystem(replace(self.params, **updates))

    def residuals(self, controls, name="hard"):
        return np.array([np.sum(controls.channel("u")) - 1.0])

    def jacobian(self, controls, name="hard"):
        return np.ones((1, self.control_spec().size))


class MissingGradientSystem:
    def control_spec(self):
        return ControlSpec(keys=("u",), control_dim=2)

    def evaluate(self, controls):
        return {"J": 0.0}

    def with_secondary(self, **updates):
        return self


class BadMetricsSystem(DummySystem):
    def evaluate(self, controls):
        return {"energy": 1.0}


class BadGradientSystem(DummySystem):
    def gradient(self, controls):
        spec = ControlSpec(keys=("u", "v"), control_dim=self.params.control_dim)
        return Controls.zeros(spec)


class SystemContractTests(unittest.TestCase):
    def test_probe_and_require_system(self):
        system = DummySystem()
        probe = probe_system(system)

        self.assertTrue(probe.required_ok)
        self.assertTrue(probe.has_residuals)
        self.assertTrue(probe.has_jacobian)
        self.assertIs(require_system(system), system)

        with self.assertRaises(TypeError):
            require_system(MissingGradientSystem())

    def test_evaluate_and_gradient(self):
        system = DummySystem()
        controls = Controls.from_dict(system.control_spec(), {"u": [1.0, 2.0, 3.0, 4.0]})

        metrics = evaluate_system(system, controls)
        gradient = gradient_system(system, controls)

        self.assertEqual(metrics["J"], 30.0)
        np.testing.assert_allclose(gradient.channel("u"), [2.0, 4.0, 6.0, 8.0])

    def test_with_secondary_changes_system_weights(self):
        system = DummySystem().with_secondary(weight=3.0)
        controls = Controls.from_dict(system.control_spec(), {"u": [1.0, 1.0, 1.0, 1.0]})

        metrics = evaluate_system(system, controls)
        gradient = gradient_system(system, controls)

        self.assertEqual(metrics["J"], 12.0)
        np.testing.assert_allclose(gradient.channel("u"), [6.0, 6.0, 6.0, 6.0])

    def test_controls_must_match_system_layout(self):
        system = DummySystem()
        wrong = Controls.zeros(ControlSpec(keys=("u", "v"), control_dim=4))

        with self.assertRaises(ValueError):
            validate_controls_for_system(system, wrong)

    def test_bad_metrics_and_bad_gradient_are_rejected(self):
        controls = Controls.zeros(DummySystem().control_spec())

        with self.assertRaises(KeyError):
            evaluate_system(BadMetricsSystem(), controls)
        with self.assertRaises(ValueError):
            gradient_system(BadGradientSystem(), controls)

    def test_optional_residuals_and_jacobian(self):
        system = DummySystem()
        controls = Controls.from_dict(system.control_spec(), {"u": [0.25, 0.25, 0.25, 0.25]})

        residuals = optional_residuals(system, controls)
        jacobian = optional_jacobian(system, controls)

        np.testing.assert_allclose(residuals, [0.0])
        self.assertEqual(jacobian.shape, (1, 4))


if __name__ == "__main__":
    unittest.main()
