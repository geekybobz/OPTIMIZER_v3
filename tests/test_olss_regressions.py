"""Regression tests for the 2026-10-12 OLSS review fixes.

Each test class covers one fix: evaluation failures during shooting runs,
system-declared verification tolerance, JSON-safe results, numerical Jacobian
checks, and selection ranking on verified metrics.
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Mapping

import numpy as np

import optimizer as opt
from optimizer import OLSS, ShootingBlock, ShootingSpec, basic_shooting_system_spec


def _variables(system: Any, values: list[float]) -> opt.ShootingVariables:
    return opt.ShootingVariables.from_flat(system.shooting_spec(), values)


class CubicDomain(OLSS):
    """r(z) = z^3 - c, defined only for |z| < 3 and c - z^3 <= jump_limit.

    Outside that region the residual is non-finite, which stands in for an
    integration that blows up for a bad trial or a too-large continuation step.
    """

    def __init__(
        self,
        primary: Mapping[str, Any] | None = None,
        secondary: Mapping[str, Any] | None = None,
    ) -> None:
        self.primary = {"jacobian_limit": None, "jump_limit": None, **dict(primary or {})}
        self.secondary = {"c": 8.0, **dict(secondary or {})}
        self._spec = ShootingSpec((ShootingBlock("z", shape=(1,)),))

    def system_spec(self) -> dict[str, Any]:
        return basic_shooting_system_spec(
            "cubic-domain",
            shooting_spec=self._spec,
            residual_size=1,
            parameters={"primary": dict(self.primary), "secondary": dict(self.secondary)},
            verification_options={"accuracy": "verification"},
        )

    def shooting_spec(self) -> ShootingSpec:
        return self._spec

    def residuals(self, variables, *, options=None) -> np.ndarray:
        z = float(variables.flatten()[0])
        c = float(self.secondary["c"])
        jump_limit = self.primary["jump_limit"]
        if abs(z) >= 3.0 or (jump_limit is not None and c - z**3 > jump_limit):
            return np.array([np.nan])
        return np.array([z**3 - c])

    def jacobian(self, variables, *, options=None) -> np.ndarray:
        z = float(variables.flatten()[0])
        limit = self.primary["jacobian_limit"]
        if limit is not None and z > limit:
            raise FloatingPointError("variational equations diverged")
        return np.array([[3.0 * z**2]])

    def with_secondary(self, **updates: Any) -> "CubicDomain":
        return CubicDomain(self.primary, {**self.secondary, **updates})


class BandedLine(OLSS):
    """One residual x - p on two unknowns; evaluation fails inside a band of x."""

    def __init__(self, primary=None, secondary=None) -> None:
        self.primary = dict(primary or {})
        self.secondary = dict(secondary or {})
        self._spec = ShootingSpec(
            (
                ShootingBlock("x"),
                ShootingBlock("p", role="continuation_parameter"),
            )
        )

    def system_spec(self) -> dict[str, Any]:
        return basic_shooting_system_spec(
            "banded-line",
            shooting_spec=self._spec,
            residual_size=1,
            verification_options={"accuracy": "verification"},
        )

    def shooting_spec(self) -> ShootingSpec:
        return self._spec

    def residuals(self, variables, *, options=None) -> np.ndarray:
        x, p = variables.flatten()
        if 0.45 < x < 0.55:
            raise FloatingPointError("integration failed inside the band")
        return np.array([x - p])

    def jacobian(self, variables, *, options=None) -> np.ndarray:
        return np.array([[1.0, -1.0]])

    def with_secondary(self, **updates: Any) -> "BandedLine":
        return BandedLine(self.primary, {**self.secondary, **updates})


class EulerDecay(OLSS):
    """Find y(0) so that explicit Euler for y' = -y reaches the target at t = 1.

    Working evaluations use ``options['steps']`` (default 100); verification uses
    1000 steps, so the verified residual contains the working discretization
    error (about 4.6e-3), far above a 1e-10 solver tolerance.
    """

    def __init__(self, primary=None, secondary=None) -> None:
        self.primary = {"verification_tolerance": None, **dict(primary or {})}
        self.secondary = {"target": 1.0, **dict(secondary or {})}
        self._spec = ShootingSpec((ShootingBlock("y0", shape=(1,)),))

    def system_spec(self) -> dict[str, Any]:
        return basic_shooting_system_spec(
            "euler-decay",
            shooting_spec=self._spec,
            residual_size=1,
            parameters={"primary": dict(self.primary), "secondary": dict(self.secondary)},
            verification_options={"steps": 1000},
            verification_tolerance=self.primary["verification_tolerance"],
        )

    def shooting_spec(self) -> ShootingSpec:
        return self._spec

    @staticmethod
    def _decay(options) -> float:
        steps = int(dict(options or {}).get("steps", 100))
        return (1.0 - 1.0 / steps) ** steps

    def residuals(self, variables, *, options=None) -> np.ndarray:
        y0 = float(variables.flatten()[0])
        return np.array([y0 * self._decay(options) - float(self.secondary["target"])])

    def jacobian(self, variables, *, options=None) -> np.ndarray:
        return np.array([[self._decay(options)]])

    def with_secondary(self, **updates: Any) -> "EulerDecay":
        return EulerDecay(self.primary, {**self.secondary, **updates})


class FencedCircle(OLSS):
    """Unit circle x^2 + y^2 = 1, minimize y; evaluation fails below y = -0.5.

    Direct shooting walks the circle downhill until its tangent steps enter the
    failing region; those trials must be rejected rather than raise.
    """

    def __init__(self, primary=None, secondary=None) -> None:
        self.primary = dict(primary or {})
        self.secondary = dict(secondary or {})
        self._spec = ShootingSpec((ShootingBlock("x"), ShootingBlock("y")))

    def system_spec(self) -> dict[str, Any]:
        return basic_shooting_system_spec(
            "fenced-circle",
            shooting_spec=self._spec,
            residual_size=1,
            metrics={"height": {"label": "Height", "required": True}},
            selection_metric="height",
            verification_options={"accuracy": "verification"},
        )

    def shooting_spec(self) -> ShootingSpec:
        return self._spec

    def residuals(self, variables, *, options=None) -> np.ndarray:
        x, y = variables.flatten()
        if y < -0.5:
            raise FloatingPointError("propagation failed below the fence")
        return np.array([x * x + y * y - 1.0])

    def jacobian(self, variables, *, options=None) -> np.ndarray:
        x, y = variables.flatten()
        return np.array([[2.0 * x, 2.0 * y]])

    def evaluate(self, variables, *, options=None):
        residual = self.residuals(variables, options=options)
        return {**self._residual_metrics(residual), "height": float(variables.flatten()[1])}

    def verify(self, variables, *, options=None):
        residual = self.residuals(variables, options=options)
        return opt.ShootingVerification(
            residuals=residual,
            metrics=self.evaluate(variables, options=options),
            options=dict(options or {}),
        )

    def cost_gradient(self, variables, *, options=None) -> np.ndarray:
        return np.array([0.0, 1.0])

    def with_secondary(self, **updates: Any) -> "FencedCircle":
        return FencedCircle(self.primary, {**self.secondary, **updates})


class FailedEvaluationTests(unittest.TestCase):
    """Fix 1: evaluation failures after the start are evidence, not exceptions."""

    def test_direct_shooting_rejects_trials_that_cannot_be_evaluated(self):
        system = FencedCircle()
        result = opt.solve_direct_shooting(
            system, _variables(system, [1.0, 0.0]), step_size=0.4, maxiter=40
        )

        rejected = [row for row in result.history if row.get("failed_trial_evaluations")]
        self.assertTrue(result.qualified, result.to_dict())
        self.assertTrue(rejected, "no trial entered the failing region")
        self.assertGreaterEqual(float(result.variables.block("y")), -0.5)
        self.assertLess(float(result.variables.block("y")), -0.3)

    def test_failed_trial_is_rejected_and_the_step_shrinks(self):
        system = CubicDomain()
        result = opt.solve_shooting(system, _variables(system, [0.5]), method="newton")

        self.assertTrue(result.qualified, result.to_dict())
        self.assertEqual(result.stop_reason, "root_tolerance")
        self.assertAlmostEqual(float(result.variables.block("z")[0]), 2.0, places=10)
        self.assertEqual(result.history[0]["failed_trial_evaluations"], 3)
        self.assertIn("trial_error", result.history[0])

    def test_a_start_that_cannot_be_evaluated_raises_clearly(self):
        system = CubicDomain()
        with self.assertRaisesRegex(ValueError, "initial shooting variables failed evaluation"):
            opt.solve_shooting(system, _variables(system, [5.0]))

    def test_every_trial_failing_stops_as_nonfinite_trial(self):
        system = CubicDomain()
        result = opt.solve_shooting(
            system, _variables(system, [0.5]), method="newton", line_search=False
        )

        self.assertEqual(result.stop_reason, "nonfinite_trial")
        self.assertFalse(result.qualified)
        self.assertIn("finite", result.error)
        self.assertAlmostEqual(float(result.variables.block("z")[0]), 0.5)

    def test_jacobian_failure_after_a_step_is_a_clean_stop(self):
        system = CubicDomain(primary={"jacobian_limit": 1.5})
        result = opt.solve_shooting(system, _variables(system, [0.5]), method="newton")

        z = float(result.variables.block("z")[0])
        self.assertEqual(result.stop_reason, "jacobian_failed")
        self.assertIn("FloatingPointError", result.error)
        self.assertGreater(z, 1.5)
        self.assertAlmostEqual(result.residual_norm, abs(z**3 - 8.0))

    def test_multistart_records_a_failed_seed_and_keeps_good_results(self):
        system = CubicDomain()
        seeds = [_variables(system, [value]) for value in (5.0, 0.5, 1.5)]
        campaign = opt.solve_shooting_multistart(
            system, seeds, solver_options={"method": "newton"}
        )

        failed = campaign.results[0]
        self.assertEqual(failed.stop_reason, "initial_evaluation_failed")
        self.assertFalse(failed.qualified)
        self.assertIn("initial shooting variables failed evaluation", failed.error)
        self.assertTrue(np.all(np.isnan(failed.residuals)))
        self.assertEqual(len(campaign.qualified), 2)
        self.assertEqual(len(campaign.unique_qualified), 1)
        json.dumps(campaign.to_dict())

    def test_parameter_continuation_shrinks_past_a_failing_stage(self):
        system = CubicDomain(primary={"jump_limit": 10.0})
        start = opt.solve_shooting(system, _variables(system, [1.9]), method="newton")
        branch = opt.continue_parameter(
            system,
            start.variables,
            parameter="c",
            start=8.0,
            target=23.0,
            step_size=15.0,
            solver_options={"method": "newton"},
        )

        self.assertTrue(branch.reached_target, branch.to_dict())
        self.assertAlmostEqual(
            float(branch.final.variables.block("z")[0]), 23.0 ** (1.0 / 3.0), places=9
        )

    def test_pseudo_arclength_corrector_failure_shrinks_instead_of_raising(self):
        system = BandedLine()
        previous = _variables(system, [0.0, 0.0])
        current = _variables(system, [0.1, 0.1])
        branch = opt.continue_shooting(
            system, previous, current, steps=10, step_size=0.2, min_step=1.0e-3
        )

        self.assertEqual(branch.stop_reason, "minimum_step_reached", branch.to_dict())
        self.assertTrue(any("error" in row for row in branch.step_history))
        json.dumps(branch.to_dict())


class VerificationToleranceTests(unittest.TestCase):
    """Fix 2: a system can declare its verification tolerance; results say which one applied."""

    def test_without_a_declaration_the_working_tolerance_applies_and_is_recorded(self):
        system = EulerDecay()
        result = opt.solve_shooting(system, _variables(system, [1.0]))

        self.assertTrue(result.solver_converged)
        self.assertFalse(result.qualified)
        self.assertEqual(result.stop_reason, "verification_failed")
        self.assertEqual(result.verification_tolerance, 1.0e-10)
        self.assertEqual(result.verification_tolerance_source, "working_tolerance")
        self.assertGreater(result.verification.residual_norm, 1.0e-3)

    def test_a_declared_tolerance_qualifies_the_root(self):
        system = EulerDecay(primary={"verification_tolerance": 1.0e-2})
        result = opt.solve_shooting(system, _variables(system, [1.0]))

        self.assertTrue(result.qualified, result.to_dict())
        self.assertEqual(result.verification_tolerance, 1.0e-2)
        self.assertEqual(result.verification_tolerance_source, "system_default")
        self.assertEqual(system.system_spec()["verification"]["default_tolerance"], 1.0e-2)

    def test_an_explicit_tolerance_overrides_the_declaration(self):
        system = EulerDecay(primary={"verification_tolerance": 1.0e-2})
        result = opt.solve_shooting(system, _variables(system, [1.0]), verification_tolerance=1.0e-4)

        self.assertFalse(result.qualified)
        self.assertEqual(result.verification_tolerance_source, "explicit")

    def test_the_specification_rejects_an_invalid_declared_tolerance(self):
        spec = ShootingSpec((ShootingBlock("z"),))
        for bad in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                basic_shooting_system_spec("bad", shooting_spec=spec, residual_size=1, verification_tolerance=bad)

    def test_parameter_continuation_reports_why_each_attempt_failed(self):
        system = EulerDecay()
        start = opt.solve_shooting(system, _variables(system, [1.0]))
        branch = opt.continue_parameter(
            system, start.variables, parameter="target", start=1.0, target=2.0, step_size=0.5
        )

        self.assertEqual(branch.stop_reason, "minimum_step_reached")
        self.assertTrue(branch.step_history)
        last = branch.step_history[-1]
        self.assertEqual(last["stop_reason"], "verification_failed")
        self.assertTrue(last["solver_converged"])
        self.assertGreater(last["verification_residual_norm"], last["verification_tolerance"])

        declared = EulerDecay(primary={"verification_tolerance": 1.0e-2})
        landed = opt.continue_parameter(
            declared, start.variables, parameter="target", start=1.0, target=2.0, step_size=0.5
        )
        self.assertTrue(landed.reached_target)
        self.assertEqual(len(landed.step_history), len(landed.points))

    def test_pseudo_arclength_records_the_tolerance_it_used(self):
        system = BandedLine()
        branch = opt.continue_shooting(
            system, _variables(system, [0.0, 0.0]), _variables(system, [0.1, 0.1]), steps=1, step_size=0.1
        )

        self.assertEqual(branch.verification_tolerance_source, "working_tolerance")
        self.assertEqual(branch.verification_tolerance, 1.0e-10)


class NoisyFiniteDifferenceSystem(OLSS):
    """Numerical-declared Jacobian by central differences with the project's own step.

    The residual carries a tiny step-scale ripple, like adaptive-integrator noise.
    At the project's step the finite-difference Jacobian is dominated by that
    ripple, so it is wrong; only a comparison at a different step can see it.
    """

    def __init__(self, primary=None, secondary=None) -> None:
        self.primary = {"step": 1.0e-5, "ripple": 1.0e-7, **dict(primary or {})}
        self.secondary = dict(secondary or {})
        self._spec = ShootingSpec((ShootingBlock("z"),))

    def system_spec(self) -> dict[str, Any]:
        return basic_shooting_system_spec(
            "noisy-fd",
            shooting_spec=self._spec,
            residual_size=1,
            jacobian_source="numerical-declared",
            verification_options={"accuracy": "verification"},
        )

    def shooting_spec(self) -> ShootingSpec:
        return self._spec

    def residuals(self, variables, *, options=None) -> np.ndarray:
        z = float(variables.flatten()[0])
        return np.array([np.sin(3.0 * z) + self.primary["ripple"] * np.sin(1.0e7 * z)])

    def jacobian(self, variables, *, options=None) -> np.ndarray:
        z = float(variables.flatten()[0])
        h = float(self.primary["step"])
        plus = self.residuals(variables.with_flat([z + h]))[0]
        minus = self.residuals(variables.with_flat([z - h]))[0]
        return np.array([[(plus - minus) / (2.0 * h)]])

    def with_secondary(self, **updates: Any) -> "NoisyFiniteDifferenceSystem":
        return NoisyFiniteDifferenceSystem(self.primary, {**self.secondary, **updates})


class NumericalJacobianCheckTests(unittest.TestCase):
    """Fix 4: a numerical-declared Jacobian is never checked against its own stencil only."""

    def test_a_coinciding_step_cannot_produce_a_hollow_pass(self):
        system = NoisyFiniteDifferenceSystem()  # project step 1e-5 == 10 * default eps
        check = opt.verify_shooting_jacobian(system, _variables(system, [0.4]))

        self.assertAlmostEqual(check["comparison_steps"][0], 1.0e-5, delta=1.0e-15)
        self.assertEqual(len(check["step_max_abs_errors"]), 2)
        self.assertLess(check["step_max_abs_errors"][0], 1.0e-12)  # identical stencil
        self.assertGreater(check["step_max_abs_errors"][1], 1.0e-5)  # the ripple is exposed
        self.assertFalse(check["passed"])

    def test_a_clean_numerical_jacobian_still_passes(self):
        system = NoisyFiniteDifferenceSystem(primary={"ripple": 0.0})
        check = opt.verify_shooting_jacobian(system, _variables(system, [0.4]))

        self.assertTrue(check["passed"], check)
        self.assertGreater(check["max_abs_error"], 0.0)

    def test_declared_analytical_jacobians_use_one_comparison(self):
        system = CubicDomain()
        check = opt.verify_shooting_jacobian(system, _variables(system, [1.2]))

        self.assertTrue(check["passed"])
        self.assertEqual(len(check["comparison_steps"]), 1)
        self.assertEqual(check["comparison_strategy"], "central_difference")


class TwoRootCost(OLSS):
    """Roots z = -1 and z = +1; the working cost estimate misorders them.

    Verified cost (z - 0.1)^2 prefers +1.  The working estimate adds 0.3 z and
    prefers -1, standing in for a coarse discretization biasing a cost metric.
    """

    def __init__(self, primary=None, secondary=None) -> None:
        self.primary = dict(primary or {})
        self.secondary = dict(secondary or {})
        self._spec = ShootingSpec((ShootingBlock("z"),))

    def system_spec(self) -> dict[str, Any]:
        return basic_shooting_system_spec(
            "two-root-cost",
            shooting_spec=self._spec,
            residual_size=1,
            metrics={"cost": {"label": "Cost", "required": True}},
            selection_metric="cost",
            verification_options={"accuracy": "verification"},
        )

    def shooting_spec(self) -> ShootingSpec:
        return self._spec

    def residuals(self, variables, *, options=None) -> np.ndarray:
        z = float(variables.flatten()[0])
        return np.array([z * z - 1.0])

    def jacobian(self, variables, *, options=None) -> np.ndarray:
        return np.array([[2.0 * float(variables.flatten()[0])]])

    def evaluate(self, variables, *, options=None):
        z = float(variables.flatten()[0])
        residual = self.residuals(variables)
        verified = dict(options or {}).get("accuracy") == "verification"
        return {
            **self._residual_metrics(residual),
            "cost": (z - 0.1) ** 2 + (0.0 if verified else 0.3 * z),
        }

    def verify(self, variables, *, options=None):
        return opt.ShootingVerification(
            residuals=self.residuals(variables),
            metrics=self.evaluate(variables, options=options),
            options=dict(options or {}),
        )

    def with_secondary(self, **updates: Any) -> "TwoRootCost":
        return TwoRootCost(self.primary, {**self.secondary, **updates})


class VerifiedRankingTests(unittest.TestCase):
    """Fix 5: qualified roots are ranked by the verified selection metric."""

    def test_ranking_uses_verification_metrics(self):
        system = TwoRootCost()
        seeds = [_variables(system, [value]) for value in (-1.3, 1.3)]
        campaign = opt.solve_shooting_multistart(system, seeds)

        self.assertEqual(len(campaign.unique_qualified), 2)
        self.assertEqual(campaign.selection_source, "verification")
        self.assertAlmostEqual(float(campaign.best.variables.block("z")), 1.0)
        working = sorted(campaign.unique_qualified, key=lambda result: result.metrics["cost"])
        self.assertAlmostEqual(float(working[0].variables.block("z")), -1.0)


class StrictJsonTests(unittest.TestCase):
    """Fix 3: every OLSS result serializes as strict JSON (no NumPy types)."""

    def assertStrictJson(self, payload: Any) -> None:
        json.dumps(payload, allow_nan=False)

    def test_every_result_type_is_strict_json(self):
        system = EulerDecay(primary={"verification_tolerance": 1.0e-2})
        start = opt.solve_shooting(system, _variables(system, [1.0]))
        self.assertStrictJson(start.to_dict())
        self.assertStrictJson(opt.ShootingEvaluator(system).evaluate(start.variables).to_dict())

        campaign = opt.solve_shooting_multistart(
            system, [_variables(system, [value]) for value in (0.5, 3.0)]
        )
        self.assertStrictJson(campaign.to_dict())

        branch = opt.continue_parameter(
            system,
            start.variables,
            parameter="target",
            start=np.float64(1.0),
            target=np.float64(1.5),
            step_size=np.float64(0.25),
        )
        self.assertTrue(branch.reached_target)
        self.assertIs(type(branch.reached_target), bool)
        self.assertStrictJson(branch.to_dict())

        line = BandedLine()
        arclength = opt.continue_shooting(
            line, _variables(line, [0.0, 0.0]), _variables(line, [0.1, 0.1]), steps=2, step_size=0.1
        )
        self.assertStrictJson(arclength.to_dict())

    def test_numpy_flags_from_a_project_verify_are_converted(self):
        verification = opt.ShootingVerification(
            residuals=np.zeros(1),
            metrics={"residual_norm": np.float64(0.0), "residual_max_abs": np.float64(0.0)},
            numerical_success=np.bool_(True),
            checks=(opt.VerificationCheck("energy", passed=np.bool_(True), value=np.float32(1.5)),),
        )
        self.assertStrictJson(verification.to_dict())


if __name__ == "__main__":
    unittest.main()
