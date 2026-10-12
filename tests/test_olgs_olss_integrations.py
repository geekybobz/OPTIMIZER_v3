"""Regression tests for the integrated OLGS/OLSS high- and medium-priority tools."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import optimizer as opt
from optimizer.core import default_accept
from optimizer.system_olgs.scaffold import scaffold_system
from test_contained.olgs_convergence.scenarios import BoxQuadratic, IsotropicQuadratic
from test_contained.olss_scenarios.systems import BoundedSystem


class StationaryMaximum(IsotropicQuadratic):
    def __init__(self):
        super().__init__(primary={"N": 2})

    def evaluate(self, controls):
        vector = controls.flatten(copy=False)
        return {"J": -0.5 * float(vector @ vector)}

    def gradient(self, controls):
        return opt.Controls.from_flat(self.control_spec(), -controls.flatten(copy=False))

    def hvp(self, controls, vector):
        del controls
        values = vector.flatten(copy=False) if isinstance(vector, opt.Controls) else np.asarray(vector)
        return opt.Controls.from_flat(self.control_spec(), -values)


class CheckedRoot(opt.OLSS):
    def __init__(self, primary=None, secondary=None):
        self.primary = dict(primary or {})
        self.secondary = dict(secondary or {})
        self._spec = opt.ShootingSpec((opt.ShootingBlock("x"),))

    def shooting_spec(self):
        return self._spec

    def system_spec(self):
        return opt.basic_shooting_system_spec(
            "checked-root",
            shooting_spec=self._spec,
            residual_size=1,
            parameters={"primary": self.primary, "secondary": self.secondary},
            verification_options={"profile": "independent"},
        )

    def residuals(self, variables, *, options=None):
        del options
        return np.array([float(variables.block("x")) - 1.0])

    def jacobian(self, variables, *, options=None):
        del variables, options
        return np.ones((1, 1))

    def verify(self, variables, *, options=None):
        residuals = self.residuals(variables, options=options)
        return opt.ShootingVerification(
            residuals=residuals,
            metrics=self._residual_metrics(residuals),
            options=dict(options or {}),
            checks=(
                opt.VerificationCheck(
                    "external_replay",
                    False,
                    value=2.0,
                    tolerance=1.0,
                    message="intentional regression fixture",
                ),
            ),
        )

    def recover_controls(self, variables, *, times):
        value = float(variables.block("x"))
        return opt.Controls.constant(opt.ControlSpec(("u",), len(times)), value)

    def with_secondary(self, **updates):
        return type(self)(self.primary, {**self.secondary, **updates})


class ReplayGradientSystem:
    def __init__(self, size):
        self._spec = opt.ControlSpec(("u",), size)

    def system_spec(self):
        return opt.basic_system_spec("replay-gradient", control_spec=self._spec)

    def control_spec(self):
        return self._spec

    def evaluate(self, controls):
        return {"J": float(np.mean(controls.flatten(copy=False) ** 2))}

    def gradient(self, controls):
        return opt.Controls.from_flat(self._spec, 2.0 * controls.flatten(copy=False) / self._spec.size)

    def with_secondary(self, **updates):
        if updates:
            raise KeyError(updates)
        return self


class LinearBranch(CheckedRoot):
    def __init__(self, primary=None, secondary=None):
        self.primary = dict(primary or {})
        self.secondary = dict(secondary or {})
        self._spec = opt.ShootingSpec(
            (
                opt.ShootingBlock("x"),
                opt.ShootingBlock("parameter", role="continuation_parameter"),
            )
        )

    def residuals(self, variables, *, options=None):
        del options
        return np.array([float(variables.block("x")) - float(variables.block("parameter"))])

    def jacobian(self, variables, *, options=None):
        del variables, options
        return np.array([[1.0, -1.0]])

    def verify(self, variables, *, options=None):
        residuals = self.residuals(variables, options=options)
        return opt.ShootingVerification(
            residuals=residuals,
            metrics=self._residual_metrics(residuals),
            options=dict(options or {}),
        )


class IntegratedContractTests(unittest.TestCase):
    def test_standard_first_order_acceptance_schedules_and_bounds(self):
        system = BoxQuadratic()
        unbounded = IsotropicQuadratic(primary={"N": 2})
        controls = opt.Controls.zeros(system.control_spec())
        schedule = opt.CosineDecaySchedule(initial_step=0.2, total_steps=4, min_step=0.05)

        result = opt.adam(system, controls, step_size=schedule, maxiter=4)

        lower, upper = system.control_bounds()
        self.assertTrue(np.all(result.controls.flatten() >= np.asarray(lower).reshape(-1)))
        self.assertTrue(np.all(result.controls.flatten() <= np.asarray(upper).reshape(-1)))
        self.assertEqual(result.diagnostics['rejected'], 0)
        self.assertLessEqual(result.state.step_size, schedule.initial())

        adaptive_schedule = opt.AdaptiveStepSchedule(initial_step=0.2, grow=0.5)
        first_chunk = opt.adam(
            unbounded,
            opt.Controls.zeros(unbounded.control_spec()),
            step_size=adaptive_schedule,
            maxiter=2,
        )
        continued_chunk = opt.adam(
            unbounded,
            state=first_chunk.state,
            step_size=adaptive_schedule,
            maxiter=1,
        )
        self.assertAlmostEqual(continued_chunk.state.step_size, 0.025)

        projected_start = opt.adam(
            system,
            opt.Controls.constant(system.control_spec(), 5.0),
            maxiter=0,
        )
        self.assertTrue(np.all(projected_start.controls.flatten() <= np.asarray(upper).reshape(-1)))

        guarded = opt.momentum(
            unbounded,
            opt.Controls.constant(unbounded.control_spec(), 1.0),
            step_size=100.0,
            maxiter=1,
            accept=default_accept,
        )
        self.assertEqual(guarded.diagnostics['accepted'], 0)

    def test_short_run_diagnostics_and_stationary_curvature(self):
        system = StationaryMaximum()
        controls = opt.Controls.zeros(system.control_spec())
        result = opt.newton_cg(system, controls, maxiter=2)
        curvature = opt.curvature_probe(system, controls)
        diagnosis = opt.diagnose_run(
            opt.adam(system, opt.Controls.constant(system.control_spec(), 0.2), maxiter=3)
        )

        self.assertEqual(result.stop_reason, "stationary_nonminimum")
        self.assertTrue(curvature["negative_curvature"])
        self.assertIn(diagnosis["classification"], {"converging", "slow", "stalled"})

    def test_compare_methods_is_in_memory_and_ranked(self):
        system = IsotropicQuadratic(primary={"N": 2})
        controls = opt.Controls.constant(system.control_spec(), 1.0)
        comparison = opt.compare_methods(
            system,
            controls,
            ("adam", "line_search"),
            iterations=3,
            options={"line_search": {"variant": "backtracking"}},
        )
        self.assertEqual(len(comparison["ranking"]), 2)
        values = [row["objective"] for row in comparison["ranking"]]
        self.assertEqual(values, sorted(values))

    def test_rejected_curvature_steps_shrink_and_newton_restores_full_step(self):
        system = IsotropicQuadratic(primary={"N": 2})
        controls = opt.Controls.constant(system.control_spec(), 1.0)

        for method in (opt.lbfgs, opt.nonlinear_cg):
            result = method(system, controls, step_size=10.0, maxiter=3)
            self.assertEqual(result.state.step_size, 1.25)
            self.assertEqual(result.diagnostics['rejected'], 3)
            self.assertTrue(np.all(np.isfinite(result.controls.flatten())))

        newton = opt.newton_cg(system, controls, step_size=10.0, maxiter=4)
        self.assertGreater(newton.diagnostics['accepted'], 0)
        self.assertEqual(newton.state.step_size, 10.0)

    def test_cma_es_supports_state_warmstart_and_standard_stopping(self):
        system = IsotropicQuadratic(primary={"N": 2})
        controls = opt.Controls.zeros(system.control_spec())
        first = opt.cma_es(system, controls, maxiter=1, population_size=4, seed=3)
        continued = opt.cma_es(system, state=first.state, maxiter=1, population_size=4, seed=4)
        warm = opt.cma_es(
            system,
            warmstart=continued.warmstart(target_optimizer="cma_es"),
            maxiter=0,
        )
        stopped = opt.cma_es(
            system,
            controls,
            maxiter=3,
            target_value=10.0,
            target_metric="J",
            target_mode="le",
        )
        structured = opt.cma_es(
            system,
            controls,
            maxiter=5,
            population_size=4,
            seed=5,
            stopping=opt.StoppingConfig(maxiter=1),
        )

        self.assertGreater(continued.state.global_iteration, first.state.global_iteration)
        self.assertTrue(np.allclose(warm.controls.flatten(), continued.controls.flatten()))
        self.assertEqual(stopped.stop_reason, "target")
        self.assertEqual(structured.iterations, 1)
        for method in (
            opt.optimizers.adam,
            opt.optimizers.momentum,
            opt.optimizers.lbfgs,
            opt.optimizers.nonlinear_cg,
            opt.optimizers.newton_cg,
        ):
            self.assertIn("gradient_tolerance", inspect.signature(method).parameters)

    def test_olss_named_checks_geometry_seeds_resume_and_dispatch(self):
        system = CheckedRoot()
        initial = opt.ShootingVariables.from_flat(system.shooting_spec(), [0.0])
        events = []
        result = opt.solve_shooting(
            system,
            initial,
            method="newton",
            tolerance=1.0e-12,
            on_event=events.append,
        )
        resumed = opt.solve_shooting(
            system,
            result.variables,
            method="newton",
            tolerance=1.0e-12,
            previous_result=result,
        )
        geometry = opt.shooting_geometry_probe(system, result.variables)
        seeds = opt.random_shooting_seeds(system.shooting_spec(), 4, seed=7)
        cloud = opt.perturb_shooting_seeds(result.variables, 3, seed=8)
        refined = opt.refine_shooting_seeds(
            system,
            seeds[:2],
            elite_count=1,
            perturbations_per_elite=2,
            seed=9,
            solver_options={"method": "newton", "maxiter": 5},
        )

        self.assertTrue(result.solver_converged)
        self.assertFalse(result.qualified)
        self.assertEqual(result.verification.checks[0].name, "external_replay")
        self.assertEqual([events[0]["event"], events[-1]["event"]], ["start", "terminal"])
        self.assertTrue(any(event["event"] == "iteration" for event in events))
        self.assertGreater(len(resumed.history), len(result.history))
        self.assertEqual(geometry["rank"], 1)
        self.assertEqual(len(seeds), 4)
        self.assertEqual(len(cloud), 3)
        self.assertEqual(len(refined.results), 3)
        self.assertEqual(opt.evaluate(system, result.variables).residual_norm, 0.0)
        with self.assertRaisesRegex(TypeError, "residuals and jacobian"):
            opt.gradient(system, result.variables)

    def test_parameter_continuation_lands_exactly_and_replay_is_explicit(self):
        system = BoundedSystem(secondary={"target": 0.2})
        initial = opt.ShootingVariables.from_flat(system.shooting_spec(), [0.2])
        branch = opt.continue_parameter(
            system,
            initial,
            parameter="target",
            start=0.2,
            target=0.85,
            step_size=0.3,
            max_step=0.4,
            solver_options={"method": "newton", "tolerance": 1.0e-10},
        )
        checked = CheckedRoot()
        root = opt.ShootingVariables.from_flat(checked.shooting_spec(), [1.0])
        replay = opt.compare_shooting_replay(
            checked,
            root,
            ReplayGradientSystem(5),
            times=np.linspace(0.0, 1.0, 5),
        )

        self.assertTrue(branch.reached_target, branch.to_dict())
        self.assertAlmostEqual(branch.points[-1].parameter, 0.85)
        self.assertEqual(replay["grid"]["count"], 5)
        self.assertAlmostEqual(replay["gradient_metrics"]["J"], 1.0)
        json.dumps(replay)

    def test_pseudo_arclength_can_stop_with_exact_target_landing(self):
        system = LinearBranch()
        previous = opt.ShootingVariables.from_flat(system.shooting_spec(), [0.0, 0.0])
        current = opt.ShootingVariables.from_flat(system.shooting_spec(), [0.1, 0.1])
        branch = opt.continue_shooting(
            system,
            previous,
            current,
            steps=5,
            step_size=0.2,
            target_block="parameter",
            target_value=0.35,
        )

        self.assertEqual(branch.stop_reason, "target_reached", branch.to_dict())
        self.assertAlmostEqual(branch.points[-1].variables.block("parameter"), 0.35)
        self.assertLess(branch.completed_steps, branch.requested_steps)

    def test_transactions_and_olss_scaffold(self):
        system = CheckedRoot()
        variables = opt.ShootingVariables.from_flat(system.shooting_spec(), [1.0])
        response = opt.transact(
            system,
            opt.SystemRequest("residuals", {"variables": variables}, family="olss"),
        )
        self.assertTrue(response.ok)
        self.assertEqual(response.to_dict()["result"], [0.0])
        updated = opt.transact(
            system,
            opt.SystemRequest("with_secondary", {"target": 2.0}, family="olss"),
        )
        json.dumps(updated.to_dict())
        self.assertEqual(updated.result.secondary["target"], 2.0)

        gradient_system = IsotropicQuadratic(primary={"N": 2})
        controls = opt.Controls.zeros(gradient_system.control_spec())
        gradient_response = opt.transact(
            gradient_system,
            opt.SystemRequest("gradient", {"controls": controls}, family="olgs"),
        )
        self.assertTrue(gradient_response.ok)
        json.dumps(gradient_response.to_dict())
        mismatch = opt.transact(
            gradient_system,
            opt.SystemRequest("gradient", {"controls": controls}, family="olss"),
        )
        self.assertFalse(mismatch.ok)
        self.assertEqual(mismatch.error["type"], "family_mismatch")

        with tempfile.TemporaryDirectory() as directory:
            created = scaffold_system(Path(directory), family="olss", class_name="ExampleSystem")
            self.assertEqual(len(created), 2)
            source = created[0].read_text(encoding="utf-8")
            self.assertIn("class ExampleSystem(OLSS)", source)
            self.assertIn("primary:", source)
            self.assertIn("secondary:", source)


if __name__ == "__main__":
    unittest.main()
