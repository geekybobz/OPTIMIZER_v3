"""Small independent OLSS problems used by the shooting robustness suite."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import cos, pi, sin
from typing import Any, Mapping

import numpy as np

from optimizer.system_olss import (
    OLSS,
    ShootingBlock,
    ShootingSpec,
    ShootingVariables,
    ShootingVerification,
    basic_shooting_system_spec,
    shooting_metric_spec,
)


def _base_metrics(residuals: np.ndarray) -> dict[str, float]:
    return {
        "residual_norm": float(np.linalg.norm(residuals)),
        "residual_max_abs": float(np.max(np.abs(residuals))) if residuals.size else 0.0,
    }


class RectangularShootingSystem(OLSS):
    """Consistent three-residual, two-variable algebraic shooting problem."""

    def __init__(self) -> None:
        self._spec = ShootingSpec((ShootingBlock("x"), ShootingBlock("y")))
        self.combined_calls = 0

    def shooting_spec(self):
        return self._spec

    def system_spec(self):
        return basic_shooting_system_spec(
            "rectangular-shooting",
            shooting_spec=self._spec,
            residual_size=3,
            jacobian_source="analytical",
            capabilities={"combined_residual_jacobian": True},
        )

    def residuals(self, variables, *, options=None):
        del options
        x, y = variables.flatten(copy=False)
        return np.array([x - 1.0, y + 2.0, x + y + 1.0])

    def jacobian(self, variables, *, options=None):
        del variables, options
        return np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    def residuals_and_jacobian(self, variables, *, options=None):
        self.combined_calls += 1
        return self.residuals(variables, options=options), self.jacobian(variables, options=options)

    def with_secondary(self, **updates):
        if updates:
            raise KeyError(f"Unknown secondary parameters: {sorted(updates)}")
        return RectangularShootingSystem()


@dataclass(frozen=True)
class PendulumParameters:
    final_time: float = 1.2
    target_angle: float = 0.6
    default_steps: int = 800


class PendulumBoundaryShootingSystem(OLSS):
    """Nonlinear ODE boundary shooting with a variational Jacobian."""

    def __init__(self, params: PendulumParameters | None = None) -> None:
        self.params = params or PendulumParameters()
        self._spec = ShootingSpec(
            (ShootingBlock("initial_velocity", unit="rad/s", role="initial_costate"),)
        )

    def shooting_spec(self):
        return self._spec

    def system_spec(self):
        return basic_shooting_system_spec(
            "pendulum-boundary-shooting",
            shooting_spec=self._spec,
            residual_size=1,
            system_type="nonlinear-ode-boundary-value",
            jacobian_source="variational",
            parameters=dict(vars(self.params)),
            capabilities={"accuracy_override": True},
        )

    def _integrate(self, initial_velocity: float, steps: int) -> np.ndarray:
        dt = self.params.final_time / int(steps)
        state = np.array([0.0, initial_velocity, 0.0, 1.0], dtype=float)

        def rhs(value):
            theta, omega, sensitivity_theta, sensitivity_omega = value
            return np.array(
                [
                    omega,
                    -np.sin(theta),
                    sensitivity_omega,
                    -np.cos(theta) * sensitivity_theta,
                ]
            )

        for _ in range(int(steps)):
            k1 = rhs(state)
            k2 = rhs(state + 0.5 * dt * k1)
            k3 = rhs(state + 0.5 * dt * k2)
            k4 = rhs(state + dt * k3)
            state += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return state

    def _steps(self, options: Mapping[str, Any] | None) -> int:
        return int((options or {}).get("steps", self.params.default_steps))

    def residuals(self, variables, *, options=None):
        value = self._integrate(float(variables.block("initial_velocity")), self._steps(options))
        return np.array([value[0] - self.params.target_angle])

    def jacobian(self, variables, *, options=None):
        value = self._integrate(float(variables.block("initial_velocity")), self._steps(options))
        return np.array([[value[2]]])

    def residuals_and_jacobian(self, variables, *, options=None):
        value = self._integrate(float(variables.block("initial_velocity")), self._steps(options))
        return (
            np.array([value[0] - self.params.target_angle]),
            np.array([[value[2]]]),
        )

    def with_secondary(self, **updates):
        return PendulumBoundaryShootingSystem(replace(self.params, **updates))


class MultipleShootingHarmonicSystem(OLSS):
    """Two-segment harmonic oscillator multiple-shooting problem."""

    final_time = pi / 2.0
    target = 1.0

    def __init__(self) -> None:
        self._spec = ShootingSpec(
            (
                ShootingBlock("initial_velocity", role="initial_costate"),
                ShootingBlock("midpoint_position", role="segment_state"),
                ShootingBlock("midpoint_velocity", role="segment_state"),
            )
        )

    @staticmethod
    def _propagate(position: float, velocity: float, duration: float) -> tuple[float, float]:
        return (
            position * cos(duration) + velocity * sin(duration),
            -position * sin(duration) + velocity * cos(duration),
        )

    def shooting_spec(self):
        return self._spec

    def system_spec(self):
        return basic_shooting_system_spec(
            "harmonic-multiple-shooting",
            shooting_spec=self._spec,
            residual_size=3,
            system_type="multiple-shooting",
            jacobian_source="analytical",
        )

    def residuals(self, variables, *, options=None):
        del options
        v0, midpoint_position, midpoint_velocity = variables.flatten(copy=False)
        half = self.final_time / 2.0
        first_position, first_velocity = self._propagate(0.0, v0, half)
        terminal_position, _ = self._propagate(midpoint_position, midpoint_velocity, half)
        return np.array(
            [
                first_position - midpoint_position,
                first_velocity - midpoint_velocity,
                terminal_position - self.target,
            ]
        )

    def jacobian(self, variables, *, options=None):
        del variables, options
        half = self.final_time / 2.0
        return np.array(
            [
                [sin(half), -1.0, 0.0],
                [cos(half), 0.0, -1.0],
                [0.0, cos(half), sin(half)],
            ]
        )

    def with_secondary(self, **updates):
        if updates:
            raise KeyError(f"Unknown secondary parameters: {sorted(updates)}")
        return MultipleShootingHarmonicSystem()


class QubitTransferShootingSystem(OLSS):
    """Constant-y-control Bloch transfer from +z to +x at unit final time."""

    def __init__(self) -> None:
        self._spec = ShootingSpec(
            (ShootingBlock("control_amplitude", unit="rad/s", role="initial_control"),)
        )

    def shooting_spec(self):
        return self._spec

    def system_spec(self):
        return basic_shooting_system_spec(
            "qubit-state-transfer-shooting",
            shooting_spec=self._spec,
            residual_size=2,
            system_type="closed-quantum-state-transfer",
            residual_groups=(
                {
                    "name": "terminal_bloch",
                    "shape": [2],
                    "meaning": "Terminal x and z errors for +z to +x transfer",
                },
            ),
            jacobian_source="analytical",
        )

    def residuals(self, variables, *, options=None):
        del options
        amplitude = float(variables.block("control_amplitude"))
        return np.array([sin(amplitude) - 1.0, cos(amplitude)])

    def jacobian(self, variables, *, options=None):
        del options
        amplitude = float(variables.block("control_amplitude"))
        return np.array([[cos(amplitude)], [-sin(amplitude)]])

    def with_secondary(self, **updates):
        if updates:
            raise KeyError(f"Unknown secondary parameters: {sorted(updates)}")
        return QubitTransferShootingSystem()


class AccuracySensitiveShootingSystem(OLSS):
    """A controlled example where a coarse numerical root fails tight verification."""

    def __init__(self) -> None:
        self._spec = ShootingSpec((ShootingBlock("x"),))

    def shooting_spec(self):
        return self._spec

    def system_spec(self):
        return basic_shooting_system_spec(
            "accuracy-sensitive-shooting",
            shooting_spec=self._spec,
            residual_size=1,
            capabilities={"accuracy_override": True},
        )

    @staticmethod
    def _steps(options):
        return int((options or {}).get("steps", 10))

    def residuals(self, variables, *, options=None):
        numerical_target = 1.0 + 1.0 / self._steps(options)
        return np.array([float(variables.block("x")) - numerical_target])

    def jacobian(self, variables, *, options=None):
        del variables, options
        return np.ones((1, 1))

    def with_secondary(self, **updates):
        if updates:
            raise KeyError(f"Unknown secondary parameters: {sorted(updates)}")
        return AccuracySensitiveShootingSystem()


class CubicRootShootingSystem(OLSS):
    """Three isolated roots used for multistart and ranking tests."""

    def __init__(self) -> None:
        self._spec = ShootingSpec((ShootingBlock("x"),))

    def shooting_spec(self):
        return self._spec

    def system_spec(self):
        return basic_shooting_system_spec(
            "cubic-root-shooting",
            shooting_spec=self._spec,
            residual_size=1,
            metrics={"cost": shooting_metric_spec("Root cost", role="selection")},
            selection_metric="cost",
            verification_options={"profile": "independent"},
        )

    def residuals(self, variables, *, options=None):
        del options
        x = float(variables.block("x"))
        return np.array([x * (x - 1.0) * (x + 1.0)])

    def jacobian(self, variables, *, options=None):
        del options
        x = float(variables.block("x"))
        return np.array([[3.0 * x * x - 1.0]])

    def evaluate(self, variables, *, options=None):
        residuals = self.residuals(variables, options=options)
        x = float(variables.block("x"))
        return {**_base_metrics(residuals), "cost": x * x}

    def verify(self, variables, *, options=None):
        residuals = self.residuals(variables, options=options)
        return ShootingVerification(
            residuals=residuals,
            metrics=self.evaluate(variables, options=options),
            options=dict(options or {}),
        )

    def with_secondary(self, **updates):
        if updates:
            raise KeyError(f"Unknown secondary parameters: {sorted(updates)}")
        return CubicRootShootingSystem()


class FoldContinuationShootingSystem(OLSS):
    """The fold lambda=x^2 with lambda included in the shooting vector."""

    def __init__(self) -> None:
        self._spec = ShootingSpec(
            (
                ShootingBlock("x"),
                ShootingBlock("lambda", role="continuation_parameter"),
            )
        )

    def shooting_spec(self):
        return self._spec

    def system_spec(self):
        return basic_shooting_system_spec(
            "fold-continuation-shooting",
            shooting_spec=self._spec,
            residual_size=1,
            capabilities={"pseudo_arclength": True},
        )

    def residuals(self, variables, *, options=None):
        del options
        x, parameter = variables.flatten(copy=False)
        return np.array([parameter - x * x])

    def jacobian(self, variables, *, options=None):
        del options
        x = float(variables.block("x"))
        return np.array([[-2.0 * x, 1.0]])

    def with_secondary(self, **updates):
        if updates:
            raise KeyError(f"Unknown secondary parameters: {sorted(updates)}")
        return FoldContinuationShootingSystem()


class RedundantFoldContinuationSystem(OLSS):
    """Fold curve with one redundant residual row and Jacobian rank one."""

    def __init__(self) -> None:
        self._spec = ShootingSpec(
            (
                ShootingBlock("x"),
                ShootingBlock("lambda", role="continuation_parameter"),
            )
        )

    def shooting_spec(self):
        return self._spec

    def system_spec(self):
        return basic_shooting_system_spec(
            "redundant-fold-continuation",
            shooting_spec=self._spec,
            residual_size=2,
            capabilities={"pseudo_arclength": True},
            verification_options={"profile": "independent"},
        )

    def residuals(self, variables, *, options=None):
        del options
        x, parameter = variables.flatten(copy=False)
        value = parameter - x * x
        return np.array([value, 2.0 * value])

    def jacobian(self, variables, *, options=None):
        del options
        x = float(variables.block("x"))
        return np.array([[-2.0 * x, 1.0], [-4.0 * x, 2.0]])

    def with_secondary(self, **updates):
        if updates:
            raise KeyError(f"Unknown secondary parameters: {sorted(updates)}")
        return RedundantFoldContinuationSystem()


class SymmetryEquivalentCircleSystem(OLSS):
    """Unit-circle roots identified by a system-owned rotational symmetry."""

    def __init__(self) -> None:
        self._spec = ShootingSpec((ShootingBlock("x"), ShootingBlock("y")))

    def shooting_spec(self):
        return self._spec

    def system_spec(self):
        return basic_shooting_system_spec(
            "symmetry-equivalent-circle",
            shooting_spec=self._spec,
            residual_size=1,
            verification_options={"profile": "independent"},
        )

    def residuals(self, variables, *, options=None):
        del options
        x, y = variables.flatten(copy=False)
        return np.array([x * x + y * y - 1.0])

    def jacobian(self, variables, *, options=None):
        del options
        x, y = variables.flatten(copy=False)
        return np.array([[2.0 * x, 2.0 * y]])

    def roots_equivalent(self, left, right, *, tolerance):
        left_radius = float(np.linalg.norm(left.flatten(copy=False)))
        right_radius = float(np.linalg.norm(right.flatten(copy=False)))
        return abs(left_radius - right_radius) <= float(tolerance)

    def with_secondary(self, **updates):
        if updates:
            raise KeyError(f"Unknown secondary parameters: {sorted(updates)}")
        return SymmetryEquivalentCircleSystem()


class NumericalDeclaredJacobianSystem(OLSS):
    """Nonlinear residual whose declared Jacobian uses the checker's old stencil."""

    def __init__(self) -> None:
        self._spec = ShootingSpec((ShootingBlock("x"),))

    def shooting_spec(self):
        return self._spec

    def system_spec(self):
        return basic_shooting_system_spec(
            "numerical-declared-jacobian",
            shooting_spec=self._spec,
            residual_size=1,
            jacobian_source="numerical-declared",
        )

    def residuals(self, variables, *, options=None):
        del options
        x = float(variables.block("x"))
        return np.array([np.exp(x)])

    def jacobian(self, variables, *, options=None):
        del options
        x = float(variables.block("x"))
        step = 1.0e-6 * max(1.0, abs(x))
        return np.array([[(np.exp(x + step) - np.exp(x - step)) / (2.0 * step)]])

    def with_secondary(self, **updates):
        if updates:
            raise KeyError(f"Unknown secondary parameters: {sorted(updates)}")
        return NumericalDeclaredJacobianSystem()


class DirectCostShootingSystem(OLSS):
    """Underdetermined root line y=0 with minimum cost at x=2."""

    def __init__(self) -> None:
        self._spec = ShootingSpec((ShootingBlock("x"), ShootingBlock("y")))

    def shooting_spec(self):
        return self._spec

    def system_spec(self):
        return basic_shooting_system_spec(
            "direct-cost-shooting",
            shooting_spec=self._spec,
            residual_size=1,
            metrics={
                "cost": shooting_metric_spec("Quadratic selection cost", role="selection", required=True)
            },
            selection_metric="cost",
            capabilities={"cost_gradient": True},
        )

    def residuals(self, variables, *, options=None):
        del options
        return np.array([float(variables.block("y"))])

    def jacobian(self, variables, *, options=None):
        del variables, options
        return np.array([[0.0, 1.0]])

    def evaluate(self, variables, *, options=None):
        residuals = self.residuals(variables, options=options)
        x = float(variables.block("x"))
        return {**_base_metrics(residuals), "cost": (x - 2.0) ** 2}

    def verify(self, variables, *, options=None):
        residuals = self.residuals(variables, options=options)
        return ShootingVerification(
            residuals=residuals,
            metrics=self.evaluate(variables, options=options),
            options=dict(options or {}),
        )

    def cost_gradient(self, variables, *, options=None):
        del options
        x = float(variables.block("x"))
        return np.array([2.0 * (x - 2.0), 0.0])

    def with_secondary(self, **updates):
        if updates:
            raise KeyError(f"Unknown secondary parameters: {sorted(updates)}")
        return DirectCostShootingSystem()
