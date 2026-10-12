"""OLSS project system: minimum-energy Duffing transfer by indirect (PMP) shooting.

Filled in from the `optimizer-system --family olss` scaffold.

    x1' = x2,  x2' = -x1 - eps x1^3 + u,  minimize 1/2 int u^2 dt
    PMP: u = -p2,  p1' = p2 (1 + 3 eps x1^2),  p2' = -p1
    shooting variables: p(0) = (p1, p2); residuals: x(T) - x_target

Shooting pattern: guess p(0) -> propagate state+costate (+ variational equations)
-> residuals and Jacobian -> library update step -> independent verification.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from optimizer import OLSS, ShootingBlock, ShootingSpec, ShootingVariables, VerificationCheck
from optimizer import ShootingVerification, basic_shooting_system_spec, shooting_metric_spec


class DuffingShooting(OLSS):
    """Minimum-energy Duffing rest-to-origin transfer."""

    def __init__(
        self,
        primary: Mapping[str, Any] | None = None,
        secondary: Mapping[str, Any] | None = None,
    ) -> None:
        primary_defaults = {"T": 2.0, "x0": (1.0, 0.0), "x_target": (0.0, 0.0)}
        secondary_defaults: dict[str, Any] = {"epsilon": 0.0}
        self.primary = {**primary_defaults, **dict(primary or {})}
        self.secondary = {**secondary_defaults, **dict(secondary or {})}
        self._shooting_spec = ShootingSpec(
            (
                ShootingBlock(
                    "initial_costate",
                    shape=(2,),
                    meaning="PMP costate (p1, p2) at t = 0",
                    scale=1.0,
                ),
            )
        )

    def system_spec(self) -> dict[str, Any]:
        return basic_shooting_system_spec(
            "duffing-min-energy",
            name="Duffing minimum-energy transfer",
            system_type="pmp_indirect",
            description="Indirect shooting on p(0); residual x(T) - x_target; RK4 with options['steps'].",
            shooting_spec=self.shooting_spec(),
            residual_size=2,
            residual_groups=[{"name": "terminal_state", "shape": (2,)}],
            metrics={
                "energy": shooting_metric_spec("Control energy", role="selection", required=True),
                "hamiltonian_drift": shooting_metric_spec("max |H(t) - H(0)|", role="diagnostic", required=True),
            },
            parameters={
                "primary": dict(self.primary),
                "secondary": dict(self.secondary),
            },
            jacobian_source="variational",
            selection_metric="energy",
            verification_options={"steps": 4000},
            verification_tolerance=1e-7,
        )

    def shooting_spec(self) -> ShootingSpec:
        return self._shooting_spec

    # ---------------------------------------------------------------- physics
    def _rhs(self, y: np.ndarray, with_sensitivity: bool) -> np.ndarray:
        eps = float(self.secondary["epsilon"])
        x1, x2, p1, p2 = y[:4]
        f = np.array([x2, -x1 - eps * x1**3 - p2, p2 * (1.0 + 3.0 * eps * x1**2), -p1])
        if not with_sensitivity:
            return f
        a = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [-1.0 - 3.0 * eps * x1**2, 0.0, 0.0, -1.0],
                [6.0 * eps * x1 * p2, 0.0, 0.0, 1.0 + 3.0 * eps * x1**2],
                [0.0, 0.0, -1.0, 0.0],
            ]
        )
        s = y[4:].reshape(4, 2)
        return np.concatenate((f, (a @ s).reshape(-1)))

    def forward_prop(self, variables: ShootingVariables, *, options=None, with_sensitivity=False):
        """RK4 propagation of state, costate, and optionally the variational equations."""

        steps = int(dict(options or {}).get("steps", 400))
        h = float(self.primary["T"]) / steps
        p0 = np.asarray(variables.block("initial_costate"), dtype=float)
        y = np.concatenate((np.asarray(self.primary["x0"], dtype=float), p0))
        if with_sensitivity:
            y = np.concatenate((y, np.array([[0, 0], [0, 0], [1, 0], [0, 1]], dtype=float).reshape(-1)))
        path = [y[:4].copy()]
        for _ in range(steps):
            k1 = self._rhs(y, with_sensitivity)
            k2 = self._rhs(y + 0.5 * h * k1, with_sensitivity)
            k3 = self._rhs(y + 0.5 * h * k2, with_sensitivity)
            k4 = self._rhs(y + h * k3, with_sensitivity)
            y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            path.append(y[:4].copy())
        return {"final": y, "path": np.array(path), "h": h}

    def residuals(self, variables, *, options=None) -> np.ndarray:
        final = self.forward_prop(variables, options=options)["final"]
        return final[:2] - np.asarray(self.primary["x_target"], dtype=float)

    def jacobian(self, variables, *, options=None) -> np.ndarray:
        final = self.forward_prop(variables, options=options, with_sensitivity=True)["final"]
        return final[4:].reshape(4, 2)[:2, :]

    def residuals_and_jacobian(self, variables, *, options=None):
        final = self.forward_prop(variables, options=options, with_sensitivity=True)["final"]
        return final[:2] - np.asarray(self.primary["x_target"], dtype=float), final[4:].reshape(4, 2)[:2, :]

    def _metrics(self, variables, options):
        run = self.forward_prop(variables, options=options)
        eps = float(self.secondary["epsilon"])
        x1, x2, p1, p2 = run["path"].T
        u = -p2
        weights = np.full(u.size, run["h"])
        weights[[0, -1]] *= 0.5
        hamiltonian = 0.5 * u**2 + p1 * x2 + p2 * (-x1 - eps * x1**3 + u)
        residual = run["path"][-1, :2] - np.asarray(self.primary["x_target"], dtype=float)
        return residual, {
            "residual_norm": float(np.linalg.norm(residual)),
            "residual_max_abs": float(np.max(np.abs(residual))),
            "energy": float(0.5 * np.sum(weights * u**2)),
            "hamiltonian_drift": float(np.max(np.abs(hamiltonian - hamiltonian[0]))),
        }

    def evaluate(self, variables, *, options=None):
        return self._metrics(variables, options)[1]

    def verify(self, variables, *, options=None):
        residual, metrics = self._metrics(variables, options)
        return ShootingVerification(
            residuals=residual,
            metrics=metrics,
            options=dict(options or {}),
            checks=(
                VerificationCheck(
                    "hamiltonian_conserved",
                    passed=metrics["hamiltonian_drift"] <= 1e-8,
                    value=metrics["hamiltonian_drift"],
                    tolerance=1e-8,
                ),
            ),
        )

    def with_secondary(self, **updates: Any) -> "DuffingShooting":
        secondary = dict(self.secondary)
        secondary.update(updates)
        return type(self)(primary=self.primary, secondary=secondary)


def system(config: Mapping[str, Any] | None = None, **updates: Any) -> DuffingShooting:
    """Construct the project system from dict-first configuration."""

    payload = dict(config or {})
    primary = payload.pop("primary", None)
    secondary = payload.pop("secondary", None)
    if payload:
        raise KeyError(f"Unknown top-level configuration keys: {sorted(payload)}")
    return DuffingShooting(primary=primary, secondary={**dict(secondary or {}), **updates})
