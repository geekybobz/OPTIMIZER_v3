"""Pi-pulse sample system, OLSS (shooting) route.

Same physics as pi_pulse_olgs.py, written as indirect (PMP) shooting in the
reduced rotation-angle coordinate theta (Bloch vector in the y-z plane):

    theta' = u,   cost C = 0.5 * int u^2 dt  [+ w (1 - F(theta(T))) when soft]
    H = p u - u^2 / 2   ->   u = p,   p' = 0

Shooting variable z = p(0) (one costate). Residual (one row):
    hard: Re <0|psi(T)> = cos(theta(T) / 2) = 0
    soft: p(T) - (w / 2) sin(theta(T)) = 0,  w = 1 / (2 alpha)
The soft cost C is (1 / (2 alpha)) * [(1 - F) + alpha E], the OLGS soft J.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from optimizer import OLSS, ShootingBlock, ShootingSpec, ShootingVariables
from optimizer import basic_shooting_system_spec
from optimizer.system_olss.results import ShootingVerification


def _merge(defaults: Mapping[str, Any], updates: Mapping[str, Any] | None, label: str) -> dict[str, Any]:
    updates = dict(updates or {})
    unknown = sorted(set(updates) - set(defaults))
    if unknown:
        raise KeyError(f"Unknown {label} parameters: {unknown}")
    return {**defaults, **updates}


class PiPulseShooting(OLSS):
    MODE = "hard"

    def __init__(self, primary: Mapping[str, Any] | None = None, secondary: Mapping[str, Any] | None = None) -> None:
        self.primary = _merge({"T": 1.0, "steps": 200}, primary, "primary")
        self.secondary = _merge({"alpha": 0.01}, secondary, "secondary")
        if float(self.primary["T"]) <= 0.0 or int(self.primary["steps"]) < 1:
            raise ValueError("T must be positive and steps >= 1.")
        if float(self.secondary["alpha"]) <= 0.0:
            raise ValueError("alpha must be positive.")
        self._spec = ShootingSpec((ShootingBlock(
            "p0", shape=(1,), role="costate",
            meaning="initial costate; the extremal control is u(t) = p(t)"),))

    def shooting_spec(self) -> ShootingSpec:
        return self._spec

    def system_spec(self) -> dict[str, Any]:
        return basic_shooting_system_spec(
            f"pi-pulse-olss-{self.MODE}",
            name=f"Pi pulse, shooting, {self.MODE}",
            system_type="qubit-indirect-pmp",
            description="One-channel resonant qubit |0> -> |1>, reduced PMP shooting.",
            shooting_spec=self._spec,
            residual_size=1,
            metrics={
                "energy": {"label": "Control energy int u^2 dt", "role": "selection"},
                "fidelity": {"label": "Transfer fidelity", "role": "diagnostic"},
                "soft_cost": {"label": "(1 - F) + alpha E", "role": "selection"},
            },
            parameters={"primary": {**self.primary, "mode": self.MODE}, "secondary": dict(self.secondary)},
            jacobian_source="variational",
            selection_metric="energy" if self.MODE == "hard" else "soft_cost",
            verification_options={"steps": 4 * int(self.primary["steps"])},
            verification_tolerance=1.0e-9,
        )

    def _propagate(self, z: np.ndarray, options: Mapping[str, Any] | None) -> dict[str, float]:
        """RK4 for y = (theta, p, E, dtheta/dz, dp/dz)."""

        steps = int(dict(options or {}).get("steps", self.primary["steps"]))
        h = float(self.primary["T"]) / steps

        def rhs(y: np.ndarray) -> np.ndarray:
            return np.array([y[1], 0.0, y[1] ** 2, y[4], 0.0])

        y = np.array([0.0, float(z[0]), 0.0, 0.0, 1.0])
        for _ in range(steps):
            k1 = rhs(y)
            k2 = rhs(y + 0.5 * h * k1)
            k3 = rhs(y + 0.5 * h * k2)
            k4 = rhs(y + h * k3)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return {"theta": y[0], "p": y[1], "energy": y[2], "dtheta": y[3], "dp": y[4]}

    def _residual(self, end: Mapping[str, float]) -> tuple[float, float]:
        if self.MODE == "hard":
            return np.cos(0.5 * end["theta"]), -0.5 * np.sin(0.5 * end["theta"]) * end["dtheta"]
        w = 1.0 / (2.0 * float(self.secondary["alpha"]))
        return (end["p"] - 0.5 * w * np.sin(end["theta"]),
                end["dp"] - 0.5 * w * np.cos(end["theta"]) * end["dtheta"])

    def residuals(self, variables: ShootingVariables, *, options: Mapping[str, Any] | None = None) -> np.ndarray:
        return np.array([self._residual(self._propagate(variables.flatten(), options))[0]])

    def jacobian(self, variables: ShootingVariables, *, options: Mapping[str, Any] | None = None) -> np.ndarray:
        return np.array([[self._residual(self._propagate(variables.flatten(), options))[1]]])

    def _metrics(self, end: Mapping[str, float]) -> dict[str, float]:
        residual = np.array([self._residual(end)[0]])
        fidelity = float(np.sin(0.5 * end["theta"]) ** 2)
        soft = (1.0 - fidelity) + float(self.secondary["alpha"]) * float(end["energy"])
        return {**self._residual_metrics(residual), "energy": float(end["energy"]),
                "fidelity": fidelity, "soft_cost": soft}

    def evaluate(self, variables: ShootingVariables, *, options: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self._metrics(self._propagate(variables.flatten(), options))

    def verify(self, variables: ShootingVariables, *, options: Mapping[str, Any] | None = None) -> ShootingVerification:
        end = self._propagate(variables.flatten(), options)
        return ShootingVerification(residuals=np.array([self._residual(end)[0]]), metrics=self._metrics(end),
                                    options=dict(options or {}), independent=True,
                                    independence_source="rk4_step_refinement")

    def recover_controls(self, variables: ShootingVariables, *, times: np.ndarray) -> dict[str, Any]:
        grid = np.asarray(times, dtype=float).reshape(-1)
        return {"times": grid, "channels": ("ux",),
                "values": np.full((1, grid.size), float(variables.flatten()[0]))}

    def with_secondary(self, **updates: Any) -> "PiPulseShooting":
        return type(self)(primary=self.primary, secondary={**self.secondary, **updates})


class PiPulseShootingSoft(PiPulseShooting):
    MODE = "soft"


def _build(cls, config: Mapping[str, Any] | None, updates: Mapping[str, Any]):
    payload = dict(config or {})
    primary = payload.pop("primary", None)
    secondary = payload.pop("secondary", None)
    if payload:
        raise KeyError(f"Unknown top-level configuration keys: {sorted(payload)}")
    return cls(primary=primary, secondary={**dict(secondary or {}), **updates})


def hard(config: Mapping[str, Any] | None = None, **updates: Any) -> PiPulseShooting:
    return _build(PiPulseShooting, config, updates)


def soft(config: Mapping[str, Any] | None = None, **updates: Any) -> PiPulseShootingSoft:
    return _build(PiPulseShootingSoft, config, updates)


system = hard
