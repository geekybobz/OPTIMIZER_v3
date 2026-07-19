"""Neutral vector-control fixture for optimizer API tests."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from optimizer.controls import ControlSpec, Controls


class QuadraticVectorSystem:
    """Small OLGS-compatible system with named channels and analytical gradient."""

    def __init__(
        self,
        *,
        N: int = 17,
        tau: float = 1.0,
        objective_weight: float = 1.0,
        residual_weight: float = 0.25,
        energy_weight: float = 1.0e-3,
    ) -> None:
        self.primary = {"N": int(N), "tau": float(tau)}
        self.secondary = {
            "objective_weight": float(objective_weight),
            "residual_weight": float(residual_weight),
            "energy_weight": float(energy_weight),
        }
        self.params = dict(self.secondary)
        if self.primary["N"] < 2:
            raise ValueError("N must be >= 2.")
        if self.primary["tau"] <= 0.0:
            raise ValueError("tau must be positive.")
        self.N = self.primary["N"]
        self.dt = self.primary["tau"] / float(self.N - 1)
        self.tgrid = np.linspace(0.0, self.primary["tau"], self.N)

    def control_spec(self) -> ControlSpec:
        return ControlSpec(
            keys=("ux", "uy", "uz"),
            control_dim=self.N,
            dtype=float,
            dt=self.dt,
            meta={"fixture": "quadratic vector system"},
        )

    def _target_matrix(self) -> np.ndarray:
        t = self.tgrid
        return np.vstack(
            (
                0.30 * np.sin(np.pi * t / self.primary["tau"]),
                -0.20 * np.sin(2.0 * np.pi * t / self.primary["tau"]),
                0.15 * np.cos(np.pi * t / self.primary["tau"]),
            )
        )

    def reference_controls(self, *, amplitude: float = 1.0, name: str = "reference") -> Controls:
        return Controls.from_matrix(
            self.control_spec(),
            float(amplitude) * self._target_matrix(),
            name=name,
        )

    def evaluate(self, controls: Controls) -> dict[str, Any]:
        self.validate_controls(controls)
        u = controls.as_matrix(copy=False)
        target = self._target_matrix()
        diff = u - target
        residuals = self.residuals(controls)
        objective_error = 0.5 * self.dt * float(np.sum(diff * diff))
        residual_norm2 = float(np.dot(residuals, residuals))
        energy = self.dt * float(np.sum(u * u))
        J = (
            self.secondary["objective_weight"] * objective_error
            + self.secondary["residual_weight"] * residual_norm2
            + self.secondary["energy_weight"] * energy
        )
        fidelity = 1.0 / (1.0 + objective_error)
        return {
            "J": float(J),
            "objective_error": float(objective_error),
            "residual_norm2": residual_norm2,
            "energy": float(energy),
            "fidelity": float(fidelity),
        }

    def gradient(self, controls: Controls) -> Controls:
        self.validate_controls(controls)
        u = controls.as_matrix(copy=False)
        target = self._target_matrix()
        matrix = self.secondary["objective_weight"] * self.dt * (u - target)
        matrix += 2.0 * self.secondary["residual_weight"] * (u - target)
        matrix += 2.0 * self.secondary["energy_weight"] * self.dt * u
        return Controls.from_matrix(self.control_spec(), matrix, name="gradient")

    def residuals(self, controls: Controls, *, name: str = "hard") -> np.ndarray:
        self.validate_controls(controls)
        _ = name
        return (controls.as_matrix(copy=False) - self._target_matrix()).reshape(-1)

    def jacobian(self, controls: Controls, *, name: str = "hard") -> np.ndarray:
        self.validate_controls(controls)
        _ = name
        return np.eye(self.control_spec().size)

    def simulate(
        self,
        controls: Controls,
        *,
        alpha: float | None = None,
        directions: str | np.ndarray | None = None,
        parallel: bool = False,
    ) -> dict[str, Any]:
        metrics = self.evaluate(controls)
        if directions is None or isinstance(directions, str):
            dirs = np.eye(3)
        else:
            dirs = np.asarray(directions, dtype=float).reshape(-1, 3)
        alpha_value = 0.0 if alpha is None else float(alpha)
        channel_mean = np.mean(controls.as_matrix(copy=False), axis=1)
        directional_signal = dirs @ channel_mean
        fidelity = np.clip(metrics["fidelity"] - alpha_value * alpha_value * directional_signal * directional_signal, 0.0, 1.0)
        out = dict(metrics)
        out.update(
            {
                "terminal_overlap": complex(np.sqrt(metrics["fidelity"])),
                "robust_backend": "vectorized",
                "robust_parallel_used": bool(parallel and False),
                "robust_directions": dirs,
                "robust_fidelity_per_direction": fidelity,
            }
        )
        return out

    def with_secondary(self, **updates: Any) -> "QuadraticVectorSystem":
        payload = dict(self.secondary)
        payload.update(updates)
        return self.__class__(N=self.N, tau=self.primary["tau"], **payload)

    def validate_controls(self, controls: Controls) -> ControlSpec:
        spec = self.control_spec()
        if controls.spec.keys != spec.keys or controls.spec.control_dim != spec.control_dim:
            raise ValueError("controls do not match this system.")
        return spec

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "fixture",
            "class": self.__class__.__name__,
            "primary": dict(self.primary),
            "secondary": dict(self.secondary),
            "control_spec": self.control_spec().to_dict(),
        }


def system(config: Mapping[str, Any] | None = None, **updates: Any) -> QuadraticVectorSystem:
    payload = dict(config or {})
    payload.update(updates)
    return QuadraticVectorSystem(**payload)
