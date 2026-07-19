"""Minimal OLGS project-system template.

Copy this shape into a project-local ``system.py`` and replace the toy quadratic
with physical forward propagation, backward propagation, objective metrics, and an
analytical gradient.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from optimizer.controls import ControlSpec, Controls
from optimizer.system_olgs import OLGS


@dataclass(frozen=True)
class PrimaryParams:
    control_dim: int = 100
    dt: float = 0.01
    control_channels: tuple[str, ...] = ("u",)


@dataclass(frozen=True)
class SecondaryParams:
    target_weight: float = 1.0
    energy_weight: float = 0.0


class System(OLGS):
    """Toy OLGS system with one control channel."""

    def __init__(
        self,
        primary: PrimaryParams | dict[str, Any] | None = None,
        secondary: SecondaryParams | dict[str, Any] | None = None,
    ) -> None:
        self.primary = self._coerce_primary(primary)
        self.secondary = self._coerce_secondary(secondary)
        self._latest_metrics: dict[str, Any] | None = None

    @staticmethod
    def _coerce_primary(params: PrimaryParams | dict[str, Any] | None) -> PrimaryParams:
        if params is None:
            return PrimaryParams()
        if isinstance(params, PrimaryParams):
            return params
        if isinstance(params, dict):
            return PrimaryParams(**params)
        raise TypeError("primary must be None, dict, or PrimaryParams.")

    @staticmethod
    def _coerce_secondary(params: SecondaryParams | dict[str, Any] | None) -> SecondaryParams:
        if params is None:
            return SecondaryParams()
        if isinstance(params, SecondaryParams):
            return params
        if isinstance(params, dict):
            return SecondaryParams(**params)
        raise TypeError("secondary must be None, dict, or SecondaryParams.")

    def control_spec(self) -> ControlSpec:
        return ControlSpec(
            keys=self.primary.control_channels,
            control_dim=int(self.primary.control_dim),
            dtype=float,
            dt=float(self.primary.dt),
            meta={"template": "replace with physical channel metadata"},
        )

    def forward_prop(self, controls: Controls) -> dict[str, Any]:
        self.validate_controls(controls)
        u = controls.channel("u", copy=False)
        state = {
            "u": u.copy(),
            "target_error": u - 1.0,
            "energy_density": u * u,
        }
        self._latest_state = state
        return state

    def back_prop(self, controls: Controls) -> dict[str, Any]:
        state = self.forward_prop(controls)
        costate = {
            "d_target": 2.0 * float(self.secondary.target_weight) * state["target_error"],
            "d_energy": 2.0 * float(self.secondary.energy_weight) * float(self.primary.dt) * state["u"],
        }
        self._latest_costate = costate
        return costate

    def evaluate(self, controls: Controls) -> dict[str, Any]:
        state = self.forward_prop(controls)
        target_term = float(self.secondary.target_weight) * float(np.sum(state["target_error"] ** 2))
        energy = float(np.sum(state["energy_density"]) * float(self.primary.dt))
        J = target_term + float(self.secondary.energy_weight) * energy
        metrics = {
            "J": J,
            "target_term": target_term,
            "energy": energy,
        }
        self._latest_controls = controls.copy(name=controls.name)
        self._latest_metrics = metrics
        return metrics

    def gradient(self, controls: Controls) -> Controls:
        costate = self.back_prop(controls)
        grad = costate["d_target"] + costate["d_energy"]
        return Controls.from_dict(self.control_spec(), {"u": grad}, name="gradient")

    def with_secondary(self, **updates: Any) -> "System":
        payload = asdict(self.secondary)
        payload.update(updates)
        return System(primary=self.primary, secondary=replace(self.secondary, **payload))

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload["primary"] = asdict(self.primary)
        payload["secondary"] = asdict(self.secondary)
        return payload
