"""Minimal OPTIMIZER v3 system template.

Why this file exists
--------------------
This template shows the expected shape of a project-specific ``system.py``.  It is
not a physical model.  It is a small, executable example of the contract from
``optimizer/system.py``.

The key design idea is that the system owns the cost and derivative logic.  A real
quantum-control system should replace the toy quadratic in this file with analytical
forward propagation, metrics, and gradient/costate equations.  The optimizer should
still call the same methods:

    system.control_spec()
    system.evaluate(controls)
    system.gradient(controls)
    system.with_params(...)

How it fits the architecture
----------------------------
- New downstream systems can copy this file as a starting point.
- Tests can use it as a simple contract example.
- Later documentation can point users here before they write a real system.

What to replace in a real system
--------------------------------
- ``SystemParams`` should become the physical and cost-prefactor parameter set.
- ``control_spec`` should expose the real control names and dimension.
- ``evaluate`` should run the real dynamics and return all useful metrics.
- ``gradient`` should return the analytical gradient in ``Controls`` format.
- optional ``residuals`` and ``jacobian`` should be added for hard-constraint repairs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from optimizer.controls import ControlSpec, Controls


@dataclass(frozen=True)
class SystemParams:
    control_dim: int = 100
    dt: float = 0.01
    target_weight: float = 1.0
    energy_weight: float = 0.0


class System:
    """Toy template system with one control channel.

    Real systems should keep the same public method names and replace the internals
    with analytical propagation and gradients.
    """

    def __init__(self, params: SystemParams | None = None) -> None:
        self.params = params or SystemParams()

    def control_spec(self) -> ControlSpec:
        return ControlSpec(
            keys=("u",),
            control_dim=int(self.params.control_dim),
            dtype=float,
            dt=float(self.params.dt),
            meta={"template": "replace with physical channel metadata"},
        )

    def evaluate(self, controls: Controls) -> dict[str, Any]:
        u = controls.channel("u")
        target_term = float(self.params.target_weight) * float(np.sum((u - 1.0) ** 2))
        energy = float(np.sum(u * u) * float(self.params.dt))
        J = target_term + float(self.params.energy_weight) * energy
        return {
            "J": J,
            "target_term": target_term,
            "energy": energy,
        }

    def gradient(self, controls: Controls) -> Controls:
        u = controls.channel("u")
        grad = (
            2.0 * float(self.params.target_weight) * (u - 1.0)
            + 2.0 * float(self.params.energy_weight) * float(self.params.dt) * u
        )
        return Controls.from_dict(self.control_spec(), {"u": grad}, name="gradient")

    def with_params(self, **updates: Any) -> "System":
        return System(replace(self.params, **updates))

    # Optional advanced hooks can be added by systems that need repairs/projected tools:
    #
    # def residuals(self, controls: Controls, name: str = "hard") -> np.ndarray:
    #     ...
    #
    # def jacobian(self, controls: Controls, name: str = "hard") -> np.ndarray:
    #     ...
