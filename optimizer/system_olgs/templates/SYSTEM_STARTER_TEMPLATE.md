---
title: Name Your System
type: system_starter_template
module: systems/name_your_system
tags:
  - optimizer
  - system
  - olgs
  - template
depends_on:
  - README
  - CONTRACT
  - PARAMS
  - LIFECYCLE
connects_to:
  - DERIVATIVES
  - LOGGING_BOUNDARY
---

# Name Your System

Use this template when starting a new physical OLGS system.

Copy the Python skeleton into:

```text
systems/name_your_system/system.py
```

Then rename:

```text
NameYourSystem
SYSTEM_NAME
systems/name_your_system
control channel names
primary defaults
secondary defaults
physical forward/backward/gradient logic
```

## Reference Map

- [[README|system_olgs]]
- [[CONTRACT|OLGS API Contract]]
- [[PARAMS|Primary and Secondary Params]]
- [[LIFECYCLE|Forward, Backward, Gradient Flow]]
- [[DERIVATIVES|Optional Derivative Hooks]]
- [[LOGGING_BOUNDARY|Logging Boundary Placeholder]]

## Design Rule

The system file describes the physical model and OLGS API only.

Keep these outside the system file:

```text
saved best controls
control loading
optimizer runs
notebook plotting
experiment-specific wrappers
ad hoc aliases
```

Keep these inside the system file:

```text
control_spec
forward_prop
back_prop
evaluate
gradient
with_secondary
simulate when the simulation is part of the system behavior
optional residual/Jacobian/Hessian hooks
```

## Minimal `system.py`

```python
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from optimizer.controls import ControlSpec, Controls
from optimizer.system_olgs import (
    BackwardResult,
    ForwardResult,
    OLGS,
    SimulationResult,
)


SYSTEM_NAME = "NameYourSystem"

_DEFAULT_PRIMARY = {
    # Physical and numerical constants. These normally stay fixed during a run.
    "tau": 1.0,
    "N": 1001,
    "control_channels": ("u",),
}

_DEFAULT_SECONDARY = {
    # Objective and curriculum weights. These can change through with_secondary.
    "target_weight": 1.0,
    "energy_weight": 0.0,
}


def _merge_config(
    defaults: Mapping[str, Any],
    values: Mapping[str, Any] | None = None,
    **updates: Any,
) -> dict[str, Any]:
    """Merge user dictionaries with explicit defaults.

    None means use defaults.
    dict means override named defaults.
    unknown keys fail early so notebook mistakes are visible.
    """

    payload = dict(defaults)

    if values is not None:
        if not isinstance(values, Mapping):
            raise TypeError("configuration must be a mapping/dict.")
        unknown = set(values) - set(defaults)
        if unknown:
            raise KeyError(f"Unknown configuration keys: {sorted(unknown)}")
        payload.update(dict(values))

    if updates:
        unknown = set(updates) - set(defaults)
        if unknown:
            raise KeyError(f"Unknown configuration keys: {sorted(unknown)}")
        payload.update(updates)

    return payload


class NameYourSystem(OLGS):
    """Physical OLGS system description.

    This class should contain physics and system-owned API behavior only.
    Controls are created outside and passed in by notebooks, tests, optimizers,
    or experiment scripts.
    """

    def __init__(
        self,
        primary: Mapping[str, Any] | None = None,
        secondary: Mapping[str, Any] | None = None,
    ) -> None:
        self.primary = _merge_config(_DEFAULT_PRIMARY, primary)
        self.secondary = _merge_config(_DEFAULT_SECONDARY, secondary)

        self.N = int(self.primary["N"])
        self.tau = float(self.primary["tau"])
        if self.N < 2:
            raise ValueError("N must be >= 2.")
        if not np.isfinite(self.tau) or self.tau <= 0.0:
            raise ValueError("tau must be finite and positive.")

        # Endpoint-sampled convention:
        # N samples, N - 1 intervals, dt = tau / (N - 1).
        self.dt = self.tau / float(self.N - 1)

        self._latest_controls: Controls | None = None
        self._latest_metrics: dict[str, Any] | None = None
        self._latest_state: dict[str, Any] = {}

    def control_spec(self) -> ControlSpec:
        """Declare the control layout expected by this system.

        The canonical Controls matrix shape is:
            (n_controls, control_dim)

        The row order is exactly the order of ControlSpec.keys.
        """

        return ControlSpec(
            keys=tuple(self.primary["control_channels"]),
            control_dim=self.N,
            dtype=float,
            dt=self.dt,
            meta={"system": SYSTEM_NAME},
        )

    def forward_prop(self, controls: Controls) -> ForwardResult:
        """Run physical forward propagation.

        Replace the toy cumulative integral below with the real physical
        propagator, trajectory, terminal state, and diagnostic quantities.
        """

        self.validate_controls(controls)
        u = controls.channel("u", copy=False)

        trajectory = np.cumsum(u) * self.dt
        terminal = float(trajectory[-1])

        data = {
            "trajectory": trajectory,
            "terminal": terminal,
        }

        self._latest_controls = controls.copy(name=controls.name)
        self._latest_state = data
        return ForwardResult(controls=controls, data=data)

    def back_prop(self, controls: Controls) -> BackwardResult:
        """Run adjoint/costate propagation.

        This method should return whatever gradient needs. It may call
        forward_prop when the matching forward state is not already available.
        """

        fwd = self.forward_prop(controls)

        terminal_error = fwd.data["terminal"] - 1.0
        costate = np.full(self.N, terminal_error, dtype=float)

        return BackwardResult(
            controls=controls,
            data={
                "costate": costate,
                "terminal_error": terminal_error,
            },
        )

    def evaluate(self, controls: Controls) -> dict[str, Any]:
        """Return optimizer-facing metrics.

        Required:
            metrics["J"] must be a finite scalar.

        Extra metric names are system-defined and should stay stable enough for
        notebooks, logs, and comparison tables.
        """

        fwd = self.forward_prop(controls)
        u = controls.channel("u", copy=False)

        target_term = float(self.secondary["target_weight"]) * fwd.data["terminal"] ** 2
        energy = float(np.sum(u * u) * self.dt)
        J = target_term + float(self.secondary["energy_weight"]) * energy

        metrics = {
            "J": float(J),
            "target_term": float(target_term),
            "energy": float(energy),
        }

        if not np.isfinite(metrics["J"]):
            raise FloatingPointError("Objective J must be finite.")

        self._latest_metrics = metrics
        return metrics

    def gradient(self, controls: Controls) -> Controls:
        """Return analytical dJ/du as Controls.

        Gradient is required for OLGS. Do not rely on numerical fallback for the
        normal optimizer path.
        """

        self.validate_controls(controls)
        bwd = self.back_prop(controls)
        u = controls.channel("u", copy=False)

        grad_u = (
            2.0 * float(self.secondary["target_weight"]) * bwd.data["costate"]
            + 2.0 * float(self.secondary["energy_weight"]) * self.dt * u
        )

        return Controls.from_dict(
            self.control_spec(),
            {"u": grad_u},
            name="gradient",
        )

    def with_secondary(self, **updates: Any) -> "NameYourSystem":
        """Return an equivalent system with updated secondary parameters.

        This is the curriculum/objective-weight update hook.
        Primary physical parameters are carried through unchanged.
        """

        secondary = _merge_config(self.secondary, updates)
        return NameYourSystem(primary=self.primary, secondary=secondary)

    def simulate(
        self,
        controls: Controls,
        *,
        direction: np.ndarray | None = None,
        points: np.ndarray | None = None,
        n_points: int | None = None,
        alpha_range: tuple[float, float, int] | None = None,
    ) -> SimulationResult:
        """Run richer system-owned simulation.

        Use this for expensive checks, ensemble scans, robustness curves,
        projection data, Bloch data, or other plot-ready diagnostics.

        Optimizers should normally use evaluate and gradient instead.
        """

        metrics = self.evaluate(controls)

        # Replace with system-specific simulation data.
        data = {
            "direction": direction,
            "points": points,
            "n_points": n_points,
            "alpha_range": alpha_range,
        }

        return SimulationResult(
            controls=controls,
            metrics=metrics,
            trajectories={},
            data=data,
        )

    def residuals(self, controls: Controls, name: str = "default") -> np.ndarray:
        """Optional residual vector for repair/projected-gradient tools."""

        raise NotImplementedError("Define residuals only when this system needs them.")

    def jacobian(self, controls: Controls, name: str = "default") -> np.ndarray:
        """Optional analytical residual Jacobian.

        If this is missing, explicit numerical fallback tools can be used from
        optimizer.system_olgs.derivatives.
        """

        raise NotImplementedError("Define jacobian only when this system needs it.")

    def describe(self) -> dict[str, Any]:
        """Return compact metadata for notebooks, logs, and inspections."""

        payload = super().describe()
        payload["name"] = SYSTEM_NAME
        return payload


system = NameYourSystem

__all__ = ["NameYourSystem", "system"]
```

## Notebook/API Smoke Case

```python
import numpy as np

from optimizer.controls import Controls
from systems.name_your_system.system import system


primary = {
    "tau": 1.0,
    "N": 1001,
    "control_channels": ("u",),
}

secondary = {
    "target_weight": 1.0,
    "energy_weight": 0.0,
}

q = system(primary=primary, secondary=secondary)

spec = q.control_spec()
u0 = Controls.zeros(spec, name="zero")

fwd = q.forward_prop(u0)
bwd = q.back_prop(u0)
metrics = q.evaluate(u0)
grad = q.gradient(u0)
sim = q.simulate(u0, n_points=250, alpha_range=(0.0, 0.5, 501))

print(type(fwd), fwd.controls.spec.shape, fwd.data.keys())
print(type(bwd), bwd.data.keys())
print(metrics)
print(type(grad), grad.spec.shape)
print(type(sim), sim.metrics.keys(), sim.data.keys())
```

## Required API Checklist

```text
control_spec()
forward_prop(controls)
back_prop(controls)
evaluate(controls)
gradient(controls)
with_secondary(**updates)
```

## Optional API Checklist

```text
simulate(controls, ...)
residuals(controls, name=...)
jacobian(controls, name=...)
hessian(controls)
hvp(controls, vector)
second_derivative(controls, ...)
metric_schema()
residual_schema()
describe()
cache_reset()
cache_status()
```

## Before First Optimizer Run

Check these manually in a lightweight notebook:

```text
import path works
system constructs from primary/secondary dicts
control_spec returns the expected names and shape
Controls.zeros or Controls.from_matrix matches the spec
forward_prop returns a readable result
back_prop returns the adjoint/costate data expected by gradient
evaluate returns finite J
gradient returns Controls with the same spec and finite values
with_secondary returns a new system with updated secondary dict
simulate returns plot-ready data when the system defines richer scans
```
