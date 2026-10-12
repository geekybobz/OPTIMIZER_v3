"""OLGS project system: qubit |0> -> |1> transfer with piecewise-constant x/y drives.

Filled in from the `optimizer-system --profile qubit` scaffold. The physics follows
the forward -> backward -> gradient pattern: forward_prop stores the states,
back_prop stores the costates, gradient assembles dJ/du from both.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.linalg import expm_frechet

from optimizer import ControlSpec, Controls, basic_system_spec, metric_spec

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
PSI0 = np.array([1, 0], dtype=complex)
TARGET = np.array([0, 1], dtype=complex)


class QubitTransfer:
    """Qubit state transfer with drift detuning and x/y control fields."""

    def __init__(
        self,
        primary: Mapping[str, Any] | None = None,
        secondary: Mapping[str, Any] | None = None,
    ) -> None:
        primary_defaults = {
            "control_keys": ("ux", "uy"),
            "control_dim": 100,
            "dt": 0.05,
        }
        secondary_defaults: dict[str, Any] = {"delta": 1.0, "energy_weight": 1.0e-3}
        self.primary = {**primary_defaults, **dict(primary or {})}
        self.secondary = {**secondary_defaults, **dict(secondary or {})}
        self._cache: dict[bytes, dict[str, Any]] = {}

    def system_spec(self) -> dict[str, Any]:
        return basic_system_spec(
            "qubit-transfer",
            name="Qubit transfer",
            system_type="qubit",
            description="|0> -> |1> transfer, H = delta/2 sz + ux/2 sx + uy/2 sy, J = 1 - F + w/2 sum u^2 dt.",
            control_spec=self.control_spec(),
            parameters={
                "primary": dict(self.primary),
                "secondary": dict(self.secondary),
            },
            metrics={
                "J": metric_spec("Objective", role="objective", required=True),
                "fidelity": metric_spec("Fidelity", role="figure_of_merit", required=True),
                "infidelity": metric_spec("Infidelity", role="objective_term", required=True),
                "energy": metric_spec("Control energy", role="objective_term", required=True),
            },
        )

    def control_spec(self) -> ControlSpec:
        return ControlSpec(
            keys=tuple(self.primary["control_keys"]),
            control_dim=int(self.primary["control_dim"]),
            dt=float(self.primary["dt"]),
        )

    # ---------------------------------------------------------------- physics
    def _generators(self, u: np.ndarray) -> list[np.ndarray]:
        delta = float(self.secondary["delta"])
        dt = float(self.primary["dt"])
        return [-1j * dt * 0.5 * (delta * SZ + u[0, k] * SX + u[1, k] * SY) for k in range(u.shape[1])]

    def forward_prop(self, controls: Controls) -> dict[str, Any]:
        """Propagate psi_0 forward; store states and slice propagators."""

        u = controls.as_matrix(copy=False)
        key = u.tobytes()
        if key in self._cache:
            return self._cache[key]
        dt = float(self.primary["dt"])
        states = [PSI0]
        props = []
        for generator in self._generators(u):
            unitary, _ = expm_frechet(generator, np.zeros((2, 2), dtype=complex))
            props.append(unitary)
            states.append(unitary @ states[-1])
        overlap = np.vdot(TARGET, states[-1])
        fidelity = float(abs(overlap) ** 2)
        energy = float(0.5 * np.sum(u**2) * dt)
        self._cache = {key: {"states": states, "props": props, "overlap": overlap, "fidelity": fidelity, "energy": energy}}
        return self._cache[key]

    def back_prop(self, controls: Controls) -> list[np.ndarray]:
        """Propagate the target costate backward: chi_k = (U_N ... U_{k+1})^dagger |target>."""

        forward = self.forward_prop(controls)
        costates = [TARGET]
        for unitary in reversed(forward["props"]):
            costates.append(unitary.conj().T @ costates[-1])
        costates.reverse()  # costates[k] pairs with props[k-1]; costates[N] = target
        return costates

    def evaluate(self, controls: Controls) -> dict[str, Any]:
        """Return declared metrics including finite scalar J."""

        forward = self.forward_prop(controls)
        infidelity = 1.0 - forward["fidelity"]
        weight = float(self.secondary["energy_weight"])
        return {
            "J": infidelity + weight * forward["energy"],
            "fidelity": forward["fidelity"],
            "infidelity": infidelity,
            "energy": forward["energy"],
        }

    def gradient(self, controls: Controls) -> Controls:
        """Return analytical dJ/du in the input control layout."""

        u = controls.as_matrix(copy=False)
        forward = self.forward_prop(controls)
        costates = self.back_prop(controls)
        dt = float(self.primary["dt"])
        weight = float(self.secondary["energy_weight"])
        overlap = forward["overlap"]
        grad = np.zeros_like(u)
        for k, generator in enumerate(self._generators(u)):
            for c, sigma in enumerate((SX, SY)):
                _, d_unitary = expm_frechet(generator, -1j * dt * 0.5 * sigma)
                d_overlap = np.vdot(costates[k + 1], d_unitary @ forward["states"][k])
                d_fidelity = 2.0 * np.real(np.conj(overlap) * d_overlap)
                grad[c, k] = -d_fidelity + weight * u[c, k] * dt
        return Controls.from_matrix(controls.spec, grad, name="gradient")

    def with_secondary(self, **updates: Any) -> "QubitTransfer":
        secondary = dict(self.secondary)
        secondary.update(updates)
        return type(self)(primary=self.primary, secondary=secondary)


def system(config: Mapping[str, Any] | None = None, **updates: Any) -> QubitTransfer:
    """Construct the project system from dict-first configuration."""

    payload = dict(config or {})
    primary = payload.pop("primary", None)
    secondary = payload.pop("secondary", None)
    if payload:
        raise KeyError(f"Unknown top-level configuration keys: {sorted(payload)}")
    secondary_payload = {**dict(secondary or {}), **updates}
    return QubitTransfer(primary=primary, secondary=secondary_payload)
