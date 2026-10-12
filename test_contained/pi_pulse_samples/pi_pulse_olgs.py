"""Pi-pulse sample system, OLGS (gradient) route.

Model: one-channel resonant qubit, H(t) = 0.5 * u(t) * sigma_x, piecewise
constant on N intervals of length dt = T / N; transfer |0> -> |1>.

    F = |<1|psi(T)>|^2          E = sum_k u_k^2 dt   (full energy)

Factories:
    soft(...)  J = (1 - F) + alpha * E, no constraint hooks.
    hard(...)  J = E, equality constraint "hard": Re <0|psi(T)> = 0.
               (<0|psi(T)> = cos(theta/2) is real for this model.)
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from optimizer import ControlSpec, Controls, basic_system_spec

_KET0 = np.array([1.0, 0.0], dtype=complex)
_KET1 = np.array([0.0, 1.0], dtype=complex)


def _merge(defaults: Mapping[str, Any], updates: Mapping[str, Any] | None, label: str) -> dict[str, Any]:
    updates = dict(updates or {})
    unknown = sorted(set(updates) - set(defaults))
    if unknown:
        raise KeyError(f"Unknown {label} parameters: {unknown}")
    return {**defaults, **updates}


class PiPulseSoft:
    """Soft route: infidelity plus weighted energy."""

    MODE = "soft"

    def __init__(self, primary: Mapping[str, Any] | None = None, secondary: Mapping[str, Any] | None = None) -> None:
        self.primary = _merge({"N": 100, "T": 1.0}, primary, "primary")
        self.secondary = _merge({"alpha": 0.01}, secondary, "secondary")
        if int(self.primary["N"]) < 1 or float(self.primary["T"]) <= 0.0:
            raise ValueError("N must be >= 1 and T must be positive.")
        if float(self.secondary["alpha"]) < 0.0:
            raise ValueError("alpha must be nonnegative.")
        self._key: bytes | None = None
        self._forward: dict[str, Any] = {}

    def control_spec(self) -> ControlSpec:
        N, T = int(self.primary["N"]), float(self.primary["T"])
        return ControlSpec(keys=("ux",), control_dim=N, dt=T / N)

    def _spec_extras(self) -> dict[str, Any]:
        return {}

    def system_spec(self) -> dict[str, Any]:
        return basic_system_spec(
            f"pi-pulse-olgs-{self.MODE}",
            name=f"Pi pulse, gradient, {self.MODE}",
            system_type="qubit",
            description="One-channel resonant qubit |0> -> |1>; H = u sigma_x / 2.",
            control_spec=self.control_spec(),
            parameters={"primary": {**self.primary, "mode": self.MODE}, "secondary": dict(self.secondary)},
            metrics={
                "fidelity": {"label": "Transfer fidelity", "role": "quality"},
                "infidelity": {"label": "1 - F", "role": "quality"},
                "energy": {"label": "Control energy sum u^2 dt", "role": "resource"},
                "soft_cost": {"label": "(1 - F) + alpha E", "role": "diagnostic"},
                "c0_re": {"label": "Re <0|psi(T)>", "role": "constraint"},
            },
            **self._spec_extras(),
        )

    # Physics: forward propagation -> backward propagation -> gradient.

    def forward_prop(self, controls: Controls) -> dict[str, Any]:
        u = np.asarray(controls.as_matrix(copy=False), dtype=float)[0]
        key = u.tobytes()
        if self._key != key:
            dt = float(self.primary["T"]) / int(self.primary["N"])
            c, s = np.cos(0.5 * u * dt), np.sin(0.5 * u * dt)
            psi = np.empty((u.size + 1, 2), dtype=complex)
            psi[0] = _KET0
            for k in range(u.size):
                a, b = psi[k]
                psi[k + 1] = (c[k] * a - 1j * s[k] * b, c[k] * b - 1j * s[k] * a)
            self._forward = {"u": u.copy(), "dt": dt, "c": c, "s": s, "psi": psi}
            self._key = key
        return self._forward

    def _adjoint(self, forward: Mapping[str, Any], target: np.ndarray) -> np.ndarray:
        """chi[k] = (U_N ... U_{k+1})^dagger target, k = 1..N."""

        c, s = forward["c"], forward["s"]
        n = c.size
        chi = np.empty((n + 1, 2), dtype=complex)
        chi[n] = target
        for k in range(n, 1, -1):
            a, b = chi[k]
            chi[k - 1] = (c[k - 1] * a + 1j * s[k - 1] * b, c[k - 1] * b + 1j * s[k - 1] * a)
        return chi

    def _amplitude_derivative(self, forward: Mapping[str, Any], target: np.ndarray) -> np.ndarray:
        """d <target|psi(T)> / d u_k, exact: sigma_x commutes with each step."""

        chi, psi = self._adjoint(forward, target)[1:], forward["psi"][1:]
        overlap = np.conj(chi[:, 0]) * psi[:, 1] + np.conj(chi[:, 1]) * psi[:, 0]
        return -0.5j * forward["dt"] * overlap

    def back_prop(self, controls: Controls) -> dict[str, Any]:
        forward = self.forward_prop(controls)
        return {"dc1": self._amplitude_derivative(forward, _KET1)}

    def _metrics(self, forward: Mapping[str, Any]) -> dict[str, float]:
        psi_t = forward["psi"][-1]
        fidelity = float(abs(psi_t[1]) ** 2)
        energy = float(np.sum(forward["u"] ** 2) * forward["dt"])
        soft = (1.0 - fidelity) + float(self.secondary["alpha"]) * energy
        return {"fidelity": fidelity, "infidelity": 1.0 - fidelity, "energy": energy,
                "soft_cost": soft, "c0_re": float(psi_t[0].real)}

    def evaluate(self, controls: Controls) -> dict[str, Any]:
        metrics = self._metrics(self.forward_prop(controls))
        return {"J": metrics["soft_cost"], **metrics}

    def gradient(self, controls: Controls) -> Controls:
        forward = self.forward_prop(controls)
        c1 = forward["psi"][-1][1]
        d_fidelity = 2.0 * np.real(np.conj(c1) * self.back_prop(controls)["dc1"])
        grad = -d_fidelity + 2.0 * float(self.secondary["alpha"]) * forward["u"] * forward["dt"]
        return Controls.from_matrix(self.control_spec(), grad[None, :], name="gradient")

    def cache_reset(self) -> None:
        self._key, self._forward = None, {}

    def with_secondary(self, **updates: Any):
        return type(self)(primary=self.primary, secondary={**self.secondary, **updates})


class PiPulseHard(PiPulseSoft):
    """Hard route: minimum energy subject to Re <0|psi(T)> = 0."""

    MODE = "hard"

    def _spec_extras(self) -> dict[str, Any]:
        return {
            "derivatives": {"constraint_jacobian": {"hook": "jacobian", "source": "analytical"}},
            "constraints": {"hard": {"kind": "equality", "provider": "residuals"}},
        }

    def evaluate(self, controls: Controls) -> dict[str, Any]:
        metrics = self._metrics(self.forward_prop(controls))
        return {"J": metrics["energy"], **metrics}

    def gradient(self, controls: Controls) -> Controls:
        forward = self.forward_prop(controls)
        grad = 2.0 * forward["u"] * forward["dt"]
        return Controls.from_matrix(self.control_spec(), grad[None, :], name="gradient")

    def residuals(self, controls: Controls, *, name: str = "hard") -> np.ndarray:
        if name != "hard":
            raise KeyError(f"Unknown residual set {name!r}.")
        return np.array([self.forward_prop(controls)["psi"][-1][0].real])

    def jacobian(self, controls: Controls, *, name: str = "hard") -> np.ndarray:
        if name != "hard":
            raise KeyError(f"Unknown residual set {name!r}.")
        forward = self.forward_prop(controls)
        return np.real(self._amplitude_derivative(forward, _KET0))[None, :]


def _build(cls, config: Mapping[str, Any] | None, updates: Mapping[str, Any]):
    payload = dict(config or {})
    primary = payload.pop("primary", None)
    secondary = payload.pop("secondary", None)
    if payload:
        raise KeyError(f"Unknown top-level configuration keys: {sorted(payload)}")
    return cls(primary=primary, secondary={**dict(secondary or {}), **updates})


def soft(config: Mapping[str, Any] | None = None, **updates: Any) -> PiPulseSoft:
    return _build(PiPulseSoft, config, updates)


def hard(config: Mapping[str, Any] | None = None, **updates: Any) -> PiPulseHard:
    return _build(PiPulseHard, config, updates)


system = soft
