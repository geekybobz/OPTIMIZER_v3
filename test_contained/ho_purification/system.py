"""Contained OLGS adapter for the qubit--oscillator purification problem.

This is an independent, small test implementation of the project model.  It
uses the same piecewise-constant Hamiltonian and exact Fréchet derivative as
the authoritative project implementation, but is deliberately restricted to
the fock and thermal resource preparations needed for tracker trials.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from scipy.linalg import expm, expm_frechet

from optimizer.controls import ControlSpec, Controls
from optimizer.system_olgs import basic_system_spec
from optimizer.visualization import PlotData, PlotLayer


DTYPE = np.complex128


@dataclass(frozen=True)
class HOPurificationParams:
    tf: float = 1.0
    N: int = 24
    omega_B: float = 1.0
    g: float = 0.2
    n_osc: int = 4
    r: float = 0.6
    oscillator_state: str = "fock"
    fock_N: int = 1
    theta_B: float | None = None
    target_A_state: str = "e"
    control_weight: float = 0.0


class HOPurificationTrackerSystem:
    """One-channel, closed qubit--oscillator purification system.

    ``gradient`` is the exact derivative of the discretized objective, not the
    continuum PMP signal.  The latter remains useful physics, but is not used
    as an optimizer gradient in this test adapter.
    """

    def __init__(self, params: HOPurificationParams | dict[str, Any] | None = None) -> None:
        if params is None:
            params = HOPurificationParams()
        elif isinstance(params, dict):
            params = HOPurificationParams(**params)
        if not isinstance(params, HOPurificationParams):
            raise TypeError("params must be HOPurificationParams, a dict, or None.")
        self.params = params
        if params.N < 1 or params.tf <= 0 or params.n_osc < 2:
            raise ValueError("Require N >= 1, tf > 0, and n_osc >= 2.")
        if abs(params.r) > 1 or params.fock_N < 0 or params.fock_N >= params.n_osc:
            raise ValueError("Invalid initial-state parameters.")
        if params.oscillator_state not in {"fock", "thermal"}:
            raise ValueError("This contained adapter supports oscillator_state='fock' or 'thermal'.")
        if params.oscillator_state == "thermal" and (params.theta_B is None or params.theta_B <= 0):
            raise ValueError("A positive theta_B is required for a thermal oscillator.")
        if params.target_A_state not in {"e", "g"} or params.control_weight < 0:
            raise ValueError("Invalid target_A_state or control_weight.")

        self.dt = float(params.tf) / int(params.N)
        self.dim = 2 * int(params.n_osc)
        self._build_operators()
        self.rho0 = self._initial_state()
        self.target = self.proj_e if params.target_A_state == "e" else self.proj_g
        self.Pi = np.kron(self.target, self.Iosc)
        self._cached_controls: np.ndarray | None = None
        self._cache: dict[str, Any] = {}

    def _build_operators(self) -> None:
        I2 = np.eye(2, dtype=DTYPE)
        sx = np.array([[0, 1], [1, 0]], dtype=DTYPE)
        sz = np.array([[1, 0], [0, -1]], dtype=DTYPE)
        self.e = np.array([1, 0], dtype=DTYPE)
        self.ground = np.array([0, 1], dtype=DTYPE)
        self.proj_e = np.outer(self.e, self.e.conj())
        self.proj_g = np.outer(self.ground, self.ground.conj())
        self.Iosc = np.eye(self.params.n_osc, dtype=DTYPE)
        a = np.diag(np.sqrt(np.arange(1, self.params.n_osc)), 1).astype(DTYPE)
        adag = a.conj().T
        number = adag @ a
        self.Hc = 0.5 * np.kron(sz, self.Iosc)
        self.H0 = (
            self.params.omega_B * np.kron(I2, number)
            + self.params.g * (np.kron(sx, self.Iosc) @ np.kron(I2, adag + a))
        )

    def _initial_state(self) -> np.ndarray:
        rho_a = 0.5 * ((1 - self.params.r) * self.proj_e + (1 + self.params.r) * self.proj_g)
        if self.params.oscillator_state == "fock":
            ket = np.zeros(self.params.n_osc, dtype=DTYPE)
            ket[self.params.fock_N] = 1.0
            rho_b = np.outer(ket, ket.conj())
        else:
            weights = np.exp(-float(self.params.theta_B) * np.arange(self.params.n_osc))
            weights /= weights.sum()
            rho_b = np.diag(weights).astype(DTYPE)
        return np.kron(rho_a, rho_b)

    def control_spec(self) -> ControlSpec:
        return ControlSpec(keys=("omega_A",), control_dim=self.params.N, dtype=float, dt=self.dt)

    def system_spec(self) -> dict[str, Any]:
        return basic_system_spec(
            "ho-purification-tracker",
            name="Qubit-oscillator purification tracker system",
            system_type="qubit-composite",
            description="Closed qubit-oscillator purification adapter with an exact discrete gradient.",
            control_spec=self.control_spec(),
            parameters={"primary": asdict(self.params), "secondary": {"control_weight": self.params.control_weight}},
            metrics={
                "fidelity": {"label": "Fidelity", "role": "quality"},
                "infidelity": {"label": "Infidelity", "role": "quality"},
                "control_fluence": {"label": "Control fluence", "role": "resource"},
                "max_abs_omega_A": {"label": "Maximum control amplitude", "role": "diagnostic"},
            },
            derivatives={
                "gradient": {
                    "target": "J",
                    "hook": "gradient",
                    "source": "analytical",
                    "convention": "exact discrete dJ/du from Frechet propagator derivatives",
                    "derivation_reference": "test_contained/ho_purification/PROBLEM_CONTEXT.md",
                }
            },
            visualizations={
                "bloch": {
                    "provider": "plot_data",
                    "status": "implemented",
                    "description": "Generic named layers of reduced-qubit Bloch coordinates",
                    "automatic": False,
                }
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": type(self).__name__,
            "model": "closed qubit-oscillator Rabi purification",
            "source": "test_contained/ho_purification/PROBLEM_CONTEXT.md",
            "discretization": "left piecewise-constant controls; N propagator steps",
        }


    def with_secondary(self, **updates: Any) -> "HOPurificationTrackerSystem":
        return type(self)(replace(self.params, **updates))

    def _omega(self, controls: Controls) -> np.ndarray:
        if controls.spec != self.control_spec():
            raise ValueError("Controls do not match this system's ControlSpec.")
        omega = np.asarray(controls.channel("omega_A"), dtype=float)
        if not np.all(np.isfinite(omega)):
            raise ValueError("omega_A must be finite.")
        return omega

    def _propagate(self, omega: np.ndarray) -> dict[str, Any]:
        if self._cached_controls is not None and np.array_equal(omega, self._cached_controls):
            return self._cache
        rho = np.empty((self.params.N + 1, self.dim, self.dim), dtype=DTYPE)
        steps = np.empty((self.params.N, self.dim, self.dim), dtype=DTYPE)
        rho[0] = self.rho0
        for j, value in enumerate(omega):
            steps[j] = expm(-1j * (self.H0 + value * self.Hc) * self.dt)
            rho[j + 1] = steps[j] @ rho[j] @ steps[j].conj().T
        fidelity = float(np.real(np.trace(self.Pi @ rho[-1])))
        fluence = float(0.5 * self.dt * np.dot(omega, omega))
        self._cached_controls = omega.copy()
        self._cache = {"rho": rho, "steps": steps, "fidelity": fidelity, "fluence": fluence}
        return self._cache

    def evaluate(self, controls: Controls) -> dict[str, float]:
        omega = self._omega(controls)
        cache = self._propagate(omega)
        fidelity = float(np.clip(cache["fidelity"], 0.0, 1.0))
        fluence = float(cache["fluence"])
        return {
            "J": float(1.0 - fidelity + self.params.control_weight * fluence),
            "fidelity": fidelity,
            "infidelity": float(1.0 - fidelity),
            "control_fluence": fluence,
            "max_abs_omega_A": float(np.max(np.abs(omega))),
        }

    def plot_data(self, controls: Controls, *, kind: str, **options: Any) -> PlotData:
        """Return generic Bloch-coordinate data for the reduced qubit state."""

        del options
        if kind != "bloch":
            raise KeyError(f"Unsupported plot-data kind {kind!r}.")
        cache = self._propagate(self._omega(controls))
        rho = np.asarray(cache["rho"], dtype=DTYPE)
        reshaped = rho.reshape(rho.shape[0], 2, self.params.n_osc, 2, self.params.n_osc)
        reduced = np.trace(reshaped, axis1=2, axis2=4)
        sy = np.array([[0, -1j], [1j, 0]], dtype=DTYPE)
        points = np.column_stack(
            [
                np.real(np.einsum("nij,ji->n", reduced, axis))
                for axis in (
                    np.array([[0, 1], [1, 0]], dtype=DTYPE),
                    sy,
                    np.array([[1, 0], [0, -1]], dtype=DTYPE),
                )
            ]
        )
        return PlotData(
            kind="bloch",
            layers=(
                PlotLayer(
                    name="reduced_state_trajectory",
                    coordinates={
                        "points": points,
                        "parameter": np.linspace(0.0, self.params.tf, self.params.N + 1),
                    },
                    label="Reduced-state trajectory",
                ),
            ),
            annotations={"initial": {"point": points[0], "label": "initial"}},
            metadata={"parameter_label": "time", "source": "system forward propagation"},
        )

    def gradient(self, controls: Controls) -> Controls:
        omega = self._omega(controls)
        cache = self._propagate(omega)
        rho = cache["rho"]
        steps = cache["steps"]
        chi = np.empty_like(rho)
        chi[-1] = self.Pi
        for j in range(self.params.N - 1, -1, -1):
            chi[j] = steps[j].conj().T @ chi[j + 1] @ steps[j]
        grad = np.empty(self.params.N, dtype=float)
        for j, value in enumerate(omega):
            A = -1j * (self.H0 + value * self.Hc) * self.dt
            dU = expm_frechet(A, -1j * self.Hc * self.dt, compute_expm=False)
            dF = 2.0 * np.real(np.trace(chi[j + 1] @ dU @ rho[j] @ steps[j].conj().T))
            grad[j] = -float(dF) + self.params.control_weight * self.dt * value
        return Controls.from_dict(self.control_spec(), {"omega_A": grad}, name="exact_gradient_J")

    def static_parameters(self) -> dict[str, Any]:
        return asdict(self.params)
