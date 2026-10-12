"""Contained OLGS system for second-order universally robust state transfer.

The objective is the Method 1 scalar-channel penalty from the second-order PMP
formulation.  The gradient differentiates the chosen discrete objective exactly
through each piecewise-constant propagator.  This intentionally differs from
the old source's continuous-time correction signal: it is suitable for the
generic optimizer's ``dJ/du`` contract and is checked by the trial runner.
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
class SecondOrderPMPParams:
    tf: float = 1.0
    # Preserve the original Method 1 grid: 1001 stored time nodes yield
    # 1000 propagator intervals with dt = 1e-3 at tf = 1.
    N: int = 1001
    control_weight: float = 0.0
    initial_basis: int = 0
    target_basis: int = 1
    active_axes: tuple[str, ...] = ("x", "y", "z")


class SecondOrderPMPSystem:
    """Three-control qubit transfer with a second-order robustness penalty."""

    _axis_names = ("x", "y", "z")

    def __init__(self, params: SecondOrderPMPParams | dict[str, Any] | None = None) -> None:
        if params is None:
            params = SecondOrderPMPParams()
        elif isinstance(params, dict):
            params = SecondOrderPMPParams(**params)
        if not isinstance(params, SecondOrderPMPParams):
            raise TypeError("params must be SecondOrderPMPParams, a dict, or None.")
        if params.N < 2 or params.tf <= 0 or params.control_weight < 0:
            raise ValueError("Require N >= 2, tf > 0, and control_weight >= 0.")
        if params.initial_basis not in (0, 1) or params.target_basis not in (0, 1):
            raise ValueError("initial_basis and target_basis must be 0 or 1.")
        if not params.active_axes or not set(params.active_axes).issubset(self._axis_names):
            raise ValueError("active_axes must be a non-empty subset of x, y, z.")
        self.params = params
        self.dt = float(params.tf) / (int(params.N) - 1)
        self.sigma = np.array(
            [
                [[0, 1], [1, 0]],
                [[0, -1j], [1j, 0]],
                [[1, 0], [0, -1]],
            ],
            dtype=DTYPE,
        )
        self.psi0 = self._basis(params.initial_basis)
        self.target = self._basis(params.target_basis)
        self.psi_perp0 = np.array([-self.psi0[1].conjugate(), self.psi0[0].conjugate()], dtype=DTYPE)
        self._cached_controls: np.ndarray | None = None
        self._cache: dict[str, Any] = {}

    @staticmethod
    def _basis(index: int) -> np.ndarray:
        state = np.zeros(2, dtype=DTYPE)
        state[index] = 1.0
        return state

    def control_spec(self) -> ControlSpec:
        return ControlSpec(keys=("ux", "uy", "uz"), control_dim=self.params.N, dtype=float, dt=self.dt)

    def system_spec(self) -> dict[str, Any]:
        return basic_system_spec(
            "second-order-pmp",
            name="Second-order robust qubit transfer",
            system_type="qubit",
            description="Three-control qubit state transfer with an analytical discrete gradient.",
            control_spec=self.control_spec(),
            parameters={"primary": asdict(self.params), "secondary": {"control_weight": self.params.control_weight}},
            metrics={
                "fidelity": {"label": "Fidelity", "role": "quality"},
                "infidelity": {"label": "Infidelity", "role": "quality"},
                "robustness_penalty": {"label": "Robustness penalty", "role": "resource"},
                "control_energy": {"label": "Control energy", "role": "resource"},
                "abs_Fx": {"label": "Absolute Fx channel", "role": "diagnostic"},
                "abs_Fy": {"label": "Absolute Fy channel", "role": "diagnostic"},
                "abs_Fz": {"label": "Absolute Fz channel", "role": "diagnostic"},
            },
            derivatives={
                "gradient": {
                    "target": "J",
                    "hook": "gradient",
                    "source": "analytical",
                    "convention": "exact discrete dJ/du from Frechet propagator derivatives",
                    "derivation_reference": "test_contained/second_order_pmp/PROBLEM_CONTEXT.md",
                }
            },
            visualizations={
                "bloch": {
                    "provider": "plot_data",
                    "status": "implemented",
                    "description": "Generic named layers of Bloch coordinates",
                    "automatic": False,
                }
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": type(self).__name__,
            "model": "second-order universally robust two-level state transfer",
            "source": "test_contained/second_order_pmp/PROBLEM_CONTEXT.md",
            "discretization": "Method 1 right-grid storage: N time nodes and N-1 propagator intervals",
        }


    def with_secondary(self, **updates: Any) -> "SecondOrderPMPSystem":
        return type(self)(replace(self.params, **updates))

    def _matrix(self, controls: Controls) -> np.ndarray:
        if controls.spec != self.control_spec():
            raise ValueError("Controls do not match this system's ControlSpec.")
        matrix = np.asarray(controls.as_matrix(copy=False), dtype=float)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Controls must be finite.")
        return matrix

    def _propagate(self, matrix: np.ndarray) -> dict[str, Any]:
        if self._cached_controls is not None and np.array_equal(matrix, self._cached_controls):
            return self._cache
        # This is deliberately source-compatible with Method 1: controls have
        # one value per stored node, while only entries 0..N-2 propagate the
        # N-1 intervals.  The terminal node contributes to the channel rule.
        psi = np.empty((self.params.N, 2), dtype=DTYPE)
        perp = np.empty_like(psi)
        steps = np.empty((self.params.N - 1, 2, 2), dtype=DTYPE)
        psi[0], perp[0] = self.psi0, self.psi_perp0
        for j in range(self.params.N - 1):
            H = 0.5 * np.tensordot(matrix[:, j], self.sigma, axes=1)
            steps[j] = expm(-1j * H * self.dt)
            psi[j + 1] = steps[j] @ psi[j]
            perp[j + 1] = steps[j] @ perp[j]
        channels = np.empty(3, dtype=DTYPE)
        for axis in range(3):
            channels[axis] = self.dt * sum(np.vdot(perp[j], self.sigma[axis] @ psi[j]) for j in range(self.params.N))
        overlap = np.vdot(self.target, psi[-1])
        fidelity = float(abs(overlap) ** 2)
        active = np.array([name in self.params.active_axes for name in self._axis_names])
        robustness = float(np.sum(abs(channels[active]) ** 2))
        # Match the legacy Method 1 reporting convention exactly.
        energy = float(self.dt * np.sum(matrix * matrix))
        self._cached_controls = matrix.copy()
        self._cache = {
            "psi": psi, "perp": perp, "steps": steps, "channels": channels,
            "overlap": overlap, "fidelity": fidelity, "robustness": robustness, "energy": energy,
        }
        return self._cache

    def evaluate(self, controls: Controls) -> dict[str, float]:
        matrix = self._matrix(controls)
        cache = self._propagate(matrix)
        channels = cache["channels"]
        return {
            "J": float(1.0 - cache["fidelity"] + cache["robustness"] + self.params.control_weight * cache["energy"]),
            "fidelity": float(cache["fidelity"]),
            "infidelity": float(1.0 - cache["fidelity"]),
            "robustness_penalty": float(cache["robustness"]),
            "control_energy": float(cache["energy"]),
            "abs_Fx": float(abs(channels[0])),
            "abs_Fy": float(abs(channels[1])),
            "abs_Fz": float(abs(channels[2])),
        }

    def plot_data(self, controls: Controls, *, kind: str, **options: Any) -> PlotData:
        """Return explicit generic Bloch-coordinate data for the propagated state."""

        del options
        if kind != "bloch":
            raise KeyError(f"Unsupported plot-data kind {kind!r}.")
        cache = self._propagate(self._matrix(controls))
        psi = np.asarray(cache["psi"], dtype=DTYPE)
        points = np.column_stack(
            [np.real(np.einsum("ni,ij,nj->n", psi.conj(), axis, psi)) for axis in self.sigma]
        )
        target = np.array([np.real(np.vdot(self.target, axis @ self.target)) for axis in self.sigma])
        return PlotData(
            kind="bloch",
            layers=(
                PlotLayer(
                    name="state_trajectory",
                    coordinates={"points": points, "parameter": np.linspace(0.0, self.params.tf, self.params.N)},
                    label="State trajectory",
                ),
            ),
            annotations={
                "initial": {"point": points[0], "label": "initial"},
                "target": {"point": target, "label": "target"},
            },
            metadata={"parameter_label": "time", "source": "system forward propagation"},
        )

    def gradient(self, controls: Controls) -> Controls:
        """Exact O(N) discrete-adjoint derivative of the source-aligned objective.

        The state/channel convention matches Method 1: N stored nodes, N-1
        propagators, and a channel sum that includes the terminal node.  Each
        interval uses one 2x2 Frechet derivative per control axis; no finite
        differences or legacy continuous-time correction signal enter the update.
        """
        matrix = self._matrix(controls)
        cache = self._propagate(matrix)
        psi, perp, steps = cache["psi"], cache["perp"], cache["steps"]
        channels, overlap = cache["channels"], cache["overlap"]
        active = np.array([name in self.params.active_axes for name in self._axis_names], dtype=bool)
        gradient = 2.0 * self.params.control_weight * self.dt * matrix

        def channel_adjoint(node: int) -> tuple[np.ndarray, np.ndarray]:
            """Return d robust-penalty / d(psi, perp) at one stored node."""
            adjoint_psi = np.zeros(2, dtype=DTYPE)
            adjoint_perp = np.zeros(2, dtype=DTYPE)
            for axis in range(3):
                if active[axis]:
                    adjoint_psi += self.dt * channels[axis] * (self.sigma[axis] @ perp[node])
                    # q enters the channel through q^dagger, hence its real
                    # adjoint carries the conjugated channel coefficient.
                    adjoint_perp += self.dt * np.conj(channels[axis]) * (self.sigma[axis] @ psi[node])
            return adjoint_psi, adjoint_perp

        # For real J, dJ = 2 Re[a^dagger d state].  Start with the terminal
        # infidelity adjoint, then add the terminal channel contribution.
        adjoint_psi = -overlap * self.target
        adjoint_perp = np.zeros(2, dtype=DTYPE)
        terminal_psi, terminal_perp = channel_adjoint(self.params.N - 1)
        adjoint_psi += terminal_psi
        adjoint_perp += terminal_perp

        for pixel in range(self.params.N - 2, -1, -1):
            H = 0.5 * np.tensordot(matrix[:, pixel], self.sigma, axes=1)
            generator = -1j * H * self.dt
            for axis in range(3):
                direction = -0.5j * self.sigma[axis] * self.dt
                dU = expm_frechet(generator, direction, compute_expm=False)
                gradient[axis, pixel] += 2.0 * np.real(
                    np.vdot(adjoint_psi, dU @ psi[pixel])
                    + np.vdot(adjoint_perp, dU @ perp[pixel])
                )
            adjoint_psi = steps[pixel].conj().T @ adjoint_psi
            adjoint_perp = steps[pixel].conj().T @ adjoint_perp
            node_psi, node_perp = channel_adjoint(pixel)
            adjoint_psi += node_psi
            adjoint_perp += node_perp
        return Controls.from_matrix(self.control_spec(), gradient, name="exact_discrete_gradient_J")

    def robust_fidelity(self, controls: Controls, *, alpha: float, directions: np.ndarray) -> dict[str, float]:
        """Explicit post-run finite-perturbation diagnostic; never used by evaluate."""
        matrix = self._matrix(controls)
        normalized = np.asarray(directions, dtype=float)
        if normalized.ndim != 2 or normalized.shape[1] != 3:
            raise ValueError("directions must have shape (n, 3).")
        values: list[float] = []
        for vector in normalized:
            vector = vector / np.linalg.norm(vector)
            state = self.psi0.copy()
            for j in range(self.params.N - 1):
                H = 0.5 * np.tensordot(matrix[:, j], self.sigma, axes=1) + float(alpha) * np.tensordot(vector, self.sigma, axes=1)
                state = expm(-1j * H * self.dt) @ state
            values.append(float(abs(np.vdot(self.target, state)) ** 2))
        return {"robust_fidelity_mean": float(np.mean(values)), "robust_fidelity_min": float(np.min(values)), "robust_samples": int(len(values))}

    def static_parameters(self) -> dict[str, Any]:
        return asdict(self.params)
