"""Temporary v3-style universal fourth-order robust-control system fixture.

Why this file exists
--------------------
Future optimizer phases should be checked against a system that resembles the real
downstream problem, not only against abstract quadratic examples.  This fixture is
based on the structure of
``/Users/billabobz/phd/CODES/universal_robust_control_4th/model_note/main.tex``:

    controls: ux, uy, uz on an endpoint-including grid
    objective: nominal infidelity + lambda2*|F(T)|^2 + lambda4*|C_sym(T)|^2 + energy
    residuals: the three F constraints plus six independent C_sym constraints
    gradient: analytical derivative of the implemented discrete objective
    simulation: vectorized finite-alpha direction-fidelity diagnostics

It is intentionally a lightweight surrogate, not the full physical qubit propagator.
The real system has matrix exponentials and adjoint propagation; this fixture keeps
the same optimizer-facing shape while staying cheap enough for unit tests.

How it fits the architecture
----------------------------
- Phase 6 public API tests import this as a temporary downstream-like system.
- future optimizer tests can reuse it until real integration tests are connected.
- the fixture exercises complex metrics, vectorized robust-direction simulation, and
  system-owned curriculum weights.

What this file deliberately does not do
---------------------------------------
It does not replace the real fourth-order model, prove physical fidelity, or implement
the full PMP/costate equations.  It only validates that the optimizer library can work
with a realistic system contract.

Reviewer invariants
-------------------
- public hooks are exactly ``control_spec/evaluate/gradient/with_params``.
- changing ``lambda2/lambda4/energy_weight`` changes system-owned objective terms.
- the gradient is analytical for this fixture's discrete objective.
- robust direction evaluation is vectorized over directions by default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from optimizer.controls import ControlSpec, Controls


CONTROL_KEYS = ("ux", "uy", "uz")


@dataclass(frozen=True)
class UniversalFourthOrderParams:
    """System-owned weights and grid parameters for the fixture."""

    tau: float = 1.0
    N: int = 33
    target_area: float = 1.0
    infidelity_weight: float = 1.0
    lambda2: float = 0.5
    lambda4: float = 0.1
    energy_weight: float = 1.0e-3
    robust_n_dirs: int = 250
    robust_seed: int | None = 12345


def _sample_unit_vectors_spherical(n: int, *, seed: int | None) -> np.ndarray:
    """Vectorized deterministic S2 sampling used by ``simulate``."""

    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0, size=int(n))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=int(n))
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.stack((radius * np.cos(phi), radius * np.sin(phi), z), axis=1)


def _normalize_directions(directions: np.ndarray) -> np.ndarray:
    """Return finite unit direction vectors with shape ``(n_dirs, 3)``."""

    dirs = np.asarray(directions, dtype=float)
    if dirs.ndim == 1:
        dirs = dirs.reshape(1, 3)
    if dirs.ndim != 2 or dirs.shape[1] != 3:
        raise ValueError("directions must have shape (n_dirs, 3).")
    norms = np.linalg.norm(dirs, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("directions must be finite nonzero vectors.")
    return dirs / norms[:, None]


class TemporaryUniversalFourthOrderSystem:
    """V3-compatible fixture with fourth-order robust-control metric structure."""

    # ------------------------------------------------------------------
    # Construction and system contract
    # ------------------------------------------------------------------

    def __init__(self, params: UniversalFourthOrderParams | None = None, **updates: Any) -> None:
        base = params or UniversalFourthOrderParams()
        self.params = replace(base, **updates) if updates else base
        self._validate_params()
        self.dt = float(self.params.tau) / float(self.params.N - 1)
        self._spec = ControlSpec(
            CONTROL_KEYS,
            self.params.N,
            dtype=float,
            dt=self.dt,
            meta={
                "fixture": "temporary universal fourth-order robust system",
                "control_convention": "N endpoint samples; first N-1 samples drive intervals",
            },
        )

    def control_spec(self) -> ControlSpec:
        """Return the v3 control layout expected by this system."""

        return self._spec

    def evaluate(self, controls: Controls) -> dict[str, Any]:
        """Return objective metrics for controls."""

        parts = self._components(controls)
        F = parts["F_terminal"]
        C = parts["C_terminal"]
        Csym = parts["C_sym_terminal"]
        F_norm2 = float(np.sum(F * F))
        C_sym_norm2 = float(np.sum(Csym * Csym))
        energy = float(self.dt * np.sum(parts["matrix"] * parts["matrix"]))
        infidelity = F_norm2

        J_infidelity = float(self.params.infidelity_weight) * infidelity
        J_second_order = float(self.params.lambda2) * F_norm2
        J_fourth_order = float(self.params.lambda4) * C_sym_norm2
        J_energy = float(self.params.energy_weight) * energy
        J = float(J_infidelity + J_second_order + J_fourth_order + J_energy)
        fidelity = float(max(0.0, 1.0 - infidelity))

        return {
            "J": J,
            "J_infidelity": J_infidelity,
            "J_second_order": J_second_order,
            "J_fourth_order": J_fourth_order,
            "J_energy": J_energy,
            "fidelity": fidelity,
            "infidelity": float(infidelity),
            "energy": energy,
            "F_terminal": F.astype(np.complex128),
            "F_norm2": F_norm2,
            "C_terminal": C.astype(np.complex128),
            "C_sym_terminal": Csym.astype(np.complex128),
            "C_sym_norm2": C_sym_norm2,
            "terminal_overlap": complex(np.sqrt(fidelity), 0.0),
            "system_params": asdict(self.params),
        }

    def gradient(self, controls: Controls) -> Controls:
        """Return the analytical gradient of this fixture's discrete objective."""

        parts = self._components(controls)
        matrix = parts["matrix"]
        active = parts["active"]
        F = parts["F_terminal"]
        F_cum = parts["F_cum"]
        Csym = parts["C_sym_terminal"]
        M = active.shape[1]

        grad = 2.0 * float(self.params.energy_weight) * self.dt * matrix
        active_grad = np.zeros_like(active)

        second_order_weight = float(self.params.infidelity_weight) + float(self.params.lambda2)
        active_grad += 2.0 * second_order_weight * self.dt * F[:, None]

        if float(self.params.lambda4) != 0.0:
            c_sensitivity = 2.0 * Csym
            for axis in range(3):
                direct = self.dt * (c_sensitivity[axis, :] @ F_cum)
                tail_source = c_sensitivity[:, axis] @ active
                tail_after = np.cumsum(tail_source[::-1])[::-1] - tail_source
                active_grad[axis] += float(self.params.lambda4) * (
                    direct + self.dt * self.dt * tail_after
                )

        grad[:, :M] += active_grad
        return Controls.from_matrix(self.control_spec(), grad, name="gradient")

    def with_params(self, **updates: Any) -> "TemporaryUniversalFourthOrderSystem":
        """Return an equivalent fixture with updated objective/grid params."""

        return TemporaryUniversalFourthOrderSystem(replace(self.params, **updates))

    # ------------------------------------------------------------------
    # Optional hooks matching planned downstream needs
    # ------------------------------------------------------------------

    def residuals(self, controls: Controls, *, name: str = "hard") -> np.ndarray:
        """Return hard residual vector ``[F, C_sym independent entries]``."""

        if name != "hard":
            raise ValueError("temporary fixture only provides residuals(name='hard').")
        parts = self._components(controls)
        Csym = parts["C_sym_terminal"]
        return np.asarray(
            [
                *parts["F_terminal"].tolist(),
                Csym[0, 0],
                Csym[1, 1],
                Csym[2, 2],
                Csym[0, 1],
                Csym[0, 2],
                Csym[1, 2],
            ],
            dtype=float,
        )

    def simulate(
        self,
        controls: Controls,
        *,
        alpha: float = 0.1,
        directions: str | np.ndarray | None = "sphere",
        n_dirs: int | None = None,
        seed: int | None = None,
        parallel: bool | None = None,
        max_workers: int | None = None,
    ) -> dict[str, Any]:
        """Vectorized finite-alpha direction-fidelity diagnostic."""

        del max_workers
        parts = self._components(controls)
        dirs = self._direction_set(directions, n_dirs=n_dirs, seed=seed)
        F = parts["F_terminal"]
        C = parts["C_terminal"]
        directional_c = np.einsum("ki,ij,kj->k", dirs, C, dirs)
        perturbed = F[None, :] + float(alpha) * dirs
        infidelity = np.sum(perturbed * perturbed, axis=1) + (float(alpha) ** 4) * (
            directional_c * directional_c
        )
        fidelity = np.exp(-np.maximum(0.0, infidelity))
        worst = int(np.argmin(fidelity))
        best = int(np.argmax(fidelity))
        return {
            **self.evaluate(controls),
            "robust_backend": "vectorized",
            "robust_parallel_requested": bool(parallel) if parallel is not None else False,
            "robust_parallel_used": False,
            "robust_alpha": float(alpha),
            "robust_n_dirs": int(dirs.shape[0]),
            "robust_directions": dirs.copy(),
            "robust_fidelity_per_direction": fidelity.copy(),
            "robust_infidelity_per_direction": (1.0 - fidelity).copy(),
            "robust_G4_per_direction": directional_c.astype(np.complex128),
            "robust_fidelity_mean": float(np.mean(fidelity)),
            "robust_fidelity_min": float(np.min(fidelity)),
            "robust_fidelity_max": float(np.max(fidelity)),
            "robust_worst_direction": dirs[worst].copy(),
            "robust_best_direction": dirs[best].copy(),
        }

    def reference_controls(self, *, amplitude: float | None = None) -> Controls:
        """Return a simple ux-only control near the nominal area target."""

        amp = float(self.params.target_area / self.params.tau if amplitude is None else amplitude)
        matrix = np.zeros(self.control_spec().shape, dtype=float)
        matrix[0, :] = amp
        return Controls.from_matrix(self.control_spec(), matrix, name="reference")

    # ------------------------------------------------------------------
    # Internal vectorized objective helpers
    # ------------------------------------------------------------------

    def _validate_params(self) -> None:
        if float(self.params.tau) <= 0.0:
            raise ValueError("tau must be positive.")
        if int(self.params.N) < 3:
            raise ValueError("N must be >= 3.")
        if int(self.params.robust_n_dirs) <= 0:
            raise ValueError("robust_n_dirs must be positive.")
        for name in ("infidelity_weight", "lambda2", "lambda4", "energy_weight"):
            value = float(getattr(self.params, name))
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and nonnegative.")

    def _matrix(self, controls: Controls) -> np.ndarray:
        if not isinstance(controls, Controls):
            raise TypeError("controls must be a Controls object.")
        if controls.spec.keys != self.control_spec().keys or controls.spec.control_dim != self.params.N:
            raise ValueError("controls do not match the fixture control_spec.")
        return controls.as_matrix(copy=False)

    def _components(self, controls: Controls) -> dict[str, np.ndarray]:
        """Compute vectorized terminal channel quantities for the fixture."""

        matrix = self._matrix(controls)
        active = matrix[:, :-1]
        target_rate = np.array([self.params.target_area / self.params.tau, 0.0, 0.0], dtype=float)
        f_samples = active - target_rate[:, None]
        inclusive = self.dt * np.cumsum(f_samples, axis=1)
        F_cum = np.concatenate((np.zeros((3, 1), dtype=float), inclusive[:, :-1]), axis=1)
        F_terminal = self.dt * np.sum(f_samples, axis=1)
        C_terminal = self.dt * active @ F_cum.T
        C_sym = 0.5 * (C_terminal + C_terminal.T)
        return {
            "matrix": matrix,
            "active": active,
            "f_samples": f_samples,
            "F_cum": F_cum,
            "F_terminal": F_terminal,
            "C_terminal": C_terminal,
            "C_sym_terminal": C_sym,
        }

    def _direction_set(
        self,
        directions: str | np.ndarray | None,
        *,
        n_dirs: int | None,
        seed: int | None,
    ) -> np.ndarray:
        if directions is None:
            count = int(self.params.robust_n_dirs if n_dirs is None else n_dirs)
            seed_eff = self.params.robust_seed if seed is None else seed
            return _sample_unit_vectors_spherical(count, seed=seed_eff)
        if isinstance(directions, str):
            if directions == "sphere":
                count = int(self.params.robust_n_dirs if n_dirs is None else n_dirs)
                seed_eff = self.params.robust_seed if seed is None else seed
                return _sample_unit_vectors_spherical(count, seed=seed_eff)
            if directions in {"axes", "xyz"}:
                return np.eye(3, dtype=float)
            raise ValueError(f"unknown direction set {directions!r}.")
        return _normalize_directions(np.asarray(directions, dtype=float))
