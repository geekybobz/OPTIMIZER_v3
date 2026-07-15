"""Universal fourth-order robust two-level state-transfer model.

This module implements the channel-matrix formulation in ``model_note/main.tex``.
Training uses only the nominal Hamiltonian and accumulates the perturbation
response channels

    F_k(T) = int <psi_perp|sigma_k|psi_0> dt
    C_kl(T) = int <psi_0|sigma_k|psi_0> F_l dt

for k,l in {x,y,z}.  No fixed perturbation direction V is chosen during
training.  A candidate pulse is universally fourth-order robust when the
nominal transfer is exact, all F_k(T) vanish, and the transpose-symmetric
part of C(T) vanishes.

The discrete grid follows the existing ST1 article implementation: ``N`` is
the number of endpoint-including time samples, ``dt=tau/(N-1)``, controls have
length ``N``, and only the first ``N-1`` control samples generate propagation
intervals.  The final control sample is retained so saved article pulses can be
evaluated without changing their quadrature convention.

v3 adapter note
---------------
This file is intentionally local to ``systems/universal_robust_4th`` rather than
inside ``optimizer/``.  It is a real downstream-style ``system.py`` used to test the
library contract:

    control_spec()        describe the ux/uy/uz control layout
    evaluate(controls)    return objective metrics, including scalar J
    gradient(controls)    return analytical dJ/du as Controls
    residuals(...)        expose hard/fourth-order conditions for repair tools
    jacobian(...)         expose analytical seeded-adjoint residual Jacobians

The physics and adjoint formulas come from the previous fourth-order implementation;
only the optimizer-facing names and local reference-control helpers are adapted to v3.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, is_dataclass
from itertools import repeat
from pathlib import Path
from typing import Any

import numpy as np

from optimizer.controls import ControlSpec, Controls
from optimizer.guesses import fourier_guess as _v3_fourier_guess

try:
    from scipy.linalg import expm as _scipy_expm
    from scipy.linalg import expm_frechet as _scipy_expm_frechet
except Exception:  # pragma: no cover - SciPy is expected in the optimizer env.
    _scipy_expm = None
    _scipy_expm_frechet = None


DTYPE = np.complex128
CHANNEL_KEYS = ("x", "y", "z")
CONTROL_KEYS = ("ux", "uy", "uz")
MODULE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = MODULE_DIR / "reference"
FOURTH_ORDER_BEST_CONTROLS = REFERENCE_DIR / "fourth_order_best_controls.npz"


def _mat_expm(matrix: np.ndarray) -> np.ndarray:
    """Small dense matrix exponential with a NumPy fallback."""
    if _scipy_expm is not None:
        return _scipy_expm(matrix)
    vals, vecs = np.linalg.eig(matrix)
    return vecs @ np.diag(np.exp(vals)) @ np.linalg.inv(vecs)


def _mat_expm_frechet(matrix: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Frechet derivative of exp(matrix) in ``direction``."""
    if _scipy_expm_frechet is not None:
        return _scipy_expm_frechet(matrix, direction, compute_expm=False)
    n = matrix.shape[0]
    block = np.zeros((2 * n, 2 * n), dtype=DTYPE)
    block[:n, :n] = matrix
    block[:n, n:] = direction
    block[n:, n:] = matrix
    return _mat_expm(block)[:n, n:]


def _su2_step_apply(
    psi0: complex,
    psi1: complex,
    hx: float,
    hy: float,
    hz: float,
    dt: float,
) -> tuple[complex, complex]:
    """Apply exp(-i * (hx*sx + hy*sy + hz*sz) * dt) to one qubit state."""
    r = float(np.sqrt(hx * hx + hy * hy + hz * hz))
    if r > 1.0e-14:
        c = float(np.cos(r * dt))
        s_over_r = float(np.sin(r * dt) / r)
    else:
        c = 1.0
        s_over_r = float(dt)

    u00 = c - 1j * s_over_r * hz
    u11 = c + 1j * s_over_r * hz
    u01 = -1j * s_over_r * (hx - 1j * hy)
    u10 = -1j * s_over_r * (hx + 1j * hy)
    return u00 * psi0 + u01 * psi1, u10 * psi0 + u11 * psi1


def _normalize_direction_array(direction: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    d = np.asarray(direction, dtype=float).reshape(3)
    norm = float(np.linalg.norm(d))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Perturbation direction must be a finite nonzero 3-vector.")
    return d / norm


def _robust_worker_direction(
    direction: np.ndarray,
    alpha: float,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    dt: float,
    psi_initial: np.ndarray,
    psi_target: np.ndarray,
    C_terminal: np.ndarray,
) -> tuple[float, complex, complex]:
    d = _normalize_direction_array(direction)
    p0 = complex(psi_initial[0])
    p1 = complex(psi_initial[1])
    ax, ay, az = float(alpha) * d
    for n in range(int(ux.shape[0]) - 1):
        p0, p1 = _su2_step_apply(
            p0,
            p1,
            0.5 * float(ux[n]) + float(ax),
            0.5 * float(uy[n]) + float(ay),
            0.5 * float(uz[n]) + float(az),
            float(dt),
        )
    overlap = np.conjugate(psi_target[0]) * p0 + np.conjugate(psi_target[1]) * p1
    g4 = complex(d @ np.asarray(C_terminal, dtype=DTYPE) @ d)
    return float(abs(overlap) ** 2), complex(overlap), g4


def sample_unit_vectors_spherical(
    n: int,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample unit vectors uniformly on S^2."""
    if int(n) <= 0:
        raise ValueError("n must be positive.")
    if rng is None:
        rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0, size=int(n))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=int(n))
    s = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.stack((s * np.cos(phi), s * np.sin(phi), z), axis=1)


def load_control_npz(
    path: str | Path,
    *,
    drop_endpoint: bool = False,
    target_N: int | None = None,
) -> dict[str, np.ndarray]:
    """Load saved ``ux,uy,uz`` controls from a paper artifact ``.npz``.

    The article controls are stored on a time grid with the endpoint included.
    This model keeps that endpoint by default.  ``drop_endpoint=True`` is kept
    only for comparing against older slice-grid experiments.
    """
    data = np.load(Path(path), allow_pickle=True)
    if "u" in data.files:
        arr = np.asarray(data["u"], dtype=float)
        if arr.shape[0] != 3:
            raise ValueError(f"Expected saved 'u' to have shape (3, n), got {arr.shape}.")
        ux, uy, uz = arr[0], arr[1], arr[2]
    else:
        missing = [key for key in CONTROL_KEYS if key not in data.files]
        if missing:
            raise KeyError(f"Missing control arrays in {path}: {missing}")
        ux = np.asarray(data["ux"], dtype=float)
        uy = np.asarray(data["uy"], dtype=float)
        uz = np.asarray(data["uz"], dtype=float)

    if drop_endpoint:
        ux, uy, uz = ux[:-1], uy[:-1], uz[:-1]

    if target_N is not None:
        n = int(target_N)
        if ux.shape != (n,) or uy.shape != (n,) or uz.shape != (n,):
            raise ValueError(
                "Loaded controls have shape "
                f"{ux.shape}, {uy.shape}, {uz.shape}; expected ({n},)."
            )
    return {"ux": ux.copy(), "uy": uy.copy(), "uz": uz.copy()}


@dataclass
class SystemParams:
    """Parameters for the universal fourth-order robust two-level model."""

    tau: float = 1.0
    N: int = 1001

    infidelity_weight: float = 1.0
    lambda2: float = 1.0
    lambda4: float = 1.0
    energy_weight: float = 0.0

    psi0: tuple[complex, complex] = (1.0 + 0.0j, 0.0 + 0.0j)
    psi_target: tuple[complex, complex] = (0.0 + 0.0j, 1.0 + 0.0j)
    psi_perp_seed: tuple[complex, complex] | None = None

    robust_n_dirs: int = 250
    robust_pair_n_dirs: int = 120
    robust_alpha_default: float | None = None
    robust_seed: int | None = 12345
    robust_parallel: bool = True
    robust_leave_cores_free: int = 2


class FourthOrderUniversalRobustSystem:
    """Optimizer-compatible universal fourth-order robust state-transfer model."""

    def __init__(
        self,
        params: dict[str, Any] | SystemParams | None = None,
        **param_overrides: Any,
    ) -> None:
        self.params = self._coerce_params(params, param_overrides)
        self._validate_params()

        self.N = int(self.params.N)
        self.dt = float(self.params.tau) / float(self.N - 1)
        self.tgrid = np.linspace(0.0, float(self.params.tau), self.N, dtype=float)
        self.control_tgrid = self.tgrid.copy()

        self.I2, self.sx, self.sy, self.sz = self._paulis()
        self.paulis = np.stack((self.sx, self.sy, self.sz), axis=0)
        self.control_generators = 0.5 * self.paulis

        self.psi_initial = self._normalize_state(self.params.psi0, label="psi0")
        self.psi_target = self._normalize_state(self.params.psi_target, label="psi_target")
        if self.params.psi_perp_seed is None:
            self.psi_perp_seed = self._orthogonal_companion(self.psi_initial)
        else:
            self.psi_perp_seed = self._normalize_state(
                self.params.psi_perp_seed,
                label="psi_perp_seed",
            )
            overlap = complex(np.vdot(self.psi_initial, self.psi_perp_seed))
            if abs(overlap) > 1.0e-10:
                raise ValueError("psi_perp_seed must be orthogonal to psi0.")

        self.state: dict[str, Any] = {}

    @staticmethod
    def _coerce_params(
        params: dict[str, Any] | SystemParams | None,
        overrides: dict[str, Any] | None = None,
    ) -> SystemParams:
        overrides = {} if overrides is None else dict(overrides)
        if params is None:
            return SystemParams(**overrides)
        if isinstance(params, SystemParams):
            payload = asdict(params)
            payload.update(overrides)
            return SystemParams(**payload)
        if isinstance(params, dict):
            payload = dict(params)
            payload.update(overrides)
            return SystemParams(**payload)
        if is_dataclass(params):
            payload = asdict(params)
            payload.update(overrides)
            return SystemParams(**payload)
        raise TypeError("params must be None, dict, SystemParams, or compatible dataclass.")

    def _validate_params(self) -> None:
        p = self.params
        if int(p.N) < 2:
            raise ValueError("N must be >= 2 for endpoint-sampled propagation.")
        if not np.isfinite(float(p.tau)) or float(p.tau) <= 0.0:
            raise ValueError("tau must be finite and positive.")
        for name in ("infidelity_weight", "lambda2", "lambda4", "energy_weight"):
            value = float(getattr(p, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if int(p.robust_n_dirs) <= 0:
            raise ValueError("robust_n_dirs must be positive.")
        if int(p.robust_pair_n_dirs) <= 0:
            raise ValueError("robust_pair_n_dirs must be positive.")
        if int(p.robust_leave_cores_free) < 0:
            raise ValueError("robust_leave_cores_free must be nonnegative.")

    @staticmethod
    def _paulis(dtype=DTYPE) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
        sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=dtype)
        sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=dtype)
        return np.eye(2, dtype=dtype), sx, sy, sz

    @staticmethod
    def _normalize_state(
        values: tuple[complex, complex] | list[complex] | np.ndarray,
        *,
        label: str,
    ) -> np.ndarray:
        psi = np.asarray(values, dtype=DTYPE).reshape(-1)
        if psi.shape != (2,):
            raise ValueError(f"{label} must contain exactly two amplitudes.")
        norm = float(np.linalg.norm(psi))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError(f"{label} must have finite nonzero norm.")
        return psi / norm

    @staticmethod
    def _orthogonal_companion(psi: np.ndarray) -> np.ndarray:
        a, b = complex(psi[0]), complex(psi[1])
        out = np.array([-np.conjugate(b), np.conjugate(a)], dtype=DTYPE)
        return out / np.linalg.norm(out)

    def control_spec(self) -> ControlSpec:
        """Return the v3 control layout expected by this system.

        The old fourth-order scripts used ``get_control_spec()``.  v3 standardizes
        the name to ``control_spec()`` so optimizers, guesses, diagnostics, and repair
        tools can call every system through one contract.  The final endpoint sample
        is part of the control vector for artifact compatibility, even though only
        the first ``N-1`` samples generate propagation intervals.
        """

        return ControlSpec(
            keys=CONTROL_KEYS,
            control_dim=self.N,
            dtype=float,
            dt=self.dt,
            meta={
                "model": "universal fourth-order robust two-level state transfer",
                "control_convention": "N endpoint samples; first N-1 controls generate propagation intervals",
            },
        )

    def get_control_spec(self) -> ControlSpec:
        """Backward-compatible alias for older fourth-order helper scripts."""

        return self.control_spec()

    def get_control_info(self) -> dict[str, Any]:
        return self.control_spec().to_dict()

    def with_params(self, **updates: Any) -> "FourthOrderUniversalRobustSystem":
        """Return a new system with updated model/objective parameters.

        This is the curriculum hook.  Optimizers do not need to know whether a stage
        is emphasizing fidelity, ``F_norm2``, ``C_sym_norm2``, or energy.  A caller
        changes those weights by creating a new system with updated params, then
        continues from the current controls or a warmstart.
        """

        payload = asdict(self.params)
        payload.update(updates)
        return self.__class__(payload)

    def controls_from_arrays(
        self,
        arrays: dict[str, np.ndarray],
        *,
        name: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Controls:
        """Build v3 ``Controls`` from ux/uy/uz arrays in canonical order."""

        missing = [key for key in CONTROL_KEYS if key not in arrays]
        if missing:
            raise KeyError(f"Missing control arrays: {missing}")
        matrix = np.vstack([np.asarray(arrays[key], dtype=float) for key in CONTROL_KEYS])
        return Controls.from_matrix(
            self.control_spec(),
            matrix,
            copy=True,
            name=name,
            meta=meta,
        )

    def controls_from_npz(self, path: str | Path, *, name: str | None = None) -> Controls:
        """Load endpoint-sampled ux/uy/uz controls from a saved NPZ artifact."""

        path_obj = Path(path)
        return self.controls_from_arrays(
            load_control_npz(path_obj, target_N=self.N),
            name=name or path_obj.stem,
            meta={"source_npz": str(path_obj)},
        )

    def reference_controls(self, *, name: str = "fourth_order_previous_best") -> Controls:
        """Return the copied canonical previous-best fourth-order control."""

        return self.controls_from_npz(FOURTH_ORDER_BEST_CONTROLS, name=name)

    def evaluate(self, controls: Controls) -> dict[str, Any]:
        """Evaluate controls and return v3 metrics with scalar ``J``.

        ``simulate`` is reused because it already performs validation, propagation,
        optional finite-alpha robustness checks, and metric collection.  With the
        default params no expensive random-direction robustness scan is run during
        optimizer iterations because ``robust_alpha_default`` is ``None``.
        """

        return self.simulate(controls)

    def gradient(self, controls: Controls) -> Controls:
        """Return analytical ``dJ/du`` in the same layout as ``controls``.

        The old implementation names the descent direction ``corrections`` and stores
        the actual gradient in ``state["grad_J"]``.  v3 optimizers expect a gradient
        object and then choose their own update rule, so this method returns
        ``grad_J`` rather than the negative correction.
        """

        self.forwardprop(controls)
        self.corrections(controls)
        grad = self.state["grad_J"]
        matrix = np.vstack([np.asarray(grad[key], dtype=float) for key in CONTROL_KEYS])
        return Controls.from_matrix(
            self.control_spec(),
            matrix,
            copy=True,
            name="gradient_J",
            meta={"quantity": "dJ_du"},
        )

    def system_info(self, *, compact: bool = False, render_math: bool = True) -> dict[str, Any]:
        payload = {
            "title": "Universal Fourth-Order Robust Two-Level System",
            "model": {
                "name": self.__class__.__name__,
                "dimension": 2,
                "task": "state transfer",
                "training_formulation": "Pauli-channel universal objective; no fixed V in training",
            },
            "time_grid": {
                "tau": float(self.params.tau),
                "N_samples": int(self.N),
                "dt": float(self.dt),
                "state_samples": int(self.N),
                "propagation_intervals": int(self.N - 1),
            },
            "controls": {
                "keys": CONTROL_KEYS,
                "hamiltonian": "H0(t)=0.5*(ux*sigma_x+uy*sigma_y+uz*sigma_z)",
            },
            "initial_target": {
                "psi0": self.psi_initial.copy(),
                "psi_target": self.psi_target.copy(),
                "psi_perp_seed": self.psi_perp_seed.copy(),
            },
            "objective": {
                "J": "w_inf*(1-F0)+lambda2*sum_k|F_k(T)|^2+lambda4*||C_sym(T)||_F^2+w_E*energy",
                "F_k": "integral <psi_perp|sigma_k|psi0> dt",
                "C_kl": "integral <psi0|sigma_k|psi0> F_l dt",
                "universal_condition": "F(T)=0 and (C(T)+C(T)^T)/2=0",
            },
            "parameters": asdict(self.params),
            "equations_latex": {
                "H0": r"H_0(t)=\frac12\sum_{k=x,y,z}u_k(t)\sigma_k",
                "F": r"\dot F_k=\langle\psi_\perp|\sigma_k|\psi_0\rangle",
                "C": r"\dot C_{kl}=\langle\psi_0|\sigma_k|\psi_0\rangle F_l",
                "J": r"J=w_{\rm inf}(1-\mathcal F_0)+\lambda_2\sum_k|F_k(T)|^2+\lambda_4\|C^{\rm sym}(T)\|_F^2+w_EE",
            },
        }

        def fmt(value: Any) -> str:
            if isinstance(value, np.ndarray):
                if value.size <= 12:
                    return np.array2string(value, precision=6, suppress_small=True)
                return f"array(shape={value.shape}, dtype={value.dtype})"
            if isinstance(value, (float, np.floating)):
                return f"{float(value):.10g}"
            return str(value)

        print("=" * 72)
        print(payload["title"])
        print("=" * 72)
        for section in ("model", "time_grid", "controls", "initial_target", "objective", "parameters"):
            print(f"\n[{section.replace('_', ' ').title()}]")
            for key, value in payload[section].items():
                print(f"  {key}: {fmt(value)}")
        if not compact:
            print("\n[Notes]")
            print("  - Training propagates only H0; V is represented by Pauli channels.")
            print("  - simulate(..., alpha=...) performs finite-alpha tests for chosen V directions.")
            print("  - Use load_control_npz(...) for endpoint-sampled saved controls.")
        if render_math:
            try:
                from IPython.display import Math, Markdown, display  # type: ignore
            except Exception:
                pass
            else:
                display(Markdown("### Rendered Equations (`system_info`)"))
                for label, expr in payload["equations_latex"].items():
                    display(Markdown(f"- `{label}`"))
                    display(Math(expr))
        print("=" * 72)
        return payload

    def _validate_controls(
        self,
        ux: Any,
        uy: np.ndarray | None = None,
        uz: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(ux, Controls):
            if uy is not None or uz is not None:
                raise TypeError("When passing Controls, do not also pass uy/uz.")
            ctrl = Controls.from_matrix(self.get_control_spec(), ux.as_matrix(copy=False), copy=True)
            arrays = tuple(np.asarray(ctrl.channel(key), dtype=float) for key in CONTROL_KEYS)
        elif uy is None and uz is None:
            payload = ux
            if hasattr(payload, "as_dict"):
                payload = payload.as_dict(copy=False)
            if isinstance(payload, dict):
                missing = [key for key in CONTROL_KEYS if key not in payload]
                if missing:
                    raise KeyError(f"Control dict missing keys: {missing}")
                arrays = tuple(np.asarray(payload[key], dtype=float) for key in CONTROL_KEYS)
            else:
                arr = np.asarray(payload, dtype=float)
                if arr.shape != (3, self.N):
                    raise TypeError(
                        "Controls must be Controls, dict with ux/uy/uz, matrix shape (3,N), "
                        "or explicit ux,uy,uz arrays."
                    )
                arrays = (arr[0], arr[1], arr[2])
        else:
            if uy is None or uz is None:
                raise TypeError("Expected either one controls object or explicit ux,uy,uz arrays.")
            arrays = (
                np.asarray(ux, dtype=float),
                np.asarray(uy, dtype=float),
                np.asarray(uz, dtype=float),
            )

        out = []
        for key, arr in zip(CONTROL_KEYS, arrays):
            if arr.shape != (self.N,):
                raise ValueError(f"{key} must have shape ({self.N},), got {arr.shape}.")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{key} contains non-finite values.")
            out.append(arr.astype(float, copy=True))
        return out[0], out[1], out[2]

    def _controls_match_cache(self, ux: np.ndarray, uy: np.ndarray, uz: np.ndarray) -> bool:
        try:
            return (
                np.array_equal(self.state.get("ux"), ux)
                and np.array_equal(self.state.get("uy"), uy)
                and np.array_equal(self.state.get("uz"), uz)
            )
        except Exception:
            return False

    def buildH(
        self,
        ux: Any,
        uy: np.ndarray | None = None,
        uz: np.ndarray | None = None,
    ) -> np.ndarray:
        ux, uy, uz = self._validate_controls(ux, uy, uz)
        H = np.empty((self.N, 2, 2), dtype=DTYPE)
        H[:, 0, 0] = 0.5 * uz
        H[:, 1, 1] = -0.5 * uz
        H[:, 0, 1] = 0.5 * (ux - 1j * uy)
        H[:, 1, 0] = 0.5 * (ux + 1j * uy)
        return H

    def buildH_perturbed(
        self,
        ux: Any,
        uy: np.ndarray | None = None,
        uz: np.ndarray | None = None,
        *,
        alpha: float,
        direction: np.ndarray | list[float] | tuple[float, float, float],
    ) -> np.ndarray:
        H = self.buildH(ux, uy, uz)
        d = self._normalize_direction(direction)
        V = d[0] * self.sx + d[1] * self.sy + d[2] * self.sz
        return H + float(alpha) * V[None, :, :]

    @staticmethod
    def _normalize_direction(direction: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        return _normalize_direction_array(direction)

    def forwardprop(
        self,
        ux: Any,
        uy: np.ndarray | None = None,
        uz: np.ndarray | None = None,
    ) -> np.ndarray:
        ux, uy, uz = self._validate_controls(ux, uy, uz)
        H = self.buildH(ux, uy, uz)

        psi0 = np.zeros((self.N, 2), dtype=DTYPE)
        psip = np.zeros((self.N, 2), dtype=DTYPE)
        psi0[0] = self.psi_initial
        psip[0] = self.psi_perp_seed

        U_steps = np.zeros((self.N - 1, 2, 2), dtype=DTYPE)
        e_samples = np.zeros((self.N, 3), dtype=float)
        f_samples = np.zeros((self.N, 3), dtype=DTYPE)
        F_traj = np.zeros((self.N + 1, 3), dtype=DTYPE)
        C_traj = np.zeros((self.N + 1, 3, 3), dtype=DTYPE)

        for n in range(self.N - 1):
            U = _mat_expm(-1j * H[n] * self.dt)
            U_steps[n] = U
            psi0[n + 1] = U @ psi0[n]
            psip[n + 1] = U @ psip[n]

        for n in range(self.N):
            psi_n = psi0[n]
            psip_n = psip[n]
            for k, sigma in enumerate(self.paulis):
                e_samples[n, k] = float(np.real(np.vdot(psi_n, sigma @ psi_n)))
                f_samples[n, k] = np.vdot(psip_n, sigma @ psi_n)

            F_traj[n + 1] = F_traj[n] + self.dt * f_samples[n]
            C_traj[n + 1] = C_traj[n] + self.dt * np.outer(e_samples[n], F_traj[n])

        overlap = complex(np.vdot(self.psi_target, psi0[-1]))
        fidelity = float(np.abs(overlap) ** 2)
        infidelity = float(1.0 - fidelity)
        infidelity_clipped = float(max(0.0, infidelity))
        F_terminal = F_traj[-1]
        C_terminal = C_traj[-1]
        C_sym = 0.5 * (C_terminal + C_terminal.T)
        F_norm2 = float(np.sum(np.abs(F_terminal) ** 2))
        C_sym_norm2 = float(np.sum(np.abs(C_sym) ** 2))
        energy = float(self.dt * np.sum(ux * ux + uy * uy + uz * uz))
        J_infidelity = float(self.params.infidelity_weight) * infidelity
        J_second_order = float(self.params.lambda2) * F_norm2
        J_fourth_order = float(self.params.lambda4) * C_sym_norm2
        J_energy = float(self.params.energy_weight) * energy
        J = float(J_infidelity + J_second_order + J_fourth_order + J_energy)
        psi0_norm_error = np.abs(np.sum(np.abs(psi0) ** 2, axis=1) - 1.0)
        psip_norm_error = np.abs(np.sum(np.abs(psip) ** 2, axis=1) - 1.0)
        frame_overlap = np.einsum("ni,ni->n", np.conj(psip), psi0)

        self.state = {
            "ux": ux.copy(),
            "uy": uy.copy(),
            "uz": uz.copy(),
            "H": H,
            "U_steps": U_steps,
            "psi0_traj": psi0,
            "psi_perp_traj": psip,
            "psi": psi0,
            "psi_perp": psip,
            "e_samples": e_samples,
            "f_samples": f_samples,
            "F_traj": F_traj,
            "C_traj": C_traj,
            "F_terminal": F_terminal.copy(),
            "C_terminal": C_terminal.copy(),
            "C_sym_terminal": C_sym.copy(),
            "F_norm2": F_norm2,
            "C_sym_norm2": C_sym_norm2,
            "terminal_overlap": overlap,
            "a": overlap,
            "fidelity": fidelity,
            "infidelity": infidelity,
            "infidelity_raw": infidelity,
            "infidelity_clipped": infidelity_clipped,
            "energy": energy,
            "J_infidelity": J_infidelity,
            "J_second_order": J_second_order,
            "J_fourth_order": J_fourth_order,
            "J_energy": J_energy,
            "J": J,
            "psi_T": psi0[-1].copy(),
            "psi_perp_T": psip[-1].copy(),
            "bloch_x": np.real(np.einsum("ni,ij,nj->n", np.conj(psi0), self.sx, psi0)),
            "bloch_y": np.real(np.einsum("ni,ij,nj->n", np.conj(psi0), self.sy, psi0)),
            "bloch_z": np.real(np.einsum("ni,ij,nj->n", np.conj(psi0), self.sz, psi0)),
            "psi0_norm_max_error": float(np.max(psi0_norm_error)),
            "psi_perp_norm_max_error": float(np.max(psip_norm_error)),
            "frame_orthogonality_max_abs": float(np.max(np.abs(frame_overlap))),
        }
        return psi0

    def backwardprop(self) -> dict[str, np.ndarray]:
        if "U_steps" not in self.state:
            raise RuntimeError("Run forwardprop(...) before backwardprop().")

        psi0 = np.asarray(self.state["psi0_traj"], dtype=DTYPE)
        psip = np.asarray(self.state["psi_perp_traj"], dtype=DTYPE)
        U_steps = np.asarray(self.state["U_steps"], dtype=DTYPE)
        e_samples = np.asarray(self.state["e_samples"], dtype=float)
        F_traj = np.asarray(self.state["F_traj"], dtype=DTYPE)

        chi0 = np.zeros((self.N, 2), dtype=DTYPE)
        chip = np.zeros((self.N, 2), dtype=DTYPE)
        p = np.zeros((self.N + 1, 3), dtype=DTYPE)
        q = np.zeros((self.N + 1, 3, 3), dtype=DTYPE)

        a = complex(self.state["terminal_overlap"])
        chi0[-1] = -float(self.params.infidelity_weight) * a * self.psi_target
        p[-1] = float(self.params.lambda2) * np.asarray(self.state["F_terminal"], dtype=DTYPE)
        q[-1] = float(self.params.lambda4) * np.asarray(self.state["C_sym_terminal"], dtype=DTYPE)

        for n in range(self.N - 1, -1, -1):
            q[n] = q[n + 1]
            p[n] = p[n + 1] + self.dt * np.einsum("k,kl->l", e_samples[n], q[n + 1])

            source_f = np.zeros(2, dtype=DTYPE)
            source_fp = np.zeros(2, dtype=DTYPE)
            for k, sigma in enumerate(self.paulis):
                source_f += p[n + 1, k] * (sigma @ psip[n])
                source_fp += np.conjugate(p[n + 1, k]) * (sigma @ psi0[n])

            r = np.einsum("kl,l->k", np.conjugate(q[n + 1]), F_traj[n])
            source_e = np.zeros(2, dtype=DTYPE)
            for k, sigma in enumerate(self.paulis):
                source_e += 2.0 * float(np.real(r[k])) * (sigma @ psi0[n])

            sample_chi0 = self.dt * (source_f + source_e)
            sample_chip = self.dt * source_fp
            if n < self.N - 1:
                Udag = U_steps[n].conj().T
                chi0[n] = Udag @ chi0[n + 1] + sample_chi0
                chip[n] = Udag @ chip[n + 1] + sample_chip
            else:
                chi0[n] += sample_chi0
                chip[n] += sample_chip

        self.state.update(
            {
                "chi0": chi0,
                "chi_perp": chip,
                "p_costate": p,
                "q_costate": q,
                "terminal_chi0": chi0[-1].copy(),
                "terminal_chi_perp": chip[-1].copy(),
            }
        )
        return {"chi0": chi0, "chi_perp": chip, "p": p, "q": q}

    def corrections(
        self,
        ux: Any,
        uy: np.ndarray | None = None,
        uz: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        ux, uy, uz = self._validate_controls(ux, uy, uz)
        if not self._controls_match_cache(ux, uy, uz):
            self.forwardprop(ux, uy, uz)
        if "chi0" not in self.state:
            self.backwardprop()

        H = np.asarray(self.state["H"], dtype=DTYPE)
        psi0 = np.asarray(self.state["psi0_traj"], dtype=DTYPE)
        psip = np.asarray(self.state["psi_perp_traj"], dtype=DTYPE)
        chi0 = np.asarray(self.state["chi0"], dtype=DTYPE)
        chip = np.asarray(self.state["chi_perp"], dtype=DTYPE)

        grads = np.zeros((3, self.N), dtype=float)
        control_arrays = (ux, uy, uz)
        for k, arr in enumerate(control_arrays):
            grads[k] += (
                2.0
                * float(self.params.energy_weight)
                * self.dt
                * np.asarray(arr, dtype=float)
            )

        for n in range(self.N - 1):
            A = -1j * H[n] * self.dt
            for k, sigma_half in enumerate(self.control_generators):
                E = -1j * sigma_half * self.dt
                dU = _mat_expm_frechet(A, E)
                term = np.vdot(chi0[n + 1], dU @ psi0[n])
                term += np.vdot(chip[n + 1], dU @ psip[n])
                grads[k, n] += 2.0 * float(np.real(term))

        corrections = -grads
        self.state["grad_J"] = {
            "ux": grads[0].copy(),
            "uy": grads[1].copy(),
            "uz": grads[2].copy(),
        }
        self.state["corrections"] = {
            "ux": corrections[0].copy(),
            "uy": corrections[1].copy(),
            "uz": corrections[2].copy(),
        }
        return {
            "ux": corrections[0].copy(),
            "uy": corrections[1].copy(),
            "uz": corrections[2].copy(),
        }

    # ------------------------------------------------------------------
    # v3 hard-condition residuals and seeded-adjoint Jacobians
    # ------------------------------------------------------------------

    def _target_perp(self) -> np.ndarray:
        """Return a normalized vector orthogonal to the target state.

        The nominal-transfer hard condition is that the final state has no component
        along this target-orthogonal vector.  Splitting that complex leakage into real
        and imaginary parts gives two real residual equations.
        """

        return self._orthogonal_companion(self.psi_target)

    @staticmethod
    def _complex_to_real_rows(values: np.ndarray) -> np.ndarray:
        """Interleave real/imag parts of a complex vector as real residual rows."""

        out: list[float] = []
        for value in np.asarray(values, dtype=DTYPE).reshape(-1):
            out.append(float(np.real(value)))
            out.append(float(np.imag(value)))
        return np.asarray(out, dtype=float)

    def _residual_blocks_from_state(self) -> dict[str, np.ndarray]:
        """Return complex residual blocks from the latest forward propagation."""

        if "psi_T" not in self.state:
            raise RuntimeError("Run forwardprop(...) before building residuals.")

        eta = self._target_perp()
        leak = np.asarray([complex(np.vdot(eta, self.state["psi_T"]))], dtype=DTYPE)
        F = np.asarray(self.state["F_terminal"], dtype=DTYPE).reshape(3)
        C = np.asarray(self.state["C_terminal"], dtype=DTYPE).reshape(3, 3)

        # The fourth-order universal condition uses the transpose-symmetric part of
        # C.  For residuals we store the six independent complex equations.  Off
        # diagonals are represented by the unnormalized sums C_ij + C_ji because the
        # old best-control workflow used those exact hard conditions.
        c_sym_values = np.asarray(
            [
                C[0, 0],
                C[1, 1],
                C[2, 2],
                C[0, 1] + C[1, 0],
                C[0, 2] + C[2, 0],
                C[1, 2] + C[2, 1],
            ],
            dtype=DTYPE,
        )
        return {"nominal": leak, "F": F, "C_sym": c_sym_values}

    def residuals(self, controls: Controls, *, name: str = "hard") -> np.ndarray:
        """Return named real residual vectors for repair/projection tools.

        Names:
        - ``"nominal"``: 2 real rows, target-orthogonal terminal leakage.
        - ``"F"`` / ``"second_order"``: 6 real rows, the three complex F_k(T).
        - ``"hard"``: 8 rows, nominal leakage plus F_k(T).  This is the standard
          hard constraint used before energy or fourth-order polishing.
        - ``"C_sym"``: 12 rows, the six complex transpose-symmetric C conditions.
        - ``"fourth_order"`` / ``"full"``: 20 rows, nominal + F + C_sym.
        """

        self.forwardprop(controls)
        blocks = self._residual_blocks_from_state()
        key = str(name).lower()
        if key in {"nominal", "leak", "fidelity"}:
            values = blocks["nominal"]
        elif key in {"f", "second_order", "second-order"}:
            values = blocks["F"]
        elif key in {"hard", "nominal_f", "nominal+f"}:
            values = np.concatenate((blocks["nominal"], blocks["F"]))
        elif key in {"c", "c_sym", "csym", "fourth"}:
            values = blocks["C_sym"]
        elif key in {"fourth_order", "fourth-order", "full", "all"}:
            values = np.concatenate((blocks["nominal"], blocks["F"], blocks["C_sym"]))
        else:
            raise ValueError(
                "Unknown residual name. Use nominal, F, hard, C_sym, or fourth_order."
            )
        return self._complex_to_real_rows(values)

    @staticmethod
    def _flatten_gradient_dict(gradient: dict[str, np.ndarray]) -> np.ndarray:
        """Flatten a ux/uy/uz gradient dictionary in the Controls row order."""

        return np.concatenate(
            [np.asarray(gradient[key], dtype=float).reshape(-1) for key in CONTROL_KEYS]
        )

    def _seeded_gradient(
        self,
        controls: Controls,
        *,
        chi0_terminal: np.ndarray | None = None,
        p_terminal: np.ndarray | None = None,
        q_terminal: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return one terminal-seeded adjoint gradient row.

        This is the exact row-building idea used by the old fourth-order projected
        workflow.  A terminal seed selects one scalar residual component; propagating
        the corresponding adjoint backward gives the derivative of that scalar with
        respect to every control sample.

        The caller uses a zero-weight worker system for Jacobian rows so objective
        weights, especially energy, cannot leak into residual derivatives.
        """

        self.forwardprop(controls)

        psi0 = np.asarray(self.state["psi0_traj"], dtype=DTYPE)
        psip = np.asarray(self.state["psi_perp_traj"], dtype=DTYPE)
        U_steps = np.asarray(self.state["U_steps"], dtype=DTYPE)
        e_samples = np.asarray(self.state["e_samples"], dtype=float)
        F_traj = np.asarray(self.state["F_traj"], dtype=DTYPE)

        chi0 = np.zeros((self.N, 2), dtype=DTYPE)
        chip = np.zeros((self.N, 2), dtype=DTYPE)
        p = np.zeros((self.N + 1, 3), dtype=DTYPE)
        qcost = np.zeros((self.N + 1, 3, 3), dtype=DTYPE)

        if chi0_terminal is not None:
            chi0[-1] = np.asarray(chi0_terminal, dtype=DTYPE).reshape(2)
        if p_terminal is not None:
            p[-1] = np.asarray(p_terminal, dtype=DTYPE).reshape(3)
        if q_terminal is not None:
            qcost[-1] = np.asarray(q_terminal, dtype=DTYPE).reshape(3, 3)

        for n in range(self.N - 1, -1, -1):
            qcost[n] = qcost[n + 1]
            p[n] = p[n + 1] + self.dt * np.einsum("k,kl->l", e_samples[n], qcost[n + 1])

            source_f = np.zeros(2, dtype=DTYPE)
            source_fp = np.zeros(2, dtype=DTYPE)
            for k, sigma in enumerate(self.paulis):
                source_f += p[n + 1, k] * (sigma @ psip[n])
                source_fp += np.conjugate(p[n + 1, k]) * (sigma @ psi0[n])

            r = np.einsum("kl,l->k", np.conjugate(qcost[n + 1]), F_traj[n])
            source_e = np.zeros(2, dtype=DTYPE)
            for k, sigma in enumerate(self.paulis):
                source_e += 2.0 * float(np.real(r[k])) * (sigma @ psi0[n])

            sample_chi0 = self.dt * (source_f + source_e)
            sample_chip = self.dt * source_fp
            if n < self.N - 1:
                Udag = U_steps[n].conj().T
                chi0[n] = Udag @ chi0[n + 1] + sample_chi0
                chip[n] = Udag @ chip[n + 1] + sample_chip
            else:
                chi0[n] += sample_chi0
                chip[n] += sample_chip

        self.state.update({"chi0": chi0, "chi_perp": chip, "p_costate": p, "q_costate": qcost})
        self.corrections(controls)
        return self._flatten_gradient_dict(self.state["grad_J"])

    def _zero_weight_worker(self) -> "FourthOrderUniversalRobustSystem":
        """Return an equivalent system whose objective weights are all zero."""

        return self.with_params(
            infidelity_weight=0.0,
            lambda2=0.0,
            lambda4=0.0,
            energy_weight=0.0,
        )

    def _hard_jacobian_rows(self, controls: Controls) -> list[np.ndarray]:
        """Return rows for nominal leakage and F_k(T) residuals."""

        worker = self._zero_weight_worker()
        eta = worker._target_perp()
        rows: list[np.ndarray] = [
            worker._seeded_gradient(controls, chi0_terminal=0.5 * eta),
            worker._seeded_gradient(controls, chi0_terminal=0.5j * eta),
        ]
        for k in range(3):
            seed_re = np.zeros(3, dtype=DTYPE)
            seed_im = np.zeros(3, dtype=DTYPE)
            seed_re[k] = 0.5
            seed_im[k] = 0.5j
            rows.append(worker._seeded_gradient(controls, p_terminal=seed_re))
            rows.append(worker._seeded_gradient(controls, p_terminal=seed_im))
        return rows

    def _c_sym_jacobian_rows(self, controls: Controls) -> list[np.ndarray]:
        """Return rows for the six complex fourth-order C_sym conditions."""

        worker = self._zero_weight_worker()
        specs = [
            ((0, 0),),
            ((1, 1),),
            ((2, 2),),
            ((0, 1), (1, 0)),
            ((0, 2), (2, 0)),
            ((1, 2), (2, 1)),
        ]
        rows: list[np.ndarray] = []
        for entries in specs:
            seed_re = np.zeros((3, 3), dtype=DTYPE)
            seed_im = np.zeros((3, 3), dtype=DTYPE)
            for i, j in entries:
                seed_re[i, j] = 0.5
                seed_im[i, j] = 0.5j
            rows.append(worker._seeded_gradient(controls, q_terminal=seed_re))
            rows.append(worker._seeded_gradient(controls, q_terminal=seed_im))
        return rows

    def jacobian(self, controls: Controls, *, name: str = "hard") -> np.ndarray:
        """Return analytical residual Jacobian rows in flattened control space."""

        key = str(name).lower()
        if key in {"nominal", "leak", "fidelity"}:
            rows = self._hard_jacobian_rows(controls)[:2]
        elif key in {"f", "second_order", "second-order"}:
            rows = self._hard_jacobian_rows(controls)[2:]
        elif key in {"hard", "nominal_f", "nominal+f"}:
            rows = self._hard_jacobian_rows(controls)
        elif key in {"c", "c_sym", "csym", "fourth"}:
            rows = self._c_sym_jacobian_rows(controls)
        elif key in {"fourth_order", "fourth-order", "full", "all"}:
            rows = self._hard_jacobian_rows(controls) + self._c_sym_jacobian_rows(controls)
        else:
            raise ValueError(
                "Unknown jacobian name. Use nominal, F, hard, C_sym, or fourth_order."
            )
        return np.vstack(rows).astype(float, copy=False)

    def directional_obstruction(
        self,
        direction: np.ndarray | list[float] | tuple[float, float, float],
        *,
        use_symmetric: bool = False,
    ) -> complex:
        if "C_terminal" not in self.state:
            raise RuntimeError("Run forwardprop(...) before directional_obstruction(...).")
        v = self._normalize_direction(direction)
        C = self.state["C_sym_terminal"] if use_symmetric else self.state["C_terminal"]
        return complex(v @ np.asarray(C, dtype=DTYPE) @ v)

    def _constraint_metrics(self) -> dict[str, Any]:
        F = np.asarray(self.state["F_terminal"], dtype=DTYPE)
        C = np.asarray(self.state["C_terminal"], dtype=DTYPE)
        Csym = np.asarray(self.state["C_sym_terminal"], dtype=DTYPE)
        constraints = {
            "C_xx": complex(C[0, 0]),
            "C_yy": complex(C[1, 1]),
            "C_zz": complex(C[2, 2]),
            "C_xy_plus_yx": complex(C[0, 1] + C[1, 0]),
            "C_xz_plus_zx": complex(C[0, 2] + C[2, 0]),
            "C_yz_plus_zy": complex(C[1, 2] + C[2, 1]),
        }
        out: dict[str, Any] = {
            "F_x": complex(F[0]),
            "F_y": complex(F[1]),
            "F_z": complex(F[2]),
            "F_x_abs2": float(abs(F[0]) ** 2),
            "F_y_abs2": float(abs(F[1]) ** 2),
            "F_z_abs2": float(abs(F[2]) ** 2),
            "C_xx_abs2": float(abs(C[0, 0]) ** 2),
            "C_yy_abs2": float(abs(C[1, 1]) ** 2),
            "C_zz_abs2": float(abs(C[2, 2]) ** 2),
            "C_xy_plus_yx_abs2": float(abs(C[0, 1] + C[1, 0]) ** 2),
            "C_xz_plus_zx_abs2": float(abs(C[0, 2] + C[2, 0]) ** 2),
            "C_yz_plus_zy_abs2": float(abs(C[1, 2] + C[2, 1]) ** 2),
            "C_antisym_norm2": float(np.sum(np.abs(0.5 * (C - C.T)) ** 2)),
            "C_trace": complex(np.trace(C)),
            "C_sym_trace": complex(np.trace(Csym)),
            "G_axis_x": complex(C[0, 0]),
            "G_axis_y": complex(C[1, 1]),
            "G_axis_z": complex(C[2, 2]),
            "G_diag111_normalized": complex(
                self._normalize_direction((1.0, 1.0, 1.0)) @ C @ self._normalize_direction((1.0, 1.0, 1.0))
            ),
        }
        out.update(constraints)
        return out

    def metrics(self) -> dict[str, Any]:
        if "J" not in self.state:
            raise RuntimeError("Run forwardprop(...) before metrics().")
        overlap = complex(self.state["terminal_overlap"])
        out: dict[str, Any] = {
            "J": float(self.state["J"]),
            "J_infidelity": float(self.state["J_infidelity"]),
            "J_second_order": float(self.state["J_second_order"]),
            "J_fourth_order": float(self.state["J_fourth_order"]),
            "J_energy": float(self.state["J_energy"]),
            "fidelity": float(self.state["fidelity"]),
            "infidelity": float(self.state["infidelity"]),
            "infidelity_raw": float(self.state["infidelity_raw"]),
            "infidelity_clipped": float(self.state["infidelity_clipped"]),
            "terminal_overlap": overlap,
            "terminal_overlap_abs": float(abs(overlap)),
            "terminal_overlap_phase": float(np.angle(overlap)),
            "energy": float(self.state["energy"]),
            "F_terminal": np.asarray(self.state["F_terminal"], dtype=DTYPE).copy(),
            "F_norm2": float(self.state["F_norm2"]),
            "C_terminal": np.asarray(self.state["C_terminal"], dtype=DTYPE).copy(),
            "C_sym_terminal": np.asarray(self.state["C_sym_terminal"], dtype=DTYPE).copy(),
            "C_sym_norm2": float(self.state["C_sym_norm2"]),
            "psi_T": np.asarray(self.state["psi_T"], dtype=DTYPE).copy(),
            "psi_perp_T": np.asarray(self.state["psi_perp_T"], dtype=DTYPE).copy(),
            "bloch_x": np.asarray(self.state["bloch_x"], dtype=float).copy(),
            "bloch_y": np.asarray(self.state["bloch_y"], dtype=float).copy(),
            "bloch_z": np.asarray(self.state["bloch_z"], dtype=float).copy(),
            "psi0_norm_max_error": float(self.state["psi0_norm_max_error"]),
            "psi_perp_norm_max_error": float(self.state["psi_perp_norm_max_error"]),
            "frame_orthogonality_max_abs": float(self.state["frame_orthogonality_max_abs"]),
        }
        out.update(self._constraint_metrics())

        robust_keys = [key for key in self.state if key.startswith("robust_")]
        for key in robust_keys:
            value = self.state[key]
            if isinstance(value, np.ndarray):
                out[key] = value.copy()
            else:
                out[key] = value
        return out

    def _direction_set(
        self,
        directions: str | np.ndarray | list[list[float]] | None,
        *,
        n_dirs: int | None,
        seed: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if directions is None:
            directions = "sphere"

        if isinstance(directions, str):
            spec = directions.strip().lower()
            if spec in {"axes", "axis", "xyz"}:
                dirs = np.eye(3, dtype=float)
                labels = np.asarray(["x", "y", "z"], dtype=object)
            elif spec in {"x", "axis_x"}:
                dirs = np.array([[1.0, 0.0, 0.0]], dtype=float)
                labels = np.asarray(["x"], dtype=object)
            elif spec in {"y", "axis_y"}:
                dirs = np.array([[0.0, 1.0, 0.0]], dtype=float)
                labels = np.asarray(["y"], dtype=object)
            elif spec in {"z", "axis_z"}:
                dirs = np.array([[0.0, 0.0, 1.0]], dtype=float)
                labels = np.asarray(["z"], dtype=object)
            elif spec in {"diag", "111", "diagonal"}:
                dirs = np.array([[1.0, 1.0, 1.0]], dtype=float)
                labels = np.asarray(["111"], dtype=object)
            elif spec in {"sphere", "random", "s2"}:
                count = int(self.params.robust_n_dirs if n_dirs is None else n_dirs)
                dirs = sample_unit_vectors_spherical(count, seed=seed)
                labels = np.asarray([f"sphere_{i:04d}" for i in range(count)], dtype=object)
            elif spec in {"pair_xy", "xy", "pair_yz", "yz", "pair_zx", "zx", "pairs", "pairwise"}:
                count = int(self.params.robust_pair_n_dirs if n_dirs is None else n_dirs)
                angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)

                def circle(pair: str) -> np.ndarray:
                    out = np.zeros((count, 3), dtype=float)
                    c = np.cos(angles)
                    s = np.sin(angles)
                    if pair == "xy":
                        out[:, 0], out[:, 1] = c, s
                    elif pair == "yz":
                        out[:, 1], out[:, 2] = c, s
                    elif pair == "zx":
                        out[:, 2], out[:, 0] = c, s
                    else:
                        raise ValueError(pair)
                    return out

                if spec in {"pair_xy", "xy"}:
                    dirs = circle("xy")
                    labels = np.asarray([f"xy_{i:04d}" for i in range(count)], dtype=object)
                elif spec in {"pair_yz", "yz"}:
                    dirs = circle("yz")
                    labels = np.asarray([f"yz_{i:04d}" for i in range(count)], dtype=object)
                elif spec in {"pair_zx", "zx"}:
                    dirs = circle("zx")
                    labels = np.asarray([f"zx_{i:04d}" for i in range(count)], dtype=object)
                else:
                    dirs = np.vstack((circle("xy"), circle("yz"), circle("zx")))
                    labels = np.asarray(
                        [f"xy_{i:04d}" for i in range(count)]
                        + [f"yz_{i:04d}" for i in range(count)]
                        + [f"zx_{i:04d}" for i in range(count)],
                        dtype=object,
                    )
            else:
                raise ValueError(f"Unknown directions specifier: {directions!r}")
        else:
            dirs = np.asarray(directions, dtype=float)
            if dirs.ndim == 1:
                dirs = dirs.reshape(1, 3)
            if dirs.ndim != 2 or dirs.shape[1] != 3:
                raise ValueError("directions array must have shape (n_dirs, 3).")
            labels = np.asarray([f"dir_{i:04d}" for i in range(dirs.shape[0])], dtype=object)

        norms = np.linalg.norm(dirs, axis=1)
        if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
            raise ValueError("directions must be finite nonzero vectors.")
        return dirs / norms[:, None], labels

    def _evaluate_one_direction(
        self,
        ux: np.ndarray,
        uy: np.ndarray,
        uz: np.ndarray,
        *,
        alpha: float,
        direction: np.ndarray,
    ) -> tuple[float, complex]:
        d = self._normalize_direction(direction)
        p0 = complex(self.psi_initial[0])
        p1 = complex(self.psi_initial[1])
        ax, ay, az = float(alpha) * d
        for n in range(self.N - 1):
            p0, p1 = _su2_step_apply(
                p0,
                p1,
                0.5 * float(ux[n]) + ax,
                0.5 * float(uy[n]) + ay,
                0.5 * float(uz[n]) + az,
                self.dt,
            )
        overlap = np.conjugate(self.psi_target[0]) * p0 + np.conjugate(self.psi_target[1]) * p1
        return float(abs(overlap) ** 2), complex(overlap)

    def _robust_eval_serial(
        self,
        ux: np.ndarray,
        uy: np.ndarray,
        uz: np.ndarray,
        *,
        alpha: float,
        dirs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fids = np.empty(dirs.shape[0], dtype=float)
        overlaps = np.empty(dirs.shape[0], dtype=DTYPE)
        g4 = np.empty(dirs.shape[0], dtype=DTYPE)
        C = np.asarray(self.state["C_terminal"], dtype=DTYPE)
        for i, direction in enumerate(dirs):
            fids[i], overlaps[i] = self._evaluate_one_direction(
                ux,
                uy,
                uz,
                alpha=float(alpha),
                direction=direction,
            )
            g4[i] = complex(direction @ C @ direction)
        return fids, overlaps, g4

    def _robust_eval_parallel(
        self,
        ux: np.ndarray,
        uy: np.ndarray,
        uz: np.ndarray,
        *,
        alpha: float,
        dirs: np.ndarray,
        max_workers: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        cpu = os.cpu_count() or 1
        if max_workers is None:
            leave = max(0, int(self.params.robust_leave_cores_free))
            max_workers = max(1, cpu - leave)
        max_workers = max(1, min(int(max_workers), int(dirs.shape[0])))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(
                    _robust_worker_direction,
                    dirs,
                    repeat(float(alpha)),
                    repeat(np.asarray(ux, dtype=float)),
                    repeat(np.asarray(uy, dtype=float)),
                    repeat(np.asarray(uz, dtype=float)),
                    repeat(float(self.dt)),
                    repeat(np.asarray(self.psi_initial, dtype=DTYPE)),
                    repeat(np.asarray(self.psi_target, dtype=DTYPE)),
                    repeat(np.asarray(self.state["C_terminal"], dtype=DTYPE)),
                )
            )
        fids = np.asarray([row[0] for row in results], dtype=float)
        overlaps = np.asarray([row[1] for row in results], dtype=DTYPE)
        g4 = np.asarray([row[2] for row in results], dtype=DTYPE)
        return fids, overlaps, g4, max_workers

    def _robust_eval(
        self,
        ux: np.ndarray,
        uy: np.ndarray,
        uz: np.ndarray,
        *,
        alpha: float,
        directions: str | np.ndarray | list[list[float]] | None,
        n_dirs: int | None,
        seed: int | None,
        parallel: bool,
        max_workers: int | None,
    ) -> dict[str, Any]:
        dirs, labels = self._direction_set(directions, n_dirs=n_dirs, seed=seed)
        used_parallel = False
        used_workers = 1
        parallel_error = None
        if bool(parallel) and int(dirs.shape[0]) > 1:
            try:
                fids, overlaps, g4, used_workers = self._robust_eval_parallel(
                    ux,
                    uy,
                    uz,
                    alpha=float(alpha),
                    dirs=dirs,
                    max_workers=max_workers,
                )
                used_parallel = True
            except Exception as exc:
                parallel_error = repr(exc)
                fids, overlaps, g4 = self._robust_eval_serial(
                    ux,
                    uy,
                    uz,
                    alpha=float(alpha),
                    dirs=dirs,
                )
        else:
            fids, overlaps, g4 = self._robust_eval_serial(
                ux,
                uy,
                uz,
                alpha=float(alpha),
                dirs=dirs,
            )

        infids = 1.0 - fids
        worst = int(np.argmin(fids))
        best = int(np.argmax(fids))
        q = np.percentile(fids, [5, 25, 50, 75, 95])
        payload: dict[str, Any] = {
            "robust_alpha": float(alpha),
            "robust_n_dirs": int(dirs.shape[0]),
            "robust_direction_labels": labels,
            "robust_directions": dirs.astype(float, copy=True),
            "robust_fidelity_per_direction": fids.copy(),
            "robust_infidelity_per_direction": infids.copy(),
            "robust_terminal_overlap_per_direction": overlaps.copy(),
            "robust_G4_per_direction": g4.copy(),
            "robust_fidelity_mean": float(np.mean(fids)),
            "robust_infidelity_mean": float(np.mean(infids)),
            "robust_fidelity_std": float(np.std(fids)),
            "robust_fidelity_min": float(np.min(fids)),
            "robust_fidelity_max": float(np.max(fids)),
            "robust_fidelity_median": float(q[2]),
            "robust_fidelity_p05": float(q[0]),
            "robust_fidelity_p25": float(q[1]),
            "robust_fidelity_p75": float(q[3]),
            "robust_fidelity_p95": float(q[4]),
            "robust_worst_direction": dirs[worst].copy(),
            "robust_best_direction": dirs[best].copy(),
            "robust_worst_label": str(labels[worst]),
            "robust_best_label": str(labels[best]),
            "robust_worst_overlap": complex(overlaps[worst]),
            "robust_best_overlap": complex(overlaps[best]),
            "robust_G4_max_abs": float(np.max(np.abs(g4))),
            "robust_G4_mean_abs": float(np.mean(np.abs(g4))),
            "robust_parallel_requested": bool(parallel),
            "robust_parallel_used": bool(used_parallel),
            "robust_max_workers": int(used_workers),
            "robust_parallel_error": parallel_error,
        }

        label_strings = [str(x) for x in labels.tolist()]
        for prefix in ("x", "y", "z"):
            if prefix in label_strings:
                idx = label_strings.index(prefix)
                payload[f"robust_fidelity_axis_{prefix}"] = float(fids[idx])
        for pair in ("xy", "yz", "zx"):
            mask = np.asarray([label.startswith(pair + "_") for label in label_strings])
            if np.any(mask):
                payload[f"robust_fidelity_{pair}_mean"] = float(np.mean(fids[mask]))
                payload[f"robust_fidelity_{pair}_min"] = float(np.min(fids[mask]))
                payload[f"robust_fidelity_{pair}_std"] = float(np.std(fids[mask]))
        return payload

    def simulate(
        self,
        ux: Any,
        uy: np.ndarray | None = None,
        uz: np.ndarray | None = None,
        *,
        alpha: float | None = None,
        directions: str | np.ndarray | list[list[float]] | None = None,
        n_dirs: int | None = None,
        seed: int | None = None,
        parallel: bool | None = None,
        max_workers: int | None = None,
    ) -> dict[str, Any]:
        ux, uy, uz = self._validate_controls(ux, uy, uz)
        self.reset_state()
        self.forwardprop(ux, uy, uz)

        alpha_eff = alpha
        if alpha_eff is None:
            alpha_eff = self.params.robust_alpha_default
        if alpha_eff is not None:
            seed_eff = self.params.robust_seed if seed is None else seed
            parallel_eff = bool(self.params.robust_parallel if parallel is None else parallel)
            robust = self._robust_eval(
                ux,
                uy,
                uz,
                alpha=float(alpha_eff),
                directions=directions,
                n_dirs=n_dirs,
                seed=seed_eff,
                parallel=parallel_eff,
                max_workers=max_workers,
            )
            self.state.update(robust)
        return self.metrics()

    def validate_optimizer_compatibility(
        self,
        ux: Any,
        uy: np.ndarray | None = None,
        uz: np.ndarray | None = None,
    ) -> dict[str, Any]:
        ux, uy, uz = self._validate_controls(ux, uy, uz)
        self.forwardprop(ux, uy, uz)
        corr = self.corrections(ux, uy, uz)
        metrics = self.metrics()
        if "J" not in metrics:
            raise KeyError("metrics() must include 'J'.")
        for key, arr in corr.items():
            if key not in CONTROL_KEYS:
                raise KeyError(f"Unexpected correction key: {key}")
            if np.asarray(arr).shape != (self.N,):
                raise ValueError(f"corrections()['{key}'] must have shape ({self.N},).")
        return metrics

    def check_contract(
        self,
        ux: Any,
        uy: np.ndarray | None = None,
        uz: np.ndarray | None = None,
    ) -> dict[str, Any]:
        return self.validate_optimizer_compatibility(ux, uy, uz)

    def check_ctrl(
        self,
        ux: Any,
        uy: np.ndarray | None = None,
        uz: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self.simulate(ux, uy, uz, **kwargs)

    def get_state(self) -> dict[str, Any]:
        return dict(self.state)

    def reset_state(self) -> None:
        self.state.clear()

    def save_state(self, path: str | Path) -> None:
        if not self.state:
            raise RuntimeError("No state to save. Run forwardprop/simulate first.")
        payload: dict[str, Any] = {}
        for key, value in self.state.items():
            if isinstance(value, np.ndarray) or np.isscalar(value):
                payload[key] = value
        np.savez(Path(path), **payload)

    def fourier_guess_controls(
        self,
        *,
        n_terms: int = 8,
        coeff_scale: float = 0.1,
        include_dc: bool = True,
        seed: int | None = None,
    ) -> Controls:
        """Backward-compatible Fourier guess helper built on v3 guess tools."""

        del include_dc, seed
        return _v3_fourier_guess(
            self.control_spec(),
            modes=n_terms,
            amplitude=coeff_scale,
            decay="1/k",
            name="fourier_guess_controls",
        )


system = FourthOrderUniversalRobustSystem


__all__ = [
    "SystemParams",
    "FourthOrderUniversalRobustSystem",
    "system",
    "load_control_npz",
    "sample_unit_vectors_spherical",
    "FOURTH_ORDER_BEST_CONTROLS",
    "REFERENCE_DIR",
]
