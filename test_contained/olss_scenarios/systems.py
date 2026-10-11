"""OLSS scenario systems used to check the shooting tools end to end.

Problems taken from ``~/phd/rob_unversal_article`` (problem statements only):

- ``BVP_problem/intro/latex/sections/05_cart_worked.tex``: the minimum-energy
  cart by the indirect route and its three variants (integral ledger, free
  arrival speed, bounded force with minimum time).
- ``.../06_two_routes.tex``: the same cart by the direct route (square and
  oversized polynomial families, energy as the selection cost).
- ``.../03_continuation.tex``: pendulum swing-up reached by continuation in
  the target fraction ``s``.
- ``shooting/second_order/latex/shooting_methods.tex``: hard minimum-energy
  universally robust qubit transfer |0> -> |1>, reduced eight-variable,
  nine-residual formulation, and its non-robust limit with the rotational
  gauge left free.

Generic scenarios: the Bratu fold, an unstable linear boundary-value problem
(single and multiple shooting), an inconsistent overdetermined system, badly
scaled variables, bounds, and a deliberately wrong Jacobian.

The ``Article*ShapeSystem`` classes below are dimension- and workflow-matched
contract probes, not replacements for the article's physical models.  They
exercise the exact rectangular/square layouts documented for the gate,
fourth-order, and planar sixth-order cases with deterministic known roots.

Every system follows the OLGS-style dictionary construction
``System(primary={...}, secondary={...})``.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from optimizer.system_olss import (
    OLSS,
    ShootingBlock,
    ShootingSpec,
    ShootingVerification,
    basic_shooting_system_spec,
    shooting_metric_spec,
)


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


class ScenarioSystem(OLSS):
    """Dictionary-configured OLSS base with a one-entry computation cache."""

    system_id = "scenario"
    system_type = "shooting"
    PRIMARY: dict[str, Any] = {}
    SECONDARY: dict[str, Any] = {}
    residual_size = 1
    jacobian_source = "analytical"
    extra_metrics: dict[str, Any] = {}
    selection_metric: str | None = None

    def __init__(
        self,
        primary: Mapping[str, Any] | None = None,
        secondary: Mapping[str, Any] | None = None,
    ) -> None:
        self.primary = self._merge(self.PRIMARY, primary, "primary")
        self.secondary = self._merge(self.SECONDARY, secondary, "secondary")
        self._spec = self.build_spec()
        self._last: tuple[Any, dict[str, Any]] | None = None

    @staticmethod
    def _merge(defaults: Mapping[str, Any], updates: Mapping[str, Any] | None, label: str) -> dict[str, Any]:
        updates = dict(updates or {})
        unknown = sorted(set(updates) - set(defaults))
        if unknown:
            raise KeyError(f"Unknown {label} parameters: {unknown}")
        merged = dict(defaults)
        merged.update(updates)
        return merged

    def build_spec(self) -> ShootingSpec:
        raise NotImplementedError

    def shooting_spec(self) -> ShootingSpec:
        return self._spec

    def capabilities(self) -> dict[str, Any]:
        return {"accuracy_option": "steps"}

    def verification_default_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {"profile": "independent"}
        if "steps" in self.primary:
            options["steps"] = 4 * int(self.primary["steps"])
        return options

    def system_spec(self) -> dict[str, Any]:
        return basic_shooting_system_spec(
            self.system_id,
            shooting_spec=self._spec,
            residual_size=self.residual_size,
            system_type=self.system_type,
            metrics=dict(self.extra_metrics) or None,
            parameters={"primary": dict(self.primary), "secondary": dict(self.secondary)},
            jacobian_source=self.jacobian_source,
            selection_metric=self.selection_metric,
            capabilities=self.capabilities(),
            verification_options=self.verification_default_options(),
        )

    def with_secondary(self, **updates: Any) -> "ScenarioSystem":
        return type(self)(self.primary, {**self.secondary, **updates})

    def _compute(self, z: np.ndarray, options: dict[str, Any], need_jacobian: bool) -> dict[str, Any]:
        raise NotImplementedError

    def _cached(self, variables: Any, options: Mapping[str, Any] | None, need_jacobian: bool) -> dict[str, Any]:
        opts = dict(options or {})
        key = (variables.flatten(copy=False).tobytes(), json.dumps(opts, sort_keys=True, default=str))
        if self._last is not None and self._last[0] == key:
            if not need_jacobian or self._last[1].get("jacobian") is not None:
                return self._last[1]
        payload = self._compute(variables.flatten(), opts, need_jacobian)
        self._last = (key, payload)
        return payload

    def residuals(self, variables, *, options=None):
        return np.array(self._cached(variables, options, False)["residuals"], dtype=float)

    def jacobian(self, variables, *, options=None):
        return np.array(self._cached(variables, options, True)["jacobian"], dtype=float)

    def residuals_and_jacobian(self, variables, *, options=None):
        payload = self._cached(variables, options, True)
        return np.array(payload["residuals"], dtype=float), np.array(payload["jacobian"], dtype=float)

    def evaluate(self, variables, *, options=None):
        payload = self._cached(variables, options, False)
        return {**self._residual_metrics(payload["residuals"]), **payload.get("metrics", {})}

    def verify(self, variables, *, options=None):
        opts = dict(options or {})
        payload = self._compute(variables.flatten(), opts, False)
        return ShootingVerification(
            residuals=payload["residuals"],
            metrics={**self._residual_metrics(payload["residuals"]), **payload.get("metrics", {})},
            options=opts,
        )


def _energy_metric(description: str = "Control energy") -> dict[str, Any]:
    return {"energy": shooting_metric_spec(description, role="selection", unit="arb")}


def _batch_central_jacobian(fn, z: np.ndarray, rel_step: float = 1.0e-6) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate ``fn`` on z and its central-difference stencil in one batch."""

    n = z.size
    steps = rel_step * np.maximum(1.0, np.abs(z))
    batch = np.repeat(z[None, :], 2 * n + 1, axis=0)
    batch[1 : n + 1] += np.diag(steps)
    batch[n + 1 :] -= np.diag(steps)
    values = fn(batch)
    jac = (values[1 : n + 1] - values[n + 1 :]).T / (2.0 * steps)
    return values[0], jac


# ---------------------------------------------------------------------------
# Cart, indirect route and variants (05_cart_worked.tex)
# ---------------------------------------------------------------------------


class CartMinEnergySystem(ScenarioSystem):
    """Rest-to-rest cart, unknowns (mu_x, mu_v(0)); u = mu_v / 2."""

    system_id = "cart-min-energy-indirect"
    system_type = "linear-ode-indirect"
    PRIMARY = {"final_time": 1.0}
    residual_size = 2
    extra_metrics = _energy_metric()
    selection_metric = "energy"

    def build_spec(self):
        return ShootingSpec(
            (
                ShootingBlock("mu_x", meaning="frozen position costate", role="costate"),
                ShootingBlock("mu_v0", meaning="initial velocity costate", role="costate"),
            )
        )

    def _compute(self, z, options, need_jacobian):
        T = float(self.primary["final_time"])
        mu_x, mu_v0 = z
        x_T = 0.5 * (mu_v0 * T**2 / 2.0 - mu_x * T**3 / 6.0)
        v_T = 0.5 * (mu_v0 * T - mu_x * T**2 / 2.0)
        energy = 0.25 * (mu_v0**2 * T - mu_v0 * mu_x * T**2 + mu_x**2 * T**3 / 3.0)
        jac = np.array([[-T**3 / 12.0, T**2 / 4.0], [-T**2 / 4.0, T / 2.0]])
        return {"residuals": np.array([x_T - 1.0, v_T]), "jacobian": jac, "metrics": {"energy": float(energy)}}


def _poly_cart(coefficients_u: np.ndarray, T: float) -> dict[str, float]:
    """Integrate x'' = u for a polynomial u(t) = sum c_k t^k from rest."""

    u = np.polynomial.Polynomial(coefficients_u)
    v = u.integ()
    x = v.integ()
    z = x.integ()
    energy = (u * u).integ()
    return {"x": float(x(T)), "v": float(v(T)), "z": float(z(T)), "u_T": float(u(T)), "energy": float(energy(T))}


class CartLedgerSystem(ScenarioSystem):
    """Variant 1: integral demand int_0^T x dt = c through a ledger state."""

    system_id = "cart-integral-ledger"
    system_type = "linear-ode-indirect"
    PRIMARY = {"final_time": 1.0, "ledger_target": 0.3}
    residual_size = 3
    extra_metrics = _energy_metric()
    selection_metric = "energy"

    def build_spec(self):
        return ShootingSpec(
            (
                ShootingBlock("mu_x0", role="costate"),
                ShootingBlock("mu_v0", role="costate"),
                ShootingBlock("mu_z", meaning="frozen ledger price", role="multiplier"),
            )
        )

    def _residual_vector(self, z):
        T = float(self.primary["final_time"])
        mu_x0, mu_v0, mu_z = z
        # mu_x = mu_x0 - mu_z t, mu_v = mu_v0 - mu_x0 t + mu_z t^2 / 2, u = mu_v / 2.
        state = _poly_cart(np.array([mu_v0, -mu_x0, mu_z / 2.0]) / 2.0, T)
        return np.array([state["x"] - 1.0, state["v"], state["z"] - self.primary["ledger_target"]]), state

    def _compute(self, z, options, need_jacobian):
        residuals, state = self._residual_vector(z)
        jac = None
        if need_jacobian:
            base = self._residual_vector(np.zeros(3))[0]
            jac = np.column_stack([self._residual_vector(e)[0] - base for e in np.eye(3)])
        return {"residuals": residuals, "jacobian": jac, "metrics": {"energy": state["energy"]}}


class CartFreeSpeedSystem(ScenarioSystem):
    """Variant 2: arrival speed free, so mu_v(T) = 0 replaces v(T) = 0."""

    system_id = "cart-free-arrival-speed"
    system_type = "linear-ode-indirect"
    PRIMARY = {"final_time": 1.0}
    residual_size = 2
    extra_metrics = _energy_metric()
    selection_metric = "energy"

    def build_spec(self):
        return ShootingSpec((ShootingBlock("mu_x", role="costate"), ShootingBlock("mu_v0", role="costate")))

    def _compute(self, z, options, need_jacobian):
        T = float(self.primary["final_time"])
        mu_x, mu_v0 = z
        state = _poly_cart(np.array([mu_v0, -mu_x]) / 2.0, T)
        residuals = np.array([state["x"] - 1.0, mu_v0 - mu_x * T])
        jac = np.array([[-T**3 / 12.0, T**2 / 4.0], [-T, 1.0]])
        return {"residuals": residuals, "jacobian": jac, "metrics": {"energy": state["energy"]}}


class CartMinTimeBangBangSystem(ScenarioSystem):
    """Variant 3: |u| <= 1, minimum time; unknowns (mu_x, mu_v(0), T); H(T) = 0."""

    system_id = "cart-min-time-bang-bang"
    system_type = "switching-indirect-free-time"
    residual_size = 3
    jacobian_source = "numerical-declared"
    extra_metrics = {"final_time": shooting_metric_spec("Final time", role="selection")}
    selection_metric = "final_time"

    def build_spec(self):
        return ShootingSpec(
            (
                ShootingBlock("mu_x", role="costate"),
                ShootingBlock("mu_v0", role="costate"),
                ShootingBlock("final_time", unit="s", role="free_final_time"),
            )
        )

    def capabilities(self):
        return {
            **super().capabilities(),
            "jacobian_method": "central differences, relative step 1e-6",
        }

    @staticmethod
    def _batch_residuals(Z):
        out = np.empty((Z.shape[0], 3))
        for row, (mu_x, mu_v0, T) in enumerate(Z):
            first = 1.0 if mu_v0 > 0 else -1.0
            t_s = mu_v0 / mu_x if mu_x != 0 else np.inf
            t1 = float(np.clip(t_s, 0.0, T)) if np.isfinite(t_s) else T
            v1 = first * t1
            x1 = 0.5 * first * t1**2
            dt2 = T - t1
            x_T = x1 + v1 * dt2 - 0.5 * first * dt2**2
            v_T = v1 - first * dt2
            mu_v_T = mu_v0 - mu_x * T
            out[row] = (x_T - 1.0, v_T, mu_x * v_T + abs(mu_v_T) - 1.0)
        return out

    def _compute(self, z, options, need_jacobian):
        if need_jacobian:
            residuals, jac = _batch_central_jacobian(self._batch_residuals, z)
        else:
            residuals, jac = self._batch_residuals(z[None, :])[0], None
        return {"residuals": residuals, "jacobian": jac, "metrics": {"final_time": float(z[2])}}


# ---------------------------------------------------------------------------
# Cart, direct route (06_two_routes.tex)
# ---------------------------------------------------------------------------


class CartDirectPolynomialSystem(ScenarioSystem):
    """u(t; p) = sum_k p_k t^k with demands x(T) = 1, v(T) = 0; cost = energy."""

    system_id = "cart-direct-polynomial"
    system_type = "linear-direct-route"
    PRIMARY = {"final_time": 1.0, "coefficients": 3}
    residual_size = 2
    extra_metrics = _energy_metric()
    selection_metric = "energy"

    def build_spec(self):
        n = int(self.primary["coefficients"])
        return ShootingSpec((ShootingBlock("coefficients", shape=(n,), meaning="polynomial drive coefficients"),))

    def _gram(self):
        T = float(self.primary["final_time"])
        n = int(self.primary["coefficients"])
        idx = np.arange(n)
        return T ** (idx[:, None] + idx[None, :] + 1) / (idx[:, None] + idx[None, :] + 1)

    def _compute(self, z, options, need_jacobian):
        T = float(self.primary["final_time"])
        k = np.arange(z.size)
        jac = np.vstack((T ** (k + 2) / ((k + 1) * (k + 2)), T ** (k + 1) / (k + 1)))
        residuals = jac @ z - np.array([1.0, 0.0])
        energy = float(z @ self._gram() @ z)
        return {"residuals": residuals, "jacobian": jac, "metrics": {"energy": energy}}

    def cost_gradient(self, variables, *, options=None):
        return 2.0 * self._gram() @ variables.flatten()


# ---------------------------------------------------------------------------
# Pendulum swing-up with continuation (03_continuation.tex)
# ---------------------------------------------------------------------------


def _pendulum_terminal(p: float, T: float, steps: int) -> tuple[float, float]:
    """Return y(T) and dy(T)/dp for y'' = -sin y, y(0) = 0, y'(0) = p."""

    state = np.array([0.0, p, 0.0, 1.0])
    dt = T / steps

    def rhs(s):
        y, w, a, b = s
        return np.array([w, -np.sin(y), b, -np.cos(y) * a])

    for _ in range(steps):
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return float(state[0]), float(state[2])


class PendulumSwingUpSystem(ScenarioSystem):
    """Unknowns (p = y'(0), s); residual y(T; p) - s A; s is a continuation block."""

    system_id = "pendulum-swing-up-arclength"
    system_type = "nonlinear-ode-boundary-value"
    jacobian_source = "variational"
    PRIMARY = {"final_time": 2.0, "target_angle": float(np.pi), "steps": 400}
    residual_size = 1

    def build_spec(self):
        return ShootingSpec(
            (
                ShootingBlock("initial_velocity", unit="rad/s"),
                ShootingBlock("target_fraction", role="continuation_parameter"),
            )
        )

    def _compute(self, z, options, need_jacobian):
        steps = int(options.get("steps", self.primary["steps"]))
        y_T, dy_dp = _pendulum_terminal(float(z[0]), float(self.primary["final_time"]), steps)
        A = float(self.primary["target_angle"])
        return {"residuals": np.array([y_T - z[1] * A]), "jacobian": np.array([[dy_dp, -A]])}


class PendulumFixedTargetSystem(ScenarioSystem):
    """Unknown p only; the target fraction s lives in secondary (natural continuation)."""

    system_id = "pendulum-swing-up-natural"
    system_type = "nonlinear-ode-boundary-value"
    jacobian_source = "variational"
    PRIMARY = {"final_time": 2.0, "target_angle": float(np.pi), "steps": 400}
    SECONDARY = {"target_fraction": 0.1}
    residual_size = 1

    def build_spec(self):
        return ShootingSpec((ShootingBlock("initial_velocity", unit="rad/s"),))

    def _compute(self, z, options, need_jacobian):
        steps = int(options.get("steps", self.primary["steps"]))
        y_T, dy_dp = _pendulum_terminal(float(z[0]), float(self.primary["final_time"]), steps)
        target = float(self.secondary["target_fraction"]) * float(self.primary["target_angle"])
        return {"residuals": np.array([y_T - target]), "jacobian": np.array([[dy_dp]])}


# ---------------------------------------------------------------------------
# Robust qubit transfer (shooting_methods.tex)
# ---------------------------------------------------------------------------


def _pauli(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b = v[:, 0], v[:, 1]
    return (
        np.stack((b, a), axis=1),
        np.stack((-1j * b, 1j * a), axis=1),
        np.stack((a, -b), axis=1),
    )


def _controls(psi: np.ndarray, chi: np.ndarray) -> np.ndarray:
    """u_j = Im <chi | sigma_j | psi> (lambda_0 = 1/4 normalization)."""

    return np.stack([np.imag(np.sum(np.conj(chi) * s, axis=1)) for s in _pauli(psi)], axis=1)


def _extremal_rhs(Y: np.ndarray, p: np.ndarray) -> np.ndarray:
    psi, perp, chi = Y[:, 0:2], Y[:, 2:4], Y[:, 7:9]
    u = _controls(psi, chi)

    def minus_i_h(v):
        sx, sy, sz = _pauli(v)
        return -0.5j * (u[:, 0:1] * sx + u[:, 1:2] * sy + u[:, 2:3] * sz)

    s_perp = _pauli(perp)
    out = np.empty_like(Y)
    out[:, 0:2] = minus_i_h(psi)
    out[:, 2:4] = minus_i_h(perp)
    out[:, 4:7] = np.stack([np.sum(np.conj(perp) * s, axis=1) for s in _pauli(psi)], axis=1)
    out[:, 7:9] = minus_i_h(chi) + 2.0 * (p[:, 0:1] * s_perp[0] + p[:, 1:2] * s_perp[1] + p[:, 2:3] * s_perp[2])
    out[:, 9] = np.sum(u * u, axis=1)
    return out


def _integrate_extremal(chi0: np.ndarray, p: np.ndarray, final_time: float, steps: int, record: bool = False):
    """Batch RK4 for the hard extremal system; state = (psi0, psi_perp, F, chi, E)."""

    B = chi0.shape[0]
    Y = np.zeros((B, 10), dtype=complex)
    Y[:, 0] = 1.0  # |psi_0(0)> = |0>
    Y[:, 3] = 1.0  # |psi_perp(0)> = C|0> = |1>
    Y[:, 7:9] = chi0
    dt = final_time / steps
    samples = [_controls(Y[:, 0:2], Y[:, 7:9])] if record else None
    for _ in range(steps):
        k1 = _extremal_rhs(Y, p)
        k2 = _extremal_rhs(Y + 0.5 * dt * k1, p)
        k3 = _extremal_rhs(Y + 0.5 * dt * k2, p)
        k4 = _extremal_rhs(Y + dt * k3, p)
        Y = Y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if record:
            samples.append(_controls(Y[:, 0:2], Y[:, 7:9]))
    return Y, (np.stack(samples, axis=-1) if record else None)


def _bloch(psi: np.ndarray) -> np.ndarray:
    return np.stack([np.real(np.sum(np.conj(psi) * s, axis=1)) for s in _pauli(psi)], axis=1)


class RobustTransferSecondOrderSystem(ScenarioSystem):
    """Hard robust |0> -> |1>: z = (u_y(0), u_z(0), Re/Im p_x, p_y, p_z); 9 residuals."""

    system_id = "robust-transfer-second-order-hard-pmp"
    system_type = "qubit-indirect-pmp"
    jacobian_source = "numerical-declared"
    PRIMARY = {"final_time": 1.0, "steps": 1500}
    residual_size = 9
    extra_metrics = {
        **_energy_metric("E = int ||u||^2 dt"),
        "fidelity": shooting_metric_spec("Nominal transfer fidelity", role="diagnostic"),
        "ledger_norm": shooting_metric_spec("||F(t_f)||_2", role="diagnostic"),
    }
    selection_metric = "energy"

    def build_spec(self):
        return ShootingSpec(
            (
                ShootingBlock("u_y0", meaning="initial control, gauge u_x(0) = 0", scale=10.0),
                ShootingBlock("u_z0", meaning="initial control z component", scale=1.0),
                ShootingBlock("p", shape=(6,), meaning="robustness multipliers (Re, Im) x, y, z", scale=30.0),
            )
        )

    def capabilities(self):
        return {
            **super().capabilities(),
            "jacobian_method": "batched central differences",
        }

    def _split(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        chi0 = np.stack((-1j * Z[:, 1], Z[:, 0] + 0j), axis=1)
        p = Z[:, 2::2] + 1j * Z[:, 3::2]
        return chi0, p

    def _terminal(self, Z: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
        chi0, p = self._split(Z)
        Y, _ = _integrate_extremal(chi0, p, float(self.primary["final_time"]), steps)
        psi, F, chi = Y[:, 0:2], Y[:, 4:7], Y[:, 7:9]
        b = -psi[:, 0]  # <psi_f^perp| psi(t_f)>, psi_f^perp = C|1> = -|0>
        u_dot_r = np.sum(_controls(psi, chi) * _bloch(psi), axis=1)
        residuals = np.column_stack((b.real, b.imag, F.real, F.imag, u_dot_r))
        extras = np.column_stack((Y[:, 9].real, np.abs(psi[:, 1]) ** 2, np.linalg.norm(F, axis=1)))
        return residuals, extras

    def _compute(self, z, options, need_jacobian):
        steps = int(options.get("steps", self.primary["steps"]))
        if need_jacobian:
            n = z.size
            stencil = 1.0e-6 * np.maximum(1.0, np.abs(z))
            batch = np.repeat(z[None, :], 2 * n + 1, axis=0)
            batch[1 : n + 1] += np.diag(stencil)
            batch[n + 1 :] -= np.diag(stencil)
            residuals, extras = self._terminal(batch, steps)
            jac = (residuals[1 : n + 1] - residuals[n + 1 :]).T / (2.0 * stencil)
            residuals, extras = residuals[0], extras[0]
        else:
            residuals, extras = self._terminal(z[None, :], steps)
            residuals, extras, jac = residuals[0], extras[0], None
        metrics = {"energy": float(extras[0]), "fidelity": float(extras[1]), "ledger_norm": float(extras[2])}
        return {"residuals": residuals, "jacobian": jac, "metrics": metrics}

    def recover_controls(self, variables, *, times):
        times = np.asarray(times, dtype=float)
        steps = times.size - 1
        if not np.allclose(np.diff(times), times[1] - times[0]) or abs(times[-1] - self.primary["final_time"]) > 1e-12:
            raise ValueError("recover_controls expects a uniform grid on [0, t_f].")
        chi0, p = self._split(variables.flatten()[None, :])
        _, samples = _integrate_extremal(chi0, p, float(self.primary["final_time"]), steps, record=True)
        return {"times": times, "channels": ("ux", "uy", "uz"), "values": samples[0]}


class QubitMinEnergyTransferSystem(ScenarioSystem):
    """Non-robust limit (p = 0) with the rotational gauge free: z = chi(0) in C^2."""

    system_id = "qubit-min-energy-transfer-gauge-free"
    system_type = "qubit-indirect-pmp"
    jacobian_source = "numerical-declared"
    PRIMARY = {"final_time": 1.0, "steps": 600}
    residual_size = 4
    extra_metrics = _energy_metric("E = int ||u||^2 dt")
    selection_metric = "energy"

    def build_spec(self):
        return ShootingSpec((ShootingBlock("chi0", shape=(4,), meaning="initial costate (Re chi1, Im chi1, Re chi2, Im chi2)"),))

    def _terminal(self, Z, steps):
        chi0 = Z[:, 0::2] + 1j * Z[:, 1::2]
        Y, _ = _integrate_extremal(chi0, np.zeros((Z.shape[0], 3), dtype=complex), float(self.primary["final_time"]), steps)
        psi, chi = Y[:, 0:2], Y[:, 7:9]
        b = -psi[:, 0]
        eta = np.sum(np.conj(psi) * chi, axis=1)
        return np.column_stack((b.real, b.imag, eta.real, eta.imag)), Y[:, 9].real

    def _compute(self, z, options, need_jacobian):
        steps = int(options.get("steps", self.primary["steps"]))
        fn = lambda Z: self._terminal(Z, steps)[0]
        if need_jacobian:
            residuals, jac = _batch_central_jacobian(fn, z)
        else:
            residuals, jac = fn(z[None, :])[0], None
        energy = float(self._terminal(z[None, :], steps)[1][0])
        return {"residuals": residuals, "jacobian": jac, "metrics": {"energy": energy}}

    def roots_equivalent(self, left, right, *, tolerance):
        """Identify costates that differ only by their unobservable global phase."""

        left_flat = left.flatten(copy=False)
        right_flat = right.flatten(copy=False)
        left_complex = left_flat[0::2] + 1j * left_flat[1::2]
        right_complex = right_flat[0::2] + 1j * right_flat[1::2]
        overlap = np.vdot(right_complex, left_complex)
        if abs(overlap) > np.finfo(float).tiny:
            right_complex = right_complex * (overlap / abs(overlap))
        delta = float(np.linalg.norm(left_complex - right_complex))
        scale = max(
            1.0,
            float(np.linalg.norm(left_complex)),
            float(np.linalg.norm(right_complex)),
        )
        return delta <= float(tolerance) * scale


# ---------------------------------------------------------------------------
# Article-dimension contract probes (known-root structural systems)
# ---------------------------------------------------------------------------


class _ArticleShapeSystem(ScenarioSystem):
    """Deterministic full-column-rank system for an article shooting layout."""

    variable_size = 1
    residual_size = 1
    target_scale = 1.0
    extra_metrics = _energy_metric("Quadratic probe cost")
    selection_metric = "energy"

    def build_spec(self):
        return ShootingSpec(
            (
                ShootingBlock(
                    "shooting_data",
                    shape=(self.variable_size,),
                    meaning="dimension-matched article shooting variables",
                    role="article_shape_probe",
                ),
            )
        )

    def _matrix(self) -> np.ndarray:
        rows = np.arange(1, self.residual_size + 1, dtype=float)[:, None]
        cols = np.arange(1, self.variable_size + 1, dtype=float)[None, :]
        matrix = np.sin(0.37 * rows * cols) + np.cos(0.19 * rows * (cols + 1.0))
        matrix[: self.variable_size] += 2.0 * np.eye(self.variable_size)
        return matrix

    def target(self) -> np.ndarray:
        return self.target_scale * np.linspace(-0.8, 0.9, self.variable_size)

    def _compute(self, z, options, need_jacobian):
        matrix = self._matrix()
        delta = np.asarray(z, dtype=float) - self.target()
        return {
            "residuals": matrix @ delta,
            "jacobian": matrix,
            "metrics": {"energy": float(np.dot(z, z))},
        }


class ArticleGateTwelveSystem(_ArticleShapeSystem):
    """Square 12-variable/12-residual hard-gate contract probe."""

    system_id = "article-hard-gate-12x12-shape"
    system_type = "article-shape-probe"
    variable_size = 12
    residual_size = 12


class ArticleFourthOrderTwentySystem(_ArticleShapeSystem):
    """Full fourth-order 20-variable/21-residual contract probe."""

    system_id = "article-fourth-order-20x21-shape"
    system_type = "article-shape-probe"
    variable_size = 20
    residual_size = 21
    jacobian_source = "variational"


class ArticleSixthOrderPlanarSystem(_ArticleShapeSystem):
    """Planar sixth-order 4-variable/4-residual contract probe."""

    system_id = "article-sixth-order-planar-4x4-shape"
    system_type = "article-shape-probe"
    variable_size = 4
    residual_size = 4


# ---------------------------------------------------------------------------
# Generic scenarios
# ---------------------------------------------------------------------------


def _bratu_terminal(s: float, lam: float, steps: int) -> np.ndarray:
    """y'' = -lam e^y with sensitivities to s = y'(0) and lam; returns (y, dy/ds, dy/dlam) at 1."""

    state = np.array([0.0, s, 0.0, 1.0, 0.0, 0.0])
    dt = 1.0 / steps

    def rhs(v):
        y, w, a, da, q, dq = v
        e = np.exp(y)
        return np.array([w, -lam * e, da, -lam * e * a, dq, -e - lam * e * q])

    for _ in range(steps):
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state[[0, 2, 4]]


class BratuSystem(ScenarioSystem):
    """Bratu y'' + lam e^y = 0, y(0) = y(1) = 0; unknowns (y'(0), lam); fold at lam ~ 3.5138."""

    system_id = "bratu-fold"
    system_type = "nonlinear-ode-boundary-value"
    jacobian_source = "variational"
    PRIMARY = {"steps": 400}
    residual_size = 1

    def build_spec(self):
        return ShootingSpec((ShootingBlock("initial_slope"), ShootingBlock("lambda", role="continuation_parameter")))

    def _compute(self, z, options, need_jacobian):
        y1, dys, dyl = _bratu_terminal(float(z[0]), float(z[1]), int(options.get("steps", self.primary["steps"])))
        return {"residuals": np.array([y1]), "jacobian": np.array([[dys, dyl]])}


class BratuFixedLambdaSystem(ScenarioSystem):
    """Same problem with lam in secondary: natural continuation only."""

    system_id = "bratu-natural"
    system_type = "nonlinear-ode-boundary-value"
    jacobian_source = "variational"
    PRIMARY = {"steps": 400}
    SECONDARY = {"lam": 0.5}
    residual_size = 1

    def build_spec(self):
        return ShootingSpec((ShootingBlock("initial_slope"),))

    def _compute(self, z, options, need_jacobian):
        y1, dys, _ = _bratu_terminal(float(z[0]), float(self.secondary["lam"]), int(options.get("steps", self.primary["steps"])))
        return {"residuals": np.array([y1]), "jacobian": np.array([[dys]])}


def _hyperbolic_map(k: float, h: float) -> np.ndarray:
    return np.array([[np.cosh(k * h), np.sinh(k * h) / k], [k * np.sinh(k * h), np.cosh(k * h)]])


class UnstableSingleShootingSystem(ScenarioSystem):
    """y'' = k^2 y on [0, 1], y(0) = 1, y(1) = 0; one unknown y'(0)."""

    system_id = "unstable-linear-single"
    system_type = "linear-ode-boundary-value"
    PRIMARY = {"k": 40.0}
    residual_size = 1

    def build_spec(self):
        return ShootingSpec((ShootingBlock("initial_slope"),))

    def _compute(self, z, options, need_jacobian):
        M = _hyperbolic_map(float(self.primary["k"]), 1.0)
        y1 = M[0, 0] * 1.0 + M[0, 1] * z[0]
        return {"residuals": np.array([y1]), "jacobian": np.array([[M[0, 1]]])}


class UnstableMultipleShootingSystem(ScenarioSystem):
    """Same BVP with K segments: unknowns y'(0) and interior node states (y, y')."""

    system_id = "unstable-linear-multiple"
    system_type = "linear-ode-boundary-value"
    PRIMARY = {"k": 40.0, "segments": 20}

    def __init__(self, primary=None, secondary=None):
        super().__init__(primary, secondary)
        self.residual_size = 2 * (int(self.primary["segments"]) - 1) + 1

    def build_spec(self):
        K = int(self.primary["segments"])
        return ShootingSpec(
            (
                ShootingBlock("initial_slope"),
                ShootingBlock("nodes", shape=(K - 1, 2), meaning="interior node states (y, y')", role="node_state"),
            )
        )

    def _compute(self, z, options, need_jacobian):
        K = int(self.primary["segments"])
        M = _hyperbolic_map(float(self.primary["k"]), 1.0 / K)
        nodes = z[1:].reshape(K - 1, 2)
        starts = np.vstack(([1.0, z[0]], nodes))
        ends = starts @ M.T
        residuals = np.concatenate(((ends[:-1] - nodes).reshape(-1), [ends[-1, 0]]))
        n = z.size
        jac = np.zeros((self.residual_size, n))
        for seg in range(K - 1):  # continuity rows: M @ start_seg - node_seg
            rows = slice(2 * seg, 2 * seg + 2)
            if seg == 0:
                jac[rows, 0] = M[:, 1]
            else:
                jac[rows, 1 + 2 * (seg - 1) : 1 + 2 * seg] = M
            jac[rows, 1 + 2 * seg : 3 + 2 * seg] = -np.eye(2)
        jac[-1, n - 2 : n] = M[0]
        return {"residuals": residuals, "jacobian": jac}


class InconsistentOverdeterminedSystem(ScenarioSystem):
    """One unknown, residuals (x - 1, x - 2): no exact root, least squares at 1.5."""

    system_id = "inconsistent-overdetermined"
    residual_size = 2

    def build_spec(self):
        return ShootingSpec((ShootingBlock("x"),))

    def _compute(self, z, options, need_jacobian):
        return {"residuals": np.array([z[0] - 1.0, z[0] - 2.0]), "jacobian": np.array([[1.0], [1.0]])}


class BadlyScaledSystem(ScenarioSystem):
    """Two roots (1e6, 1e-3) and (1e6, 2e-3); declared scales 1e6 and 1e-3."""

    system_id = "badly-scaled-two-roots"
    residual_size = 2

    def build_spec(self):
        return ShootingSpec((ShootingBlock("a", scale=1.0e6), ShootingBlock("b", scale=1.0e-3)))

    def _compute(self, z, options, need_jacobian):
        a, b = z[0] / 1.0e6, z[1] / 1.0e-3
        residuals = np.array([a - 1.0, (b - 1.0) * (b - 2.0)])
        jac = np.array([[1.0e-6, 0.0], [0.0, (2.0 * b - 3.0) / 1.0e-3]])
        return {"residuals": residuals, "jacobian": jac}


class BadlyScaledLineSystem(ScenarioSystem):
    """One residual a/1e6 + b/1e-3 - 2 = 0: a root line; scaled min-norm root is (1e6, 1e-3)."""

    system_id = "badly-scaled-line"
    residual_size = 1

    def build_spec(self):
        return ShootingSpec((ShootingBlock("a", scale=1.0e6), ShootingBlock("b", scale=1.0e-3)))

    def _compute(self, z, options, need_jacobian):
        return {"residuals": np.array([z[0] / 1.0e6 + z[1] / 1.0e-3 - 2.0]), "jacobian": np.array([[1.0e-6, 1.0e3]])}


class BoundedSystem(ScenarioSystem):
    """x in [0, 1]; residual x^2 - target^2 (target in secondary)."""

    system_id = "bounded-root"
    residual_size = 1
    SECONDARY = {"target": 0.5}

    def build_spec(self):
        return ShootingSpec((ShootingBlock("x"),))

    def shooting_bounds(self):
        return np.array([0.0]), np.array([1.0])

    def _compute(self, z, options, need_jacobian):
        t = float(self.secondary["target"])
        return {"residuals": np.array([z[0] ** 2 - t**2]), "jacobian": np.array([[2.0 * z[0]]])}


class WrongJacobianSystem(ScenarioSystem):
    """Residual sin x - 0.5 with a Jacobian that is wrong by a factor of 2."""

    system_id = "wrong-jacobian"
    residual_size = 1

    def build_spec(self):
        return ShootingSpec((ShootingBlock("x"),))

    def _compute(self, z, options, need_jacobian):
        return {"residuals": np.array([np.sin(z[0]) - 0.5]), "jacobian": np.array([[2.0 * np.cos(z[0])]])}


def make_robust_transfer_system() -> RobustTransferSecondOrderSystem:
    """Importable, picklable factory for process-parallel multistart."""

    return RobustTransferSecondOrderSystem()
