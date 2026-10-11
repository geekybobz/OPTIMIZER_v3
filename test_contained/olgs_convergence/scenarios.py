"""OLGS convergence scenarios from OLGS_CONVERGENCE_TEST_DESIGN.md, plus extra stress cases.

S0 isotropic quadratic, S1 rotated ill-conditioned quadratic, S2 extended
Rosenbrock, S3 double well (negative curvature), S4 channel-scale and grid
family, S5 bounds and equality constraints, S6 smooth plus L1 composite, and
the extra cases X1 large dimension, X2 restricted domain, X3 inconsistent
gradient. S7 workflows and the physical S8/S9 adapters live in the runner.

Every system is a plain OLGS object built as ``System(primary={...},
secondary={...})`` and works in flat control coordinates internally.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from optimizer.controls import ControlSpec, Controls
from optimizer.system_olgs import basic_system_spec


def _merge(defaults: Mapping[str, Any], updates: Mapping[str, Any] | None, label: str) -> dict[str, Any]:
    updates = dict(updates or {})
    unknown = sorted(set(updates) - set(defaults))
    if unknown:
        raise KeyError(f"Unknown {label} parameters: {unknown}")
    return {**defaults, **updates}


class VectorSystem:
    """OLGS base over ``len(keys) x N`` controls; subclasses define f and g (and optionally hv)."""

    system_id = "vector"
    keys: tuple[str, ...] = ("u",)
    PRIMARY: dict[str, Any] = {"N": 20}
    SECONDARY: dict[str, Any] = {}
    direction = "minimize"

    def __init__(self, primary: Mapping[str, Any] | None = None, secondary: Mapping[str, Any] | None = None) -> None:
        self.primary = _merge(self.PRIMARY, primary, "primary")
        self.secondary = _merge(self.SECONDARY, secondary, "secondary")
        self.N = int(self.primary["N"])
        self.size = len(self.keys) * self.N
        self.setup()

    # -- configuration -----------------------------------------------------
    def setup(self) -> None:
        """Precompute scenario data."""

    def time_step(self) -> float | None:
        return None

    def control_spec(self) -> ControlSpec:
        return ControlSpec(keys=self.keys, control_dim=self.N, dtype=float, dt=self.time_step())

    def extra_metric_declarations(self) -> dict[str, Any]:
        return {}

    def spec_extras(self) -> dict[str, Any]:
        return {}

    def system_spec(self) -> dict[str, Any]:
        return basic_system_spec(
            self.system_id,
            control_spec=self.control_spec(),
            objective_direction=self.direction,
            metrics=self.extra_metric_declarations() or None,
            parameters={"primary": dict(self.primary), "secondary": dict(self.secondary)},
            **self.spec_extras(),
        )

    def with_secondary(self, **updates: Any) -> "VectorSystem":
        return type(self)(self.primary, {**self.secondary, **updates})

    # -- math (flat coordinates) ---------------------------------------------
    def f(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def g(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def extra_metrics(self, x: np.ndarray) -> dict[str, float]:
        return {}

    def sign(self) -> float:
        return -1.0 if self.direction == "maximize" else 1.0

    # -- OLGS transactions ---------------------------------------------------------
    def controls(self, x: Any, name: str = "x") -> Controls:
        return Controls.from_flat(self.control_spec(), np.asarray(x, dtype=float).reshape(-1), name=name)

    def evaluate(self, controls: Controls) -> dict[str, float]:
        x = controls.flatten()
        return {"J": float(self.sign() * self.f(x)), **self.extra_metrics(x)}

    def gradient(self, controls: Controls) -> Controls:
        return self.controls(self.sign() * self.g(controls.flatten()), name="gradient")

    # -- oracle -----------------------------------------------------------------
    def oracle(self) -> dict[str, Any]:
        """Return {'x_star': array or None, 'J_star': float or None, 'kind': str}."""

        return {"x_star": None, "J_star": None, "kind": "unknown"}


class HVPMixin:
    """Adds the optional analytical Hessian-vector product capability."""

    def hv(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def hvp(self, controls: Controls, vector: Any) -> Controls:
        v = vector.flatten() if isinstance(vector, Controls) else np.asarray(vector, dtype=float).reshape(-1)
        return self.controls(self.sign() * self.hv(controls.flatten(), v), name="hvp")

    def hessian_eigenvalues(self, x: np.ndarray) -> np.ndarray:
        n = x.size
        H = np.column_stack([self.hv(x, e) for e in np.eye(n)])
        return np.linalg.eigvalsh(0.5 * (H + H.T))


# ---------------------------------------------------------------------------
# S0-S4: unconstrained smooth
# ---------------------------------------------------------------------------


class IsotropicQuadratic(HVPMixin, VectorSystem):
    """S0: f = 1/2 ||x - c||^2; maximize variant uses J = -f."""

    system_id = "s0-isotropic-quadratic"
    PRIMARY = {"N": 20}

    def setup(self):
        self.c = 0.5 * np.linspace(-1.0, 1.0, self.size)

    def f(self, x):
        d = x - self.c
        return 0.5 * float(d @ d)

    def g(self, x):
        return x - self.c

    def hv(self, x, v):
        return v.copy()

    def oracle(self):
        return {"x_star": self.c.copy(), "J_star": 0.0, "kind": "global"}


class IsotropicQuadraticMax(IsotropicQuadratic):
    system_id = "s0-isotropic-quadratic-max"
    direction = "maximize"


class RotatedQuadratic(HVPMixin, VectorSystem):
    """S1: f = 1/2 (x - c)^T A (x - c), A = Q diag(logspace(0, log10 cond)) Q^T."""

    system_id = "s1-rotated-quadratic"
    PRIMARY = {"N": 20, "condition": 1.0e4, "seed": 3}

    def setup(self):
        rng = np.random.default_rng(int(self.primary["seed"]))
        Q, _ = np.linalg.qr(rng.standard_normal((self.size, self.size)))
        lam = np.logspace(0.0, np.log10(float(self.primary["condition"])), self.size)
        self.A = (Q * lam) @ Q.T
        self.c = rng.uniform(-1.0, 1.0, self.size)

    def f(self, x):
        d = x - self.c
        return 0.5 * float(d @ self.A @ d)

    def g(self, x):
        return self.A @ (x - self.c)

    def hv(self, x, v):
        return self.A @ v

    def oracle(self):
        return {"x_star": self.c.copy(), "J_star": 0.0, "kind": "global"}


class ExtendedRosenbrock(HVPMixin, VectorSystem):
    """S2: sum over pairs of 100 (x_{2i+1} - x_{2i}^2)^2 + (1 - x_{2i})^2."""

    system_id = "s2-extended-rosenbrock"
    PRIMARY = {"N": 10}

    def f(self, x):
        a, b = x[0::2], x[1::2]
        return float(np.sum(100.0 * (b - a * a) ** 2 + (1.0 - a) ** 2))

    def g(self, x):
        a, b = x[0::2], x[1::2]
        out = np.empty_like(x)
        out[0::2] = -400.0 * a * (b - a * a) - 2.0 * (1.0 - a)
        out[1::2] = 200.0 * (b - a * a)
        return out

    def hv(self, x, v):
        a, b = x[0::2], x[1::2]
        va, vb = v[0::2], v[1::2]
        haa = 1200.0 * a * a - 400.0 * b + 2.0
        hab = -400.0 * a
        out = np.empty_like(v)
        out[0::2] = haa * va + hab * vb
        out[1::2] = hab * va + 200.0 * vb
        return out

    def start(self):
        x = np.empty(self.size)
        x[0::2], x[1::2] = -1.2, 1.0
        return x

    def oracle(self):
        return {"x_star": np.ones(self.size), "J_star": 0.0, "kind": "global"}


class DoubleWell(HVPMixin, VectorSystem):
    """S3: f = sum 1/4 (x_i^2 - 1)^2; stationary values -1, 0, +1; minima at +-1."""

    system_id = "s3-double-well"
    PRIMARY = {"N": 10}

    def f(self, x):
        return float(np.sum(0.25 * (x * x - 1.0) ** 2))

    def g(self, x):
        return x * (x * x - 1.0)

    def hv(self, x, v):
        return (3.0 * x * x - 1.0) * v

    def oracle(self):
        return {"x_star": None, "J_star": 0.0, "kind": "local: every |x_i| = 1"}


class ChannelScale(HVPMixin, VectorSystem):
    """S4: three channels weighted 1e-3, 1, 1e3 with a dt-normalized objective on N samples."""

    system_id = "s4-channel-scale"
    keys = ("slow", "mid", "fast")
    PRIMARY = {"N": 101}
    weights = np.array([1.0e-3, 1.0, 1.0e3])

    def setup(self):
        t = np.linspace(0.0, 1.0, self.N)
        self.target = np.vstack((np.sin(np.pi * t), 0.5 * np.cos(2.0 * np.pi * t), 0.2 * np.sin(3.0 * np.pi * t))).reshape(-1)
        self.w = np.repeat(self.weights, self.N) * self.time_step()

    def time_step(self):
        return 1.0 / (self.N - 1)

    def f(self, x):
        d = x - self.target
        return 0.5 * float(np.sum(self.w * d * d))

    def g(self, x):
        return self.w * (x - self.target)

    def hv(self, x, v):
        return self.w * v

    def oracle(self):
        return {"x_star": self.target.copy(), "J_star": 0.0, "kind": "global"}


# ---------------------------------------------------------------------------
# S5: bounds and equality constraints
# ---------------------------------------------------------------------------


class _TargetQuadratic(VectorSystem):
    """f = 1/2 ||x - c||^2 with constraint hooks added by subclasses."""

    c = np.zeros(1)

    def setup(self):
        self.c = np.asarray(type(self).c, dtype=float)

    def f(self, x):
        d = x - self.c
        return 0.5 * float(d @ d)

    def g(self, x):
        return x - self.c


def _equality_spec() -> dict[str, Any]:
    return {
        "derivatives": {"constraint_jacobian": {"hook": "jacobian", "source": "analytical"}},
        "constraints": {"hard": {"kind": "equality", "provider": "residuals"}},
    }


class BoxQuadratic(_TargetQuadratic):
    """S5a: active box bounds; x* = clip(c, -1, 1)."""

    system_id = "s5a-box"
    PRIMARY = {"N": 5}
    c = np.array([2.0, -2.0, 0.5, -0.5, 3.0])

    def control_bounds(self):
        return -np.ones((1, self.N)), np.ones((1, self.N))

    def oracle(self):
        x = np.clip(self.c, -1.0, 1.0)
        return {"x_star": x, "J_star": self.f(x), "kind": "global"}


class LinearEqualityQuadratic(_TargetQuadratic):
    """S5b: residual A x - b; x* is the projection of c onto the affine set."""

    system_id = "s5b-linear-equality"
    PRIMARY = {"N": 6}
    c = np.array([1.0, -2.0, 0.5, 3.0, -1.0, 2.0])

    def setup(self):
        super().setup()
        self.A = np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, -1.0, 2.0, 0.0, 1.0]])
        self.b = np.array([1.0, -0.5])

    def spec_extras(self):
        return _equality_spec()

    def residuals(self, controls, *, name="hard"):
        return self.A @ controls.flatten() - self.b

    def jacobian(self, controls, *, name="hard"):
        return self.A.copy()

    def oracle(self):
        x = self.c - self.A.T @ np.linalg.solve(self.A @ self.A.T, self.A @ self.c - self.b)
        return {"x_star": x, "J_star": self.f(x), "kind": "global"}


def _project_simplex(v: np.ndarray) -> np.ndarray:
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    rho = np.nonzero(u - css / (np.arange(v.size) + 1) > 0)[0][-1]
    return np.maximum(v - css[rho] / (rho + 1.0), 0.0)


class SimplexQuadratic(_TargetQuadratic):
    """S5c: sum(x) = 1 with x >= 0 (upper 10); x* is the simplex projection of c."""

    system_id = "s5c-equality-plus-bounds"
    PRIMARY = {"N": 4}
    c = np.array([2.0, -1.0, 0.5, 0.1])

    def spec_extras(self):
        return _equality_spec()

    def residuals(self, controls, *, name="hard"):
        return np.array([np.sum(controls.flatten()) - 1.0])

    def jacobian(self, controls, *, name="hard"):
        return np.ones((1, self.N))

    def control_bounds(self):
        return np.zeros((1, self.N)), 10.0 * np.ones((1, self.N))

    def oracle(self):
        x = _project_simplex(self.c)
        return {"x_star": x, "J_star": self.f(x), "kind": "global"}


class SphereQuadratic(_TargetQuadratic):
    """S5d: nonlinear equality ||x||^2 = 1; x* = c / ||c||."""

    system_id = "s5d-sphere"
    PRIMARY = {"N": 3}
    c = np.array([2.0, -1.0, 0.5])

    def spec_extras(self):
        return _equality_spec()

    def residuals(self, controls, *, name="hard"):
        x = controls.flatten()
        return np.array([x @ x - 1.0])

    def jacobian(self, controls, *, name="hard"):
        return 2.0 * controls.flatten()[None, :]

    def oracle(self):
        x = self.c / np.linalg.norm(self.c)
        return {"x_star": x, "J_star": self.f(x), "kind": "global"}


class InconsistentEquality(_TargetQuadratic):
    """S5e: residuals (x0 - 1, x0 - 2) cannot both vanish."""

    system_id = "s5e-inconsistent"
    PRIMARY = {"N": 2}
    c = np.zeros(2)

    def spec_extras(self):
        return _equality_spec()

    def residuals(self, controls, *, name="hard"):
        x = controls.flatten()
        return np.array([x[0] - 1.0, x[0] - 2.0])

    def jacobian(self, controls, *, name="hard"):
        return np.array([[1.0, 0.0], [1.0, 0.0]])

    def oracle(self):
        return {"x_star": None, "J_star": None, "kind": "infeasible"}


# ---------------------------------------------------------------------------
# S6: composite smooth plus L1
# ---------------------------------------------------------------------------


class LassoIdentity(VectorSystem):
    """S6: J = 1/2 ||x - b||^2 + lam ||x||_1; x* = soft(b, lam)."""

    system_id = "s6-composite-l1"
    PRIMARY = {"N": 6}
    SECONDARY = {"lam": 1.0}
    b = np.array([3.0, -0.5, 0.2, -2.0, 0.05, 1.5])

    def f(self, x):
        d = x - self.b
        return 0.5 * float(d @ d) + float(self.secondary["lam"]) * float(np.sum(np.abs(x)))

    def g(self, x):  # subgradient; the required OLGS gradient of a nonsmooth objective
        return x - self.b + float(self.secondary["lam"]) * np.sign(x)

    def smooth_gradient(self, controls):
        return self.controls(controls.flatten() - self.b, name="smooth_gradient")

    def prox(self, controls, step_size):
        x = controls.flatten()
        t = float(step_size) * float(self.secondary["lam"])
        return self.controls(np.sign(x) * np.maximum(np.abs(x) - t, 0.0), name="prox")

    def oracle(self):
        lam = float(self.secondary["lam"])
        x = np.sign(self.b) * np.maximum(np.abs(self.b) - lam, 0.0)
        return {"x_star": x, "J_star": self.f(x), "kind": "global"}


# ---------------------------------------------------------------------------
# Extra stress cases
# ---------------------------------------------------------------------------


class LargeChannelScale(ChannelScale):
    """X1: S4 at N = 100000 (300000 variables) for per-iteration cost."""

    system_id = "x1-large-channel-scale"
    PRIMARY = {"N": 100_000}
    weights = np.array([1.0, 1.0, 1.0])


class RestrictedDomain(VectorSystem):
    """X2: f = 1/2 ||x - c||^2 - mu sum log(1 - x^2); undefined for |x_i| >= 1."""

    system_id = "x2-restricted-domain"
    PRIMARY = {"N": 8}
    SECONDARY = {"mu": 1.0e-3}

    def setup(self):
        self.c = np.linspace(-0.97, 0.97, self.size)

    def _check(self, x):
        if np.any(np.abs(x) >= 1.0):
            raise ValueError("controls outside the domain |x| < 1")

    def f(self, x):
        self._check(x)
        d = x - self.c
        return 0.5 * float(d @ d) - float(self.secondary["mu"]) * float(np.sum(np.log(1.0 - x * x)))

    def g(self, x):
        self._check(x)
        return x - self.c + float(self.secondary["mu"]) * 2.0 * x / (1.0 - x * x)

    def oracle(self):
        mu = float(self.secondary["mu"])
        x = self.c.copy()
        for _ in range(100):  # Newton on each coordinate
            r = x - self.c + mu * 2.0 * x / (1.0 - x * x)
            dr = 1.0 + mu * 2.0 * (1.0 + x * x) / (1.0 - x * x) ** 2
            x = np.clip(x - r / dr, -0.999999, 0.999999)
        return {"x_star": x, "J_star": self.f(x), "kind": "global"}


class InconsistentGradient(IsotropicQuadratic):
    """X3: S0 with a gradient that is 0.1 % too large plus a small bias."""

    system_id = "x3-inconsistent-gradient"

    def g(self, x):
        return 1.001 * (x - self.c) + 1.0e-4 * np.sin(np.arange(x.size))
