# `system.py`: OptimizerSystem

Status: review draft.
Last updated: 2026-07-15.

## Purpose

`system.py` is the model contract. It is the equivalent of an RL environment style
contract: the library knows what functions it can ask for, and every project system
implements the same shape.

The system owns the physical meaning. The optimizer does not.

## Required Responsibilities

The system must define:

```text
control names
control dimension
time grid / dt
physical dynamics
objective value J
metrics
analytical gradient
cost prefactors in params
```

## Proposed Interface

```python
class OptimizerSystem:
    def control_spec(self) -> ControlSpec:
        ...

    def evaluate(self, controls: Controls) -> Evaluation:
        ...

    def gradient(self, controls: Controls) -> Controls:
        ...

    def with_params(self, **updates) -> "OptimizerSystem":
        ...
```

Optional advanced hooks:

```python
    def residuals(self, controls: Controls, name: str = "hard") -> np.ndarray:
        ...

    def jacobian(self, controls: Controls, name: str = "hard") -> np.ndarray:
        ...

    def hessian(self, controls: Controls):
        ...

    def hvp(self, controls: Controls, vector: np.ndarray) -> np.ndarray:
        ...

    def simulate(self, controls: Controls, **kwargs) -> dict:
        ...
```

## Cost Parameters

Multi-objective costs are represented as system params:

```python
params = {
    "infidelity_weight": 1.0,
    "lambda2": 1.0,
    "lambda4": 1.0,
    "energy_weight": 0.0,
}
```

The system uses these params inside its analytical derivation:

```text
forward dynamics
costate dynamics
gradient
metrics
```

The optimizer does not know how the formula is built.

## Evaluation

`evaluate()` should run the minimum required propagation and return all metrics that
are cheap or naturally available from that propagation.

Example metrics:

```text
J
fidelity
infidelity
energy
F_norm2
C_sym_norm2
terminal_overlap
control_norm
```

## Gradient

Gradient is expected to exist analytically for this project. Numerical fallback is for
diagnostics and emergency checks, not the normal optimization path.

The returned gradient should match `ControlSpec` shape and channel names.

## Residuals And Jacobian

Use residuals for hard equality-style conditions:

```text
terminal leakage
F_k(T)
exact constraints
```

Use Jacobian for:

```text
Newton repair
projected descent
geometry probe
Levenberg-Marquardt
```

## Naming Note

Prefer `control_spec()` over `get_control_spec()` in v3, but an adapter can support
old systems with `get_control_spec()`.

