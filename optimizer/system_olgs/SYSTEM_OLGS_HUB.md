---
role: front-door
parent: ../../README.md
summary: "Entry point for the Open Loop Gradient System contract and its supporting tools."
read_when: "Implementing a differentiable optimization system or choosing OLGS capabilities."
status: active
reviewed: 2026-10-12
---
# OLGS System Hub

OLGS is the reusable boundary for differentiable optimization problems. A project
owns its physics and objective; the library owns optimizers, diagnostics, schedules,
validation, state transfer, and result containers.

## Required Surface

```text
system_spec() -> mapping
control_spec() -> ControlSpec
evaluate(controls) -> metrics mapping containing J
gradient(controls) -> Controls
with_secondary(**updates) -> new compatible system
```

The five calls stay fixed. Optional capabilities add constrained, second-order,
proximal, combined-evaluation, and visualization workflows without enlarging the
base contract.

## Read by Task

| Task | Read |
|---|---|
| Implement or validate a system | [Contract](CONTRACT.md) |
| Separate fixed and curriculum parameters | [Parameters](PARAMS.md) |
| Understand propagation and evaluation order | [Lifecycle](LIFECYCLE.md) |
| Add HVP, constraints, or proximal behavior | [Derivatives](DERIVATIVES.md) |
| Use short-run diagnostics | [Utils Hub](../utils/UTILS_HUB.md) |
| Inspect verification evidence | [OLGS campaign report](../../test_contained/olgs_convergence/REPORT.md) |

## Minimal Use

```python
import optimizer as opt

system = System(primary={"tau": 1.0}, secondary={"energy_weight": 0.0})
controls = opt.guesses.random_fourier_guess(system, amplitude=0.1, modes=4)
result = opt.adam(system, controls, maxiter=5)
diagnosis = opt.diagnose_run(result)
```

Use `opt.system_info(system)` before selecting a conditional method. Use
`optimizer-system path/to/system.py --profile qubit` to create a starting scaffold;
the generated system must still be adapted to the project physics. The scaffold
is laid out as forward propagation → backward propagation → gradient, with the
last forward propagation cached; see
[One propagation per iteration](LIFECYCLE.md#one-propagation-per-iteration).

## Ownership Boundary

OLGS owns system semantics, control layout, objective metrics, and analytical
derivatives. It does not own optimizer update rules, persistent run logging, generic
finite-difference diagnostics, or result ranking.

## Connections

- Related: [V4 entry](../../README.md)
- Related: [Contract](CONTRACT.md)
- Related: [Parameters](PARAMS.md)
- Related: [Lifecycle](LIFECYCLE.md)
- Related: [Derivative capabilities](DERIVATIVES.md)
- Deeper: [Interaction boundary](LOGGING_BOUNDARY.md)
- Related: [Transactions](../TRANSACTIONS.md), [Optimizer methods](../optimizers/OPTIMIZERS_HUB.md), [OLSS](../system_olss/SYSTEM_OLSS_HUB.md)

---

[⌂ Home](../../README.md)
