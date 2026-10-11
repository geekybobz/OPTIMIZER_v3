---
title: SYSTEM_OLSS_HUB
type: module_hub
module: optimizer/system_olss
tags:
  - optimizer
  - system
  - olss
  - shooting
---

# OLSS System Hub

OLSS means **Open Loop Shooting System**. It is the project-facing contract for
single shooting, multiple shooting, multistart root discovery, continuation, and
cost-aware direct shooting.

Each project owns its complete shooting formulation and physics. OLSS does not
derive PMP equations, boundary conditions, costates, gauges, or continuation
parameters from a Hamiltonian.

## Stable boundary

The project system supplies:

```text
ordered shooting variables
unscaled residual equations
the residual Jacobian
declared metrics
higher-accuracy verification
immutable secondary-parameter updates
optional bounds, cost gradient, reconstruction, and exports
```

The library supplies:

```text
variable containers and validation
accuracy-aware evaluation and caching
Newton and damped least-squares root solving
Jacobian verification
multistart, deduplication, and ranking
pseudo-arclength and natural-parameter continuation
scale-aware seed clouds and focused refinement
cross-family OLSS-to-OLGS replay comparison
cost-aware direct shooting
structured result evidence
```

Solver convergence, residual acceptance, tighter verification, candidate
qualification, candidate ranking, and optimality certification are separate facts.

## Start here

- [OLSS Contract](CONTRACT.md)
- [Issues found and remaining validation](OLSS_ISSUES_FOUND.md)
- [Independent shooting problem fixtures](../../tests/fixtures/shooting_systems.py)
- [OLSS robustness tests](../../tests/test_olss.py)

## Details

Minimal use:

```python
import optimizer as opt

system = ProjectShootingSystem(primary={...}, secondary={...})
z0 = opt.ShootingVariables.from_blocks(
    system.shooting_spec(),
    {"initial_costate": [...], "continuation_parameter": 0.2},
)

check = opt.verify_shooting_jacobian(system, z0)
result = opt.solve_shooting(
    system,
    z0,
    verification_options={"rtol": 1e-11, "atol": 1e-13},
)
```

The system decides which evaluation options it supports. The library passes those
options explicitly and includes them in its cache key.

## Numerical operations

| Call | Purpose |
|---|---|
| `opt.solve_shooting` | Solve and independently verify one residual root |
| `opt.solve_shooting_multistart` | Run independent seeds, deduplicate qualified roots, and rank them |
| `opt.continue_shooting` | Follow a one-dimensional branch with pseudo-arclength continuation |
| `opt.continue_parameter` | Continue one secondary parameter with adaptive steps and exact target landing |
| `opt.solve_direct_shooting` | Optimize a declared cost along the feasible residual set |
| `opt.verify_shooting_jacobian` | Compare the declared Jacobian with explicit finite differences |
| `opt.shooting_geometry_probe` | Inspect rank, condition, weak blocks, and residual groups |
| `opt.random_shooting_seeds` / `opt.perturb_shooting_seeds` | Build scale-aware reproducible seed clouds |
| `opt.refine_shooting_seeds` | Refine an elite subset without project-specific policy in OLSS |
| `opt.compare_shooting_replay` | Recover controls and compare an OLSS candidate in a separate OLGS system |
| `opt.system_info` | Inspect OLSS declarations and hooks without propagation |

Parallel multistart requires a system factory. Each worker owns a reconstructed
system instance; one mutable system is never silently shared between workers.
When a picklable factory is supplied, process execution is the default parallel
choice; otherwise execution remains serial.

`solve_shooting(..., previous_result=...)` continues one history, while
`on_event=` receives stable start/iteration/terminal dictionaries for future
terminal and MCP rendering. This lifecycle hook is intentionally independent of
the later logging redesign.

Create an OLSS starting point with the same scaffold command used for OLGS:

```bash
optimizer-system my_problem/system.py --family olss
```

## Implementation map

```text
optimizer/system_olss/
  variables.py     ShootingBlock, ShootingSpec, ShootingVariables
  specs.py         static system declaration and validation
  contract.py      protocol, discovery, and capability probe
  validation.py    validated numerical transactions
  results.py       evaluation, verification, solver, campaign, and continuation evidence
  olss.py          optional base class with default metric and verification behavior

optimizer/shooting/
  evaluator.py     caching, counters, and combined evaluation
  derivatives.py   finite-difference Jacobian and verification
  solvers.py       root solving and direct shooting
  campaigns.py     multistart, deduplication, and ranking
  continuation.py  pseudo-arclength and natural-parameter continuation
  replay.py        explicit OLSS-to-OLGS replay comparison
```

---

[⌂ Home](../../README.md)
