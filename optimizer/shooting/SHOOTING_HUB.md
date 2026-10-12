---
role: front-door
parent: ../../README.md
summary: "Entry point for OLSS root solvers, continuation, seed campaigns, derivative checks, and replay."
read_when: "Start here after an OLSS project system and initial shooting variables exist."
status: active
reviewed: 2026-10-12
---
# Shooting Hub

This package contains reusable numerical operations that consume the OLSS
contract. It does not own project equations or acceptance physics.

## Tool hierarchy

```text
root solving
  solve_shooting
  solve_direct_shooting

derivative inspection
  finite_difference_shooting_jacobian
  verify_shooting_jacobian
  shooting_geometry_probe

branch following
  continue_shooting
  continue_parameter

seed campaigns
  random_shooting_seeds
  perturb_shooting_seeds
  refine_shooting_seeds
  solve_shooting_multistart

cross-formulation checking
  compare_shooting_replay
```

## Standard workflow

1. Construct `ShootingVariables` from the system's `ShootingSpec`.
2. Check the Jacobian when implementing or changing the project equations.
3. Solve one root or a seed cloud.
4. Require independent verification before accepting a root.
5. Inspect rank and weak directions when progress is poor.
6. Continue a branch, refine seeds, or replay recovered controls as needed.

## Main references

- [Shooting Methods](SHOOTING_METHODS.md) — inputs, parameter groups, returns,
  use cases, and limitations.
- [OLSS Contract](../system_olss/CONTRACT.md) — project operations consumed here.
- [OLSS Verification](../system_olss/VERIFICATION.md) — qualification rules.
- [API Reference](../../API_REFERENCE.md) — exact signatures and defaults.

## Implementation ownership

| Source | Responsibility |
|---|---|
| `evaluator.py` | Validated calls, bounded caching, working/verification accuracy, and counters |
| `solvers.py` | Root and direct constrained-cost methods |
| `derivatives.py` | Finite-difference Jacobian and independent comparison |
| `continuation.py` | Pseudo-arclength and secondary-parameter continuation |
| `campaigns.py` | Multistart, root deduplication, random/perturbed/refined seeds |
| `replay.py` | Recover OLSS controls and compare them through OLGS |

## Connections

- Related: [V4 entry](../../README.md)
- Prerequisite: [OLSS System Hub](../system_olss/SYSTEM_OLSS_HUB.md)
- Related: [Shooting Methods](SHOOTING_METHODS.md)
- Related: [Utils Hub](../utils/UTILS_HUB.md)

---

[⌂ Home](../../README.md)
