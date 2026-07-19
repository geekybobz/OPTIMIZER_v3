---
title: system_olgs
type: module_index
module: optimizer/system_olgs
tags:
  - optimizer
  - system
  - olgs
connects_to:
  - CONTRACT
  - PARAMS
  - LIFECYCLE
  - DERIVATIVES
  - LOGGING_BOUNDARY
  - OLGS_SYSTEM_BUILD_SKILL
---

# OLGS System Documentation

OLGS means **Open Loop Gradient System**.

This folder contains the system-facing documentation and Python implementation for
the OLGS class format.  OLGS is the system template format for OPTIMIZER v3: a
project system with analytical forward propagation, backward propagation, objective
evaluation, and gradient.

The intent is compact documentation.  This README is the entry point; each linked
file owns one technical part of the system API.

## Design Scope

OLGS owns the contract between a physical system and the optimizer.

The system provides:

```text
control layout
primary physical parameters
secondary objective/curriculum parameters
forward propagation
backward propagation
objective metrics
analytical gradient
optional residuals and higher derivatives
```

The optimizer library provides:

```text
Controls and ControlSpec
optimizer loops
guess generators
diagnostics and repairs
result objects
logging and blackbox connection
```

## Documentation Map

Read in this order:

```text
README.md
  Entry point and design scope.

CONTRACT.md
  Required and optional OLGS methods.

PARAMS.md
  Primary and secondary parameter split.

LIFECYCLE.md
  Forward, backward, evaluate, gradient, metrics flow.

DERIVATIVES.md
  Optional Jacobian, Hessian, HVP, and numerical fallback policy.

LOGGING_BOUNDARY.md
  Cache/logging placeholders and what OLGS should not own yet.

templates/OLGS_SYSTEM_BUILD_SKILL.md
  Build protocol for turning theory notes into a project-local system.py.
```

## System Map

Start here:

- [[OLGS_SYSTEM_BUILD_SKILL|OLGS System Build Skill]]
- [[CONTRACT|API Contract]]
- [[PARAMS|System Configuration]]
- [[LIFECYCLE|Computation Lifecycle]]

Advanced:

- [[DERIVATIVES|Derivative Hooks]]
- [[LOGGING_BOUNDARY|Logging Boundary]]

Reading guide:

- Building a new project system from theory: [[OLGS_SYSTEM_BUILD_SKILL|OLGS System Build Skill]]
- Implementing a new system: [[CONTRACT|API Contract]]
- Defining primary/secondary dicts: [[PARAMS|System Configuration]]
- Calling from a notebook: [[LIFECYCLE|Computation Lifecycle]]
- Building repair/Newton tools: [[DERIVATIVES|Derivative Hooks]]
- Connecting logs or blackbox: [[LOGGING_BOUNDARY|Logging Boundary]]

## Target User Shape

```python
from systems.my_project.system import System
import optimizer as opt

system = System(
    primary={"tau": 1.0, "N": 1001},
    secondary={"infidelity_weight": 1.0, "energy_weight": 0.0},
)

controls = opt.guesses.random_fourier_guess(system, amplitude=0.1, modes=4)
result = opt.optimizers.adam(system, controls, maxiter=100)
```

Curriculum updates should use secondary params:

```python
system = system.with_secondary(residual_weight=0.2, energy_weight=1.0e-3)
```

## Decisions Already Fixed

```text
Name: OLGS
Folder: optimizer/system_olgs
Only one system class format for now
Use primary and secondary params only
No runtime parameter group for now
Use row-wise Controls layout: (n_controls, control_dim)
Require analytical gradient
Allow optional higher derivatives
Numerical fallback only for optional advanced derivatives
Leave cache/logging as placeholders for later blackbox redesign
```

## Implementation Files

Current implementation files:

```text
optimizer/system_olgs/
  __init__.py
  olgs.py
  contract.py
  validation.py
  derivatives.py
  results.py
  templates/
    OLGS_SYSTEM_BUILD_SKILL.md
```

## Reading Path

For a new system author:

```text
README.md
templates/OLGS_SYSTEM_BUILD_SKILL.md
CONTRACT.md
PARAMS.md
LIFECYCLE.md
```

For optimizer/tool integration:

```text
CONTRACT.md
DERIVATIVES.md
LOGGING_BOUNDARY.md
```
