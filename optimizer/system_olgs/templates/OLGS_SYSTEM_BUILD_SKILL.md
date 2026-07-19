---
title: OLGS System Build Skill
type: build_skill
module: optimizer/system_olgs/templates
tags:
  - optimizer
  - system
  - olgs
  - skill
  - theory_to_code
depends_on:
  - README
  - CONTRACT
  - PARAMS
  - LIFECYCLE
connects_to:
  - DERIVATIVES
  - LOGGING_BOUNDARY
---

# OLGS System Build Skill

Use this document as the build protocol for creating a new physical OLGS system
from theory notes and user requirements.

The output is a project-local `system.py` that implements the OLGS API.
This is not a copy-paste class template. It is a constraint document for turning a
mathematical model into a clean system file.

## Trigger

Use this skill when the user asks for:

```text
build system file
build an OLGS system file
create system.py from theory
implement system.py
make a system file
design the system file
build the physical system model
turn this theory into a system file
```

Also use it for closely related requests where the user provides theory notes and
expects a new OLGS-compatible `system.py`.

## Environment

Use the OPTIMIZER v3 conda environment for implementation and checks.

Expected Python/Jupyter checks should run inside the project environment, for
example through the configured optimizer v3 conda env rather than the system
Python.

## Reference Map

- [[README|system_olgs]]
- [[CONTRACT|OLGS API Contract]]
- [[PARAMS|Primary and Secondary Params]]
- [[LIFECYCLE|Forward, Backward, Gradient Flow]]
- [[DERIVATIVES|Optional Derivative Hooks]]
- [[LOGGING_BOUNDARY|Logging Boundary Placeholder]]

## Role

When asked to build a system, act as the system implementer.

Build a physical model file that:

```text
derives from the provided theory
matches the OLGS API
keeps the system file focused on system-owned behavior
uses dict-first primary and secondary params
returns stable, inspectable API outputs
passes lightweight sanity checks
reports any unresolved assumptions
```

Do not invent missing physics silently. If a missing assumption affects the model
or gradient, ask the user before implementing that part.

## Inputs

The user may provide:

```text
theory .tex files
compact derivation notes
model assumptions
target objective
required control channels
required simulation behavior
saved-control helper requirements
plotting requirements
logging requirements or placeholder expectations
```

The user does not need to provide an existing `system.py`. This protocol is for a
new project built from theory and requirements.

## Theory Audit

Before writing code, read the theory and extract:

```text
state variables
state representation
state dimension
control variables
control channel names
evolution equation
drift terms
control terms
initial conditions
target conditions
perturbation variables
objective terms
residual constraints
analytical gradient route
simulation or ensemble requirements
```

Double-check:

```text
array dimensions
control dimensions
time-grid convention
normalization conventions
complex vs real dtype needs
endpoint handling
objective sign and scaling
gradient sign and scaling
whether the derivation can support the requested OLGS methods
```

If the theory and requested API do not align, stop and report the mismatch before
writing fragile code.

## OLGS Mapping

Map theory objects into the OLGS structure:

```text
physical constants -> primary dict
grid/time/state definitions -> primary dict
objective weights -> secondary dict
curriculum weights -> secondary dict
controls -> ControlSpec
forward evolution -> forward_prop
adjoint/costate evolution -> back_prop
scalar objective -> evaluate
analytical derivative -> gradient
robustness, ensemble, trajectory, or projection scans -> simulate
```

No runtime parameter group is used for now. Runtime choices are either internal
system choices or notebook/test choices unless the OLGS contract is expanded later.

## System File Boundary

The system file should contain:

```text
physical model constants and defaults
dict-first primary and secondary params
control layout
forward propagation
backward propagation
objective evaluation
analytical gradient
system-native simulation
optional residual, Jacobian, Hessian, HVP hooks
saved-control helpers when explicitly requested
plotting helpers when explicitly requested
logging placeholder hooks until the logging layer is finalized
```

The system file should not contain:

```text
optimizer calls
notebook wrappers
experiment-specific absolute file paths
large performance or stress-test loops
full logging implementation until the logging layer is finalized
```

Performance loops mean repeated timing, scaling, stress, or optimizer-run sweeps.
Those belong in tests, notebooks, benchmark scripts, or experiment folders unless
the user explicitly asks for system-owned performance diagnostics.

## Mandatory API

Every generated system must implement:

```text
__init__(primary=None, secondary=None)
control_spec() -> ControlSpec
forward_prop(controls)
back_prop(controls)
evaluate(controls) -> dict
gradient(controls) -> Controls
with_secondary(**updates)
describe()
```

Hard requirements:

```text
evaluate returns a finite scalar metrics["J"]
gradient is analytical
gradient returns Controls
gradient has the same ControlSpec as the input controls
with_secondary returns a new system instance
unknown primary keys fail clearly
unknown secondary keys fail clearly
control validation fails clearly on wrong shape or keys
```

## Optional API

Add optional methods only when required by the theory or explicitly requested by
the user:

```text
simulate(controls, ...)
residuals(controls, name="hard")
jacobian(controls, name="hard")
hessian(controls)
hvp(controls, vector)
second_derivative(...)
metric_schema()
residual_schema()
load_*_controls(...)
plot_*(...)
cache_reset()
cache_status()
```

If an optional method would help testing or analysis but is outside the requested
API, ask the user before adding it.

## Build Protocol

Follow this order:

```text
1. Read all provided theory and requirement files.
2. Summarize the physical model.
3. Double-check derivation details against dimensions and OLGS needs.
4. Extract state, controls, perturbations, objectives, constraints, and gradient route.
5. Define primary and secondary dict defaults.
6. Define ControlSpec and control channel order.
7. Implement forward_prop.
8. Implement evaluate.
9. Implement back_prop.
10. Implement analytical gradient.
11. Add simulate only when system-native simulation is required.
12. Add optional hooks only when theory or user requirements support them.
13. Run sanity checks.
14. Report assumptions and any optional additions needing user approval.
```

## Sanity Checks

After implementation, check:

```text
import works
system constructs from primary and secondary dicts
control_spec returns expected keys and shape
zero controls validate
random controls validate
forward_prop returns expected structures
back_prop returns expected structures
evaluate returns finite J
gradient returns finite Controls
gradient shape matches controls shape
with_secondary returns a new instance
simulate works when implemented
```

For gradient confidence, compare the analytical gradient against a small finite
difference check when the model cost is reasonable and the user has not asked to
skip it.

## User Approval Points

Ask the user before adding:

```text
optional physics not present in the theory
extra residual families
extra derivative hooks beyond the requested API
system-owned plotting helpers
saved-control helper formats
large simulation scans
performance diagnostics
```

Do not ask before implementing mandatory OLGS methods when the theory provides
enough information.

## Output Report

When finished, report:

```text
files created or updated
physical model summary
primary params
secondary params
implemented mandatory API methods
implemented optional API methods
sanity checks run
unresolved assumptions
suggested notebook or test calls
```
