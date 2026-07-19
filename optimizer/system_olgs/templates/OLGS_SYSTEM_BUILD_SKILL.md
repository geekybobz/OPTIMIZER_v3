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
---

# OLGS System Build Skill

Use this document as the build protocol for creating a new physical OLGS system
from theory notes and user requirements.

The output is a project-local `system.py` that implements the OLGS API.
This is not a copy-paste class template. It is a build protocol with a small set
of mandatory API rules and a larger set of implementation hints for turning a
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

The operative Claude Code trigger lives at
`.claude/skills/olgs-system-build/SKILL.md`, which points back to this protocol as
the single source of truth. It is a hidden tooling shim, not an Obsidian graph node.

## Environment

Use the `optimizer_v3` conda environment for implementation and checks.

Expected Python/Jupyter checks should run inside the project environment, for
example through `conda activate optimizer_v3` rather than the system Python.

## Reference Map

- [OLGS System Hub](../SYSTEM_OLGS_HUB.md)
- [OLGS API Contract](../CONTRACT.md)
- [Primary and Secondary Params](../PARAMS.md)
- [Forward, Backward, Gradient Flow](../LIFECYCLE.md)
- [Optional Derivative Hooks](../DERIVATIVES.md)
- [Logging Boundary Placeholder](../LOGGING_BOUNDARY.md)

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

If the theory and requested API do not align, report the mismatch before writing
fragile code. Ask the user only when the missing assumption affects correctness.

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

## Computational Lifecycle Planning

Before writing code, sketch the cheapest valid computation path for each public
API call. Treat this as a design aid, not a rigid rule:

```text
evaluate -> usually final metrics only
forward_prop -> full forward data when it is useful for gradients, debugging, or inspection
back_prop/gradient -> the history, checkpoints, or recomputation needed by the derivative
simulate -> system-native scans or requested diagnostic trajectories
```

Prefer purpose-specific internal kernels when one universal propagation routine
would store much more data than a call actually needs:

```text
_propagate_final(...)        # final-state metrics; often O(state_dim^2) memory
_propagate_for_gradient(...) # derivative data; may be O(N*state_dim^2)
_propagate_observables(...)  # notebook trajectories; store observables, not full states
```

Useful estimates to include in the plan when the model is large:

```text
state dimension
number of time/control steps
largest arrays and dtypes
evaluate memory scale
gradient memory scale
simulate/trajectory memory scale
whether checkpointing or recomputation might help later
```

The goal is to make cheap API calls naturally cheap while keeping expensive data
available when the math or user workflow genuinely needs it.

## System File Boundary

The system file should usually contain:

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

The system file should avoid:

```text
optimizer calls
notebook wrappers
experiment-specific absolute file paths
large performance or stress-test loops
full logging implementation until the logging layer is finalized
```

Performance loops mean repeated timing, scaling, stress, or optimizer-run sweeps.
Those usually belong in tests, notebooks, benchmark scripts, or experiment folders
unless the user explicitly asks for system-owned performance diagnostics.

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

Implementation hints:

```text
reuse stored forward state when it is simple and safe
recompute forward state when it makes correctness clearer
avoid making cache behavior part of the public contract
keep performance choices local to the system until they are measured
```

## Optional API

Add optional methods when required by the theory, requested by the user, or clearly
useful for normal inspection/testing:

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

Ask the user before adding optional methods only when they introduce substantial
new physics, file formats, plotting surfaces, or expensive simulation behavior.

## Build Protocol

Follow this order:

```text
1. Read all provided theory and requirement files.
2. Summarize the physical model.
3. Double-check derivation details against dimensions and OLGS needs.
4. Extract state, controls, perturbations, objectives, constraints, and gradient route.
5. Sketch the computational lifecycle and expected memory bottlenecks.
6. Define primary and secondary dict defaults.
7. Define ControlSpec and control channel order.
8. Implement evaluate using the cheapest correct path when practical.
9. Implement forward_prop for full forward data when needed.
10. Implement back_prop.
11. Implement analytical gradient.
12. Add simulate when system-native simulation is useful or requested.
13. Add optional hooks when theory, user requirements, or normal diagnostics support them.
14. Run sanity checks.
15. Report assumptions and any optional additions needing user approval.
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
gradient after evaluate on the same controls returns correct finite values
with_secondary returns a new instance
simulate works when implemented
```

Useful performance checks when practical:

```text
repeat evaluate/gradient on the same controls and confirm cache behavior is understandable
check whether avoiding duplicate forward propagation matters for the system cost
document any deliberate recomputation in the output report
for large models, confirm evaluate does not store full trajectory data unless needed
for trajectory simulations, prefer scalar/vector observables over full state histories when that answers the question
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
computational lifecycle summary
implemented mandatory API methods
implemented optional API methods
sanity checks run
unresolved assumptions
suggested notebook or test calls
```
