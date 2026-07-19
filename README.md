---
title: OPTIMIZER_V3_ENTRY
type: root_hub
module: root
tags:
  - optimizer
  - entry
  - hub
---

# OPTIMIZER v3 Entry

OPTIMIZER v3 is a namespace-first optimization toolbox for system-defined control
problems.

A project system owns the physics:

```text
control layout
metrics and objective value
analytical gradient
optional residuals and Jacobians
primary and secondary parameters
```

The library owns the reusable optimizer engineering:

```text
Controls
initial guesses
optimizer loops
diagnostics and repairs
schedules
logs and checkpoints
blackbox run records
result objects
```

Use this README as the root Obsidian entry point. It links to hubs and placeholders,
not every leaf note.

## Fast Start

```python
import optimizer as opt

sys = system(params)

controls = opt.guesses.random_fourier_guess(
    sys,
    amplitude=0.1,
    modes=5,
    seed=1,
)

check = opt.utils.verify_gradient(sys, controls, eps=1.0e-6, directions=8)

r1 = opt.optimizers.adam(sys, controls, step_size=0.05, maxiter=50)
r2 = opt.optimizers.lbfgs(sys, r1.controls, warmstart=r1.warmstart(), maxiter=20)

report = opt.utils.diagnostic_report(sys, r2.controls)
fixed = opt.utils.repair_newton(sys, r2.controls, residuals="hard")
```

Preferred public style groups calls by role:

```text
opt.optimizers.*
opt.guesses.*
opt.utils.*
opt.schedules.*
```

Compact aliases such as `opt.adam(...)`, `opt.random_fourier_guess(...)`, and
`opt.repair_newton(...)` remain available for notebooks.

## Runtime Discovery

When working from a fresh notebook or after losing context, use the root helper API:

```python
opt.info()                 # structured overview of groups, paths, and examples
opt.info(h=True)           # readable overview
opt.list()                 # full structured catalog grouped by role
opt.list(h=True)           # readable full catalog
opt.info("adam", h=True)   # details for one method or object
opt.search("gradient")     # search across methods, utilities, and paths
opt.path("beginner", h=True)
```

Each role namespace also exposes focused helpers:

```python
opt.optimizers.list(h=True)
opt.optimizers.info("adam", h=True)
opt.guesses.list(h=True)
opt.utils.list(kind="derivative", h=True)
opt.core.info("run_chunk", h=True)
```

## Graph Entry Map

Start with the runtime hubs:

```text
README.md
  -> optimizer/system_olgs/SYSTEM_OLGS_HUB.md
  -> optimizer/guesses/GUESSES_HUB.md
  -> optimizer/optimizers/OPTIMIZERS_HUB.md
  -> optimizer/core/CORE_HUB.md
  -> optimizer/utils/UTILS_HUB.md
```

Then use theory hubs when mathematical context is needed:

```text
Theory/guess/GUESS_THEORY_HUB.md
Theory/optimizers/OPTIMIZER_THEORY_HUB.md
Theory/utils/UTILS_THEORY_HUB.md
```

Use placeholder hubs for folders that are intentionally not fully documented yet:

```text
optimizer/schedules/SCHEDULES_HUB.md
optimizer/logs/LOGS_HUB.md
optimizer/blackbox/BLACKBOX_HUB.md
tests/TESTING_HUB.md
```

## Graph Color Code

Use the Obsidian graph colors as a quick map:

```text
gold      README entry point
orange    OLGS system contract and theory-to-system build path
green     guess generators and initial-control design
red       optimizer methods and optimizer theory
blue      core engine, acceptance, stopping, and shared runtime objects
teal      diagnostics, derivative checks, repair, spectrum, and analysis utilities
purple    theory notes
gray      placeholder or not-yet-expanded folders: schedules, logs, blackbox, tests
```

## Main Workflow

```text
system.control_spec()
        |
        v
initial Controls from opt.guesses.*
        |
        v
gradient sanity check from opt.utils.verify_gradient
        |
        v
optimizer chunk from opt.optimizers.*
        |
        v
OptimizerResult with controls, metrics, state, trace
        |
        v
diagnostics, projection, or repair from opt.utils.*
        |
        v
optional warmstart into another optimizer
```

## Architecture

OPTIMIZER v3 separates project physics from optimizer engineering.

The project system owns:

```text
control layout
dynamics or simulation
objective value J
metric names
objective weights and system parameters
analytical gradient
optional residuals and Jacobians
```

The library owns:

```text
control containers
optimizer loops
step proposals and acceptance
warmstart state
guess generation
diagnostics and repairs
trace records and checkpoints
standard result objects
```

This keeps optimizer calls general. A different project can use different metric
names without changing optimizer code.

The native unit of optimization is a short chunk. Longer workflows should chain
chunks and carry `WarmStartState` or `OptimizerResult.warmstart()` when moving
between methods.

`ControlSpec` defines the channel layout, control dimension, time step, dtype, and
metadata. `Controls` stores numeric values as a vectorized channel-by-time matrix,
so optimizers, guesses, diagnostics, checkpoints, and repairs all share one
control representation.

## Runtime Hubs

- [OLGS System Hub](optimizer/system_olgs/SYSTEM_OLGS_HUB.md)
- [Guess Documentation Hub](optimizer/guesses/GUESSES_HUB.md)
- [Optimizer Documentation Hub](optimizer/optimizers/OPTIMIZERS_HUB.md)
- [Core Documentation Hub](optimizer/core/CORE_HUB.md)
- [Utils Documentation Hub](optimizer/utils/UTILS_HUB.md)

## Theory Hubs

- [Guess Theory Hub](Theory/guess/GUESS_THEORY_HUB.md)
- [Optimizer Theory Hub](Theory/optimizers/OPTIMIZER_THEORY_HUB.md)
- [Utils Theory Hub](Theory/utils/UTILS_THEORY_HUB.md)

## Placeholder Hubs

These folders are real parts of the library, but their Obsidian documentation is
intentionally shallow for now. Keep them connected through placeholders until each
folder receives the same structured treatment as optimizers, guesses, and utils.

- [Schedules Placeholder Hub](optimizer/schedules/SCHEDULES_HUB.md)
- [Logs Placeholder Hub](optimizer/logs/LOGS_HUB.md)
- [Blackbox Placeholder Hub](optimizer/blackbox/BLACKBOX_HUB.md)
- [Testing Placeholder Hub](tests/TESTING_HUB.md)

## Current Package Map

```text
optimizer/
  controls.py          vectorized ControlSpec and Controls
  system_olgs/         OLGS class, system contract helpers, params, derivatives, template
  result.py            Evaluation and OptimizerResult
  state.py             RunState and WarmStartState
  library.py           root facade and bound-system context
  core/                shared chunk engine, evaluation, stopping, acceptance guards
  optimizers/          Adam, momentum, line search, AdaGrad, RMSProp, L-BFGS, NCG, CMA-ES
  guesses/             simple, harmonic, random, and composite initial controls
  utils/               diagnostics, derivatives, geometry, repair, spectrum, parallel tools
  schedules/           step-size policies
  logs/                in-memory trace records and checkpoints
  blackbox/            run records, artifact policy, and analysis helpers
```

Concrete project systems should live outside this library repository. This package
only provides the generic `optimizer.system_olgs` contract and helper layer.

## System Contract

Every OLGS project system must provide:

```text
control_spec()
evaluate(controls) -> metrics dict with scalar J
gradient(controls) -> Controls
with_secondary(**updates)
```

Recommended for inspectable analytical systems:

```text
forward_prop(controls)
back_prop(controls)
metrics()
describe()
```

Optional but important for constrained or staged work:

```text
residuals(controls, name=...)
jacobian(controls, name=...)
metric_schema()
residual_schema()
```

The system owns metric names and objective weights. Optimizer methods do not assemble
physics objectives externally.

For the detailed contract, read:

- [OLGS System Hub](optimizer/system_olgs/SYSTEM_OLGS_HUB.md)
- [OLGS Contract](optimizer/system_olgs/CONTRACT.md)
- [OLGS Lifecycle](optimizer/system_olgs/LIFECYCLE.md)

## Installation

Create or update the standard conda environment and install this repo as an editable
library:

```bash
./install.sh
conda activate optimizer_v3
```

After activation, this works from any folder:

```python
import optimizer as opt
```

Editable install means source edits in this checkout are visible after restarting
Python. The installer records the live paths in:

```text
~/.optimizer_v3/install.json
```

Useful commands:

```bash
./update.sh --check
./update.sh
./uninstall.sh
./uninstall.sh --remove-env
```

## Testing

Run the full suite:

```bash
python -m unittest discover -s tests
```

Inside the standard environment:

```bash
conda run -n optimizer_v3 python -m unittest discover -s tests
```

## Repository Hygiene

Generated artifacts should stay out of the graph:

```text
__pycache__/
.pytest_cache/
*.egg-info/
runs/
outputs/
checkpoints/
large numeric arrays
local notebooks under tests/system_check/
```

Tracked Markdown should either be a useful graph node or be removed after its content
is migrated into structured docs.
