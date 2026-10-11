---
title: OPTIMIZER_V4_ENTRY
type: root_hub
module: root
tags:
  - optimizer
  - entry
  - hub
---

# OPTIMIZER V4 entry

This is the single current OPTIMIZER V4 checkout. Commits and annotated tags record
release revisions; ordinary V4 work stays in this folder. It is a namespace-first
optimization toolbox for system-defined control problems.

A project system owns the physics:

```text
control layout
metrics and objective value
analytical gradient
optional residuals and Jacobians
optional combined evaluation, bounds, HVP, or proximal capability
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
result objects
```

Use this README as the root Obsidian entry point. It links to hubs, not every leaf
note.

## Fast Start

```python
import optimizer as opt
from my_project.system import System

system = System(
    primary={"tau": 1.0, "N": 1001},
    secondary={"infidelity_weight": 1.0, "energy_weight": 0.0},
)

controls = opt.guesses.random_fourier_guess(
    system,
    amplitude=0.1,
    modes=5,
    seed=1,
)

check = opt.utils.verify_gradient(system, controls, eps=1.0e-6, directions=8)

r1 = opt.optimizers.adam(system, controls, step_size=0.05, maxiter=50)
r2 = opt.optimizers.lbfgs(system, r1.controls, warmstart=r1.warmstart(), maxiter=20)

report = opt.utils.diagnostic_report(system, r2.controls)
fixed = opt.utils.repair_newton(system, r2.controls, residuals="hard")
```

Capability-gated numerical methods remain direct:

```python
feasible = opt.optimizers.projected_gradient(system, controls, maxiter=50)
newton = opt.optimizers.newton_cg(system, controls, maxiter=20)
trusted = opt.optimizers.trust_region(system, controls, maxiter=20)
composite = opt.optimizers.proximal_gradient(system, controls, maxiter=50)
```

`opt.system_info(system)["capabilities"]` reports which calls are compatible before
any numerical work. `opt.continuation(...)` coordinates explicit parameter stages
and requires an explicit control-transfer callable when a stage changes the control
layout or time spacing.

Preferred public style groups calls by role:

```text
opt.optimizers.*
opt.guesses.*
opt.utils.*
opt.schedules.*
opt.campaigns.*
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

## Public API

The runtime catalog is the declared public API. `opt.list()` returns every entry
by group, each with a summary, requirements, return value, and status, and each
group's hub documents its entries. Compact root aliases such as `opt.adam(...)`
call the same entries.

An entry is removed only after one release in which it still works, warns when
used, and appears with its replacement in the release evidence.

## Graph Entry Map

Start with the runtime hubs:

```text
README.md
  -> optimizer/BASE_HUB.md
  -> optimizer/system_olgs/SYSTEM_OLGS_HUB.md
  -> optimizer/guesses/GUESSES_HUB.md
  -> optimizer/optimizers/OPTIMIZERS_HUB.md
  -> optimizer/core/CORE_HUB.md
  -> optimizer/utils/UTILS_HUB.md
  -> optimizer/tracker/TRACKER_HUB.md
  -> optimizer/campaigns/CAMPAIGNS.md
  -> optimizer/visualization/VISUALIZATION_HUB.md
  -> optimizer_log/OPTIMIZER_LOG_READER.md
```

Then use theory hubs when mathematical context is needed:

```text
Theory/guess/GUESS_THEORY_HUB.md
Theory/optimizers/OPTIMIZER_THEORY_HUB.md
Theory/utils/UTILS_THEORY_HUB.md
```

Small support folders have compact hubs:

```text
optimizer/schedules/SCHEDULES_HUB.md
optimizer/logs/LOGS_HUB.md
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
gray      small support folders: schedules, logs, tests
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

OPTIMIZER v4 separates project physics from optimizer engineering.

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

- [Base API Hub](optimizer/BASE_HUB.md)
- [OLGS System Hub](optimizer/system_olgs/SYSTEM_OLGS_HUB.md)
- [Guess Documentation Hub](optimizer/guesses/GUESSES_HUB.md)
- [Optimizer Documentation Hub](optimizer/optimizers/OPTIMIZERS_HUB.md)
- [Core Documentation Hub](optimizer/core/CORE_HUB.md)
- [Utils Documentation Hub](optimizer/utils/UTILS_HUB.md)
- [Tracker Hub](optimizer/tracker/TRACKER_HUB.md)
- [Campaign Modes](optimizer/campaigns/CAMPAIGNS.md)
- [Visualization Hub](optimizer/visualization/VISUALIZATION_HUB.md)
- [Run Log Reader](optimizer_log/OPTIMIZER_LOG_READER.md)

## Theory Hubs

- [Guess Theory Hub](Theory/guess/GUESS_THEORY_HUB.md)
- [Optimizer Theory Hub](Theory/optimizers/OPTIMIZER_THEORY_HUB.md)
- [Utils Theory Hub](Theory/utils/UTILS_THEORY_HUB.md)

## Support Hubs

- [Schedules Hub](optimizer/schedules/SCHEDULES_HUB.md)
- [Logs Hub](optimizer/logs/LOGS_HUB.md)
- [Testing Hub](tests/TESTING_HUB.md)

## Current Package Map

```text
optimizer/
  controls.py          vectorized ControlSpec and Controls
  system_olgs/         OLGS class, system contract helpers, params, derivatives, template
  result.py            Evaluation and OptimizerResult
  state.py             RunState and WarmStartState
  library.py           root facade and bound-system context
  catalog.py           runtime catalog behind opt.list, opt.info, opt.search, opt.path
  introspection.py     runtime self-description of registered APIs and signatures
  continuation.py      explicit parameter-stage continuation
  core/                shared chunk engine, evaluation, stopping, acceptance guards
  optimizers/          Adam, momentum, line search, AdaGrad, RMSProp, L-BFGS, NCG, CMA-ES,
                       projected gradient, Newton-CG, trust region, proximal gradient
  guesses/             simple, harmonic, random, and composite initial controls
  utils/               diagnostics, derivatives, geometry, repair, spectrum, parallel tools
  visualization/       generic plot-data models and optional renderers
  campaigns/           multi-stage scout, comparison, refinement, challenge, and validation
  schedules/           step-size policies
  logs/                in-memory trace records and checkpoints
  tracker/             durable run journal, observer bridge, checkpoint artifacts

optimizer_log/         standalone read-only reader for tracker run folders
```

Concrete project systems should live outside this library repository. This package
only provides the generic `optimizer.system_olgs` contract and helper layer.

## System Contract

Every OLGS project system must provide:

```text
system_spec() -> complete static metric/API/derivative declaration
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
plot_data(controls, kind=...)
```

The system owns metric names, meanings, analytical derivative provenance, and
objective weights through `system_spec()`. Optimizer methods do not assemble physics
objectives externally.

Create a small capability-selected system scaffold with:

```bash
python -m optimizer.system_olgs.scaffold my_problem/system.py --profile qubit
```

For the detailed contract, read:

- [OLGS System Hub](optimizer/system_olgs/SYSTEM_OLGS_HUB.md)
- [OLGS Contract](optimizer/system_olgs/CONTRACT.md)
- [OLGS Lifecycle](optimizer/system_olgs/LIFECYCLE.md)

## Installation

Create or update the release environment and install the built wheel. This is a
non-editable install: source changes cannot alter the installed release.

```bash
./install.sh
conda activate optimizer_v4
```

After activation, this works from any folder:

```python
import optimizer as opt
```

The installer records the wheel hash and installed release provenance in:

```text
~/.optimizer_v4/install.json
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
conda run -n optimizer_v4 python -m unittest discover -s tests
```

## Releases and Plans

- [Release evidence](RELEASE_EVIDENCE.md): the current release record
- [Stabilization plan](STABILIZATION.md): the steps that make V4 a stable baseline

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

---

[⌂ Home](README.md)
