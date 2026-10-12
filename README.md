---
role: front-door
summary: "Entry point for the OPTIMIZER V4 contracts, numerical tools, workflows, and verification evidence."
read_when: "Start here when using, extending, or reviewing OPTIMIZER V4."
status: active
reviewed: 2026-10-12
---
# OPTIMIZER V4

OPTIMIZER is a Python toolbox that connects project-owned control problems to
reusable gradient optimizers, shooting solvers, diagnostics, continuation, and
campaign tools through stable system contracts.

## In brief

Projects implement one of two contracts:

| System family | Project supplies | Library supplies |
|---|---|---|
| [OLGS](optimizer/system_olgs/SYSTEM_OLGS_HUB.md) | Controls, objective metrics, analytical gradient, and optional numerical capabilities | Gradient and derivative-free optimizers, schedules, diagnostics, repair, and continuation |
| [OLSS](optimizer/system_olss/SYSTEM_OLSS_HUB.md) | Shooting variables, residuals, residual Jacobian, metrics, and independent verification | Root solving, multistart, continuation, seed campaigns, geometry diagnostics, and replay |

Project code normally calls the contracts directly, and the numerical tools
call systems through their validated evaluators. The
[transaction envelope](optimizer/TRANSACTIONS.md) exposes the same system
operations in a validated request-response form. The
[foreground operation service](optimizer/operations/SERVICE.md) adds selected
systems, owned results and candidate activation. The
[structured terminal](optimizer/cli/TERMINAL.md) exposes it through interactive
semicolon commands and scripts; a future MCP adapter uses the same service.

## Main routes

- To choose a path for a task, use the [Route Map](MAP.md).
- To understand components and source ownership, use the [Code Map](CODE_MAP.md).
- To find every maintained topic, use the [Topic Index](INDEX.md).
- To look up an exact public signature, use the generated
  [API Reference](API_REFERENCE.md).
- To build a gradient project, start with the [OLGS System Hub](optimizer/system_olgs/SYSTEM_OLGS_HUB.md).
- To build or solve a shooting project, start with the [OLSS System Hub](optimizer/system_olss/SYSTEM_OLSS_HUB.md).
- To choose a numerical method, start with the [Optimizer Hub](optimizer/optimizers/OPTIMIZERS_HUB.md)
  or [Shooting Hub](optimizer/shooting/SHOOTING_HUB.md).
- To inspect a short run, start with the [Utils Hub](optimizer/utils/UTILS_HUB.md).
- To run interactive commands or scripts, use the [Structured Terminal](optimizer/cli/TERMINAL.md).
- To save, reopen, fork or roll back an experiment, use [Persistent Experiments](optimizer/persistence/PERSISTENCE.md).

## Normal workflow

```text
declare a project system
    -> construct Controls or ShootingVariables
    -> verify derivatives and shapes
    -> run a short numerical method chunk
    -> diagnose the result
    -> continue, switch, refine, or verify
```

The runtime catalog is the declared public API. Every export has exactly one
catalog declaration, and the namespace exports, `opt.methods()`, and the
[API Reference](API_REFERENCE.md) are built from those declarations. An entry is
removed only after one release in which it still works, warns when used, and
appears with its replacement in the release evidence. The explicitly approved
logger retirement is an exception; [migration notes](optimizer/history/HISTORY.md#removed-logging-api)
list its API removals. Use `opt.info()`,
`opt.list()`, `opt.search()`, and `opt.path()` when exact runtime discovery is
preferable to browsing documentation.

## Details

### Minimal OLGS example

```python
import optimizer as opt
from my_project.system import System

system = System(primary={"tau": 1.0, "N": 1001}, secondary={})
controls = opt.guesses.random_fourier_guess(system, amplitude=0.1, modes=5, seed=1)
check = opt.utils.verify_gradient(system, controls, directions=8)
result = opt.adam(system, controls, maxiter=5)
diagnosis = opt.diagnose_run(result)
```

### Minimal OLSS example

```python
import optimizer as opt
from my_project.system import System

system = System(primary={...}, secondary={...})
initial = opt.ShootingVariables.from_blocks(system.shooting_spec(), {...})
result = opt.solve_shooting(system, initial, verification_options={"accuracy": "verification"})
geometry = opt.shooting_geometry_probe(system, result.variables)
```

### Installation

```bash
./install.sh
conda activate optimizer_v4
```

The release installation is non-editable. When validating current source rather
than the installed wheel, run from this checkout with `PYTHONPATH=.`.

### Verification

```bash
python -m pytest -q
mdp verify . --require-properties
```

Release provenance is recorded in [Release Evidence](RELEASE_EVIDENCE.md).
Numerical scenario evidence is owned by the contained test campaigns linked
from the [Testing Hub](tests/TESTING_HUB.md).

## Connections

- Deeper: [Foreground operation service](optimizer/operations/SERVICE.md)
- Deeper: [Structured terminal](optimizer/cli/TERMINAL.md)
- Deeper: [Persistent experiments](optimizer/persistence/PERSISTENCE.md)

- Related: [Base API Hub](optimizer/BASE_HUB.md)
- Related: [OLGS System Hub](optimizer/system_olgs/SYSTEM_OLGS_HUB.md)
- Related: [OLSS System Hub](optimizer/system_olss/SYSTEM_OLSS_HUB.md)
- Related: [Optimizer Hub](optimizer/optimizers/OPTIMIZERS_HUB.md)
- Related: [Shooting Hub](optimizer/shooting/SHOOTING_HUB.md)
- Related: [Transaction model](optimizer/TRANSACTIONS.md)
- Related: [Code map](CODE_MAP.md)
- Related: [Core Hub](optimizer/core/CORE_HUB.md)
- Related: [Utils Hub](optimizer/utils/UTILS_HUB.md)
- Deeper: [Campaigns](optimizer/campaigns/CAMPAIGNS.md)
- Deeper: [Command history](optimizer/history/HISTORY.md)
- Related: [Schedules Hub](optimizer/schedules/SCHEDULES_HUB.md)
- Related: [Testing Hub](tests/TESTING_HUB.md)

---

[⌂ Home](README.md) · [Next →](MAP.md)
