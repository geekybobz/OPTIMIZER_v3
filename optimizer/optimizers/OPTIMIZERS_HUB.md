---
title: OPTIMIZERS_HUB
type: module_hub
module: optimizer/optimizers
tags:
  - optimizer
  - optimizers
  - api
---

# Optimizer Methods Documentation

This folder contains the optimizer method layer for OPTIMIZER v3.

The intent is compact documentation. Use this hub as the entry point, then open
only the focused file needed for the current task.

## Design Scope

Optimizers move `Controls` to improve a scalar system metric.

The optimizer method layer owns:

```text
method-specific update rules
method-specific state stored in RunState.optimizer_state
trial control proposals
variant validation
warmstart-compatible memory for engine-based methods
compact API references
```

The optimizer method layer does not own:

```text
physical objectives
system propagation
analytical gradient derivation
guess generation
residual repair
projected-gradient tools
deep mathematical theory notes
```

## Documentation Map

Read in this order:

```text
OPTIMIZERS_HUB.md
  Entry point and module map.

OPTIMIZER_CONTRACT.md
  Public optimizer API contract and common arguments.

OPTIMIZER_METHODS.md
  Compact table of implemented optimizers.

OPTIMIZER_LIFECYCLE.md
  Engine-driven proposal/evaluate/accept flow and CMA-ES exception.

OPTIMIZER_STATE_WARMSTART.md
  RunState, optimizer_state, and safe handoff behavior.

OPTIMIZER_ACCEPTANCE.md
  Accept/reject metrics and stopping rules.

OPTIMIZER_BOUNDARY.md
  What belongs in optimizers, systems, utils, guesses, and theory.

OPTIMIZER_COMMON.md
  Shared helper behavior from _common.py.
```

Method references:

```text
LINE_SEARCH.md
MOMENTUM.md
ADAM.md
ADAGRAD_RMSPROP.md
LBFGS.md
NONLINEAR_CG.md
CMA_ES.md
```

## Public Entry Points

All methods are available through the namespace:

```python
import optimizer as opt

result = opt.optimizers.adam(system, controls, maxiter=100)
```

Most methods also have root shortcuts:

```python
result = opt.adam(system, controls, maxiter=100)
```

Bound contexts are available for notebook and curriculum workflows:

```python
ctx = opt.context(system)
result = ctx.adam(controls, maxiter=100)
```

Runtime discovery:

```python
opt.optimizers.list(h=True)
opt.optimizers.info("adam", h=True)
opt.info("optimizers.adam", h=True)
opt.search("warmstart", h=True)
```

## Implementation Files

Current source files:

```text
optimizer/optimizers/
  __init__.py
  _common.py
  line_search.py
  momentum.py
  adam.py
  adaptive.py
  lbfgs.py
  nonlinear_cg.py
  cma_es.py
```

## Theory Map

Long mathematical notes live outside this runtime package:

```text
Theory/optimizers/
  OPTIMIZER_THEORY_HUB.md
  THEORY_LINE_SEARCH.md
  THEORY_MOMENTUM.md
  THEORY_ADAM_FAMILY.md
  THEORY_ADAPTIVE_SCALING.md
  THEORY_NONLINEAR_CG.md
  THEORY_LBFGS.md
  THEORY_CMA_ES.md
```

Each method reference links to its corresponding theory note, and each theory note
links back to the API reference.

## Related Notes

- [Optimizer Contract](OPTIMIZER_CONTRACT.md)
- [Optimizer Methods](OPTIMIZER_METHODS.md)
- [Optimizer Lifecycle](OPTIMIZER_LIFECYCLE.md)
- [State and Warmstart](OPTIMIZER_STATE_WARMSTART.md)
- [Theory Hub](../../Theory/optimizers/OPTIMIZER_THEORY_HUB.md)
