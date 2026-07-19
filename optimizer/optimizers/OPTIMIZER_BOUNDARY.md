---
title: OPTIMIZER_BOUNDARY
type: module_boundary
module: optimizer/optimizers
tags:
  - optimizer
  - boundary
  - architecture
---

# Optimizer Boundary

This file defines what belongs in `optimizer/optimizers`.

## Optimizers Own

```text
control update rules
method variants
method hyperparameter validation
method-specific proposal logic
method-specific optimizer_state payloads
compact API docs
links to deeper theory docs
```

## Systems Own

```text
physics
control_spec
evaluate
gradient
forward propagation
back propagation
objective terms
primary and secondary params
optional residuals/Jacobians/Hessians/HVPs
```

See:

- [OLGS README](../system_olgs/README.md)
- [OLGS Contract](../system_olgs/CONTRACT.md)

## Core Owns

```text
run_chunk
StepContext
StepProposal
AcceptanceDecision
StopTracker
SystemEvaluator
```

Optimizers call core; core stays method-agnostic.

## Utils Own

```text
diagnostic reports
finite-difference checks
gradient verification
Jacobian verification
repair tools
projected gradients
spectrum diagnostics
geometry probes
```

## Guesses Own

```text
zero/constant/ramp/sine/cosine/gaussian/sinc controls
Fourier controls
random smooth controls
restart perturbations
guess mixing and scaling
```

## Theory Owns

```text
derivations
update equations
mathematical assumptions
geometric intuition
convergence notes
method caveats
references
figures and illustrations
```

Long theory notes should live in `Theory/optimizers`, not in runtime modules.

## Related Notes

- [Optimizer Contract](OPTIMIZER_CONTRACT.md)
- [Theory Hub](../../Theory/optimizers/OPTIMIZER_THEORY_HUB.md)
- [Common Helpers](OPTIMIZER_COMMON.md)
