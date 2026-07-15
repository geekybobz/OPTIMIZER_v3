# OPTIMIZER v3 Context

Status: Phase 6 complete for review.
Last updated: 2026-07-15.

This file captures the durable context for implementation work. It should be read
before editing code.

## Current Goal

Build a new optimizer library in this folder, layer by layer. The old optimizer lives
outside this folder and should remain untouched unless a later migration step is
explicitly approved.

## Core Architecture Decisions

1. The public API should be direct:

   ```python
   opt.adam(...)
   opt.line_search(...)
   opt.fourier_guess(...)
   opt.repair_newton(...)
   ```

   Both styles should be supported:

   ```python
   opt.adam(system, controls, ...)
   ctx = opt.context(system)
   ctx.adam(controls, ...)
   ```

   The context style is for notebooks and curriculum. It must be explicit object
   state, not a mutable global such as `opt.system = system`.

2. `system.py` owns the physics, objective components, cost prefactors, analytical
   gradient, hard residuals, Jacobians, and higher derivative hooks.

3. Optimizers only move controls using the current system:

   ```python
   evaluation = system.evaluate(controls)
   gradient = system.gradient(controls)
   ```

4. Multi-objective optimization is controlled by system params:

   ```text
   J = wF * infidelity
     + w2 * F_norm2
     + w4 * C_sym_norm2
     + wE * energy
   ```

5. Curriculum means short optimizer chunks plus system-param changes, logs, guards,
   and checkpoint rollback.

   Preferred bound-system curriculum style:

   ```python
   ctx = opt.context(system, trace="run_001")
   r1 = ctx.adam(controls, maxiter=10)
   ctx2 = ctx.with_params(lambda2=1e6, lambda4=0.0)
   r2 = ctx2.adam(r1.controls, maxiter=10, warmstart=r1)
   ```

6. Numerical finite-difference derivatives are diagnostics or explicit fallback
   experiments. Analytical gradients are the normal path.

7. The first skeleton should avoid wrapper-heavy ideas. In particular, no first-pass
   `problem.py`; the active task is `system + controls + current system params`.

8. Modes are deferred until low-level tools are reliable.

## Important Downstream Context

The old optimizer is used in:

```text
../phd/
../phd/CODES/
```

Representative downstream systems include:

```text
../phd/HO_purification/system.py
../phd/CODES/three_level_system/system.py
../phd/CODES/non_unitary/system.py
../phd/CODES/non_unitary/system_6control.py
../phd/CODES/universal_robust_control_4th/system.py
```

Observed downstream pattern:

```text
systems already contain control dimensions and control names
systems already compute J and metrics
systems already derive gradients analytically
advanced scripts implement hard residuals, Jacobians, projected gradients, and repairs
```

This is why v3 should not assemble objectives externally.

## Build Protocol

For each phase:

```text
explain the phase
implement only that phase
run focused checks
summarize results
wait for review before the next phase
```

## Testing Rule

Future optimizer-facing tests should not rely only on toy/sample objective systems.
Use a temporary system shaped from the universal fourth-order robust-control model
until real downstream integration tests are connected:

```text
source idea:
  /Users/billabobz/phd/CODES/universal_robust_control_4th/model_note/main.tex
  /Users/billabobz/phd/CODES/universal_robust_control_4th/system.py

temporary fixture:
  tests/fixtures/universal_robust_4th/system.py

required shape:
  controls ux, uy, uz with endpoint samples
  system-owned weights infidelity_weight, lambda2, lambda4, energy_weight
  metrics J, infidelity, F_norm2, C_sym_norm2, energy
  analytical gradient
  hard residual vector [F, independent C_sym terms]
  vectorized finite-alpha direction simulation
```

Small toy systems can still be kept for very narrow edge cases, but new optimizer
behavior should be demonstrated on the fourth-order-style fixture first.

## Editing Rules

```text
do not modify old optimizer outside this folder unless explicitly approved
do not build future phases early
do not add auto modes before low-level tools are stable
do not treat generated optimizer outputs as source
```

## Python File Documentation Rule

Every implementation `.py` file should start with a detailed module introduction that
explains:

```text
what the file contains
why it exists in the architecture
which later modules depend on it
what it deliberately does not do
the key invariants or contracts a reviewer should check
```

Implementation files should also use structured section comments for readability.
Comments should explain design relevance and non-obvious choices, not restate trivial
assignments.

## Tracking Policy

Track:

```text
Python source
tests
templates
Markdown docs
package metadata
small intentional fixtures
```

Ignore:

```text
bytecode
virtual environments
test caches
build artifacts
runtime outputs
large generated arrays
logs/checkpoints produced by runs
```
