# OPTIMIZER v3 Restructure Plan

Status: review draft.
Last updated: 2026-07-15.

This folder stores the current redesign plan layer by layer. It supersedes the older
top-level draft wording for the architecture discussion, but it does not delete the
older notes. No code skeleton is implied yet. This is still the design review layer.

## Reading Order

Read the main files first:

1. `ARCHITECTURE.md` - complete architecture and module layout.
2. `FLOW.md` - how a run moves through system, controls, optimizer, logs, and checkpoints.
3. `PUBLIC_API.md` - intended direct call style.
4. `CURRICULUM_AND_CHECKPOINTS.md` - staged cost-parameter tuning and rollback.
5. `BUILD_ORDER.md` - layer-by-layer implementation order.

Then read the class and module details:

1. `classes/system.md`
2. `classes/controls.md`
3. `classes/evaluation_result_state.md`
4. `classes/stage.md`
5. `classes/trace_checkpoint.md`
6. `classes/core_engine.md`
7. `classes/optimizers.md`
8. `classes/schedules.md`
9. `classes/guesses.md`
10. `classes/constraints.md`
11. `classes/utils_logs_modes.md`

## Core Decision

The system owns the physics, objective components, cost prefactors, analytical
gradient, residuals, Jacobians, and higher derivative hooks.

The optimizer only moves controls using the current system:

```python
metrics = system.evaluate(controls)
gradient = system.gradient(controls)
```

For multi-objective optimization, the system exposes cost prefactors in its params.
Curriculum means changing those params between short optimizer chunks, checking logs,
and rolling back if a chunk damages important metrics.

## Design Rules

- Direct user API: `opt.adam(...)`, `opt.line_search(...)`, `opt.fourier_guess(...)`.
- Avoid wrapper-heavy concepts. Keep objects tied to real responsibilities.
- `system.py` is like an RL environment contract: it defines the functions every
  optimizer/tool can rely on.
- Use vectorized controls and multicore-ready evaluation utilities from the start.
- Build layer by layer: contract, controls, result/state, logs, engine, optimizers,
  guesses, diagnostics/repairs, modes.
- `modes/` is a placeholder until the low-level tools are reliable.

