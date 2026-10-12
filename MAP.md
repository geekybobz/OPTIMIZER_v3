---
role: map
parent: README.md
summary: "Recommended documentation paths for common OPTIMIZER tasks."
read_when: "Use when you know what you want to accomplish but not which topic owns it."
status: active
reviewed: 2026-10-12
---
# Route Map

## If you are new

1. Read the [V4 entry](README.md).
2. Choose [OLGS](optimizer/system_olgs/SYSTEM_OLGS_HUB.md) for objective-gradient
   problems or [OLSS](optimizer/system_olss/SYSTEM_OLSS_HUB.md) for shooting
   residual equations.
3. Read the relevant contract before selecting a numerical method.
4. Use the [Optimizer Hub](optimizer/optimizers/OPTIMIZERS_HUB.md) or
   [Shooting Hub](optimizer/shooting/SHOOTING_HUB.md).
5. Verify shapes and derivatives before interpreting convergence.

## If you are implementing a project system

- OLGS required calls, shapes, and optional capabilities:
  [OLGS Contract](optimizer/system_olgs/CONTRACT.md).
- OLGS primary and secondary configuration:
  [OLGS Parameters](optimizer/system_olgs/PARAMS.md).
- OLSS required calls and invariants:
  [OLSS Contract](optimizer/system_olss/CONTRACT.md).
- OLSS variables, specifications, and result objects:
  [OLSS Data Model](optimizer/system_olss/DATA_MODEL.md).
- OLSS independent acceptance evidence:
  [OLSS Verification](optimizer/system_olss/VERIFICATION.md).
- Create a starting implementation with `optimizer-system --family olgs` or
  `optimizer-system --family olss`.

## If you are choosing a numerical method

- Compare gradient and derivative-free methods:
  [Optimizer Methods](optimizer/optimizers/OPTIMIZER_METHODS.md).
- Check common inputs, stopping, state, and results:
  [Optimizer Contract](optimizer/optimizers/OPTIMIZER_CONTRACT.md).
- Compare shooting root, continuation, seed, and replay tools:
  [Shooting Methods](optimizer/shooting/SHOOTING_METHODS.md).
- Look up exact signatures and defaults:
  [API Reference](API_REFERENCE.md).

## If you run short experiments

1. Run five to ten iterations.
2. Classify the history with `diagnose_run`.
3. Probe local curvature with `curvature_probe` when appropriate.
4. Compare candidate methods with `compare_methods`.
5. For OLSS, inspect rank and weak blocks with `shooting_geometry_probe`.
6. Continue or switch using the returned state, variables, or warm start.

Start with the [Utils Hub](optimizer/utils/UTILS_HUB.md) and
[State and Warmstart](optimizer/optimizers/OPTIMIZER_STATE_WARMSTART.md).

## If you are integrating another interface

- Read [System Transactions](optimizer/TRANSACTIONS.md) for the shared
  request-response boundary.
- Read the [Code Map](CODE_MAP.md) before changing dispatch, serialization, or
  runtime discovery.
- Treat the runtime catalog as the public API source. Do not scrape prose to
  discover signatures.

## If you are changing the library

- Shared iteration behavior: [Core Hub](optimizer/core/CORE_HUB.md).
- Method-specific mathematics: [Optimizer Hub](optimizer/optimizers/OPTIMIZERS_HUB.md).
- Shooting numerical operations: [Shooting Hub](optimizer/shooting/SHOOTING_HUB.md).
- Complete file ownership: [Code Map](CODE_MAP.md).
- Required checks: [Testing Hub](tests/TESTING_HUB.md).

---

[← Previous](README.md) · [⌂ Home](README.md) · [Next →](INDEX.md)
