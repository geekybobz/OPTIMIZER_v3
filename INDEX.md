---
role: index
parent: README.md
summary: "Complete semantic index of authored OPTIMIZER V4 documentation."
read_when: "Use when locating the authoritative topic for a known concept."
status: active
reviewed: 2026-10-12
---
# Topic Index

## Orientation

- [V4 entry](README.md) — scope, boundaries, and primary routes.
- [Route Map](MAP.md) — reading paths by task.
- [Code Map](CODE_MAP.md) — runtime components, ownership, and change impact.
- [API Reference](API_REFERENCE.md) — generated public signatures and catalog metadata.
- [Markdown Inventory](INVENTORY.md) — generated exhaustive file listing.

## System contracts

- [OLGS System Hub](optimizer/system_olgs/SYSTEM_OLGS_HUB.md) — gradient-system entry point.
- [OLGS Contract](optimizer/system_olgs/CONTRACT.md) — required and optional OLGS calls.
- [OLGS Parameters](optimizer/system_olgs/PARAMS.md) — primary and secondary configuration.
- [OLGS Lifecycle](optimizer/system_olgs/LIFECYCLE.md) — propagation, evaluation, and gradient flow.
- [OLGS Derivatives](optimizer/system_olgs/DERIVATIVES.md) — derivative capabilities and fallback policy.
- [OLSS System Hub](optimizer/system_olss/SYSTEM_OLSS_HUB.md) — shooting-system entry point.
- [OLSS Contract](optimizer/system_olss/CONTRACT.md) — required calls and invariants.
- [OLSS Data Model](optimizer/system_olss/DATA_MODEL.md) — variable, specification, and result structures.
- [OLSS Verification](optimizer/system_olss/VERIFICATION.md) — independent qualification evidence.
- [System Transactions](optimizer/TRANSACTIONS.md) — shared request-response envelope.
- [Foreground Operation Service](optimizer/operations/SERVICE.md) — selected contexts, native dispatch, candidates and bounded results.
- [Structured Terminal](optimizer/cli/TERMINAL.md) — interactive calls, scripts, names, inspection and foreground lifetime.

## Numerical execution

- [Base API Hub](optimizer/BASE_HUB.md) — shared public objects and root calls.
- [Core Hub](optimizer/core/CORE_HUB.md) — shared engine and method boundary.
- [Optimizer Hub](optimizer/optimizers/OPTIMIZERS_HUB.md) — gradient and derivative-free methods.
- [Optimizer Contract](optimizer/optimizers/OPTIMIZER_CONTRACT.md) — common call and result rules.
- [Optimizer Methods](optimizer/optimizers/OPTIMIZER_METHODS.md) — method taxonomy and selection.
- [Shooting Hub](optimizer/shooting/SHOOTING_HUB.md) — OLSS numerical operations.
- [Shooting Methods](optimizer/shooting/SHOOTING_METHODS.md) — root, continuation, seed, and replay tools.
- [Schedules Hub](optimizer/schedules/SCHEDULES_HUB.md) — reusable step-size policies.

## Preparation and inspection

- [Guess Hub](optimizer/guesses/GUESSES_HUB.md) — initial-control generators.
- [Utils Hub](optimizer/utils/UTILS_HUB.md) — diagnostics, checks, geometry, repair, and spectrum.
- [Visualization Hub](optimizer/visualization/VISUALIZATION_HUB.md) — explicit visualization data products.

## Coordination and evidence

- [Campaigns](optimizer/campaigns/CAMPAIGNS.md) — durable multi-stage workflows.
- [Tracker Hub](optimizer/tracker/TRACKER_HUB.md) — journals, checkpoints, and observer bridge.
- [Logs Hub](optimizer/logs/LOGS_HUB.md) — in-memory trace and checkpoint behavior.
- [Standalone Log Reader](optimizer_log/OPTIMIZER_LOG_READER.md) — read-only run-folder inspection.
- [Testing Hub](tests/TESTING_HUB.md) — test ownership and scenario evidence.
- [Release Evidence](RELEASE_EVIDENCE.md) — released-wheel provenance.

## Theory

- [Optimizer Theory](Theory/optimizers/OPTIMIZER_THEORY_HUB.md)
- [Guess Theory](Theory/guess/GUESS_THEORY_HUB.md)
- [Utility Theory](Theory/utils/UTILS_THEORY_HUB.md)

---

[← Previous](MAP.md) · [⌂ Home](README.md) · [Next →](CODE_MAP.md)
