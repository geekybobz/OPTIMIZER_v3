---
role: deep
parent: ../../tests/TESTING_HUB.md
summary: "OLSS issue record: resolution status, original findings, and campaign evidence."
read_when: "Read when checking an OLSS issue or rerunning the scenario campaign."
status: active
reviewed: 2026-10-12
---
# OLSS Issues Found

This record began as the 2026-10-11 OLSS robustness review. The original finding
descriptions remain below for provenance; the status tables are authoritative for
the current checkout. These are library findings, not scientific claims about the
retained `rob_unversal_article` solutions.

## Resolution status — 2026-10-12

| ID | Status | Current result |
|---|---|---|
| OLSS-001 | Partly resolved | Root deduplication uses `ShootingBlock.scale`; full solver-coordinate scaling remains later numerical work. |
| OLSS-002 | Deferred | Dense Jacobians remain the intended small/medium-system default; sparse/operator solvers remain later work. |
| OLSS-003 | Resolved | `continue_parameter` supplies adaptive natural continuation and exact target landing. |
| OLSS-004 | Deferred | Production SQP/null-space direct shooting remains later numerical work. |
| OLSS-005 | Resolved | `VerificationCheck` gives project-owned named gates and all supplied checks participate in qualification. |
| OLSS-006 | Deferred | Additional independently accepted physical regression fixtures remain project work. |
| OLSS-007 | Resolved | Scaled default equivalence plus optional project `roots_equivalent` canonicalization are consumed by multistart. |
| OLSS-008 | Resolved | Random, perturbation-cloud, and elite-refinement seed tools are public. |
| OLSS-009 | Resolved | OLSS construction types are in the runtime catalog with regression coverage. |
| OLSS-010 | Resolved for execution | Continuous resume history and start/iteration/terminal event hooks cover CLI/MCP execution; persistent logging/checkpoints stay in the later logging redesign. |
| OLSS-011 | Resolved | `compare_shooting_replay` reports explicit grid, interpolation, units, and metric differences. |
| OLSS-012 | Resolved | Root solves distinguish `least_squares_stationary`, `bound_stalled`, `singular_step`, and ordinary line-search failure. |

Additional review findings recorded after the original file:

| ID | Status | Current result |
|---|---|---|
| OLSS-013 | Resolved | Qualification requires independent verification options or a distinct system default and records provenance. |
| OLSS-014 | Resolved 2026-10-12 | The single separated step could still coincide with the project's own step and pass hollowly; numerical-declared Jacobians are now compared at two separated steps ([2026-10-12 review](../review_2026_10_12/REPORT.md), fix 4). |
| OLSS-015 | Resolved | Pseudo-arclength checks numerical rank `n-1` and accepts redundant rows. |
| OLSS-016 | Resolved | Top-level evaluation/gradient/continuation dispatch is family-aware and names the OLSS equivalent. |
| OLSS-017 | Resolved | OLSS uses `System(primary=..., secondary=...)` and declares both parameter groups. |
| OLSS-018 | Resolved | Multistart defaults to processes when a factory is available, otherwise serial. |
| OLSS-019 | Deferred with OLSS-004 | Direct-shooting verification cost and stationarity policy belong to the later solver upgrade. |
| OLSS-020 | Resolved | Continuation retries with step shrink and grows successful fast steps within declared limits. |
| OLSS-021 | Resolved 2026-10-12 | Both evaluators use one bounded least-recently-used cache ([2026-10-12 review](../review_2026_10_12/REPORT.md), issue S-2). |
| OLSS-022 | Resolved 2026-10-12 | Multistart ranks on verified selection metrics and records `selection_source` ([2026-10-12 review](../review_2026_10_12/REPORT.md), fix 5). |
| OLSS-023 | Deferred | The current `lm` name still denotes fixed-damping Gauss-Newton; no measured practical deficit justified renaming in this pass. |
| OLSS-024 | Resolved | `optimizer-system --family olss` generates a contract-shaped system and test. |
| OLSS-025 | Resolved 2026-10-12 | Evaluation failures after the start become stop reasons; failed seeds and stages are recorded instead of aborting campaigns ([2026-10-12 review](../review_2026_10_12/REPORT.md), fix 1). |
| OLSS-026 | Resolved 2026-10-12 | Systems declare a verification tolerance; results record the tolerance and its source; natural continuation records every attempt ([2026-10-12 review](../review_2026_10_12/REPORT.md), fix 2). |
| OLSS-027 | Resolved 2026-10-12 | All OLSS result objects serialize as strict JSON ([2026-10-12 review](../review_2026_10_12/REPORT.md), fix 3). |
| OLSS-028 | Design | The root-solver update rule is fixed (`newton`, `lm`, `damped`); OLSS root methods become step functions in shared-core phase 4 ([2026-10-12 review](../review_2026_10_12/REPORT.md), issue S-1). |

The high- and medium-priority contract, integration, diagnostic, and workflow gaps
are therefore closed. The remaining items are explicitly the later numerical or
project-evidence group.

## Original findings and rationale

### OLSS-001 — Variable scales are declared but not consumed numerically

`ShootingBlock.scale` records a positive characteristic scale, but the current
root, continuation, and direct-shooting steps operate in raw variable
coordinates. Root deduplication also compares raw vectors. This matters when
initial controls, costates, multipliers, and continuation parameters differ by
several orders of magnitude. In the scenario campaign, two roots differing in
the small-scale variable were incorrectly merged because the large coordinate
dominated the relative-distance test.

Required follow-up: define one solver-coordinate scaling rule and apply it
consistently to linear solves, damping, step limits, tangent calculations,
finite-difference steps, and root deduplication. Residuals must remain the
unscaled physical residuals required by the [OLSS contract](../../optimizer/system_olss/CONTRACT.md).

### OLSS-002 — Dense Jacobians are the only consumed representation

Sparse and matrix-free hooks are discoverable, but every implemented solver
currently converts or requests a dense Jacobian. This is sufficient for the
tested 4, 8, 12, 20, and 39-variable scenarios, but it can become the limiting
factor for large multiple-shooting systems with many internal nodes.

Required follow-up: add a solver path that consumes `sparse_jacobian` or
`jacobian_operator` without changing the required OLSS calls. Dense behavior
should remain the default for small systems.

### OLSS-003 — Continuation lacks natural campaigns and exact landing

Natural parameter stepping works by calling `with_secondary()` and solving
each successive system, and this workflow is covered by tests. The public
continuation operation, however, is only the pseudo-arclength driver for a
one-dimensional root branch. Pseudo-arclength also has no stop-at-parameter or
exact-landing operation, so the scenario campaign had to detect that it crossed
the desired parameter and run a separate root solve.

Required follow-up: decide whether repeated secondary-parameter stepping needs
a named convenience operation with structured step history, failure recovery,
and verification evidence. Until then, callers must orchestrate that loop.

### OLSS-004 — Direct shooting uses a basic projected-gradient iteration

The direct-shooting test preserves feasibility and reduces the declared energy,
but the three-coefficient cart scenario reaches its iteration limit before the
known energy-stationary point. A 200-iteration probe reduced the energy from
`12.2283` to `12.0254` but stopped at `maxiter` instead of reaching the known
`E=12` solution. The returned candidate remains qualified as a verified
feasible point; `stop_reason="maxiter"` correctly shows that cost stationarity
was not reached.

Required follow-up: evaluate a constrained SQP, null-space quasi-Newton, or
trust-region method before treating direct shooting as a production-scale
energy optimizer. Do not weaken the separation between feasibility,
stationarity, and global optimality.

### OLSS-005 — Verification gates are project-owned but not structurally named

The required `verify()` transaction can perform tighter reintegration and can
set `numerical_success=False`, while its metrics can carry endpoint,
stationarity, matching, and conservation checks. The generic result does not,
however, expose standardized named acceptance gates for the article workflow:
residual acceptance, tighter replay, stationarity audit, and cost comparison
only among candidates that passed the earlier gates.

Required follow-up: before terminal or MCP work, decide whether verification
needs a small structured gate list. Any such structure must preserve
project-owned physics and must not imply a local- or global-optimality
certificate.

### OLSS-006 — Article-level scientific replay remains incomplete

The published 8-variable/9-residual second-order root was reproduced: its
working residual was `3.53e-9`, tighter 6000-step replay gave `1.35e-11`, and
the energy was `64.754737709`. The 12/12 hard-gate, 20/21 fourth-order, and 4/4
planar sixth-order cases are currently deterministic dimension- and
workflow-matched contract probes. They do not reproduce the complete article
equations or retained roots.

Required follow-up: add read-only regression fixtures from independently
accepted project artifacts for each physical formulation. Compare residual
groups, tighter replay, stationarity audit, reconstructed controls, and declared
costs without importing the article's optimizer policy into OLSS.

### OLSS-007 — Root equivalence is not scale- or symmetry-aware

Multistart deduplication uses a relative Euclidean distance in raw variable
coordinates. Besides the scaled-root collision in OLSS-001, a gauge-free qubit
campaign reported several points on one equal-energy rotational orbit as
different roots.

Required follow-up: allow the project to provide a canonicalization or
root-equivalence hook. The default should remain ordinary numerical distance,
but it should use the agreed solver-coordinate scaling once OLSS-001 is fixed.

### OLSS-008 — Research campaign construction is still manual

The multistart executor accepts explicit seeds, but the article workflow also
needs scale-aware random starts, perturbation clouds around anchors, elite
selection, child generation, and focused refinement. In the broad probe, only
2 of 24 cold random starts qualified and neither reached the retained
`E=64.75` branch; a small cloud around the retained root returned 8 of 8 seeds
to one qualified root.

Required follow-up: design reusable seed-cloud and refinement tools without
putting project-specific campaign policy into the OLSS contract.

### OLSS-009 — OLSS root types are missing from the runtime catalog

The numerical shooting operations are discoverable, but the catalog does not
currently expose `OLSS`, `ShootingBlock`, `ShootingSpec`, `ShootingVariables`,
or `basic_shooting_system_spec` as root public entries. This weakens the
declared-API discovery path intended for later terminal and MCP consumers.

Required follow-up: add the public OLSS construction and inspection types to
the catalog and cover them with catalog tests.

### OLSS-010 — Shooting solvers bypass shared run infrastructure

`ShootingResult.to_dict()` is JSON-serializable, but `solve_shooting` currently
has no observer, trace, checkpoint, warm-start, or shared stopping hooks. This
is separate from the future logging redesign, but the missing integration must
be resolved before a uniform terminal or MCP execution workflow is promised.

Required follow-up: decide which existing core lifecycle pieces are genuinely
generic and connect shooting without forcing waveform-gradient assumptions.

### OLSS-011 — Cross-contract replay has no reusable helper

`recover_controls()` allowed the retained shooting control to be evaluated by
the contained OLGS system, but the cross-check required manual grid conversion
and the two systems used different quadrature conventions. The result was
close but not identical.

Required follow-up: define a small, explicit replay comparison tool that reports
grid, interpolation, units, metric identity, and tolerance provenance. It must
not require OLGS and OLSS to share a physics class.

### OLSS-012 — Some stop reasons are too coarse

An inconsistent overdetermined problem correctly remained unqualified, but it
ended as `line_search_failed` after reaching the nonzero least-squares minimum.
An unreachable bounded root also ended with the same reason after clipping at
the bound.

Required follow-up: distinguish nonzero least-squares stationarity,
bound-stalled steps, singular corrections, and ordinary line-search failure in
structured result evidence.

## Findings that passed

- Square, overdetermined, and underdetermined residual systems behave as
  declared.
- Inconsistent equations are not promoted as qualified roots.
- A deliberately wrong Jacobian is rejected by the derivative checker.
- Bounds are enforced during root steps.
- Pseudo-arclength continuation crosses a fold and verifies every retained
  point.
- Thread and process multistart use worker-owned system instances, deduplicate
  roots, and rank only qualified candidates.
- Multiple shooting, higher-accuracy verification, and optional control
  reconstruction work through the existing contract.
- The retained second-order 8/9 root and a nearby eight-seed process campaign
  were reproduced successfully.

## Evidence

- [Article-shaped and numerical-stress systems](systems.py)
- [Method compatibility probe](probe.py)
- [Broad executable scenario campaign](run_scenarios.py)
- [Article scenario tests](../../tests/test_olss_scenarios.py)
- [Core OLSS robustness tests](../../tests/test_olss.py)
- [Root and direct solvers](../../optimizer/shooting/solvers.py)
- [Pseudo-arclength continuation](../../optimizer/shooting/continuation.py)

Verification performed for this issue record:

```text
complete V4 fast suite                         189 passed
focused OLSS contract/scenario tests            21 passed
integrated OLGS/OLSS regression tests             9 passed
full updated article-shaped campaign              20 pass, 2 gaps, 1 deferred numerical failure
campaign runtime / process workers                123.8 s / 8 workers
physical OLSS-to-OLGS replay helper               passed and JSON-serialized
```

The remaining campaign failure is the explicitly deferred OLSS-004 direct-shooting
solver: it stays feasible and reduces energy but does not reach the known stationary
point within the declared first-order budget. The remaining gaps are later numerical
work (full solver scaling) or fixture/evidence work, not missing high/medium contract
tools. The campaign runner now calls the new continuation, seed, event, catalog, and
cross-family replay tools directly.

---

[⌂ Home](../../optimizer/system_olss/SYSTEM_OLSS_HUB.md)
