---
role: deep
parent: ../../tests/TESTING_HUB.md
summary: "2026-10-12 review: two rounds of fixes with tests, issue statuses, the shared-core inspection, and shared-core phase 1."
read_when: "Read when checking the 2026-10-12 fixes and their evidence, or before designing or implementing the shared core."
status: active
reviewed: 2026-10-12
---
# Review 2026-10-12: fixes and shared-core inspection

This review tested the library the way a user would. Two new problems were
written from the scaffolds and solved with library tools only: a qubit state
transfer (OLGS) and a Duffing minimum-energy transfer by indirect shooting
(OLSS). The full suite, both scenario campaigns, and targeted probes were run,
and the code was inspected for logic that is duplicated between the two
families or tied to one method. Round 1 fixed five OLSS defects. Round 2, after
the user's answers, fixed every remaining bug-level issue and wrote the
shared-core design for review. After the user approved the design, shared-core
phase 1 was implemented.

## Verdict

- **The contracts are generic.** Both new problems fit OLGS and OLSS without
  any library change and run end to end without solver code.
- **Your workflow patterns are supported.** Gradient: forward propagation →
  backward propagation → gradient → library update, at one forward and one
  backward propagation per iteration with the template's cache. Shooting: guess
  → propagate → residuals and Jacobian → library update → independent
  verification.
- **Every bug-level issue is fixed and tested.** Each fix has regression tests
  that fail on the previous code (`20533da`). The suite passes under both
  `unittest` and pytest; the OLSS campaign and the OLGS matrix are unchanged.
- **The core is partly shared.** OLGS methods run through one engine and OLSS
  through separate loops; the same mechanics exist two to eight times. The
  user approved the shared-core design (`plans/shared_core/` in project
  memory). [Phase 1](#shared-core-phase-1) of five is implemented: one
  stop-reason registry, one serialization policy, one family rule, and catalog
  declarations that build the exports. Phases 2 to 5 share the numerical
  kernels, scales, the run loop, and continuation.

## Decisions

Taken on 2026-10-12:

1. The shared core gets a design document before the terminal interface.
2. The names added by the fixes are agreed.
3. The remaining issues are fixed in a structured way, with the documentation
   and memory updated.

After the design review:

4. The design is approved, and phase 1 started.
5. Non-finite numbers in JSON are the strings `"nan"`, `"inf"`, and `"-inf"`.
6. OLSS root methods become step functions of the shared engine (phase 4).
7. Phase 1 names: `StopReason` entries in `STOP_REASONS`, `to_plain`, and
   `system_family`. `CatalogItem` stays the one declaration per tool; the
   exports are built from the declarations, and a `root` field marks the root
   shortcuts.

Still open: the names of the concepts that later phases introduce, listed in
the design.

## Fixed in round 1: OLSS

| Fix | Problem | Change | Tests |
|---|---|---|---|
| 1 | An evaluation failure after the start (for example an integration blow-up) raised out of `solve_shooting`; one bad seed aborted a whole multistart and lost the good results | As in the OLGS engine, an unusable start raises `ValueError`. After the start a failing trial is rejected and the line search shrinks; other failures end with `nonfinite_trial`, `jacobian_failed`, `evaluation_failed`, or `verification_failed`, with the message in `ShootingResult.error`. Multistart and natural continuation record a failed seed or stage as `initial_evaluation_failed`; pseudo-arclength shrinks when a corrector point fails; direct shooting rejects a failed corrected trial | `FailedEvaluationTests`, 8 |
| 2 | The verification tolerance defaulted to the working tolerance (1e-10). Verification on a finer grid also sees the working grid's discretization error, so real problems never qualified, and continuation stopped at `minimum_step_reached` without saying why | A system declares its tolerance once with `basic_shooting_system_spec(..., verification_tolerance=...)`. Precedence: explicit argument, then system default, then working tolerance. Results record the value and its source; `continue_parameter` keeps one `step_history` row per attempt with the reason it failed | `VerificationToleranceTests`, 6 |
| 3 | `ParameterContinuationResult.to_dict()` failed `json.dumps` (a NumPy bool); NumPy flags returned by a project `verify` leaked into results | Every OLSS result converts to plain Python values; checked with strict JSON | `StrictJsonTests`, 2 |
| 4 (OLSS-014) | A numerical-declared Jacobian was compared at one step. When the project's own step coincided, the check reproduced the same stencil and passed with error 0 | Compared at two separated steps; passes only when both agree; reports each step's error | `NumericalJacobianCheckTests`, 3 |
| 5 (OLSS-022) | Multistart ranked qualified roots on working-accuracy metrics | Ranks on the verified selection metric when verification reports it; `selection_source` records which | `VerifiedRankingTests`, 1 |

These tests are in [test_olss_regressions.py](../../tests/test_olss_regressions.py).

## Issue status

Status values: **Fixed** (with tests), **Done** (documentation or files),
**Design** (resolved by the shared-core design, implemented after its review),
**Backlog** (deferred numerical method, after the shared core).

### Transactions

| ID | Issue | Status | Change |
|---|---|---|---|
| T-1 | No input decoding: a request that went through JSON failed | Fixed | Payloads are rebuilt against the system's own layout from live objects, `to_dict()` forms, channel or block mappings, or arrays; a mismatched layout returns `invalid_payload` |
| T-2 | Results were not validated: a NaN `J` with an undeclared metric returned `ok: True` | Fixed | Every operation calls the validated contract helpers; a violation returns the validator's error |
| T-3 | Only required calls; optional hooks refused; operations hard-coded in one `if/elif` chain | Fixed | One operation table per family with the optional hooks; a missing hook returns `capability_missing` before any computation. Run-level operations belong to the interactive interface design |
| T-4 | The V4 README said the core uses the envelope | Done | Sentence corrected |

Tests: [test_transactions.py](../../tests/test_transactions.py), 10.

### OLGS

| ID | Issue | Status | Change |
|---|---|---|---|
| G-1 | `diagnose_run` labels misled: absolute thresholds, trend starting after the first step, divergence tested before oscillation | Fixed | Thresholds relative to the run's starting value, taken from its chunk record; oscillation tested first; thresholds returned with the diagnosis |
| G-2 | An OLGS run without a trace was reported as OLSS and `stalled` | Fixed | Dispatch by result type; `no_trace` label; other objects raise `TypeError` |
| G-3 | The template did not scaffold forward → backward → gradient or a forward cache | Fixed | `forward_prop` with a one-entry cache, `_propagate_forward`, `back_prop`, `evaluate`, `gradient`, `cache_reset`; the generated test still passes unchanged |
| G-4 | `value_and_gradient` was used only at the initial point | Fixed | Under standard acceptance a trial is evaluated with `value_and_gradient` when the system has it: identical iterates with half the forward propagations |
| G-5 | The docs did not describe standard acceptance, bounds projection, failed-trial retries, stationarity, or the new stop reasons | Done | Core acceptance, stopping, and evaluation pages; method pages; OLGS lifecycle and hub |
| G-6 | Scale-aware tolerances (OLGS-010); equality plus active bounds (OLGS-011) | Design / Backlog | OLGS-010 is shared-core phase 3; OLGS-011 stays in the numerical backlog |

Tests: [test_run_diagnostics.py](../../tests/test_run_diagnostics.py), 6;
[test_templates.py](../../tests/test_templates.py), 2;
[test_engine_trial_gradient.py](../../tests/test_engine_trial_gradient.py), 3.

### OLSS

| ID | Issue | Status | Change |
|---|---|---|---|
| S-1 (OLSS-028) | The root-solver update rule is fixed (`newton`, `lm`, `damped`) | Design | OLSS root methods become step functions in shared-core phase 4 |
| S-2 (OLSS-021) | Evaluator caches grew without bound, in both families | Fixed | One bounded least-recently-used store (32 entries by default, `cache_size=None` keeps all) behind both evaluators |
| S-3 | NaN residuals of a failed seed are not valid strict JSON | Fixed | Phase 1: every result converts through `to_plain`, so a failed seed's residuals serialize as `"nan"`; `RepairResult` and `OptimizerResult` numbers too |
| S-4 | Deferred numerical work: OLSS-001 solver scaling, OLSS-002 sparse Jacobians, OLSS-004 and OLSS-019 SQP direct shooting, OLSS-023 adaptive damping, OLSS-006 article fixtures | Design / Backlog | OLSS-001 is shared-core phase 3; the rest stay in the numerical backlog |

### Catalog, tests, and documentation

| ID | Issue | Status | Change |
|---|---|---|---|
| C-1 | Exported OLSS tools and results missing from the catalog, including `finite_difference_shooting_jacobian` | Fixed | Ten catalog entries added in round 2; phase 1 declared every other export, so the known-gap list is empty and the exports are built from the declarations |
| C-2 | `optimizer.shooting.__all__` named a `search` function that does not exist, so `from optimizer.shooting import *` failed | Fixed | Phase 1: exports are built from the declarations; a test star-imports every namespace |
| D-1 | The API-reference test never ran under `unittest` and failed under pytest without `PYTHONPATH` | Fixed | `unittest` test case; the generator puts the checkout on its own path |
| D-2 | 91 front-matter summaries were boilerplate | Done | 90 specific summaries; the 91st page was parked (D-3) |
| D-3 | `STABILIZATION.md` was due to move to the workspace `temp/` | Done | Parked in `temp/v4/` as the unedited committed version, with its origin recorded |
| D-4 | The documentation pass dropped the README rule for removing a public entry, which the migration plan relies on | Done | Restored in the V4 README |

Tests: [test_catalog_coverage.py](../../tests/test_catalog_coverage.py), 8;
[test_api_reference.py](../../tests/test_api_reference.py), 1.

## Shared-core inspection

### What is shared today

Parallel execution (`utils/parallel.py`), the scaffold command for both
families, the runtime catalog, family-aware `system_info` and `evaluate`, the
validated `transact` envelope, the cross-family replay comparison, and, since
round 2, the bounded evaluator cache.

### Duplicated logic

| # | Concern | Separate implementations | Consequence |
|---:|---|---|---|
| 1 | Gauss–Newton / Levenberg–Marquardt residual solve | `repair_newton` with `_solve_repair_delta` (`utils/repairs.py:105`); `solve_shooting` with `_solve_delta` (`shooting/solvers.py:66`); the pseudo-arclength corrector and target landing (`shooting/continuation.py`); projected-gradient restoration (`optimizers/projected.py:158`) | Same method names, two LM formulations (algebraically equal), different stop names (`converged` versus `root_tolerance`), different history fields; a robustness fix must be repeated in five loops (fix 1 did it for the three OLSS loops) |
| 2 | Tangent-space projection (gradient minus Jacobian multiplier) | `utils/geometry.py:148`, `optimizers/projected.py:118`, direct shooting (`shooting/solvers.py:405`) | Three places to keep numerically consistent |
| 3 | Finite-difference Jacobian and its check | `utils/derivatives.py:208` and `system_olgs/derivatives.py:46` (identical bodies), `shooting/derivatives.py:13`; checks `verify_jacobian` (random directions) and `verify_shooting_jacobian` (full matrix) | Fix 4 had to be made in OLSS only; outputs use different keys |
| 4 | Rank and conditioning | `jacobian_geometry` (`utils/geometry.py`), `shooting_geometry_probe` (own SVD), the pseudo-arclength rank check (own SVD) | Three rank tolerances; keys `condition_number` versus `condition` |
| 5 | Validated, counted evaluation | `SystemEvaluator` and `ShootingEvaluator` | Only OLGS has structured `try_*` outcomes, so fix 1 needed local handling; the caches are now shared |
| 6 | Run loop: stopping, failure handling, history, events | `run_chunk` (`core/engine.py`); the CMA-ES generation loop with its own stopping and observer wiring (`optimizers/cma_es.py:402-495`); `solve_shooting`; `solve_direct_shooting`; two continuation correctors; `repair_newton` | Seven loops; each new rule (retries, events, interruption) is added per loop |
| 7 | Stop-reason vocabulary | OLGS: `gradient_converged`, `converged`, `proposal_failed`, `nonfinite_trial`, …; OLSS: `root_tolerance`, `least_squares_stationary`, `line_search_failed`, …; continuation: `completed`, `minimum_step_reached`, `max_steps`, … | No shared list; the diagnosis tool, terminal, and MCP must each learn every vocabulary |
| 8 | Iteration history and events | OLGS `Trace.iteration_records` plus `RunObserver` events and the tracker; OLSS `history` rows plus `on_event` dictionaries | Two event models reach the terminal interface and the logging redesign |
| 9 | Continue a run | OLGS `state=` and `warmstart=`; OLSS `previous_result=` | Two idioms for the same short-run workflow |
| 10 | Continuation | OLGS `opt.continuation` (gated stages, control transfer); OLSS `continue_parameter` (adaptive steps on one parameter) | Each lacks the other's features |
| 11 | Results | `OptimizerResult`, `RepairResult`, `ContinuationResult`; `ShootingResult`, `MultiStartResult`, `ShootingContinuationResult`, `ParameterContinuationResult` | Common fields (stop reason, iterations, history, counters, `to_dict`) without a common protocol |
| 12 | JSON conversion | Eight private helpers: `transactions.py`, `result.py`, `continuation.py`, `system_olgs/results.py`, `utils/diagnostics.py`, `system_olss/results.py`, `campaigns/__init__.py`, `shooting/evaluator.py` | Different handling of NumPy values, callables, unknown objects, and non-finite numbers; fix 3 was one symptom |
| 13 | Contract plumbing | `metric_spec` and `shooting_metric_spec` are identical; the hook probes list 25 and 19 hand-written flags; both base classes reimplement `cache_reset` and `cache_status` | Every new hook is added by hand in several places |
| 14 | Family detection | Four copies in `library.py`, one in `introspection.py`, one in `transactions.py`; the OLGS spec declares no `family` | Each new consumer adds another copy |
| 15 | Campaigns and seeds | OLGS `guesses/` and durable `campaigns/`; OLSS seed tools with in-memory multistart and refinement | OLSS runs cannot be journaled by the tracker |

Phase 1 resolves rows 7 (one registry, `opt.core.STOP_REASONS`), 12 (one
`opt.to_plain`), and 14 (one `opt.system_family`).

### Code tied to one family or method

- **The engine is the OLGS engine.** The Code Map calls it the shared gradient
  engine. It is typed to `Controls`, the objective `J`, and objective gradients,
  and its `StepContext` offers only `gradient`, so a root-solving method cannot
  run in it.
- **Method-specific branches inside the engine.** `gradient_source="smooth_gradient"`
  serves only the proximal method (`core/engine.py:762`). The HVP curvature
  check in `_stationary_reason` serves only second-order systems
  (`core/engine.py:286`).
- **Acceptance policy is split.** The engine default is monotone
  (`default_accept`). Adam, momentum, AdaGrad, and RMSProp pass
  `accept_finite_trial`. L-BFGS, nonlinear CG, and Newton-CG shrink rejected
  steps inside their own modules.
- **Tools serve one family.** `compare_methods`, `curvature_probe`,
  `verify_gradient`, `metric_report`, `diagnostic_report`, the tracker, and the
  campaigns serve OLGS only. `shooting_geometry_probe` serves OLSS only.
- **Registering is manual and repeated.** `diagnose_run` is wired in four files
  (nine references), and the catalog is a 2,240-line hand-written file. Three
  separate optimizer-name registries exist: `continuation.py:144`,
  `library.py:894`, and `campaigns/__init__.py:68`. Phase 1 replaced the three
  registries with the catalog declarations, which also build the exports.
- **Only one family declares scales.** `ShootingBlock` declares characteristic
  scales and `ControlSpec` does not.

### Shared-core design

The proposal is written up as a design for review in project memory,
`plans/shared_core/`: the front door with the open decisions, the nine
components (variables protocol, evaluator base, step engine, numerical kernels,
run record, stop-reason registry, serialization policy, continuation
coordinator, tool declarations, plus one family-detection function), and five
migration phases aligned with the interactive interface reference. Phase 1 is
implemented, as described below; phases 2 to 5 are not.

## Shared-core phase 1

Implemented after the design review. Every numerical result is unchanged.

| Component | Change | Tests |
|---|---|---|
| Stop-reason registry | `opt.core.STOP_REASONS` holds 38 `StopReason` entries: meaning, outcome class, families, continuability, and the tools that assign it. `diagnose_run` reports `stop_outcome` and `stop_continuable`; the API reference generates the table; the stopping pages link to it | [test_stop_reasons.py](../../tests/test_stop_reasons.py), 4; a source scan fails on any assigned reason the registry lacks |
| Serialization policy | `opt.to_plain` replaces the eight private helpers. Non-finite numbers become `"nan"`, `"inf"`, and `"-inf"`; an object without a known form raises instead of using `repr()` | [test_serialization.py](../../tests/test_serialization.py), 11, including strict JSON for every result type |
| Family rule | `opt.system_family` replaces six detection copies. OLGS specifications declare `family: "olgs"`; a missing family still means OLGS | `test_serialization.py` |
| Tool declarations | Every export has one `CatalogItem` (287 in all) with new `family` and `root` fields. Namespace and root `__all__`, `opt.methods()`, the continuation and campaign optimizer lookups, and the API reference are built from them | [test_catalog_coverage.py](../../tests/test_catalog_coverage.py), 8 |

Found and fixed during phase 1:

- `optimizer.shooting.__all__` named a nonexistent `search` (C-2).
- OLSS results, `RepairResult`, and `OptimizerResult` wrote raw floats, so a
  failed seed's NaN residuals produced invalid strict JSON (S-3).
- Declaring `family` in OLGS specifications changed every campaign
  fingerprint, so campaign evidence saved earlier could no longer resolve its
  system. The fingerprint now leaves out the implied OLGS family; a test in
  [test_campaigns.py](../../tests/test_campaigns.py) checks evidence without it.
- Bare-name lookups such as `opt.info("follow_timeline")` resolve to the root
  declaration, or to any of several declarations of one re-exported object.
- The README rule for removing a public entry was restored (D-4).

Deliberate limits:

- Tracker journals keep their stricter rule: records reject non-finite values,
  and the standalone reader validates them.
- `optimizer_log` keeps a hand-written export list because it must not import
  the optimizer package; a test checks the list against its declarations.
- The root shortcut functions, such as `opt.adam`, are still written by hand;
  the design generates them with the interactive operation registry.
- `opt.methods()` now covers all 114 root functions; the 50 earlier entries are
  unchanged.

## Verification

| Check | Result |
|---|---|
| `python -B -m unittest discover -s tests` | 255 tests pass (189 before the review, 44 in rounds 1 and 2, 22 in phase 1) |
| `python -B -m pytest tests` | 255 pass, without `PYTHONPATH` |
| New tests on the previous code (`20533da`, and `b2be361` for phase 1) | Fail, as recorded per fix below |
| OLSS scenario campaign | 20 pass, 2 gaps, 1 deferred failure (A5); every observed value identical after phase 1 |
| OLGS matrix | 425 runs, no non-finite result, stop reasons, acceptance counts, and values identical after phase 1; campaign summaries gain the declared `family` |
| Run time | Unchanged: alternating runs of `b2be361` and phase 1 overlap (300,000-variable Newton-CG 183 to 315 ms per iteration before, 185 to 225 after; four serial OLSS solves 45 to 48 s before, 44 to 47 after) |
| Review evidence scripts | Pass, including the transaction and diagnosis probes; the Duffing script checks strict JSON for every result |
| `tools/generate_api_reference.py --check` | Current |
| `mdp verify .` and `mdp inventory . --check` | Pass |

## Details

### Fix evidence

- **Fix 1.** With the Cubic test system (residual non-finite for `abs(z) >= 3`),
  Newton from 0.5 first tries z = 11. The previous code raised `ValueError`; the
  fix rejects three trials, accepts the fourth, and converges to z = 2. For the
  Duffing system at ε = 2, a multistart of 12 random seeds plus one seed that
  overflows previously aborted. It now returns 13 results: 3 qualified, 1
  unique root with energy 0.815562, and the overflowing seed recorded as
  `initial_evaluation_failed`.
- **Fix 2.** The Duffing working grid has 400 RK4 steps and verification uses
  4,000. Its verification residuals were 1.0e-10 to 2.7e-10 against a 1e-10
  tolerance, so continuation in ε stopped at 0.756. With
  `verification_tolerance=1e-7` declared once, continuation reaches ε = 2 in
  four steps (0.5, 1.0, 1.5, 2.0) and the multistart qualifies roots without
  any extra argument.
- **Fix 3.** `json.dumps(ParameterContinuationResult.to_dict())` raised
  `TypeError: Object of type bool is not JSON serializable`. Every OLSS result
  now passes `json.dumps(..., allow_nan=False)` for qualified runs.
- **Fix 4.** For a numerical-declared Jacobian whose residual has a small
  step-scale ripple, like adaptive-integrator noise, the previous check
  reported `passed=True` with `max_abs_error=0.0`. It now reports step errors
  `[0.0, 1.56e-4]` and `passed=False`.
- **Fix 5.** With two roots whose working cost estimate is biased, the
  previous multistart chose z = −1. It now chooses z = +1 by the verified cost
  and reports `selection_source="verification"`.
- **T-1 to T-3.** A request sent through JSON previously failed with
  `'dict' object has no attribute 'as_matrix'`, and a NaN objective returned
  `ok: True`. Both now behave correctly, and a missing optional hook returns
  `capability_missing`.
- **G-1, G-2.** On the qubit problem, Adam cut J from 0.0173 to 0.0045 while
  swinging and was labelled `diverging`; a steady 29 % drop was `slow`. They are
  now `oscillating` (with a 74 % net improvement) and `converging`, and the
  labels do not change when the objective is scaled by 1e-9 or 1e6.
- **G-3, G-4.** A system filled in from the new template runs 20 Adam
  iterations with 21 forward propagations. A system with only `evaluate` and
  `gradient`, which redoes the forward pass inside `gradient`, needs 41; giving
  it `value_and_gradient` brings it to 21 with bit-identical iterates.
- **S-2.** Evaluating 100 distinct points keeps the newest 32 entries; the
  newest point is still a cache hit.

### Files changed in this review

Git history records the commits: rounds 1 and 2 in `b2be361`, phase 1 in the
commit that adds this report's phase 1 section.

- Round 1 code: `optimizer/shooting/solvers.py`, `campaigns.py`,
  `continuation.py`, `derivatives.py`, `evaluator.py`;
  `optimizer/system_olss/results.py`, `specs.py`.
- Round 2 code: `optimizer/transactions.py`, `optimizer/utils/run_diagnostics.py`,
  `optimizer/core/engine.py`, `optimizer/core/evaluate.py`,
  `optimizer/core/_cache.py` (new), `optimizer/optimizers/_common.py`,
  `optimizer/shooting/evaluator.py`, `optimizer/catalog.py`,
  `optimizer/system_olgs/templates/system.py.tmpl` and `test_system.py.tmpl`,
  `optimizer_context.py`, `tools/generate_api_reference.py`.
- Tests (new): `test_olss_regressions.py`, `test_transactions.py`,
  `test_run_diagnostics.py`, `test_templates.py`,
  `test_engine_trial_gradient.py`, `test_catalog_coverage.py`; rewritten:
  `test_api_reference.py`.
- Documentation: the OLSS verification, data-model, and contract pages; the
  shooting methods page and hub; the transactions page and the V4 README; the
  core acceptance, stopping, and evaluation pages; the Adam, momentum,
  AdaGrad/RMSProp, L-BFGS, nonlinear CG, acceptance, and common-helper pages;
  the diagnostics page; the OLGS lifecycle page and hub; the OLSS scenario
  report's status rows; the Testing Hub; `INDEX.md`; 90 front-matter summaries;
  regenerated `API_REFERENCE.md` and `INVENTORY.md`.
- Moved: `STABILIZATION.md` to the workspace `temp/v4/`, recorded in
  `temp/README.md`.
- Evidence: this folder, with the qubit OLGS system, the Duffing OLSS system,
  and their run and probe scripts.
- Phase 1 code: `optimizer/core/stop_reasons.py`, `optimizer/serialization.py`,
  and `optimizer/families.py` (new); `catalog.py`, `library.py`,
  `introspection.py`, `transactions.py`, `continuation.py`, `result.py`,
  `campaigns/__init__.py`, `core/observer.py`, `shooting/evaluator.py`,
  `system_olgs/specs.py` and `results.py`, `system_olss/results.py`,
  `utils/diagnostics.py`, `repairs.py`, and `run_diagnostics.py`, every
  namespace `__init__.py`, and `tools/generate_api_reference.py`.
- Phase 1 tests: `test_stop_reasons.py` and `test_serialization.py` (new),
  `test_catalog_coverage.py` (rewritten), `test_campaigns.py`.
- Phase 1 documentation: the V4 README and Code Map, the Base and Core hubs,
  Core Stopping, Shooting Methods, Transactions, the OLGS contract, the OLSS
  data model, the diagnostics page, the Testing Hub, and this report;
  regenerated `API_REFERENCE.md` and `INVENTORY.md`.

### Reproduce

From the V4 checkout root:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests
```

```bash
PYTHONPATH=.:test_contained/review_2026_10_12/qubit_olgs python -B test_contained/review_2026_10_12/qubit_olgs/run_loop.py
```

```bash
PYTHONPATH=.:test_contained/review_2026_10_12/qubit_olgs python -B test_contained/review_2026_10_12/qubit_olgs/probes.py
```

```bash
PYTHONPATH=.:test_contained/review_2026_10_12/duffing_olss python -B test_contained/review_2026_10_12/duffing_olss/run_loop.py
```

## Connections

- Parent: [Testing Hub](../../tests/TESTING_HUB.md)
- Related: [OLSS Scenario Report](../olss_scenarios/REPORT.md)
- Related: [OLGS Convergence Test Design](../olgs_convergence/REPORT.md)
- Related: [Code Map](../../CODE_MAP.md)
- Related: [OLSS Verification](../../optimizer/system_olss/VERIFICATION.md)

---

[⌂ Home](../../README.md)
