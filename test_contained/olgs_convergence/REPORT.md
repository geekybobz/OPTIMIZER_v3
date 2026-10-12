---
role: deep
parent: ../../tests/TESTING_HUB.md
summary: "OLGS validation design, scenario matrix, method coverage, and the 2026-10-12 outcome."
read_when: "Read when checking OLGS method structure or rerunning the matrix."
status: active
reviewed: 2026-10-12
---
# OLGS Convergence Test Design

This page records the OLGS validation design and its 2026-10-12 implementation
outcome. Scenario results are configuration-specific numerical evidence, never a
universal optimizer ranking.

## Implementation outcome — 2026-10-12

The contract stayed unchanged. The analytical S0-S6 fixtures, physical S8/S9
adapters, workflow checks, stress cases, and fast regression checks now live under
`test_contained/olgs_convergence/` and `tests/test_olgs_olss_integrations.py`.

The high- and medium-priority tool gaps found by the campaign were fixed:

- standard Adam, momentum, AdaGrad, and RMSProp accept finite trials by default;
  monotone acceptance remains an explicit opt-in guard;
- L-BFGS and nonlinear CG shrink rejected steps, and nonlinear CG restarts invalid
  directions;
- Newton-CG restores its configured full step after acceptance;
- the shared engine projects declared box bounds and retries failed trials with a
  bounded shrink policy;
- first-order methods accept serializable constant, adaptive, exponential,
  warm-up, and cosine schedules;
- stationary HVP-capable points report negative-curvature evidence instead of a
  hollow gradient-converged result;
- CMA-ES accepts state, warmstart, and standard target/stall stopping options;
- visible gradient tolerances provide default stationarity stops;
- `diagnose_run`, `curvature_probe`, and `compare_methods` support the intended
  five-to-ten-iteration inspect-and-decide workflow.

Still deferred as later numerical design: scale-aware gradient norms and
preconditioning (OLGS-010), exact projection onto intersecting equality/bound
constraints or a stronger constrained solver (OLGS-011), strong-Wolfe search,
advanced quasi-Newton updates, and accelerated proximal methods.

Verification evidence:

```text
complete V4 fast suite               189 passed
contained OLGS convergence matrix    425 jobs, 0 exceptions
retained controls                    0 non-finite
result serialization                 425/425 passed
workflow checks                      no errors
serial/thread/process campaigns      all operational
physical derivative checks           all passed
large-dimension stress               300,000 variables retained finite controls
```

Thirteen deliberately aggressive momentum configurations ended with the bounded
`nonfinite_trial` stop after three failed-trial recovery attempts. This is the
expected structural response to unsuitable parameters; no invalid trial replaced
the last valid controls.

## Decision Summary

The present OLGS contract is sufficient for the planned gradient test program.
No second gradient-system class is needed. The required five-method transaction

```text
system_spec -> control_spec -> evaluate -> gradient -> with_secondary
```

already feeds every ordinary gradient optimizer through the shared engine. The
optional OLGS capabilities also cover combined value/gradient evaluation,
constraints, bounds, Hessian-vector products, proximal objectives, and explicit
control transfer.

The validation scenarios now exist. Further API changes should be considered only
when a retained scenario exposes a repeatable limitation that cannot be addressed
by optimizer configuration or an existing optional capability.

## What Must Be Established

The future suite must answer five different questions without mixing them:

1. **Contract correctness:** does a system obey OLGS shapes, declarations,
   objective direction, finiteness, and capability rules?
2. **Derivative correctness:** does the analytical gradient, Jacobian, or HVP
   match an independent numerical check?
3. **Optimizer correctness:** does each method preserve its own state,
   acceptance, stopping, and stationarity invariants?
4. **Convergence behavior:** which methods are robust to conditioning,
   curvature, scale imbalance, constraints, and poor initial controls?
5. **Workflow integration:** can methods be warm-started, switched, continued,
   inspected, and later driven through CLI and MCP without special project code?

A single assertion such as `final_J < initial_J` cannot establish all five.

## Existing Architecture to Reuse

The design deliberately reuses these current boundaries:

| Boundary | Existing owner | Role in the test design |
|---|---|---|
| System semantics | `system_spec()` | Objective direction, metrics, derivatives, constraints |
| Control layout | `ControlSpec` and `Controls` | Canonical shapes and method-independent handoff |
| Numerical calls | `SystemEvaluator` | Validation, caching, and operation counters |
| Iteration transaction | `run_chunk` | Gradient, proposal, trial, acceptance, state, stopping |
| Method state | `RunState.optimizer_state` | Warm start and branch-correct private memory |
| Result | `OptimizerResult` | Controls, metrics, trace, counters, stop reason |
| Method switching | `WarmStartState` | Compatible control/metric handoff without leaking private state |
| Curriculum | `continuation` and `with_secondary` | Explicit staged objective or parameter changes |

This reuse is important: convergence tests should exercise the same public
transactions that a future CLI or MCP server will call.

## Two-Layer Test Architecture

### Layer A: deterministic regression suite

Location proposed for later implementation:

```text
tests/olgs_convergence/
  fixtures.py
  assertions.py
  test_derivatives.py
  test_first_order.py
  test_second_order.py
  test_constraints.py
  test_workflows.py
```

Properties:

- small analytical systems;
- fixed seeds where randomness is essential;
- seconds, not minutes;
- invariant-based assertions;
- no claims that one optimizer is universally best;
- suitable for source and installed-wheel verification.

### Layer B: contained convergence studies

Location proposed for later implementation:

```text
test_contained/olgs_convergence/
  README.md
  scenarios.py
  methods.py
  run_matrix.py
  summarize.py
  baselines/
```

Properties:

- larger grids and multiple seeds;
- work-normalized method comparisons;
- the existing HO purification and second-order PMP adapters;
- machine-readable reports plus a compact human summary;
- numerical evidence only, never a global-optimality or physics certificate;
- not part of every fast CI run.

This separation prevents expensive physical examples from becoming flaky unit
tests while still testing the exact public integration used by real projects.

## Proposed Test Transactions

The following names are **discussion candidates**, not accepted public API.
They describe the structure that the test harness needs and may remain private to
the tests.

```text
ScenarioSpec
  identity, system_factory, initial-control factory, oracle, capabilities,
  seeds, scale family, claim boundary

MethodSpec
  public method name, variant, options, supported capabilities, budget policy

RunRequest
  scenario id, method id, seed, initial controls, budget, tolerances

RunEvidence
  initial/final metrics, stationarity evidence, feasibility, call counters,
  acceptance history, stop reason, finite-state checks, reproducibility ids

SuiteReport
  scenario matrix, invariant results, work-normalized summaries, limitations
```

The transaction should be JSON-safe through explicit `to_dict()` output so the
same evidence can later be rendered by Python, CLI, or MCP without changing the
optimizer contract.

## Scenario Matrix

### S0: exact isotropic quadratic

Purpose:

- smoke-test every compatible method;
- verify minimize/maximize direction symmetry;
- verify known optimum and stationarity;
- verify stopping, counters, and result serialization.

Oracle: exact minimizer/maximizer, gradient, Hessian, HVP, and objective gap.

### S1: rotated ill-conditioned quadratic

Use deterministic orthogonal rotations and condition numbers such as
`1e2`, `1e4`, and `1e6`.

Purpose:

- expose zigzagging and scale sensitivity;
- compare normalized descent, momentum, Adam-family scaling, RMSProp,
  L-BFGS, nonlinear CG, Newton-CG, and trust region;
- measure the value of preconditioning or control scaling before proposing an
  API extension.

Oracle: exact optimum, gradient norm, objective gap, and spectral condition.

### S2: Rosenbrock-type curved valley

Purpose:

- exercise nonlinear curvature and narrow valleys;
- expose fixed-step fragility;
- evaluate globalization for L-BFGS and nonlinear CG;
- compare rough-search then polish workflows.

Oracle: known minimizer and local objective value. The test must not require a
specific iteration count from every method.

### S3: nonconvex saddle and double-well family

Purpose:

- test negative-curvature behavior in Newton-CG and trust region;
- ensure accepted states stay finite;
- compare deterministic outcomes across declared initial points;
- verify that the report says "local candidate" rather than "global optimum."

Oracle: known stationary points and Hessian signatures, not a universal global
minimum claim.

### S4: channel-scale and discretization family

Construct a multi-channel objective where channels differ by several orders of
magnitude, then repeat it at multiple control dimensions.

Purpose:

- distinguish coordinate scaling from physical time-step scaling;
- detect methods whose behavior changes unexpectedly as the grid is refined;
- test whether tolerances and gradient norms need scale-aware definitions.

Oracle: exact scaled optimum and a discretization-normalized objective.

### S5: bounds and equality constraints

Include:

- pure box constraints with an active optimum;
- a linear equality manifold;
- equality plus active bounds;
- a mildly nonlinear equality requiring restoration;
- deliberately inconsistent constraints.

Purpose:

- validate feasibility restoration;
- validate projected-KKT stationarity;
- distinguish `restoration_failed` from ordinary objective rejection;
- assess whether projected gradient is sufficient before considering an
  augmented-Lagrangian or SQP family.

### S6: composite smooth plus L1 objective

Purpose:

- validate `smooth_gradient` and `prox` as one capability bundle;
- use the proximal-gradient mapping as stationarity evidence;
- test sparse support recovery and zero-coordinate stability;
- provide a baseline before considering acceleration.

Oracle: closed-form soft-threshold solution.

### S7: continuation and method switching

Workflows:

```text
Adam -> L-BFGS
line search -> L-BFGS
Adam -> trust region when HVP is available
coarse secondary parameters -> strict secondary parameters
coarse control grid -> fine grid through explicit control transfer
```

Purpose:

- verify controls and metrics transfer;
- verify incompatible optimizer-private state is discarded;
- verify compatible same-method state is retained;
- verify a changed objective is re-evaluated;
- verify layout changes require explicit system-owned transfer.

### S8: contained HO purification

Reuse `test_contained/ho_purification/system.py`.

Purpose:

- directional verification of the exact discrete gradient;
- compare conservative line search, Adam-family exploration, and deterministic
  polishing from identical controls;
- test sensitivity to grid size, energy weight, and initial guess family;
- preserve the existing boundary that this is an engineering convergence study,
  not a purification or thermodynamic certificate.

### S9: contained second-order robust-control objective

Reuse `test_contained/second_order_pmp/system.py`.

Purpose:

- directional verification of the exact discrete adjoint gradient;
- test multi-channel scale imbalance;
- compare single methods with `Adam -> L-BFGS`;
- keep finite-perturbation robustness as a post-run diagnostic rather than a
  hidden optimized metric;
- report only numerical candidates and local stationarity evidence.

## Method Coverage

| Family | Variants or modes to design for | Main scenarios |
|---|---|---|
| Line search | fixed, normalized, backtracking, Armijo | S0-S4, S8-S9 |
| Momentum | heavy-ball, Nesterov, restart, clipped | S0-S4 |
| Adam family | Adam, AMSGrad, AdamW, RAdam, AdaBelief | S0-S4, S8-S9 |
| Adaptive first order | AdaGrad, RMSProp | S1, S4 |
| L-BFGS | history sizes, curvature rejection, step clipping | S0-S4, S7-S9 |
| Nonlinear CG | FR, PR, PR+, HS, non-descent restart | S0-S3 |
| Projected gradient | bounds, equality, combined | S5 |
| Newton-CG | HVP, damping, negative curvature | S0-S4 |
| Trust region | radius shrink/grow, negative curvature | S0-S4 |
| Proximal gradient | fixed step, rejection shrink | S6 |

CMA-ES may be included only as a derivative-free reference in contained reports.
It should not be counted as OLGS gradient-method validation.

## Evidence and Assertions

### Contract and numerical safety

Every run should establish:

- controls preserve keys, shape, dtype contract, and finite values;
- all declared scalar metrics remain finite;
- missing capabilities fail before numerical iterations;
- stop reasons are explicit;
- rejected trials do not replace current or best controls;
- trace counters match the system calls visible to the optimizer;
- serialization contains no hidden Python-only object needed by CLI or MCP.

### Derivative evidence

For each analytical derivative:

- compare several seeded directions, not only coordinate directions;
- use central differences with a small step sweep;
- report absolute error, relative error, and the selected step;
- test at the initial point and at least one nontrivial interior point;
- include a deliberately wrong derivative fixture to prove the checker fails;
- keep numerical derivatives diagnostic-only, never a silent optimizer fallback.

Exact analytical fixtures can use strict tolerances. Physical discrete-adjoint
fixtures should use tolerances justified by their propagator and grid rather than
copying the quadratic threshold.

### Stationarity evidence

Use the correct measure for each problem:

| Problem | Required evidence |
|---|---|
| Unconstrained smooth | native gradient norm and objective gap when known |
| Bound/equality constrained | feasibility norm and projected-KKT norm |
| Composite/proximal | proximal-gradient mapping norm |
| Newton/trust region | gradient norm, CG residual, curvature/radius evidence |
| Physical nonconvex | gradient norm plus explicit local-candidate wording |

### Convergence comparisons

Record at least:

```text
initial and final objective
best objective
stationarity measure
accepted and rejected steps
evaluate, gradient, value-and-gradient, HVP, residual, Jacobian, and prox counts
non-finite or failed calls
stop reason
seed and initial-control identity
```

Wall time is useful diagnostic information but should not be a hard regression
assertion because it is machine-dependent.

### Budget fairness

Iteration counts are not comparable across methods: Armijo may evaluate several
trials per iteration, while Newton-CG may perform many HVPs. Initial contained
reports should therefore show both:

- progress by outer iteration;
- progress by operation counters.

The first implementation should not change the engine merely to force equal work
budgets. If the studies need hard evaluation/HVP ceilings, that is evidence for a
future shared work-budget policy.

## Advanced Convergence Ideas to Evaluate

These are ordered by how much can be learned without changing OLGS.

### Priority 0: use existing composition better

1. **Derivative preflight:** run explicit directional checks before expensive
   physical campaigns.
2. **Exploration then polish:** compare Adam or RMSProp followed by L-BFGS,
   nonlinear CG, or trust region.
3. **Continuation:** gradually change objective weights or discretization through
   `with_secondary` and explicit control transfer.
4. **Step clipping and restart policies:** test the existing safeguards on poorly
   scaled or curved problems.

These require scenario and workflow code, not a new system contract.

### Priority 1: shared globalization for curvature methods

Current L-BFGS and nonlinear CG use configured step sizes and default acceptance;
the L-BFGS source explicitly notes that strong-Wolfe search is not yet available.
S1-S3 should determine whether rejected steps, lost curvature pairs, or slow
polishing justify a reusable line-search service.

Candidate design, to discuss after evidence:

```text
LineSearchPolicy
  armijo
  strong_wolfe
  bounds on trial count and step size
```

This belongs on the optimizer/core side, not in OLGS, because it chooses steps
rather than describing the physical system.

### Priority 1: scale and preconditioning policy

S1 and S4 should determine whether scale imbalance remains after Adam-family and
normalized methods are used. Possible designs, not yet selected:

- strict positive control scales declared in `system_spec`;
- an optional system scaling/preconditioner transaction;
- an optimizer-supplied preconditioner callable;
- scale-aware norms and tolerances without modifying the gradient itself.

Naming and ownership must be discussed before implementation. A generic
preconditioner is more powerful, while declared control scales are easier for CLI
and MCP discovery.

### Priority 2: spectral and scheduled first-order steps

Candidates:

- Barzilai-Borwein initial steps for deterministic first-order methods;
- a shared step-size schedule transaction for Adam/momentum;
- reduce-on-rejection or reduce-on-stall policies;
- periodic momentum and nonlinear-CG restarts.

The schedule should be optimizer-owned and serializable. It should not add another
required OLGS method.

### Priority 2: more robust quasi-Newton updates

Evaluate:

- damped or Powell-corrected curvature pairs;
- scale-aware curvature acceptance;
- history reset after repeated rejection;
- strong-Wolfe integration.

The test evidence must separate "objective decreased" from "useful curvature was
retained."

### Priority 2: accelerated proximal methods

Use S6 to evaluate backtracking proximal gradient and FISTA-style acceleration.
Require a restart safeguard and retain the proximal-gradient mapping as the
stationarity measure.

### Priority 3: stronger constrained methods

Only if S5 shows repeatable restoration or progress failures should the library
consider augmented-Lagrangian or SQP-style coordination. Those methods would reuse
the existing residual, Jacobian, bounds, and possibly HVP transactions; they should
not create a new system class.

### Out of current scope

Stochastic-gradient semantics, minibatches, and noisy-oracle convergence are not
part of the present OLGS contract, which requires an analytical gradient of the
declared objective. They should not be introduced through a test fixture without a
separate design decision.

## Implementation Phases and Gates

### Phase 1: analytical fixtures and oracles

Create S0-S6 systems with exact solutions or independently checkable derivatives.

Gate:

- every fixture passes the five-method OLGS contract;
- optional capability declarations match callable hooks;
- every oracle is stated explicitly.

### Phase 2: derivative validation

Design gradient, Jacobian, and HVP checks across scales and directions.

Gate:

- correct derivatives pass justified tolerances;
- intentionally wrong derivatives fail;
- no numerical fallback enters a normal optimizer run.

### Phase 3: method invariants

Cover every variant and optimizer-private state branch.

Gate:

- finite state and explicit stopping;
- accepted/rejected branch correctness;
- correct stationarity measure;
- minimize/maximize symmetry where the method supports both directions.

### Phase 4: convergence matrix

Run S0-S6 with declared configurations and operation-counter reporting.

Gate:

- deterministic reports are reproducible;
- no brittle universal ranking is embedded in unit tests;
- each suspected convergence limitation has a retained trace and minimal
  reproducer.

### Phase 5: workflow composition

Exercise method switches, same-method continuation, parameter continuation, and
grid transfer.

Gate:

- no private state leaks across incompatible methods;
- changed objectives are re-evaluated;
- provenance identifies every stage and transfer.

### Phase 6: physical contained studies

Run S8 and S9 with several declared initial controls and seeds.

Gate:

- discrete gradients pass their own directional checks;
- conclusions remain numerical and local;
- comparisons are work-normalized;
- post-run physics diagnostics are not confused with optimized metrics.

### Phase 7: architecture decision

Classify every observed limitation as one of:

```text
configuration issue
test/fixture issue
optimizer implementation issue
shared-core policy gap
optional OLGS capability gap
research limitation or unproved claim
```

Only then approve or reject new names such as a line-search policy, scale
declaration, preconditioner transaction, or work-budget policy.

## CLI and MCP Readiness Criteria

Before the later CLI and MCP phases, the convergence harness should expose one
stable machine-readable report containing:

- system and capability identity;
- method and complete serializable options;
- initial-control identity and seed;
- budgets and operation counters;
- final metrics and stationarity evidence;
- stop reason and bounded failure details;
- artifact paths and claim boundary.

The CLI should submit a request and render this report. The MCP server should expose
the same request/report transaction. Neither interface should reimplement optimizer
logic or define its own method-specific result format.

## Later decisions still reserved for discussion

Before implementing the deferred numerical group, agree on:

1. whether the proposed scenario/method/evidence names remain test-private or become
   public API;
2. whether physical convergence studies live under `test_contained` or a separate
   benchmark package;
3. how control scaling should be owned and named if S1/S4 justify it;
4. whether a shared strong-Wolfe policy is the first optimizer enhancement;
5. whether hard work budgets are needed or operation-normalized reports are enough;
6. which physical grids, seeds, and objective weights are authoritative for S8/S9.

The fixtures and harness are now implemented. None of the remaining choices requires
an OLGS contract rewrite before evidence justifies it.

## Related Notes

- [OLGS System Hub](../../optimizer/system_olgs/SYSTEM_OLGS_HUB.md)
- [OLGS Contract](../../optimizer/system_olgs/CONTRACT.md)
- [OLGS Derivatives](../../optimizer/system_olgs/DERIVATIVES.md)
- [Optimizer Methods](../../optimizer/optimizers/OPTIMIZER_METHODS.md)
- [Conditional Numerical Capabilities](../../optimizer/optimizers/NUMERICAL_CAPABILITIES.md)
- [Testing Hub](../../tests/TESTING_HUB.md)

---

[⌂ Home](../../optimizer/system_olgs/SYSTEM_OLGS_HUB.md)
