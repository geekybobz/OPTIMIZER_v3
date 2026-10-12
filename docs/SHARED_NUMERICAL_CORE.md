---
role: topic
parent: ../README.md
summary: "Shared numerical mechanics, coordinate scaling, verification evidence and remaining solver limitations."
read_when: "Read to understand the completed shared core and the numerical work that remains outside it."
status: active
reviewed: 2026-10-13
---
# Shared numerical core

The shared core provides the numerical plumbing for OLGS optimization and OLSS
shooting. Their project contracts remain separate: a system still owns its
physics, objective or residual equations, derivatives, and verification rules.

The consolidation shares evaluator caches and call accounting, coordinate
scaling, linear algebra kernels, iteration budgets and cooperative boundaries,
bounded numerical records, and continuation coordination. Method adapters own
their proposal mathematics and acceptance requirements. A root solver, a
gradient method and a population method therefore share the lifecycle driver
without pretending to have the same mathematical iteration.

## The two mathematical problems

An OLGS method optimizes a scalar objective, for example pulse energy or
infidelity. Its gradient tells it which local direction changes that objective.
Bounds and residual constraints may restrict the allowed controls.

An OLSS root solver instead solves a vector equation, R(x) = 0. The entries of R
can represent endpoint errors, robustness conditions, or matching conditions
between trajectory segments. A small residual establishes approximate
feasibility for those declared equations. It does not establish minimum energy
or global optimality. Independent verification and named project checks still
decide whether a converged candidate is qualified.

Direct shooting adds a cost C(x) to the root problem: minimize C while keeping
R(x) = 0. Finding one feasible root and finding the best feasible root are
different tasks. This distinction explains several of the remaining upgrades.

## Coordinates and units

Declare a positive characteristic scale for each OLSS block or each OLGS
control channel. Unit scales preserve the previous behavior. If a coordinate
typically has magnitude one million and another has magnitude one thousandth,
using their raw magnitudes makes Euclidean distances misleading.

The numerical coordinates are z = S⁻¹x, where S is the diagonal matrix of
declared scales. Project hooks receive x in physical units. Gradients transform
as ∇z C = S ∇x C and residual Jacobians transform as Jz = Jx S. No residual is
silently reweighted: choosing variable units and choosing the relative
importance of residual equations are separate decisions.

Newton solves Jx S Δz = −R and returns Δx = S Δz. In an underdetermined problem,
the least-squares solution minimizes the dimensionless step norm. Fixed-damping
LM solves the regularized problem

```text
minimize over Δz:  ||Jx S Δz + R||² + damping ||Δz||².
```

For example, let R(a,b) = a/10⁶ + b/10⁻³ − 2. Starting from zero, the scaled
minimum-norm Newton step gives a = 10⁶ and b = 10⁻³. Each normalized coordinate
moves by one. A raw-coordinate minimum-norm solve puts almost the entire change
into b because its Jacobian coefficient is much larger. Both can satisfy the
equation, but only the scaled choice respects the declared characteristic units.

OLGS gradient methods use the same change of coordinates automatically when
nonunit scales are declared. Their internal method memory belongs to those
numerical coordinates; returned controls and state controls use physical units.
Shooting and repair step-length caps use dimensionless norms. Root convergence
continues to use the physical residual norm and the declared residual tolerance.
Finite differences use characteristic coordinate scales; OLSS also retains its
relative-to-current-value step rule.

The existing proximal hook defines a physical Euclidean proximal operation.
An anisotropic change of coordinates would require a weighted proximal operator
that cannot be inferred from that hook. Proximal methods therefore retain
physical coordinates and expose a scaled displacement norm for stationarity.
This is a limitation of the current hook, not a claim of full unit invariance
for proximal updates.

## Shared implementation and extension points

| Mechanic | Owner | Consumers |
|---|---|---|
| Variable structure and reconstruction | `core/variables.py` | Controls and ShootingVariables |
| Cache identity, bounded caches, attempted/successful calls, captured failures | `core/runtime.py` | Both evaluators |
| Newton/LM steps, tangent projections, finite differences and SVD rank rules | `core/numerics.py` | Repair, projected methods, shooting, continuation and diagnostics |
| Finite iteration lifecycle and failure-aware residual backtracking | `core/iteration.py` | Gradient engine, root/direct solvers, repair and CMA-ES |
| Coordinate transformation for OLGS hooks and state | `core/coordinates.py` | Gradient engine |
| Population evaluation and private worker factories | `core/population.py` | CMA-ES |
| Adaptive step policy, attempt records and Newton corrector | `core/coordinator.py` | Staged, natural and arclength continuation |
| Bounded numerical evidence | `core/record.py` | Results and diagnosis |

Wide residual problems use residual-space linear algebra to avoid allocating
an identity matrix with one row per control variable. Shooting retains its
augmented least-squares formulation where appropriate to avoid unnecessarily
squaring the Jacobian condition number.

Custom OLSS updates are supported by `solve_shooting(step=...)`, and the simple
`run_chunk(step=...)` route dispatches by family. The shared `StepContext`
provides `variables`, residuals and Jacobian for OLSS; its gradient is absent.
Return ShootingVariables or StepProposal containing them. The root adapter
still validates the layout, backtracks for residual decrease, applies declared
bounds and independently verifies converged candidates. Use `solve_shooting`
directly for its verification options, damping and restart arguments.

Each result has a bounded RunRecord, retaining the latest 256 samples by
default and scalar aggregates for the whole chunk. Truncation is explicit.
Native shooting and repair history fields remain available but are bounded.
Reading diagnosis, serializing or saving a result performs no new physics
evaluations. Live callbacks are optional and are not persistent iteration logs.
Frontend command history remains separate.

CMA-ES retains generator state across explicit chunks. This does not expand
the durable checkpoint capabilities advertised by the operation service: a
saved physical candidate and an exact live solver continuation are still
distinct operations. Old checkpoints missing the newly added optional fields
receive their compatibility defaults.

## CPU safeguard

Local executor worker counts are capped at available logical CPUs minus three,
with a minimum of one for machines with three or fewer available CPUs. CPU
affinity or a process-specific count is used when the platform exposes it.
Managed jobs reject configurations whose worker count times native threads per
worker exceeds this budget. A ten-logical-core machine has a seven-slot budget.

These are admission limits for one executor or one job coordinator. They do
not allocate CPUs exclusively or subtract unrelated applications' current CPU
usage. Two separately launched executors can still overlap. Native-thread
environment limits also depend on project libraries honoring them. Verification
campaigns should run sequentially, with explicit one-thread BLAS/OpenMP settings,
so their combined demand stays within the budget.

## Remaining numerical work

These items are outside the shared-core consolidation. They do not prevent
ordinary small/medium dense problems from running, but they matter in the
specific situations below.

### Sparse and matrix-free Jacobians — OLSS-002

A dense Jacobian stores every one of its m×n entries. With many shooting nodes,
most entries may be zero because a segment only depends on neighboring states.
Storing and factoring the full array then wastes memory and time. A sparse
solver stores nonzero entries; a matrix-free solver requests products Jv and
Jᵀv without building J. Discoverable optional hooks alone do not implement that
solver path. The current dense solvers remain appropriate for the small
regression systems. Large multiple-shooting problems need a separately tested
linear solver and preconditioner path.

### Stronger constrained direct shooting — OLSS-004 and OLSS-019

The current direct solver projects a cost gradient onto the local tangent
space, tries a step, and corrects the residual equations. A tangent direction
satisfies Jd ≈ 0 to first order; a finite step can still leave the curved
feasible set. Restoration can undo useful cost progress, and clipping against
bounds can destroy tangency. Near active constraints or poor conditioning this
method can stall even when a better feasible solution exists.

Sequential quadratic programming (SQP) would solve a local quadratic objective
with linearized constraints. A null-space method would optimize in independent
directions that preserve those constraints locally. Either needs a clear
stationarity test, bound handling, globalization and independent cost evidence.
The current projected-gradient method is usable as a local heuristic, with its
declared feasibility and verification gates. Its stop reason does not certify a
constrained optimum or a global minimum.

The cart scenario A5 provides a concrete example. Its exact reference is
(6, −12, 0), with energy 12. The current 200-iteration direct run retains about
(5.644, −9.864, −2.136), with energy 12.02536, and stops at its iteration budget.
It improves the initial feasible root (energy about 12.2283), but misses the
benchmark's requested accuracy. Sharing the engine preserves that algorithm;
it does not replace it with SQP.

### Adaptive LM damping — OLSS-023

The current `lm` method uses a fixed damping value. Larger damping shortens
unstable directions but can slow progress; smaller damping approaches Newton
but may produce poor steps near singular Jacobians. Backtracking limits the
accepted displacement, but it does not adjust the regularization itself.
Adaptive LM would compare predicted residual improvement with actual
improvement and change damping accordingly. It requires persistent damping
state and regression evidence on ill-conditioned examples. The existing name
therefore denotes fixed-damping regularized Gauss-Newton.

### Equality constraints together with bounds — OLGS-011

Projection onto equality tangents followed by clipping into a box generally
does not project onto their intersection. For example, x+y=1 with x≤0.4:
clipping the equality-feasible point (0.5,0.5) produces (0.4,0.5), which violates
the equality. Restoration and feasibility checks can expose or reduce the
error, but do not make this sequence an exact intersection projection. A
bounded least-squares or active-set projection is needed for that guarantee.

### Physical regression fixtures — OLSS-006

Synthetic tests establish algorithmic properties and catch software
regressions. They do not independently validate a project's physical equations,
truncations or accepted research solutions. Additional project-approved
fixtures should include the physical model/version, candidate, independently
verified residuals and costs, and tolerances. Creating those references requires
scientific acceptance; importing an old numerical root alone is insufficient.

General operator preconditioning, stronger line-search variants and accelerated
proximal algorithms are also separate method enhancements. The new scaling
addresses differences in coordinate units; it does not solve every source of
ill-conditioning.

## Verification and release boundary

`tests/test_shared_core.py` checks scale invariance, physical hook arguments,
underdetermined roots, custom OLSS updates, bounded records, call accounting,
legacy checkpoint defaults, split-run state, population parity, continuation,
rank rules and CPU admission. Existing tests and numerical campaigns provide
the broader compatibility evidence. Campaign comparison separates expected
scaling changes and metadata changes from unit-scale numerical regressions.
The [verification report](../test_contained/shared_core/REPORT.md) records the
current baseline comparison and retained campaign findings.

The source checkout contains this development work. The installed wheel is a
separate release; promotion follows the explicit
[release procedure](RELEASE_PROCEDURE.md).

---

[⌂ Home](../README.md)
