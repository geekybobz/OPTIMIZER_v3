---
role: generated
parent: README.md
summary: Generated inventory of protocol-managed Markdown files.
read_when: Read when an exhaustive file listing is needed.
---
<!-- Generated file. Do not edit directly.
Source: Markdown properties in this collection.
Regenerate with: markdown_protocol.py inventory <root> --write
-->

# Markdown Inventory

Generated from the structured properties of this collection. The authored
Front Door or `INDEX.md` remains the semantic route map.

## Core files

| path | role | summary | read when |
|---|---|---|---|
| [AGENTS.md](AGENTS.md) | topic | Maintainer rules for source ownership, generated docs and explicit release promotion. | Read before changing this repository. |
| [CODE_MAP.md](CODE_MAP.md) | topic | Runtime components, execution paths, source ownership, and change-impact guide. | Read before changing architecture, dispatch, contracts, or shared execution behavior. |
| [INDEX.md](INDEX.md) | index | Complete semantic index of authored OPTIMIZER V4 documentation. | Use when locating the authoritative topic for a known concept. |
| [MAP.md](MAP.md) | map | Recommended documentation paths for common OPTIMIZER tasks. | Use when you know what you want to accomplish but not which topic owns it. |
| [README.md](README.md) | front-door | Entry point for the OPTIMIZER V4 contracts, numerical tools, workflows, and verification evidence. | Start here when using, extending, or reviewing OPTIMIZER V4. |
| [RELEASE_EVIDENCE.md](RELEASE_EVIDENCE.md) | topic | Release verification scope and the authority for exact artifact provenance. | Read when checking what a release verifies and where its hashes live. |
| [Theory/guess/GUESS_THEORY_HUB.md](Theory/guess/GUESS_THEORY_HUB.md) | front-door | Entry to the mathematical notes behind each initial-guess generator. | Read when you need the mathematics of a guess generator rather than its call. |
| [Theory/guess/THEORY_COMPOSITE_RESTARTS.md](Theory/guess/THEORY_COMPOSITE_RESTARTS.md) | topic | Mathematics of scaling, mixing, and perturbing controls for restarts. | Read when combining or perturbing earlier controls to restart a run. |
| [Theory/guess/THEORY_CONSTANT_RAMP.md](Theory/guess/THEORY_CONSTANT_RAMP.md) | topic | Constant, linear, quadratic, and smoothstep ramp controls and where each fits. | Read when choosing a simple deterministic initial control. |
| [Theory/guess/THEORY_FOURIER_SERIES.md](Theory/guess/THEORY_FOURIER_SERIES.md) | topic | Finite sine-series guesses, their low-frequency bias, and per-channel coefficients. | Read when building a smooth band-limited initial control. |
| [Theory/guess/THEORY_HARMONIC_WAVES.md](Theory/guess/THEORY_HARMONIC_WAVES.md) | topic | Sine and cosine guesses on normalized time, their frequency content, and envelopes. | Read when using a single-harmonic initial control. |
| [Theory/guess/THEORY_LOCALIZED_PULSES.md](Theory/guess/THEORY_LOCALIZED_PULSES.md) | topic | Gaussian and sinc pulse guesses and their time-frequency localization trade-off. | Read when an initial control should be concentrated in time. |
| [Theory/guess/THEORY_RANDOM_CONTROLS.md](Theory/guess/THEORY_RANDOM_CONTROLS.md) | topic | Random control matrices: distributions, amplitude range, and roughness. | Read when using an unstructured random initial control. |
| [Theory/guess/THEORY_RANDOM_FOURIER.md](Theory/guess/THEORY_RANDOM_FOURIER.md) | topic | Random Fourier coefficients: smoothness bias and seeded reproducibility. | Read when generating reproducible smooth random starts. |
| [Theory/guess/THEORY_SMOOTH_RANDOM.md](Theory/guess/THEORY_SMOOTH_RANDOM.md) | topic | Smoothed random controls: smoothing model, correlation length, edge padding, and scaling. | Read when tuning the smoothness of a random initial control. |
| [Theory/optimizers/OPTIMIZER_THEORY_HUB.md](Theory/optimizers/OPTIMIZER_THEORY_HUB.md) | front-door | Entry to the derivations and convergence notes behind each optimizer. | Read when you need why a method works rather than how to call it. |
| [Theory/optimizers/THEORY_ADAM_FAMILY.md](Theory/optimizers/THEORY_ADAM_FAMILY.md) | topic | Adam moments and the AMSGrad, AdamW, RAdam, and AdaBelief variants, with convergence notes. | Read when choosing an Adam variant or interpreting its moment state. |
| [Theory/optimizers/THEORY_ADAPTIVE_SCALING.md](Theory/optimizers/THEORY_ADAPTIVE_SCALING.md) | topic | AdaGrad and RMSProp per-coordinate scaling and how they relate to Adam. | Read when choosing between AdaGrad, RMSProp, and Adam. |
| [Theory/optimizers/THEORY_CMA_ES.md](Theory/optimizers/THEORY_CMA_ES.md) | topic | Population search with isotropic or diagonal sampling, elite means, and sigma adaptation. | Read when using the derivative-free CMA-ES method. |
| [Theory/optimizers/THEORY_LBFGS.md](Theory/optimizers/THEORY_LBFGS.md) | topic | Quasi-Newton curvature pairs, limited memory, the two-loop recursion, and restarts. | Read when using L-BFGS or interpreting its history state. |
| [Theory/optimizers/THEORY_LINE_SEARCH.md](Theory/optimizers/THEORY_LINE_SEARCH.md) | topic | Descent steps, directional derivatives, backtracking, and the Armijo condition. | Read when choosing a line-search variant or step rule. |
| [Theory/optimizers/THEORY_MOMENTUM.md](Theory/optimizers/THEORY_MOMENTUM.md) | topic | Heavy-ball and Nesterov momentum, restarts, and clipping. | Read when choosing a momentum variant or step size. |
| [Theory/optimizers/THEORY_NONLINEAR_CG.md](Theory/optimizers/THEORY_NONLINEAR_CG.md) | topic | Fletcher-Reeves, Polak-Ribiere, and Hestenes-Stiefel directions and restart logic. | Read when choosing a nonlinear CG variant. |
| [Theory/utils/THEORY_DIAGNOSTIC_REPORTS.md](Theory/utils/THEORY_DIAGNOSTIC_REPORTS.md) | topic | What metric and diagnostic reports measure: control, gradient, and residual norms. | Read when interpreting the numbers in a diagnostic report. |
| [Theory/utils/THEORY_FINITE_DIFFERENCE_CHECKS.md](Theory/utils/THEORY_FINITE_DIFFERENCE_CHECKS.md) | topic | Central-difference gradient and Jacobian checks, error interpretation, and choosing eps. | Read when a derivative check fails or eps needs choosing. |
| [Theory/utils/THEORY_NEWTON_REPAIR.md](Theory/utils/THEORY_NEWTON_REPAIR.md) | topic | Least-squares and damped Newton steps that reduce a residual vector, with line search and stop reasons. | Read when repairing controls onto residual constraints. |
| [Theory/utils/THEORY_NULLSPACE_PROJECTION.md](Theory/utils/THEORY_NULLSPACE_PROJECTION.md) | topic | Constraint-preserving directions: gradient decomposition and the J J^T residual-space solve. | Read when projecting a gradient onto the tangent space of constraints. |
| [Theory/utils/THEORY_RESIDUAL_JACOBIAN_GEOMETRY.md](Theory/utils/THEORY_RESIDUAL_JACOBIAN_GEOMETRY.md) | topic | Local residual linearization: rank, singular values, condition number, and nullspace dimension. | Read when interpreting a geometry probe. |
| [Theory/utils/THEORY_SPECTRUM_SMOOTHNESS.md](Theory/utils/THEORY_SPECTRUM_SMOOTHNESS.md) | topic | Real FFT spectra, high-frequency fraction, difference roughness, and total variation of controls. | Read when judging how smooth or band-limited a control is. |
| [Theory/utils/UTILS_THEORY_HUB.md](Theory/utils/UTILS_THEORY_HUB.md) | front-door | Entry to the mathematical notes behind diagnostics, derivative checks, repair, and geometry. | Read when you need the mathematics behind a utility tool. |
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | topic | Learn the installed optimizer through a small, reproducible terminal example. | Start here for your first installation and interactive run. |
| [docs/INSTALLATION.md](docs/INSTALLATION.md) | topic | Install a committed release in Optimizer and use it from any project folder. | Read when installing, checking global commands, or configuring notebooks. |
| [docs/MIGRATION.md](docs/MIGRATION.md) | topic | Use the current system contracts and retire legacy trees and logger integrations. | Read when converting an older project or looking for the former workspace layout. |
| [docs/RELEASE_PROCEDURE.md](docs/RELEASE_PROCEDURE.md) | topic | Promote an approved commit into an isolated global wheel, verify provenance and roll back. | Read before publishing, updating or restoring the installed optimizer. |
| [docs/WORKFLOWS.md](docs/WORKFLOWS.md) | topic | Find the supported routes for independent experiments, control exchange, jobs and verification. | Read after the first run to choose your next workflow. |
| [optimizer/BASE_HUB.md](optimizer/BASE_HUB.md) | front-door | Shared data types and top-level calls of the base catalog group: controls, results, state, and root calls. | Read when working with Controls, results, run state, or a top-level opt call. |
| [optimizer/TRANSACTIONS.md](optimizer/TRANSACTIONS.md) | topic | Family-aware request-response boundary shared by OLGS, OLSS, and future adapters. | Read when integrating systems with shared core, terminal, or MCP-facing code. |
| [optimizer/campaigns/CAMPAIGNS.md](optimizer/campaigns/CAMPAIGNS.md) | front-door | In-memory multi-stage campaigns, owned candidates and explicit numerical validation. | Read when comparing initial guesses, methods or refinement candidates in Python. |
| [optimizer/core/CORE_ACCEPTANCE_GUARDS.md](optimizer/core/CORE_ACCEPTANCE_GUARDS.md) | topic | Engine acceptance: the default rule, each method's default, metric guards, and failure reporting. | Read when a trial is rejected unexpectedly or a stricter acceptance rule is needed. |
| [optimizer/core/CORE_BOUNDARY.md](optimizer/core/CORE_BOUNDARY.md) | topic | What belongs in optimizer/core and what does not. | Read before adding shared behavior to the engine. |
| [optimizer/core/CORE_CONTRACT.md](optimizer/core/CORE_CONTRACT.md) | topic | The step and accept callables that connect any optimizer method to the shared engine. | Read when writing a custom update rule for run_chunk. |
| [optimizer/core/CORE_EVALUATION.md](optimizer/core/CORE_EVALUATION.md) | topic | SystemEvaluator: validated system calls, the bounded evaluation cache, and call counters. | Read when counting system calls or checking what the engine caches. |
| [optimizer/core/CORE_HUB.md](optimizer/core/CORE_HUB.md) | front-door | Entry to the shared chunk engine: system boundary, acceptance, stopping, and the method boundary. | Start here before changing or extending the engine. |
| [optimizer/core/CORE_LIFECYCLE.md](optimizer/core/CORE_LIFECYCLE.md) | topic | Exact order of one chunk, its accepted and rejected branches, failure exits, and chaining. | Read when a chunk ends unexpectedly or chunks are chained. |
| [optimizer/core/CORE_STOPPING.md](optimizer/core/CORE_STOPPING.md) | topic | Stopping rules, stationarity, and every stop reason a chunk can report. | Read when interpreting result.stop_reason or configuring stops. |
| [optimizer/guesses/GUESSES_HUB.md](optimizer/guesses/GUESSES_HUB.md) | front-door | Entry to the initial-control generators: families, common options, and theory. | Start here when choosing an initial control. |
| [optimizer/guesses/GUESS_BOUNDARY.md](optimizer/guesses/GUESS_BOUNDARY.md) | topic | What belongs in optimizer/guesses and what belongs to systems, optimizers, and theory. | Read before adding a guess generator. |
| [optimizer/guesses/GUESS_COMMON_API.md](optimizer/guesses/GUESS_COMMON_API.md) | topic | Shared guess options: target resolution, time grid, channels, envelopes, endpoints, and scaling. | Read when a guess option behaves differently than expected. |
| [optimizer/guesses/GUESS_COMPOSITE.md](optimizer/guesses/GUESS_COMPOSITE.md) | topic | Building new starting controls by scaling, mixing, and perturbing existing Controls. | Read when restarting from earlier controls. |
| [optimizer/guesses/GUESS_CONTRACT.md](optimizer/guesses/GUESS_CONTRACT.md) | topic | Input and return rules every guess generator follows. | Read when writing or validating a guess generator. |
| [optimizer/guesses/GUESS_FOURIER.md](optimizer/guesses/GUESS_FOURIER.md) | topic | fourier_guess: deterministic low-frequency sine-series controls. | Read when you want a smooth deterministic start. |
| [optimizer/guesses/GUESS_HARMONIC.md](optimizer/guesses/GUESS_HARMONIC.md) | topic | Sine and cosine controls on the normalized time grid. | Read when you want a single-frequency start. |
| [optimizer/guesses/GUESS_LIFECYCLE.md](optimizer/guesses/GUESS_LIFECYCLE.md) | topic | The steps a fresh or composite guess passes through, its metadata, and validation. | Read when a guess's metadata or validation needs explaining. |
| [optimizer/guesses/GUESS_LOCALIZED.md](optimizer/guesses/GUESS_LOCALIZED.md) | topic | Gaussian and sinc pulses concentrated around a center in normalized time. | Read when a start should be localized in time. |
| [optimizer/guesses/GUESS_METHODS.md](optimizer/guesses/GUESS_METHODS.md) | topic | Lookup table of the implemented guess families with a suggested workflow. | Read to pick a guess family quickly. |
| [optimizer/guesses/GUESS_RANDOM.md](optimizer/guesses/GUESS_RANDOM.md) | topic | random_guess: raw random controls with optional envelope, endpoint, and scaling. | Read when using an unstructured random start. |
| [optimizer/guesses/GUESS_RANDOM_FOURIER.md](optimizer/guesses/GUESS_RANDOM_FOURIER.md) | topic | random_fourier_guess: smooth random starts from random Fourier coefficients and phases. | Read when you want reproducible smooth random starts. |
| [optimizer/guesses/GUESS_RANDOM_SMOOTH.md](optimizer/guesses/GUESS_RANDOM_SMOOTH.md) | topic | random_smooth_guess: random controls smoothed with a Gaussian correlation kernel. | Read when tuning the correlation length of a random start. |
| [optimizer/guesses/GUESS_SIMPLE.md](optimizer/guesses/GUESS_SIMPLE.md) | topic | Constant and ramp controls. | Read when you want an explicit deterministic start. |
| [optimizer/jobs/JOBS.md](optimizer/jobs/JOBS.md) | topic | Bounded process runs, persistent coordinators, safe live pause and explicit physical recovery. | Read before running several optimization starts through the terminal or service. |
| [optimizer/mcp/MCP.md](optimizer/mcp/MCP.md) | topic | Local MCP tools for native optimizer operations, shared experiment ownership and bounded inspection. | Read when connecting an MCP client or composing optimizer operations with object references. |
| [optimizer/optimizers/ADAGRAD_RMSPROP.md](optimizer/optimizers/ADAGRAD_RMSPROP.md) | topic | AdaGrad and RMSProp: per-coordinate scaling by squared-gradient history, arguments, and state. | Read when calling adagrad or rmsprop. |
| [optimizer/optimizers/ADAM.md](optimizer/optimizers/ADAM.md) | topic | Adam and its variants: arguments, standard acceptance, schedules, and moment state. | Read when calling adam or continuing an Adam run. |
| [optimizer/optimizers/CMA_ES.md](optimizer/optimizers/CMA_ES.md) | topic | Derivative-free population search: variants, arguments, and sampling state. | Read when calling cma_es. |
| [optimizer/optimizers/LBFGS.md](optimizer/optimizers/LBFGS.md) | topic | L-BFGS polish: arguments, rejection shrink, and curvature-history state. | Read when calling lbfgs or switching to it from another method. |
| [optimizer/optimizers/LINE_SEARCH.md](optimizer/optimizers/LINE_SEARCH.md) | topic | Gradient descent with fixed, normalized, backtracking, and Armijo steps. | Read when calling line_search. |
| [optimizer/optimizers/MOMENTUM.md](optimizer/optimizers/MOMENTUM.md) | topic | Heavy-ball, Nesterov, restart, and clipped momentum: arguments and velocity state. | Read when calling momentum. |
| [optimizer/optimizers/NEWTON_CG.md](optimizer/optimizers/NEWTON_CG.md) | topic | Contract and use cases for matrix-free Newton-CG optimization. | An OLGS system supplies analytical Hessian-vector products. |
| [optimizer/optimizers/NONLINEAR_CG.md](optimizer/optimizers/NONLINEAR_CG.md) | topic | Nonlinear conjugate gradient variants for large control vectors: arguments and direction state. | Read when calling nonlinear_cg. |
| [optimizer/optimizers/NUMERICAL_CAPABILITIES.md](optimizer/optimizers/NUMERICAL_CAPABILITIES.md) | topic | Optional system capabilities that unlock projected, second-order, and proximal methods. | Read when a method needs a hook beyond the five required OLGS calls. |
| [optimizer/optimizers/OPTIMIZERS_HUB.md](optimizer/optimizers/OPTIMIZERS_HUB.md) | front-door | Entry to the optimizer methods: map, public entry points, and theory. | Start here when choosing or calling an optimizer. |
| [optimizer/optimizers/OPTIMIZER_ACCEPTANCE.md](optimizer/optimizers/OPTIMIZER_ACCEPTANCE.md) | topic | Per-method acceptance, custom accept functions, line-search specifics, stopping inputs, and best-so-far tracking. | Read when a method rejects steps or a stopping input needs choosing. |
| [optimizer/optimizers/OPTIMIZER_BOUNDARY.md](optimizer/optimizers/OPTIMIZER_BOUNDARY.md) | topic | What belongs in optimizer/optimizers versus systems, core, utils, guesses, and theory. | Read before adding an optimizer method. |
| [optimizer/optimizers/OPTIMIZER_COMMON.md](optimizer/optimizers/OPTIMIZER_COMMON.md) | topic | Shared plumbing in _common.py: shapes, clipping, warm starts, schedule resolution, and standard acceptance. | Read when writing an optimizer method or changing a shared helper. |
| [optimizer/optimizers/OPTIMIZER_CONTRACT.md](optimizer/optimizers/OPTIMIZER_CONTRACT.md) | topic | Common inputs, arguments, run recording, and the return object of every optimizer call. | Read when calling any optimizer or writing a new one. |
| [optimizer/optimizers/OPTIMIZER_LIFECYCLE.md](optimizer/optimizers/OPTIMIZER_LIFECYCLE.md) | topic | How a method's proposal function runs inside the shared engine, and the CMA-ES exception. | Read when writing a step function or tracing one iteration. |
| [optimizer/optimizers/OPTIMIZER_METHODS.md](optimizer/optimizers/OPTIMIZER_METHODS.md) | topic | Lookup table of the implemented optimizer methods with a suggested workflow. | Read to pick an optimizer quickly. |
| [optimizer/optimizers/OPTIMIZER_STATE_WARMSTART.md](optimizer/optimizers/OPTIMIZER_STATE_WARMSTART.md) | topic | RunState, WarmStartState, method state keys, and how state carries between chunks and methods. | Read when continuing a run or switching methods. |
| [optimizer/optimizers/PROJECTED_GRADIENT.md](optimizer/optimizers/PROJECTED_GRADIENT.md) | topic | Contract and use cases for equality- and bound-constrained projected gradient. | Optimizing an OLGS system with hard equality constraints or box bounds. |
| [optimizer/optimizers/PROXIMAL_GRADIENT.md](optimizer/optimizers/PROXIMAL_GRADIENT.md) | topic | Contract and use cases for composite-objective proximal-gradient optimization. | An OLGS system has a smooth term plus a system-owned proximal operator. |
| [optimizer/optimizers/TRUST_REGION.md](optimizer/optimizers/TRUST_REGION.md) | topic | Contract and use cases for Steihaug trust-region minimization. | A guarded second-order method is needed for an OLGS minimization problem. |
| [optimizer/schedules/SCHEDULES_HUB.md](optimizer/schedules/SCHEDULES_HUB.md) | front-door | Step-size schedules for optimizers and repair: constant, adaptive, exponential decay, warm-up, and cosine decay. | Read when giving a first-order method a step-size schedule. |
| [optimizer/shooting/SHOOTING_HUB.md](optimizer/shooting/SHOOTING_HUB.md) | front-door | Entry point for OLSS root solvers, continuation, seed campaigns, derivative checks, and replay. | Start here after an OLSS project system and initial shooting variables exist. |
| [optimizer/shooting/SHOOTING_METHODS.md](optimizer/shooting/SHOOTING_METHODS.md) | topic | Selection guide and semantic parameter reference for OLSS numerical tools. | Read when choosing or configuring a shooting solver, continuation, seed campaign, or replay. |
| [optimizer/system_olgs/CONTRACT.md](optimizer/system_olgs/CONTRACT.md) | topic | Normative OLGS request, response, invariant, and optional capability contract. | Implementing a system class or integrating an optimizer with OLGS. |
| [optimizer/system_olgs/DERIVATIVES.md](optimizer/system_olgs/DERIVATIVES.md) | topic | The required analytical gradient, optional derivative hooks, and the numerical-fallback policy. | Read when implementing derivatives for an OLGS system. |
| [optimizer/system_olgs/LIFECYCLE.md](optimizer/system_olgs/LIFECYCLE.md) | topic | How an OLGS system goes from controls to metrics and gradients, with one propagation per iteration. | Read when implementing evaluate and gradient for an OLGS system. |
| [optimizer/system_olgs/LOGGING_BOUNDARY.md](optimizer/system_olgs/LOGGING_BOUNDARY.md) | topic | Numerical system ownership and frontend command-history boundaries. | Read before adding persistence or interaction recording to a system. |
| [optimizer/system_olgs/PARAMS.md](optimizer/system_olgs/PARAMS.md) | topic | Primary and secondary parameter groups and dict-first construction of OLGS systems. | Read when defining a system's configuration. |
| [optimizer/system_olgs/SYSTEM_OLGS_HUB.md](optimizer/system_olgs/SYSTEM_OLGS_HUB.md) | front-door | Entry point for the Open Loop Gradient System contract and its supporting tools. | Implementing a differentiable optimization system or choosing OLGS capabilities. |
| [optimizer/system_olss/CONTRACT.md](optimizer/system_olss/CONTRACT.md) | topic | Required and optional project-system operations for the OLSS contract. | Read when implementing an OLSS system or validating a shooting-system call. |
| [optimizer/system_olss/DATA_MODEL.md](optimizer/system_olss/DATA_MODEL.md) | topic | OLSS system declaration, shooting-variable layout, and structured result objects. | Read when defining variable blocks, metric declarations, serialization, or result handling. |
| [optimizer/system_olss/SYSTEM_OLSS_HUB.md](optimizer/system_olss/SYSTEM_OLSS_HUB.md) | front-door | Entry point for implementing, validating, and using Open Loop Shooting Systems. | Start here when a project is formulated through shooting variables and boundary residuals. |
| [optimizer/system_olss/VERIFICATION.md](optimizer/system_olss/VERIFICATION.md) | topic | Independent OLSS verification, named checks, qualification, and scientific evidence boundaries. | Read when defining acceptance criteria or interpreting a shooting result. |
| [optimizer/utils/UTILS_HUB.md](optimizer/utils/UTILS_HUB.md) | front-door | Entry to diagnostics, derivative checks, geometry, repair, and the short-run decision loop. | Start here when inspecting or repairing a run. |
| [optimizer/utils/UTIL_BOUNDARY.md](optimizer/utils/UTIL_BOUNDARY.md) | topic | What belongs in utils versus optimizers, guesses, systems, core, and theory. | Read before adding a utility. |
| [optimizer/utils/UTIL_COMMON_API.md](optimizer/utils/UTIL_COMMON_API.md) | topic | Shared utility arguments: residual selection, finite-difference scale and method, directions, fallback, rank, and spectra. | Read when a utility argument needs explaining. |
| [optimizer/utils/UTIL_CONTRACT.md](optimizer/utils/UTIL_CONTRACT.md) | topic | Control, hook, metric, shape, return, and failure rules every utility follows. | Read when writing or validating a utility. |
| [optimizer/utils/UTIL_CONTROL_TRANSFORMS.md](optimizer/utils/UTIL_CONTROL_TRANSFORMS.md) | topic | Explicit grids, owned control candidates, waveform transformations and shooting recovery. | Read when exchanging controls between grids or composing a new starting waveform. |
| [optimizer/utils/UTIL_DERIVATIVE_CHECKS.md](optimizer/utils/UTIL_DERIVATIVE_CHECKS.md) | topic | Gradient and Jacobian checks against finite differences. | Read before trusting an optimizer on a new system. |
| [optimizer/utils/UTIL_DIAGNOSTICS.md](optimizer/utils/UTIL_DIAGNOSTICS.md) | topic | Metric and diagnostic reports, and short-run diagnosis with its labels and thresholds. | Read when deciding what to do after a short run. |
| [optimizer/utils/UTIL_LIFECYCLE.md](optimizer/utils/UTIL_LIFECYCLE.md) | topic | The debug order: metrics, gradient and Jacobian trust, geometry, projection or repair, and shape. | Read when a run misbehaves and the cause is unclear. |
| [optimizer/utils/UTIL_METHODS.md](optimizer/utils/UTIL_METHODS.md) | topic | Index of the public utilities by category. | Read to find a utility quickly. |
| [optimizer/utils/UTIL_REPAIR.md](optimizer/utils/UTIL_REPAIR.md) | topic | Newton and damped repair that reduces named residuals, with line search and RepairResult. | Read when controls must move back onto residual constraints. |
| [optimizer/utils/UTIL_RESIDUAL_GEOMETRY.md](optimizer/utils/UTIL_RESIDUAL_GEOMETRY.md) | topic | Geometry probe, nullspace basis, and gradient projection for residual constraints. | Read when examining or projecting onto a constraint surface. |
| [optimizer/utils/UTIL_SPECTRUM_SMOOTHNESS.md](optimizer/utils/UTIL_SPECTRUM_SMOOTHNESS.md) | topic | Control spectra and smoothness measures that leave controls unchanged. | Read when checking how smooth or band-limited a control is. |
| [optimizer/visualization/VISUALIZATION_HUB.md](optimizer/visualization/VISUALIZATION_HUB.md) | front-door | System-owned plot data and reusable renderers for Bloch, projection, and control panels. | Read when plotting a system's data. |
| [test_contained/ho_purification/PROBLEM_CONTEXT.md](test_contained/ho_purification/PROBLEM_CONTEXT.md) | topic | Contained harmonic-oscillator purification system and historical tracker-trial provenance. | Read when running or changing the HO purification trial. |
| [test_contained/second_order_pmp/PROBLEM_CONTEXT.md](test_contained/second_order_pmp/PROBLEM_CONTEXT.md) | topic | Contained second-order qubit-transfer system and historical tracker-campaign provenance. | Read when running or changing the second-order PMP trial. |
| [tests/TESTING_HUB.md](tests/TESTING_HUB.md) | front-door | The test suite, its fixtures, contained scenario evidence, and contained numerical systems. | Read when running tests or locating test evidence. |

## Optional

| path | role | summary | read when |
|---|---|---|---|
| [API_REFERENCE.md](API_REFERENCE.md) | generated | Generated exact signatures and catalog semantics for every public OPTIMIZER tool. | Looking up a callable signature, default, requirement, return, or runtime identifier. |
| [optimizer/cli/TERMINAL.md](optimizer/cli/TERMINAL.md) | deep | Structured terminal syntax, launch modes, retained names, display and foreground lifetime. | Read when running optimizer interactively or writing a command script. |
| [optimizer/core/CORE_ENGINE.md](optimizer/core/CORE_ENGINE.md) | deep | run_chunk arguments, start-state precedence and compact numerical results. | Read when calling run_chunk directly or interpreting native chunk state. |
| [optimizer/history/HISTORY.md](optimizer/history/HISTORY.md) | deep | Frontend command history, durable statement commits, explicit migration and bounded replay. | Read before inspecting or replaying terminal history or integrating an MCP recorder. |
| [optimizer/operations/SERVICE.md](optimizer/operations/SERVICE.md) | deep | Foreground operation requests, isolated sessions, activation, continuation and retained-data limits. | Read when integrating the operation service or preparing a terminal/MCP adapter. |
| [optimizer/persistence/PERSISTENCE.md](optimizer/persistence/PERSISTENCE.md) | deep | Independent experiments, typed checkpoints, exact continuation, rollback and bounded disk storage. | Read before saving, reopening, restoring or pruning an interactive experiment. |
| [test_contained/olgs_convergence/REPORT.md](test_contained/olgs_convergence/REPORT.md) | deep | OLGS validation design, scenario matrix, method coverage, and the 2026-10-12 outcome. | Read when checking OLGS method structure or rerunning the matrix. |
| [test_contained/olss_scenarios/REPORT.md](test_contained/olss_scenarios/REPORT.md) | deep | OLSS issue record: resolution status, original findings, and campaign evidence. | Read when checking an OLSS issue or rerunning the scenario campaign. |
| [test_contained/pi_pulse_samples/REPORT.md](test_contained/pi_pulse_samples/REPORT.md) | deep | Known-answer pi-pulse tests for gradient and shooting, soft and hard, through the Python API, the operation service and the terminal. | Read when checking the interfaces on small problems with analytical answers, or rerunning the sample suite. |
| [test_contained/review_2026_10_12/REPORT.md](test_contained/review_2026_10_12/REPORT.md) | deep | 2026-10-12 review: two rounds of fixes with tests, issue statuses, the shared-core inspection, and shared-core phase 1. | Read when checking the 2026-10-12 fixes and their evidence, or before designing or implementing the shared core. |

---

[⌂ Home](README.md)
