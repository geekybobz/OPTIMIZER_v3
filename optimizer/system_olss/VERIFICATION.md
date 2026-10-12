---
role: topic
parent: SYSTEM_OLSS_HUB.md
summary: "Independent OLSS verification, named checks, qualification, and scientific evidence boundaries."
read_when: "Read when defining acceptance criteria or interpreting a shooting result."
status: active
reviewed: 2026-10-12
---
# OLSS Verification

Solver termination, independent verification, and scientific qualification are
separate facts.

## Qualification pipeline

```text
working residual solve
    -> solver residual meets working tolerance
    -> verify with distinct options or a distinct declared route
    -> verification residual meets verification tolerance
    -> every named VerificationCheck passes
    -> qualified root
```

`qualified` is false if any stage is absent or fails.

## Verification tolerance

Verification re-evaluates at distinct accuracy, so its residual also contains
the working discretization error. The tolerance is resolved in this order:

1. `verification_tolerance` passed to the solver or continuation call;
2. the system's declared `verification.default_tolerance`, set with
   `basic_shooting_system_spec(..., verification_tolerance=...)`;
3. the working solver `tolerance`.

`ShootingResult` and `ShootingContinuationResult` record the value used in
`verification_tolerance` and its origin in `verification_tolerance_source`
(`explicit`, `system_default`, or `working_tolerance`). A system whose working
accuracy is coarser than its verification accuracy should declare its default
tolerance once rather than leave every caller to pass one.

## Independence

Verification is independent only when explicit `verification_options` or the
system's declared default verification options are non-empty and differ from
the working options. The result records:

| Field | Meaning |
|---|---|
| `independent` | Whether the verification is distinct |
| `independence_source` | `explicit_options`, `system_default`, or the reason independence is absent |
| `numerical_success` | Whether the verification computation completed |
| `message` | Project or runtime explanation |

Reusing working options may still return useful metrics, but it cannot qualify
the candidate.

## Named checks

Each `VerificationCheck` contains:

```text
name, passed, value, tolerance, message
```

Names and physics remain project-owned. Typical checks include tighter replay,
normalization, conserved quantities, terminal fidelity, transversality,
stationarity, or physical admissibility. Every supplied check must pass.

## Jacobian verification

`verify_shooting_jacobian` compares the system Jacobian with a central finite
difference calculation. If the declared Jacobian is already numerical, it is
compared at two separated steps (`comparison_steps`). One of them may coincide
with the project's own step and agree to roundoff; the other cannot, and the
check passes only when both comparisons agree. The diagnostic reports strategy,
steps, the error at each step (`step_max_abs_errors`), absolute and relative
errors, tolerances, and both matrices.

This diagnostic checks local derivative consistency. It does not replace the
declared Jacobian during a solve.

## Root comparison and ranking

Multistart first keeps qualified roots, then removes duplicates using either:

1. the project `roots_equivalent` hook for a known gauge or symmetry; or
2. the scale-aware numerical default.

Only unique qualified roots are ranked. A declared selection metric orders the
available set; it does not prove global optimality. Ranking uses the verified
value of the selection metric when verification reports it, otherwise the
working value; `MultiStartResult.selection_source` records `verification`,
`working`, or `mixed`.

## Evidence meanings

| Evidence | Establishes | Does not establish |
|---|---|---|
| Solver convergence | Working residual criterion was met | Independent correctness |
| Verification | Distinct numerical check completed | Local minimum |
| Qualification | Required verification gates passed | Global optimum |
| Selection ranking | Best declared metric among supplied qualified roots | Exhaustive root search |
| Curvature or stationarity audit | Local evidence under its assumptions | Global certificate |

## Connections

- Related: [OLSS System Hub](SYSTEM_OLSS_HUB.md)
- Prerequisite: [OLSS Contract](CONTRACT.md)
- Related: [OLSS Data Model](DATA_MODEL.md)
- Related: [Shooting Methods](../shooting/SHOOTING_METHODS.md)
- Related: [OLSS scenario evidence](../../test_contained/olss_scenarios/REPORT.md)

---

[⌂ Home](SYSTEM_OLSS_HUB.md)
