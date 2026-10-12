---
role: deep
parent: ../../tests/TESTING_HUB.md
summary: "Shared-core verification: exact OLGS parity, intended OLSS scaling change, CPU reserve and known solver limits."
read_when: "Read when reviewing the 2026-10-13 numerical consolidation and its baseline comparisons."
status: active
reviewed: 2026-10-13
---
# Shared-core verification — 2026-10-13

The shared-core implementation is verified in the development checkout against
baseline commit `8d21ad7a152bdbb4ebaa48aae208e3c052c871af`. The baseline was
extracted with `git archive`; it was not installed. These campaign comparisons were captured before release promotion. The shared
core is included in version 4.0.6; exact committed bytes and installed-wheel
verification belong to the [release artifacts](../../RELEASE_EVIDENCE.md).

The [implementation guide](../../docs/SHARED_NUMERICAL_CORE.md) explains the
coordinate equations, component ownership and remaining numerical work.
[verification.json](verification.json) records a digest of the campaign-stage Python
source before the 4.0.6 version bump, comparison results and artifact digests.
That digest is a campaign receipt, not the final release checksum.

## Results

| Check | Result |
|---|---|
| Full source suite | 491 tests and 450 subtests passed |
| Final repair/core/control checks | 45 tests and 24 subtests passed |
| OLGS matrix | 425/425 rows exactly match baseline after excluding timings |
| System-call accounting | Every matrix row has identical attempted and successful call counts |
| Additional OLGS evidence | Six workflows, eight physical studies and five large-vector cases match numerically |
| Serial/thread/process campaigns | All succeeded; summary size changes because results now include bounded records |
| OLSS campaign | 21 pass, 1 gap, 1 fail; baseline was 20 pass, 2 gaps, 1 fail |
| Review scripts | Qubit loop, structural probes and Duffing loop completed; strict JSON checks passed |
| Documentation | API generator, inventory freshness, Markdown properties/links and whitespace checks passed |

All observed OLSS values match after excluding timing text, except the intended
G6 scaling correction. The underdetermined line formerly returned normalized
coordinates approximately (0,2); it now returns (1,1), the minimum-norm step in
the declared numerical coordinates. G6 changes from gap to pass. The small
campaign snapshots are retained as [baseline](olss-baseline.json) and
[source](olss-source.json).

## Remaining campaign findings

**A5 still fails its accuracy target in both versions.** The cart direct solver
reduces energy from about 12.2283 to 12.025356 within 200 iterations, while the
known optimum is 12. This is the existing limitation of projected-gradient
direct shooting; the shared core does not implement SQP. Feasible roots and
budget stops are not optimality certificates.

**G3 remains a fixture gap in both versions.** Both single and multiple shooting
solve the example using an exact transition matrix. That propagation does not
test the integration-driven instability the scenario is meant to compare.
An independently accepted finite-precision integration fixture is still needed.

Sparse/operator Jacobians, stronger constrained direct shooting, adaptive LM,
exact equality-plus-bound projection and additional accepted physical fixtures
remain outside this consolidation. Their consequences and proposed remedies
are explained in the implementation guide.

## Resource use and timing

The machine reports ten logical CPUs. Executors cap worker counts at seven;
managed jobs require workers times native threads to fit the same seven-slot
budget. The final campaigns ran sequentially with four workers and six explicit
BLAS/OpenMP thread environment settings set to one. Full tests and review
scripts ran separately from campaigns. Earlier overlapping exploratory runs
were stopped after the user's CPU-limit instruction; their timings are not
used to claim performance parity.

[Alternating serial timing samples](timing.json) compare baseline/source twice
on 300,000 control variables with one native thread. Typical gradient-method
iterations remain a few milliseconds and the Newton-CG case about 0.2 seconds.
Timing varies between samples; these measurements do not establish a speedup
or a precise overhead bound. Numerical and call-count parity are the strong
claims supported by the campaign.

Worker admission is per executor/coordinator, not a machine-wide allocator.
Unrelated applications and independently launched executors can add CPU load;
native libraries must honor their thread limits.

## Reproduction

Use the Optimizer Python environment, the checkout on PYTHONPATH, bytecode
disabled, and each of OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS,
VECLIB_MAXIMUM_THREADS, NUMEXPR_NUM_THREADS and BLIS_NUM_THREADS set to 1.
Run each item to completion before starting the next:

```text
python -B -m pytest -q tests
python -B test_contained/olgs_convergence/run_matrix.py --workers 4 --json OLGS.json
python -B test_contained/olss_scenarios/run_scenarios.py --workers 4 --json OLSS.json
python -B test_contained/review_2026_10_12/qubit_olgs/run_loop.py
python -B test_contained/review_2026_10_12/qubit_olgs/probes.py
python -B test_contained/review_2026_10_12/duffing_olss/run_loop.py
```

Campaign runners report known gaps in JSON and may exit successfully even when
a scenario is marked fail. Inspect their rows and compare to the baseline;
exit code alone does not establish numerical acceptance.

---

[⌂ Home](../../README.md)
