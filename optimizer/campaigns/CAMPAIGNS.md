---
role: front-door
parent: ../../README.md
summary: In-memory multi-stage campaigns, owned candidates and explicit numerical validation.
read_when: Read when comparing initial guesses, methods or refinement candidates in Python.
status: active
reviewed: 2026-10-12
---

# Campaign modes

`optimizer.campaigns` coordinates native optimizer calls and retains typed
candidates in RAM. It creates no files, journals or automatic history. `root`
is an optional compatibility label; `campaigns.open` and durable campaign
artifacts are removed. Ordinary Python owns the lifetime of these objects.

```python
campaign = opt.campaigns.create(system, system_factory=make_system)
guesses = opt.campaigns.random_smooth_guesses(system, 10, amplitude=0.1, seed=7)
short = opt.campaigns.MethodSpec("adam", {"maxiter":20, "step_size":0.05})
scout = campaign.scout(guesses, method=short)
```

Selection uses the system-declared objective and direction. `report()` reads
retained candidates and never performs hidden evaluation. `load_candidate()`
returns owned numerical state. Lifetime and process memory remain caller-owned;
this coordinator is not the future bounded CLI job scheduler.

## Details

## Six modes

1. `scout` compares supplied guesses with one declared method.
2. `portfolio` compares methods from identical physical controls.
3. `refine` runs longer from a selected candidate, preserving compatible memory.
4. `challenge` makes explicit Fourier compression/local perturbations; these
   are local sensitivity checks, not global-optimality claims.
5. `validate` evaluates the candidate and optionally measures its gradient;
   the returned ValidationResult carries metrics and numerical counters.
6. `continue_parameter` uses `with_secondary` and requires the same layout.

`CandidateRef.result` supplies an OptimizerResult. Export that result or its
controls explicitly for checkpoint-supported workflows. There is no durable
codec for a live Campaign, and no terminal campaign lifecycle adapter yet.

## Parallel execution

Existing `ParallelConfig` supports native map-style Python workflows.
Multi-worker campaigns require `system_factory` so every worker constructs an
independent system/cache. A process factory and its captured objects must be
picklable. Results are returned to the coordinator; workers share no journal.
These APIs do not enforce an RSS budget, CPU reservations or cooperative pause;
managed interactive jobs remain a separately reviewed phase.

## Connections

- Parent: [V4 entry](../../README.md)
- Related: [Optimizer contract](../optimizers/OPTIMIZER_CONTRACT.md)
- Related: [Checkpoint persistence](../persistence/PERSISTENCE.md)

---

[⌂ Home](../../README.md)
