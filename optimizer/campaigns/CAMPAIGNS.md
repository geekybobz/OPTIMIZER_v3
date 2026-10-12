---
role: front-door
parent: ../../README.md
summary: "Multi-stage campaigns that coordinate existing optimizer calls into evidence-preserving modes."
read_when: "Read when running scout, comparison, refinement, challenge, or validation campaigns."
status: active
reviewed: 2026-10-12
---
# Campaign Modes

`optimizer.campaigns` coordinates existing optimizer calls into an explicit,
evidence-preserving campaign. It does not own a system objective, invent metrics, or
run hidden diagnostics.

```python
import optimizer as opt

system = make_system()
campaign = opt.campaigns.create(
    system,
    "runs/example-campaign",
    system_factory=make_system,
)

guesses = opt.campaigns.random_smooth_guesses(
    system, 10, amplitude=0.1, seed=7,
)
short = opt.campaigns.MethodSpec("adam", {"maxiter": 20, "step_size": 0.05})
scout = campaign.scout(guesses, method=short)
```

The required `amplitude` above is intentional: a generic campaign library cannot
choose a physically meaningful control scale for a research system.

## Details

## The six modes

1. `scout(guesses, method)` runs one declared method from many supplied guesses.
2. `portfolio(sources, methods)` compares methods from exactly identical source
   controls. It is a fair comparison, not a warm-start chain.
3. `refine(source, method)` performs one longer solve from a selected candidate.
   Same-method optimizer memory may be retained; cross-method private memory is not.
4. `challenge(source, ...)` retains the source pulse and adds controlled Fourier
   compression and smooth local perturbations. It is local-basin sensitivity evidence,
   never a global-optimality claim.
5. `validate(source)` performs a fresh evaluation and, by default, one explicit
   gradient measurement. The resulting counters are saved under `diagnostics/`.
6. `continue_parameter(source, updates, method)` updates the system through its
   `with_secondary` hook. It requires exactly the same control layout.

## Parallel execution

`scout`, `portfolio`, and `challenge` accept `ParallelConfig`. Multi-worker
execution requires `system_factory`, because every worker must receive an isolated
system instance, evaluator cache, tracker journal, and stage directory.

```python
from optimizer.utils.parallel import ParallelConfig

scout = campaign.scout(
    guesses,
    method=short,
    parallel=ParallelConfig(backend="thread", workers=4),
)
```

Use a process backend only when the system factory and all objects it closes over are
picklable in the target environment. Never share a mutable system instance or a
tracker writer between candidates.

## Durable artifacts

```text
campaign_root/
  campaign_receipt.json
  campaign_timeline.jsonl
  selected.json
  stages/<stage-id>/       # ordinary tracker receipts, timeline, checkpoints, terminal
  diagnostics/<source>.json
```

`campaign.report()` only reads these artifacts. It never calls `evaluate`,
`gradient`, propagation, residual, or repair hooks.

---

[⌂ Home](../../README.md)
