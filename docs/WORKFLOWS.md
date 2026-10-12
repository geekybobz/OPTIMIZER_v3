---
role: topic
parent: ../README.md
summary: "Find the supported routes for independent experiments, control exchange, jobs and verification."
read_when: "Read after the first run to choose your next workflow."
status: active
reviewed: 2026-10-13
---
# Everyday workflows

Start with [Getting started](GETTING_STARTED.md). Each topic below owns its exact
commands, validation rules and limitations. Use discovery (`opt.help();` and
`opt.info("adam");`) to inspect the installed API rather than guessing signatures.
Direct Python uses the same numerical methods without automatic command history.

## Details

| Goal | Supported route and authority |
|---|---|
| Compare several initial guesses | Create independent named experiments; use [managed jobs](../optimizer/jobs/JOBS.md) for bounded workers and explicit result selection |
| Preserve one run while exploring another | Save a checkpoint and fork the [experiment](../optimizer/persistence/PERSISTENCE.md); initialization never silently clears an existing experiment |
| Undo an iteration or return to a baseline | `opt.session.undo();`, `redo()` or `restore("baseline")`; all remain in [history](../optimizer/history/HISTORY.md) |
| Try another method | Select physical controls/variables or pass compatible method state; see [warm starts](../optimizer/optimizers/OPTIMIZER_STATE_WARMSTART.md) |
| Resample, smooth, filter, scale or normalize controls | Use [control transforms](../optimizer/utils/UTIL_CONTROL_TRANSFORMS.md); inspect the returned candidate before explicit activation |
| Exchange shooting and gradient controls | Preserve channel names, time/grid representation and units; waveform recovery does not reconstruct missing shooting costates |
| Pause and resume long work | [Jobs](../optimizer/jobs/JOBS.md): exact resume needs the same live worker; saved boundary snapshots permit explicit physical restart |
| Keep work running after terminal exit | Start a persistent experiment coordinator and detach; ordinary foreground job ownership ends with the client |
| Connect an assistant | Configure the [MCP server](../optimizer/mcp/MCP.md); its typed requests share CLI state/history through one writer |
| Bound disk use | Check checkpoint quotas, pin important snapshots and explicitly prune payloads; history remains retained |
| Verify a result | Check derivatives/layouts and physical metrics, then apply family-specific [OLSS qualification](../optimizer/system_olss/VERIFICATION.md) or OLGS diagnostics |

### Short continuation and rollback

Inside a loaded gradient experiment:

```python
base = opt.session.save(name="before-trial");
a = opt.adam(maxiter=5, step_size=0.01);
a.metrics;
opt.session.undo();
opt.session.redo();
opt.session.restore(base);
b = opt.lbfgs(maxiter=5, activate=False);
b.metrics;
opt.session.use(b.controls);
```

Keeping `b` as a candidate does not overwrite the active head. Activating physical
controls deliberately starts new method memory. Same-method implicit continuation
can preserve compatible numerical state; checkpoints validate source, environment
and state compatibility before claiming exact restoration.

### Resource limits

Worker and thread limits apply per experiment coordinator. Memory reservations
and RSS sampling are safeguards, not an OS-enforced memory cap. Coordinate budgets
across multiple experiments yourself; creating several coordinators multiplies
their allowed resources. Begin with small worker counts before using all cores.
See the [jobs reference](../optimizer/jobs/JOBS.md) for default limits and pause
support by method.

The real article benchmark campaign remains a separate reviewed task. Library
and installation tests do not establish those scientific comparison results.

---

[← Previous](GETTING_STARTED.md) · [⌂ Home](../README.md)
