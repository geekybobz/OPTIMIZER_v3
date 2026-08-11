# Optimizer Log Portal: plan

## Status

Approved design plan. This document defines intended scope and sequencing; it
does not claim that the portal, CLI, discovery, retention, or verification
features are implemented.

## 1. Aim

Build `optimizer_log` as a local analysis-and-management portal for generic
optimization runs. It must make a run understandable without hard-coding a
physics problem, metric name, or optimizer method.

For one selected run, the user should be able to answer:

1. Which numerical values were saved and where did they come from?
2. How did selected metrics, optimizer costs, and technical quantities evolve?
3. Which method and phase were active over time?
4. Which steps were accepted, rejected, failed, analysed, or changed by an
   intervention?
5. Which checkpoint and terminal evidence support the selected result?
6. Which evidence is absent, rather than silently inferred?
7. Later, which verification/test results support or challenge the run?

The portal has two separate responsibilities:

~~~text
analysis:    explain existing numerical evidence
management:  safely manage local portal state and selected old run bundles
~~~

The analysis path is passive. It must not call a system, optimizer, evaluator,
gradient, residual, Jacobian, Hessian, HVP, simulation, or diagnostic hook.

\[
N_{\mathrm{evaluate}}^{\mathrm{portal}}
=
N_{\mathrm{gradient}}^{\mathrm{portal}}
=
N_{\mathrm{diagnostic}}^{\mathrm{portal}}
=0.
\]

Management operations are explicit, local, reviewed actions. They must never
rewrite an append-only scientific timeline.

## 2. Scope and exclusions

### In scope

- local CLI and local browser portal;
- discovery of user-selected run roots;
- adaptive rendering of saved scalar system metrics, counters, and bounded
  technical values;
- user-selected metric plots;
- method, phase, acceptance, intervention, checkpoint, and terminal history;
- incremental live following from durable files;
- local viewer preferences, cache clearing, hide/unhide, and later safe
  archive/trash management;
- later static verification/test scanning and controlled replay workflows.

### Explicitly out of scope for the first implementation

- remote hosting, accounts, external telemetry, databases, WebSockets, or a
  third-party tracking platform;
- automatic optimizer method changes or numerical recovery actions;
- UI-triggered expensive diagnostics;
- global-optimality or convergence claims inferred from curves;
- control-array plots from JSONL;
- all-metrics-at-once dashboards;
- campaign/multi-start ranking before campaign identity and comparability rules;
- direct deletion of raw timeline content.

## 3. Existing source-of-truth model

The portal extends the current tracker rather than creating a competing
history system.

~~~text
<run>/
  run_receipt.json       immutable run identity and configuration
  timeline.jsonl         append-only causal event stream
  live.json              replaceable latest display snapshot
  terminal.json          direct terminal-result index
  writer.json            writer identity, heartbeat, and closed state
  .writer.lock           present only while the writer owns the run
  checkpoints/*.npz      selected control and resume artifacts
~~~

The existing standalone reader provides:

~~~python
optimizer_log.read_receipt(run_dir)
optimizer_log.read_timeline(run_dir)
optimizer_log.read_live(run_dir)
optimizer_log.read_terminal(run_dir)
optimizer_log.read_writer(run_dir)
optimizer_log.inspect_writer(run_dir)
optimizer_log.follow_timeline(run_dir, offset=...)
optimizer_log.project_timeline(records, fields=...)
~~~

The current writer/event path is:

~~~mermaid
flowchart LR
    S["System evaluate()<br/>existing scalar metrics"] --> E["Optimizer engine"]
    G["Engine facts<br/>technical values, counters,<br/>method, phase, acceptance"] --> E
    E --> O["RunObserver event stream"]
    O --> J["JournalObserver"]
    J --> F["Durable tracker files"]
    F --> P["RunProjection"]
    P --> C["optimizer_log CLI"]
    P --> U["Local portal"]
~~~

### Existing fact categories

| Category | Current source | Portal use |
|---|---|---|
| System metric | `ScalarSnapshot.metrics` | Metric plot |
| Metric origin | `ScalarSnapshot.metric_sources` | Source label |
| Cost counter | `ScalarSnapshot.counters` | Counter/cost plot |
| Optimizer technical evidence | `IterationRecord.proposal` | Optional technical plot/detail |
| Step result | `accepted`, `reason` | Acceptance strip/event table |
| Method/phase | Event fields | Method history strip |
| Analysis/intervention | Dedicated records | Causal event detail |
| Checkpoint | `CheckpointReference` | Artifact link |
| Terminal result | `TerminalRecord` | Final evidence |

The portal never calls a system to fill a missing metric. If it is absent from
the saved record, it remains absent in the view.

## 4. Required contract gate: metric meaning

`system.tracking_spec()` remains optional. A system without it must still
work with raw metric keys.

When it exists, the resolved static metadata is copied into
`run_receipt.json` once, when the run begins:

~~~json
{
  "metric_id": "metric.fidelity",
  "label": "target fidelity",
  "unit": "",
  "direction": "maximize",
  "source": "system.evaluate",
  "role": "diagnostic"
}
~~~

The receipt must also carry an explicit result-selection rule:

~~~json
{
  "metric_id": "J",
  "direction": "minimize",
  "rule": "best_finite_value"
}
~~~

Without descriptor metadata, the portal uses:

~~~text
label      = raw field key
unit       = unspecified
direction  = unspecified
source     = event-provided source or unspecified
~~~

It may plot a numeric field, but must not label it as improved, optimal, or
best. A direction-dependent selected value is only shown when the receipt
explicitly declares it:

\[
q_{\mathrm{best},k} =
\begin{cases}
\min_{i \leq k} q_i, & \text{declared minimize},\\
\max_{i \leq k} q_i, & \text{declared maximize}.
\end{cases}
\]

The writer contract distinguishes `live`, `closed`, `abandoned` on the local
host, and `remote_or_unknown`. A later portal must preserve that uncertainty
instead of guessing from elapsed time alone.

## 5. RunProjection: adaptive data model

`RunProjection` is the local adapter between tracker files and both user
interfaces. It dynamically discovers fields and provides a stable view model.

~~~text
RunProjection
├── receipt
├── run_status
├── integrity_state
├── metric_catalog
├── metric_series
├── method_phase_segments
├── acceptance_history
├── cost_history
├── causal_events
├── checkpoint_references
└── verification_links
~~~

### Dynamic metric discovery

For every causal event, use these snapshots for the ordinary run trajectory:

~~~text
run_start      -> initial
iteration      -> after
intervention   -> after
run_end        -> final
~~~

Keep `before`, `trial`, and `best_after` in the expandable event detail,
not as default charts. This avoids visually confusing a proposed/trial state
with the committed optimizer state.

Dynamic field identifiers:

~~~text
metric.<system key>
counter.<counter key>
technical.<numeric proposal path>
~~~

Examples:

~~~text
metric.J
metric.energy
metric.constraint_residual
counter.evaluation
counter.gradient
technical.gradient_norm
technical.proposal.step_size
technical.acceptance.threshold
~~~

Each discovered field has:

~~~text
id, category, raw_key, label, unit, source, direction,
sample_count, first_iteration, last_iteration, plot_eligibility,
availability
~~~

Rules:

1. A numeric field with at least two observations is plot-eligible.
2. A single observation appears in event details only.
3. Missing values create gaps; they are never changed to zero or interpolated.
4. Strings and booleans remain event metadata.
5. Arrays and control histories remain checkpoint/artifact references.
6. Technical fields stay visibly classified as technical evidence.
7. The reader preserves source and event ID with every plotted point.

For a scalar (q), the plot data are:

\[
\mathcal P_q = \{(k, q_k, t_k, e_k)\},
\]

where (k) is outer iteration, (t_k) elapsed time, and (e_k) the causal
event identifier.

## 6. User selections and local preferences

The user chooses which available metrics are visible. The metric browser is
fully adaptive:

~~~text
SYSTEM METRICS
[x] objective
[x] fidelity
[ ] energy

OPTIMIZER TECHNICAL
[x] gradient_norm
[ ] proposal.step_size

COUNTERS
[x] evaluation
[ ] gradient
~~~

New fields discovered in later events appear under `New metrics`. They are
available immediately but are not automatically plotted, avoiding a crowded
dashboard.

The user may pin fields for future visits. These are local display preferences,
not scientific records:

~~~json
{
  "schema_version": "optimizer_log.viewer/v1",
  "pinned_metrics": [
    "metric.J",
    "technical.gradient_norm",
    "counter.evaluation"
  ],
  "smoothing": 0.0,
  "hidden_runs": []
}
~~~

## 7. Minimal portal design

The first portal must be smaller and clearer than a general TensorBoard clone.

~~~text
┌ Run root · selected run · method · phase · last update ┐
│ [Refresh] [Follow live] [Storage]                       │
├───────────────────────┬─────────────────────────────────┤
│ Runs                  │ Primary selected metric plot     │
│                        ├─────────────────────────────────┤
│ Available metrics      │ Optional second/third plot       │
│ [x] objective          ├─────────────────────────────────┤
│ [x] gradient norm      │ acceptance and method strip      │
│ [ ] energy             ├─────────────────────────────────┤
│ [x] evaluations        │ causal event table/detail        │
└────────────────────────┴─────────────────────────────────┘
~~~

### Default plot policy

1. Plot the declared objective/selection metric first.
2. Plot up to two user-pinned additional fields.
3. List all other discovered fields in the metric browser.
4. Do not overlay incompatible units by default.
5. Do not display more than three charts initially.

### Plot types

| Data type | Initial display |
|---|---|
| System scalar | Raw point + line plot |
| Counter | Cumulative line plot, zero lower bound |
| Gradient norm | Technical line plot when saved |
| Step size | Technical line plot when saved |
| Acceptance | Compact accepted/rejected strip |
| Method/phase | Horizontal method segment strip |
| Intervention/checkpoint/terminal | Causal marker |
| Arrays/artifacts | Linked detail only |

Optional display smoothing is visual only:

\[
\tilde q_0=q_0,
\qquad
\tilde q_k=(1-s)q_k+s\tilde q_{k-1}.
\]

Raw points remain visible. Smoothing changes neither the run, evidence,
selection result, nor numerical claim.

The portal may display progress evidence:

\[
\Delta q_k=q_k-q_{k-1},
\qquad
r_k=\frac{|q_k-q_{k-1}|}{\max(|q_{k-1}|,\varepsilon)}.
\]

It must not turn these plots into a convergence or optimality theorem.

## 8. Controls

### Analysis controls

| Control | Behaviour |
|---|---|
| Run selector | Choose a discovered run |
| Metric checkbox | Show/hide a dynamic field |
| Pin metric | Use field by default next visit |
| Search | Filter keys, labels, category, and source |
| Refresh | Re-read saved files now |
| Follow live | Incrementally read new records |
| Smoothing | Display-only transform |
| Event row | Open exact before/trial/after/action evidence |
| Checkpoint row | Show artifact metadata/reference |

### Management controls

| Control | Behaviour |
|---|---|
| Clear viewer cache | Remove UI cache and preferences only |
| Hide/unhide run | Reversible list operation |
| Storage plan | Produce a dry-run archive/delete manifest |
| Archive run | Later: move one completed run bundle to recoverable trash |
| Apply plan | Later: explicit confirmation required |
| Purge trash | Later: permanent removal after review |

`timeline.jsonl` is not directly editable from the portal. Clearing history
must always name the exact target: local UI cache, hidden-run list, archived
bundle, or reviewed trash purge.

## 9. CLI and local service

The intended entry point is:

~~~text
optimizer_log
~~~

Run from a chosen working folder.

| Command | Intended result |
|---|---|
| `optimizer_log` | Discover candidate run roots and select/launch |
| `optimizer_log --run <path>` | Open one explicit run |
| `optimizer_log --port 3000` | Start local portal |
| `optimizer_log --list` | List discovered runs |
| `optimizer_log --follow <id>` | Compact live terminal summary |
| `optimizer_log storage plan <id>` | Preview storage action |
| `optimizer_log storage apply <plan>` | Later, execute approved plan |
| `optimizer_log verify <id>` | Later, read/run verification scan |

Discovery scans only the explicit/current folder and declared descendants. If
more than one candidate root is found, the CLI presents a numbered selection;
it never guesses which old/new/multi-start result the user intended.

The local service needs only:

~~~text
GET  /api/runs
GET  /api/runs/{id}/view
GET  /api/runs/{id}/tail?after_sequence=...
POST /api/storage/plan
POST /api/storage/apply
~~~

The first three operations are read-only. Storage operations are separate,
local, and confirmation-gated.

## 10. Live following and integrity

For an active run:

1. Read the receipt once.
2. Read the initial timeline.
3. Record its byte offset and last complete sequence.
4. On refresh, read only appended bytes.
5. Read `live.json` as the latest display convenience.
6. Refresh only changed series and events.

Integrity behaviour:

| Condition | Portal result |
|---|---|
| Incomplete final JSONL line | Wait/retry; do not show partial event |
| Midstream corruption | Integrity warning; do not invent a repaired history |
| No new records | Show last update only |
| No metric descriptor | Plot raw key; do not infer semantics |
| Missing metric at one event | Plot gap |
| Artifact unavailable | Explicit unavailable state |

## 11. Safe storage/retention design

Current tracker storage has writer ownership/liveness but no retention API.
Therefore:

### First safe operations

- clear local portal cache;
- reset pins/filters;
- hide/unhide runs from the portal;
- generate a storage dry-run only.

### Later guarded operations

1. Permit only completed/non-active run bundles.
2. Create a manifest of exact files and sizes.
3. Move the complete bundle to recoverable local trash.
4. Preserve an audit receipt for the archive action.
5. Permit permanent purge only from trash, after explicit confirmation.

The first archive/delete feature should operate on a whole completed run
bundle, not individual checkpoints. Fine-grained artifact deletion waits until
a reference graph can prove that selected terminal evidence remains readable.

## 12. Verification and test-scanning module

Verification is a later module with its own records. It must not be mixed into
the raw optimization timeline.

~~~mermaid
flowchart LR
    R["Saved run evidence"] --> V["Verification scanner"]
    T["Saved test evidence"] --> V
    V --> Q["Verification report"]
    Q --> P["Portal verification view"]
~~~

### Verification levels

| Level | Input | Numerical work |
|---|---|---|
| Static integrity | Receipt, timeline, artifacts | None |
| Tracker regression | Saved test/pytest evidence | None |
| Reproducibility comparison | Two saved run receipts/results | None |
| Controlled replay | Explicit selected test configuration | Only with user request |
| Campaign comparison | Later comparable multi-start groups | None unless replay requested |

The initial static scan should report:

- schema/event order and tail-recovery state;
- metric/counter consistency;
- receipt/provenance/configuration differences;
- checkpoint existence and loadability;
- selection rule availability;
- tracker-on/off numerical-equivalence evidence;
- non-finite trial handling;
- line-search trial evidence;
- technical-array omission;
- standalone reader isolation from optimizer imports.

Existing focused tests already cover these tracker behaviours. The future scan
should read their recorded result/evidence rather than pretending a test was
run.

A verification report must carry:

~~~text
scan ID, run ID, input receipt/hash, test version, environment,
mode, result, evidence links, absent checks, and timestamp
~~~

Controlled replay is a separate explicit action with a selected test, budget,
environment, and target. It must never happen merely because a user opens a
plot.

## 13. Delivery phases

| Phase | Deliverable | Approval condition |
|---|---|---|
| 0 | This plan | Scope agreed |
| 1 — complete | Metric descriptors, selection rule, writer/liveness contract | Metric meaning survives reading |
| 2a — complete | Generic projection and incremental validated reader | Arbitrary keys can be projected dynamically |
| 2b | Read-only run discovery/index | Runs can be located without UI assumptions |
| 3 | CLI list/show/follow/local launch | One real run opens locally |
| 4 | Small portal: dynamic metrics, up to three plots, events | One run is understandable |
| 5 | Incremental live following and UI preferences | No full rescans or numerical work |
| 6 | Cache clear/hide and dry-run storage manager | No run evidence is mutated |
| 7 | Archive/trash after retention contract | Recoverable and reviewed |
| 8 | Static verification/test scan | Tested and untested facts distinguished |
| 9 | Campaign/multi-start comparison | Comparability contract exists |
| 10 | Controlled replay and advanced analysis | Explicit authority and budget |

## 14. First implementation boundary

The first build after the contract work is deliberately small:

~~~text
local run discovery
+ adaptive metric catalogue
+ objective and selected-metric plots
+ method/acceptance strips
+ causal event inspector
+ local live refresh
+ clear viewer cache and hide/unhide run
~~~

This must work on a real run before adding storage mutation, test scans,
campaign comparison, or automated decision support.
