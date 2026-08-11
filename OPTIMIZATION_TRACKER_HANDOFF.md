# Optimization tracker and closed-loop plan

## Status and scope

This is the design and implementation plan for the contained optimizer only.
It replaces the earlier tracker handoff plan.

The reference lifecycle is an OLGS quantum-control run:

\[
u^k \longrightarrow \text{forward propagation} \longrightarrow
\text{endpoint metrics} \longrightarrow \text{backward propagation}
\longrightarrow \nabla J(u^k) \longrightarrow u^{k+1}.
\]

The design remains generic over systems satisfying the OLGS contract, but the
HO forward--backward lifecycle is the required stress test and determines the
first tracking API. All work remains under:

~~~text
/Users/billabobz/OPTIMIZER/contained
~~~

This plan does not change stable/v3 or the HO-purification source tree.

## 1. Aim

Build a small, durable record of an optimization that answers, from initial
control to final selected control:

1. How did the objective and declared physical metrics move at each optimizer
   iteration?
2. Which trials were accepted or rejected, and how many forward/gradient calls
   did they cost?
3. Did progress become too small, did the step strategy fail, or was there
   evidence of feasible local stationarity?
4. Why did a human/Codex or a later deterministic policy continue, refine,
   recover, switch method, restart, verify, or stop?
5. Which saved pulse and diagnostic evidence support the final claim, and which
   claims remain unsupported?

The tracker is an evidence and control-loop layer, not a replacement quantum
system, a general experiment platform, or an optimality oracle.

## 2. The two levels that must not be confused

### Quantum propagation level

For one candidate pulse, the system evolves over its time pixels
\(j=0,\ldots,N\). A full \(\rho(t_j)\), costate, and propagator history can be
large. It is not written for every candidate pulse.

### Optimizer-iteration level

The optimizer moves between pulses \(u^k\). One outer iteration records the
existing numerical transaction:

~~~text
current pulse u^k
  -> gradient at u^k
  -> proposed trial pulse u_trial
  -> evaluate(u_trial)
  -> accept or reject
~~~

Every outer iteration receives a compact raw record. This is necessary to find
wasted iterations. The tracker does not record every internal quantum time
step by default.

### Segment level

A segment is a bounded group of outer iterations using one chosen method and
configuration, for example Adam for 20 iterations or L-BFGS for 30 iterations
from the previous best pulse. The strategic loop acts only at segment
boundaries, or immediately after a declared failure.

## 3. Numerical lifecycle for the HO/OLGS case

For the current HO v3 implementation, evaluate(controls) performs a forward
propagation and produces endpoint metrics. gradient(controls) reuses the
matching cached forward history, constructs the backward costate as needed,
and returns the exact discrete gradient. An accepted trial's forward history
is then reused by the next gradient call.

~~~mermaid
flowchart LR
    U["Current pulse u^k"] --> G["gradient(u^k): cached forward + backward"]
    G --> P["Optimizer proposes u_trial"]
    P --> E["evaluate(u_trial): forward propagation + endpoint metrics"]
    E --> A{"Accept?"}
    A -->|yes| N["u^(k+1) = u_trial"]
    A -->|no| U
    E --> R["Raw iteration record"]
    A --> R
~~~

For a simple accepted gradient step this normally costs one new forward
propagation for the proposed pulse and one backward propagation for the
current gradient. A backtracking line search may evaluate several trial pulses
but does not need a new backward propagation for every rejected trial.

If a trial is rejected, the system may have to regenerate the old current
forward history before its next gradient. Retaining full histories for every
trial would trade that computation for potentially excessive memory. The
system owns this cache trade-off; the tracker never changes it.

## 4. Ownership and API boundaries

~~~mermaid
flowchart LR
    S["OLGS system"] -->|"evaluate(u): declared metrics"| E["SystemEvaluator"]
    S -->|"gradient(u): analytical gradient"| E
    E --> O["Optimizer engine"]
    O -->|"immutable engine facts"| B["Tracker bridge"]
    B --> J["Run journal"]

    J --> A["Reader / window analysis"]
    A --> V["Decision view"]
    V --> C["Coordinator"]
    C --> O
    C -->|"explicit supported request"| S
~~~

| Component | Owns | Must not do |
|---|---|---|
| System | Dynamics, forward/backward propagation, objective, physical metrics, analytical derivatives, optional quantum diagnostics. | Write tracker records or choose optimizer strategy. |
| SystemEvaluator | System validation, exact-control evaluation cache, objective/gradient counters. | Invent physical data or alter a method. |
| Optimizer engine | Proposals, trial evaluation, acceptance, standard stopping, current/best control state. | Know HO physics or manually assemble tracker payloads. |
| Tracker bridge | Convert existing system/engine facts to a validated compact tracking record. | Call evaluate, gradient, propagation, residuals, or diagnostics. |
| Journal | Durable causal storage and replaceable live snapshot. | Compute numerical values or make decisions. |
| Analyzer | Read saved records and derive window summaries. | Rerun the system or claim evidence it lacks. |
| Coordinator | Run segments, validate actions, request explicit diagnostics, and preserve handoffs. | Perform system internals or bypass the journal. |

The journal's current writer methods are low-level persistence methods. They
are not the intended public user API. A user must never pass an arbitrary
metrics mapping to describe a real run.

## 5. Planned system-owned tracking contract

The existing required OLGS calls remain:

~~~python
system.evaluate(controls) -> Mapping[str, scalar]
system.gradient(controls) -> Controls
~~~

evaluate is the source of endpoint physical metrics. For HO, it may return J,
fidelity, infidelity, purity, entropy, energy, work, trace, and other finite
endpoint scalars that the system already computed during its normal forward
pass.

The planned optional system call is:

~~~python
system.tracking_spec() -> TrackingSpec
~~~

TrackingSpec is static metadata, not a calculation. It declares:

~~~text
per-iteration scalar fields:
  J                required objective
  fidelity         physical progress field
  control_fluence  objective component
  purity_A         physical state field
  energy/work      selected ledger fields

diagnostic catalog:
  residuals, projected-KKT, Jacobian geometry, trajectory simulation

artifact catalog:
  selected pulse checkpoint, selected physical trajectory artifact
~~~

The system selects field names, definitions, units, and source labels. At run
start the tracker bridge validates this specification. At each evaluation it
extracts only declared finite scalar fields from the already-returned
Evaluation.metrics object.

The intended public optimization shape is:

~~~python
result = opt.adam(
    system,
    initial_controls,
    maxiter=20,
    tracking=TrackingConfig(profile="standard"),
)
~~~

The user chooses a profile and an allowed diagnostic budget, never numerical
metric values. The engine supplies those values automatically from the system
and its own existing state.

## 6. Explicit diagnostics

Some evidence is too expensive or too large for every iteration:

~~~text
projected-KKT residual
constraint residual/Jacobian geometry
Hessian-vector evidence
selected rho(t), chi(t), populations, or Fock-sector response
~~~

The coordinator may request an advertised system diagnostic only at an
explicit decision point:

~~~text
Coordinator -> diagnostic broker -> supported system API -> analysis record
~~~

The request must name its purpose, input checkpoint, allowed cost, and output
fields. Its extra calls are counted and stored. The journal never requests a
diagnostic by itself.

## 7. What is saved

~~~text
<run>/
  run_receipt.json      small immutable run configuration
  timeline.jsonl        ordered source of numerical and decision evidence
  checkpoints/
    *.npz               selected full control arrays and optional resume state
  artifacts/
    *.npz or *.json     selected system-owned diagnostic output only
  live.json             latest replaceable progress display
  terminal.json         direct pointer to the selected final event
~~~

### 7.1 run_receipt.json

This remains small: run ID, objective direction, method configuration, seed,
and references to the system/problem authority. It is not a duplicate problem
statement and it is not the main numerical record.

### 7.2 timeline.jsonl

Every event has a schema version, run ID, event ID, causal sequence number,
UTC timestamp, elapsed time, event kind, and JSON-safe payload. The timeline
is the scientific source of truth.

| Event | Purpose | Core contents |
|---|---|---|
| run_start | Initial evaluated state. | Initial declared scalars, method, counters. |
| iteration | One outer optimizer transaction. | Before/trial/after scalars, proposal summary, acceptance result, counters, best-so-far. |
| analysis | Derived reading of a declared raw window. | Window bounds, formulas/thresholds, inputs, findings, diagnostic counters. |
| checkpoint | Pointer to a selected full control state. | ID, label, path, shape. |
| segment_start and segment_end (planned) | One method/budget interval. | Segment ID, method/configuration, initial/final checkpoint, budget, entry/exit reason. |
| decision (planned) | A manual or policy recommendation. | Authority, evidence window, permitted actions, selected/rejected directive, validator outcome. |
| intervention | An action outside normal stepping. | Action, reason, parameters, source decision, before/after state, outcome. |
| run_end | Selected terminal result. | Status, reason, final scalars, selected checkpoint, explicit certificate evidence. |

### 7.3 Per-iteration raw record

The raw iteration data is the minimal material required to diagnose wasted
optimization work:

~~~text
method, segment, iteration number
before: declared scalar metrics and cumulative counters
proposal: gradient norm, step size, method technical summary
trial: line-search trial scalar values when they occurred
after: declared scalar metrics and cumulative counters
best-after: declared scalar metrics for the best accepted pulse
accepted/rejected and exact reason
~~~

No complete pulse, gradient array, density-matrix history, costate history, or
propagator array is written in a normal iteration record.

### 7.4 Checkpoints and selected artifacts

Checkpoints save the full Controls matrix, ControlSpec, control metadata, and
JSON-safe optimizer resume data. They are written at initial, meaningful best,
handoff, pre-intervention, failure, or final boundaries—not at arbitrary
record counts.

System trajectory data is saved only when an explicit diagnostic requests it
for a selected checkpoint. The timeline then stores the artifact pointer and
the diagnostic configuration that produced it.

## 8. Raw records, window analysis, and wasted-work detection

Raw records are always kept. Analysis is a separate, inexpensive reading of
those records; it adds no propagation or gradient call.

For a window of \(W\) outer iterations, the analyzer derives:

~~~text
best-objective improvement and objective slope per evaluation
fidelity and objective-component changes
gradient-norm trend
accepted-step fraction and rejection streak
number of trial forwards spent per accepted move
elapsed time and forward/gradient budget consumed
availability of feasibility or KKT evidence
~~~

| Window evidence | Interpretation | Permitted next direction |
|---|---|---|
| Objective and fidelity improve meaningfully. | Productive segment. | Continue/refine. |
| Objective flat but gradient remains large. | Stalled or poorly conditioned, not convergence. | Change step strategy, switch method, or restart. |
| Rejection fraction is high. | Trial forwards are being wasted. | Recover line-search/step configuration. |
| Objective improves while fidelity falls and fluence falls. | Objective trade-off, not unqualified physical improvement. | Review weights or objective decomposition. |
| Gradient is small but feasibility/KKT is absent. | Insufficient stopping evidence. | Request an explicit diagnostic. |
| Feasible with a small projected-KKT residual. | Local-stationarity evidence. | Stop/refine; never call it global proof. |
| No best improvement over several windows. | Budget may be wasted in this basin. | Diversify/restart/terminate branch. |

The analysis schedule is configurable, for example every 10 or 20 outer
iterations, plus immediate checks for non-finite trials or a rejection streak.
It reads existing scalar values only.

## 9. Closed-loop modes

### 9.1 Manual/Codex mode — first operational mode

1. The user chooses an objective, initial pulse, allowed methods, and budget.
2. The coordinator runs one bounded segment.
3. Raw iteration records and a window analysis are written.
4. User/Codex reviews the evidence and chooses the next allowed directive.
5. The decision, action, selected checkpoint, and later outcome are recorded.
6. The next segment starts from that checkpoint.

### 9.2 Deterministic policy shadow mode — later

A fixed, versioned rule set receives only a compact DecisionView from saved
records. It can recommend a directive but cannot alter the run. Its
recommendation and the eventual human action are both logged for comparison.

### 9.3 Deterministic active policy mode — later and only after shadow tests

The same fixed rule set may choose only declared actions at a segment boundary:

~~~text
continue, refine, recover, switch_method, restart,
request_diagnostic, checkpoint, stop
~~~

The coordinator validates a directive against available methods, system
capabilities, budget, bounds, checkpoint availability, and mode transition
rules before executing it. The policy has no system object and no journal
write access.

There is no Ollama/LLM control path and no RL, self-learning, or online policy
training in this plan.

## 10. Efficiency rules

1. Tracking alone adds zero system evaluate, gradient, forward, backward,
   residual, Jacobian, HVP, or trajectory calls.
2. Record outer optimizer iterations, not internal propagation pixels.
3. Use scalar numerical values in JSONL; write complete arrays only to selected
   compressed artifacts.
4. Reuse the engine's existing Evaluation, gradient, proposal, acceptance, and
   cache counters; do not reconstruct them afterwards.
5. Run analysis from the saved scalar stream and only at the chosen cadence.
6. Give each concurrent start its own run folder and one writer; never let
   workers share a JSONL file.
7. Keep logging profiles small: minimal, standard, and explicit diagnostic
   requests. Benchmark tracker-on/off before adopting a profile.
8. If strict per-event durability measurably harms a cheap numerical problem,
   introduce a bounded writer buffer with explicit flushes at checkpoints,
   decisions, and terminal events. It must never silently drop evidence.

## 11. Implementation phases

| Phase | Deliverable | Completion condition |
|---|---|---|
| 0 — complete | v2 record model, reader, checkpoint artifacts, and non-live HO Trace-projection pilot. | Pilot validates the layout but is explicitly labelled non-live. |
| 1 — next | Optional live engine observer and tracker bridge at core.run_chunk. | Enabled/disabled runs have identical controls, metrics, and evaluator counts; existing engine facts produce every raw iteration record. |
| 2 | Reader window analysis and tiny read-only live view based on live.json. | Detects progress, stall, rejection, and insufficient evidence without system calls. |
| 3 | Manual segment coordinator, explicit handoffs, decisions, and selected artifacts. | A method switch/restart is reproducible from its decision, checkpoint, and outcome. |
| 4 | Deterministic policy shadow mode. | It produces replayable recommendations and cannot change any run. |
| 5 | Deterministic active policy for approved modes/directives. | Validation rejects disallowed actions; every action has evidence, policy version, and outcome. |
| Later if needed | Multi-start/campaign comparison. | Candidate runs are independently durable and compared on declared common settings. |

## 12. Required validation

### Phase 1

1. Tracker enabled/disabled gives identical terminal controls, scalar metrics,
   stop reason, and system evaluator counts for deterministic systems.
2. The live timeline contains initial state plus every attempted outer
   iteration in causal order.
3. Each iteration is built from the existing engine facts; no tracker system
   calls occur.
4. Line-search trial scalar values are preserved without trial pulse arrays.
5. live.json is always valid JSON and identifies the last complete event.
6. Checkpoint loading reproduces the stored pulse and control specification
   without a system object.
7. A small real HO v3 run exposes declared endpoint metrics without an
   HO-specific tracker special case.

### Phases 2--5

1. Window analysis gives the same result when replayed from the same timeline.
2. An unavailable diagnostic produces an explicit not available result, not a
   fallback numerical calculation.
3. A policy-shadow run changes no numerical result or evaluator count.
4. Active-policy directives are rejected when unsupported, over budget, or
   outside the configured mode.
5. Every decision links to raw evidence, selected checkpoint, action, and
   measured later outcome.
6. Concurrent starts write separate valid journals without shared-writer
   corruption.

## 13. Claim boundary

The tracker can show objective trajectory, physical-metric trajectory, cost,
accepted/rejected steps, method handoffs, and evidence actually requested.

It cannot turn a loss curve into a theorem. In particular:

~~~text
small raw gradient alone              != constrained local stationarity
one apparently converged run          != global optimum
better final objective from one start != basin/global reliability
~~~

Feasibility and projected-KKT evidence are required for a constrained local
stationarity claim. Comparable multi-start, continuation, or other declared
campaign evidence is required for basin/global conclusions.

## 14. Explicit exclusions

- no change outside contained;
- no automatic promotion to stable/v3;
- no HO source-tree modifications;
- no journal-triggered system calculations;
- no raw quantum trajectory saved at every optimizer iteration;
- no hidden finite-difference, KKT, or diagnostic work;
- no LLM/Ollama decision layer;
- no RL, online learning, or self-modifying policy;
- no TensorBoard-scale platform, database, server, WebSockets, or third-party
  tracking service; and
- no global-optimality claim without the required mathematical/campaign
  evidence.

## 15. Review rule

Only one phase is integrated at a time. The user reviews the artifact,
tracker-on/off equivalence evidence, and real HO smoke result before the next
phase begins.
