# OPTIMIZER v3 — Design & Implementation (self-contained)

Status: **Phase 6 complete for review** — public facade implemented; Phase 7 first
optimizers are next only after approval.
Last updated: 2026-07-15

> Start with the top-level docs in this README's reading order. The [`plan/`](plan/README.md)
> folder keeps the layer-by-layer and class-by-class detail notes that support those
> top-level docs.

This folder holds **all** design and planning for the next version (v3) of the
optimizer library. It is **self-contained**: every document needed to understand,
continue, or hand off the v3 effort lives here (design docs + copied reference
material under `reference/`). It has no dependency on files outside this folder, so it
can be relocated as a standalone project without breaking any links.

> New thread? **Read this README top to bottom first**, then the design docs in order.
> It gives the context, the status, the process, and the rules.

---

## 0) What v3 is (one paragraph)

v3 restructures the optimizer library into a direct-call toolbox. A project
`system.py` owns the physics, control layout, objective components, cost prefactors,
analytical gradient, residuals, Jacobians, and higher derivative hooks. The library
supplies vectorized controls, a shared engine, optimizers, schedules, guesses,
constraints, diagnostics, repairs, trace logs, checkpoints, and later modes. The
public style should stay direct: `opt.adam(...)`, `opt.line_search(...)`,
`opt.fourier_guess(...)`, `opt.repair_newton(...)`. Full rationale + backbone:
[`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## 1) Folder contents (read in this order)

| # | File | What it gives you |
|---|---|---|
| 1 | **`README.md`** (this file) | context, status, process, rules, resume checklist |
| 2 | [`CONTEXT.md`](CONTEXT.md) | durable implementation context and rules |
| 3 | [`BUILD_PLAN.md`](BUILD_PLAN.md) | phase-by-phase build plan and acceptance checks |
| 4 | [`ARCHITECTURE.md`](ARCHITECTURE.md) | current architecture, system-owned objective, direct API, build order |
| 5 | [`OPTIMIZERS.md`](OPTIMIZERS.md) | optimizers that move controls |
| 6 | [`AUXILIARY.md`](AUXILIARY.md) | guesses, schedules, constraints, diagnostics, repairs, logs/checkpoints |
| 7 | [`plan/`](plan/README.md) | detailed layer and class/module notes |
| — | [`reference/`](reference/) | copied source material (see [`reference/README.md`](reference/README.md)) |

`reference/` (self-contained snapshots, so no external files are needed):
- [`reference/OPTIMIZATION_METHODS.md`](reference/OPTIMIZATION_METHODS.md) — the proven
  constraint-manifold method ladder + diagnostic playbook (source of the advanced ideas).
- [`reference/p1_adaptive_energy.md`](reference/p1_adaptive_energy.md) — the adaptive
  energy/λ tradeoff problem that curriculum/stage parameter tuning targets.
- [`reference/legacy_v2_snapshot.md`](reference/legacy_v2_snapshot.md) — a self-contained
  summary of the current (v2) package that v3 restructures, so no external code needs
  reading.

---

## 2) Status by phase

| Phase | Deliverable | Status |
|---|---|---|
| **0** | project setup and build context | complete |
| **1** | vectorized control container | complete |
| **2** | optimizer-facing system contract | complete |
| **3** | result and run/warmstart state | complete |
| **4** | trace records and in-memory checkpoints | complete |
| **5** | shared core engine | complete |
| **6** | public facade (`optimizer/__init__.py`, `library.py`) | complete for review |
| **7** | first optimizers (`adam`, `momentum`, `line_search`) | next after approval |
| **8+** | schedules, guesses, constraints, diagnostics/repairs, advanced methods, modes | pending |

**Immediate next action:** review Phase 6. After approval, commit Phase 6 and move to
Phase 7 first optimizer planning/implementation.

---

## 3) How this work is run (process)

One layer at a time; each layer runs the loop:

```
info  →  plan  →  review  →  fix     (repeat until the user approves the layer)
```

- Deliverables are **markdown design docs first**; code skeleton only after the
  class/module contracts are approved.
- **Show lists/designs for review before writing them to files**, unless the user has
  already reviewed and said "save".
- Never jump ahead to a later layer while the current one is under review.
- **Naming is user-led**: propose apt/unambiguous options; the user chooses.

---

## 4) Vocabulary

Use role names internally, but keep public calls direct.

| Role | Meaning | Public style |
|---|---|---|
| optimizer | moves controls to reduce current system `J` | `opt.adam(...)` |
| guess | creates initial controls from `system.control_spec()` | `opt.fourier_guess(...)` |
| diagnostic | measures and reports; does not move controls | `opt.geometry_probe(...)` |
| repair | moves controls to restore feasibility | `opt.repair_newton(...)` |
| schedule | step-size, line-search, trust, or damping policy | config/helper |
| trace/checkpoint | logs and rollback/resume state | `opt.Trace(...)` |

Quantum-control-specific gradient formulations belong in the system, not in the
optimizer registry.

---

## 5) Hard rules

1. **Legacy is frozen.** v3 never modifies the existing (v2) package; v3 is a separate,
   opt-in layer. What v2 is and why it is frozen:
   [`reference/legacy_v2_snapshot.md`](reference/legacy_v2_snapshot.md).
2. **Docs-first.** No code or skeleton until the class/module contract layer is
   explicitly approved.
3. **Self-contained folder.** Keep all v3 information inside this folder. Do not add
   links or path dependencies to files outside it — this folder may become a standalone
   project. Record v3 changes in §7 of this README (the folder's own change log).
4. **System-owned analytics.** Normal optimization expects the system to provide
   analytical `evaluate()` and `gradient()` behavior. Numerical derivatives are for
   verification or explicit fallback experiments.
5. **Don't auto-commit.** Remind the user to commit; let them do it.

---

## 6) Key decisions: locked vs open

**Locked (this design pass):**
- Direct public API: `opt.method(...)`.
- System-owned multi-objective cost prefactors.
- One engine loop shared by optimizers.
- Vectorized `Controls` and `ControlSpec` as a foundation.
- Short-chunk curriculum with trace/checkpoint rollback.
- Modes are deferred until the toolbox layers are stable.

**Open (resolve in review loop / later layers):**
- Exact package import name and compatibility bridge.
- Final method/function names after class-level review.
- Whether old v2 utility code is wrapped, copied, or rewritten.
- Exact persistence format for trace/checkpoint files.
- Advanced method priority after the first-wave methods are stable.

---

## 7) Change log (v3 folder)

Keep v3 history here so it travels with the folder.

- **2026-07-09** — Consolidated all v3 planning into this self-contained `v3_plans/`
  folder; added this README, structured the design docs (`ARCHITECTURE.md`,
  `OPTIMIZERS.md`, `AUXILIARY.md`), and copied source material into `reference/`.
- **2026-07-09** — Layer 1/1b: optimizer catalog + auxiliary catalog (correctors,
  diagnostics, enhancers) drafted with namespaced ids.
- **2026-07-08** — Layer 0: backbone architecture drafted.
- **2026-07-15** — Added [`plan/`](plan/README.md) as the current review plan:
  direct `opt.method(...)` API, system-owned multi-objective cost params,
  short-chunk curriculum, trace/checkpoint rollback, warmstart, guesses, and
  class-by-class module notes.
- **2026-07-15** — Folded the new plan direction into top-level `ARCHITECTURE.md`,
  `OPTIMIZERS.md`, and `AUXILIARY.md`; removed stale `opt.tools`/wrapper-heavy and
  normal-FD-fallback wording.
- **2026-07-15** — Phase 0 started: added `BUILD_PLAN.md`, `CONTEXT.md`, and
  `.gitignore` for tracked implementation work.
- **2026-07-15** — Phase 1 complete for review: added `ControlSpec`, `Controls`, and
  focused control-container tests.
- **2026-07-15** — Phase 2 complete for review: added system contract helpers,
  minimal system template, and focused system-contract tests.
- **2026-07-15** — Phase 3 complete for review: added evaluation/result containers,
  run/warmstart state containers, and focused result/state tests.
- **2026-07-15** — Phase 4 complete for review: added trace records, in-memory
  checkpoints, label-based restore, and focused trace/checkpoint tests.
- **2026-07-15** — Phase 5 complete for review: added the shared chunk engine,
  evaluation/cache helpers, stopping rules, local parallel-map helpers, and focused
  engine tests.
- **2026-07-15** — Phase 6 complete for review: added `import optimizer as opt`
  facade, default `OptimizerLibrary`, bound `OptimizerContext` via `opt.context(system)`,
  reserved direct method names, and public API tests using a temporary
  universal-robust-4th-style fixture.

---

## 8) Resume checklist (new thread)

1. Read this README fully, then `CONTEXT.md`, `BUILD_PLAN.md`, `ARCHITECTURE.md`,
   then `OPTIMIZERS.md` + `AUXILIARY.md`.
2. Skim `reference/` for the advanced-method source and the v2 legacy snapshot.
3. Find the current phase in §2 and `BUILD_PLAN.md`; continue that phase's
   `info→plan→review→fix` loop — do not skip ahead.
4. Propose in chat before writing to files; keep naming user-led.
5. On any change: update §2 (status) and §7 (change log) here; remind the user to commit.
