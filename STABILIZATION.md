---
title: V4 stabilization plan
type: plan
module: root
tags:
  - optimizer
  - plan
  - stabilization
---

# V4 stabilization plan

How OPTIMIZER V4 becomes a stable baseline before the terminal interface and the
MCP server are built. Draft for review, written on 2026-10-11 from a read-only
review of commit `45ac041`. Once V4 is stable, this plan moves to the
workspace `temp/` folder.

## Definition of stable

V4 is stable when all four conditions hold:

1. **One release truth.** The installed wheel, `RELEASE_MANIFEST.json`, the Git
   tag, and the registry record name the same commit, and the registry doctor
   passes.
2. **A declared API.** The runtime catalog from `opt.list()` is the public API.
   Every entry is documented, and removing one follows a written deprecation
   rule.
3. **Green, counted tests.** The full suite passes, and its count is recorded in
   [the release evidence](RELEASE_EVIDENCE.md).
4. **Reachable docs.** Every package is reachable from [the README](README.md),
   and every doc follows Markdown Protocol.

## Steps

Work one step at a time; each step ends with the user's review before the next
one starts.

| Step | Deliverable | Done when |
|---|---|---|
| S1 Docs and API | Docs for every package and the core types, plus the declared API list | Every doc is reachable from the README, the open V3-audit gaps are closed, the API list is agreed, and the doc-link test passes |
| S2 Code tidy | A current context command and a clean checkout | `optimizer_context.py` lists only existing docs and the right environment; `build/` and the empty `.claude/skills/` are gone |
| S3 Registry clean-up | A registry that states the real layout | Legacy versions have their own group, V3's `editable` flag is correct, route meanings are written down, and `release_artifacts/` has a policy |
| S4 Release gate | A new tagged release | One release truth holds, preflight and doctor pass, and the user has approved the release |

P4 follows S4: one design doc each for the terminal interface and the MCP
server, with names chosen by the user.

## Decisions needed

| Decision | Needed by | Options |
|---|---|---|
| Version for the commits after `v4.0.3` | S4 | `4.0.4` treats them as fixes; `4.1.0` reflects the new reference-axis stubs in `draw_sphere` |
| Which catalog entries are frozen API | S1 | All 91 entries, or a smaller core with the rest marked provisional |
| The placeholder hubs for schedules, logs, and tests | S1 | Document them fully, or mark them deferred with a reason |
| Markdown Protocol scope | S1 | All 83 docs that lack the Home footer (proposed), or only docs S1 touches |
| The Git remote named `OPTIMIZER_v3` | S3 | Rename it on GitHub, or keep it |

## Boundaries

- No release tag without the user's explicit instruction; the user decides
  whether a commit is scientifically ready.
- Docs parked in the workspace `temp/` folder are raw material, not authority.
  Check them against the current code before reusing anything.
- This plan stabilizes the current design. Redesigning a module needs its own
  design doc.

## Details

### S1 work list

1. Add `tracker`, `optimizer_log`, `campaigns`, `visualization`, and the root
   modules `continuation.py`, `introspection.py`, and `catalog.py` to the
   README's package map and hub routes. Link the docs that are now unreachable:
   [release evidence](RELEASE_EVIDENCE.md),
   [campaign modes](optimizer/campaigns/CAMPAIGNS.md),
   [visualization](optimizer/visualization/VISUALIZATION_HUB.md),
   [the log reader](optimizer_log/OPTIMIZER_LOG_READER.md), and the two
   tracker trials under `test_contained/`.
2. Write a tracker doc from the parked
   `temp/v4/optimizer/tracker/TRACKER_MODEL.md`, checked against
   `optimizer/tracker/`.
3. Write one note on the core types: the exact constructor fields of
   `ControlSpec`, the `Controls` constructors, and the fields of
   `OptimizerResult`, `RunState`, `WarmStartState`, and `Evaluation`.
4. Replace the README Fast Start's undefined `sys = system(params)` with the
   construction shown in [the OLGS system hub](optimizer/system_olgs/SYSTEM_OLGS_HUB.md).
5. Document `observers`, the tracker hook, in
   [the optimizer contract](optimizer/optimizers/OPTIMIZER_CONTRACT.md) beside
   `trace`, `create_trace`, `stage`, and `use_cache`.
6. Document the `OLGS` base class, the result containers, the validators
   `probe_system` and `require_system`, and the `optimizer-system` scaffold
   command.
7. Link the worked example `tests/fixtures/quadratic_system.py` from the OLGS
   system hub and [the testing hub](tests/TESTING_HUB.md).
8. Settle the three placeholder hubs.
9. Redesign the system-build skill from the parked
   `temp/v4/optimizer/system_olgs/templates/OLGS_SYSTEM_BUILD_SKILL.md`.
10. Propose the frozen API list and a deprecation rule.
11. Add the Home footer and run the Markdown checks.

### S2 work list

1. In `optimizer_context.py`, replace the two anchors that point to parked docs
   and the hard-coded `contained` Conda environment.
2. Remove the ignored `build/` residue and the empty `.claude/skills/` folder.

### S3 work list

1. Give legacy versions their own registry group and teach `list`, `doctor`,
   and `preflight` to read it.
2. Mark V3 `editable: true`, and refresh its install receipt, which still points
   to `/Users/billabobz/OPTIMIZER_v3`.
3. Write down what the `default`, `release`, and `update` routes mean when all
   three point to this checkout.
4. Decide what `release_artifacts/` keeps; it holds only the v4.0.1 wheel.

### S4 work list

1. Set the version, rebuild the wheel, and reinstall it into `optimizer_v4`
   with `install.sh`.
2. Run the full suite, then update the release evidence, the manifest, and the
   registry record.
3. Run preflight and doctor; tag only on the user's instruction.

### Findings behind this plan

- **Code:** 159 tests in 19 files cover every package. The runtime catalog
  lists 91 entries in 11 groups.
- **Release:** five commits follow `v4.0.3`: four visualization changes and one
  docs change. The `optimizer_v4` environment runs a wheel built from
  `43a944a` that is still labelled 4.0.3 and differs from the manifest's
  wheel. The manifest records 150 tests. The registry doctor fails on `v4`, and
  the `build/` residue would fail preflight.
- **Docs:** 6 of 89 docs are unreachable from the README, which never mentions
  the tracker or the log reader. The tracker has no doc since its model note
  was parked. 83 of 89 docs lack the Home footer.
- **Not yet built:** run discovery and an `optimizer_log` command; the tracker
  reports `run_discovery: False`.

### Candidate API: runtime catalog at `45ac041`

- **optimizers (13):** adam, momentum, line_search, adagrad, rmsprop, lbfgs,
  nonlinear_cg, ncg, cma_es, projected_gradient, newton_cg, trust_region,
  proximal_gradient
- **guesses (13):** constant_guess, ramp_guess, sine_guess, cosine_guess,
  gaussian_guess, sinc_guess, fourier_guess, random_guess,
  random_smooth_guess, random_fourier_guess, scale_guess, mix_guess,
  perturb_guess
- **utils (13):** metric_report, diagnostic_report, geometry_probe,
  finite_difference_gradient, verify_gradient, finite_difference_jacobian,
  verify_jacobian, nullspace_basis, project_gradient, repair_newton,
  control_spectrum, smoothness_report, RepairResult
- **schedules (2):** ConstantSchedule, AdaptiveStepSchedule
- **base (13):** ControlSpec, Controls, evaluate, gradient,
  value_and_gradient, OptimizerResult, WarmStartState, Trace, context,
  system_info, plot_data, context_info, continuation
- **core (7):** run_chunk, StepContext, StepProposal, AcceptanceDecision,
  SystemEvaluator, StoppingConfig, metric_guard
- **campaigns (6):** Campaign, create, open, MethodSpec, SelectionSpec,
  SystemReference
- **tracker (5):** RunJournal, JournalObserver, CheckpointPolicy,
  read_timeline, load_checkpoint
- **visualization (12):** PlotData, plot_bloch, draw_timeseries, draw_sphere,
  style_context, validate_bloch_data, plot_bloch_panel, plot_projection_panel,
  plot_controls_panel, validate_controls_data, save_figure, save_panel
- **system_olgs (4):** get_system_spec, require_system, with_secondary,
  optional_control_bounds
- **optimizer_log (3):** read_timeline, follow_timeline, inspect_writer

### Inputs for P4

- The catalog and discovery API (`opt.list`, `opt.info`, `opt.search`,
  `opt.path`) can back the command list of a terminal interface and the tool
  list of an MCP server.
- `optimizer_context.py` already prints a JSON context snapshot.
- The `optimizer_log` reader already reads, follows, and inspects run files.
- The parked portal plan (`temp/v4/OPTIMIZER_LOG_PORTAL_PLAN.md`, section 9)
  sketches CLI commands and a small local service; the parked tracker plan
  describes manual and policy-driven closed-loop modes.

---

[⌂ Home](README.md)
