---
role: topic
parent: UTILS_HUB.md
summary: "Explicit grids, owned control candidates, waveform transformations and shooting recovery."
read_when: "Read when exchanging controls between grids or composing a new starting waveform."
status: active
reviewed: 2026-10-12
---
# Control transformations

`opt.utils` transforms arbitrary named floating or complex control channels.
Every operation returns a `ControlTransformResult` with `.controls`, parameters,
source/target grids, numerical facts and service source references. Inputs and
their metadata remain independent. Old metrics and solver memory are discarded.
Select a candidate explicitly with `opt.session.use(candidate.controls, context=...)`.

Pure Python calls install no history writer. Terminal operations use the existing
[command history](../history/HISTORY.md), including failures and inspections.
Transform reports can be saved with `opt.checkpoints.save(candidate, kind="report")`;
save `.controls` with `kind="controls"`. These are physical restarts, not exact
continuations of the originating optimizer.

## Required declarations

- `ControlGrid(times, start, stop, placement, units=None)` records strictly
  increasing sample times and the physical interval. `placement` is `nodes`,
  `left_cells` or `cell_centers`. At least two samples are required.
- `uniform_control_grid(..., count=N)` always means **N stored samples**.
  1001 nodes contain 1000 intervals; 2000 intervals require 2001 nodes.
- A supplied grid must match the sample count and any declared `ControlSpec.dt`.
  A grid already stored in `Controls.meta["control_grid"]` cannot be contradicted.
  Unknown units stay unknown. Time units must match across a conversion; no
  implicit unit conversion occurs. Control units use `ControlSpec.meta["units"]`,
  either one label or a mapping keyed by channel.
- Resampling uses an explicit target `ControlSpec`; preserve channel order/units.
  Linear interpolation is supported, including complex components. Extrapolation
  defaults to `error`; explicitly choose `hold` or `linear` outside coverage.
  Cell-center refinement may require extrapolation even on the same interval.
- Downsampling reports an aliasing warning. Apply a suitable explicit filter
  first when needed; interpolation does not automatically filter.
- Integer inputs/targets and unsafe output casts are rejected. These operations
  do not certify hard-constraint feasibility. Evaluate, verify or repair afterward.
- `max_bytes` preflights estimated numerical arrays (default 64 MiB in Python).
  The service caps it at the session data budget. Retention/persistence quotas
  still apply. This is not an OS RSS limit or a bound on project-hook allocations.

## Operations

| Call | Meaning and important arguments |
|---|---|
| `uniform_control_grid(start=..., stop=..., count=..., placement=..., units=...)` | Explicit uniform grid. Nonuniform grids use the `ControlGrid` constructor. |
| `resample_controls(controls, source_grid=..., target_grid=..., target=..., method="linear", extrapolation="error")` | Interpolation or explicitly selected boundary extension. |
| `smooth_controls(controls, sigma=..., boundary=..., grid=None, channels=None)` | Gaussian convolution, sigma in **samples**, truncated at four sigma; supplied grid must be uniform. |
| `filter_controls(controls, grid=..., kind=..., cutoff=..., boundary=..., channels=None)` | Ideal FFT lowpass/highpass with one cutoff; bandpass/bandstop with two. Uniform grid required. Cutoff is cycles per declared time unit, bounded by Nyquist. |
| `scale_controls(controls, factors=..., offsets=0, channels=None)` | Exact affine arithmetic. Scalar, selected-channel mapping or selected-row sequence. |
| `normalize_controls(controls, norm="max_abs", value=1, scope="global", channels=None, grid=None, zero="error")` | `max_abs`, `l2`, `rms`, `time_l2` or `time_rms`; scope `global` or `channel`; `zero="keep"` preserves zero channels. |
| `mix_controls(inputs, weights=..., grids=None)` | Explicit real signed weights, never implicitly normalized. Full layout, grid and control units must agree. |
| `select_channels(controls, channels=...)` | Select/reorder arbitrary keys; creates the corresponding layout. |
| `replace_channel(controls, channel=..., replacement=..., replacement_channel=..., grid=None, replacement_grid=None)` | Replace one row on the identical declared grid with matching dtype/units. |
| `assemble_channels(inputs, channels=..., target=..., grids=None)` | One `(input_index, source_key)` pair per target row, in target order. |
| `join_controls_in_time(inputs, grids=..., boundary="error")` | Absolute-time contiguous segments; duplicate nodes require `left` or `right` to select the retained sample. |
| `apply_control_bounds(controls, lower=..., upper=..., channels=None)` | Real clipping and correction count/norm; no nonlinear constraint repair. |
| `recover_shooting_controls(variables, grid=..., context=...)` | Terminal OLSS recovery through the optional project hook. Python additionally supplies `system` first. |

Smoothing/filtering boundaries are `periodic`, `reflect` or `hold`. The boundary
is mandatory because endpoint assumptions change the result. Ideal FFT masks can
ring; assess the returned waveform rather than assuming physical admissibility.

Time-weighted norms use trapezoid node weights, left-cell widths, or cell-center
midpoint boundaries. Global RMS averages across selected channels as well as
time/samples. `time_l2` sums channel integrals. Different conventions are explicit.

## Details

### Terminal workflow

The example uses the neutral fixture from the checkout root. Launch `optimizer`
once, then submit these statements:

```python
opt.session.load("coarse", "tests/fixtures/quadratic_system.py", parameters={"N": 1001});
old = opt.guesses.constant_guess(value=0.1);
source = opt.utils.uniform_control_grid(start=0, stop=1, count=1001, placement="nodes", units="s");
fine = opt.utils.uniform_control_grid(start=0, stop=1, count=2001, placement="nodes", units="s");
opt.session.load("fine", "tests/fixtures/quadratic_system.py", parameters={"N": 2001});
target = opt.control_spec(context="fine");
candidate = opt.utils.resample_controls(old, source_grid=source, target_grid=fine, target=target, context="fine");
filtered = opt.utils.filter_controls(candidate.controls, grid=fine, kind="lowpass", cutoff=20, boundary="reflect");
check = opt.utils.metric_report(controls=filtered.controls, context="fine");
filtered.facts;
opt.session.use(filtered.controls, context="fine");
trial = opt.adam(maxiter=5, activate=False, context="fine");
trial.metrics;
```

`opt.control_spec(context=...)` is a service convenience returning the selected
OLGS layout without overrides. Python uses `system.control_spec()` or the existing
`opt.control_spec(keys, control_dim, ...)` constructor. If resampling/assembly
omits `target`, a selected/explicit OLGS context supplies it. With an explicit
target and no context, pure transforms need no loaded system. Supplying a context
checks that target before computation. Ordinary optimizer validation stays strict.

### Shooting exchange

The optional `recover_controls(variables, *, times)` hook returns `Controls` or a
mapping containing `values`, a typed `spec`, and the exact requested `times`.
Recovery validates the OLSS variables and returned waveform; it does not alter
the shooting context. Transfer the waveform to a compatible OLGS context and
evaluate it explicitly. The project owns control units and channel meaning.

The inverse direction needs a project-specific initializer/fitting map. A waveform
does not determine shooting costates or constraint multipliers generically.

### Ownership and verification

[control_tools.py](control_tools.py) owns pure transformations;
[service.py](../operations/service.py) owns target-context validation and retained
references; the allowlisted [codec](../checkpoints/codec.py) owns persistence.
[test_control_tools.py](../../tests/test_control_tools.py) checks analytical
signals, arbitrary channels, complex handling, boundaries, budgets, activation,
typed checkpoints and command replay. Article benchmarks and managed jobs are
separate reviewed work.

---

[⌂ Home](UTILS_HUB.md)
