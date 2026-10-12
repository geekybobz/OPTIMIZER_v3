---
role: deep
parent: ../../tests/TESTING_HUB.md
summary: "Known-answer pi-pulse tests for gradient and shooting, soft and hard, through the Python API, the operation service and the terminal."
read_when: "Read when checking the interfaces on small problems with analytical answers, or rerunning the sample suite."
status: active
reviewed: 2026-10-12
---
# Pi-Pulse Sample Tests

The next installed-MCP campaign has a separate [four-case plan](MCP_TEST_PLAN.md)
with T=1 and 1000 intervals. It has not run; the evidence below is historical.


Small problems with analytical answers check that one workflow gives the same
correct result through every interface. They test interface behavior and tool
structure; they are not scientific benchmarks and tune no parameters.

## Status

On 2026-10-12, `run_samples.py` passed 61 checks with no failures and one
pending check. It ran on `v4.0` at `7cfa486`, `3be3bf9` and `80a311f` (the
last with the terminal UX changes), with the same result each time. Each run used
the `optimizer_v4` environment with checkout source first on the import path.
MCP is pending because no adapter exists. Terminal and MCP command history waits
for its own task.

## Model and conventions

- One-channel resonant qubit, `H(t) = u(t) σx / 2`, duration `T = 1`, transfer
  `|0⟩ → |1⟩`. The rotation angle is `θ = ∫u dt`.
- `F = |⟨1|ψ(T)⟩|² = sin²(θ/2)`, and `⟨0|ψ(T)⟩ = cos(θ/2)`. The second is real
  for this model, so the hard condition is one real equation.
- `E = ∫u² dt` is the full energy; the half energy is `E/2`.
- OLGS grid: `N = 100` piecewise-constant intervals, `dt = 0.01`, one channel
  `ux`.
- Soft: `J = (1 − F) + α E`, with `α = 0.01` as a secondary parameter. Hard:
  `J = E` subject to `Re⟨0|ψ(T)⟩ = 0`, the `hard` equality constraint.
- OLSS uses reduced Pontryagin shooting: `θ' = u`, `H = p u − u²/2`, so `u = p`
  and `p' = 0`. The one shooting variable is `z = p(0)`.
  - Hard residual: `cos(θ(T)/2)`.
  - Soft residual: `p(T) − sin θ(T) / (4α)`, the transversality condition of
    `C = J / (2α)`. Its minimizers are those of the OLGS soft problem.

## Expected answers

| Quantity | Hard | Soft (α = 0.01) |
|---|---|---|
| Optimal control | constant `u = ±π = ±3.1415926536` | constant `u = ±θ* = ±3.0204776615`, with `sin θ* = 4αθ*/T` |
| `F` | 1 | 0.9963372703 |
| `E` | π² = 9.8696044011 | 9.1232853034 |
| `J` | 9.8696044011 | 0.0948955827 |
| Other stationary points or roots | `u = ±3π, ±5π, …`: feasible, `E = 9π², 25π², …` | `u = 0` (saddle, `J = 1`); `θ ≈ 6.548` (`J = 1.411`); `θ ≈ 9.054` (`J = 0.854`); their mirrors |

`u → −u` gives an equal-cost twin. Compare `J`, `F`, `E` and residuals, never the
sign of the control.

## Tests

| ID | Route | Checks | Pass when |
|---|---|---|---|
| PP1 | OLGS soft | Short Adam, diagnosis, curvature, switch to line search, candidate run | `\|J − J*\| ≤ 1e-8`, `std(u) ≤ 1e-6`; inspect calls and `activate=False` leave the head unchanged |
| PP2 | OLGS hard | Geometry probe, projected gradient | `\|Re c0\| ≤ 1e-8`, `\|E − π²\| ≤ 2e-7`, `std(u) ≤ 1e-6` |
| PP3 | OLSS hard | Seed, solve, geometry, multistart, select by energy | qualified, residual ≤ 1e-10, `\|z\| = π` within 1e-9, best `E = π²` |
| PP4 | OLSS soft | Multistart, select by soft cost | best `\|z\| = θ*` within 1e-9; the `z = 0` root qualifies but ranks lower |
| PP5 | Cross-family replay | OLSS root replayed in OLGS, hard and soft | mapped metrics differ by ≤ 1e-10 |
| PP6 | Traps, current behavior | Zero start, weak starts, wrong-branch roots | each trap gives the answer in the [trap table](#pp6-traps) |
| PP7 | Error paths | Codes in the [error table](#pp7-error-paths) | expected code; contexts, heads and bindings unchanged |
| PP8 | Interface parity | PP1–PP4 values through Python, the service, the terminal, later MCP | values agree within 1e-12 relative |

## Pass rules

- `maxiter` is a valid stop. Pass or fail depends on the metrics; the observed
  stop reason is recorded.
- A qualified shooting root is not an optimum. Selection compares the declared
  metric among qualified roots only.
- Run settings are test inputs, not tuning claims. The gradient with respect to
  raw amplitudes scales with `dt`; the scripts use Adam `step_size=0.05`, line
  search `step_size=50` and projected gradient `step_size=25`.
- PP6 checks current behavior. When a trap is fixed, for example a saddle stop
  reason, update its expected answer here and in the runner.

## Results, 2026-10-12

| Test | Python | Service | Terminal |
|---|---|---|---|
| Contract checks | pass | — | — |
| PP1 | `J = 0.0948955827442545` | same; head kept | same; head kept |
| PP2 | `E = 9.869604401089346`, `c0 = 9.9e-16` | same | same |
| PP3 | best `z = π`, `E = π²`; first seed qualifies | same; best root activates | same |
| PP4 | best `z = θ*`; `z = 0` ranked third | same | same; seed 3 root `J = 2.647` |
| PP5 | differences 3.9e-14 hard, 8.6e-15 soft | 5.0e-14 hard | needs a typed `times` list (finding 5) |
| PP6 | all five traps as expected | — | three traps as expected |
| PP7 | — | 7 of 7 codes | 9 of 9 cases, exit status 2 |
| PP8 | PP1–PP4 values identical across the three interfaces | | MCP pending |

## Findings

1. **A saddle is reported as converged.** Adam from `u = 0` stops with
   `gradient_converged` at `J = 1`; OLGS-007 is open. `curvature_probe` reports
   negative curvature there.
2. **Default step sizes barely move these controls.** L-BFGS stalls from
   `u = 0.5`, where the curvature is negative, and works after Adam. This is
   scaling (OLGS-010), not a tool defect.
3. **Projected gradient never stops on stationarity.** It reaches `E = π²` to
   1e-14 in 30 iterations. Continued to 300 iterations, it drifts to the
   feasibility edge: `|c0| = 1e-8`, `E = π² − 1.3e-7`.
4. **The terminal cannot read properties.** `ms.best` gives
   `invalid_reference`; use `ms.ranked[0]`.
5. **Some inputs need Python.**
   - The terminal and service cannot build a chosen shooting start such as
     `z0 = 2.5`: raw lists are refused, and the `ShootingVariables` constructor
     needs `spec`. Seeds come only from `random_shooting_seeds`.
   - Replay needs a typed 100-value `times` list.
6. **One error message leaks internal text.** Using OLSS variables in an OLGS
   context gives the right code, `incompatible_layout`, with the message
   `'ControlSpec' object has no attribute 'blocks'`.
7. **Failed terminal calls shift byte accounting.** After a failed
   `capability_missing` or `incompatible_layout` terminal call, `retained_bytes`
   drops by 32–48 bytes. Contexts, heads and bindings are unchanged. Through the
   service directly, the state stays identical.

## Details

### Run

To reproduce now, from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/Optimizer/bin/python -B test_contained/pi_pulse_samples/run_samples.py
```

`--json` prints every row. The runner calls the Python API and
`OperationService` directly. It runs the terminal as
`python -m optimizer.cli --json` subprocesses from this folder, so the relative
factory paths resolve. Exit status is 0 when no check fails.

To use one script interactively, start the terminal in this folder and paste
its commands, or run:

```bash
PYTHONPATH=../.. ~/miniconda3/envs/Optimizer/bin/python -B -m optimizer.cli --script scripts/pp1_gradient_soft.opt
```

### Files

| File | Content |
|---|---|
| [pi_pulse_olgs.py](pi_pulse_olgs.py) | `PiPulseSoft` and `PiPulseHard`; factories `soft` and `hard`. Forward and backward propagation with an exact gradient; the hard class adds `residuals` and `jacobian`. |
| [pi_pulse_olss.py](pi_pulse_olss.py) | `PiPulseShooting` (hard) and `PiPulseShootingSoft`; factories `hard` and `soft`. RK4 with variational equations; independent `verify` at 4× steps; `recover_controls`. |
| [scripts/](scripts/) | Terminal scripts for PP1–PP4 and PP6 |
| [run_samples.py](run_samples.py) | Every check through all three interfaces; prints the table |

### Contract checks

- `opt.require_system` passes for both OLGS factories. `verify_gradient` passes
  on a random smooth guess. `verify_jacobian` passes for the hard system.
- `require_shooting_system` passes for both OLSS factories.
  `verify_shooting_jacobian` at `z = 2.5` passes and is independent: errors
  1.9e-10 hard and 7.5e-9 soft.

### PP1 gradient, soft

```text
opt.session.load(name="g_soft", factory="./pi_pulse_olgs.py", factory_name="soft");
u0 = opt.guesses.constant_guess(value=0.5);
opt.session.use(value=u0);
r1 = opt.optimizers.adam(maxiter=100, step_size=0.05);
d = opt.utils.diagnose_run(result=r1);
c = opt.utils.curvature_probe();
r2 = opt.optimizers.line_search(maxiter=60, step_size=50.0);
trial = opt.optimizers.adam(maxiter=5, step_size=0.05, activate=False);
opt.evaluate();
```

Observed:

- Adam reaches `J = 0.0949726`, and the diagnosis is `converging`.
- Curvature at that point is positive.
- The line search reaches the answer and stops with `gradient_converged`.
- The final `opt.evaluate()` returns the line-search metrics, so the head did
  not move.

### PP2 gradient, hard

```text
opt.session.load(name="g_hard", factory="./pi_pulse_olgs.py", factory_name="hard");
u0 = opt.guesses.gaussian_guess(amplitude=7.0);
opt.session.use(value=u0);
g = opt.utils.geometry_probe();
r = opt.optimizers.projected_gradient(maxiter=30, step_size=25.0);
opt.evaluate();
```

The guess has `c0 = 0.265`, `E = 12.91` and rank 1.

A variant first runs `rep = opt.utils.repair_newton();` and
`opt.session.use(value=rep.controls);`. Repair reaches `F = 1` at `E = 15.99`.
Thirty iterations then end at `E = π² − 1e-7` with `c0 = 7.9e-9`, inside the
pass tolerance.

### PP3 shooting, hard

```text
opt.session.load(name="s_hard", factory="./pi_pulse_olss.py", factory_name="hard");
seeds = opt.shooting.random_shooting_seeds(count=16, seed=1, radius=10.0);
opt.session.use(value=seeds[0]);
root = opt.shooting.solve_shooting();
geo = opt.utils.shooting_geometry_probe();
ms = opt.shooting.solve_shooting_multistart(starts=seeds);
ms.ranked[0].metrics;
```

Observed:

- The geometry probe gives singular value 0.5.
- All 16 seeds qualify.
- The unique roots are `{±π, ±3π, 5π, −7π}`, and the best two are `±π`.
- Ranking uses energy on verification values.

### PP4 shooting, soft

```text
opt.session.load(name="s_soft", factory="./pi_pulse_olss.py", factory_name="soft");
seeds = opt.shooting.random_shooting_seeds(count=16, seed=1, radius=10.0);
ms = opt.shooting.solve_shooting_multistart(starts=seeds);
ms.ranked[0].metrics;
low = opt.shooting.solve_shooting(initial=seeds[3]);
```

The unique roots include `0`, `±6.548`, `9.054` and `−13.12`. With seed 1,
multistart does not find `−θ*`. The root reached from seed 3 qualifies with
`J = 2.647` and `F = 0.074`. A root is not an optimum.

### PP5 cross-family replay

The arguments are:

- `shooting_system="s_hard"` and `gradient_system="g_hard"`
- `times`: the 100 interval midpoints
- `metric_map={"energy": "energy", "fidelity": "fidelity"}`

The soft run maps `soft_cost` to `J`.

### PP6 traps

| Trap | Input | Expected and observed |
|---|---|---|
| Zero start | Soft, `constant_guess(value=0.0)`, Adam 10 iterations | `gradient_converged` at `J = 1`; minimum curvature −4.3e-5 |
| Weak hard start | Hard, `constant_guess(value=0.5)`, projected gradient 100 iterations | lands on `θ = 3π`, `E = 88.826`: feasible, not minimal |
| Weak shooting start | Hard, `z0 = 0.5` (Python only, finding 5) | qualified root `z = 3π` |
| Trivial soft root | Soft, `z0 = 0.5` (Python only) | qualified root `z = 0`, `J = 1` |

### PP7 error paths

| Case | Code |
|---|---|
| Adam in an OLSS context | `wrong_family` |
| `solve_shooting` in an OLGS context | `wrong_family` |
| Projected gradient on the soft system | `capability_missing` (hook `residuals`) |
| `adam(maxiterr=5)` | `invalid_arguments` |
| Loading an existing context name | `context_exists` |
| Service `session.use` with an unknown object ID | `missing_object` |
| OLSS seed used in an OLGS context | `incompatible_layout` |
| Terminal `import os;` | `unsupported_syntax` |
| Terminal `opt.optimizers.adm(...)` | `unknown_operation` |
| Terminal: two calls without a separating semicolon | `syntax_error` |

### Harder variants, not built

- Two channels (`ux`, `uy`) give the gauge circle `u = π(cos φ, sin φ)`. The
  OLSS Jacobian is 2×2 of rank 1, so `roots_equivalent` is needed, and checks
  compare invariants only.
- Detuning `Δ` as a secondary parameter, continued from 0 with
  `continue_parameter`. OLGS hard and OLSS hard must agree on `E`.
- An X-gate target.
- Multiple shooting of the hard case, a 3×3 system with a known root.

## Connections

- Parent: [Testing Hub](../../tests/TESTING_HUB.md)
- Related: [Terminal reference](../../optimizer/cli/TERMINAL.md)
- Related: [Foreground operation service](../../optimizer/operations/SERVICE.md)
- Related: [OLGS Contract](../../optimizer/system_olgs/CONTRACT.md), [OLSS Contract](../../optimizer/system_olss/CONTRACT.md)

---

[⌂ Home](../../README.md)
