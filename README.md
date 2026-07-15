# OPTIMIZER v3

OPTIMIZER v3 is a namespace-first optimization toolbox for system-defined control
problems. A project `system.py` owns the physics, metrics, objective value,
analytical gradient, residuals, Jacobians, and parameter weights. The library provides
the reusable engineering around that system: vectorized controls, optimizer loops,
guess generators, diagnostics, repairs, schedules, trace records, checkpoints, and
result objects.

## Install

Create or update the standard conda environment and install this repo as an editable
library:

```bash
./install.sh
conda activate optimizer_v3
```

After activation, this works from any folder:

```python
import optimizer as opt
```

Editable install means source edits in this checkout are visible after restarting
Python. The installer records the live paths in:

```text
~/.optimizer_v3/install.json
```

That state file stores the repo root, conda environment, Python executable, package
name, environment file, and git commit. `uninstall.sh` reads it so removal does not
depend on remembered paths.

Useful commands:

```bash
./update.sh --check
./update.sh
./uninstall.sh
./uninstall.sh --remove-env
```

## Public API

Preferred style groups functions by role:

```python
import optimizer as opt

sys = system(params)
controls = opt.guesses.fourier_guess(sys, modes=6, amplitude=0.03)

r1 = opt.optimizers.adam(sys, controls, maxiter=50)
r2 = opt.optimizers.lbfgs(sys, r1.controls, maxiter=20, warmstart=r1.warmstart())

diag = opt.utils.geometry_probe(sys, r2.controls)
fixed = opt.utils.repair_newton(sys, r2.controls, residuals="hard")
```

Compact aliases such as `opt.adam(...)`, `opt.fourier_guess(...)`, and
`opt.repair_newton(...)` remain available for notebooks.

## Current Modules

```text
optimizer/
  controls.py          vectorized ControlSpec and Controls
  system.py            optimizer-facing system contract helpers
  result.py            Evaluation and OptimizerResult
  state.py             RunState and WarmStartState
  library.py           root facade and bound-system context
  core/                shared evaluation, stopping, guards, engine, parallel helpers
  optimizers/          Adam, momentum, line search, AdaGrad, RMSProp, L-BFGS, NCG, CMA-ES
  guesses/             simple, harmonic, random, and composite initial controls
  utils/               diagnostics, derivatives, geometry, repair, spectrum tools
  schedules/           step-size policies
  logs/                in-memory trace records and checkpoints
```

Local project adapters can live under `systems/` for testing or examples. Private
reference artifacts under `systems/*/reference/` are ignored by git.

## System Contract

A project system should provide:

```text
control_spec()
evaluate(controls) -> metrics dict with scalar J
gradient(controls) -> Controls
```

Optional but important for constrained work:

```text
residuals(controls, name=...)
jacobian(controls, name=...)
with_params(...)
metric_schema()
```

The system owns metric names and objective weights. The optimizer does not assemble a
physics objective externally.

## Testing

Run the full suite:

```bash
python -m unittest discover -s tests
```

Inside the standard environment:

```bash
conda run -n optimizer_v3 python -m unittest discover -s tests
```

