# Public API

Status: review draft.
Last updated: 2026-07-15.

## Main Principle

The public API should be direct:

```python
opt.method_name(...)
```

Avoid forcing users through nested wrappers for common work.

## Preferred Explicit Style

```python
import optimizer as opt

sys = system(params)
controls = opt.fourier_guess(sys, n_terms=8, amplitude=0.03)

result = opt.adam(sys, controls, maxiter=1000)
result = opt.line_search(sys, result, maxiter=10, warmstart=True)

fixed = opt.repair_newton(sys, result.controls)
diag = opt.geometry_probe(sys, fixed.controls)
```

This is safest for scripts and multiprocessing because the system is explicit.

## Optional Bound-System Style

This can be supported later for notebook convenience:

```python
import optimizer as opt

opt.system = system(params)
controls = opt.fourier_guess(n_terms=8, amplitude=0.03)
result = opt.adam(controls, maxiter=1000)
```

This is convenient, but it should not be the only supported style.

## Direct Functions

Optimizers:

```python
opt.adam(...)
opt.momentum(...)
opt.line_search(...)
opt.lbfgs(...)
opt.projected_descent(...)
opt.levenberg_marquardt(...)
```

Guesses:

```python
opt.zero_guess(system)
opt.constant_guess(system, amplitude=0.1)
opt.sine_guess(system, amplitude=0.05)
opt.sinc_guess(system, amplitude=0.05)
opt.fourier_guess(system, n_terms=8, amplitude=0.03)
opt.random_guess(system, amplitude=0.01)
opt.mix_controls(system, a, b, ratio=0.2)
```

Repairs and diagnostics:

```python
opt.repair_newton(system, controls)
opt.geometry_probe(system, controls)
opt.verify_derivatives(system, controls)
opt.control_spectrum(system, controls)
```

Trace and checkpoints:

```python
trace = opt.Trace("run_001")
trace.checkpoint("stage_start", system, controls, state)
controls, state = trace.restore("stage_start")
```

## Result Access

`OptimizerResult` should support attribute access and dict-like export:

```python
result.controls
result.metrics["J"]
result.metrics["fidelity"]
result.state
result.trace

payload = result.to_dict()
```

For v2 compatibility, important fields can also appear in exported dict form:

```text
controls
J
score
metrics
iterations
stop_reason
J_hist
metrics_hist
control_info
```

