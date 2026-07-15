# Public API

Status: review draft.
Last updated: 2026-07-15.

## Main Principle

The preferred public API should be namespace-first:

```python
opt.optimizers.method_name(...)
opt.utils.method_name(...)
opt.guesses.method_name(...)
```

This keeps the call readable when the library grows: an optimizer moves controls, a
utility diagnoses or repairs, and a guess function creates initial controls. Direct
shortcuts such as ``opt.adam(...)`` stay available for compact notebooks.

## Preferred Explicit Style

```python
import optimizer as opt

sys = system(params)
controls = opt.guesses.fourier_guess(sys, modes=8, amplitude=0.03)

result = opt.optimizers.adam(sys, controls, maxiter=1000)
result = opt.optimizers.line_search(sys, result.controls, maxiter=10, warmstart=result.warmstart())

fixed = opt.utils.repair_newton(sys, result.controls)
diag = opt.utils.geometry_probe(sys, fixed.controls)
```

This is safest for scripts and multiprocessing because the system is explicit.

## Bound-System Context Style

This is supported for notebook and curriculum convenience:

```python
import optimizer as opt

ctx = opt.context(system(params))
controls = ctx.fourier_guess(n_terms=8, amplitude=0.03)
result = ctx.adam(controls, maxiter=1000)

ctx = ctx.with_params(lambda2=1e6, lambda4=10.0)
result = ctx.adam(result.controls, maxiter=10, warmstart=True)
```

This is convenient, but it should not be the only supported style. Avoid a mutable
global like `opt.system = system`; it is weaker for multiprocessing, multiple systems,
and reproducibility.

## Namespaced Functions

Optimizers:

```python
opt.optimizers.adam(...)
opt.optimizers.momentum(...)
opt.optimizers.line_search(...)
opt.optimizers.adagrad(...)
opt.optimizers.rmsprop(...)
opt.optimizers.lbfgs(...)
opt.optimizers.nonlinear_cg(...)
opt.optimizers.cma_es(...)
```

Guesses:

```python
opt.guesses.zero_guess(system)
opt.guesses.constant_guess(system, value=0.1)
opt.guesses.sine_guess(system, amplitude=0.05)
opt.guesses.sinc_guess(system, amplitude=0.05)
opt.guesses.fourier_guess(system, modes=8, amplitude=0.03)
opt.guesses.random_guess(system, amplitude=0.01)
opt.guesses.mix_guess([a, b], weights=[0.8, 0.2])
```

Repairs and diagnostics:

```python
opt.utils.repair_newton(system, controls)
opt.utils.geometry_probe(system, controls)
opt.utils.verify_gradient(system, controls)
opt.utils.verify_jacobian(system, controls)
opt.utils.control_spectrum(controls)
```

The singular alias ``opt.util`` points to the same namespace as ``opt.utils`` for
notebook convenience.

## Direct Shortcuts

These remain supported and delegate to the same implementations:

```python
opt.adam(...)
opt.fourier_guess(...)
opt.repair_newton(...)
opt.geometry_probe(...)
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
