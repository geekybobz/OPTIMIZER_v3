"""Numerical coordinate adapter x = S z; project hooks always receive x."""
from __future__ import annotations
import copy
from dataclasses import replace
import numpy as np
from optimizer.controls import ControlSpec, Controls
from optimizer.system_olgs import ControlBounds, ValueGradient


class CoordinateSystem:
    """Private change of coordinates, never a shared physics model."""
    def __init__(self, system):
        self.original = system
        self.physical_spec = system.control_spec()
        self.scale = np.repeat(self.physical_spec.scales, self.physical_spec.control_dim)
        self.spec = ControlSpec(self.physical_spec.keys, self.physical_spec.control_dim,
                                dtype=self.physical_spec.dtype, dt=self.physical_spec.dt,
                                meta=self.physical_spec.meta)

    def physical(self, value):
        return Controls.from_flat(self.physical_spec, value.flatten(copy=False) * self.scale,
                                  name=value.name, meta=value.meta)

    def normalized(self, value):
        return Controls.from_flat(self.spec, value.flatten(copy=False) / self.scale,
                                  name=value.name, meta=value.meta)

    def derivative(self, value):
        return Controls.from_flat(self.spec, value.flatten(copy=False) * self.scale, name=value.name)

    def control_spec(self):
        return self.spec

    def system_spec(self):
        payload = copy.deepcopy(self.original.system_spec())
        payload["controls"].update(self.spec.to_dict())
        return payload

    def evaluate(self, controls):
        return self.original.evaluate(self.physical(controls))

    def gradient(self, controls):
        return self.derivative(self.original.gradient(self.physical(controls)))

    def __getattr__(self, name):
        hook = getattr(self.original, name)
        if name == "value_and_gradient":
            def combined(controls):
                pair = hook(self.physical(controls))
                if isinstance(pair, ValueGradient):
                    return ValueGradient(pair.metrics, self.derivative(pair.gradient))
                if isinstance(pair, tuple):
                    return pair[0], self.derivative(pair[1])
                return dict(pair, gradient=self.derivative(pair["gradient"]))
            return combined
        if name == "smooth_gradient":
            return lambda controls: self.derivative(hook(self.physical(controls)))
        if name == "hvp":
            return lambda controls, vector: self.derivative(hook(self.physical(controls), self.physical(vector)))
        if name == "jacobian":
            return lambda controls, **kw: hook(self.physical(controls), **kw) * self.scale
        if name == "residuals":
            return lambda controls, **kw: hook(self.physical(controls), **kw)
        if name == "control_bounds":
            def bounds():
                bound = hook()
                if isinstance(bound, ControlBounds):
                    return ControlBounds(self.normalized(bound.lower), self.normalized(bound.upper))
                lower, upper = bound
                lower = lower if isinstance(lower, Controls) else Controls.from_matrix(self.physical_spec, lower)
                upper = upper if isinstance(upper, Controls) else Controls.from_matrix(self.physical_spec, upper)
                return ControlBounds(self.normalized(lower), self.normalized(upper))
            return bounds
        if name == "prox":
            # The public prox hook defines a Euclidean physical-coordinate map.
            # Proximal methods explicitly keep those coordinates; no weighted
            # proximal operator can be inferred from this hook.
            raise AttributeError("Scaled coordinates require a weighted prox; use coordinate_mode='physical'.")
        return hook

    def state(self, state, *, physical=False):
        if state is None:
            return None
        convert = self.physical if physical else self.normalized
        return replace(state, controls=convert(state.controls),
                       best_controls=None if state.best_controls is None else convert(state.best_controls),
                       optimizer_state=copy.deepcopy(state.optimizer_state))


def run_scaled(run, system, controls, options):
    adapter = CoordinateSystem(system)
    options = dict(options)
    options["state"] = adapter.state(options.get("state"))
    warm = options.get("warmstart")
    if warm is not None:
        options["warmstart"] = replace(warm, controls=adapter.normalized(warm.controls))
    options["coordinate_mode"] = "physical"
    result = run(adapter, None if controls is None else adapter.normalized(controls), **options)
    result.controls = adapter.physical(result.controls)
    result.state = adapter.state(result.state, physical=True)
    if result.run_record is not None:
        result.run_record.summary["coordinates"] = "scaled"
    return result
