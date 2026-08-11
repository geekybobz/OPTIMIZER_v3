---
title: Visualization
type: module_hub
module: optimizer/visualization
---

# Visualization

Visualization is split into system-owned data and reusable rendering.

```text
controls
  -> system.plot_data(controls, kind=...)
  -> PlotData(kind, named layers, annotations, metadata)
  -> explicit renderer
```

`PlotLayer` has a name, generic coordinate arrays, an optional label, style hints,
and metadata. Layers have no built-in scientific meaning; the project system or
calling notebook chooses their names and interpretation.

The Bloch renderer expects each layer to contain `points` with shape `(n, 3)` and
an optional one-dimensional `parameter`. It validates that points lie inside the
Bloch ball, then draws the shared light wireframe, named trajectories, annotations,
and optional Cartesian projections. It contains no Hamiltonian, propagation, or
problem-specific comparison model.

```python
data = opt.plot_data(system, controls, kind="bloch")
with opt.visualization.style_context():
    fig, sphere, projections = opt.visualization.plot_bloch(data)
```

Plot-data calls are explicit and may be expensive. Optimizers and passive tracking
do not invoke them automatically.
