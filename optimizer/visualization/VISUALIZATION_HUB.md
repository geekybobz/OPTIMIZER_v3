---
role: front-door
parent: ../../README.md
summary: "System-owned plot data and reusable renderers for Bloch, projection, and control panels."
read_when: "Read when plotting a system's data."
status: active
reviewed: 2026-10-12
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

## Standalone single-panel figures

For a verification figure made of several independently sized/positioned panels
(e.g. controls | Bloch-component projections | Bloch sphere, arranged in a document
by the caller rather than baked into one composite), use the one-figure-per-panel
functions in `optimizer.visualization.panels` instead of `plot_bloch`'s combined
figure: `plot_bloch_panel` (sphere only), `plot_projection_panel` (Bloch components
vs. parameter, same palette and Bloch-ball validation as `plot_bloch`), and
`plot_controls_panel` (control amplitudes vs. parameter — a different, established
palette, and no Bloch-ball check, since controls are not unit vectors). `save_panel`
writes a finished panel figure as both `.pdf` and `.png` from one path stem.

```python
fig, sphere = opt.visualization.plot_bloch_panel(data)
opt.visualization.save_panel(fig, "out/trajectory_bloch")
```

Trajectories colored by parameter (`color_by="parameter"`, the default whenever a
layer has a `parameter`) are rasterized by default before saving, since a `Line3DCollection`
exported as pure vector PDF renders inconsistently across PDF viewers (confirmed:
poppler/Ghostscript vs. macOS Preview/Quartz can disagree on the same file). Opt out
per layer with `style={"rasterized": False}` if pure vector output is required.

---

[⌂ Home](../../README.md)
