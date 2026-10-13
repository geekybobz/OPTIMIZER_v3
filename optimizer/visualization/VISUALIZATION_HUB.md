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
OLGS Controls / OLSS ShootingVariables
  -> opt.plot_data(system, candidate, kind=...)
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

## Reviewed Bloch rendering

`plot_bloch` accepts PlotData, its mapping, or explicit real `(N, 3)` coordinates
with `parameter=...`. Mixed-state points stay inside the sphere; no normalization
or physical conversion is performed. Complex, object/string and non-finite
coordinates are rejected. The system owns the basis/reference frame and trajectory.

```python
data = opt.plot_data(system, candidate, kind="bloch")
fig, sphere = opt.visualization.plot_bloch(
    data, projections=False, appearance="paper", azimuth=36, elevation=14,
    cmap="viridis_r", axes=True, arrows=2,
)
receipt = opt.visualization.render_bloch(
    data, "figures/trajectory", options={"azimuth": 36, "axes": True, "arrows": 2}
)
```

Paper appearance uses orthographic projection, light front/back sphere guides,
shared scalar normalization, endpoint markers and optional direction arrows.
Azimuth is a camera angle in degrees about fixed Z; finite arbitrary angles are
accepted. Calling the returned axis's `view_init` updates guides/annotation positions
without changing data or propagating physics. Legacy appearance is the default
for existing figure calls. Per-layer styles remain available; explicit call options
override them. A paper shared colorbar requires compatible coloring fields, colormaps
and declared units. `color_range` sets an explicit shared range. Generic parameter
labels come from metadata; normalized time must be explicitly supplied/labeled.

Annotations can distinguish desired target from actual endpoint. Paper labels accept
`offset_points` in screen points. Automatic endpoint markers never invent a target.
One camera view cannot guarantee that every crossing is visible.

`render_bloch` defaults to paper appearance and sphere-only PNG/PDF export. It accepts
an explicit output stem, a bounded DPI and native renderer `options`, returning
`optimizer.bloch-render/1` with artifact paths/hashes, input hash and effective settings.
It closes its figure, refuses existing outputs and does no numerical work. Rendering
is an explicit filesystem effect; a later history failure/lost reply does not erase
completed files. Inspect files/history before retrying. Save the receipt as an existing
report checkpoint when reproducibility is required; no second log is introduced.
CLI/MCP expose `visualization.render_bloch`; live figures/axes remain Python-only.
The skill reviews data/cost/style/camera/output choices before generating plots.

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
