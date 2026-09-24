"""Modular Bloch-sphere rendering for generic named plot-data layers.

This module owns presentation only.  A project system owns propagation, basis
conventions, and conversion of its state representation to Bloch coordinates.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

from optimizer.visualization.models import PlotData, coerce_plot_data


DEFAULT_ELEVATION = 14.0
DEFAULT_AZIMUTH = -54.0
COMPONENT_COLORS = ("#0072B2", "#E69F00", "#009E73")
LAYER_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")


def _matplotlib() -> tuple[Any, Any, Any]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise ImportError(
            "Visualization rendering requires matplotlib; install optimizer-v4[scientific]."
        ) from exc
    return plt, Line2D, Line3DCollection


@contextmanager
def style_context() -> Iterator[None]:
    """Apply the compact serif plotting style without mutating global settings."""

    plt, _line2d, _collection = _matplotlib()
    style = {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.45,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.bbox": "tight",
    }
    with plt.rc_context(style):
        yield


def validate_bloch_data(value: PlotData | dict[str, Any], *, tolerance: float = 1.0e-8) -> PlotData:
    """Validate generic plot data for rendering inside the Bloch ball."""

    data = coerce_plot_data(value)
    if data.kind != "bloch":
        raise ValueError(f"Bloch renderer requires plot kind 'bloch', got {data.kind!r}.")
    for layer in data.layers:
        if "points" not in layer.coordinates:
            raise KeyError(f"Bloch layer {layer.name!r} must provide coordinates['points'].")
        points = np.asarray(layer.coordinates["points"], dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1:
            raise ValueError(f"Bloch layer {layer.name!r} points must have shape (n, 3).")
        if np.any(np.linalg.norm(points, axis=1) > 1.0 + float(tolerance)):
            raise ValueError(f"Bloch layer {layer.name!r} contains points outside the Bloch ball.")
        parameter = layer.coordinates.get("parameter")
        if parameter is not None and np.asarray(parameter).shape != (points.shape[0],):
            raise ValueError(
                f"Bloch layer {layer.name!r} parameter must have shape ({points.shape[0]},)."
            )
    return data


def draw_sphere(
    ax: Any,
    *,
    equator: bool = True,
    elevation: float = DEFAULT_ELEVATION,
    azimuth: float = DEFAULT_AZIMUTH,
    nu: int = 25,
    nv: int = 13,
) -> Any:
    """Draw a light unit-sphere scaffold on an existing 3D axis."""

    uu, vv = np.meshgrid(np.linspace(0.0, 2.0 * np.pi, int(nu)), np.linspace(0.0, np.pi, int(nv)))
    ax.plot_wireframe(
        np.cos(uu) * np.sin(vv),
        np.sin(uu) * np.sin(vv),
        np.cos(vv),
        color="#cccccc",
        lw=0.28,
        alpha=0.5,
    )
    if equator:
        angle = np.linspace(0.0, 2.0 * np.pi, 120)
        ax.plot(np.cos(angle), np.sin(angle), np.zeros_like(angle), color="#999999", lw=0.9, ls=(0, (4, 3)))
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.view_init(elev=float(elevation), azim=float(azimuth))
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.set_zlim(-1.02, 1.02)
    return ax


def _draw_layer(ax: Any, layer: Any, index: int) -> Any:
    plt, _line2d, Line3DCollection = _matplotlib()
    points = np.asarray(layer.coordinates["points"], dtype=float)
    style = dict(layer.style)
    linewidth = float(style.pop("linewidth", 1.8))
    parameter = layer.coordinates.get("parameter")
    color_by = style.pop("color_by", "parameter" if parameter is not None else "constant")
    if color_by == "parameter" and parameter is not None and len(points) > 1:
        parameter_values = np.asarray(parameter, dtype=float)
        parameter_min = float(np.min(parameter_values))
        parameter_max = float(np.max(parameter_values))
        if np.isclose(parameter_min, parameter_max):
            parameter_max = parameter_min + 1.0
        segments = np.concatenate((points[:-1, None, :], points[1:, None, :]), axis=1)
        # Rasterized by default: mplot3d exports a Line3DCollection to vector
        # PDF as many small overlapping path patches, and different PDF
        # renderers z-sort those patches differently -- the same trajectory
        # can render as a clean arc in one viewer and show a spurious gap in
        # another (poppler/Ghostscript vs. macOS Preview/Quartz, confirmed).
        # Rasterizing removes that fragile vector code path entirely; a
        # caller that genuinely wants pure vector output can opt back out
        # via style={"rasterized": False}.
        rasterized = bool(style.pop("rasterized", True))
        collection = Line3DCollection(
            segments,
            cmap=style.pop("cmap", "Blues"),
            lw=linewidth,
            norm=plt.Normalize(parameter_min, parameter_max),
            **style,
        )
        collection.set_array(parameter_values[:-1])
        collection.set_rasterized(rasterized)
        ax.add_collection3d(collection)
        return collection
    color = style.pop("color", LAYER_COLORS[index % len(LAYER_COLORS)])
    return ax.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color=color,
        lw=linewidth,
        label=layer.label or layer.name,
        **style,
    )[0]


def _draw_annotations(ax: Any, annotations: dict[str, Any]) -> None:
    for name, raw in annotations.items():
        if not isinstance(raw, dict) or "point" not in raw:
            continue
        point = np.asarray(raw["point"], dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError(f"Bloch annotation {name!r} point must be a finite shape-(3,) vector.")
        color = str(raw.get("color", "#222222"))
        ax.plot([point[0]], [point[1]], [point[2]], "o", ms=float(raw.get("markersize", 4.5)), color=color)
        label = raw.get("label", name)
        if label:
            ax.text(point[0], point[1], point[2] + float(raw.get("z_offset", 0.12)), str(label), ha="center")


def draw_timeseries(
    ax: Any,
    data: PlotData,
    *,
    palette: Sequence[str] = COMPONENT_COLORS,
    component_labels: Sequence[str] = ("x", "y", "z"),
    ylabel: str = "Bloch components",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = (-1.14, 1.14),
    prefix_labels: bool = True,
    legend_ncols: int = 2,
) -> None:
    """Draw named per-component channels vs. a scalar parameter on a 2D axis.

    Generalizes the panel this module has always drawn as the right-hand
    side of ``plot_bloch(..., projections=True)`` into a public, reusable
    primitive: ``plot_bloch`` itself now calls this with its historical
    defaults, and ``optimizer.visualization.panels`` builds standalone
    single-panel figures (Bloch-component or control-amplitude) on top of
    it with a different palette/validation per panel kind.

    Components beyond ``min(points.shape[1], len(palette), len(component_labels))``
    are not drawn.
    """

    line_styles = ("-", "--", ":", "-.")
    for layer_index, layer in enumerate(data.layers):
        points = np.asarray(layer.coordinates["points"], dtype=float)
        parameter = layer.coordinates.get("parameter")
        x = np.arange(len(points), dtype=float) if parameter is None else np.asarray(parameter, dtype=float)
        n_components = min(points.shape[1], len(palette), len(component_labels))
        for component in range(n_components):
            label = (
                f"{layer.label or layer.name}:{component_labels[component]}"
                if prefix_labels
                else component_labels[component]
            )
            ax.plot(
                x,
                points[:, component],
                color=palette[component],
                ls=line_styles[layer_index % len(line_styles)],
                lw=1.4,
                label=label,
            )
    ax.axhline(0.0, color="#bbbbbb", lw=0.6, zorder=0)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(str(data.metadata.get("parameter_label", "parameter")))
    ax.set_ylabel(ylabel)
    ax.legend(ncols=legend_ncols, loc="best")


def plot_bloch(
    value: PlotData | dict[str, Any],
    *,
    projections: bool = True,
    equator: bool = True,
    elevation: float = DEFAULT_ELEVATION,
    azimuth: float = DEFAULT_AZIMUTH,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, ...]:
    """Render all generic layers and return the figure plus created axes."""

    data = validate_bloch_data(value)
    plt, _line2d, _collection = _matplotlib()
    if projections:
        figure = plt.figure(figsize=figsize or (6.4, 3.0))
        sphere_axis = figure.add_subplot(1, 2, 1, projection="3d")
        projection_axis = figure.add_subplot(1, 2, 2)
    else:
        figure = plt.figure(figsize=figsize or (3.3, 3.3))
        sphere_axis = figure.add_subplot(projection="3d")
        projection_axis = None
    draw_sphere(
        sphere_axis,
        equator=equator,
        elevation=elevation,
        azimuth=azimuth,
    )
    for index, layer in enumerate(data.layers):
        _draw_layer(sphere_axis, layer, index)
    _draw_annotations(sphere_axis, dict(data.annotations))
    if projection_axis is not None:
        draw_timeseries(projection_axis, data)
        return figure, sphere_axis, projection_axis
    return figure, sphere_axis


def save_figure(figure: Any, path: Any, *, dpi: int = 400, close: bool = False) -> None:
    """Save a rendered figure; closing remains an explicit caller choice."""

    plt, _line2d, _collection = _matplotlib()
    figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    if close:
        plt.close(figure)
