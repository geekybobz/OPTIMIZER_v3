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


def _real_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite real numbers (no complex, object or string coordinates).")
    return array.astype(float, copy=False)


def validate_bloch_data(value: PlotData | dict[str, Any], *, tolerance: float = 1.0e-8) -> PlotData:
    """Validate generic plot data for rendering inside the Bloch ball."""

    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Bloch radius tolerance must be finite and nonnegative.")
    data = coerce_plot_data(value)
    if data.kind != "bloch":
        raise ValueError(f"Bloch renderer requires plot kind 'bloch', got {data.kind!r}.")
    for layer in data.layers:
        if "points" not in layer.coordinates:
            raise KeyError(f"Bloch layer {layer.name!r} must provide coordinates['points'].")
        points = _real_array(layer.coordinates["points"], f"Bloch layer {layer.name!r} points")
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1:
            raise ValueError(f"Bloch layer {layer.name!r} points must have shape (n, 3).")
        if np.any(np.linalg.norm(points, axis=1) > 1.0 + float(tolerance)):
            raise ValueError(f"Bloch layer {layer.name!r} contains points outside the Bloch ball.")
        parameter = layer.coordinates.get("parameter")
        if parameter is not None and _real_array(parameter, "parameter").shape != (points.shape[0],):
            raise ValueError(
                f"Bloch layer {layer.name!r} parameter must have shape ({points.shape[0]},)."
            )
    for name, annotation in data.annotations.items():
        if isinstance(annotation, dict) and "point" in annotation:
            point = _real_array(annotation["point"], f"annotation {name!r}")
            if point.shape != (3,) or np.linalg.norm(point) > 1 + tolerance:
                raise ValueError("Bloch annotation points must be shape-(3,) vectors inside the Bloch ball.")
    return data


DEFAULT_AXES_COLOR = "#555555"


def _draw_reference_axes(
    ax: Any, *, inner: float = 1.0, outer: float = 1.32, color: str = DEFAULT_AXES_COLOR, fontsize: float = 9
) -> None:
    """Small x/y/z stub indicators so orientation stays legible under any
    rotation (view_init elevation/azimuth) -- deliberately short and
    unobtrusive rather than full axis lines through the sphere."""

    for label, (dx, dy, dz) in (("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1))):
        ax.plot(
            [inner * dx, outer * dx], [inner * dy, outer * dy], [inner * dz, outer * dz],
            color=color, lw=1.1, zorder=8,
        )
        ax.text(
            outer * dx * 1.12, outer * dy * 1.12, outer * dz * 1.12, label,
            color=color, fontsize=fontsize, style="italic", ha="center", va="center", zorder=8,
        )


def draw_sphere(
    ax: Any,
    *,
    equator: bool = True,
    elevation: float = DEFAULT_ELEVATION,
    azimuth: float = DEFAULT_AZIMUTH,
    nu: int = 25,
    nv: int = 13,
    axes: bool = False,
    axes_color: str = DEFAULT_AXES_COLOR,
) -> Any:
    """Draw a light unit-sphere scaffold on an existing 3D axis.

    axes=True adds small x/y/z reference-axis stubs (see
    _draw_reference_axes) -- useful whenever the view is rotated away from
    the default elevation/azimuth and orientation would otherwise be
    ambiguous. Off by default to keep existing figures unchanged.
    """

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
    if axes:
        _draw_reference_axes(ax, color=axes_color)
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


def _paper_guides(ax: Any, *, equator: bool, axes: bool) -> None:
    """A light scaffold whose front/back guides follow subsequent camera changes."""
    theta = np.linspace(0, 2 * np.pi, 241)
    circles = []
    for z in (-0.5, 0.0, 0.5):
        if z == 0 and not equator:
            continue
        radius = np.sqrt(1 - z * z)
        circles.append(np.column_stack((radius * np.cos(theta), radius * np.sin(theta), np.full_like(theta, z))))
    for phi in (0.0, np.pi / 2):
        circles.append(np.column_stack((np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta))))
    artists = [(ax.plot([], [], [], color="#aeb6bf", lw=.55, alpha=.7)[0],
                ax.plot([], [], [], color="#aeb6bf", lw=.55, alpha=.35, ls=(0, (2, 3)))[0])
               for _ in circles]
    outline = ax.plot([], [], [], color="#bdc4cc", lw=.7)[0]
    if axes:
        _draw_reference_axes(ax, inner=1.03, outer=1.18)
    previous = [None]
    def update(event=None):
        view = (ax.azim, ax.elev, ax.roll)
        if view == previous[0]:
            return
        previous[0] = view
        az, el = np.deg2rad([ax.azim, ax.elev])
        camera = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        for circle, (front_artist, back_artist) in zip(circles, artists):
            front = circle @ camera >= 0
            for mask, artist in ((front, front_artist), (~front, back_artist)):
                artist.set_data_3d(*np.where(mask[:, None], circle, np.nan).T)
        horizontal = np.array([-np.sin(az), np.cos(az), 0.0])
        vertical = np.cross(camera, horizontal)
        ring = np.cos(theta)[:, None] * horizontal + np.sin(theta)[:, None] * vertical
        outline.set_data_3d(*ring.T)
        if event is not None:
            ax.figure.canvas.draw_idle()
    update()
    ax.figure.canvas.mpl_connect("draw_event", update)


def _paper_annotations(ax: Any, annotations: dict[str, Any]) -> None:
    from mpl_toolkits.mplot3d import proj3d
    entries = []
    for name, raw in annotations.items():
        if not isinstance(raw, dict) or "point" not in raw:
            continue
        point = _real_array(raw["point"], f"annotation {name}")
        color = str(raw.get("color", "#222222"))
        ax.scatter(*point, color=color, s=float(raw.get("markersize", 4.5)) ** 2,
                   marker=raw.get("marker", "o"), depthshade=False)
        label = raw.get("label", name)
        if label:
            offset = raw.get("offset_points", (0, 9))
            if len(offset) != 2 or not np.all(np.isfinite(offset)):
                raise ValueError("annotation offset_points must be two finite numbers.")
            text = ax.annotate(str(label), (0, 0), xytext=offset, textcoords="offset points", ha="center")
            entries.append((point, text))
    previous = [None]
    def update(event=None):
        matrix = ax.get_proj()
        if previous[0] is not None and np.array_equal(matrix, previous[0]):
            return
        previous[0] = matrix.copy()
        for point, text in entries:
            x, y, _z = proj3d.proj_transform(*point, matrix)
            text.xy = (x, y)
        if event is not None:
            ax.figure.canvas.draw_idle()
    update()
    ax.figure.canvas.mpl_connect("draw_event", update)


def plot_bloch(
    value: Any,
    *,
    parameter: Any = None,
    projections: bool = True,
    equator: bool = True,
    elevation: float = DEFAULT_ELEVATION,
    azimuth: float = DEFAULT_AZIMUTH,
    figsize: tuple[float, float] | None = None,
    colorbar: bool = True,
    axes: bool = False,
    appearance: str = "legacy",
    cmap: str | None = None,
    color_by: str | None = None,
    color_range: tuple[float, float] | None = None,
    linewidth: float | None = None,
    markers: bool | None = None,
    arrows: int = 0,
) -> tuple[Any, ...]:
    """Render PlotData, its mapping, or an explicit real (N,3) Bloch trajectory.

    Camera angles are degrees. The paper appearance uses orthographic projection,
    a light scaffold, shared scalar normalization and optional direction arrows.
    Rendering never calls a system. Existing calls retain the legacy appearance.
    """
    from collections.abc import Mapping
    if not isinstance(value, (PlotData, Mapping)):
        coordinates = {"points": _real_array(value, "trajectory")}
        if parameter is not None:
            coordinates["parameter"] = _real_array(parameter, "parameter")
        from optimizer.visualization.models import PlotLayer
        value = PlotData(kind="bloch", layers=(PlotLayer("trajectory", coordinates),))
    elif parameter is not None:
        raise ValueError("parameter belongs to each PlotData layer; do not supply it twice.")
    data = validate_bloch_data(value)
    elevation, azimuth = float(elevation), float(azimuth)
    if not np.isfinite(elevation) or not np.isfinite(azimuth):
        raise ValueError("Camera angles must be finite degrees.")
    if appearance not in {"legacy", "paper"}:
        raise ValueError("appearance must be 'legacy' or 'paper'.")
    if type(arrows) is not int or not 0 <= arrows <= 50:
        raise ValueError("arrows must be an integer from 0 to 50 per trajectory.")
    if linewidth is not None and (not np.isfinite(linewidth) or linewidth <= 0):
        raise ValueError("linewidth must be finite and positive.")
    for name, flag in (("projections", projections), ("equator", equator), ("colorbar", colorbar), ("axes", axes)):
        if type(flag) is not bool:
            raise TypeError(f"{name} must be bool.")
    if markers is not None and type(markers) is not bool:
        raise TypeError("markers must be bool or None.")
    if color_range is not None:
        bounds = _real_array(color_range, "color_range")
        if bounds.shape != (2,) or bounds[0] >= bounds[1]:
            raise ValueError("color_range must contain increasing finite lower/upper values.")
    plt, _line2d, Line3DCollection = _matplotlib()
    if cmap is not None:
        plt.get_cmap(cmap)  # Validate before allocating a figure.
    prepared = []
    groups = {}
    for layer in data.layers:
        style = dict(layer.style)
        key = color_by or style.get("color_by", "parameter" if "parameter" in layer.coordinates else "constant")
        palette = cmap or style.get("cmap", "viridis_r" if appearance == "paper" else "Blues")
        scalar = None
        if key != "constant":
            if key not in layer.coordinates:
                raise ValueError(f"Layer {layer.name!r} has no coloring coordinate {key!r}.")
            scalar = _real_array(layer.coordinates[key], f"color coordinate {key!r}")
            if scalar.shape != (len(layer.coordinates["points"]),):
                raise ValueError("Color coordinates must have one scalar per trajectory point.")
            plt.get_cmap(palette)
            identity = (key, palette, layer.metadata.get("parameter_unit"))
            groups.setdefault(identity, []).append(scalar)
        prepared.append((layer, style, key, palette, scalar))
    if colorbar and appearance == "paper" and len(groups) > 1:
        raise ValueError("A shared colorbar requires the same coloring coordinate, colormap and units; use colorbar=False or compatible layers.")
    ranges = {}
    for identity, scalars in groups.items():
        low, high = color_range or (min(float(np.min(v)) for v in scalars), max(float(np.max(v)) for v in scalars))
        if low == high:
            high = low + 1.0
        ranges[identity] = (low, high)
    figure = plt.figure(figsize=figsize or ((6.4, 3.0) if projections else ((4.0, 3.3) if colorbar else (3.3, 3.3))))
    try:
        sphere_axis = figure.add_subplot(1, 2, 1, projection="3d") if projections else figure.add_subplot(projection="3d")
        projection_axis = figure.add_subplot(1, 2, 2) if projections else None
        if appearance == "paper":
            sphere_axis.set_proj_type("ortho")
            sphere_axis.set_box_aspect([1, 1, 1])
            sphere_axis.set_axis_off()
            sphere_axis.view_init(elev=elevation, azim=azimuth % 360)
            for setter in (sphere_axis.set_xlim, sphere_axis.set_ylim, sphere_axis.set_zlim):
                setter(-1.08, 1.08)
            _paper_guides(sphere_axis, equator=equator, axes=axes)
        else:
            draw_sphere(sphere_axis, equator=equator, elevation=elevation, azimuth=azimuth, axes=axes)
        mappable = None
        endpoint_annotations = dict(data.annotations)
        for index, (layer, style, key, palette, scalar) in enumerate(prepared):
            points = _real_array(layer.coordinates["points"], "points")
            width = linewidth if linewidth is not None else float(style.pop("linewidth", 1.8))
            if not np.isfinite(width) or width <= 0:
                raise ValueError("Layer linewidth must be finite and positive.")
            for reserved in ("color_by", "cmap", "linewidth"):
                style.pop(reserved, None)
            if scalar is None:
                color = style.pop("color", LAYER_COLORS[index % len(LAYER_COLORS)])
                if len(points) == 1:
                    sphere_axis.scatter(*points[0], color=color, **style)
                else:
                    sphere_axis.plot(*points.T, color=color, lw=width, label=layer.label or layer.name, **style)
            else:
                identity = (key, palette, layer.metadata.get("parameter_unit"))
                limits = ranges[identity] if appearance == "paper" or color_range is not None else (float(np.min(scalar)), float(np.max(scalar)))
                if limits[0] == limits[1]:
                    limits = (limits[0], limits[0] + 1)
                norm = plt.Normalize(*limits)
                if len(points) == 1:
                    artist = sphere_axis.scatter(*points.T, c=scalar, cmap=palette, norm=norm, **style)
                else:
                    rasterized = bool(style.pop("rasterized", True))
                    segments = np.stack((points[:-1], points[1:]), axis=1)
                    if appearance == "paper":
                        style.setdefault("capstyle", "round")
                    artist = Line3DCollection(segments, cmap=palette, norm=norm, lw=width, **style)
                    artist.set_array(.5 * (scalar[:-1] + scalar[1:]) if appearance == "paper" else scalar[:-1])
                    artist.set_rasterized(rasterized)
                    sphere_axis.add_collection3d(artist)
                mappable = mappable or artist
            if (markers if markers is not None else appearance == "paper"):
                for suffix, point, color, marker in (("start", points[0], "#0072B2", "o"), ("end", points[-1], "#D55E00", "D")):
                    # Existing scientific annotations are preserved; automatic endpoints never invent a target.
                    if not any(isinstance(a, dict) and "point" in a and np.array_equal(a["point"], point) for a in endpoint_annotations.values()):
                        endpoint_annotations[f"{layer.name}_{suffix}"] = {"point": point, "color": color, "marker": marker, "label": ""}
            for fraction in np.linspace(0, 1, arrows + 2)[1:-1]:
                if len(points) < 2:
                    break
                i = min(int(fraction * (len(points) - 1)), len(points) - 2)
                delta = points[i + 1] - points[i]
                length = np.linalg.norm(delta)
                if length > 1e-14:
                    arrow_color = plt.get_cmap(palette)(norm(scalar[i])) if scalar is not None else color
                    sphere_axis.quiver(*points[i], *(delta / length * .075), color=arrow_color,
                                       arrow_length_ratio=.5, lw=width, normalize=False)
        if appearance == "paper":
            _paper_annotations(sphere_axis, endpoint_annotations)
        else:
            _draw_annotations(sphere_axis, endpoint_annotations)
        if colorbar and mappable is not None:
            bar = figure.colorbar(mappable, ax=sphere_axis, shrink=.62, pad=.02)
            bar.set_label(str(data.metadata.get("parameter_label", "parameter")))
        if projection_axis is not None:
            draw_timeseries(projection_axis, data)
            return figure, sphere_axis, projection_axis
        return figure, sphere_axis
    except Exception:
        plt.close(figure)
        raise


def save_figure(figure: Any, path: Any, *, dpi: int = 400, close: bool = False) -> None:
    """Save a rendered figure; closing remains an explicit caller choice."""

    plt, _line2d, _collection = _matplotlib()
    figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    if close:
        plt.close(figure)
