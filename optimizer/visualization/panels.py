"""Standalone single-panel renderers: one matplotlib figure, one file.

Bloch-sphere, Bloch-component-projection, and control-amplitude panels are
common building blocks for qubit-control verification figures, and are
deliberately kept as separate figures rather than baked into one combined,
multi-panel composite -- so callers (and the LaTeX/HTML/etc. documents that
embed them) can arrange, resize, or reorder panels without regenerating
plots.

This module owns the two panel kinds ``bloch.py`` does not: it must skip the
Bloch-ball validation for control amplitudes (which are not unit vectors)
and use a different, separately-established color palette for them. It
imports one-directionally from ``optimizer.visualization.bloch`` (never the
reverse), so there is no circular import.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from optimizer.visualization.bloch import (
    COMPONENT_COLORS,
    DEFAULT_AZIMUTH,
    DEFAULT_ELEVATION,
    _matplotlib,
    draw_timeseries,
    plot_bloch,
    save_figure,
    validate_bloch_data,
)
from optimizer.visualization.models import PlotData, coerce_plot_data

# Established, project-wide control-channel convention (vermillion middle
# channel), kept distinct from COMPONENT_COLORS's amber middle channel used
# for Bloch-vector components -- these are two different things and are
# never merged into one palette.
CONTROL_COLORS: tuple[str, ...] = ("#0072B2", "#D55E00", "#009E73")


def validate_controls_data(value: PlotData | Mapping[str, Any]) -> PlotData:
    """Validate generic plot data for a control-amplitude-vs-parameter panel.

    Deliberately does not call ``validate_bloch_data``: control amplitudes
    are not constrained to the Bloch unit ball, so no norm check applies.
    """

    data = coerce_plot_data(value)
    if data.kind != "controls":
        raise ValueError(f"Controls panel requires plot kind 'controls', got {data.kind!r}.")
    for layer in data.layers:
        if "points" not in layer.coordinates:
            raise KeyError(f"Controls layer {layer.name!r} must provide coordinates['points'].")
        points = np.asarray(layer.coordinates["points"], dtype=float)
        if points.ndim != 2 or points.shape[0] < 1:
            raise ValueError(f"Controls layer {layer.name!r} points must have shape (n, k).")
        parameter = layer.coordinates.get("parameter")
        if parameter is not None and np.asarray(parameter).shape != (points.shape[0],):
            raise ValueError(
                f"Controls layer {layer.name!r} parameter must have shape ({points.shape[0]},)."
            )
    return data


def plot_bloch_panel(
    value: PlotData | Mapping[str, Any],
    *,
    equator: bool = True,
    elevation: float = DEFAULT_ELEVATION,
    azimuth: float = DEFAULT_AZIMUTH,
    figsize: tuple[float, float] = (3.3, 3.3),
    colorbar: bool = True,
    axes: bool = False,
) -> tuple[Any, Any]:
    """Standalone single-panel Bloch-sphere figure.

    Thin, discoverability-only alias for ``plot_bloch(..., projections=False)``
    -- that call already produces exactly this panel. axes=True adds small
    x/y/z reference-axis stubs, useful whenever elevation/azimuth is rotated
    away from the default view.
    """

    return plot_bloch(
        value,
        projections=False,
        equator=equator,
        elevation=elevation,
        azimuth=azimuth,
        figsize=figsize,
        colorbar=colorbar,
        axes=axes,
    )


def plot_projection_panel(
    value: PlotData | Mapping[str, Any],
    *,
    component_labels: Sequence[str] = (r"$r_x$", r"$r_y$", r"$r_z$"),
    xlim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (3.4, 3.0),
) -> tuple[Any, Any]:
    """Standalone Bloch-component-vs-parameter panel (e.g. r_x,r_y,r_z(t))."""

    data = validate_bloch_data(value)
    plt, *_rest = _matplotlib()
    figure, axis = plt.subplots(figsize=figsize)
    draw_timeseries(
        axis,
        data,
        palette=COMPONENT_COLORS,
        component_labels=component_labels,
        ylabel="Bloch component",
        xlim=xlim,
        ylim=(-1.14, 1.14),
        prefix_labels=False,
        legend_ncols=3,
    )
    return figure, axis


def plot_controls_panel(
    value: PlotData | Mapping[str, Any],
    *,
    component_labels: Sequence[str] = (r"$u_x$", r"$u_y$", r"$u_z$"),
    xlim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (4.5, 2.75),
) -> tuple[Any, Any]:
    """Standalone control-amplitude-vs-parameter panel (e.g. u_x,u_y,u_z(t)).

    Uses CONTROL_COLORS and validate_controls_data -- no Bloch-ball check.
    """

    data = validate_controls_data(value)
    plt, *_rest = _matplotlib()
    figure, axis = plt.subplots(figsize=figsize)
    draw_timeseries(
        axis,
        data,
        palette=CONTROL_COLORS,
        component_labels=component_labels,
        ylabel="control amplitude",
        xlim=xlim,
        ylim=None,
        prefix_labels=False,
        legend_ncols=3,
    )
    return figure, axis


def save_panel(
    figure: Any,
    out_stem: str | Path,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 400,
    close: bool = True,
) -> tuple[Path, ...]:
    """Save one panel figure under ``out_stem`` for every requested format.

    Built on ``save_figure`` (reused, not reimplemented); adds the
    stem-to-multi-format loop and the "done with this panel" defaults
    (dpi=400 so a rasterized trajectory layer stays crisp; close=True since
    a finished standalone panel has nothing left to draw on).
    """

    stem = Path(out_stem)
    written = tuple(stem.with_suffix(f".{fmt.lstrip('.')}") for fmt in formats)
    for path in written:
        save_figure(figure, path, dpi=dpi, close=False)
    if close:
        plt, *_rest = _matplotlib()
        plt.close(figure)
    return written
