"""Pure, explicit control-grid transformations. No physics or implicit activation."""
from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from optimizer.controls import Controls, ControlSpec, _validate_control_spec_compatibility

_DEFAULT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class ControlGrid:
    """Sample locations plus their physical interval and placement convention.

    Units are a declared label, never automatically converted. Nonuniform grids
    are supported except by FFT filtering and sample-based Gaussian smoothing.
    """
    times: tuple[float, ...]
    start: float
    stop: float
    placement: str
    units: str | None = None

    def __post_init__(self):
        times = tuple(float(t) for t in self.times)
        start, stop = float(self.start), float(self.stop)
        if len(times) < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise ValueError("Grid needs at least two finite, strictly increasing sample times.")
        if not np.isfinite(start) or not np.isfinite(stop) or start >= stop:
            raise ValueError("Grid interval must be finite and increasing.")
        if self.placement not in {"nodes", "left_cells", "cell_centers"}:
            raise ValueError("placement must be nodes, left_cells or cell_centers.")
        if times[0] < start or times[-1] > stop:
            raise ValueError("Samples must lie in the physical interval.")
        if self.placement == "nodes" and (times[0] != start or times[-1] != stop):
            raise ValueError("Node grid must include both physical endpoints.")
        if self.placement == "left_cells" and (times[0] != start or times[-1] >= stop):
            raise ValueError("Left-cell grid starts at start and excludes stop.")
        if self.placement == "cell_centers" and (times[0] <= start or times[-1] >= stop):
            raise ValueError("Cell centers must lie strictly inside the interval.")
        if self.units is not None and (not isinstance(self.units, str) or not self.units):
            raise ValueError("units must be a nonempty label or None.")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)

    @property
    def count(self):
        return len(self.times)

    @property
    def spacing(self):
        steps = np.diff(self.times)
        return float(steps[0]) if np.allclose(steps, steps[0], rtol=1e-10, atol=0) else None


@dataclass(frozen=True)
class ControlTransformResult:
    """An owned candidate and numerical facts, never a feasibility certificate."""
    controls: Controls
    operation: str
    parameters: dict[str, Any] = field(default_factory=dict)
    source_grids: tuple[ControlGrid | None, ...] = ()
    target_grid: ControlGrid | None = None
    facts: dict[str, Any] = field(default_factory=dict)
    source_refs: tuple[Any, ...] = ()


def _budget(max_bytes, *sizes):
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer.")
    if sum(sizes) > max_bytes:
        raise MemoryError("Estimated transformation arrays exceed max_bytes; increase explicitly or reduce the grid.")


def uniform_control_grid(*, start, stop, count, placement, units=None, max_bytes=_DEFAULT_BYTES):
    """Create a uniform declared grid; count always means stored samples."""
    if type(count) is not int or count < 2:
        raise ValueError("count must be an integer >= 2.")
    _budget(max_bytes, count * 64)
    start, stop = float(start), float(stop)
    if placement == "nodes":
        times = np.linspace(start, stop, count)
    elif placement == "left_cells":
        times = start + np.arange(count) * ((stop - start) / count)
    elif placement == "cell_centers":
        times = start + (np.arange(count) + 0.5) * ((stop - start) / count)
    else:
        raise ValueError("Unknown sample placement.")
    return ControlGrid(tuple(times), start, stop, placement, units)


def _controls(value):
    if not isinstance(value, Controls):
        raise TypeError("Expected Controls, or select a transform's .controls member.")
    if value.spec.dtype.kind not in "fc" or value.u.dtype != value.spec.dtype or value.u.shape != value.spec.shape or not np.all(np.isfinite(value.u)):
        raise ValueError("Transform input requires finite floating or complex controls with the declared shape.")
    return value


def _grid(controls, grid=None, *, required=False):
    stored = controls.meta.get("control_grid")
    if grid is None:
        grid = stored
    if grid is None:
        if required:
            raise ValueError("An explicit ControlGrid is required; dt alone does not declare placement.")
        return None
    if not isinstance(grid, ControlGrid) or grid.count != controls.spec.control_dim:
        raise ValueError("ControlGrid count must match control samples.")
    # Revalidate descriptors, including ones obtained from trusted mutable callers.
    ControlGrid(grid.times, grid.start, grid.stop, grid.placement, grid.units)
    if stored is not None and stored != grid:
        raise ValueError("Explicit grid conflicts with the control's stored grid.")
    if controls.spec.dt is not None and (grid.spacing is None or not np.isclose(grid.spacing, controls.spec.dt, rtol=1e-10, atol=0)):
        raise ValueError("ControlSpec.dt disagrees with the declared grid spacing.")
    return grid


def _channels(controls, channels):
    keys = controls.keys if channels is None else (channels,) if isinstance(channels, str) else tuple(channels)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Select nonempty unique channel keys.")
    return tuple(controls.spec.channel_index(key) for key in keys)


def _channel_values(controls, value, rows):
    if isinstance(value, Mapping):
        keys = tuple(controls.keys[r] for r in rows)
        if set(value) != set(keys):
            raise ValueError("Per-channel values must exactly match selected keys.")
        array = np.asarray([value[k] for k in keys])
    else:
        array = np.asarray(value)
        if array.ndim == 0:
            array = np.repeat(array, len(rows))
    if array.shape != (len(rows),) or array.dtype.kind not in "iufc" or not np.all(np.isfinite(array)):
        raise ValueError("Values must be finite scalars or one per selected channel.")
    return array[:, None]


def _result(inputs, matrix, operation, parameters, *, spec=None, grid=None, facts=None):
    inputs = tuple(inputs)
    spec = inputs[0].spec if spec is None else spec
    matrix = np.asarray(matrix)
    # Source precision cannot be reduced. Intermediate arithmetic can use wider
    # precision, then round to the declared output precision like other solvers.
    safe_sources = all(np.can_cast(c.u.dtype, spec.dtype, casting="safe") for c in inputs)
    if spec.dtype.kind not in "fc" or not safe_sources or (np.iscomplexobj(matrix) and spec.dtype.kind != "c"):
        raise ValueError("Transformation would require a lossy dtype cast; declare a compatible floating/complex layout.")
    # Deliberately discard stale evaluation/solver metadata. Only grid provenance
    # belongs on the new Controls; all numerical facts belong on this report.
    controls = Controls(copy.deepcopy(spec), matrix, meta={} if grid is None else {"control_grid": grid})
    declarations = (facts or {}).get("source_grids")
    if declarations is None:
        declarations = ((facts["source_grid"],) if facts and "source_grid" in facts else
                        (grid,) if len(inputs) == 1 and grid is not None else tuple(_grid(c) for c in inputs))
    return ControlTransformResult(controls, operation, copy.deepcopy(dict(parameters)),
                                  tuple(declarations), grid,
                                  copy.deepcopy({"feasibility": "not_evaluated", **(facts or {})}))


def resample_controls(controls, *, source_grid, target_grid, target, method="linear",
                      extrapolation="error", max_bytes=_DEFAULT_BYTES):
    """Linearly interpolate real/complex channels; extrapolation is explicit."""
    controls = _controls(controls)
    source = _grid(controls, source_grid, required=True)
    if not isinstance(target, ControlSpec) or not isinstance(target_grid, ControlGrid):
        raise TypeError("target and target_grid must be ControlSpec and ControlGrid.")
    if controls.keys != target.keys or controls.spec.meta.get("units") != target.meta.get("units"):
        raise ValueError("Resampling requires identical channel order and declared control units.")
    if source.units != target_grid.units:
        raise ValueError("Time-grid units must match; no implicit conversion.")
    if method != "linear" or extrapolation not in {"error", "hold", "linear"}:
        raise ValueError("Use method='linear' and extrapolation='error', 'hold' or 'linear'.")
    if target_grid.count != target.control_dim or target.dtype.kind not in "fc" or not np.can_cast(controls.u.dtype, target.dtype, casting="safe"):
        raise ValueError("Target grid/layout mismatch or unsupported dtype.")
    if target.dt is not None and (target_grid.spacing is None or not np.isclose(target.dt, target_grid.spacing, rtol=1e-10, atol=0)):
        raise ValueError("Target dt disagrees with grid.")
    _budget(max_bytes, controls.u.nbytes, target.size * 32, (source.count + target_grid.count) * 64)
    x, y = np.asarray(source.times), np.asarray(target_grid.times)
    outside = (y < x[0]) | (y > x[-1])
    if np.any(outside) and extrapolation == "error":
        raise ValueError("Target samples extend outside source coverage; declare extrapolation.")
    matrix = np.stack([np.interp(y, x, row) for row in controls.u])
    if extrapolation == "linear":
        for left, a, b in ((True, 0, 1), (False, -2, -1)):
            mask = y < x[0] if left else y > x[-1]
            matrix[:, mask] = controls.u[:, a, None] + (y[mask] - x[a]) * (
                (controls.u[:, b] - controls.u[:, a]) / (x[b] - x[a]))[:, None]
    return _result([controls], matrix, "resample_controls", {"method": method, "extrapolation": extrapolation},
                   spec=target, grid=target_grid, facts={"source_grid": source,
                   "extrapolated_samples": int(np.count_nonzero(outside)),
                   "physical_interval_changed": (source.start, source.stop) != (target_grid.start, target_grid.stop),
                   "downsampled": target_grid.count < source.count,
                   "aliasing_warning": "Downsampling may alias; apply an explicit anti-alias filter first."
                       if target_grid.count < source.count else None,
                   "anti_alias_filter_applied": False})


def scale_controls(controls, *, factors, offsets=0.0, channels=None, max_bytes=_DEFAULT_BYTES):
    """Exact affine scaling (without amplitude normalization)."""
    controls = _controls(controls)
    rows = _channels(controls, channels)
    factors, offsets = _channel_values(controls, factors, rows), _channel_values(controls, offsets, rows)
    _budget(max_bytes, controls.u.nbytes * 4)
    dtype = np.result_type(controls.u.dtype, factors.dtype, offsets.dtype)
    matrix = controls.u.astype(dtype, copy=True)
    matrix[list(rows)] = matrix[list(rows)] * factors + offsets
    return _result([controls], matrix, "scale_controls", {"factors": factors, "offsets": offsets,
                   "channels": tuple(controls.keys[r] for r in rows)}, grid=_grid(controls))


def _quadrature(grid):
    times = np.asarray(grid.times)
    if grid.placement == "nodes":
        steps = np.diff(times)
        return np.r_[steps[0] / 2, (steps[:-1] + steps[1:]) / 2, steps[-1] / 2]
    if grid.placement == "left_cells":
        return np.diff(np.r_[times, grid.stop])
    return np.diff(np.r_[grid.start, (times[:-1] + times[1:]) / 2, grid.stop])


def normalize_controls(controls, *, norm="max_abs", value=1.0, scope="global", channels=None,
                       grid=None, zero="error", max_bytes=_DEFAULT_BYTES):
    """Normalize max_abs, discrete l2/rms, or quadrature time_l2/time_rms."""
    controls = _controls(controls)
    rows = _channels(controls, channels)
    value = float(value)
    if not np.isfinite(value) or value < 0 or scope not in {"global", "channel"} or zero not in {"error", "keep"}:
        raise ValueError("Require finite nonnegative value, global/channel scope and error/keep zero policy.")
    grid = _grid(controls, grid, required=norm in {"time_l2", "time_rms"})
    _budget(max_bytes, controls.u.nbytes * 6)
    selected = controls.u[list(rows)]
    axis = None if scope == "global" else 1
    if norm == "max_abs":
        measured = np.max(np.abs(selected), axis=axis)
    elif norm in {"l2", "rms", "time_l2", "time_rms"}:
        square = np.abs(selected) ** 2
        if norm.startswith("time_"):
            square = square * _quadrature(grid)
        measured = np.sum(square, axis=axis)
        if norm == "rms":
            measured /= selected.size if axis is None else selected.shape[1]
        elif norm == "time_rms":
            measured /= (grid.stop - grid.start) * (len(rows) if axis is None else 1)
        measured = np.sqrt(measured)
    else:
        raise ValueError("Unknown norm: max_abs, l2, rms, time_l2 or time_rms required.")
    if not np.all(np.isfinite(measured)):
        raise ValueError("Measured norm overflowed; rescale input before normalization.")
    if np.any(np.asarray(measured) == 0) and zero == "error" and value != 0:
        raise ValueError("Cannot normalize a zero control to a positive norm; choose zero='keep'.")
    factors = np.divide(value, measured, out=np.zeros_like(np.asarray(measured, dtype=float)), where=np.asarray(measured) != 0)
    matrix = controls.u.copy()
    matrix[list(rows)] = selected * (factors if axis is None else factors[:, None])
    return _result([controls], matrix, "normalize_controls", {"norm": norm, "value": value,
                   "scope": scope, "zero": zero, "channels": tuple(controls.keys[r] for r in rows)},
                   grid=grid, facts={"input_norm": measured, "zero_preserved": bool(np.any(np.asarray(measured) == 0))})


def _padding(boundary):
    try:
        return {"periodic": "wrap", "reflect": "reflect", "hold": "edge"}[boundary]
    except KeyError as exc:
        raise ValueError("boundary must be periodic, reflect or hold.") from exc


def smooth_controls(controls, *, sigma, boundary, grid=None, method="gaussian", channels=None,
                    max_bytes=_DEFAULT_BYTES):
    """Gaussian convolution; sigma is explicitly in samples, truncated at 4 sigma."""
    controls = _controls(controls)
    rows = _channels(controls, channels)
    grid = _grid(controls, grid)
    if grid is not None and grid.spacing is None:
        raise ValueError("Sample-based Gaussian smoothing requires a uniform grid.")
    sigma = float(sigma)
    if method != "gaussian" or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("method='gaussian' requires finite positive sigma in samples.")
    mode = _padding(boundary)
    radius = max(1, int(np.ceil(4 * sigma)))
    _budget(max_bytes, controls.u.nbytes * 4, (controls.spec.control_dim + 2 * radius) * 32, (2 * radius + 1) * 32)
    distance = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (distance / sigma) ** 2)
    kernel /= kernel.sum()
    matrix = controls.u.copy()
    for row in rows:
        matrix[row] = np.convolve(np.pad(controls.u[row], radius, mode=mode), kernel, mode="valid")
    return _result([controls], matrix, "smooth_controls", {"method": method, "sigma": sigma,
                   "sigma_units": "samples", "boundary": boundary,
                   "channels": tuple(controls.keys[r] for r in rows)}, grid=grid, facts={"radius": radius})


def filter_controls(controls, *, grid, kind, cutoff, boundary, channels=None, max_bytes=_DEFAULT_BYTES):
    """Ideal FFT mask, cutoff in cycles per declared time unit; no hidden filter."""
    controls = _controls(controls)
    grid = _grid(controls, grid, required=True)
    dt = grid.spacing
    if dt is None:
        raise ValueError("FFT filtering requires a uniform grid.")
    rows = _channels(controls, channels)
    mode = _padding(boundary)
    cut = np.asarray(cutoff, dtype=float).reshape(-1)
    count = 2 if kind in {"bandpass", "bandstop"} else 1
    if kind not in {"lowpass", "highpass", "bandpass", "bandstop"} or cut.size != count:
        raise ValueError("Use lowpass/highpass with one cutoff or bandpass/bandstop with two.")
    if not np.all(np.isfinite(cut)) or np.any(cut < 0) or np.any(cut > 0.5 / dt) or (count == 2 and cut[0] >= cut[1]):
        raise ValueError("Cutoffs must be ordered, nonnegative and within Nyquist frequency.")
    padding = 0 if boundary == "periodic" else controls.spec.control_dim - 1
    length = controls.spec.control_dim + 2 * padding
    _budget(max_bytes, controls.u.nbytes * 4, length * 128)
    frequencies = np.abs(np.fft.fftfreq(length, d=dt))
    mask = frequencies <= cut[0] if kind == "lowpass" else frequencies >= cut[0] if kind == "highpass" else (
        (frequencies >= cut[0]) & (frequencies <= cut[1]))
    if kind == "bandstop":
        mask = ~mask
    matrix = controls.u.copy()
    for row in rows:
        extended = np.pad(controls.u[row], padding, mode=mode) if padding else controls.u[row]
        filtered = np.fft.ifft(np.fft.fft(extended) * mask)
        if controls.spec.dtype.kind == "f":
            filtered = filtered.real
        matrix[row] = filtered[padding:padding + controls.spec.control_dim]
    return _result([controls], matrix, "filter_controls", {"kind": kind, "cutoff": cut,
                   "boundary": boundary, "channels": tuple(controls.keys[r] for r in rows)}, grid=grid,
                   facts={"frequency_units": None if grid.units is None else f"cycles/{grid.units}",
                          "nyquist": 0.5 / dt, "padded_samples": padding, "ideal_mask": True})


def _compatible_controls(items, grids=None):
    if not items:
        raise ValueError("At least one control is required.")
    items = tuple(_controls(c) for c in items)
    if grids is not None and len(grids) != len(items):
        raise ValueError("Supply one grid per input.")
    declarations = tuple(_grid(c, None if grids is None else grids[i], required=True) for i, c in enumerate(items))
    for item, grid in zip(items, declarations):
        _validate_control_spec_compatibility(items[0].spec, item.spec)
        if grid != declarations[0] or item.spec.meta.get("units") != items[0].spec.meta.get("units"):
            raise ValueError("Mixing requires identical grids and declared control units.")
    return items, declarations


def mix_controls(inputs, *, weights, grids=None, max_bytes=_DEFAULT_BYTES):
    """Explicit weighted sum; negative weights are allowed and never normalized."""
    items, declarations = _compatible_controls(tuple(inputs), grids)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(items),) or not np.all(np.isfinite(weights)):
        raise ValueError("Supply one finite real weight per input.")
    _budget(max_bytes, sum(c.u.nbytes for c in items), items[0].u.nbytes * 4)
    matrix = np.zeros(items[0].shape, dtype=np.result_type(items[0].u.dtype, weights.dtype))
    for weight, item in zip(weights, items):
        matrix += weight * item.u
    return _result(items, matrix, "mix_controls", {"weights": weights}, grid=declarations[0],
                   facts={"source_grids": declarations})


def select_channels(controls, *, channels, max_bytes=_DEFAULT_BYTES):
    """Select/reorder named channels, keeping their sampling convention."""
    controls = _controls(controls)
    rows = _channels(controls, channels)
    _budget(max_bytes, controls.u.nbytes * 3)
    keys = tuple(controls.keys[r] for r in rows)
    spec = ControlSpec(keys, controls.spec.control_dim, controls.spec.dtype, controls.spec.dt,
                       {**controls.spec.meta, "units": _selected_units(controls, keys)})
    return _result([controls], controls.u[list(rows)].copy(), "select_channels", {"channels": keys},
                   spec=spec, grid=_grid(controls))


def _selected_units(controls, keys):
    units = controls.spec.meta.get("units")
    return {k: units[k] for k in keys} if isinstance(units, Mapping) else units


def replace_channel(controls, *, channel, replacement, replacement_channel, grid=None,
                    replacement_grid=None, max_bytes=_DEFAULT_BYTES):
    """Replace one channel from another Controls on the identical declared grid."""
    controls, replacement = _controls(controls), _controls(replacement)
    first = _grid(controls, grid, required=True)
    second = _grid(replacement, replacement_grid, required=True)
    if first != second or controls.spec.dtype != replacement.spec.dtype:
        raise ValueError("Replacement requires identical grid and dtype.")
    units_a, units_b = _selected_units(controls, [channel]), _selected_units(replacement, [replacement_channel])
    if isinstance(units_a, Mapping): units_a = units_a[channel]
    if isinstance(units_b, Mapping): units_b = units_b[replacement_channel]
    if units_a != units_b:
        raise ValueError("Replacement channel units differ.")
    _budget(max_bytes, controls.u.nbytes * 3, replacement.u.nbytes)
    matrix = controls.u.copy()
    matrix[controls.spec.channel_index(channel)] = replacement.channel(replacement_channel)
    return _result([controls, replacement], matrix, "replace_channel", {"channel": channel,
                   "replacement_channel": replacement_channel}, grid=first, facts={"source_grids": (first, second)})


def assemble_channels(inputs, *, channels, target, grids=None, max_bytes=_DEFAULT_BYTES):
    """Build target rows from explicit (input_index, channel_key) pairs."""
    items = tuple(_controls(c) for c in inputs)
    if not items or not isinstance(target, ControlSpec) or len(channels) != target.n_controls:
        raise ValueError("Inputs and one channel source per target row are required.")
    if grids is not None and len(grids) != len(items):
        raise ValueError("Supply one grid per input.")
    declarations = tuple(_grid(c, None if grids is None else grids[i], required=True) for i, c in enumerate(items))
    if any(g != declarations[0] for g in declarations) or target.control_dim != declarations[0].count:
        raise ValueError("Assembly requires identical grids and matching target count.")
    _budget(max_bytes, sum(c.u.nbytes for c in items), target.size * 32)
    matrix = []
    for target_key, pair in zip(target.keys, channels):
        index, key = pair
        if type(index) is not int or not 0 <= index < len(items):
            raise ValueError("Channel input index is out of range.")
        item = items[index]
        source_units = _selected_units(item, [key])
        if isinstance(source_units, Mapping): source_units = source_units[key]
        units = target.meta.get("units")
        if isinstance(units, Mapping): units = units[target_key]
        if source_units != units:
            raise ValueError("Assembled channel units differ from target.")
        matrix.append(item.channel(key))
    if target.dt is not None and (declarations[0].spacing is None or not np.isclose(target.dt, declarations[0].spacing, rtol=1e-10, atol=0)):
        raise ValueError("Target dt disagrees with grid.")
    return _result(items, np.stack(matrix), "assemble_channels", {"channels": tuple(channels)},
                   spec=target, grid=declarations[0], facts={"source_grids": declarations})


def join_controls_in_time(inputs, *, grids, boundary="error", max_bytes=_DEFAULT_BYTES):
    """Join contiguous absolute-time segments; duplicated nodes require a policy.

    boundary='left' or 'right' chooses which sample to keep; 'error' rejects
    duplicates. Intervals must be contiguous; time shifts are never inferred.
    """
    items = tuple(_controls(c) for c in inputs)
    if not items or len(grids) != len(items) or boundary not in {"error", "left", "right"}:
        raise ValueError("Supply controls, one grid each, and error/left/right boundary policy.")
    grids = tuple(_grid(c, g, required=True) for c, g in zip(items, grids))
    base = items[0].spec
    for item, grid in zip(items, grids):
        if item.keys != base.keys or item.spec.dtype != base.dtype or item.spec.meta.get("units") != base.meta.get("units"):
            raise ValueError("Segments require identical channel order, dtype and units.")
        if grid.units != grids[0].units or grid.placement != grids[0].placement:
            raise ValueError("Segments require identical time units and placement.")
    _budget(max_bytes, sum(c.u.nbytes * 4 + g.count * 64 for c, g in zip(items, grids)))
    times, matrices = list(grids[0].times), [items[0].u.copy()]
    dropped = 0
    for previous, grid, item in zip(grids, grids[1:], items[1:]):
        if previous.stop != grid.start:
            raise ValueError("Physical intervals must be exactly contiguous.")
        row = item.u
        if grid.times[0] == times[-1]:
            if boundary == "error":
                raise ValueError("Duplicate boundary node; choose left or right.")
            dropped += 1
            if boundary == "left":
                row = row[:, 1:]
                new_times = grid.times[1:]
            else:
                times.pop()
                matrices[-1] = matrices[-1][:, :-1]
                new_times = grid.times
        else:
            new_times = grid.times
        times.extend(new_times)
        matrices.append(row)
    grid = ControlGrid(tuple(times), grids[0].start, grids[-1].stop, grids[0].placement, grids[0].units)
    spec = ControlSpec(base.keys, grid.count, base.dtype, grid.spacing, base.meta)
    return _result(items, np.concatenate(matrices, axis=1), "join_controls_in_time", {"boundary": boundary},
                   spec=spec, grid=grid, facts={"dropped_boundary_samples": dropped, "source_grids": grids})


def apply_control_bounds(controls, *, lower, upper, channels=None, max_bytes=_DEFAULT_BYTES):
    """Clip real channels to finite bounds; does not repair nonlinear residuals."""
    controls = _controls(controls)
    if controls.spec.dtype.kind != "f":
        raise ValueError("Ordered clipping requires real controls; complex bounds need a project operation.")
    rows = _channels(controls, channels)
    lower, upper = _channel_values(controls, lower, rows), _channel_values(controls, upper, rows)
    if np.iscomplexobj(lower) or np.iscomplexobj(upper) or np.any(lower > upper):
        raise ValueError("Require real lower <= upper bounds.")
    _budget(max_bytes, controls.u.nbytes * 5)
    matrix = controls.u.copy()
    before = matrix[list(rows)].copy()
    matrix[list(rows)] = np.clip(before, lower, upper)
    delta = matrix[list(rows)] - before
    return _result([controls], matrix, "apply_control_bounds", {"lower": lower, "upper": upper,
                   "channels": tuple(controls.keys[r] for r in rows)}, grid=_grid(controls),
                   facts={"changed_samples": int(np.count_nonzero(delta)), "correction_norm": float(np.linalg.norm(delta))})


def recover_shooting_controls(system, variables, *, grid, max_bytes=_DEFAULT_BYTES):
    """Use optional OLSS recover_controls(variables, times=...) without inverse inference.

    The hook returns Controls, or {'values': channel mapping/matrix, 'spec':
    ControlSpec, 'times': requested times}. A mapping's spec and times are mandatory.
    """
    from optimizer.system_olss import validate_variables_for_shooting_system
    if not isinstance(grid, ControlGrid):
        raise TypeError("grid must be ControlGrid.")
    validate_variables_for_shooting_system(system, variables)
    hook = getattr(system, "recover_controls", None)
    if not callable(hook):
        raise NotImplementedError("System requires recover_controls(variables, *, times).")
    _budget(max_bytes, grid.count * 64, variables.values.nbytes)
    recovered = hook(variables.copy(), times=np.asarray(grid.times))
    if isinstance(recovered, Mapping):
        if not {"values", "spec", "times"} <= set(recovered) or not np.array_equal(recovered["times"], grid.times):
            raise ValueError("Recovery mapping requires values/spec and the exact requested times.")
        spec, values = recovered["spec"], recovered["values"]
        if not isinstance(spec, ControlSpec):
            raise TypeError("Recovery spec must be ControlSpec.")
        matrix = np.stack([values[k] for k in spec.keys]) if isinstance(values, Mapping) else np.asarray(values)
        if not np.can_cast(matrix.dtype, spec.dtype, casting="safe"):
            raise ValueError("Recovery would discard dtype information.")
        recovered = Controls(spec, matrix)
    recovered = _controls(recovered)
    _grid(recovered, grid, required=True)
    _budget(max_bytes, recovered.u.nbytes * 4, grid.count * 64)
    return _result([recovered], recovered.u.copy(), "recover_shooting_controls", {}, grid=grid,
                   facts={"project_hook": "recover_controls", "inverse_mapping": False})
