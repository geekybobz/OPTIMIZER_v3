"""Control spectrum and smoothness diagnostics.

Why this file exists
--------------------
A pulse can improve an objective while becoming physically ugly: large jumps, high
frequency content, channel imbalance, or excessive roughness.  These diagnostics make
that visible without changing the controls.  Filtering and bandwidth projection can
be added later as separate tools.

How it fits the architecture
----------------------------
- controls remain the only required input.
- reports use ``ControlSpec.dt`` when available so frequencies have physical units.
- diagnostic reports can include spectrum/smoothness summaries during training.

What this file deliberately does not do
---------------------------------------
It does not smooth, filter, clip, or otherwise modify controls.  It measures only.

Reviewer invariants
-------------------
- FFT is vectorized over control channels.
- reported arrays are JSON-friendly lists.
- roughness uses finite differences along the time/control dimension only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from optimizer.controls import Controls


def control_spectrum(
    controls: Controls,
    *,
    dt: float | None = None,
    high_frequency_cutoff: float | None = None,
) -> dict[str, Any]:
    """Return channel-wise real FFT amplitude/power summaries."""

    matrix = controls.as_matrix(copy=False)
    sample_dt = controls.spec.dt if dt is None else float(dt)
    if sample_dt is None:
        sample_dt = 1.0
    if sample_dt <= 0.0 or not np.isfinite(sample_dt):
        raise ValueError("dt must be finite and positive.")

    spectrum = np.fft.rfft(matrix, axis=1)
    frequencies = np.fft.rfftfreq(controls.spec.control_dim, d=sample_dt)
    amplitude = np.abs(spectrum)
    power = amplitude * amplitude
    total_power = np.sum(power, axis=1)
    dominant_index = np.argmax(power, axis=1)

    high_fraction: dict[str, float] | None = None
    if high_frequency_cutoff is not None:
        cutoff = float(high_frequency_cutoff)
        if cutoff < 0.0 or not np.isfinite(cutoff):
            raise ValueError("high_frequency_cutoff must be finite and nonnegative.")
        mask = frequencies >= cutoff
        high_power = np.sum(power[:, mask], axis=1)
        high_fraction = {
            key: float(high_power[row] / max(total_power[row], np.finfo(float).tiny))
            for row, key in enumerate(controls.keys)
        }

    return {
        "kind": "control_spectrum",
        "dt": float(sample_dt),
        "frequencies": frequencies.tolist(),
        "dominant_frequency": {
            key: float(frequencies[dominant_index[row]]) for row, key in enumerate(controls.keys)
        },
        "total_power": {key: float(total_power[row]) for row, key in enumerate(controls.keys)},
        "high_frequency_fraction": high_fraction,
        "amplitude": {key: amplitude[row].tolist() for row, key in enumerate(controls.keys)},
        "power": {key: power[row].tolist() for row, key in enumerate(controls.keys)},
    }


def smoothness_report(controls: Controls) -> dict[str, Any]:
    """Return finite-difference roughness and jump summaries."""

    matrix = controls.as_matrix(copy=False)
    if controls.spec.control_dim < 2:
        first = np.zeros((controls.spec.n_controls, 0), dtype=float)
    else:
        first = np.diff(matrix, axis=1)
    if controls.spec.control_dim < 3:
        second = np.zeros((controls.spec.n_controls, 0), dtype=float)
    else:
        second = np.diff(matrix, n=2, axis=1)

    first_norm = np.linalg.norm(first, axis=1) if first.size else np.zeros(controls.spec.n_controls)
    second_norm = np.linalg.norm(second, axis=1) if second.size else np.zeros(controls.spec.n_controls)
    total_variation = np.sum(np.abs(first), axis=1) if first.size else np.zeros(controls.spec.n_controls)
    max_jump = np.max(np.abs(first), axis=1) if first.size else np.zeros(controls.spec.n_controls)

    return {
        "kind": "smoothness_report",
        "first_difference_norm": {
            key: float(first_norm[row]) for row, key in enumerate(controls.keys)
        },
        "second_difference_norm": {
            key: float(second_norm[row]) for row, key in enumerate(controls.keys)
        },
        "total_variation": {
            key: float(total_variation[row]) for row, key in enumerate(controls.keys)
        },
        "max_jump": {key: float(max_jump[row]) for row, key in enumerate(controls.keys)},
        "global_first_difference_norm": float(np.linalg.norm(first)) if first.size else 0.0,
        "global_second_difference_norm": float(np.linalg.norm(second)) if second.size else 0.0,
        "global_total_variation": float(np.sum(np.abs(first))) if first.size else 0.0,
    }
