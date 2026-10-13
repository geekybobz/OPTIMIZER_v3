"""Explicit artifact export for Python, CLI and MCP, with no figure transport."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from optimizer.visualization.bloch import plot_bloch, style_context
from optimizer.serialization import to_plain


def render_bloch(
    value: Any,
    output: str,
    *,
    parameter: Any = None,
    options: Mapping[str, Any] | None = None,
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 400,
) -> dict[str, Any]:
    """Export to a new file stem and return paths/hashes/settings.

    Refuses existing files. Exports are external effects: a later session/history
    failure or lost reply does not erase completed files; inspect before retrying.
    No propagation, optimizer calls or additional persistent log are performed.
    """
    if not isinstance(output, (str, os.PathLike)) or not str(output).strip():
        raise ValueError("output must name an explicit file stem.")
    if not isinstance(formats, (tuple, list)) or not formats or any(fmt not in {"png", "pdf"} for fmt in formats) or len(set(formats)) != len(formats):
        raise ValueError("formats must contain unique png/pdf entries.")
    if type(dpi) is not int or not 72 <= dpi <= 1200:
        raise ValueError("dpi must be an integer from 72 to 1200.")
    if options is not None and not isinstance(options, Mapping):
        raise TypeError("options must be a mapping.")
    settings = {"projections": False, "appearance": "paper", **dict(options or {})}
    if "value" in settings or "parameter" in settings:
        raise ValueError("value/parameter must be supplied as top-level arguments.")
    # Only the public renderer's explicit keywords are transportable.
    import inspect
    valid = set(inspect.signature(plot_bloch).parameters) - {"value", "parameter"}
    if set(settings) - valid:
        raise ValueError(f"Unknown rendering options: {sorted(set(settings) - valid)}.")
    bound = inspect.signature(plot_bloch).bind(value, parameter=parameter, **settings)
    bound.apply_defaults()
    effective = {key: val for key, val in bound.arguments.items() if key not in {"value", "parameter"}}
    plain_settings = to_plain(effective)
    data_bytes = json.dumps(to_plain({"value": value, "parameter": parameter}), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    stem = Path(output).expanduser().absolute()
    paths = tuple(stem.with_suffix("." + fmt) for fmt in formats)
    if any(path.exists() for path in paths):
        raise FileExistsError("Bloch exports require new filenames; existing artifacts are preserved.")
    temporary = []
    created = []
    figure = None
    try:
        # Validate/render before touching destination directories.
        with style_context():
            figure, *_axes = plot_bloch(value, parameter=parameter, **settings)
            paths[0].parent.mkdir(parents=True, exist_ok=True)
            for fmt, destination in zip(formats, paths):
                fd, name = tempfile.mkstemp(prefix=".bloch-", suffix="." + fmt, dir=destination.parent)
                os.close(fd)
                temporary.append(Path(name))
                figure.savefig(name, format=fmt, dpi=dpi, bbox_inches="tight")
            # Exclusive hard-link publication refuses overwrite even under races.
            for source, destination in zip(temporary, paths):
                os.link(source, destination)
                created.append(destination)
        import optimizer
        artifacts = [{"path": str(path), "format": fmt, "bytes": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                     for path, fmt in zip(paths, formats)]
        return {"schema": "optimizer.bloch-render/1", "optimizer_version": optimizer.__version__,
                "input_sha256": hashlib.sha256(data_bytes).hexdigest(), "settings": plain_settings,
                "dpi": dpi, "artifacts": artifacts, "numerical_calls": 0}
    except Exception:
        # Remove only files this call published, never a pre-existing/racing file.
        for path in created:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
        if figure is not None:
            import matplotlib.pyplot as plt
            plt.close(figure)
