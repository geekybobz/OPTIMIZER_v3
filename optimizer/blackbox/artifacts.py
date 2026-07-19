"""Array artifact storage for blackbox runs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from optimizer.controls import Controls
from optimizer.blackbox.records import array_summary, controls_summary, json_safe


_SAFE_PART = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_part(value: Any) -> str:
    """Return a filesystem-safe artifact id part."""

    text = str(value).strip().replace("/", "_")
    text = _SAFE_PART.sub("_", text)
    return text.strip("._") or "artifact"


def artifact_id(kind: str, label: str, *, seq: int | None = None, iteration: int | None = None) -> str:
    """Build a stable readable artifact id."""

    parts = [safe_part(kind), safe_part(label)]
    if iteration is not None:
        parts.append(f"i{int(iteration)}")
    if seq is not None:
        parts.append(f"s{int(seq):06d}")
    return "_".join(parts)


def _artifact_path(run_dir: Path, artifact: str) -> Path:
    arrays = run_dir / "arrays"
    arrays.mkdir(parents=True, exist_ok=True)
    return arrays / f"{safe_part(artifact)}.npz"


def save_array(
    run_dir: Path | str,
    artifact: str,
    array: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save one array-like artifact and return its manifest entry."""

    root = Path(run_dir)
    path = _artifact_path(root, artifact)
    arr = np.asarray(array)
    meta = json_safe(dict(metadata or {}))
    np.savez_compressed(path, array=arr, metadata_json=json.dumps(meta, sort_keys=True))
    relpath = str(path.relative_to(root))
    return {
        "id": safe_part(artifact),
        "kind": "array",
        "path": relpath,
        "summary": array_summary(arr),
        "metadata": meta,
    }


def save_controls(
    run_dir: Path | str,
    artifact: str,
    controls: Controls,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a Controls object as a compressed NumPy artifact."""

    root = Path(run_dir)
    path = _artifact_path(root, artifact)
    meta = json_safe(dict(metadata or {}))
    spec = controls.spec.to_dict()
    np.savez_compressed(
        path,
        matrix=controls.as_matrix(copy=True),
        keys=np.asarray(list(controls.keys), dtype=str),
        control_dim=np.asarray(int(controls.spec.control_dim)),
        name=np.asarray("" if controls.name is None else str(controls.name), dtype=str),
        spec_json=json.dumps(spec, sort_keys=True),
        meta_json=json.dumps(json_safe(controls.meta), sort_keys=True),
        metadata_json=json.dumps(meta, sort_keys=True),
    )
    relpath = str(path.relative_to(root))
    return {
        "id": safe_part(artifact),
        "kind": "controls",
        "path": relpath,
        "summary": controls_summary(controls),
        "metadata": meta,
    }


def load_artifact(run_dir: Path | str, relpath: str) -> dict[str, Any]:
    """Load a saved artifact as raw NumPy payload plus decoded metadata."""

    path = Path(run_dir) / relpath
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    for key in ("metadata_json", "spec_json", "meta_json"):
        if key in payload:
            raw = payload[key]
            payload[key] = json.loads(str(raw.item() if raw.shape == () else raw))
    return payload
