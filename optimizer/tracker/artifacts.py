"""Explicit checkpoint artifact storage for the passive tracker.

Checkpoint files are never created by constructing a journal or logging an
event.  They are written only when ``RunJournal.save_checkpoint`` is called by
the optimizer/controller that has already chosen a checkpoint trigger.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from optimizer.controls import ControlSpec, Controls
from optimizer.tracker.model import json_safe


def _safe_file_stem(value: str, *, field: str) -> str:
    text = str(value)
    if not text or text in {".", ".."} or Path(text).name != text:
        raise ValueError(f"{field} must be a non-empty filename-safe identifier.")
    return text


def _encode_json(value: Mapping[str, Any] | None) -> np.ndarray:
    return np.asarray(json.dumps(json_safe(value or {}), sort_keys=True), dtype=np.str_)


def _decode_json(value: np.ndarray) -> dict[str, Any]:
    decoded = json.loads(str(value.item()))
    if not isinstance(decoded, dict):
        raise ValueError("Checkpoint JSON metadata must decode to an object.")
    return decoded


def _encode_optimizer_state(
    value: Mapping[str, Any] | None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Encode nested optimizer state with native, dtype-preserving arrays."""

    arrays: dict[str, np.ndarray] = {}

    def encode(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            name = f"optimizer_array_{len(arrays):04d}"
            arrays[name] = np.asarray(item)
            return {"__optimizer_array__": name}
        if isinstance(item, np.generic):
            name = f"optimizer_array_{len(arrays):04d}"
            arrays[name] = np.asarray(item)
            return {"__optimizer_array__": name, "scalar": True}
        if isinstance(item, Mapping):
            return {str(key): encode(child) for key, child in item.items()}
        if isinstance(item, list):
            return [encode(child) for child in item]
        if isinstance(item, tuple):
            return {"__optimizer_tuple__": [encode(child) for child in item]}
        try:
            return json_safe(item)
        except (TypeError, ValueError):
            return {"__optimizer_omitted_type__": type(item).__name__}

    encoded = encode(value or {})
    return np.asarray(json.dumps(encoded, sort_keys=True), dtype=np.str_), arrays


def _decode_optimizer_state(artifact: Any) -> dict[str, Any]:
    encoded = json.loads(str(artifact["optimizer_state_json"].item()))

    def decode(item: Any) -> Any:
        if isinstance(item, dict) and "__optimizer_array__" in item:
            array = np.asarray(artifact[str(item["__optimizer_array__"])]).copy()
            return array.item() if item.get("scalar") else array
        if isinstance(item, dict) and "__optimizer_tuple__" in item:
            return tuple(decode(child) for child in item["__optimizer_tuple__"])
        if isinstance(item, dict):
            return {str(key): decode(child) for key, child in item.items()}
        if isinstance(item, list):
            return [decode(child) for child in item]
        return item

    decoded = decode(encoded)
    if not isinstance(decoded, dict):
        raise ValueError("Checkpoint optimizer state must decode to an object.")
    return decoded


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class CheckpointReference:
    """Small JSON-safe pointer to a separately stored control artifact."""

    checkpoint_id: str
    label: str
    relative_path: str
    shape: tuple[int, int]
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "label": self.label,
            "relative_path": self.relative_path,
            "shape": list(self.shape),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class CheckpointData:
    """Loaded controls plus caller-supplied JSON metadata."""

    reference: CheckpointReference
    controls: Controls
    optimizer_state: dict[str, Any]
    snapshot: dict[str, Any]
    api_context: dict[str, Any]
    metadata: dict[str, Any]


def save_checkpoint(
    run_dir: str | Path,
    *,
    checkpoint_id: str,
    label: str,
    controls: Controls,
    optimizer_state: Mapping[str, Any] | None = None,
    snapshot: Mapping[str, Any] | None = None,
    api_context: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CheckpointReference:
    """Atomically save controls and explicitly supplied resume metadata.

    ``optimizer_state`` and ``metadata`` must be JSON-safe after NumPy scalars
    and arrays are converted.  Arbitrary Python objects are intentionally not
    pickled, keeping artifacts portable and safe to inspect.
    """

    checkpoint_id = _safe_file_stem(checkpoint_id, field="checkpoint_id")
    label = _safe_file_stem(label, field="label")
    root = Path(run_dir)
    directory = root / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    relative_path = Path("checkpoints") / f"{checkpoint_id}.npz"
    target = root / relative_path
    if target.exists():
        raise FileExistsError(f"Checkpoint already exists: {target}")

    temporary = target.with_suffix(".npz.tmp")
    spec = controls.spec.to_dict()
    optimizer_state_json, optimizer_arrays = _encode_optimizer_state(optimizer_state)
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            controls=controls.as_matrix(copy=True),
            spec_json=_encode_json(spec),
            controls_name=np.asarray("" if controls.name is None else controls.name, dtype=np.str_),
            controls_meta_json=_encode_json(controls.meta),
            optimizer_state_json=optimizer_state_json,
            snapshot_json=_encode_json(snapshot),
            api_context_json=_encode_json(api_context),
            metadata_json=_encode_json(metadata),
            **optimizer_arrays,
        )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)
    _fsync_directory(directory)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return CheckpointReference(
        checkpoint_id=checkpoint_id,
        label=label,
        relative_path=str(relative_path),
        shape=controls.shape,
        sha256=digest,
    )


def load_checkpoint(
    run_dir: str | Path,
    reference: CheckpointReference | Mapping[str, Any],
    *,
    verify: bool = True,
) -> CheckpointData:
    """Load a checkpoint without needing the original system object."""

    if isinstance(reference, Mapping):
        reference = CheckpointReference(
            checkpoint_id=str(reference["checkpoint_id"]),
            label=str(reference["label"]),
            relative_path=str(reference["relative_path"]),
            shape=tuple(int(value) for value in reference["shape"]),
            sha256=None if reference.get("sha256") is None else str(reference["sha256"]),
        )
    relative_path = Path(reference.relative_path)
    if relative_path.is_absolute():
        raise ValueError("Checkpoint relative_path must not be absolute.")
    root = Path(run_dir).resolve()
    checkpoint_root = (root / "checkpoints").resolve()
    target = (root / relative_path).resolve()
    if not target.is_relative_to(checkpoint_root):
        raise ValueError("Checkpoint relative_path must remain inside the run checkpoints directory.")
    if verify and reference.sha256 is None:
        raise ValueError("Checkpoint SHA-256 is required unless verify=False is explicit.")
    if verify:
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != reference.sha256:
            raise ValueError(
                f"Checkpoint digest mismatch for {reference.checkpoint_id!r}: "
                f"expected {reference.sha256}, got {digest}."
            )
    with np.load(target, allow_pickle=False) as artifact:
        spec_data = _decode_json(artifact["spec_json"])
        spec = ControlSpec(
            keys=tuple(spec_data["keys"]),
            control_dim=int(spec_data["control_dim"]),
            dtype=spec_data["dtype"],
            dt=spec_data.get("dt"),
            meta=spec_data.get("meta", {}),
        )
        controls = Controls(
            spec,
            artifact["controls"],
            name=str(artifact["controls_name"].item()) or None,
            meta=_decode_json(artifact["controls_meta_json"]),
        )
        return CheckpointData(
            reference=reference,
            controls=controls,
            optimizer_state=_decode_optimizer_state(artifact),
            snapshot=_decode_json(artifact["snapshot_json"])
            if "snapshot_json" in artifact.files
            else {},
            api_context=_decode_json(artifact["api_context_json"])
            if "api_context_json" in artifact.files
            else {},
            metadata=_decode_json(artifact["metadata_json"]),
        )
