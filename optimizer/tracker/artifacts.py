"""Explicit checkpoint artifact storage for the passive tracker.

Checkpoint files are never created by constructing a journal or logging an
event.  They are written only when ``RunJournal.save_checkpoint`` is called by
the optimizer/controller that has already chosen a checkpoint trigger.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
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


@dataclass(frozen=True)
class CheckpointReference:
    """Small JSON-safe pointer to a separately stored control artifact."""

    checkpoint_id: str
    label: str
    relative_path: str
    shape: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "label": self.label,
            "relative_path": self.relative_path,
            "shape": list(self.shape),
        }


@dataclass(frozen=True)
class CheckpointData:
    """Loaded controls plus caller-supplied JSON metadata."""

    reference: CheckpointReference
    controls: Controls
    optimizer_state: dict[str, Any]
    metadata: dict[str, Any]


def save_checkpoint(
    run_dir: str | Path,
    *,
    checkpoint_id: str,
    label: str,
    controls: Controls,
    optimizer_state: Mapping[str, Any] | None = None,
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
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            controls=controls.as_matrix(copy=True),
            spec_json=_encode_json(spec),
            controls_name=np.asarray("" if controls.name is None else controls.name, dtype=np.str_),
            controls_meta_json=_encode_json(controls.meta),
            optimizer_state_json=_encode_json(optimizer_state),
            metadata_json=_encode_json(metadata),
        )
        handle.flush()
    temporary.replace(target)
    return CheckpointReference(
        checkpoint_id=checkpoint_id,
        label=label,
        relative_path=str(relative_path),
        shape=controls.shape,
    )


def load_checkpoint(run_dir: str | Path, reference: CheckpointReference | Mapping[str, Any]) -> CheckpointData:
    """Load a checkpoint without needing the original system object."""

    if isinstance(reference, Mapping):
        reference = CheckpointReference(
            checkpoint_id=str(reference["checkpoint_id"]),
            label=str(reference["label"]),
            relative_path=str(reference["relative_path"]),
            shape=tuple(int(value) for value in reference["shape"]),
        )
    target = Path(run_dir) / reference.relative_path
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
            optimizer_state=_decode_json(artifact["optimizer_state_json"]),
            metadata=_decode_json(artifact["metadata_json"]),
        )
