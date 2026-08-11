"""Read canonical tracker files without importing an optimizer system or policy."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TimelineReadResult:
    """Complete timeline records and, if present, one ignored damaged tail line."""

    records: tuple[dict[str, Any], ...]
    recovered_tail: dict[str, Any] | None = None


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return payload


def read_receipt(run_dir: str | Path) -> dict[str, Any]:
    """Read the stored run receipt exactly as written."""

    return _read_object(Path(run_dir) / "run_receipt.json")


def read_live(run_dir: str | Path) -> dict[str, Any] | None:
    """Read the current replaceable observer snapshot, if the run has started."""

    path = Path(run_dir) / "live.json"
    return None if not path.exists() else _read_object(path)


def read_terminal(run_dir: str | Path) -> dict[str, Any] | None:
    """Read the direct terminal-result index, if the run has ended."""

    path = Path(run_dir) / "terminal.json"
    return None if not path.exists() else _read_object(path)


def read_timeline(run_dir: str | Path) -> TimelineReadResult:
    """Read JSONL and recover only a final incomplete line.

    Mid-stream corruption is an error: skipping it could change the recorded
    initial -> iteration -> analysis -> intervention -> terminal causality.
    """

    path = Path(run_dir) / "timeline.jsonl"
    if not path.exists():
        return TimelineReadResult(records=())
    lines = path.read_text(encoding="utf-8").splitlines()
    non_empty = [index for index, line in enumerate(lines) if line.strip()]
    last_non_empty = non_empty[-1] if non_empty else None
    records: list[dict[str, Any]] = []
    for index in non_empty:
        try:
            record = json.loads(lines[index])
        except json.JSONDecodeError as exc:
            if index == last_non_empty:
                return TimelineReadResult(
                    records=tuple(records),
                    recovered_tail={"line": index + 1, "error": str(exc)},
                )
            raise ValueError(f"Corrupt non-tail timeline record at line {index + 1}.") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Timeline record at line {index + 1} must be a JSON object.")
        records.append(record)
    return TimelineReadResult(records=tuple(records))
