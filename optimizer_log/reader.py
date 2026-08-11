"""Read canonical tracker files without importing an optimizer or system."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import socket
import subprocess
from typing import Any, Mapping


SUPPORTED_SCHEMAS = frozenset({"optimizer.tracker/v3"})
_EVENT_KINDS = {
    "run_start",
    "iteration",
    "checkpoint",
    "chunk",
    "analysis",
    "intervention",
    "run_end",
    "tracking_failure",
    "observer_failure",
}


@dataclass(frozen=True)
class TimelineReadResult:
    """Complete timeline records and, if present, one ignored damaged tail line."""

    records: tuple[dict[str, Any], ...]
    recovered_tail: dict[str, Any] | None = None
    next_offset: int = 0
    last_sequence: int | None = None
    unsupported_events: tuple[dict[str, Any], ...] = ()
    stalled: bool = False
    reason: str | None = None
    corrupt_at_offset: int | None = None


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return payload


def _require_keys(payload: Mapping[str, Any], keys: set[str], *, context: str) -> None:
    missing = keys.difference(payload)
    if missing:
        raise ValueError(f"{context} is missing required key(s): {sorted(missing)!r}.")


def _validate_schema(payload: Mapping[str, Any], *, context: str) -> None:
    version = payload.get("schema_version")
    if version not in SUPPORTED_SCHEMAS:
        raise ValueError(
            f"Unsupported {context} schema_version {version!r}; "
            f"supported versions are {sorted(SUPPORTED_SCHEMAS)!r}."
        )


def validate_receipt(payload: Mapping[str, Any]) -> None:
    """Validate the generic receipt fields required by read-only consumers."""

    _require_keys(
        payload,
        {"schema_version", "run_id", "objective", "references"},
        context="run receipt",
    )
    _validate_schema(payload, context="run receipt")
    objective = payload["objective"]
    if not isinstance(objective, Mapping):
        raise ValueError("run receipt objective must be an object.")
    _require_keys(objective, {"label", "direction"}, context="objective")
    if objective["direction"] not in {"minimize", "maximize"}:
        raise ValueError("objective direction must be 'minimize' or 'maximize'.")
    references = payload["references"]
    if not isinstance(references, Mapping) or not {"problem", "system"}.issubset(references):
        raise ValueError("run receipt references must contain problem and system.")


def validate_event(
    payload: Mapping[str, Any],
    *,
    expected_run_id: str | None = None,
    previous_sequence: int | None = None,
) -> bool:
    """Validate one causal event envelope without importing optimizer code."""

    _require_keys(
        payload,
        {"schema_version", "run_id", "event_id", "sequence", "kind", "payload"},
        context="timeline event",
    )
    _validate_schema(payload, context="timeline event")
    if expected_run_id is not None and payload["run_id"] != expected_run_id:
        raise ValueError("timeline event run_id does not match the receipt.")
    sequence = payload["sequence"]
    if isinstance(sequence, bool) or int(sequence) != sequence or int(sequence) <= 0:
        raise ValueError("timeline event sequence must be a positive integer.")
    if previous_sequence is not None and int(sequence) != int(previous_sequence) + 1:
        raise ValueError("timeline event sequence is not contiguous.")
    if not isinstance(payload["payload"], Mapping):
        raise ValueError("timeline event payload must be an object.")
    return payload["kind"] in _EVENT_KINDS


def validate_timeline(records: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> None:
    """Validate the one-start, one-terminal lifecycle of a complete prefix."""

    lifecycle = "created"
    for record in records:
        kind = record["kind"]
        if kind not in _EVENT_KINDS:
            continue
        if lifecycle == "created":
            if kind != "run_start":
                raise ValueError("timeline must begin with run_start.")
            lifecycle = "running"
            continue
        if lifecycle == "ended":
            raise ValueError("timeline contains an event after run_end.")
        if kind == "run_start":
            raise ValueError("timeline contains more than one run_start.")
        if kind == "run_end":
            lifecycle = "ended"


def read_receipt(run_dir: str | Path) -> dict[str, Any]:
    """Read the stored run receipt exactly as written."""

    payload = _read_object(Path(run_dir) / "run_receipt.json")
    validate_receipt(payload)
    return payload


def read_live(run_dir: str | Path) -> dict[str, Any] | None:
    """Read the current replaceable observer snapshot, if the run has started."""

    path = Path(run_dir) / "live.json"
    if not path.exists():
        return None
    payload = _read_object(path)
    _validate_schema(payload, context="live snapshot")
    return payload


def read_terminal(run_dir: str | Path) -> dict[str, Any] | None:
    """Read the direct terminal-result index, if the run has ended."""

    path = Path(run_dir) / "terminal.json"
    if not path.exists():
        return None
    payload = _read_object(path)
    _validate_schema(payload, context="terminal index")
    return payload


def read_writer(run_dir: str | Path) -> dict[str, Any] | None:
    """Read explicit writer ownership and heartbeat state, if present."""

    path = Path(run_dir) / "writer.json"
    if not path.exists():
        return None
    payload = _read_object(path)
    _validate_schema(payload, context="writer record")
    return payload


def _process_start_token(pid: int) -> tuple[str, str | None]:
    """Inspect a local process without delivering any signal."""

    try:
        completed = subprocess.run(
            ["ps", "-p", str(int(pid)), "-o", "lstart="],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError, TypeError, ValueError):
        return "unknown", None
    token = completed.stdout.strip()
    if completed.returncode != 0 or not token:
        return "missing", None
    return "present", token


def inspect_writer(run_dir: str | Path) -> dict[str, Any]:
    """Report local writer liveness without taking ownership or modifying files."""

    root = Path(run_dir)
    writer = read_writer(root)
    lock_exists = (root / ".writer.lock").exists()
    if writer is None:
        return {"state": "missing", "lock_exists": lock_exists, "writer": None}
    status = str(writer.get("status", "unknown"))
    state = status if status in {"closed", "failed"} else "unknown"
    if status == "live" and writer.get("host") == socket.gethostname():
        process_state, token = _process_start_token(writer.get("pid"))
        if process_state == "missing":
            state = "abandoned"
        elif process_state == "present":
            expected = writer.get("process_start_token")
            if expected is not None and token != expected:
                state = "abandoned_pid_reused"
            else:
                state = "live"
        else:
            state = "unknown"
    elif status == "live":
        state = "remote_or_unknown"
    return {"state": state, "lock_exists": lock_exists, "writer": writer}


def _decode_event(
    raw: bytes,
    *,
    context: str,
    receipt: Mapping[str, Any],
    previous_sequence: int | None,
) -> tuple[dict[str, Any], bool]:
    record = json.loads(raw.decode("utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"{context} must be a JSON object.")
    known = validate_event(
        record,
        expected_run_id=str(receipt["run_id"]),
        previous_sequence=previous_sequence,
    )
    return record, known


def _partial_tail(raw: bytes) -> bool:
    return not raw.endswith((b"\n", b"\r"))


def read_timeline(run_dir: str | Path) -> TimelineReadResult:
    """Read one byte-consistent JSONL snapshot and recover one partial tail."""

    path = Path(run_dir) / "timeline.jsonl"
    if not path.exists():
        return TimelineReadResult(records=(), next_offset=0)
    receipt = read_receipt(run_dir)
    data = path.read_bytes()
    records: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    previous_sequence: int | None = None
    offset = 0
    lines = data.splitlines(keepends=True)
    for index, raw in enumerate(lines):
        line_start = offset
        offset += len(raw)
        if not raw.strip():
            continue
        try:
            record, known = _decode_event(
                raw,
                context=f"Timeline record at line {index + 1}",
                receipt=receipt,
                previous_sequence=previous_sequence,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if index == len(lines) - 1 and _partial_tail(raw):
                validate_timeline(records)
                return TimelineReadResult(
                    records=tuple(records),
                    recovered_tail={"line": index + 1, "error": str(exc)},
                    next_offset=line_start,
                    last_sequence=previous_sequence,
                    unsupported_events=tuple(unsupported),
                    reason="partial_tail",
                )
            raise ValueError(f"Corrupt non-tail timeline record at line {index + 1}.") from exc
        previous_sequence = int(record["sequence"])
        if known:
            records.append(record)
        else:
            unsupported.append(record)
    validate_timeline(records)
    return TimelineReadResult(
        records=tuple(records),
        next_offset=len(data),
        last_sequence=previous_sequence,
        unsupported_events=tuple(unsupported),
    )


def follow_timeline(
    run_dir: str | Path,
    *,
    offset: int = 0,
    previous_sequence: int | None = None,
) -> TimelineReadResult:
    """Read complete appended events and distinguish partial tails from corruption."""

    path = Path(run_dir) / "timeline.jsonl"
    if not path.exists():
        return TimelineReadResult(records=(), next_offset=0, last_sequence=previous_sequence)
    if int(offset) < 0 or int(offset) != offset:
        raise ValueError("offset must be a non-negative integer.")
    if previous_sequence is not None and (
        isinstance(previous_sequence, bool)
        or int(previous_sequence) < 1
        or int(previous_sequence) != previous_sequence
    ):
        raise ValueError("previous_sequence must be a positive integer when supplied.")
    receipt = read_receipt(run_dir)
    records: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    last_sequence = None if previous_sequence is None else int(previous_sequence)
    with path.open("rb") as handle:
        handle.seek(int(offset))
        while True:
            line_start = handle.tell()
            raw = handle.readline()
            if not raw:
                return TimelineReadResult(
                    records=tuple(records),
                    next_offset=handle.tell(),
                    last_sequence=last_sequence,
                    unsupported_events=tuple(unsupported),
                )
            if not raw.strip():
                continue
            try:
                record, known = _decode_event(
                    raw,
                    context=f"Timeline record at byte {line_start}",
                    receipt=receipt,
                    previous_sequence=last_sequence,
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                writer_state = inspect_writer(run_dir)["state"]
                if _partial_tail(raw) and writer_state in {
                    "live",
                    "remote_or_unknown",
                    "unknown",
                }:
                    return TimelineReadResult(
                        records=tuple(records),
                        recovered_tail={"offset": line_start, "error": str(exc)},
                        next_offset=line_start,
                        last_sequence=last_sequence,
                        unsupported_events=tuple(unsupported),
                        reason="partial_tail",
                    )
                return TimelineReadResult(
                    records=tuple(records),
                    next_offset=handle.tell(),
                    last_sequence=last_sequence,
                    unsupported_events=tuple(unsupported),
                    stalled=True,
                    reason="corrupt_record",
                    corrupt_at_offset=line_start,
                )
            last_sequence = int(record["sequence"])
            if known:
                records.append(record)
            else:
                unsupported.append(record)


def _path_value(payload: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for part in str(dotted_path).split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def project_timeline(
    records: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    kind: str | None = None,
    fields: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Create a generic read-only projection for future APIs or UI consumers."""

    if kind is not None and kind not in _EVENT_KINDS:
        raise ValueError(f"Unknown timeline event kind: {kind!r}.")
    selected: list[dict[str, Any]] = []
    for record in records:
        if kind is not None and record.get("kind") != kind:
            continue
        if fields is None:
            selected.append(dict(record))
        else:
            selected.append(
                {str(name): _path_value(record, path) for name, path in fields.items()}
            )
    return tuple(selected)


def metric_coverage(
    receipt: Mapping[str, Any],
    records: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """Compare declared metrics with observed and explicitly omitted metrics."""

    validate_receipt(receipt)
    declared = {str(name) for name in (receipt.get("metrics") or {})}
    observed: set[str] = set()
    omitted: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            metrics = value.get("metrics")
            if isinstance(metrics, Mapping):
                observed.update(str(name) for name in metrics)
            omitted_values = value.get("omitted_metrics")
            if isinstance(omitted_values, (list, tuple)):
                omitted.update(str(name) for name in omitted_values)
            elif isinstance(omitted_values, Mapping):
                for names in omitted_values.values():
                    if isinstance(names, (list, tuple)):
                        omitted.update(str(name) for name in names)
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    for record in records:
        visit(record.get("payload", {}))
    return {
        "declared": tuple(sorted(declared)),
        "observed": tuple(sorted(observed)),
        "omitted": tuple(sorted(omitted)),
        "never_observed": tuple(sorted(declared.difference(observed))),
    }
